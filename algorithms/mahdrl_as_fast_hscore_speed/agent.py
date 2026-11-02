"""Same learners, sample order and TD math; batch CPU work before CUDA."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.d3qn_agent_EXDehid import _polyak_update
from algorithms.mahdrl_as_fast_hscore import agent as base
from algorithms.mahdrl_as_fast_hscore.config import VM_INPUT_DIM, VM_CAPACITY
from .replay import IndexedReplay


def pack_fixed_batch(batch):
    obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
    state, legal, following, next_legal = (np.stack(x) for x in
                                          (obs, masks, successors, next_masks))
    b = len(batch)
    if (state.shape != (b, VM_INPUT_DIM) or following.shape != state.shape
            or legal.shape != (b, VM_CAPACITY) or next_legal.shape != legal.shape):
        raise ValueError("Fixed VA requires a batch of 93-vectors and 9-action masks")
    if not all(np.isfinite(x).all() for x in (state, legal, following, next_legal)):
        raise ValueError("Invalid fixed VA state or binary mask")
    if any(np.any((m != 0) & (m != 1)) for m in (legal, next_legal)):
        raise ValueError("Invalid fixed VA state or binary mask")
    ongoing = np.flatnonzero(~np.asarray(done, dtype=bool))
    if np.any(~legal.any(1)) or np.any(~next_legal[ongoing].any(1)):
        raise ValueError("No legal VM at a non-terminal decision")
    return state, legal, actions, rewards, following[ongoing], next_legal[ongoing], ongoing


class _IndexedLearner:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = IndexedReplay(self.buffer.maxlen)


class TAAgent(_IndexedLearner, base.TAAgent):
    pass


class HAAgent(_IndexedLearner, base.HAAgent):
    pass


class FixedVAAgent(_IndexedLearner, base.FixedVAAgent):
    def _update(self):
        p = self.profiler
        with p.stage("replay_pack"):
            batch = random.sample(self.buffer, self.batch_size)
            state, legal, actions, rewards, following, next_legal, ongoing = pack_fixed_batch(batch)
        with p.stage("h2d", self.device):
            state, legal, following, next_legal = [torch.as_tensor(a, dtype=torch.float32, device=self.device)
                                                   for a in (state, legal, following, next_legal)]
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            indices = torch.as_tensor(ongoing, dtype=torch.int64, device=self.device)
        with p.stage("online_current_forward", self.device):
            q = self.online(state, legal).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            if len(ongoing):
                with p.stage("online_next_forward", self.device):
                    best = self.online(following, next_legal).masked_fill(next_legal <= .5, -torch.inf).argmax(1)
                with p.stage("target_forward", self.device):
                    future[indices] = self.gamma*self.target(following, next_legal).gather(1, best[:, None]).squeeze(1)
            target = reward + future
        loss = F.smooth_l1_loss(q, target)
        with p.stage("diagnostics"):
            self.f11.target(future, target)
            self.f11.numerics.observe("Q_selected", q, scalar=True)
            self.f11.numerics.observe("TD_target", target, scalar=True)
            self.f11.numerics.observe("TD_error", target-q, scalar=True)
            value = self.f11.loss(loss)
        with p.stage("backward_clip_optimizer", self.device):
            self.optim.zero_grad()
            loss.backward()
            self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip, error_if_nonfinite=True)
            self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        with p.stage("target_update", self.device):
            if self.target_update_tau > 0:
                _polyak_update(self.target, self.online, self.target_update_tau)
            elif self._updates % self.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        return value
