"""Set-valued D3QN; scalar TA uses the unchanged original learner."""
from collections import deque
from copy import deepcopy
import random
import uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.d3qn_agent_EXDehid import D3QNAgent as BaseD3QN, _polyak_update
from common.f11 import LearningDiagnostics, LEARNING_MODEL
from common.f18_state_normalization import check_checkpoint
from .config import APF, TRUNK, ENTITY, variant_metadata
from .networks import CandidateDuelingQNet, collate, copy_observation, validate_observation


class CheckpointIdentity:
    def bind(self, scenario, role):
        self.role = role
        shapes = {k: list(v.shape) for k, v in self.online.state_dict().items()}
        self.identity = dict(variant_metadata(), role=role, parameter_shapes=shapes)
        self.normalization_metadata = dict(scenario.normalization_metadata(),
                                            algorithm_variant=deepcopy(self.identity))
        return self

    def save(self, path):
        if not hasattr(self, "identity"):
            raise ValueError("Bind source normalization and variant role before saving")
        self.normalization_metadata["checkpoint_group"] = getattr(self, "checkpoint_group", None) or uuid.uuid4().hex
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("MAHDRL-AS requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        actual = checkpoint.get("normalization", {})
        if actual.get("algorithm_variant") != getattr(self, "identity", None):
            raise ValueError("MAHDRL-AS variant/architecture/role mismatch")
        check_checkpoint(actual, self.normalization_metadata)
        if not actual.get("checkpoint_group"):
            raise ValueError("Missing MAHDRL-AS checkpoint group")
        super().load(path, strict=True, require_learning_model=require_learning_model)
        self.loaded_group = actual["checkpoint_group"]
        self.checkpoint_group = self.loaded_group


class TAAgent(CheckpointIdentity, BaseD3QN):
    pass


class SetD3QNAgent(CheckpointIdentity, BaseD3QN):
    def __init__(self, context_dim, candidate_dim, lr=3e-4, gamma=.95, batch_size=256,
                 buffer_size=120000, eps_start=1., eps_end=.05, eps_decay_steps=80000,
                 target_update_tau=.005, target_update_freq=2000, grad_clip=10.,
                 hidden_dims=TRUNK, entity_dims=ENTITY, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.context_dim, self.candidate_dim = context_dim, candidate_dim
        self.gamma, self.batch_size, self.grad_clip = gamma, batch_size, grad_clip
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.f11 = LearningDiagnostics()
        self.online = CandidateDuelingQNet(context_dim, candidate_dim, hidden_dims, entity_dims).to(self.device)
        self.target = CandidateDuelingQNet(context_dim, candidate_dim, hidden_dims, entity_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._eps_start, self._eps_end = float(eps_start), float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = self._updates = self._action_calls = 0
        self.target_update_tau, self.target_update_freq = target_update_tau, target_update_freq
        self.cfg = APF

    def _observe_inputs(self, state, name):
        self.f11.numerics.observe(name+"_context", state[0])
        features = state[1][state[2] > .5]
        if features.numel():
            self.f11.numerics.observe(name+"_candidates", features)

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_observation(state, mask)
        valid = np.flatnonzero(np.asarray(mask) > .5)
        if count_step:
            self._action_calls += 1
        if not deterministic and random.random() < self.epsilon():
            if performance_times is None:
                raise ValueError("GE training requires candidate completion times")
            times = np.asarray(performance_times)
            if times.shape != np.shape(mask) or not np.isfinite(times[valid]).all() or np.any(times[valid] < 0):
                raise ValueError("Invalid GE candidate times")
            if random.random() < APF.guided_probability:
                fastest = times[valid].min()
                valid = valid[np.isclose(times[valid], fastest, rtol=1e-9, atol=1e-9)]
            return int(np.random.choice(valid))
        batch = collate([state], [mask], self.device)
        self._observe_inputs(batch, "online_input")
        with torch.no_grad():
            q = self.online(*batch).masked_fill(batch[3] <= .5, -torch.inf)
            return int(q.argmax(1).item())

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        validate_observation(state, mask)
        validate_observation(next_state, next_mask, decision=not done)
        if done not in (0, 1, False, True) or not np.isfinite(reward):
            raise ValueError("Invalid set transition reward/stop flag")
        if not 0 <= int(action) < len(mask) or mask[int(action)] <= .5:
            raise ValueError("Illegal recorded set action")
        if state["context"].shape != (self.context_dim,) or state["candidates"].shape[1] != self.candidate_dim:
            raise ValueError("Set transition feature schema mismatch")
        self.f11.transition(done)
        self.buffer.append((copy_observation(state), np.array(mask, copy=True), int(action), float(reward),
                            copy_observation(next_state), np.array(next_mask, copy=True), bool(done)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        obs, mask, actions, rewards, next_obs, next_mask, done = zip(*batch)
        state = collate(obs, mask, self.device)
        self._observe_inputs(state, "online_input")
        a = torch.tensor(actions, dtype=torch.int64, device=self.device)
        q = self.online(*state).gather(1, a[:, None]).squeeze(1)
        future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            indices = [i for i, stop in enumerate(done) if not stop]
            if indices:
                successor = collate([next_obs[i] for i in indices], [next_mask[i] for i in indices], self.device)
                self._observe_inputs(successor, "target_input")
                best = self.online(*successor).masked_fill(successor[3] <= .5, -torch.inf).argmax(1)
                value = self.target(*successor).gather(1, best[:, None]).squeeze(1)
                future[indices] = self.gamma * value
            target = torch.tensor(rewards, dtype=torch.float32, device=self.device) + future
        self.f11.target(future, target)
        loss = F.smooth_l1_loss(q, target)
        self.f11.numerics.observe("Q_selected", q, scalar=True)
        self.f11.numerics.observe("TD_target", target, scalar=True)
        self.f11.numerics.observe("TD_error", target-q, scalar=True)
        value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        if self.target_update_tau > 0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        elif self._updates % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return value
