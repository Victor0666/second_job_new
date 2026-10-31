"""Unchanged fixed93-to-9 VA network, trained with global Double DQN targets."""
from copy import deepcopy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.d3qn_agent_EXDehid import _polyak_update
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as BaseTA
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import frozen_array
from algorithms.mahdrl_as_fast_hscore.agent import FixedVAAgent
from algorithms.mahdrl_as_fast_vaaware.preview import behavior_probabilities
from .config import CHECKPOINT_METADATA_KEY, VM_CAPACITY, VM_INPUT_DIM, tl_policy_metadata
from .selection import selection_metadata, validate_checkpoint_selection
from .observation import validate_grouped


class TLIdentity:
    def bind(self, scenario, role):
        expected = (15, 243) if role == "TA" else (VM_INPUT_DIM, VM_CAPACITY)
        if role not in ("TA", "VA") or (self.input_dim, self.output_dim) != expected:
            raise ValueError("HScore TL requires TA15/243 and shared VA93/9 only")
        CheckpointIdentity.bind(self, scenario, role)
        self.model_version = scenario.model_version
        self.implementation = scenario.implementation_metadata()
        self.identity = dict(scenario.variant_metadata(), role=role,
            parameter_shapes={k: list(v.shape) for k, v in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=deepcopy(self.implementation), tl_policy=tl_policy_metadata())
        self.selection_expectation = dict(algorithm=scenario.algorithm, model_version=self.model_version,
            learner_role=role, checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
            model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()["source_train_scene"],
            validation_seeds=scenario.validation_seeds(), tl_policy=tl_policy_metadata())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing HScore TL checkpoint metadata")
        for key, expected in dict(model_version=self.model_version, algorithm_variant=self.identity,
                implementation=self.implementation, tl_policy=tl_policy_metadata(),
                experiment_protocol=self.selection_expectation["protocol"],
                checkpoint_selection=selection_metadata()).items():
            if actual.get(key) != expected:
                raise ValueError("HScore TL checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("HScore TL requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid HScore TL checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)


class TAAgent(TLIdentity, BaseTA):
    pass


class GroupedVAAgent(TLIdentity, FixedVAAgent):
    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_grouped(state, mask)
        self.f11.numerics.active = not deterministic and (
            self._action_calls == 0 or (self._action_calls+1) % self.f11.numerics.every == 0)
        try:
            obs = torch.as_tensor(np.array(state, copy=True), dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(np.array(mask, copy=True).reshape(-1, VM_CAPACITY), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q = self.online(obs, legal).detach().cpu().numpy().ravel()
        finally:
            self.f11.numerics.active = False
        if performance_times is None:
            if not deterministic:
                raise ValueError("Global GE requires candidate completion times")
            performance_times = np.zeros(len(mask), np.float64)
        probabilities = behavior_probabilities(q, mask, performance_times, 0. if deterministic else self.epsilon())
        if count_step:
            self._action_calls += 1
        return int(np.argmax(probabilities)) if deterministic else int(np.random.choice(len(mask), p=probabilities))

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        if done not in (0, 1, False, True):
            raise ValueError("Learning stop flag must be boolean")
        validate_grouped(state, mask)
        validate_grouped(next_state, next_mask, decision=not done)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= int(action) < len(mask) or mask[int(action)] <= .5 or not np.isfinite(reward)):
            raise ValueError("Invalid grouped transition/action")
        self.buffer.append((frozen_array(state), frozen_array(mask), int(action), float(reward),
            frozen_array(next_state), frozen_array(next_mask), bool(done)))
        self.f11.transition(done)

    def _update(self):
        p = self.profiler
        with p.stage("replay_pack"):
            batch = random.sample(self.buffer, self.batch_size)
            states, masks, actions, rewards, successors, next_masks, done = zip(*batch)
            for s, m, n, nm, d in zip(states, masks, successors, next_masks, done):
                validate_grouped(s, m)
                validate_grouped(n, nm, decision=not d)
            current = np.stack([s[a//VM_CAPACITY] for s, a in zip(states, actions)])
            current_masks = np.stack([m.reshape(-1, VM_CAPACITY)[a//VM_CAPACITY] for m, a in zip(masks, actions)])
            indices = [i for i, d in enumerate(done) if not d]
            if indices:
                following = np.concatenate([successors[i] for i in indices])
                following_masks = np.concatenate([next_masks[i].reshape(-1, VM_CAPACITY) for i in indices])
                offsets = np.cumsum([0] + [len(successors[i]) for i in indices])
        with p.stage("h2d", self.device):
            obs = torch.as_tensor(current, dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(current_masks, dtype=torch.float32, device=self.device)
            slots = torch.tensor([a % VM_CAPACITY for a in actions], dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            if indices:
                following = torch.as_tensor(following, dtype=torch.float32, device=self.device)
                following_masks = torch.as_tensor(following_masks, dtype=torch.float32, device=self.device)
        with p.stage("online_current_forward", self.device):
            q = self.online(obs, legal).gather(1, slots[:, None]).squeeze(1)
        with torch.no_grad():
            future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            if indices:
                with p.stage("online_next_forward", self.device):
                    values = self.online(following, following_masks).masked_fill(following_masks <= .5, -torch.inf)
                    best = torch.stack([values[a:b].reshape(-1).argmax() for a, b in zip(offsets[:-1], offsets[1:])])
                    rows = torch.as_tensor(offsets[:-1], dtype=torch.int64, device=self.device) + best//VM_CAPACITY
                with p.stage("target_forward", self.device):
                    next_q = self.target(following[rows], following_masks[rows]).gather(1, (best % VM_CAPACITY)[:, None]).squeeze(1)
                    future[indices] = self.gamma*next_q
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
