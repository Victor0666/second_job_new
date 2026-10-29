"""Identical D3QN math, initialization and replay sampling, with less host overhead."""
import random
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.d3qn_agent_EXDehid import _polyak_update
from common.f11 import validate_transition
from algorithms.mahdrl_oe_ge_apf_score.agent import TAAgent as OriginalTA, SetD3QNAgent as OriginalSet
from .config import APF, ALGORITHM, MODEL_VERSION, variant_metadata, implementation_metadata
from .selection import selection_metadata, validate_checkpoint_selection
from .packing import frozen_array, frozen_observation, validate_observation, pack, to_device, collate
from .diagnostics import FastDiagnostics
from .profiling import StageProfiler


class FastLearner:
    is_set = False

    def __init__(self, *args, diagnostics_every=100, profile_mode="sampled", profile_every=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.device)
        # TA's original hooks remain installed but their collector is disabled locally.
        self.f11.numerics.enabled = False
        self.f11 = FastDiagnostics(diagnostics_every)
        if not self.is_set:
            self.f11.numerics.inputs(self.online, "online_input")
            self.f11.numerics.inputs(self.target, "target_input")
        self.profiler = StageProfiler(profile_mode, profile_every)

    def bind(self, scenario, role):
        super().bind(scenario, role)
        self.identity = dict(variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata["algorithm_variant"] = deepcopy(self.identity)
        self.normalization_metadata["implementation"] = implementation_metadata()
        self.selection_expectation = dict(algorithm=ALGORITHM, model_version=MODEL_VERSION,
            learner_role=role, checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
            model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()['source_train_scene'],
            validation_seeds=scenario.validation_seeds())
        return self

    def save(self, path):
        validate_checkpoint_selection(self.normalization_metadata.get('mahdrl_as_fast'), self.selection_expectation)
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('normalization'), dict):
            raise ValueError(f'Missing {MODEL_VERSION} checkpoint normalization/selection metadata')
        actual = checkpoint.get('normalization', {})
        record = validate_checkpoint_selection(actual.get('mahdrl_as_fast'), self.selection_expectation)
        for key in ('model_version', 'checkpoint_selection', 'experiment_protocol'):
            if actual.get(key) != self.normalization_metadata.get(key):
                raise ValueError('MAHDRL-AS-Fast checkpoint selection mismatch: ' + key)
        super().load(path, strict=strict, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata['mahdrl_as_fast'] = deepcopy(record)

    def select_action(self, *args, deterministic=False, **kwargs):
        n = self._action_calls
        self.f11.numerics.active = not deterministic and (n == 0 or (n+1) % self.f11.numerics.every == 0)
        try:
            return super().select_action(*args, deterministic=deterministic, **kwargs)
        finally:
            self.f11.numerics.active = False

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        if self.is_set:
            validate_observation(state, mask)
            validate_observation(next_state, next_mask, decision=not done)
            for obs in (state, next_state):
                if obs["context"].shape != (self.context_dim,) or obs["candidates"].shape[1] != self.candidate_dim:
                    raise ValueError("Set transition feature schema mismatch")
        else:
            validate_transition(state, mask, action, reward, next_state, next_mask, done)
            if (np.shape(state) != (self.input_dim,) or np.shape(next_state) != (self.input_dim,)
                    or np.shape(mask) != (self.output_dim,) or np.shape(next_mask) != (self.output_dim,)):
                raise ValueError("TA transition feature schema mismatch")
            if any(np.any((np.asarray(m) != 0) & (np.asarray(m) != 1)) for m in (mask, next_mask)):
                raise ValueError("TA masks must be binary")
        if (done not in (0, 1, False, True) or not np.isfinite(reward)
                or int(action) != action or not 0 <= int(action) < len(mask) or mask[int(action)] <= .5):
            raise ValueError("Invalid recorded transition/action")
        copy = frozen_observation if self.is_set else frozen_array
        transition = (copy(state), frozen_array(mask), int(action), float(reward),
                      copy(next_state), frozen_array(next_mask), bool(done))
        self.buffer.append(transition)
        self.f11.transition(done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        self.profiler.select(self._updates)
        self.f11.numerics.active = self._updates == 0 or (self._updates+1) % self.f11.numerics.every == 0
        try:
            return self._update()
        finally:
            self.f11.numerics.active = False
            self.profiler.active = False

    def _update(self):
        p = self.profiler
        with p.stage("replay_pack"):
            batch = random.sample(self.buffer, self.batch_size)
            obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
            if self.is_set:
                current = pack(obs, masks)
                indices = np.flatnonzero(~np.asarray(done, dtype=bool)).tolist()
                following = pack([successors[i] for i in indices], [next_masks[i] for i in indices]) if indices else None
                if np.any(current[3].sum(1) == 0) or (following is not None and np.any(following[3].sum(1) == 0)):
                    raise ValueError("Replay nonterminal decision has no legal action")
            else:
                current = np.stack(obs)
                current_legal = np.stack(masks)
                following, next_legal = np.stack(successors), np.stack(next_masks)
                if not all(np.isfinite(x).all() for x in (current, following, current_legal, next_legal)):
                    raise ValueError("Non-finite TA replay batch")
                if (any(np.any((m != 0) & (m != 1)) for m in (current_legal, next_legal))
                        or np.any(current_legal.sum(1) == 0)
                        or np.any((next_legal.sum(1) == 0) & ~np.asarray(done, dtype=bool))):
                    raise ValueError("Invalid TA replay masks")
        with p.stage("h2d", self.device):
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            if self.is_set:
                state = to_device(current, self.device)
                following = to_device(following, self.device) if following is not None else None
            else:
                state = (torch.as_tensor(current, dtype=torch.float32, device=self.device),)
                following = (torch.as_tensor(following, dtype=torch.float32, device=self.device),)
                next_legal = torch.as_tensor(next_legal, dtype=torch.float32, device=self.device)
                stops = torch.tensor(done, dtype=torch.float32, device=self.device)
        with p.stage("diagnostics_inputs"):
            if self.is_set and self.f11.numerics.active and self.f11.numerics.enabled:
                self._observe_inputs(state, "online_input")
        with p.stage("online_current_forward", self.device):
            q = self.online(*state).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            if self.is_set:
                future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
                if following is not None:
                    with p.stage("diagnostics_inputs"):
                        if self.f11.numerics.active and self.f11.numerics.enabled:
                            self._observe_inputs(following, "target_input")
                    with p.stage("online_next_forward", self.device):
                        best = self.online(*following).masked_fill(following[3] <= .5, -torch.inf).argmax(1)
                    with p.stage("target_forward", self.device):
                        value = self.target(*following).gather(1, best[:, None]).squeeze(1)
                    future[indices] = self.gamma * value
            else:
                with p.stage("online_next_forward", self.device):
                    best = self.online(*following).masked_fill(next_legal <= .5, -1e9).argmax(1, keepdim=True)
                with p.stage("target_forward", self.device):
                    value = self.target(*following).masked_fill(next_legal <= .5, -1e9).gather(1, best).squeeze(1)
                    value = torch.where((next_legal > .5).any(1), value, torch.zeros_like(value))
                future = (1. - stops) * self.gamma * value
            target = reward + future
        loss = F.smooth_l1_loss(q, target)
        with p.stage("diagnostics"):
            self.f11.target(future, target)
            self.f11.numerics.observe("Q_selected", q, scalar=True)
            self.f11.numerics.observe("TD_target", target, scalar=True)
            if self.f11.numerics.active and self.f11.numerics.enabled:
                self.f11.numerics.observe("TD_error", target-q, scalar=True)
            value = self.f11.loss(loss)  # Nonfinite loss is rejected on EVERY update.
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


class TAAgent(FastLearner, OriginalTA):
    pass


class _SetPolicy(OriginalSet):
    def _observe_inputs(self, state, name):
        # Gate BEFORE GPU indexing or transfer.
        if self.f11.numerics.active and self.f11.numerics.enabled:
            super()._observe_inputs(state, name)

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


class SetD3QNAgent(FastLearner, _SetPolicy):
    is_set = True
