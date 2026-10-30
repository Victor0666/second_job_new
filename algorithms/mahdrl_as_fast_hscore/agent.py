"""Version-isolated TA/HA learners and masked fixed-slot Double DQN for VA."""
from collections import deque
from copy import deepcopy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.d3qn_agent_EXDehid import _polyak_update
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity, TAAgent as OriginalTA
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import (
    FastLearner, TAAgent as BaseTA, SetD3QNAgent as BaseSet,
)
from algorithms.mahdrl_oe_ge_apf_score_fast.diagnostics import FastDiagnostics
from algorithms.mahdrl_oe_ge_apf_score_fast.profiling import StageProfiler
from .config import CHECKPOINT_METADATA_KEY, VM_INPUT_DIM, VM_CAPACITY, TRUNK
from .networks import FixedVMDuelingQNet
from .observation import validate_fixed
from .preview import behavior_probabilities
from .selection import selection_metadata, validate_checkpoint_selection


class VariantIdentity:
    def bind(self, scenario, role):
        if role not in ("TA", "HA", "VA"):
            raise ValueError("Unknown learner role")
        if role == "HA" and (not self.is_set or (self.context_dim, self.candidate_dim) != (23, scenario.mechanism.host_features)):
            raise ValueError("HScore HA observation schema mismatch")
        if role == "VA" and (self.is_set or (self.input_dim, self.output_dim) != (VM_INPUT_DIM, VM_CAPACITY)):
            raise ValueError("HScore VA requires fixed 93/9 dimensions")
        if role == "TA" and (self.is_set or (self.input_dim, self.output_dim) != (15, 243)):
            raise ValueError("HScore TA observation schema mismatch")
        CheckpointIdentity.bind(self, scenario, role)
        self.mechanism = scenario.mechanism
        self.eligible_steps = 0
        self.model_version = scenario.model_version
        self.implementation = scenario.implementation_metadata()
        self.identity = dict(scenario.variant_metadata(), role=role,
            parameter_shapes={k: list(v.shape) for k, v in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=deepcopy(self.implementation))
        self.selection_expectation = dict(algorithm=scenario.algorithm, model_version=self.model_version,
            learner_role=role, checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
            model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()["source_train_scene"],
            validation_seeds=scenario.validation_seeds(), mechanism=self.identity["mechanism"])
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing HScore checkpoint metadata")
        for key, expected in dict(model_version=self.model_version, algorithm_variant=self.identity,
                implementation=self.implementation, experiment_protocol=self.selection_expectation["protocol"],
                checkpoint_selection=selection_metadata()).items():
            if actual.get(key) != expected:
                raise ValueError("HScore checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self.normalization_metadata["ha_eligible_steps"] = self.eligible_steps
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("HScore requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid HScore checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        steps = checkpoint["normalization"].get("ha_eligible_steps")
        if type(steps) is not int or steps < 0:
            raise ValueError("Invalid HA exploration clock")
        if (self.role == "HA" and checkpoint.get("updates") != steps//self.mechanism.ha_update_period
                or self.role != "HA" and steps != 0):
            raise ValueError("HA exploration and optimizer clocks disagree")
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.eligible_steps, self.loaded_selection = steps, deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)


class TAAgent(VariantIdentity, BaseTA):
    pass


class HAAgent(VariantIdentity, BaseSet):
    def epsilon(self):
        fraction = 1-min(self.eligible_steps, self._eps_decay_steps)/self._eps_decay_steps
        return float(self._eps_end+(self._eps_start-self._eps_end)*fraction)


class FixedVAAgent(VariantIdentity, FastLearner, OriginalTA):
    is_set = False

    def __init__(self, *, lr=3e-4, gamma=.95, batch_size=256, buffer_size=120000,
                 eps_start=1., eps_end=.05, eps_decay_steps=80000,
                 target_update_tau=.005, target_update_freq=2000, grad_clip=10.,
                 hidden_dims=TRUNK, device="cpu", diagnostics_every=100,
                 profile_mode="sampled", profile_every=100):
        self.input_dim, self.output_dim = VM_INPUT_DIM, VM_CAPACITY
        self.device = torch.device(device)
        self.gamma, self.batch_size, self.grad_clip = gamma, batch_size, grad_clip
        self.buffer = deque(maxlen=buffer_size)
        self.online = FixedVMDuelingQNet(hidden_dims).to(self.device)
        self.target = deepcopy(self.online)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._eps_start, self._eps_end = float(eps_start), float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = self._updates = self._action_calls = 0
        self.target_update_tau, self.target_update_freq = target_update_tau, target_update_freq
        self.f11 = FastDiagnostics(diagnostics_every)
        self.f11.numerics.inputs(self.online, "online_input")
        self.f11.numerics.inputs(self.target, "target_input")
        self.profiler = StageProfiler(profile_mode, profile_every)

    def select_preview_action(self, state, *, training):
        validate_fixed(state["obs"], state["mask"])
        if self._updates != state["va_version"]:
            raise RuntimeError("VA changed between preview and action")
        if state["epsilon"] != (self.epsilon() if training else 0.):
            raise RuntimeError("VA epsilon changed between preview and action")
        if training:
            self._action_calls += 1
            return int(np.random.choice(VM_CAPACITY, p=state["probabilities"]))
        legal = np.flatnonzero(state["mask"] > .5)
        return int(legal[np.argmax(state["q"][legal])])

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_fixed(state, mask)
        epsilon = 0. if deterministic else self.epsilon()
        if epsilon and performance_times is None:
            raise ValueError("GE training requires candidate completion times")
        obs = torch.as_tensor(np.array(state, copy=True)[None], dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(np.array(mask, copy=True)[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.online(obs, legal).cpu().numpy()[0]
        p = behavior_probabilities(q, mask, np.zeros(VM_CAPACITY) if performance_times is None else performance_times, epsilon)
        if count_step:
            self._action_calls += 1
        if deterministic:
            return int(np.flatnonzero(mask > .5)[np.argmax(q[mask > .5])])
        return int(np.random.choice(VM_CAPACITY, p=p))

    def _update(self):
        p = self.profiler
        with p.stage("replay_pack"):
            batch = random.sample(self.buffer, self.batch_size)
            obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
            for s, m, n, nm, d in zip(obs, masks, successors, next_masks, done):
                validate_fixed(s, m)
                validate_fixed(n, nm, decision=not d)
            arrays = [np.stack(x) for x in (obs, masks, successors, next_masks)]
        with p.stage("h2d", self.device):
            state, legal, following, next_legal = [torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrays]
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            ongoing = torch.tensor([not d for d in done], dtype=torch.bool, device=self.device)
        with p.stage("online_current_forward", self.device):
            q = self.online(state, legal).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            if ongoing.any():
                nxt, mask = following[ongoing], next_legal[ongoing]
                with p.stage("online_next_forward", self.device):
                    best = self.online(nxt, mask).masked_fill(mask <= .5, -torch.inf).argmax(1)
                with p.stage("target_forward", self.device):
                    future[ongoing] = self.gamma*self.target(nxt, mask).gather(1, best[:, None]).squeeze(1)
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
