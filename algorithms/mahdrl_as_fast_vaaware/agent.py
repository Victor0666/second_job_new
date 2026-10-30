"""Fast D3QN math with isolated identities and an experience-based HA epsilon."""
from copy import deepcopy
import numpy as np
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as BaseTA, SetD3QNAgent as BaseSet
from algorithms.mahdrl_oe_ge_apf_score_fast.selection import validate_checkpoint_selection, selection_metadata
from .config import CHECKPOINT_METADATA_KEY


class VariantIdentity:
    def bind(self, scenario, role):
        if role not in ("TA", "HA", "VA"):
            raise ValueError("Unknown learner role")
        self.mechanism = scenario.mechanism
        if role != "TA" and (self.context_dim, self.candidate_dim) != (
                (23, self.mechanism.host_features) if role == "HA" else (30, 7)):
            raise ValueError("VAware observation schema mismatch")
        super().bind(scenario, role)
        self.eligible_steps = 0
        self.model_version = scenario.model_version
        self.implementation = scenario.implementation_metadata()
        self.identity = dict(scenario.variant_metadata(), role=role,
            parameter_shapes={k: list(v.shape) for k, v in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=deepcopy(self.implementation))
        self.selection_expectation.update(algorithm=scenario.algorithm, model_version=self.model_version,
            mechanism=self.identity["mechanism"])
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing VAware metadata")
        for key, expected in dict(model_version=self.model_version, algorithm_variant=self.identity,
                implementation=self.implementation,
                experiment_protocol=self.selection_expectation["protocol"],
                checkpoint_selection=selection_metadata()).items():
            if actual.get(key) != expected:
                raise ValueError("VAware checkpoint mismatch: "+key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self.normalization_metadata["ha_eligible_steps"] = self.eligible_steps
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("VAware requires strict loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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


class SetD3QNAgent(VariantIdentity, BaseSet):
    def epsilon(self):
        if getattr(self, "role", None) != "HA":
            return super().epsilon()
        fraction = 1-min(self.eligible_steps, self._eps_decay_steps)/self._eps_decay_steps
        return float(self._eps_end+(self._eps_start-self._eps_end)*fraction)

    def select_preview_action(self, state, *, training):
        if self.role != "VA" or self._updates != state["va_version"]:
            raise RuntimeError("VA changed between preview and action")
        if state["epsilon"] != (self.epsilon() if training else 0.):
            raise RuntimeError("VA epsilon changed between preview and action")
        if training:
            self._action_calls += 1
            return int(np.random.choice(len(state["mask"]), p=state["probabilities"]))
        legal = np.flatnonzero(state["mask"] > .5)
        return int(legal[np.argmax(state["q"][legal])])
