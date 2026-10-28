"""Unchanged Fast set learner with strict fixed-rule, role and saved-group identity."""
from copy import deepcopy
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import SetD3QNAgent as BaseSet
from common.mahdrl_ablation import selection_metadata, validate_checkpoint_selection
from .config import get_config


class SetD3QNAgent(BaseSet):
    def bind(self, scenario, role):
        if role not in ("HA", "VA") or (self.context_dim, self.candidate_dim) != {"HA": (23, 10), "VA": (30, 7)}[role]:
            raise ValueError("Fixed-TA learner role/observation shape mismatch")
        super().bind(scenario, role)
        self.fixed_config = cfg = get_config(scenario.ordering_rule)
        self.identity = dict(cfg.variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=cfg.implementation_metadata(), ta_policy=cfg.ta_policy_metadata())
        self.selection_expectation.update(checkpoint_selection=selection_metadata(), algorithm=cfg.algorithm, model_version=cfg.model_version,
            ta_policy=cfg.ta_policy_metadata())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing fixed-TA checkpoint metadata")
        cfg = self.fixed_config
        expected = dict(model_version=cfg.model_version, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_expectation["protocol"], ta_policy=cfg.ta_policy_metadata(),
            implementation=cfg.implementation_metadata(), algorithm_variant=self.identity)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError("Fixed-TA checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(cfg.checkpoint_key), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("Fixed-TA checkpoints require strict loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid fixed-TA checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[self.fixed_config.checkpoint_key] = deepcopy(record)
