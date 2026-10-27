"""Reuse the Fast set learner with strict NTA identity and policy provenance."""
from copy import deepcopy
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import SetD3QNAgent as BaseSet
from .config import (
    ALGORITHM, MODEL_VERSION, CHECKPOINT_METADATA_KEY,
    ta_policy_metadata, variant_metadata, implementation_metadata,
)
from .selection import selection_metadata, validate_checkpoint_selection


class SetD3QNAgent(BaseSet):
    def bind(self, scenario, role):
        if role not in ("HA", "VA"):
            raise ValueError("NTA learns only HA and VA")
        super().bind(scenario, role)
        self.identity = dict(variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=implementation_metadata(), ta_policy=ta_policy_metadata())
        self.selection_expectation.update(algorithm=ALGORITHM, model_version=MODEL_VERSION,
                                          ta_policy=ta_policy_metadata())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing NTA checkpoint normalization metadata")
        expected = dict(model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_expectation["protocol"], ta_policy=ta_policy_metadata(),
            implementation=implementation_metadata(), algorithm_variant=self.identity)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError("MAHDRL-AS-Fast-V2-NTA checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        # The shared serializer enforces roles, shapes, source scales and saved groups.
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("NTA requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid NTA checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)
