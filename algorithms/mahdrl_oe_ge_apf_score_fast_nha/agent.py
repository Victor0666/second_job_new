"""Use unchanged Fast learners with strict NHA policy and saved-group identity."""
from copy import deepcopy
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as BaseTA, SetD3QNAgent as BaseSet
from .config import (
    ALGORITHM, MODEL_VERSION, CHECKPOINT_METADATA_KEY,
    ha_policy_metadata, variant_metadata, implementation_metadata,
)
from .selection import selection_metadata, validate_checkpoint_selection


class NHAIdentity:
    def bind(self, scenario, role):
        if role not in ("TA", "VA"):
            raise ValueError("NHA learns only TA and VA")
        super().bind(scenario, role)
        self.identity = dict(variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=implementation_metadata(), ha_policy=ha_policy_metadata())
        self.selection_expectation.update(checkpoint_selection=selection_metadata(), algorithm=ALGORITHM, model_version=MODEL_VERSION,
                                          ha_policy=ha_policy_metadata())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing NHA checkpoint normalization metadata")
        expected = dict(model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_expectation["protocol"], ha_policy=ha_policy_metadata(),
            implementation=implementation_metadata(), algorithm_variant=self.identity)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError("MAHDRL-AS-Fast-V2-NHA checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("NHA requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid NHA checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)


class TAAgent(NHAIdentity, BaseTA):
    pass


class SetD3QNAgent(NHAIdentity, BaseSet):
    pass
