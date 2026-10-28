"""Reuse Fast learning math with strict two-level/global-VM checkpoint identity."""
from copy import deepcopy
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as BaseTA, SetD3QNAgent as BaseSet
from .config import (
    ALGORITHM, MODEL_VERSION, CHECKPOINT_METADATA_KEY,
    tl_policy_metadata, variant_metadata, implementation_metadata, VM_CONTEXT_DIM, VM_CANDIDATE_DIM,
)
from .selection import selection_metadata, validate_checkpoint_selection


class TLIdentity:
    def bind(self, scenario, role):
        if role not in ("TA", "VA"):
            raise ValueError("TL learns only TA and VA")
        if role == "VA" and (getattr(self, "context_dim", None), getattr(self, "candidate_dim", None)) != (VM_CONTEXT_DIM, VM_CANDIDATE_DIM):
            raise ValueError("TL VA requires 23 context and 14 candidate features")
        super().bind(scenario, role)
        self.identity = dict(variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=implementation_metadata(), tl_policy=tl_policy_metadata())
        self.selection_expectation.update(checkpoint_selection=selection_metadata(), algorithm=ALGORITHM, model_version=MODEL_VERSION,
                                          tl_policy=tl_policy_metadata())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError("Missing TL checkpoint normalization metadata")
        expected = dict(model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_expectation["protocol"], tl_policy=tl_policy_metadata(),
            implementation=implementation_metadata(), algorithm_variant=self.identity)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError("MAHDRL-AS-Fast-V2-TL checkpoint mismatch: " + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError("TL requires strict checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid TL checkpoint")
        record = self._check_metadata(checkpoint.get("normalization"))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)


class TAAgent(TLIdentity, BaseTA):
    pass


class SetD3QNAgent(TLIdentity, BaseSet):
    pass
