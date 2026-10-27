"""Keep the v2 learners unchanged; bind and validate the independent R090 identity."""
from copy import deepcopy
import torch
from algorithms.mahdrl_oe_ge_apf_score.agent import CheckpointIdentity
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as BaseTA, SetD3QNAgent as BaseSet
from .config import (
    ALGORITHM, MODEL_VERSION, REWARD_PROTOCOL, CHECKPOINT_METADATA_KEY,
    reward_settings, variant_metadata, implementation_metadata,
)
from .selection import selection_metadata, validate_checkpoint_selection


class R090Identity:
    def bind(self, scenario, role):
        super().bind(scenario, role)
        self.identity = dict(variant_metadata(), role=role,
            parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity),
            implementation=implementation_metadata(), reward_protocol=REWARD_PROTOCOL,
            reward_settings=reward_settings())
        self.selection_expectation.update(algorithm=ALGORITHM, model_version=MODEL_VERSION,
            reward_protocol=REWARD_PROTOCOL, reward_settings=reward_settings())
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError('Missing MAHDRL-AS-Fast-R090 normalization metadata')
        expected = dict(model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_expectation['protocol'], reward_protocol=REWARD_PROTOCOL,
            reward_settings=reward_settings(), implementation=implementation_metadata(), algorithm_variant=self.identity)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError('MAHDRL-AS-Fast-R090 checkpoint mismatch: ' + key)
        return validate_checkpoint_selection(actual.get(CHECKPOINT_METADATA_KEY), self.selection_expectation)

    def save(self, path):
        self._check_metadata(self.normalization_metadata)
        # Use the original strict serializer, bypassing v2's hard-coded metadata key.
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError('MAHDRL-AS-Fast-R090 requires strict checkpoint loading')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('Invalid MAHDRL-AS-Fast-R090 checkpoint')
        record = self._check_metadata(checkpoint.get('normalization'))
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata[CHECKPOINT_METADATA_KEY] = deepcopy(record)


class TAAgent(R090Identity, BaseTA):
    pass


class SetD3QNAgent(R090Identity, BaseSet):
    pass
