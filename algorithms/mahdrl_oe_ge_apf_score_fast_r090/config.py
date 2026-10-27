"""Only the three reward weights differ from AS-Fast v2."""
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)

ALGORITHM = 'mahdrl_oe_ge_apf_score_fast_r090'
DISPLAY_NAME = 'MAHDRL-AS-Fast-R090'
MODEL_VERSION = 'mahdrl_as_fast_r090_v1'
REWARD_PROTOCOL = 'mahdrl_as_three_layer_delay090_energy010_v1'
CHECKPOINT_METADATA_KEY = 'mahdrl_as_fast_r090'


def reward_settings():
    return dict(manager_alpha_delay=.9, alpha_delay_host=.9, alpha_delay_vm=.9,
                energy_reward_scale=.001, task_baseline_norm=300.)


def check_reward_environment(env):
    for key, value in reward_settings().items():
        if getattr(env, key) != value:
            raise ValueError('MAHDRL-AS-Fast-R090 reward setting mismatch: ' + key)


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
                    base_implementation='mahdrl_as_fast_v2', reward_protocol=REWARD_PROTOCOL,
                    reward_settings=reward_settings())
    return metadata


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
                    scientific_variant=variant_metadata(), reward_protocol=REWARD_PROTOCOL,
                    reward_settings=reward_settings())
    return metadata
