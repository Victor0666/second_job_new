"""Keep AS-Fast v2 settings; replace only the learned sequencing policy."""
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)

ALGORITHM = "mahdrl_oe_ge_apf_score_fast_nta"
DISPLAY_NAME = "MAHDRL-AS-Fast-V2-NTA"
MODEL_VERSION = "mahdrl_as_fast_v2_nta_v1"
CHECKPOINT_METADATA_KEY = "mahdrl_as_fast_nta"


def ta_policy_metadata():
    return dict(version="random_legal_weight_delta_v1", mode="uniform_legal_increment",
                action_count=243, increments=[-.1, 0., .1],
                seed_rule="(instance_seed * 1000003 + 20260325 + 9176) & 0xFFFFFFFF",
                rng="numpy.random.RandomState", reset="each_episode_in_all_modes",
                learned_roles=["HA", "VA"], initialization="baseline_order_then_discard_TA")


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
                    base_implementation="mahdrl_as_fast_v2", ta_policy=ta_policy_metadata())
    return metadata


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
                    scientific_variant=variant_metadata(), ta_policy=ta_policy_metadata())
    return metadata
