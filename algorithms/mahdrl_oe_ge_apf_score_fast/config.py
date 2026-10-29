"""NoAPF v3 retains the AS networks, GE, rewards and learning settings."""
from common.mahdrl_ablation import no_apf_metadata
from algorithms.mahdrl_oe_ge_apf_score.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as as_variant_metadata,
)
from .selection import selection_metadata

ALGORITHM = "mahdrl_oe_ge_apf_score_fast"
DISPLAY_NAME = "MAHDRL-AS-Fast-NoAPF"
MODEL_VERSION = "mahdrl_as_fast_noapf_v3"


def variant_metadata():
    return no_apf_metadata(dict(as_variant_metadata(), algorithm=ALGORITHM,
        scheme=MODEL_VERSION, model_version=MODEL_VERSION))


def implementation_metadata():
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
                checkpoint_selection=selection_metadata(),
                scientific_variant=variant_metadata(),
                replay="validate_on_insert_owned_readonly_arrays_grouped_numpy_pack",
                environment="one_decision_cache_raw_legal_noapf_float64",
                diagnostics="sampled_details_full_update_counts_and_nonfinite_guards")
