"""Scientific settings come directly from AS; implementation identity is separate."""
from algorithms.mahdrl_oe_ge_apf_score.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs, variant_metadata,
)

ALGORITHM = "mahdrl_oe_ge_apf_score_fast"
DISPLAY_NAME = "MAHDRL-AS-Fast"
MODEL_VERSION = "mahdrl_as_fast_v1"


def implementation_metadata():
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
                scientific_variant=variant_metadata(),
                replay="validate_on_insert_owned_readonly_arrays_grouped_numpy_pack",
                environment="one_decision_cache_numpy_apf_float64",
                diagnostics="sampled_details_full_update_counts_and_nonfinite_guards")
