"""Fixed CP without host/VM APF; retain AS-Fast v2 GE and learning settings."""
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs, get_config,
)
CONFIG = get_config("cp")
ALGORITHM, DISPLAY_NAME, MODEL_VERSION = CONFIG.algorithm, CONFIG.display_name, CONFIG.model_version
CHECKPOINT_METADATA_KEY = CONFIG.checkpoint_key
variant_metadata = CONFIG.variant_metadata
implementation_metadata = CONFIG.implementation_metadata
ta_policy_metadata = CONFIG.ta_policy_metadata
