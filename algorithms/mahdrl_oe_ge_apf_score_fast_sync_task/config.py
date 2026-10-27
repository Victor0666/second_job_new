"""Explicit synchronization identity; shared AS-Fast scientific settings."""
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs, get_config,
)

CONFIG = get_config('task')
ALGORITHM = CONFIG.algorithm
DISPLAY_NAME = CONFIG.display_name
MODEL_VERSION = CONFIG.model_version
CHECKPOINT_METADATA_KEY = CONFIG.checkpoint_key
variant_metadata = CONFIG.variant_metadata
implementation_metadata = CONFIG.implementation_metadata
synchronization_metadata = CONFIG.synchronization_metadata
