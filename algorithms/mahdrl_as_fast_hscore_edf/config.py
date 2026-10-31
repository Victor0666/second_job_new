"""Version-isolated identity for HScore edf."""
from algorithms.mahdrl_as_fast_hscore_ablation.config import Variant
from algorithms.mahdrl_as_fast_hscore.config import Mechanism

CONFIG = Variant("edf")
ALGORITHM = CONFIG.algorithm
DISPLAY_NAME = "MAHDRL-AS-Fast-HScore-EDF"
MODEL_VERSION = CONFIG.model_version
OUTPUT_TAG = CHECKPOINT_METADATA_KEY = CONFIG.output_tag

def variant_metadata(mechanism=Mechanism()):
    return CONFIG.metadata(mechanism)

def implementation_metadata(mechanism=Mechanism()):
    return CONFIG.implementation(mechanism)
