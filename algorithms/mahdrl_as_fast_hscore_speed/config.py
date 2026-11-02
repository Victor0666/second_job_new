"""Independent identity; the scientific settings come from HScore v2."""
from algorithms.mahdrl_as_fast_hscore import config as base

ALGORITHM = "mahdrl_as_fast_hscore_speed"
DISPLAY_NAME = "MAHDRL-HScore-FixedVA-Speed"
MODEL_VERSION = "mahdrl_hscore_fixedva_srgt092_speed_v1"
OUTPUT_TAG = "mahdrl_hscore_speed"


def variant_metadata(mechanism):
    return dict(base.variant_metadata(mechanism), algorithm=ALGORITHM,
                scheme=MODEL_VERSION, model_version=MODEL_VERSION,
                base_implementation=base.MODEL_VERSION)


def implementation_metadata(mechanism):
    return dict(base.implementation_metadata(mechanism), algorithm=ALGORITHM,
                implementation_version=MODEL_VERSION,
                scientific_variant=variant_metadata(mechanism),
                replay="chronological_indexed_ring_same_python_random_sample",
                va_batch="vectorized_validation_cpu_nonterminal_indices")
