"""Separate identity and outputs for environment acceleration over Speed v1."""
from algorithms.mahdrl_as_fast_hscore_speed import config as base

ALGORITHM = 'mahdrl_as_fast_hscore_speed2'
DISPLAY_NAME = 'MAHDRL-HScore-FixedVA-Speed2'
MODEL_VERSION = 'mahdrl_hscore_fixedva_srgt092_speed_v2'
OUTPUT_TAG = 'mahdrl_hscore_speed2'


def variant_metadata(mechanism):
    return dict(base.variant_metadata(mechanism), algorithm=ALGORITHM,
        scheme=MODEL_VERSION, model_version=MODEL_VERSION, base_implementation=base.MODEL_VERSION)


def implementation_metadata(mechanism):
    return dict(base.implementation_metadata(mechanism), algorithm=ALGORITHM,
        implementation_version=MODEL_VERSION, scientific_variant=variant_metadata(mechanism),
        environment='raw_legal_fixed9_settled_energy_prefix_ordered_float64_cumsum_direct_normalized_queue_v1')
