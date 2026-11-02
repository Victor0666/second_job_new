"""Keep the existing sync-task definition and Speed2 implementation identity."""
from algorithms.mahdrl_as_fast_hscore_ablation.config import Variant
from algorithms.mahdrl_as_fast_hscore_speed2 import config as speed2

CONFIG = Variant('sync_task')
ALGORITHM = 'mahdrl_as_fast_hscore_speed2_sync_task'
DISPLAY_NAME = 'MAHDRL-HScore-Speed2-Sync-Task'
MODEL_VERSION = 'hscore_speed2_sync_task_srgt092_v1'
OUTPUT_TAG = CHECKPOINT_METADATA_KEY = 'hs_s2_sync_task'


def variant_metadata(mechanism):
    return dict(CONFIG.metadata(mechanism), algorithm=ALGORITHM, scheme=MODEL_VERSION,
                model_version=MODEL_VERSION, base_implementation=speed2.MODEL_VERSION)


def implementation_metadata(mechanism):
    return dict(speed2.implementation_metadata(mechanism), algorithm=ALGORITHM,
        implementation_version=MODEL_VERSION, scientific_variant=variant_metadata(mechanism),
        rollout='hscore_sync_task_real_successors_assignment_barriers_v1',
        random_policy_rng='per_environment_independent_generator')
