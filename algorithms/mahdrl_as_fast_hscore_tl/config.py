"""Two learners, no HA: the unchanged fixed93-to-9 VA sees a host batch."""
from algorithms.mahdrl_as_fast_hscore.config import (
    VM_CAPACITY, VM_INPUT_DIM, TA_HIDDEN, TRUNK, learning_kwargs,
    variant_metadata as hscore_metadata,
)

ALGORITHM = "mahdrl_as_fast_hscore_tl"
DISPLAY_NAME = "MAHDRL-AS-Fast-HScore-TL"
MODEL_VERSION = "mahdrl_hscore_tl_grouped9_v1"
OUTPUT_TAG = "mahdrl_hscore_tl"
CHECKPOINT_METADATA_KEY = "mahdrl_hscore_tl"


def tl_policy_metadata():
    return dict(version="global_grouped_fixed9_v1", learned_roles=["TA", "VA"],
        network_input_dim=VM_INPUT_DIM, network_output_dim=VM_CAPACITY,
        host_batch="one_row_per_real_host_in_host_ids_order",
        action="host_row_times_9_plus_local_slot",
        next_action_scope="all_legal_VMs_of_all_successor_hosts",
        dueling_center="per_host_legal_slots", host_parameter_sharing=True,
        va_preview="not_applicable", ha_update_period="not_applicable",
        ge_scope="all_physically_legal_idle_VMs", ge_sampling="analytic_mixture_single_draw")


def variant_metadata():
    metadata = hscore_metadata()
    for key in ("mechanism", "va_preview", "ha_epsilon_clock"):
        metadata.pop(key)
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_hscore_fixedva_v1", learned_roles=["TA", "VA"],
        architecture="shared_fixed93_to_9_host_batch_global_D3QN_v1",
        observation_layout="global_host_rows93_flat_slot_mask9H_v1",
        pooling="none", host_summary_scope="not_used", tl_policy=tl_policy_metadata(),
        components=["OE", "VA_GE", "VA_fixed_slot_D3QN", "global_VM_selection"],
        update_schedule="VA_after_assignment_TA_at_phase_v1")
    return metadata


def implementation_metadata():
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), tl_policy=tl_policy_metadata(),
        replay="frozen_Hx93_host_batches_and_9H_masks_real_global_successors",
        environment="unchanged_hscore_physics_fixed9_global_action_adapter",
        checkpoint_selection="select_sr092_minenergy_v1")
