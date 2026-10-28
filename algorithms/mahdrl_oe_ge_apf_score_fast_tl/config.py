"""Two learned levels: TA and global VA, without host or VM APF."""
from common.mahdrl_ablation import no_apf_metadata, selection_metadata
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)


ALGORITHM = "mahdrl_oe_ge_apf_score_fast_tl"
DISPLAY_NAME = "MAHDRL-AS-Fast-V2-TL"
MODEL_VERSION = 'mahdrl_as_fast_v2_tl_noapf_srgt092_v2'
CHECKPOINT_METADATA_KEY = "mahdrl_as_fast_tl"
VM_CONTEXT_DIM = 23
VM_CANDIDATE_DIM = 14


def tl_policy_metadata():
    return dict(version="ta_global_va_noapf_v2", learned_roles=["TA", "VA"],
        host_policy="removed", host_apf="disabled", vm_apf="disabled", vm_apf_scope="disabled",
        vm_selection_scope="global", candidate_scope="all_physically_legal_idle_vms",
        idle_tolerance=1e-12, vm_context_dim=VM_CONTEXT_DIM, vm_candidate_dim=VM_CANDIDATE_DIM,
        candidate_layout="vm7_owner_host5_owner_power2", action_mapping="vm_ids_order")


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_v2", tl_policy=tl_policy_metadata(),
        observation_layout="task23_globalvm14_v1", host_summary_scope="not_used",
        host_apf="disabled", vm_apf="disabled", learned_roles=["TA", "VA"],
        components=["OE", "VA_GE", "shared_scoring", "global_VM_selection"])
    return no_apf_metadata(metadata)


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), tl_policy=tl_policy_metadata(),
        environment="one_decision_cache_raw_legal_noapf_float64")
    metadata.update(checkpoint_selection=selection_metadata(), environment="one_decision_cache_raw_legal_noapf_float64")
    return metadata
