"""Remove HA and host APF; retain per-host VM APF and global learned VA."""
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)


ALGORITHM = "mahdrl_oe_ge_apf_score_fast_tl"
DISPLAY_NAME = "MAHDRL-AS-Fast-V2-TL"
MODEL_VERSION = "mahdrl_as_fast_v2_tl_v1"
CHECKPOINT_METADATA_KEY = "mahdrl_as_fast_tl"
VM_CONTEXT_DIM = 23
VM_CANDIDATE_DIM = 14


def tl_policy_metadata():
    return dict(version="ta_global_va_per_host_vm_apf_v1", learned_roles=["TA", "VA"],
        host_policy="removed", host_apf="disabled", vm_apf="enabled", vm_apf_scope="per_host",
        vm_selection_scope="global", candidate_scope="union_of_all_hosts_vm_apf_candidates",
        idle_tolerance=1e-12, vm_context_dim=VM_CONTEXT_DIM, vm_candidate_dim=VM_CANDIDATE_DIM,
        candidate_layout="vm7_owner_host5_owner_power2", action_mapping="vm_ids_order")


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_v2", tl_policy=tl_policy_metadata(),
        observation_layout="task23_globalvm14_v1", host_summary_scope="not_used",
        host_apf="disabled", vm_apf="enabled", learned_roles=["TA", "VA"],
        components=["OE", "VA_GE", "VM_APF", "shared_scoring", "global_VM_selection"])
    return metadata


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), tl_policy=tl_policy_metadata(),
        environment="one_decision_cache_global_vm_numpy_per_host_vm_apf_float64")
    return metadata
