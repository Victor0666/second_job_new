"""Remove learned VA and VM APF; retain TA, HA, host APF and OE settings."""
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)

ALGORITHM = "mahdrl_oe_ge_apf_score_fast_nva"
DISPLAY_NAME = "MAHDRL-AS-Fast-V2-NVA"
MODEL_VERSION = "mahdrl_as_fast_v2_nva_v1"
CHECKPOINT_METADATA_KEY = "mahdrl_as_fast_nva"


def vm_policy_metadata():
    return dict(version="uniform_selected_host_idle_vm_v1", mode="uniform_local_vm_slot",
        host_apf="enabled", vm_apf="disabled", candidate_scope="all_legal_idle_vms_in_selected_host",
        idle_tolerance=1e-12, rng="numpy.random.RandomState",
        seed_rule="(instance_seed * 1000003 + 20260325) & 0xFFFFFFFF",
        reset="each_episode_in_all_modes", learned_roles=["TA", "HA"],
        initialization="baseline_order_then_discard_VA")


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_v2", vm_policy=vm_policy_metadata(),
        host_apf="enabled", vm_apf="disabled",
        host_summary_scope="all_idle_vms_on_final_apf_hosts",
        components=["OE", "HA_GE", "HOST_APF", "shared_scoring"])
    return metadata


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), vm_policy=vm_policy_metadata(),
        environment="one_decision_cache_numpy_host_apf_all_idle_vms_float64")
    return metadata
