"""Random host selection with learned TA/VA and no host or VM APF."""
from common.mahdrl_ablation import no_apf_metadata, selection_metadata
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)

ALGORITHM = "mahdrl_oe_ge_apf_score_fast_nha"
DISPLAY_NAME = "MAHDRL-AS-Fast-V2-NHA"
MODEL_VERSION = 'mahdrl_as_fast_v2_nha_noapf_srgt092_v2'
CHECKPOINT_METADATA_KEY = "mahdrl_as_fast_nha"


def ha_policy_metadata():
    return dict(version="uniform_all_idle_hosts_noapf_v2", mode="uniform_host_index",
        host_apf="disabled", vm_apf="disabled",
        candidate_scope="all_hosts_with_at_least_one_legal_idle_vm",
        idle_tolerance=1e-12, rng="numpy.random.RandomState",
        seed_rule="(instance_seed * 1000003 + 20260325 + 123456 * 97) & 0xFFFFFFFF",
        reset="each_episode_in_all_modes", learned_roles=["TA", "VA"],
        initialization="baseline_order_then_discard_HA")


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_v2", ha_policy=ha_policy_metadata(),
        host_apf="disabled", vm_apf="disabled",
        components=["OE", "VA_GE", "shared_scoring"])
    return no_apf_metadata(metadata)


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), ha_policy=ha_policy_metadata(),
        environment="one_decision_cache_raw_legal_noapf_float64")
    metadata.update(checkpoint_selection=selection_metadata(), environment="one_decision_cache_raw_legal_noapf_float64")
    return metadata
