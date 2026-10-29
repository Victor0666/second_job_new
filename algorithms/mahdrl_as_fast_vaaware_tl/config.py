"""Two learned roles under the VAware training and selection protocol."""
from common.mahdrl_ablation import no_apf_metadata
from algorithms.mahdrl_oe_ge_apf_score_fast.config import TA_HIDDEN, learning_kwargs
from algorithms.mahdrl_as_fast_vaaware.config import variant_metadata as vaaware_metadata
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.config import tl_policy_metadata as global_policy
from algorithms.mahdrl_as_fast_vaaware.selection import selection_metadata

ALGORITHM = "mahdrl_as_fast_vaaware_tl"
DISPLAY_NAME = "MAHDRL-AS-Fast-VAware-TL"
MODEL_VERSION = "mahdrl_as_fast_vaaware_tl_noapf_v1"
OUTPUT_TAG = "mahdrl_vaaware_tl"
CHECKPOINT_METADATA_KEY = "mahdrl_vaaware_tl"
VM_CONTEXT_DIM, VM_CANDIDATE_DIM = 23, 14


def tl_policy_metadata():
    return dict(global_policy(), version="vaaware_ta_global_va_noapf_v1",
        va_preview="not_applicable", ha_update_period="not_applicable",
        ge_scope="all_physically_legal_idle_vms", ge_sampling="analytic_mixture_single_draw")


def variant_metadata():
    metadata = vaaware_metadata()
    for key in ("mechanism", "va_preview", "ha_epsilon_clock"):
        metadata.pop(key)
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_vaaware_noapf_v1", tl_policy=tl_policy_metadata(),
        learned_roles=["TA", "VA"], observation_layout="task23_globalvm14_v1",
        host_summary_scope="not_used", components=["OE", "VA_GE", "shared_scoring", "global_VM_selection"],
        update_schedule="VA_after_assignment_TA_at_phase_v1")
    return no_apf_metadata(metadata)


def implementation_metadata():
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(), tl_policy=tl_policy_metadata(),
        replay="frozen_historical_global_observations_fast_pack",
        environment="raw_legal_noapf_float64", checkpoint_selection=selection_metadata())
