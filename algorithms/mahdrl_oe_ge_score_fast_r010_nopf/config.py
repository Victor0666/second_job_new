"""AS-Fast with delay weight 0.1 and physical-legality-only HA/VA candidates."""
from common.mahdrl_ablation import no_apf_metadata, selection_metadata
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)

ALGORITHM = 'mahdrl_oe_ge_score_fast_r010_nopf'
DISPLAY_NAME = 'MAHDRL-AS-Fast-R010-NoPF'
MODEL_VERSION = 'mahdrl_as_fast_r010_nopf_srgt092_v2'
REWARD_PROTOCOL = 'mahdrl_as_three_layer_delay010_energy090_nopf_v1'
CHECKPOINT_METADATA_KEY = 'mahdrl_as_fast_r010_nopf'


def reward_settings():
    return dict(manager_alpha_delay=.1, alpha_delay_host=.1, alpha_delay_vm=.1,
                energy_reward_scale=.001, task_baseline_norm=300.)


def check_reward_environment(env):
    for key, value in reward_settings().items():
        if getattr(env, key) != value:
            raise ValueError('MAHDRL-AS-Fast-R010-NoPF reward setting mismatch: ' + key)


def variant_metadata():
    metadata = base_variant_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
                    base_implementation='mahdrl_as_fast_v2', reward_protocol=REWARD_PROTOCOL,
                    reward_settings=reward_settings())
    metadata.update(components=['OE', 'GE', 'shared_scoring'],
                    apf=dict(scheme='physical_legality_only_v1', host_enabled=False, vm_enabled=False),
                    host_apf_enabled=False, vm_apf_enabled=False,
                    performance_action_filter=False, candidate_filter='physical_legality_only',
                    host_summary_scope='all_physically_legal_idle_vms',
                    guided_probability=APF.guided_probability)
    return no_apf_metadata(metadata)


def implementation_metadata():
    metadata = base_implementation_metadata()
    metadata.update(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
                    scientific_variant=variant_metadata(), reward_protocol=REWARD_PROTOCOL,
                    reward_settings=reward_settings())
    metadata['environment'] = 'one_decision_cache_raw_legal_float64'
    metadata.update(checkpoint_selection=selection_metadata(), environment="one_decision_cache_raw_legal_noapf_float64")
    return metadata
