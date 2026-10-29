"""Versioned mechanism settings; the V3 baseline is never modified."""
from dataclasses import asdict, dataclass
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TA_HIDDEN, learning_kwargs, variant_metadata as base_metadata,
)

ALGORITHM = "mahdrl_as_fast_vaaware"
DISPLAY_NAME = "MAHDRL-AS-Fast-VAware"
MODEL_VERSION = "mahdrl_as_fast_vaaware_noapf_v1"
OUTPUT_TAG = "mahdrl_vaaware"
CHECKPOINT_METADATA_KEY = "mahdrl_vaaware"


@dataclass(frozen=True)
class Mechanism:
    va_preview: bool = True
    ha_update_period: int = 4

    def __post_init__(self):
        if type(self.va_preview) is not bool or type(self.ha_update_period) is not int or self.ha_update_period < 1:
            raise ValueError("VA preview must be bool; HA update period must be a positive integer")

    @property
    def host_features(self):
        return 14 if self.va_preview else 10


def variant_metadata(mechanism=Mechanism()):
    metadata = base_metadata()
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_noapf_v3", mechanism=asdict(mechanism),
        observation_layout=f"host23x{mechanism.host_features}_vm30x7_vaaware_v1",
        learned_roles=["TA", "HA", "VA"],
        va_preview=dict(version="behavior_expectation_v1", network="online",
            fields=["expected_duration", "expected_marginal_energy", "expected_delay_penalty", "expected_Q"],
            duration_scale="source_T_ref", energy_scale="100_times_source_T_ref",
            delay_scale=300., q_scale="R_ref_divided_by_1_minus_gamma",
            R_ref="0.25*0.001*100*T_ref+0.75", evaluation_epsilon=0.),
        update_schedule="lower_updates_after_assignment_TA_at_phase_v1",
        ha_epsilon_clock="eligible_real_transitions", target_clock="actual_optimizer_steps")
    metadata["components"] += (["VA_behavior_preview"] if mechanism.va_preview else [])
    metadata["components"] += (["HA_low_frequency_updates"] if mechanism.ha_update_period > 1 else [])
    return metadata


def implementation_metadata(mechanism=Mechanism()):
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(mechanism),
        replay="frozen_historical_augmented_observations_fast_pack",
        environment="raw_legal_noapf_float64", preview="batched_no_grad_no_rng",
        checkpoint_selection="select_sr092_minenergy_v1")
