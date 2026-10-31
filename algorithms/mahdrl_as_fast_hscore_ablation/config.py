"""Each ablation changes one mechanism and has an isolated checkpoint identity."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_hscore import config as base
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.config import get_config as fixed_config

SUFFIXES = ("nta", "nha", "nva", "sync_task", "sync_phase", "ertf", "edf", "cp", "r090", "r010_nopf")


@dataclass(frozen=True)
class Variant:
    suffix: str

    def __post_init__(self):
        if self.suffix not in SUFFIXES:
            raise ValueError("Unknown HScore ablation: " + self.suffix)

    @property
    def algorithm(self):
        return "mahdrl_as_fast_hscore_" + self.suffix

    @property
    def output_tag(self):
        return "hs_" + self.suffix

    @property
    def model_version(self):
        return "hscore_" + self.suffix + "_noapf_srgt092_v1"

    @property
    def roles(self):
        removed = {"nta": "TA", "nha": "HA", "nva": "VA", "ertf": "TA", "edf": "TA", "cp": "TA"}.get(self.suffix)
        return tuple(r for r in ("TA", "HA", "VA") if r != removed)

    @property
    def ordering(self):
        return self.suffix if self.suffix in ("ertf", "edf", "cp") else None

    @property
    def alpha(self):
        return {"r090": .9, "r010_nopf": .1}.get(self.suffix, .75)

    def metadata(self, mechanism):
        data = base.variant_metadata(mechanism)
        data.update(algorithm=self.algorithm, scheme=self.model_version, model_version=self.model_version,
            base_implementation=base.MODEL_VERSION, learned_roles=list(self.roles),
            ablation=self.suffix, reward_weights=dict(delay=self.alpha, energy=1-self.alpha),
            removed_policy={"nta": "uniform_legal_TA_increment", "nha": "uniform_legal_host",
                            "nva": "uniform_legal_local_VM"}.get(self.suffix, "none"))
        data["va_preview"]["R_ref"] = f"{1-self.alpha:.12g}*0.001*100*T_ref+{self.alpha:.12g}"
        if self.ordering:
            data["ta_policy"] = fixed_config(self.ordering).ta_policy_metadata()
        if self.suffix == "nva":
            data["va_preview"].update(network="none", policy="uniform_legal_local_VM",
                expected_Q="zero_placeholder_no_learned_VA", evaluation_epsilon=1.)
            data["components"] = [c for c in data["components"] if c != "VA_fixed_slot_D3QN"]
        if self.suffix == "nha":
            data["components"] = [c for c in data["components"] if c not in
                ("HA_shared_scoring", "HA_low_frequency_updates", "VA_behavior_preview")]
            data["va_preview"] = dict(policy="not_used_no_HA")
            data["ha_epsilon_clock"] = "not_used_no_HA"
        data["update_schedule"] = ("TA_per_allocation_barrier_after_assignment" if self.suffix == "sync_task"
            else "phase_boundary_deferred_transition_credits" if self.suffix == "sync_phase"
            else "lower_after_assignment_TA_at_phase")
        return data

    def implementation(self, mechanism):
        return dict(base.implementation_metadata(mechanism), algorithm=self.algorithm,
            implementation_version=self.model_version, scientific_variant=self.metadata(mechanism),
            random_policy_rng="per_environment_independent_generator",
            replay="retained_roles_only_frozen_real_successors")
