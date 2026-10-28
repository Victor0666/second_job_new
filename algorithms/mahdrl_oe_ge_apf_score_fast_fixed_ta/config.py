"""Fixed task ordering without host/VM APF; retain GE and original learning math."""
from common.mahdrl_ablation import no_apf_metadata, selection_metadata
from dataclasses import dataclass
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)


@dataclass(frozen=True)
class FixedTAConfig:
    rule: str

    def __post_init__(self):
        if self.rule not in ("ertf", "edf", "cp"):
            raise ValueError("Fixed task ordering must be ertf, edf or cp")

    @property
    def algorithm(self):
        return "mahdrl_oe_ge_apf_score_fast_" + self.rule

    @property
    def model_version(self):
        return "mahdrl_as_fast_v2_" + self.rule + "_noapf_srgt092_v3"

    @property
    def display_name(self):
        return "MAHDRL-AS-Fast-V2-" + self.rule.upper() + "-NoAPF"

    @property
    def checkpoint_key(self):
        return "mahdrl_as_fast_fixed_" + self.rule

    @property
    def weights(self):
        return {"ertf": (1., 0., 0., 0., 0.), "edf": (0., 0., 0., 0., 1.),
                "cp": (0., 0., 1., 0., 0.)}[self.rule]

    def ta_policy_metadata(self):
        keys = {"ertf": ["task_ready_time_ascending", "global_task_id_ascending"],
                "edf": ["absolute_task_subdeadline_ascending", "task_ready_time_ascending", "global_task_id_ascending"],
                "cp": ["task_up_rank_descending", "task_ready_time_ascending", "global_task_id_ascending"]}
        return dict(version="fixed_raw_task_ordering_noapf_v2", rule=self.rule, sort_keys=keys[self.rule],
            candidate_scope="arrived_ready_tasks_only", clock="original_phase_prepare",
            weights=list(self.weights), learned_roles=["HA", "VA"], host_apf="disabled", vm_apf="disabled",
            host_candidate_scope="all_hosts_with_at_least_one_legal_idle_vm",
            vm_candidate_scope="all_legal_idle_vms_on_selected_host", idle_tolerance=1e-12,
            guided_exploration="unchanged_GE", guided_probability=APF.guided_probability,
            deadline="existing_absolute_task_subdeadline", critical_path="existing_upload_compute_download_upward_rank",
            initialization="baseline_order_then_discard_TA", ta_gradient_updates=False,
            time_advance="original_physical_event_engine_only")

    def variant_metadata(self):
        result = base_variant_metadata()
        result.update(algorithm=self.algorithm, scheme=self.model_version, model_version=self.model_version,
            base_implementation="mahdrl_as_fast_v2", ta_policy=self.ta_policy_metadata(),
            learned_roles=["HA", "VA"], components=["OE", "GE", "shared_scoring"],
            host_apf="disabled", vm_apf="disabled", apf=dict(host_enabled=False, vm_enabled=False),
            host_summary_scope="all_physical_legal_candidates")
        return no_apf_metadata(result)

    def implementation_metadata(self):
        result = base_implementation_metadata()
        result.update(algorithm=self.algorithm, implementation_version=self.model_version,
            scientific_variant=self.variant_metadata(), ta_policy=self.ta_policy_metadata(),
            task_sort="raw_lexicographic_keys_at_original_phase_boundary",
            environment="one_decision_cache_raw_legal_noapf_float64")
        result.update(checkpoint_selection=selection_metadata(), environment="one_decision_cache_raw_legal_noapf_float64")
        return result


def get_config(rule):
    return FixedTAConfig(rule)
