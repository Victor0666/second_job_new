"""Two explicit update clocks; all AS-Fast v2 scientific settings are retained."""
from dataclasses import dataclass
from algorithms.mahdrl_oe_ge_apf_score_fast.config import (
    APF, TRUNK, ENTITY, TA_HIDDEN, learning_kwargs,
    variant_metadata as base_variant_metadata,
    implementation_metadata as base_implementation_metadata,
)


@dataclass(frozen=True)
class SyncConfig:
    mode: str

    def __post_init__(self):
        if self.mode not in ("task", "phase"):
            raise ValueError("Synchronization mode must be task or phase")

    @property
    def algorithm(self):
        return "mahdrl_oe_ge_apf_score_fast_sync_" + self.mode

    @property
    def model_version(self):
        return "mahdrl_as_fast_v2_sync_" + self.mode + "_v1"

    @property
    def display_name(self):
        return "MAHDRL-AS-Fast-V2-Sync-" + self.mode.title()

    @property
    def checkpoint_key(self):
        return "mahdrl_as_fast_sync_" + self.mode

    def synchronization_metadata(self):
        return dict(version="synchronization_clock_v1", mode=self.mode,
            learned_roles=["TA", "HA", "VA"], host_apf="enabled", vm_apf="enabled",
            ta_decision_clock="assignment" if self.mode == "task" else "phase",
            optimizer_clock="assignment_end" if self.mode == "task" else "phase_end",
            optimizer_budget=("one_per_eligible_role_per_assignment" if self.mode == "task"
                else "one_credit_per_warm_complete_transition_flush_all_at_phase_end"),
            ta_reward_interval="actual_TA_decisions" if self.mode == "task" else "physical_phase",
            ta_reward="original_physical_energy_and_finished_task_delay_increments",
            lower_reward="unchanged_OE_assignment_marginal_energy",
            successors="consecutive_actual_role_decisions_only", warmup="unchanged_per_role",
            gamma="unchanged_per_role_decision", exploration="unchanged_per_optimizer_update",
            terminal_flush="complete_pending_then_flush_no_extra_assignment_barrier",
            time_advance="original_physical_event_engine_only")

    def variant_metadata(self):
        result = base_variant_metadata()
        result.update(algorithm=self.algorithm, scheme=self.model_version,
            model_version=self.model_version, base_implementation="mahdrl_as_fast_v2",
            synchronization=self.synchronization_metadata())
        return result

    def implementation_metadata(self):
        result = base_implementation_metadata()
        result.update(algorithm=self.algorithm, implementation_version=self.model_version,
            scientific_variant=self.variant_metadata(), synchronization=self.synchronization_metadata(),
            environment="physical_phase_preserved_instrumented_resequencing")
        return result


def get_config(mode):
    return SyncConfig(mode)
