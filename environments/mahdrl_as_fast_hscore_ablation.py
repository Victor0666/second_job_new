"""HScore physics with fixed task ordering or task-level TA decisions."""
import numpy as np
from environments.mahdrl_as_fast_hscore import CloudWorkflowEnv_VMAgents as BaseEnv
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.policy import ordered_ready_tasks
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.config import get_config


class CloudWorkflowEnv_VMAgents(BaseEnv):
    def __init__(self, *args, variant, policy_seed, **kwargs):
        self.variant = variant
        self.policy_seed = int(policy_seed)
        self.policy_instance_seed = int(kwargs.get("random_seed", 0))
        if variant.ordering:
            kwargs["combo_weights"] = get_config(variant.ordering).weights
        super().__init__(*args, **kwargs)
        self.alpha_delay_host = self.alpha_delay_vm = self.manager_alpha_delay = variant.alpha

    def reset(self, **kwargs):
        self.policy_instance_seed = int(kwargs.get("seed", self.policy_instance_seed))
        self.policy_rng = np.random.default_rng(self.policy_seed+self.policy_instance_seed)
        return super().reset(**kwargs)

    def _phase_prepare_tasks(self):
        if not self.variant.ordering:
            return super()._phase_prepare_tasks()
        self._add_workflow_if_arrived()
        with self.profiler.stage("fixed_task_sort"):
            self._phase_tasks = ordered_ready_tasks(self, self.variant.ordering)

    def apply_manager_delta(self, delta):
        if self.variant.ordering:
            raise RuntimeError("Fixed task ordering has no TA increment actions")
        return super().apply_manager_delta(delta)

    def can_task_decide(self):
        return (not self.done_flag and not self._phase_preadvanced
            and any(self.task_state[tid] == "Ready" for tid in self.ready_task_ids)
            and np.any(self._vm_idle_mask() & (self.vm_available_at <= self.current_time+1e-12)))

    def apply_task_manager_delta(self, delta):
        if self.variant.suffix != "sync_task" or not self.can_task_decide() or self._cur_tid is not None:
            raise RuntimeError("Task-level TA requires a real unconsumed allocation decision")
        if not self._phase_started:
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.
            self._phase_preadvance_r_energy_total = 0.
            self._phase_preadvance_r_energy_by_host = {int(h): 0. for h in self.host_ids}
        super().apply_manager_delta(delta)
        self._phase_started = True
        self._phase_prepare_tasks()
