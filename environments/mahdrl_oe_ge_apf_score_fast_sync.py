"""Fast ablation scheduling with no host or VM APF and original OE accounting."""
import time
import numpy as np
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as FastEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.config import get_config


class CloudWorkflowEnv_VMAgents(FastEnv):
    def __init__(self, *args, sync_mode="task", **kwargs):
        self.sync_config = get_config(sync_mode)
        self.algorithm = self.sync_config.algorithm
        self._reset_sort_statistics()
        super().__init__(*args, **kwargs)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **self.sync_config.variant_metadata())

    def _reset_sort_statistics(self):
        self.sort_calls = self.sort_nonempty = self.sort_nontrivial = 0
        self.sort_seconds = 0.
        self.last_sort_ready_count = 0

    def reset(self, **kwargs):
        self._reset_sort_statistics()
        return super().reset(**kwargs)

    def _phase_prepare_tasks(self):
        started = time.perf_counter()
        super()._phase_prepare_tasks()
        self.sort_seconds += time.perf_counter()-started
        self.sort_calls += 1
        self.last_sort_ready_count = count = len(self._phase_tasks)
        self.sort_nonempty += int(count > 0)
        self.sort_nontrivial += int(count > 1)

    def can_task_decide(self):
        return (not self.done_flag and not self._phase_preadvanced
            and any(self.task_state[tid] == "Ready" for tid in self.ready_task_ids)
            and np.any(self._vm_idle_mask() & (self.vm_available_at <= self.current_time+1e-12)))

    def apply_task_manager_delta(self, delta):
        if self.sync_config.mode != "task" or not self.can_task_decide() or self._cur_tid is not None:
            raise RuntimeError("Task-level TA requires a real unconsumed allocation decision")
        if not self._phase_started:
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.
            self._phase_preadvance_r_energy_total = 0.
            self._phase_preadvance_r_energy_by_host = {int(h): 0. for h in self.host_ids}
        # Base weight validation clears the queue, but must not start a new physical phase.
        super().apply_manager_delta(delta)
        self._phase_started = True
        self._phase_prepare_tasks()

    def sorting_report(self, assignments):
        return dict(calls=self.sort_calls, nonempty_calls=self.sort_nonempty,
            nontrivial_calls=self.sort_nontrivial, total_seconds=self.sort_seconds,
            calls_per_1000_assignments=1000*self.sort_nonempty/max(1, assignments),
            seconds_per_assignment=self.sort_seconds/max(1, assignments))
