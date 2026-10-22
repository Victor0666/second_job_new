"""Performance candidate masks for both rollout and replay successor actions."""
from common.mahdrl_performance import candidate_durations, performance_mask
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as _Baseline
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE


class CloudWorkflowEnv_VMAgents(_Baseline):
    def _build_host_obs_for_task(self, tid):
        obs, mask = super()._build_host_obs_for_task(tid)
        # The original builder also serves callers that preview a task directly.
        times = candidate_durations(self, "host", mask, task_id=tid)
        return obs, performance_mask(mask, times)

    def _build_vm_obs_for_task_host(self, tid, host_id):
        obs, mask = super()._build_vm_obs_for_task_host(tid, host_id)
        times = candidate_durations(self, "vm", mask, task_id=tid, host_id=host_id)
        return obs, performance_mask(mask, times)

    def host_select(self, host_index):
        if self._cur_tid is not None:
            _, mask = self._build_host_obs_for_task(self._cur_tid)
            if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= 0.5:
                raise ValueError("Host excluded by the performance candidate mask")
        return super().host_select(host_index)

    def vm_assign(self, vm_slot_index):
        if self._cur_tid is not None and self._cur_host_id is not None:
            _, mask = self._build_vm_obs_for_task_host(self._cur_tid, self._cur_host_id)
            if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= 0.5:
                raise ValueError("VM excluded by the performance candidate mask")
        return super().vm_assign(vm_slot_index)
