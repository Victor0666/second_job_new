"""Fast set observations over every physically legal idle resource; no APF."""
import numpy as np
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as FastEnv
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OE
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Physical
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from common.mahdrl_ablation import no_apf_metadata


class CloudWorkflowEnv_VMAgents(FastEnv):
    def variant_metadata(self):
        return no_apf_metadata(dict(super().variant_metadata(), algorithm=self.algorithm))

    def _apf_snapshot(self, tid):
        # Keep the inherited observation interface, but never run an APF filter.
        self._ensure_decision(tid)
        if self._apf_cache is not None:
            return self._apf_cache
        with self.profiler.stage("raw_legal_candidate_preview"):
            rows = {}
            idle = self._vm_idle_mask() & (np.asarray(self.vm_available_at) <= self.current_time + 1e-12)
            for j in np.flatnonzero(idle):
                j = int(j)
                vm = self.vms[self.vm_ids[j]]
                duration = float(self._task_duration(tid, j))
                start = max(float(self.current_time), float(self.vm_available_at[j]))
                energy = self.assignment_ledger.preview(vm.host_id, start, start + duration, vm.pc)
                delay = max(0., self.vm_available_at[j] - self.current_time) + duration
                rows[j] = (float(delay), energy, max(0., self.current_time + delay - self.task_baseline_finish[tid]))
            ids = np.asarray(list(rows), dtype=np.int64)
            values = np.asarray(list(rows.values()), dtype=np.float64).reshape(-1, 3)
            if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
                raise ValueError("Non-finite or negative raw candidate prediction")
            owner = {j: hi for hi, hid in enumerate(self.host_ids) for j in self.host_to_vm_indices[hid]}
            group = np.asarray([owner[int(j)] for j in ids], dtype=np.int64)
            vms = {hid: set(indices).intersection(rows) for hid, indices in self.host_to_vm_indices.items()}
            vms = {hid: indices for hid, indices in vms.items() if indices}
            self._apf_cache = dict(hosts=set(vms), vms=vms, rows=rows, raw_count=len(rows),
                r_H=None, r_V=None, host_apf_enabled=False, vm_apf_enabled=False,
                energy_hosts=set(), energy_vms={},
                arrays=dict(ids=ids, group=group, values=values, kept=np.ones(len(ids), dtype=bool)))
        return self._apf_cache

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= .5:
            raise ValueError("Host has no physically legal idle VM")
        return Physical.host_select(self, host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = self._cur_tid, self._cur_host_id
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= .5:
            raise ValueError("VM is not a physically legal idle slot")
        snapshot = self._apf_snapshot(tid)
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]),
            host_apf_enabled=False, vm_apf_enabled=False, r_H=None, r_V=None,
            raw_vm_count=len(snapshot["rows"]), raw_host_count=len(snapshot["hosts"]),
            retained_host_count=len(snapshot["hosts"]), local_raw_vm_count=len(snapshot["vms"][hid]),
            local_retained_vm_count=len(snapshot["vms"][hid]),
            selected_energy_only_host=None, selected_energy_only_vm=None,
            predicted_ontime=bool(self.current_time + snapshot["rows"][j][0]
                                  <= self.task_baseline_finish[tid] + 1e-9)))
        try:
            with self.profiler.stage("environment_assignment"):
                return OE.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()
