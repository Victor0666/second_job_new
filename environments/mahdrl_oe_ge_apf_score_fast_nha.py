"""All physical idle hosts are selectable; only the per-host VM APF remains."""
import numpy as np
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as FastEnv
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OE
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Physical
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.config import ALGORITHM, APF, variant_metadata


def vm_apf_candidates(rows, host_rows, ready_count, now, deadline):
    """Use original VM rules/global inputs, with no host speed or energy pruning."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0., u=0., r_H=None, r_V=1.25,
                    energy_hosts=set(), energy_vms={}, raw_count=0)
    values = np.asarray(list(rows.values()), dtype=np.float64)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError("VM APF needs a current Ready task and finite times")
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
        raise ValueError("Non-finite or negative VM APF duration")
    ids = np.asarray(list(rows), dtype=np.int64)
    hosts = list(host_rows)
    owner = {j: hi for hi, h in enumerate(hosts) for j in host_rows[h]}
    group = np.asarray([owner[j] for j in ids], dtype=np.int64)
    t, energy = values[:, 0], values[:, 1]
    g = len(rows) / (len(rows) + ready_count)
    u = float(np.clip(float(t.min()) / max(deadline-now, APF.time_atol), 0., 1.))
    rv = float(np.clip(APF.vm_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))
    minimum = np.full(len(hosts), np.inf)
    np.minimum.at(minimum, group, t)
    ontime = now + t <= deadline + APF.time_atol
    local_energy = np.full(len(hosts), np.inf)
    np.minimum.at(local_energy, group[ontime], energy[ontime])
    energy_vm = ontime & np.isclose(energy, local_energy[group], rtol=APF.energy_rtol, atol=APF.energy_atol)
    speed_vm = t <= rv*minimum[group] + APF.time_atol
    kept = speed_vm | energy_vm
    present = np.isfinite(minimum)
    result = dict(hosts={hosts[i] for i in np.flatnonzero(present)},
        vms={h: set(ids[(group == i) & kept].tolist()) for i, h in enumerate(hosts) if present[i]},
        energy_hosts=set(),
        energy_vms={h: set(ids[(group == i) & energy_vm].tolist()) for i, h in enumerate(hosts) if present[i]},
        g=g, u=u, r_H=None, r_V=rv, raw_count=len(rows))
    result["arrays"] = dict(ids=ids, group=group, values=values, minimum=minimum,
                            speed_vm=speed_vm, kept=kept)
    return result


class CloudWorkflowEnv_VMAgents(FastEnv):
    algorithm = ALGORITHM

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())

    def _apf_snapshot(self, tid):
        self._ensure_decision(tid)
        if self._apf_cache is not None:
            return self._apf_cache
        with self.profiler.stage("apf_preview_and_filter"):
            rows = {}
            for j in np.flatnonzero(self._vm_idle_mask() & (np.asarray(self.vm_available_at) <= self.current_time + 1e-12)):
                j = int(j)
                vm = self.vms[self.vm_ids[j]]
                duration = float(self._task_duration(tid, j))
                start = max(float(self.current_time), float(self.vm_available_at[j]))
                energy = self.assignment_ledger.preview(vm.host_id, start, start+duration, vm.pc)
                delay = max(0., self.vm_available_at[j]-self.current_time) + duration
                rows[j] = (float(delay), energy, max(0., self.current_time+delay-self.task_baseline_finish[tid]))
            ready = len({j for j in self.ready_task_ids if self.task_state[j] == "Ready"} | {int(tid)})
            snapshot = vm_apf_candidates(rows, self.host_to_vm_indices, ready,
                                         self.current_time, self.task_baseline_finish[tid])
            snapshot["rows"] = rows
            self._apf_cache = snapshot
        return snapshot

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= .5:
            raise ValueError("Host has no legal idle VM")
        return Physical.host_select(self, host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = self._cur_tid, self._cur_host_id
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= .5:
            raise ValueError("VM excluded by VM APF")
        s = self._apf_snapshot(tid)
        a = s["arrays"]
        local = a["group"] == self.host_ids.index(hid)
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        chosen = int(np.flatnonzero(a["ids"] == j)[0])
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]),
            host_apf_enabled=False, vm_apf_enabled=True, g=s["g"], u=s["u"], r_H=None, r_V=s["r_V"],
            raw_vm_count=s["raw_count"], raw_host_count=len(s["vms"]), retained_host_count=len(s["hosts"]),
            local_raw_vm_count=int(local.sum()), local_retained_vm_count=len(s["vms"][hid]),
            energy_rescued_host_count=None,
            local_energy_rescued_vm_count=int((local & a["kept"] & ~a["speed_vm"]).sum()),
            selected_energy_only_host=None, selected_energy_only_vm=not bool(a["speed_vm"][chosen]),
            selected_time_ratio=s["rows"][j][0]/max(float(a["values"][:, 0].min()), 1e-12),
            predicted_ontime=bool(self.current_time+s["rows"][j][0] <= self.task_baseline_finish[tid]+APF.time_atol)))
        try:
            with self.profiler.stage("environment_assignment"):
                return OE.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()
