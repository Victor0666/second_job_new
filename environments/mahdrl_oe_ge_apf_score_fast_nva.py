"""Keep host APF; every legal idle VM on the selected host remains selectable."""
import numpy as np
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as FastEnv
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OE
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_nva.config import ALGORITHM, APF, variant_metadata


def host_apf_candidates(rows, host_rows, ready_count, now, deadline):
    """Original host speed/energy rules with no local VM filtering or threshold."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0., u=0., r_H=1.25, r_V=None,
                    energy_hosts=set(), energy_vms={}, raw_count=0)
    values = np.asarray(list(rows.values()), dtype=np.float64)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError("Host APF needs a current Ready task and finite times")
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
        raise ValueError("Non-finite or negative host APF duration")
    ids = np.asarray(list(rows), dtype=np.int64)
    hosts = list(host_rows)
    owner = {j: hi for hi, h in enumerate(hosts) for j in host_rows[h]}
    group = np.asarray([owner[j] for j in ids], dtype=np.int64)
    t, energy = values[:, 0], values[:, 1]
    g = len(rows) / (len(rows) + ready_count)
    fastest = float(t.min())
    u = float(np.clip(fastest / max(deadline-now, APF.time_atol), 0., 1.))
    rh = float(np.clip(APF.host_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))
    minimum = np.full(len(hosts), np.inf)
    np.minimum.at(minimum, group, t)
    ontime = now + t <= deadline + APF.time_atol
    global_energy = ontime & np.isclose(energy, energy[ontime].min() if ontime.any() else np.inf,
                                         rtol=APF.energy_rtol, atol=APF.energy_atol)
    energy_host = np.zeros(len(hosts), dtype=bool)
    np.logical_or.at(energy_host, group, global_energy)
    present = np.isfinite(minimum)
    speed_host = present & (minimum <= rh*fastest + APF.time_atol)
    result = dict(hosts={hosts[i] for i in np.flatnonzero(speed_host | energy_host)},
        vms={h: set(ids[group == i].tolist()) for i, h in enumerate(hosts) if present[i]},
        energy_hosts={hosts[i] for i in np.flatnonzero(energy_host)}, energy_vms={},
        g=g, u=u, r_H=rh, r_V=None, raw_count=len(rows))
    result["arrays"] = dict(ids=ids, group=group, values=values, minimum=minimum,
                            speed_host=speed_host, kept=np.ones(len(rows), dtype=bool))
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
            snapshot = host_apf_candidates(rows, self.host_to_vm_indices, ready,
                                           self.current_time, self.task_baseline_finish[tid])
            snapshot["rows"] = rows
            self._apf_cache = snapshot
        return snapshot

    def vm_assign(self, vm_slot_index):
        tid, hid = self._cur_tid, self._cur_host_id
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= .5:
            raise ValueError("VM is not a legal idle slot in the selected APF host")
        s = self._apf_snapshot(tid)
        a = s["arrays"]
        hi = self.host_ids.index(hid)
        local = a["group"] == hi
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]),
            host_apf_enabled=True, vm_apf_enabled=False, g=s["g"], u=s["u"], r_H=s["r_H"], r_V=None,
            raw_vm_count=s["raw_count"], raw_host_count=len(s["vms"]), retained_host_count=len(s["hosts"]),
            local_raw_vm_count=int(local.sum()), local_retained_vm_count=len(s["vms"][hid]),
            energy_rescued_host_count=len(s["hosts"])-int(a["speed_host"].sum()),
            local_energy_rescued_vm_count=None, selected_energy_only_host=not bool(a["speed_host"][hi]),
            selected_energy_only_vm=None,
            selected_time_ratio=s["rows"][j][0]/max(float(a["values"][:, 0].min()), 1e-12),
            predicted_ontime=bool(self.current_time+s["rows"][j][0] <= self.task_baseline_finish[tid]+APF.time_atol)))
        try:
            with self.profiler.stage("environment_assignment"):
                return OE.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()
