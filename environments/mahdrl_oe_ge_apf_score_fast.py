"""One-decision observation reuse and vectorized APF; AS/OE still own dynamics."""
import numpy as np
from environments.mahdrl_oe_ge_apf_score import CloudWorkflowEnv_VMAgents as OriginalAS
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OE, POWER_REF_W
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Physical
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast.config import ALGORITHM, APF
from algorithms.mahdrl_oe_ge_apf_score_fast.profiling import StageProfiler


def apf_candidates(rows, host_rows, ready_count, now, deadline):
    """Float64 grouped reductions preserve original legality, tolerances and all ties."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0., u=0., r_H=1.25, r_V=1.25,
                    energy_hosts=set(), energy_vms={}, raw_count=0)
    values = np.asarray(list(rows.values()), dtype=np.float64)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError("APF needs a current Ready task and finite times")
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
        raise ValueError("Non-finite or negative APF duration")
    ids = np.asarray(list(rows), dtype=np.int64)
    hosts = list(host_rows)
    owner = {j: hi for hi, h in enumerate(hosts) for j in host_rows[h]}
    group = np.asarray([owner[j] for j in ids], dtype=np.int64)
    t, energy = values[:, 0], values[:, 1]
    g = len(rows) / (len(rows) + ready_count)
    fastest = float(t.min())
    u = float(np.clip(fastest / max(deadline-now, APF.time_atol), 0., 1.))
    rh = float(np.clip(APF.host_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))
    rv = float(np.clip(APF.vm_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))
    minimum = np.full(len(hosts), np.inf)
    np.minimum.at(minimum, group, t)
    ontime = now + t <= deadline + APF.time_atol
    local_energy = np.full(len(hosts), np.inf)
    np.minimum.at(local_energy, group[ontime], energy[ontime])
    energy_vm = ontime & np.isclose(energy, local_energy[group], rtol=APF.energy_rtol, atol=APF.energy_atol)
    global_energy = ontime & np.isclose(energy, energy[ontime].min() if ontime.any() else np.inf,
                                         rtol=APF.energy_rtol, atol=APF.energy_atol)
    energy_host = np.zeros(len(hosts), dtype=bool)
    np.logical_or.at(energy_host, group, global_energy)
    present = np.isfinite(minimum)
    speed_host = present & (minimum <= rh*fastest + APF.time_atol)
    speed_vm = t <= rv*minimum[group] + APF.time_atol
    kept = speed_vm | energy_vm
    result = dict(hosts={hosts[i] for i in np.flatnonzero(speed_host | energy_host)},
        vms={h: set(ids[(group == i) & kept].tolist()) for i, h in enumerate(hosts) if present[i]},
        energy_hosts={hosts[i] for i in np.flatnonzero(energy_host)},
        energy_vms={h: set(ids[(group == i) & energy_vm].tolist()) for i, h in enumerate(hosts) if present[i]},
        g=g, u=u, r_H=rh, r_V=rv, raw_count=len(rows))
    result["arrays"] = dict(ids=ids, group=group, values=values, minimum=minimum,
                              speed_host=speed_host, speed_vm=speed_vm, kept=kept)
    return result


class CloudWorkflowEnv_VMAgents(OriginalAS):
    algorithm = ALGORITHM

    def __init__(self, *args, profile_mode="sampled", profile_every=100, **kwargs):
        self.profiler = StageProfiler(profile_mode, profile_every)
        self._decision_index = 0
        self._invalidate_decision()
        super().__init__(*args, **kwargs)

    def _invalidate_decision(self):
        self._fast_key = None
        self._fast_cache = {}
        self._apf_cache = None

    def _ensure_decision(self, tid):
        # Also catches explicit deadline/availability edits made by diagnostic callers.
        key = (int(tid), float(self.current_time), len(self._records),
               float(self.task_baseline_finish[tid]), tuple(self.vm_available_at),
               tuple(self.ready_task_ids), len(self.task_state))
        if key != self._fast_key:
            self._invalidate_decision()
            self._fast_key = key
            self.profiler.select(self._decision_index)
            self._decision_index += 1

    def reset(self, **kwargs):
        self._invalidate_decision()
        self._decision_index = 0
        result = super().reset(**kwargs)
        self._invalidate_decision()
        return result

    def _compute_block1(self, tid):
        key = ("block1", tid)
        if key not in self._fast_cache:
            self._fast_cache[key] = super()._compute_block1(tid)
        return self._fast_cache[key]

    def _compute_block2(self):
        if "block2" not in self._fast_cache:
            self._fast_cache["block2"] = super()._compute_block2()
        return self._fast_cache["block2"]

    def _compute_host_block(self):
        if "host_block" not in self._fast_cache:
            self._fast_cache["host_block"] = super()._compute_host_block()
        return self._fast_cache["host_block"]

    def _task_duration(self, task_id, vm_index):
        if self._fast_key is None:
            return super()._task_duration(task_id, vm_index)
        key = ("duration", int(task_id), int(vm_index))
        if key not in self._fast_cache:
            self._fast_cache[key] = super()._task_duration(task_id, vm_index)
        return self._fast_cache[key]

    def _apf_snapshot(self, tid):
        self._ensure_decision(tid)
        if self._apf_cache is not None:
            return self._apf_cache
        with self.profiler.stage("apf_preview_and_filter"):
            rows = {}
            for j in np.flatnonzero(self._vm_idle_mask() & (np.asarray(self.vm_available_at) <= self.current_time + 1e-12)):
                j = int(j)
                vm = self.vms[self.vm_ids[j]]
                # Keep the original scalar execution/ledger arithmetic and operation order.
                duration = float(self._task_duration(tid, j))
                start = max(float(self.current_time), float(self.vm_available_at[j]))
                energy = self.assignment_ledger.preview(vm.host_id, start, start+duration, vm.pc)
                delay = max(0., self.vm_available_at[j]-self.current_time) + duration
                rows[j] = (float(delay), energy, max(0., self.current_time+delay-self.task_baseline_finish[tid]))
            ready = len({j for j in self.ready_task_ids if self.task_state[j] == "Ready"} | {int(tid)})
            s = apf_candidates(rows, self.host_to_vm_indices, ready, self.current_time, self.task_baseline_finish[tid])
            s["rows"] = rows
            self._apf_cache = s
        return s

    def _build_host_obs_for_task(self, tid):
        self._ensure_decision(tid)
        if "host_obs" in self._fast_cache:
            return self._fast_cache["host_obs"]
        with self.profiler.stage("host_observation"):
            original, original_mask = Physical._build_host_obs_for_task(self, tid)
            s = self._apf_snapshot(tid)
            ref = self._obs_ref("T_ref", self.task_baseline_norm)
            summary = np.zeros((self.num_hosts, 5), np.float32)
            mask = np.asarray([h in s["hosts"] for h in self.host_ids], np.float32) * original_mask
            if s["rows"]:
                a = s["arrays"]
                group, values = a["group"], a["values"]
                kept = a["kept"] & (mask[group] > .5)
                t, e, late = values[:, 0], values[:, 1], values[:, 2]
                fast_t, cheap_e, min_late = (np.full(self.num_hosts, np.inf) for _ in range(3))
                for dest, data in ((fast_t, t), (cheap_e, e), (min_late, late)):
                    np.minimum.at(dest, group[kept], data[kept])
                fast_e, cheap_t = np.full(self.num_hosts, np.inf), np.full(self.num_hosts, np.inf)
                fastest = kept & (np.abs(t-fast_t[group]) <= APF.time_atol)
                cheapest = kept & np.isclose(e, cheap_e[group], rtol=APF.energy_rtol, atol=APF.energy_atol)
                np.minimum.at(fast_e, group[fastest], e[fastest])
                np.minimum.at(cheap_t, group[cheapest], t[cheapest])
                legal = mask > .5
                summary[legal] = np.column_stack((fast_t/ref, fast_e/(POWER_REF_W*ref),
                    min_late/ref, cheap_e/(POWER_REF_W*ref), cheap_t/ref))[legal]
            obs = np.concatenate((original, [(self.task_baseline_finish[tid]-self.current_time)/ref], summary.ravel())).astype(np.float32)
            obs.flags.writeable = mask.flags.writeable = False
            self._fast_cache["host_obs"] = obs, mask
        return obs, mask

    def _build_vm_obs_for_task_host(self, tid, host_id):
        self._ensure_decision(tid)
        key = ("vm_obs", host_id)
        if key not in self._fast_cache:
            with self.profiler.stage("vm_observation"):
                # Reuses block1/block2/host features and keeps the original power model.
                obs, original_mask = OE._build_vm_obs_for_task_host(self, tid, host_id)
                s = self._apf_snapshot(tid)
                mask = np.zeros_like(original_mask)
                if host_id in s["hosts"]:
                    indices = self.host_to_vm_indices[host_id]
                    mask[:len(indices)] = ((original_mask[:len(indices)] > .5)
                        & np.isin(indices, list(s["vms"][host_id]))).astype(np.float32)
                start = self.base_vm_obs_dim + 1 + self.num_hosts + 2
                obs[start:].reshape(self.max_vms_per_host, 2)[mask <= .5] = 0.
                obs.flags.writeable = mask.flags.writeable = False
                self._fast_cache[key] = obs, mask
        return self._fast_cache[key]

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= .5:
            raise ValueError("Host excluded by APF")
        return Physical.host_select(self, host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = self._cur_tid, self._cur_host_id
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= .5:
            raise ValueError("VM excluded by APF")
        s = self._apf_snapshot(tid)
        a = s["arrays"]
        hi = self.host_ids.index(hid)
        local = a["group"] == hi
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        chosen = int(np.flatnonzero(a["ids"] == j)[0])
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]),
            g=s["g"], u=s["u"], r_H=s["r_H"], r_V=s["r_V"], raw_vm_count=s["raw_count"],
            raw_host_count=len(s["vms"]), retained_host_count=len(s["hosts"]),
            local_raw_vm_count=int(local.sum()), local_retained_vm_count=len(s["vms"][hid]),
            energy_rescued_host_count=len(s["hosts"])-int(a["speed_host"].sum()),
            local_energy_rescued_vm_count=int((local & a["kept"] & ~a["speed_vm"]).sum()),
            selected_energy_only_host=not bool(a["speed_host"][hi]),
            selected_energy_only_vm=not bool(a["speed_vm"][chosen]),
            selected_time_ratio=s["rows"][j][0]/max(float(a["values"][:, 0].min()), 1e-12),
            predicted_ontime=bool(self.current_time+s["rows"][j][0] <= self.task_baseline_finish[tid]+APF.time_atol)))
        try:
            with self.profiler.stage("environment_assignment"):
                return OE.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        try:
            return super()._assign_task_to_specific_vm(task_id, vm_index)
        finally:
            self._invalidate_decision()

    def finish_phase_and_advance(self):
        self._invalidate_decision()
        try:
            with self.profiler.stage("environment_phase_advance"):
                return super().finish_phase_and_advance()
        finally:
            self._invalidate_decision()

    def apply_manager_delta(self, delta_vec):
        self._invalidate_decision()
        return super().apply_manager_delta(delta_vec)

    def set_manager_combo(self, weights):
        self._invalidate_decision()
        return super().set_manager_combo(weights)

    def _add_workflow_if_arrived(self):
        # No observation can survive an arrival or completion/advance boundary.
        self._invalidate_decision()
        return super()._add_workflow_if_arrived()

    def _advance_until_decision_energy_only(self, return_breakdown=False):
        self._invalidate_decision()
        try:
            return super()._advance_until_decision_energy_only(return_breakdown=return_breakdown)
        finally:
            self._invalidate_decision()

    def _process_finish_events_at_current_time(self):
        self._invalidate_decision()
        try:
            return super()._process_finish_events_at_current_time()
        finally:
            self._invalidate_decision()
