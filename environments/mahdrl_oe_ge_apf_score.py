"""Read-only APF previews and set observations; OE owns execution and rewards."""
import numpy as np
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as _OE, POWER_REF_W
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as _Physical, spaces
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score.config import APF, ALGORITHM, variant_metadata


class VariableArraySpace(spaces.Space):
    """Variable first dimension, compatible with both installed Gym and Gymnasium."""
    def __init__(self, tail=(), binary=False):
        super().__init__(shape=None, dtype=np.float32)
        self.tail, self.binary = tuple(tail), binary

    def sample(self, mask=None):
        return np.zeros((0, *self.tail), np.float32)

    def contains(self, x):
        return (isinstance(x, np.ndarray) and x.dtype == np.float32
                and x.ndim == len(self.tail)+1 and x.shape[1:] == self.tail
                and np.isfinite(x).all() and (not self.binary or np.isin(x, (0, 1)).all()))


def apf_candidates(rows, host_rows, ready_count, now, deadline):
    """rows: global VM index -> (completion delay, marginal joules, tardiness)."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0., u=0., r_H=1.25, r_V=1.25,
                    energy_hosts=set(), energy_vms={}, raw_count=0)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError("APF needs a current Ready task and finite times")
    if not all(np.isfinite(row).all() and row[0] >= 0 for row in rows.values()):
        raise ValueError("Non-finite or negative APF duration")
    g = len(rows) / (len(rows) + ready_count)
    fastest = min(row[0] for row in rows.values())
    u = float(np.clip(fastest / max(deadline - now, APF.time_atol), 0., 1.))
    rh = float(np.clip(APF.host_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))
    rv = float(np.clip(APF.vm_base + APF.spare_weight*g - APF.urgency_weight*u, APF.lower, APF.upper))

    def energy_choices(indices):
        ontime = [j for j in indices if now + rows[j][0] <= deadline + APF.time_atol]
        if not ontime:
            return set()
        minimum = min(rows[j][1] for j in ontime)
        return {j for j in ontime if np.isclose(rows[j][1], minimum,
                    rtol=APF.energy_rtol, atol=APF.energy_atol)}

    global_energy = energy_choices(rows)
    hosts, kept, energy_vms, energy_hosts = set(), {}, {}, set()
    for hid, indices in host_rows.items():
        raw = set(indices).intersection(rows)
        if not raw:
            continue
        th = min(rows[j][0] for j in raw)
        energy_vms[hid] = energy_choices(raw)
        kept[hid] = {j for j in raw if rows[j][0] <= rv*th + APF.time_atol} | energy_vms[hid]
        if raw.intersection(global_energy):
            energy_hosts.add(hid)
        if th <= rh*fastest + APF.time_atol:
            hosts.add(hid)
    return dict(hosts=hosts | energy_hosts, vms=kept, g=g, u=u, r_H=rh, r_V=rv,
                energy_hosts=energy_hosts, energy_vms=energy_vms, raw_count=len(rows))


class CloudWorkflowEnv_VMAgents(_OE):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        self._apf_cache = None
        super().__init__(*args, **kwargs)
        # Internal flat builders remain compatible with the original event engine.
        def obs_space(c, f):
            return spaces.Dict({"context": spaces.Box(-np.inf, np.inf, (c,), np.float32),
                "candidates": VariableArraySpace((f,)),
                "exists_mask": VariableArraySpace(binary=True)})
        self.observation_space = spaces.Dict({
            "host_obs": obs_space(23, 10), "host_mask": VariableArraySpace(binary=True),
            "vm_obs": obs_space(30, 7), "vm_mask": VariableArraySpace(binary=True)})
        self.apf_decisions = []

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())

    @staticmethod
    def _empty(c, f):
        return {"obs": {"context": np.zeros(c, np.float32),
                "candidates": np.zeros((0, f), np.float32), "exists_mask": np.zeros(0, np.float32)},
                "mask": np.zeros(0, np.float32), "candidate_ids": [],
                "performance_times": np.zeros(0, np.float64)}

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self._apf_cache = None
        self.apf_decisions = []
        h, v = self._empty(23, 10), self._empty(30, 7)
        return dict(host_obs=h["obs"], host_mask=h["mask"], vm_obs=v["obs"], vm_mask=v["mask"]), info

    def _apf_snapshot(self, tid):
        key = (int(tid), float(self.current_time), len(self._records),
               float(self.task_baseline_finish[tid]), tuple(self.vm_available_at), tuple(self.ready_task_ids))
        if self._apf_cache is not None and self._apf_cache[0] == key:
            return self._apf_cache[1]
        raw = _OE._candidate_rows(self, tid)
        # Respect both the original HA (1e-9) and VA (1e-12) idle checks.
        rows = {}
        for j, (_, energy, _) in raw.items():
            if self.vm_available_at[j] <= self.current_time + 1e-12:
                duration = max(0., self.vm_available_at[j] - self.current_time) + self._task_duration(tid, j)
                rows[j] = (float(duration), energy,
                           max(0., self.current_time + duration - self.task_baseline_finish[tid]))
        ready = len({j for j in self.ready_task_ids if self.task_state[j] == "Ready"} | {int(tid)})
        result = apf_candidates(rows, self.host_to_vm_indices, ready,
                                self.current_time, self.task_baseline_finish[tid])
        result["rows"] = rows
        self._apf_cache = key, result
        return result

    def _candidate_rows(self, tid):
        s = self._apf_snapshot(tid)
        return {j: s["rows"][j] for h in s["hosts"] for j in s["vms"][h]}

    def _build_host_obs_for_task(self, tid):
        original, original_mask = _Physical._build_host_obs_for_task(self, tid)
        s = self._apf_snapshot(tid)
        time_ref = self._obs_ref("T_ref", self.task_baseline_norm)
        summary = np.zeros((self.num_hosts, 5), np.float32)
        mask = np.zeros_like(original_mask)
        for hi, hid in enumerate(self.host_ids):
            if hid not in s["hosts"] or original_mask[hi] <= .5:
                continue
            mask[hi] = 1.
            values = [s["rows"][j] for j in s["vms"][hid]]
            fast_t, cheap_e = min(x[0] for x in values), min(x[1] for x in values)
            fast_e = min(x[1] for x in values if abs(x[0]-fast_t) <= APF.time_atol)
            cheap_t = min(x[0] for x in values if np.isclose(x[1], cheap_e,
                          rtol=APF.energy_rtol, atol=APF.energy_atol))
            summary[hi] = [fast_t/time_ref, fast_e/(POWER_REF_W*time_ref),
                           min(x[2] for x in values)/time_ref,
                           cheap_e/(POWER_REF_W*time_ref), cheap_t/time_ref]
        obs = np.concatenate((original, [(self.task_baseline_finish[tid]-self.current_time)/time_ref],
                              summary.ravel())).astype(np.float32)
        return obs, mask

    def _build_vm_obs_for_task_host(self, tid, host_id):
        obs, original_mask = _OE._build_vm_obs_for_task_host(self, tid, host_id)
        s = self._apf_snapshot(tid)
        mask = np.zeros_like(original_mask)
        if host_id in s["hosts"]:
            for slot, j in enumerate(self.host_to_vm_indices[host_id]):
                mask[slot] = float(original_mask[slot] > .5 and j in s["vms"][host_id])
        start = self.base_vm_obs_dim + 1 + self.num_hosts + 2
        obs[start:].reshape(self.max_vms_per_host, 2)[mask <= .5] = 0.
        return obs, mask

    def get_host_state_for_next_assignment(self):
        state, ok = super().get_host_state_for_next_assignment()
        if not ok:
            return self._empty(23, 10), False
        x, s = state["obs"], self._apf_snapshot(self._cur_tid)
        context = np.concatenate((x[:22], x[self.base_host_obs_dim:self.base_host_obs_dim+1]))
        candidates = np.concatenate((x[22:self.base_host_obs_dim].reshape(self.num_hosts, 5),
                                    x[self.base_host_obs_dim+1:].reshape(self.num_hosts, 5)), 1)
        times = np.array([min((s["rows"][j][0] for j in indices if j in s["rows"]), default=np.inf)
                          for indices in (self.host_to_vm_indices[h] for h in self.host_ids)])
        return dict(obs=dict(context=context, candidates=candidates,
                             exists_mask=np.ones(self.num_hosts, np.float32)),
                    mask=state["mask"], candidate_ids=list(self.host_ids), performance_times=times), True

    def get_vm_state_for_current_task(self):
        state, ok = super().get_vm_state_for_current_task()
        if not ok:
            return self._empty(30, 7), False
        x, b, h = state["obs"], self.base_vm_obs_dim, self._cur_host_id
        indices = self.host_to_vm_indices[h]
        k = len(indices)
        context = np.concatenate((x[:27], x[b:b+1], x[b+1+self.num_hosts:b+3+self.num_hosts]))
        candidates = np.concatenate((x[27:b].reshape(self.max_vms_per_host, 5)[:k],
                    x[b+3+self.num_hosts:].reshape(self.max_vms_per_host, 2)[:k]), 1)
        s = self._apf_snapshot(self._cur_tid)
        return dict(obs=dict(context=context, candidates=candidates, exists_mask=np.ones(k, np.float32)),
                    mask=state["mask"][:k], candidate_ids=[self.vm_ids[j] for j in indices],
                    performance_times=np.array([s["rows"].get(j, (np.inf,))[0] for j in indices])), True

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= .5:
            raise ValueError("Host excluded by APF")
        return super().host_select(host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = self._cur_tid, self._cur_host_id
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= .5:
            raise ValueError("VM excluded by APF")
        s = self._apf_snapshot(tid)
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        th = min(s["rows"][i][0] for i in self.host_to_vm_indices[hid] if i in s["rows"])
        speed_hosts = {h for h in s["hosts"] if min(s["rows"][i][0] for i in self.host_to_vm_indices[h]
                            if i in s["rows"]) <= s["r_H"]*min(x[0] for x in s["rows"].values())+APF.time_atol}
        speed_vms = {i for i in s["vms"][hid] if s["rows"][i][0] <= s["r_V"]*th+APF.time_atol}
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]),
            g=s["g"], u=s["u"], r_H=s["r_H"], r_V=s["r_V"], raw_vm_count=s["raw_count"],
            raw_host_count=len(s["vms"]), retained_host_count=len(s["hosts"]),
            local_raw_vm_count=sum(i in s["rows"] for i in self.host_to_vm_indices[hid]),
            local_retained_vm_count=len(s["vms"][hid]),
            energy_rescued_host_count=len(s["hosts"]-speed_hosts),
            local_energy_rescued_vm_count=len(s["vms"][hid]-speed_vms),
            selected_energy_only_host=hid not in speed_hosts,
            selected_energy_only_vm=j not in speed_vms,
            selected_time_ratio=s["rows"][j][0]/max(min(x[0] for x in s["rows"].values()), 1e-12),
            predicted_ontime=bool(self.current_time+s["rows"][j][0] <= self.task_baseline_finish[tid]+APF.time_atol)))
        result = super().vm_assign(vm_slot_index)
        self._apf_cache = None
        return result
