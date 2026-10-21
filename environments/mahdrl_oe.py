"""Isolated MAHDRL variant: observations 1/2/3 and assignment energy reward 5.

The original FCFS environment still owns physical execution and TA rewards.
This module neither filters deadline actions nor changes the task ordering.
"""
from bisect import bisect_right

import numpy as np

from environments.mahdrl_fcfs import (
    CloudWorkflowEnv_VMAgents as OriginalEnv, MANAGER_ACTION_TABLE,
)
from environments.mahdrl_heft import spaces
from common.f18_state_normalization import checked_observation

VARIANT_SCHEME = "mahdrl_oe_1235_v1"
POWER_REF_W = 100.0
HOST_SUMMARY_FIELDS = ("fastest_duration", "fastest_energy", "min_tardiness",
                       "min_energy", "min_energy_duration")
VM_EXTRA_FIELDS = ("predicted_tardiness", "instant_power_delta")


class AssignmentEnergyLedger:
    """Exact piecewise-load integration, with side-effect-free candidate previews.

    Like PD3QN's interval accumulator, segments include zero-load gaps inside
    used host windows. Outside a host's own window there is no baseline cost.
    """
    def __init__(self, hosts):
        self.hosts = hosts
        self.segments = {int(h): [] for h in hosts}
        self._segment_ends = {int(h): [] for h in hosts}
        self.total_energy = 0.0

    def _energy(self, host_id, load, duration):
        host = self.hosts[int(host_id)]
        return float(host.power(load / max(host.total_pc, 1e-12)) * duration)

    def _addition(self, host_id, start, end, added_pc):
        start, end, added_pc = float(start), float(end), float(added_pc)
        if not np.isfinite([start, end, added_pc]).all() or end <= start or added_pc <= 0:
            raise ValueError("Assignment energy requires a finite positive load interval")
        segs = self.segments[int(host_id)]
        extension = 0.0
        if segs:
            left, right = segs[0][0], segs[-1][1]
            if start < left:
                segs = [(start, left, 0.)] + segs
                extension += self._energy(host_id, 0., left - start)
            if end > right:
                segs = segs + [(right, end, 0.)]
                extension += self._energy(host_id, 0., end - right)
        updated = []
        index = 0
        while index < len(segs) and segs[index][1] <= start:
            updated.append(segs[index])
            index += 1
        current, delta = start, extension
        while current < end:
            if index >= len(segs) or segs[index][0] >= end:
                right = end if index >= len(segs) else min(end, segs[index][0])
                updated.append((current, right, added_pc))
                delta += self._energy(host_id, added_pc, right - current)
                current = right
                continue
            left, right, load = segs[index]
            if left > current:
                stop = min(end, left)
                updated.append((current, stop, added_pc))
                delta += self._energy(host_id, added_pc, stop - current)
                current = stop
                continue
            if left < current:
                updated.append((left, current, load))
            stop = min(end, right)
            delta += (self._energy(host_id, load + added_pc, stop - current)
                      - self._energy(host_id, load, stop - current))
            updated.append((current, stop, load + added_pc))
            current = stop
            if stop < right:
                updated.append((stop, right, load))
            index += 1
        updated.extend(segs[index:])
        merged = []
        for left, right, load in updated:
            if right <= left:
                continue
            if merged and merged[-1][1] == left and merged[-1][2] == load:
                merged[-1] = (merged[-1][0], right, load)
            else:
                merged.append((left, right, load))
        return float(delta), merged

    def preview(self, host_id, start, end, added_pc):
        start, end, added_pc = float(start), float(end), float(added_pc)
        if not np.isfinite([start, end, added_pc]).all() or end <= start or added_pc <= 0:
            raise ValueError("Assignment energy requires a finite positive load interval")
        segs = self.segments[int(host_id)]
        if not segs:
            return self._energy(host_id, added_pc, end - start)
        left, right = segs[0][0], segs[-1][1]
        # Include a new window's idle baseline, including any gap before this task.
        delta = self._energy(host_id, 0., max(0., left - start) + max(0., end - right))
        extra_duration = max(0., min(end, left) - start) + max(0., end - max(start, right))
        delta += (self._energy(host_id, added_pc, extra_duration)
                  - self._energy(host_id, 0., extra_duration))
        # Skip the historical prefix: candidate evaluation only needs intersecting loads.
        index = bisect_right(self._segment_ends[int(host_id)], start)
        while index < len(segs) and segs[index][0] < end:
            a, b, load = segs[index]
            duration = max(0., min(b, end) - max(a, start))
            delta += (self._energy(host_id, load + added_pc, duration)
                      - self._energy(host_id, load, duration))
            index += 1
        return float(delta)

    def add(self, host_id, start, end, added_pc):
        delta, updated = self._addition(host_id, start, end, added_pc)
        self.segments[int(host_id)] = updated
        self._segment_ends[int(host_id)] = [item[1] for item in updated]
        self.total_energy += delta
        return delta


class CloudWorkflowEnv_VMAgents(OriginalEnv):
    """Original dynamics with appended observations and a separate energy ledger."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_host_obs_dim = self.host_obs_dim
        self.base_vm_obs_dim = self.vm_obs_dim
        self.host_obs_dim += 1 + 5 * self.num_hosts
        self.vm_obs_dim += 1 + self.num_hosts + 2 + 2 * self.max_vms_per_host
        self.observation_space = spaces.Dict({
            "host_obs": spaces.Box(-np.inf, np.inf, (self.host_obs_dim,), np.float32),
            "host_mask": spaces.Box(0., 1., (self.host_act_dim,), np.float32),
            "vm_obs": spaces.Box(-np.inf, np.inf, (self.vm_obs_dim,), np.float32),
            "vm_mask": spaces.Box(0., 1., (self.vm_act_dim,), np.float32),
        })
        self._reset_assignment_ledger()

    def _reset_assignment_ledger(self):
        self.assignment_ledger = AssignmentEnergyLedger(self.hosts)
        self.assignment_energy_total = 0.0
        self.lower_energy_emitted = 0.0
        self._candidate_cache = {}

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._reset_assignment_ledger()
        return result

    def variant_metadata(self):
        time_ref = float(self._obs_ref("T_ref", self.task_baseline_norm))
        return dict(scheme=VARIANT_SCHEME, changes=[1, 2, 3, 5],
                    deadline_action_filter=False, time_ref_s=time_ref,
                    power_ref_w=POWER_REF_W, energy_ref_j=POWER_REF_W * time_ref,
                    host_summary_fields=list(HOST_SUMMARY_FIELDS),
                    vm_extra_fields=list(VM_EXTRA_FIELDS))

    def estimate_assignment_energy(self, tid, vm_index):
        vm = self.vms[self.vm_ids[int(vm_index)]]
        start = max(float(self.current_time), float(self.vm_available_at[int(vm_index)]))
        end = start + self._task_duration(int(tid), int(vm_index))
        return self.assignment_ledger.preview(vm.host_id, start, end, vm.pc)

    def _candidate_rows(self, tid):
        # The cache holds one actual decision; previews never assign tasks or advance time.
        key = (len(self._records), float(self.current_time), int(tid),
               float(self.task_baseline_finish[tid]))
        if key not in self._candidate_cache:
            rows = {}
            for j in np.flatnonzero(self._vm_idle_mask()):
                j = int(j)
                duration = float(self._task_duration(tid, j))
                lateness = max(0., self.current_time + duration - self.task_baseline_finish[tid])
                rows[j] = (duration, self.estimate_assignment_energy(tid, j), lateness)
            self._candidate_cache = {key: rows}
        return self._candidate_cache[key]

    def _build_host_obs_for_task(self, tid):
        original, mask = super()._build_host_obs_for_task(tid)
        metadata = self.variant_metadata()
        time_ref, energy_ref = metadata["time_ref_s"], metadata["energy_ref_j"]
        candidates = self._candidate_rows(tid)
        summary = np.zeros((self.num_hosts, 5), dtype=np.float64)
        for hi, host_id in enumerate(self.host_ids):
            indices = [j for j in self.host_to_vm_indices[host_id] if j in candidates]
            if not indices:
                continue
            fastest = min(indices, key=lambda j: (candidates[j][0], j))
            cheapest = min(indices, key=lambda j: (candidates[j][1], j))
            fast_t, fast_e, _ = candidates[fastest]
            cheap_t, cheap_e, _ = candidates[cheapest]
            summary[hi] = [fast_t / time_ref, fast_e / energy_ref,
                           min(candidates[j][2] for j in indices) / time_ref,
                           cheap_e / energy_ref, cheap_t / time_ref]
        remaining = (self.task_baseline_finish[tid] - self.current_time) / time_ref
        obs = np.concatenate((original, [remaining], summary.reshape(-1))).astype(np.float32)
        return checked_observation(self, obs), mask

    def _build_vm_obs_for_task_host(self, tid, host_id):
        original, mask = super()._build_vm_obs_for_task_host(tid, host_id)
        time_ref = self.variant_metadata()["time_ref_s"]
        now, host = float(self.current_time), self.hosts[host_id]
        active_pc = sum(self.vms[self.vm_ids[j]].pc for j in self.host_to_vm_indices[host_id]
                        if self.vm_available_at[j] > now + 1e-12)
        current_load = active_pc / max(host.total_pc, 1e-12)
        current_power = float(host.power(current_load))
        host_identity = np.zeros(self.num_hosts, dtype=np.float32)
        host_identity[self.host_ids.index(host_id)] = 1.
        extra = np.zeros((self.max_vms_per_host, 2), dtype=np.float64)
        for slot, j in enumerate(self.host_to_vm_indices[host_id]):
            vm = self.vms[self.vm_ids[j]]
            finish = max(now, float(self.vm_available_at[j])) + self._task_duration(tid, j)
            extra[slot, 0] = max(0., finish - self.task_baseline_finish[tid]) / time_ref
            if mask[slot] > .5:
                after = host.power((active_pc + vm.pc) / max(host.total_pc, 1e-12))
                extra[slot, 1] = (after - current_power) / POWER_REF_W
        remaining = (self.task_baseline_finish[tid] - now) / time_ref
        obs = np.concatenate((original, [remaining], host_identity,
                              [host.power(0.) / POWER_REF_W, current_power / POWER_REF_W],
                              extra.reshape(-1))).astype(np.float32)
        return checked_observation(self, obs), mask

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        # Covers normal actions and the original environment's forced-assignment fuse.
        super()._assign_task_to_specific_vm(task_id, vm_index)
        record = self._records[-1]
        self._last_assignment_energy = self.assignment_ledger.add(
            record.server_id, record.start_time, record.end_time, record.vm_pc)
        self.assignment_energy_total = self.assignment_ledger.total_energy
        self._candidate_cache.clear()

    def vm_assign(self, vm_slot_index):
        _old_host, _old_vm, info = super().vm_assign(vm_slot_index)
        if info.get("invalid"):
            return _old_host, _old_vm, info
        delta = float(self._last_assignment_energy)
        self.lower_energy_emitted += delta
        energy = -delta * self.energy_reward_scale
        info.update(physical_interval_r_energy_total=info["r_energy_total"],
                    physical_interval_r_energy_host=info["r_energy_host"],
                    r_energy_total=energy, r_energy_host=energy,
                    assignment_energy=delta, lower_reward_mode=VARIANT_SCHEME)
        # Single-host record addition: global and selected-host marginal energy coincide.
        host_reward = (1. - self.alpha_delay_host) * energy + self.alpha_delay_host * info["r_delay"]
        vm_reward = (1. - self.alpha_delay_vm) * energy + self.alpha_delay_vm * info["r_delay"]
        return float(host_reward), float(vm_reward), info

    def finish_phase_and_advance(self):
        manager_reward, info = super().finish_phase_and_advance()
        unreturned = self.assignment_energy_total - self.lower_energy_emitted
        info.update(lower_assignment_energy_unemitted=float(unreturned),
                    lower_energy_already_emitted=True,
                    assignment_energy_total=float(self.assignment_energy_total),
                    assignment_energy_physical_gap=float(self.assignment_energy_total - self.total_energy),
                    lower_reward_mode=VARIANT_SCHEME)
        self.lower_energy_emitted += unreturned
        return manager_reward, info
