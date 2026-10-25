"""DRL-GNN observations with public deadline, data and resource state."""
import math
from operator import index

import networkx as nx
import numpy as np

from common.workflow_opt import exec_time_components
from common.f18_state_normalization import CURRENT_VERSION
from environments.irws import IRWSEnvRouteA
from .config import TASK_FEATURES, VM_FEATURES, POWER_REFERENCE_W
from .networks import GraphState


class DRLGNNEnv(IRWSEnvRouteA):
    """Author readiness: scheduled parents and reservations at any VM's tail."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('normalization_mode') != CURRENT_VERSION:
            raise ValueError('DRL-GNN requires fixed f19_shared_v1 training references')
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self._previous_task_by_vm = [None] * self.num_vms
        self._vm_order_edges = []
        self.task_dispatch_time = []
        self.assignment_count = 0
        self.queued_assignments = 0
        return super().reset(**kwargs)

    def _add_workflows_up_to_current_time(self):
        super()._add_workflows_up_to_current_time()
        self.task_dispatch_time.extend([None] * (len(self.task_meta) - len(self.task_dispatch_time)))

    def _process_completions_at_time(self, t):
        for gid, status in enumerate(self.task_state):
            if status == "Queued" and self.task_start_time[gid] <= t + 1e-12:
                self.task_state[gid] = "Running"
        super()._process_completions_at_time(t)

    def _advance_until_decision(self):
        self._process_completions_at_time(self.current_time)
        while True:
            if self._check_done():
                self._finalize_energy_if_done()
                self.done_flag = True
                return
            if self.ready_task_ids:
                return  # A ready task can reserve a busy VM; do not wait for idle.
            events = [self.task_end_time[g] for g, s in enumerate(self.task_state) if s == "Running"]
            events.extend(self.task_start_time[g] for g, s in enumerate(self.task_state) if s == "Queued")
            if self.next_arrival_idx < len(self.arrival_times):
                events.append(float(self.arrival_times[self.next_arrival_idx]))
            future = [float(t) for t in events if t > self.current_time + 1e-12]
            if not future or not math.isfinite(min(future)):
                raise RuntimeError("Unfinished DRL-GNN reservation episode has no future event")
            self.current_time = min(future)
            self._add_workflows_up_to_current_time()
            self._process_completions_at_time(self.current_time)

    def _estimated_completions(self):
        """Estimate unassigned tasks on all VMs, including their current waits."""
        estimated = np.zeros(len(self.task_meta), dtype=np.float64)
        global_ids = {(ti.wf_id, ti.local_id): gid for gid, ti in enumerate(self.task_meta)}
        now = float(self.current_time)
        for wf in self.workflows:
            for local_id in nx.topological_sort(wf.graph):
                gid = global_ids[wf.workflow_id, local_id]
                if self.task_state[gid] in ("Queued", "Running", "Finished"):
                    estimated[gid] = float(self.task_end_time[gid])
                    continue
                parent_finish = max((estimated[global_ids[wf.workflow_id, p]]
                                     for p in self.task_parents[gid]), default=now)
                parents_loc = self._build_parents_loc(gid)
                task = wf.tasks[local_id]
                finishes = []
                for vm_id in range(self.num_vms):
                    vm = self.vms[vm_id]
                    duration = exec_time_components(
                        task, vm.pc, vm.bw, parents_loc=parents_loc,
                        child_vm_id=vm_id, child_host_id=vm.host_id,
                        zero_if_same_vm=True, zero_if_same_host=False,
                    )[-1]
                    finishes.append(max(now, float(self.vm_available_at[vm_id]),
                                        parent_finish) + duration)
                estimated[gid] = min(finishes)
        return estimated, global_ids

    def graph_state(self):
        """Snapshot only arrived workflows; the two dummy nodes never execute."""
        estimated, global_ids = self._estimated_completions()
        count = len(self.task_meta)
        end = count + 1
        now = float(self.current_time)
        workflows = {wf.workflow_id: wf for wf in self.workflows}
        task_features = np.zeros((count + 2, len(TASK_FEATURES)), dtype=np.float64)
        task_features[0, :4] = (1., 0., 0., float(self.first_arrival_time or 0.))
        for gid, status in enumerate(self.task_state):
            wf = workflows[self.task_meta[gid].wf_id]
            task_features[gid + 1] = (
                float(status in ("Queued", "Running", "Finished")), float(status == "Ready"),
                float(self.task_mi[gid]), estimated[gid],
                float(self.task_baseline_finish[gid]) - now, float(wf.deadline) - now,
                float(self.task_in_bits[gid]), float(self.task_out_bits[gid]),
            )
        dependencies = set()
        exit_estimates = []
        for wf in self.workflows:
            for local_id in wf.graph.nodes:
                gid = global_ids[wf.workflow_id, local_id]
                if not self.task_parents[gid]:
                    dependencies.add((0, gid + 1))
                for parent in self.task_parents[gid]:
                    dependencies.add((global_ids[wf.workflow_id, parent] + 1, gid + 1))
                if not self.task_children[gid]:
                    dependencies.add((gid + 1, end))
                    exit_estimates.append(estimated[gid])
        dependencies.update((previous + 1, gid + 1) for previous, gid in self._vm_order_edges)
        end_completion = (float(np.max(self.vm_available_at)) if self.done_flag
                          else max(exit_estimates, default=float(self.current_time)))
        task_features[end, :4] = (float(self.done_flag), 0., 0., end_completion)

        active_pc = dict.fromkeys(self.hosts, 0.)
        for gid, status in enumerate(self.task_state):
            if status == "Running":
                vm = self.vms[self.task_placement[gid][0]]
                active_pc[vm.host_id] += float(vm.pc)
        host_loads = {h: active_pc[h] / max(host.total_pc, 1e-12)
                      for h, host in self.hosts.items()}
        vm_features = np.empty((self.num_vms, len(VM_FEATURES)), dtype=np.float64)
        for vm_id in range(self.num_vms):
            vm = self.vms[vm_id]
            host = self.hosts[vm.host_id]
            energy_rate = (float(host.power(1.)) - float(host.power(0.))) / host.total_pc
            load = host_loads[vm.host_id]
            vm_features[vm_id] = (
                float(self.vm_available_at[vm_id]), 1. / vm.pc, energy_rate,
                float(vm.bw), max(0., float(self.vm_available_at[vm_id]) - now), load,
                float(host.power(load)), float(host.power(0.)), float(vm.host_id),
            )
        # Scale before float32 conversion/Linear so bits cannot erase flags or deadlines.
        scales = self.state_normalizer.scales
        task_features /= np.asarray((1., 1., scales['workload_mi_ref'], scales['T_ref'],
            scales['T_ref'], scales['T_ref'], scales['input_bits_ref'], scales['output_bits_ref']))
        vm_features /= np.asarray((scales['T_ref'], 1. / scales['vm_pc_ref'],
            POWER_REFERENCE_W / scales['host_pc_ref'], scales['vm_bw_ref'], scales['T_ref'],
            1., POWER_REFERENCE_W, POWER_REFERENCE_W, 1.))
        task_features = task_features.astype(np.float32)
        vm_features = vm_features.astype(np.float32)
        compatibility = np.ones((count + 2, self.num_vms), dtype=bool)
        action_mask = np.zeros_like(compatibility)
        if not self.done_flag:
            for gid in self.ready_task_ids:
                if self.task_state[gid] == "Ready":
                    action_mask[gid + 1] = compatibility[gid + 1]
        if not np.isfinite(task_features).all() or not np.isfinite(vm_features).all():
            raise ValueError("Nonfinite DRL-GNN observation")
        return GraphState(
            task_features=task_features, vm_features=vm_features,
            dependencies=np.asarray(sorted(dependencies), dtype=np.int64).reshape(-1, 2),
            compatibility_mask=compatibility, action_mask=action_mask,
            task_gids=(-1, *range(count), -2),
        )

    def pair_step(self, action):
        """Validate the joint action before any mutation; settle public reward once."""
        if self.done_flag:
            raise RuntimeError("Episode already completed")
        try:
            action = index(action)
        except TypeError as error:
            raise ValueError("Joint action must be an integer index") from error
        if not 0 <= action < (len(self.task_meta) + 2) * self.num_vms:
            raise ValueError("Joint action index out of range")
        task_index, vm_id = divmod(action, self.num_vms)
        gid = task_index - 1
        if not 0 <= gid < len(self.task_state) or self.task_state[gid] != "Ready" or gid not in self.ready_task_ids:
            raise ValueError("Joint action must select a real Ready task")
        base = gid - self.task_meta[gid].local_id
        if any(base + p not in self.task_placement for p in self.task_parents[gid]):
            raise ValueError("Joint action requires all parents to be scheduled")
        dispatched = float(self.current_time)
        before = float(self.total_energy)
        previous = self._previous_task_by_vm[vm_id]
        self.ready_task_ids.remove(gid)
        _, info = self._assign_task_to_vm(gid, vm_id)
        self.task_dispatch_time[gid] = dispatched
        queued = self.task_start_time[gid] > dispatched + 1e-12
        if queued:
            self.task_state[gid] = "Queued"
            self.queued_assignments += 1
        if previous is not None:
            self._vm_order_edges.append((previous, gid))
        self._previous_task_by_vm[vm_id] = gid
        self.assignment_count += 1
        for child in self.task_children[gid]:
            child_gid = base + child
            if (self.task_state[child_gid] == "unReady"
                    and all(base + p in self.task_placement for p in self.task_parents[child_gid])):
                self.task_state[child_gid] = "Ready"
                self.task_ready_time[child_gid] = dispatched
                self.ready_task_ids.append(child_gid)
        self._advance_until_decision()
        increment = float(self.total_energy - before)
        reward = -(1. - self.alpha_delay_vm) * self.energy_reward_scale * increment + self.alpha_delay_vm * info["r_delay"]
        info.update(task_gid=gid, vm_id=vm_id, action_count=self.assignment_count,
            dispatch_time=dispatched, queued=queued, delta_energy_total=increment,
            r_energy_total=-self.energy_reward_scale * increment, reward_mode="host_task_window_v1")
        return float(reward), bool(self.done_flag), info
