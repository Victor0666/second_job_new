"""Author task6/VM4 observations and VM-tail reservations over public physics."""
import math
from operator import index

import networkx as nx
import numpy as np

from common.workflow_opt import exec_time_components
from environments.irws import IRWSEnvRouteA
from .networks import GraphState


class TwoStageGATEnv(IRWSEnvRouteA):
    """Ready requires completed parents; busy VMs remain legal reservation targets."""

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
            if status == 'Queued' and self.task_start_time[gid] <= t + 1e-12:
                self.task_state[gid] = 'Running'
        super()._process_completions_at_time(t)

    def _advance_until_decision(self):
        self._process_completions_at_time(self.current_time)
        while True:
            if self._check_done():
                self._finalize_energy_if_done()
                self.done_flag = True
                return
            if any(self.task_state[g] == 'Ready' for g in self.ready_task_ids):
                return  # Reservation does not require an idle VM.
            events = [self.task_end_time[g] for g, s in enumerate(self.task_state) if s == 'Running']
            events.extend(self.task_start_time[g] for g, s in enumerate(self.task_state) if s == 'Queued')
            if self.next_arrival_idx < len(self.arrival_times):
                events.append(float(self.arrival_times[self.next_arrival_idx]))
            future = [float(t) for t in events if t > self.current_time + 1e-12]
            if not future or not math.isfinite(min(future)):
                raise RuntimeError('Unfinished reservation episode has no future physical event')
            self.current_time = min(future)
            self._add_workflows_up_to_current_time()
            self._process_completions_at_time(self.current_time)

    def _durations_and_completions(self):
        count = len(self.task_meta)
        durations = np.empty((count, self.num_vms), dtype=np.float64)
        global_ids = {(m.wf_id, m.local_id): g for g, m in enumerate(self.task_meta)}
        estimated = np.zeros(count, dtype=np.float64)
        for gid in range(count):
            task = self._task_obj_from_gid(gid)
            parents = self._build_parents_loc(gid)
            for vm_id, vm in self.vms.items():
                durations[gid, vm_id] = exec_time_components(
                    task, vm.pc, vm.bw, parents_loc=parents, child_vm_id=vm_id,
                    child_host_id=vm.host_id, zero_if_same_vm=True, zero_if_same_host=False)[-1]
        for wf in self.workflows:
            for local in nx.topological_sort(wf.graph):
                gid = global_ids[wf.workflow_id, local]
                if self.task_state[gid] in ('Queued', 'Running', 'Finished'):
                    estimated[gid] = self.task_end_time[gid]
                else:
                    parent = max((estimated[global_ids[wf.workflow_id, p]]
                                  for p in self.task_parents[gid]), default=self.current_time)
                    estimated[gid] = np.min(np.maximum(self.vm_available_at,
                        max(self.current_time, parent)) + durations[gid])
        return durations, estimated, global_ids

    def graph_state(self):
        durations, estimated, global_ids = self._durations_and_completions()
        count = len(self.task_meta)
        t_ref = self.state_normalizer.scales['T_ref'] if self.state_normalizer else 1.
        task_features = np.zeros((count, 6), dtype=np.float32)
        vm_features = np.ones((count, self.num_vms, 4), dtype=np.float32)
        proxies = np.empty((count, self.num_vms), dtype=np.float64)
        for vm_id, vm in self.vms.items():
            host = self.hosts[vm.host_id]
            rate = (host.power(1.) - host.power(0.)) / host.total_pc
            proxies[:, vm_id] = np.asarray(self.task_mi) * rate
            vm_features[:, vm_id, 0] = self.vm_available_at[vm_id] / t_ref
        vm_features[:, :, 2] = durations / t_ref
        vm_features[:, :, 3] = proxies * self.energy_reward_scale
        task_mask = np.asarray([s == 'Ready' for s in self.task_state], dtype=bool)
        for gid, status in enumerate(self.task_state):
            assigned = status in ('Queued', 'Running', 'Finished')
            energy = (proxies[gid, self.task_placement[gid][0]] if assigned else proxies[gid].min())
            slack = (self.task_baseline_finish[gid] - estimated[gid]) / t_ref if self.deadline_mode != 'none' else 0.
            task_features[gid] = (float(status == 'Ready'), float(assigned), estimated[gid] / t_ref,
                                  energy * self.energy_reward_scale, slack, 1.)
        dependencies = set(self._vm_order_edges)
        for gid, m in enumerate(self.task_meta):
            dependencies.update((global_ids[m.wf_id, p], gid) for p in self.task_parents[gid])
        if not np.isfinite(task_features).all() or not np.isfinite(vm_features).all():
            raise ValueError('Nonfinite two-stage GAT observation')
        return GraphState(task_features, vm_features,
            np.asarray(sorted(dependencies), dtype=np.int64).reshape(-1, 2),
            task_mask, np.ones((count, self.num_vms), dtype=bool), tuple(range(count)))

    def pair_step(self, action):
        """One joint action reserves once; public load integration settles it once."""
        if self.done_flag:
            raise RuntimeError('Episode already completed')
        try:
            action = index(action)
        except TypeError as error:
            raise ValueError('Joint action must be an integer index') from error
        if not 0 <= action < len(self.task_meta) * self.num_vms:
            raise ValueError('Joint action index out of range')
        gid, vm_id = divmod(action, self.num_vms)
        if self.task_state[gid] != 'Ready' or gid not in self.ready_task_ids:
            raise ValueError('Joint action must select an arrived task with completed parents')
        base = gid - self.task_meta[gid].local_id
        if any(self.task_state[base + p] != 'Finished' for p in self.task_parents[gid]):
            raise ValueError('Task predecessors are not physically finished')
        dispatched = float(self.current_time)
        before = float(self.total_energy)
        previous = self._previous_task_by_vm[vm_id]
        self.ready_task_ids.remove(gid)
        _, info = self._assign_task_to_vm(gid, vm_id)
        self.task_dispatch_time[gid] = dispatched
        queued = self.task_start_time[gid] > dispatched + 1e-12
        if queued:
            self.task_state[gid] = 'Queued'
            self.queued_assignments += 1
        if previous is not None:
            self._vm_order_edges.append((previous, gid))
        self._previous_task_by_vm[vm_id] = gid
        self.assignment_count += 1
        self._advance_until_decision()
        increment = float(self.total_energy - before)
        reward = -(1. - self.alpha_delay_vm) * self.energy_reward_scale * increment + self.alpha_delay_vm * info['r_delay']
        info.update(task_gid=gid, vm_id=vm_id, action_count=self.assignment_count,
            dispatch_time=dispatched, queued=queued, delta_energy_total=increment,
            r_energy_total=-self.energy_reward_scale * increment, reward_mode='host_task_window_v1')
        return float(reward), bool(self.done_flag), info
