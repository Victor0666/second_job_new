"""Exact reservation physics; cache only topology and unchanged duration inputs."""
import networkx as nx
import numpy as np

from algorithms.drl_gnn.environment import DRLGNNEnv
from common.workflow_opt import exec_time_components
from .config import TASK_FEATURES, VM_FEATURES, POWER_REFERENCE_W
from .networks import PreparedGraphState


class FastDRLGNNEnv(DRLGNNEnv):
    def reset(self, **kwargs):
        self._cached_workflows = 0
        self._global_ids = {}
        self._topology = []
        self._dag_edges = set()
        self._exit_gids = []
        self._static_tasks = []
        self._durations = {}
        self._static_vms = np.zeros((self.num_vms, len(VM_FEATURES)), dtype=np.float64)
        for j, vm in self.vms.items():
            host = self.hosts[vm.host_id]
            self._static_vms[j, [1, 2, 3, 7, 8]] = (
                1. / vm.pc, (host.power(1.) - host.power(0.)) / host.total_pc,
                vm.bw, host.power(0.), vm.host_id)
        return super().reset(**kwargs)

    def _add_workflows_up_to_current_time(self):
        super()._add_workflows_up_to_current_time()
        for wf in self.workflows[self._cached_workflows:]:
            ids = {ti.local_id: g for g, ti in enumerate(self.task_meta) if ti.wf_id == wf.workflow_id}
            self._global_ids.update({(wf.workflow_id, local): gid for local, gid in ids.items()})
            self._topology.append(tuple(ids[local] for local in nx.topological_sort(wf.graph)))
            for local, gid in ids.items():
                if not self.task_parents[gid]:
                    self._dag_edges.add((0, gid + 1))
                for parent in self.task_parents[gid]:
                    self._dag_edges.add((ids[parent] + 1, gid + 1))
                if not self.task_children[gid]:
                    self._exit_gids.append(gid)
                self._static_tasks.append((self.task_mi[gid], self.task_in_bits[gid], self.task_out_bits[gid]))
        self._cached_workflows = len(self.workflows)

    def _estimated_completions(self):
        estimated = np.zeros(len(self.task_meta), dtype=np.float64)
        now = float(self.current_time)
        for topology in self._topology:
            for gid in topology:
                if self.task_state[gid] in ('Queued', 'Running', 'Finished'):
                    estimated[gid] = float(self.task_end_time[gid])
                    continue
                meta = self.task_meta[gid]
                base = gid - meta.local_id
                parent_finish = max((estimated[base + p] for p in self.task_parents[gid]), default=now)
                parents = self._build_parents_loc(gid)
                placement = tuple(parents.items())
                cached = self._durations.get(gid)
                if cached is None or cached[0] != placement:
                    task = self._task_obj_from_gid(gid)
                    durations = tuple(exec_time_components(task, self.vms[j].pc, self.vms[j].bw,
                        parents_loc=parents, child_vm_id=j, child_host_id=self.vms[j].host_id,
                        zero_if_same_vm=True, zero_if_same_host=False)[-1] for j in range(self.num_vms))
                    self._durations[gid] = placement, durations
                else:
                    durations = cached[1]
                estimated[gid] = min(max(now, float(self.vm_available_at[j]), parent_finish) + duration
                                     for j, duration in enumerate(durations))
        return estimated, self._global_ids

    def graph_state(self):
        estimated, _ = self._estimated_completions()
        count = len(self.task_meta)
        now = float(self.current_time)
        scales = self.state_normalizer.scales
        tasks = np.zeros((count + 2, len(TASK_FEATURES)), dtype=np.float64)
        tasks[0, :4] = (1., 0., 0., float(self.first_arrival_time or 0.))
        if count:
            tasks[1:-1, [2, 6, 7]] = self._static_tasks
            tasks[1:-1, 0] = [s in ('Queued', 'Running', 'Finished') for s in self.task_state]
            tasks[1:-1, 1] = [s == 'Ready' for s in self.task_state]
            tasks[1:-1, 3] = estimated
            tasks[1:-1, 4] = np.asarray(self.task_baseline_finish) - now
            tasks[1:-1, 5] = [float(self.workflows[ti.wf_id].deadline) - now for ti in self.task_meta]
        completion = (float(np.max(self.vm_available_at)) if self.done_flag else
                      max((estimated[g] for g in self._exit_gids), default=now))
        tasks[-1, :4] = (float(self.done_flag), 0., 0., completion)
        tasks /= np.asarray((1., 1., scales['workload_mi_ref'], scales['T_ref'], scales['T_ref'],
                             scales['T_ref'], scales['input_bits_ref'], scales['output_bits_ref']))
        edges = self._dag_edges.copy()
        edges.update((g + 1, count + 1) for g in self._exit_gids)
        edges.update((p + 1, g + 1) for p, g in self._vm_order_edges)
        active = dict.fromkeys(self.hosts, 0.)
        for g, status in enumerate(self.task_state):
            if status == 'Running':
                vm = self.vms[self.task_placement[g][0]]
                active[vm.host_id] += float(vm.pc)
        loads = {h: active[h] / max(host.total_pc, 1e-12) for h, host in self.hosts.items()}
        powers = {h: float(host.power(loads[h])) for h, host in self.hosts.items()}
        vms = self._static_vms.copy()
        for j, vm in self.vms.items():
            vms[j, [0, 4, 5, 6]] = (float(self.vm_available_at[j]),
                max(0., float(self.vm_available_at[j]) - now), loads[vm.host_id], powers[vm.host_id])
        vms /= np.asarray((scales['T_ref'], 1. / scales['vm_pc_ref'],
            POWER_REFERENCE_W / scales['host_pc_ref'], scales['vm_bw_ref'], scales['T_ref'],
            1., POWER_REFERENCE_W, POWER_REFERENCE_W, 1.))
        compatibility = np.ones((count + 2, self.num_vms), dtype=bool)
        actions = np.zeros_like(compatibility)
        if not self.done_flag:
            for g in self.ready_task_ids:
                if self.task_state[g] == 'Ready':
                    actions[g + 1] = True
        return PreparedGraphState(tasks, vms, np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2),
                                  compatibility, actions, (-1, *range(count), -2))

    def pair_step(self, action):
        result = super().pair_step(action)
        self._durations.pop(int(action) // self.num_vms - 1, None)
        return result
