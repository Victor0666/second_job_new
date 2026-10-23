"""Original DRL-GNN observations over the shared online cloud simulator."""
from operator import index

import networkx as nx
import numpy as np

from common.workflow_opt import exec_time_components
from environments.irws import IRWSEnvRouteA
from .networks import GraphState


class DRLGNNEnv(IRWSEnvRouteA):
    """Expose one atomic task--VM action without adding a private cost model."""

    def reset(self, **kwargs):
        self._previous_task_by_vm = [None] * self.num_vms
        self._vm_order_edges = []
        self.assignment_count = 0
        return super().reset(**kwargs)

    def _estimated_completions(self):
        """Estimate unassigned tasks on all VMs, including their current waits."""
        estimated = np.zeros(len(self.task_meta), dtype=np.float64)
        global_ids = {(ti.wf_id, ti.local_id): gid for gid, ti in enumerate(self.task_meta)}
        now = float(self.current_time)
        for wf in self.workflows:
            for local_id in nx.topological_sort(wf.graph):
                gid = global_ids[wf.workflow_id, local_id]
                if self.task_state[gid] in ("Running", "Finished"):
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
        task_features = np.zeros((count + 2, 4), dtype=np.float32)
        task_features[0] = (1., 0., 0., float(self.first_arrival_time or 0.))
        for gid, status in enumerate(self.task_state):
            task_features[gid + 1] = (
                float(status in ("Running", "Finished")), float(status == "Ready"),
                float(self.task_mi[gid]), estimated[gid],
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
        task_features[end] = (float(self.done_flag), 0., 0., end_completion)

        vm_features = np.empty((self.num_vms, 3), dtype=np.float32)
        for vm_id in range(self.num_vms):
            vm = self.vms[vm_id]
            host = self.hosts[vm.host_id]
            energy_rate = (float(host.power(1.)) - float(host.power(0.))) / host.total_pc
            vm_features[vm_id] = (float(self.vm_available_at[vm_id]), 1. / vm.pc, energy_rate)
        compatibility = np.ones((count + 2, self.num_vms), dtype=bool)
        action_mask = np.zeros_like(compatibility)
        if not self.done_flag:
            idle = self.vm_available_at <= self.current_time + 1e-12
            for gid in self.ready_task_ids:
                if self.task_state[gid] == "Ready":
                    action_mask[gid + 1] = idle
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
        if self.vm_available_at[vm_id] > self.current_time + 1e-12:
            raise ValueError("Joint action must select an idle VM")
        previous = self._previous_task_by_vm[vm_id]
        self.select_task(gid)
        _, reward, terminated, truncated, info = super().step(vm_id)
        if previous is not None:
            self._vm_order_edges.append((previous, gid))
        self._previous_task_by_vm[vm_id] = gid
        self.assignment_count += 1
        info.update(task_gid=gid, vm_id=vm_id, action_count=self.assignment_count)
        return float(reward), bool(terminated or truncated), info
