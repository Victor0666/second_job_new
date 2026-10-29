"""Release children after completion and dispatch only onto idle VMs."""
from dataclasses import replace
from operator import index

from algorithms.drl_gnn_fast.environment import FastDRLGNNEnv
from environments.irws import IRWSEnvRouteA


class IdleDRLGNNEnv(FastDRLGNNEnv):
    # Public physics already advances until a completed-parent task and idle VM exist.
    _advance_until_decision = IRWSEnvRouteA._advance_until_decision

    def graph_state(self):
        state = super().graph_state()
        idle = self.vm_available_at <= self.current_time + 1e-12
        return replace(state, action_mask=state.action_mask & idle[None, :])

    def pair_step(self, action):
        """The policy mask excludes illegal pairs; validate before physical mutation too."""
        if self.done_flag:
            raise RuntimeError('Episode already completed')
        try:
            action = index(action)
        except TypeError as error:
            raise ValueError('Joint action must be an integer index') from error
        if not 0 <= action < (len(self.task_meta) + 2) * self.num_vms:
            raise ValueError('Joint action index out of range')
        task_index, vm_id = divmod(action, self.num_vms)
        gid = task_index - 1
        if not 0 <= gid < len(self.task_state) or self.task_state[gid] != 'Ready' or gid not in self.ready_task_ids:
            raise ValueError('Joint action must select a real Ready task')
        base = gid - self.task_meta[gid].local_id
        if any(self.task_state[base + p] != 'Finished' for p in self.task_parents[gid]):
            raise ValueError('Joint action requires all parents to be finished')
        if self.vm_available_at[vm_id] > self.current_time + 1e-12:
            raise ValueError('Joint action requires an idle VM')

        dispatched = float(self.current_time)
        before = float(self.total_energy)
        previous = self._previous_task_by_vm[vm_id]
        self.ready_task_ids.remove(gid)
        _, info = self._assign_task_to_vm(gid, vm_id)
        self.task_dispatch_time[gid] = dispatched
        if previous is not None:
            self._vm_order_edges.append((previous, gid))
        self._previous_task_by_vm[vm_id] = gid
        self.assignment_count += 1
        # The inherited completion handler, rather than dispatch, releases children.
        self._advance_until_decision()
        self._durations.pop(gid, None)
        increment = float(self.total_energy - before)
        reward = -(1. - self.alpha_delay_vm) * self.energy_reward_scale * increment + self.alpha_delay_vm * info['r_delay']
        info.update(task_gid=gid, vm_id=vm_id, action_count=self.assignment_count,
            dispatch_time=dispatched, queued=False, delta_energy_total=increment,
            r_energy_total=-self.energy_reward_scale * increment, reward_mode='host_task_window_v1')
        return float(reward), bool(self.done_flag), info
