"""Unchanged dynamics with settled energy prefixes and direct normalized queues."""
import numpy as np
from common.f18_state_normalization import checked_observation, queue_features
from environments.mahdrl_as_fast_hscore import CloudWorkflowEnv_VMAgents as BaseEnv
from .energy import PrefixEnergy


class CloudWorkflowEnv_VMAgents(BaseEnv):
    def _compute_block2(self):
        if 'block2' in self._fast_cache:
            return self._fast_cache['block2']
        if not getattr(self, 'state_normalizer', None):
            return super()._compute_block2()
        # The legacy queue calculations are overwritten by these normalized fields.
        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if self.ready_task_ids else 0.
        block = np.zeros(10, np.float32)
        block[[0, 1, 2, 3, 4, 7, 8, 6]] = queue_features(self)
        block[5] = self.state_normalizer.scale(sum_mi, 'workload_mi_ref')
        block[9] = float(len({self.task_meta[x][0] for x in self.ready_task_ids})) / max(1., len(self.workflows))
        self._fast_cache['block2'] = checked_observation(self, block)
        return block

    def _energy_reward_to_current_time_with_breakdown(self):
        if not self.workflows:
            return 0., {int(h): 0. for h in self.host_ids}
        ledger = getattr(self, '_physical_prefix', None)
        if ledger is None or ledger.records is not self._records or ledger.origin != self.first_arrival_time:
            ledger = self._physical_prefix = PrefixEnergy(self.hosts, self._records, self.first_arrival_time)
        total, per_host = ledger.at(self.current_time)
        delta = max(0., total-self._energy_cache)
        self._energy_cache = float(total)
        rewards = {}
        for hid in self.host_ids:
            value = per_host[hid]
            previous = float(self._energy_cache_by_host.get(int(hid), 0.))
            change = max(0., value-previous)
            self._energy_cache_by_host[int(hid)] = float(value)
            rewards[int(hid)] = -(float(change)*float(self.energy_reward_scale))
        self.total_energy += float(delta)
        self.lifetime_energy += float(delta)
        return -(float(delta)*float(self.energy_reward_scale)), rewards
