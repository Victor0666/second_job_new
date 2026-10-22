"""OE ledger + GE policy + original PF masks, with summaries of eligible VMs."""
import numpy as np

from common.mahdrl_combinations import OE_GE_PF_POLICY
from common.mahdrl_performance import candidate_durations, performance_mask
from environments.mahdrl_oe_ge import CloudWorkflowEnv_VMAgents as _OEGE
from environments.mahdrl_pf import CloudWorkflowEnv_VMAgents as _PF
from environments.mahdrl_oe_ge import MANAGER_ACTION_TABLE


class CloudWorkflowEnv_VMAgents(_PF, _OEGE):
    algorithm = "mahdrl_oe_ge_pf"
    policy = OE_GE_PF_POLICY
    components = ("OE", "GE", "PF")

    def _candidate_rows(self, tid):
        # Do not change OE's raw preview cache or assignment-energy ledger.
        rows = super()._candidate_rows(tid)
        retained = {}
        for host_id, indices in self.host_to_vm_indices.items():
            mask = np.zeros(self.vm_act_dim, dtype=np.float32)
            for slot, j in enumerate(indices):
                mask[slot] = float(j in rows)
            durations = candidate_durations(self, "vm", mask, task_id=tid, host_id=host_id)
            allowed = performance_mask(mask, durations, self.policy.max_duration_ratio)
            retained.update({indices[slot]: rows[indices[slot]]
                             for slot in np.flatnonzero(allowed > .5)})
        return retained

    def _build_host_obs_for_task(self, tid):
        # PF calls OE's builder first; its summaries already use retained VM rows.
        obs, mask = super()._build_host_obs_for_task(tid)
        summaries = obs[self.base_host_obs_dim + 1:].reshape(self.num_hosts, 5)
        summaries[mask <= .5] = 0.
        return obs, mask

    def _build_vm_obs_for_task_host(self, tid, host_id):
        obs, mask = super()._build_vm_obs_for_task_host(tid, host_id)
        start = self.base_vm_obs_dim + 1 + self.num_hosts + 2
        obs[start:].reshape(self.max_vms_per_host, 2)[mask <= .5] = 0.
        return obs, mask
