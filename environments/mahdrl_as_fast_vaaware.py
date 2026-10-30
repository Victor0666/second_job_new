"""Pure per-host VA previews over V3 physical dynamics and OE rewards."""
import numpy as np
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as RawEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_as_fast_vaaware.config import ALGORITHM, Mechanism, variant_metadata


class CloudWorkflowEnv_VMAgents(RawEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, mechanism=Mechanism(), experiment_metadata=None, **kwargs):
        self.mechanism = mechanism
        self._experiment_metadata = experiment_metadata
        if experiment_metadata is not None:
            self.algorithm = experiment_metadata["algorithm"]
        super().__init__(*args, **kwargs)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **(
            self._experiment_metadata if self._experiment_metadata is not None else variant_metadata(self.mechanism)))

    def preview_vm_states(self):
        if self._cur_tid is None or self._cur_host_id is not None:
            raise RuntimeError("Preview requires a current task before host selection")
        tid, states = self._cur_tid, {}
        snapshot = self._apf_snapshot(tid)
        for hid in self.host_ids:
            if hid not in snapshot["hosts"]:
                continue
            x, mask = self._build_vm_obs_for_task_host(tid, hid)
            b, indices = self.base_vm_obs_dim, self.host_to_vm_indices[hid]
            k = len(indices)
            context = np.concatenate((x[:27], x[b:b+1], x[b+1+self.num_hosts:b+3+self.num_hosts]))
            candidates = np.concatenate((x[27:b].reshape(self.max_vms_per_host, 5)[:k],
                x[b+3+self.num_hosts:].reshape(self.max_vms_per_host, 2)[:k]), axis=1)
            rows = np.array([snapshot["rows"].get(j, (0., 0., 0.)) for j in indices], dtype=np.float64)
            states[hid] = dict(obs=dict(context=context, candidates=candidates,
                exists_mask=np.ones(k, np.float32)), mask=mask[:k], rows=rows,
                performance_times=rows[:, 0], candidate_ids=[self.vm_ids[j] for j in indices])
        return states
