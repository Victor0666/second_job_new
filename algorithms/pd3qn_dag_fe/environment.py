"""Compact policy observations; scheduling, transmission and energy stay unchanged."""
import numpy as np

from algorithms.pd3qn_dag_fe import OBSERVATION_LAYOUT
from common.f18_state_normalization import CURRENT_VERSION, checked_observation
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


class CloudWorkflowEnv_DAG_FE(CloudWorkflowEnv_PD3QN_Cost):
    def __init__(self, *args, **kwargs):
        if kwargs.get('normalization_mode') != CURRENT_VERSION:
            raise ValueError('PD3QN-DAG-FE requires f19_shared_v1 calibration')
        super().__init__(*args, **kwargs)
        self.state_normalizer.metadata['observation_layout'] = OBSERVATION_LAYOUT

    def _build_obs(self):
        now = float(self.current_time)
        n = self.state_normalizer
        task = np.zeros(3, dtype=np.float32)
        if self._cur_task_gid is not None:
            gid = int(self._cur_task_gid)
            task[:] = (n.scale(self.task_mi[gid], 'workload_mi_ref'),
                       n.scale(self.task_in_bits[gid], 'input_bits_ref'),
                       n.scale(self.task_baseline_finish[gid] - now, 'T_ref'))
        vm = np.stack((n.scale(np.maximum(self.vm_available_at - now, 0.), 'T_ref'),
                       n.scale(self._vm_pc, 'vm_pc_ref'),
                       n.scale(self._vm_bw_bps / 1e6, 'vm_bw_ref')), axis=1)
        obs = np.concatenate((task, vm.reshape(-1))).astype(np.float32)
        return checked_observation(self, obs)
