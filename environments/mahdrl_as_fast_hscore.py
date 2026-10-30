"""Fixed VA observations over the unchanged VAware execution and rewards."""
import numpy as np
from environments.mahdrl_as_fast_vaaware import CloudWorkflowEnv_VMAgents as BaseEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from environments.mahdrl_heft import spaces
from algorithms.mahdrl_as_fast_hscore.config import ALGORITHM, VM_CAPACITY, VM_INPUT_DIM, variant_metadata
from algorithms.mahdrl_as_fast_hscore.observation import fixed_state


class CloudWorkflowEnv_VMAgents(BaseEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        counts = kwargs.get("vms_per_host", ())
        if counts and max(counts) > VM_CAPACITY:
            raise ValueError("Fixed VA capacity is 9 VMs per host")
        super().__init__(*args, **kwargs)
        if self.max_vms_per_host > VM_CAPACITY:
            raise ValueError("Fixed VA capacity is 9 VMs per host")
        self.observation_space = spaces.Dict(dict(self.observation_space.spaces,
            vm_obs=spaces.Box(-np.inf, np.inf, (VM_INPUT_DIM,), np.float32),
            vm_mask=spaces.Box(0., 1., (VM_CAPACITY,), np.float32)))

    def variant_metadata(self):
        metadata = dict(super().variant_metadata(), **variant_metadata(self.mechanism))
        metadata.update(self._experiment_metadata or {})
        return metadata

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        state.update(vm_obs=np.zeros(VM_INPUT_DIM, np.float32), vm_mask=np.zeros(VM_CAPACITY, np.float32))
        return state, info

    def preview_vm_states(self):
        return {hid: fixed_state(state) for hid, state in super().preview_vm_states().items()}

    def get_vm_state_for_current_task(self):
        state, ok = super().get_vm_state_for_current_task()
        return fixed_state(state), ok
