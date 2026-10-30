"""Unchanged Q4.2 dynamics with fixed variant-local reward parameters."""
from environments.marl import CloudWorkflowEnv_MARL as BaseEnvironment
from .identity import reward_settings


class CloudWorkflowEnv_MARL(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        for key, value in reward_settings().items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError('MARL_Q4.2_090 reward setting mismatch: ' + key)
            kwargs[key] = value
        super().__init__(*args, **kwargs)

    def step(self, a_host_index, a_vm_slot_index):
        for key, value in reward_settings().items():
            if getattr(self, key) != value:
                raise ValueError('MARL_Q4.2_090 reward setting mismatch: ' + key)
        return super().step(a_host_index, a_vm_slot_index)
