"""Learned TA/HA/VA, R010 rewards and all physically legal idle resources."""
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as BaseEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.config import (
    ALGORITHM, reward_settings, check_reward_environment, variant_metadata,
)


class CloudWorkflowEnv_VMAgents(BaseEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        settings = reward_settings()
        if kwargs.get('manager_alpha_delay', settings['manager_alpha_delay']) != settings['manager_alpha_delay']:
            raise ValueError('MAHDRL-AS-Fast-R010-NoPF requires manager_alpha_delay=0.1')
        kwargs['manager_alpha_delay'] = settings['manager_alpha_delay']
        super().__init__(*args, **kwargs)
        for key, value in settings.items():
            setattr(self, key, value)
        check_reward_environment(self)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())




    def finish_phase_and_advance(self):
        check_reward_environment(self)
        return super().finish_phase_and_advance()

    def vm_assign(self, vm_slot_index):
        check_reward_environment(self)
        return super().vm_assign(vm_slot_index)
