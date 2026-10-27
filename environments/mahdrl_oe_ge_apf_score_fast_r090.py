"""Unchanged AS-Fast observations/physics, with fixed R090 rewards at all levels."""
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as BaseEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_r090.config import (
    ALGORITHM, reward_settings, check_reward_environment, variant_metadata,
)


class CloudWorkflowEnv_VMAgents(BaseEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        settings = reward_settings()
        if kwargs.get('manager_alpha_delay', settings['manager_alpha_delay']) != settings['manager_alpha_delay']:
            raise ValueError('MAHDRL-AS-Fast-R090 requires manager_alpha_delay=0.9')
        kwargs['manager_alpha_delay'] = settings['manager_alpha_delay']
        super().__init__(*args, **kwargs)
        for key, value in settings.items():
            setattr(self, key, value)
        check_reward_environment(self)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())

    def vm_assign(self, vm_slot_index):
        check_reward_environment(self)
        return super().vm_assign(vm_slot_index)

    def finish_phase_and_advance(self):
        check_reward_environment(self)
        return super().finish_phase_and_advance()
