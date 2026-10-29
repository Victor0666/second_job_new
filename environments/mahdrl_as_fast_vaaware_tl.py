"""Reuse global VM observations and mapping over the same NoAPF/OE physics."""
from environments.mahdrl_oe_ge_apf_score_fast_tl import CloudWorkflowEnv_VMAgents as GlobalVMEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_as_fast_vaaware_tl.config import ALGORITHM, variant_metadata


class CloudWorkflowEnv_VMAgents(GlobalVMEnv):
    algorithm = ALGORITHM

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())
