"""Three learned levels with OE/GE and every physically legal idle resource."""
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as RawEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast.config import ALGORITHM, variant_metadata


class CloudWorkflowEnv_VMAgents(RawEnv):
    algorithm = ALGORITHM

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())
