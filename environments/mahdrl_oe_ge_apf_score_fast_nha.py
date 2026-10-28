"""NHA routing over all physically legal idle hosts and VMs."""
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as NoAPFEnv, MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.config import ALGORITHM, variant_metadata


class CloudWorkflowEnv_VMAgents(NoAPFEnv):
    algorithm = ALGORITHM

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())
