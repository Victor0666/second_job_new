"""Reuse global VM observations and mapping over the same NoAPF/OE physics."""
from environments.mahdrl_oe_ge_apf_score_fast_tl import CloudWorkflowEnv_VMAgents as GlobalVMEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_as_fast_vaaware_tl.config import ALGORITHM, variant_metadata


class CloudWorkflowEnv_VMAgents(GlobalVMEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, experiment_metadata=None, **kwargs):
        self._experiment_metadata = experiment_metadata
        if experiment_metadata is not None:
            self.algorithm = experiment_metadata["algorithm"]
        super().__init__(*args, **kwargs)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **(
            self._experiment_metadata if self._experiment_metadata is not None else variant_metadata()))
