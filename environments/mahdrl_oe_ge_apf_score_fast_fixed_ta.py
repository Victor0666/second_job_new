"""Fixed sorting, all physically legal resources, original GE/OE/Fast execution."""
import numpy as np
from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as BaseEnv
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.config import get_config
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.policy import ordered_ready_tasks


class CloudWorkflowEnv_VMAgents(BaseEnv):
    def __init__(self, *args, ordering_rule, **kwargs):
        self.fixed_config = get_config(ordering_rule)
        self.algorithm = self.fixed_config.algorithm
        super().__init__(*args, combo_weights=self.fixed_config.weights, **kwargs)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **self.fixed_config.variant_metadata())




    def reset(self, **kwargs):
        self.fixed_sort_calls = 0
        self.last_task_order = []
        return super().reset(**kwargs)

    def _phase_prepare_tasks(self):
        self._add_workflow_if_arrived()
        with self.profiler.stage("fixed_task_sort"):
            self._phase_tasks = ordered_ready_tasks(self, self.fixed_config.rule)
        self.fixed_sort_calls += 1
        self.last_task_order = list(self._phase_tasks)

    def apply_manager_delta(self, delta_vec):
        raise RuntimeError("Fixed task ordering has no TA increment actions")

    def set_manager_combo(self, weights):
        if not np.array_equal(np.asarray(weights), self.fixed_config.weights):
            raise ValueError("Fixed task ordering weights cannot be changed")
        return super().set_manager_combo(weights)
