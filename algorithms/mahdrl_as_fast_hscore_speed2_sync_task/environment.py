"""Existing task-level TA decisions with Speed2's energy and queue methods."""
from environments.mahdrl_as_fast_hscore_ablation import CloudWorkflowEnv_VMAgents as TaskEnv
from algorithms.mahdrl_as_fast_hscore_speed2.environment import CloudWorkflowEnv_VMAgents as Speed2Env


class CloudWorkflowEnv_VMAgents(TaskEnv, Speed2Env):
    pass
