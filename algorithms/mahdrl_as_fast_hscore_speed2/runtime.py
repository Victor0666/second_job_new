"""Reuse Speed v1 learners and the original training/evaluation control flow."""
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore_speed.runtime import build_agents
from .environment import CloudWorkflowEnv_VMAgents


def build_env(scene, seed):
    env = CloudWorkflowEnv_VMAgents(**base.build_env_kwargs(scene, seed), mechanism=scene.mechanism,
        experiment_metadata=scene.variant_metadata(),
        profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    env.energy_reward_scale = .001
    env.task_baseline_norm = 300.
    env.energy_norm_per_mi_ref = 20.
    env.alpha_delay_host = env.alpha_delay_vm = .75
    return env


def train(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    return base.train(scene, **kwargs)


def evaluate(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    return base.evaluate(scene, **kwargs)
