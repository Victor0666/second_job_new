"""Reuse the sync-task rollout, accelerated learners and original audit output."""
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore_ablation.runtime import run_episode
from algorithms.mahdrl_as_fast_hscore_speed.runtime import build_agents
from .environment import CloudWorkflowEnv_VMAgents


def build_env(scene, seed):
    kwargs = base.build_env_kwargs(scene, seed)
    kwargs['manager_alpha_delay'] = scene.variant.alpha
    env = CloudWorkflowEnv_VMAgents(**kwargs, variant=scene.variant,
        policy_seed=scene.model_seed*1000003, mechanism=scene.mechanism,
        experiment_metadata=scene.variant_metadata(), profile_mode=scene.profile_mode,
        profile_every=scene.profile_every)
    env.energy_reward_scale, env.task_baseline_norm, env.energy_norm_per_mi_ref = .001, 300., 20.
    return env


def load_agents(scene, *, agent_builder=build_agents):
    return base.load_agents(scene, agent_builder=agent_builder)


def train(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    kwargs.setdefault('rollout', run_episode)
    return base.train(scene, **kwargs)


def evaluate(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    kwargs.setdefault('rollout', run_episode)
    return base.evaluate(scene, **kwargs)
