"""Reuse HScore's environment, rollout, logging and validation unchanged."""
import torch
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore.config import TA_HIDDEN, learning_kwargs
from environments.mahdrl_as_fast_hscore import MANAGER_ACTION_TABLE
from .agent import TAAgent, HAAgent, FixedVAAgent


def build_agents(scene):
    device = torch.device(scene.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    options = dict(diagnostics_every=scene.diagnostics_every,
                   profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    agents = {"TA": TAAgent(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN, **options, **learning_kwargs("TA", device)),
              "HA": HAAgent(23, scene.mechanism.host_features, **options, **learning_kwargs("HA", device)),
              "VA": FixedVAAgent(**options, **learning_kwargs("VA", device))}
    return {role: agent.bind(scene, role) for role, agent in agents.items()}


def train(scene, **kwargs):
    kwargs.setdefault("agent_builder", build_agents)
    return base.train(scene, **kwargs)


def evaluate(scene, **kwargs):
    kwargs.setdefault("agent_builder", build_agents)
    return base.evaluate(scene, **kwargs)
