"""Shared runtime for the independent variant; existing agents stay unmodified."""

import numpy as np
import torch

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_fe_r090 import ALGORITHM, MODEL_VERSION, DELAY_WEIGHT, reward_metadata
from algorithms.pd3qn_fe.runtime import env_kwargs as original_env_kwargs, isolated_evaluation, set_seed
from algorithms.pd3qn_fe_r090.selection import selection_metadata, validation_selection
from common.d3qn_agent_PER import D3QNAgent
from common.experiment_measurements import EvaluationRecorder, collect_validation, episode_metrics
from common.f18_state_normalization import bind_normalization
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


def env_kwargs(scenario):
    kwargs = original_env_kwargs(scenario)
    kwargs['alpha_delay_vm'] = DELAY_WEIGHT
    return kwargs


def make_agent(scenario, env):
    settings = agent_kwargs(scenario)
    if settings['paper_mode']:
        raise ValueError('PD3QN-FE-R090 cannot use the PD3QN-DAG learner')
    obs, _ = env.reset()
    device = (scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    return bind_normalization(D3QNAgent(input_dim=int(obs.shape[0]),
        output_dim=int(env.action_space.n), device=torch.device(device), **settings), scenario)


def evaluate_instances(agent, scenario, seeds, split='validation', kwargs=None, directory=None):
    allowed = scenario.protocol().validation if split == 'validation' else scenario.protocol().test
    seeds = list(seeds)
    if split not in ('validation', 'test') or not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(allowed):
        raise ValueError('Evaluation seeds must belong to the frozen validation/test partition')
    rows = []
    with isolated_evaluation(agent):
        for seed in seeds:
            env = CloudWorkflowEnv_PD3QN_Cost(**dict(kwargs or env_kwargs(scenario), random_seed=int(seed)))
            recorder = None
            try:
                obs, _ = env.reset()
                if directory is not None:
                    recorder = EvaluationRecorder(scenario, ALGORITHM, env, {'PD3QN-FE-R090': agent},
                        directory=directory, checkpoint_directory=str(scenario.checkpoint_dir()))
                done = False
                while not done:
                    action = agent.select_action(obs, env.get_action_mask(), deterministic=True, count_step=False)
                    obs, _, terminated, truncated, _ = env.step(int(action))
                    done = bool(terminated or truncated)
                metrics = episode_metrics(env)
                if metrics['wf_count'] != env.workflows_per_episode:
                    raise RuntimeError('Evaluation ended before all workflows arrived')
                if not np.isfinite(list(metrics.values())).all():
                    raise FloatingPointError('Nonfinite evaluation metrics')
                rows.append(dict(seed=int(seed), **metrics))
                if split == 'validation':
                    collect_validation(env)
            finally:
                if recorder is not None:
                    recorder.finish()
                if hasattr(env, 'close'):
                    env.close()
    return rows


def save_checkpoint(agent, path, scenario, episode, candidate, best, rows, role):
    # D3QNAgent.save already persists this dictionary; no shared serializer changes.
    agent.normalization_metadata['pd3qn_fe_r090'] = dict(algorithm=ALGORITHM,
        model_version=MODEL_VERSION, reward=reward_metadata(), checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
        model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()['source_train_scene'],
        episode=episode, completed_episodes=episode + 1, role=role,
        validation_selection=candidate, best_validation_selection=best, validation=rows)
    agent.save(str(path))


def load_best_checkpoint(agent, path, scenario):
    obj = torch.load(path, map_location=agent.device, weights_only=True)
    metadata = obj.get('normalization', {}).get('pd3qn_fe_r090', {})
    expected = dict(algorithm=ALGORITHM, model_version=MODEL_VERSION,
        reward=reward_metadata(), checkpoint_selection=selection_metadata(),
        protocol=scenario.selection_protocol(), model_seed=scenario.model_seed, role='best',
        source_train_scene=scenario.normalization_kwargs()['source_train_scene'])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('PD3QN-FE-R090 checkpoint mismatch: ' + key)
    if obj.get('normalization', {}).get('reward') != reward_metadata():
        raise ValueError('PD3QN-FE-R090 checkpoint mismatch: reward')
    rows = metadata.get('validation', [])
    if [row['seed'] for row in rows] != scenario.validation_seeds():
        raise ValueError('PD3QN-FE-R090 checkpoint validation seeds mismatch')
    if validation_selection(rows) != metadata.get('validation_selection'):
        raise ValueError('PD3QN-FE-R090 checkpoint validation metrics mismatch')
    agent.load(str(path), require_learning_model=True)
    return metadata
