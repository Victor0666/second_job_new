"""Shared runtime for the independent variant; existing agents stay unmodified."""
from contextlib import contextmanager
import random

import numpy as np
import torch

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_fe import ALGORITHM
from algorithms.pd3qn_fe.selection import selection_metadata, validation_selection
from common.d3qn_agent_PER import D3QNAgent
from common.experiment_measurements import EvaluationRecorder, collect_validation, episode_metrics
from common.f18_state_normalization import bind_normalization
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_kwargs(scenario):
    return dict(dax_paths=scenario.dax_paths(), horizon=1e9, arrival_lambda=scenario.arrival_lambda,
        random_seed=scenario.first_training_seed(), workflows_per_episode=50,
        max_ready_tasks='auto', normalize=True, num_hosts=scenario.num_hosts,
        vms_per_host=scenario.vms_per_host, pc_tiers=(1., 2., 4., 6., 8.),
        bw_tiers=(1000., 2000., 4000., 6000., 8000.), alpha_delay_vm=.75,
        task_baseline_norm=300., energy_reward_scale=.001, energy_norm_per_mi_ref=20.,
        deadline_mode='cache_fcfs', deadline_cache_path=str(scenario.cache_path),
        deadline_alpha=1.5, deadline_alpha_choices=list(scenario.deadline_choices),
        deadline_alpha_probs=list(scenario.deadline_probs), deadline_cache_strict=True,
        **scenario.normalization_kwargs())


def make_agent(scenario, env):
    settings = agent_kwargs(scenario)
    if settings['paper_mode']:
        raise ValueError('PD3QN-FE cannot use the PD3QN-DAG learner')
    obs, _ = env.reset()
    device = (scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    return bind_normalization(D3QNAgent(input_dim=int(obs.shape[0]),
        output_dim=int(env.action_space.n), device=torch.device(device), **settings), scenario)


@contextmanager
def isolated_evaluation(agent):
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modules = (agent.online, agent.target)
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


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
                    recorder = EvaluationRecorder(scenario, ALGORITHM, env, {'PD3QN-FE': agent},
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
    agent.normalization_metadata['pd3qn_fe'] = dict(algorithm=ALGORITHM,
        checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
        model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()['source_train_scene'],
        episode=episode, completed_episodes=episode + 1, role=role,
        validation_selection=candidate, best_validation_selection=best, validation=rows)
    agent.save(str(path))


def load_best_checkpoint(agent, path, scenario):
    obj = torch.load(path, map_location=agent.device, weights_only=True)
    metadata = obj.get('normalization', {}).get('pd3qn_fe', {})
    expected = dict(algorithm=ALGORITHM, checkpoint_selection=selection_metadata(),
        protocol=scenario.selection_protocol(), model_seed=scenario.model_seed, role='best',
        source_train_scene=scenario.normalization_kwargs()['source_train_scene'])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('PD3QN-FE checkpoint mismatch: ' + key)
    rows = metadata.get('validation', [])
    if [row['seed'] for row in rows] != scenario.validation_seeds():
        raise ValueError('PD3QN-FE checkpoint validation seeds mismatch')
    if validation_selection(rows) != metadata.get('validation_selection'):
        raise ValueError('PD3QN-FE checkpoint validation metrics mismatch')
    agent.load(str(path), require_learning_model=True)
    return metadata
