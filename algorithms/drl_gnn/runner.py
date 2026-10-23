"""Frozen-protocol sampling, checkpoints and common physical evaluation for GIN-PPO."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from common.experiment_measurements import EvaluationRecorder, episode_metrics, VERSION as MEASUREMENT_VERSION
from common.f11 import protect_training_outputs
from common.f22_instances import instance_identity, write_run_metadata
from common.read_xml_opt_Tsize import load_workflow_from_dax
from common.scenarios import ROOT, parse_scenario
from common.workflow_opt import ENERGY_ACCOUNTING, COMMUNICATION_MODEL
from .config import (COMMON_REWARD, DRLGNNConfig, INPUT_ENCODING, MODEL_VERSION,
                     REWARD_PROTOCOL, SOURCE_COMMIT, SOURCE_REPOSITORY)
from .environment import DRLGNNEnv
from .ppo import PPOAgent, Transition


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sync(agent):
    if agent.device.type == 'cuda':
        torch.cuda.synchronize(agent.device)


def env_kwargs(scenario, seed):
    return dict(dax_paths=scenario.dax_paths(), horizon=1e9, arrival_lambda=.03,
        random_seed=int(seed), workflows_per_episode=50, num_hosts=scenario.num_hosts,
        vms_per_host=scenario.vms_per_host, pc_tiers=(1., 2., 4., 6., 8.),
        bw_tiers=(1000., 2000., 4000., 6000., 8000.), max_ready_tasks='auto', normalize=False,
        deadline_mode='cache_fcfs', deadline_cache_path=str(scenario.cache_path),
        deadline_cache_strict=True, deadline_alpha_choices=(2., 3.),
        deadline_alpha_probs=scenario.deadline_probs, normalization_mode='legacy',
        energy_check_on_done=True, **COMMON_REWARD)


def output_dir(scenario, kind):
    if kind == 'ckpts':
        return scenario.checkpoint_dir('drl_gnn')
    if kind == 'logs':
        return scenario.log_dir('drl_gnn')
    return ROOT / 'outputs' / kind / MODEL_VERSION / scenario.model_name('drl_gnn') / (
        scenario.task + scenario.resource + '_' + scenario.deadline)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def objective_metadata():
    return dict(source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT,
        model_version=MODEL_VERSION, policy_input=INPUT_ENCODING,
        reward_protocol=REWARD_PROTOCOL, reward_settings=COMMON_REWARD,
        energy_accounting=ENERGY_ACCOUNTING, communication_model=COMMUNICATION_MODEL,
        evaluation='argmax with author single-graph train-mode BN on an isolated copy')


def save_checkpoint(path, agent, scenario, completed, best_energy, environment_config, budget):
    obj = dict(**objective_metadata(), config=agent.cfg.as_dict(),
        network=agent.network.state_dict(), optimizer=agent.optimizer.state_dict(),
        completed_episodes=int(completed), steps=agent.steps, updates=agent.updates,
        best_validation_energy=float(best_energy), protocol=scenario.protocol().metadata(),
        shared_reference_normalization=scenario.normalization_metadata(),
        scenario=scenario.describe('drl_gnn'), environment=environment_config, budget=budget,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        resume_supported=False)
    # Best snapshots can be inside a rollout; they are evaluation checkpoints, not resumable runs.
    torch.save(obj, path)


def load_checkpoint(path, scenario, device='cpu'):
    obj = torch.load(path, map_location=device, weights_only=False)
    for key, value in objective_metadata().items():
        if obj.get(key) != value:
            raise ValueError('DRL-GNN checkpoint mismatch: ' + key)
    if obj.get('protocol') != scenario.protocol().metadata():
        raise ValueError('DRL-GNN checkpoint data partition mismatch')
    source = obj['scenario']
    expected = (scenario.train_task if scenario.mode == 'eval' else scenario.task,
                (scenario.train_resource or scenario.resource) if scenario.mode == 'eval' else scenario.resource,
                (scenario.train_deadline or scenario.deadline) if scenario.mode == 'eval' else scenario.deadline)
    if (source['task'], source['resource'], source['deadline']) != expected:
        raise ValueError('DRL-GNN checkpoint source training scenario mismatch')
    if source['model_seed'] != scenario.model_seed:
        raise ValueError('DRL-GNN checkpoint model seed mismatch')
    reference = scenario.normalization_metadata()
    for key in ('normalization_config_hash', 'source_train_scene', 'experiment_protocol_hash'):
        if obj.get('shared_reference_normalization', {}).get(key) != reference.get(key):
            raise ValueError('DRL-GNN public experiment reference mismatch: ' + key)
    config = dict(obj['config']); config['device'] = str(device)
    agent = PPOAgent(DRLGNNConfig(**config), device=device)
    agent.network.load_state_dict(obj['network'])
    agent.optimizer.load_state_dict(obj['optimizer'])
    agent.steps = int(obj['steps']); agent.updates = int(obj['updates'])
    return agent, obj


def run_episode(env, agent, seed, evaluation=None):
    env.reset(seed=int(seed))
    recorder = EvaluationRecorder(evaluation[0], 'drl_gnn', env, {'GIN-PPO': agent},
        evaluation[1], checkpoint_directory=str(evaluation[2])) if evaluation else None
    total_reward = 0.
    try:
        while not env.done_flag:
            state = env.graph_state()
            action, _, _ = agent.act(state, deterministic=True)
            reward, _, _ = env.pair_step(action)
            total_reward += reward
        metrics = episode_metrics(env)
        if env.completed_workflows != env.workflows_per_episode:
            raise RuntimeError('GIN episode stopped before physical workflow completion')
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError('Nonfinite common physical metrics')
        if recorder:
            recorder.finish(); recorder = None
        return dict(metrics=metrics, external_return=float(total_reward),
            assignments=env.assignment_count)
    finally:
        if recorder:
            recorder.finish()


def evaluate(agent, scenario, seeds, split='test', kwargs=None, directory=None, checkpoint_directory=None):
    protocol = scenario.protocol()
    if protocol is None:
        raise ValueError('DRL-GNN requires an explicit disjoint protocol')
    allowed = protocol.validation if split == 'validation' else protocol.test
    if split not in ('test', 'validation') or not seeds or not set(seeds) <= set(allowed):
        raise ValueError('Evaluation attempted to cross the frozen partition')
    # Preserve author per-graph BN without letting validation/test mutate the trained policy.
    frozen = deepcopy(agent)
    snapshot = deepcopy(frozen.network.state_dict())
    rows = []
    for seed in seeds:
        frozen.network.load_state_dict(snapshot)
        env = DRLGNNEnv(**dict(kwargs or env_kwargs(scenario, seed), random_seed=int(seed)))
        evaluation = (scenario, directory, checkpoint_directory or output_dir(scenario, 'ckpts')) if directory else None
        result = run_episode(env, frozen, seed, evaluation)
        rows.append(dict(seed=int(seed), **result['metrics']))
    summary = {key: dict(mean=float(np.mean([r[key] for r in rows])),
        std=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.)
        for key in rows[0] if key != 'seed'}
    return rows, summary


def planned_assignments(env, seeds):
    """Count training-template tasks only, without scheduling or exposing future states."""
    counts = {Path(p).name.casefold(): len(load_workflow_from_dax(p, seed=0)[1])
              for p in dict.fromkeys(env.dax_paths)}
    if env.deadline_mode != 'cache_fcfs':
        raise ValueError('Training requires the paired deadline cache to fix its budget')
    total = 0
    for seed in seeds:
        entry = env._deadline_cache[int(seed)]
        templates = entry['wf_dax_names'][:env.workflows_per_episode]
        if len(templates) != env.workflows_per_episode:
            raise ValueError('Deadline cache has too few workflows')
        total += sum(counts[Path(name).name.casefold()] for name in templates)
    return int(total)


def train(scenario, config=None, kwargs=None, out=None):
    protocol = scenario.protocol()
    if protocol is None or not 0 < scenario.episodes <= protocol.episodes:
        raise ValueError('DRL-GNN training requires the fixed common episode budget')
    scenario.normalization_metadata()  # Validate the shared experiment reference, not GIN input scaling.
    cfg = config or DRLGNNConfig(device=(scenario.paper_options or {}).get('device') or (
        'cuda' if torch.cuda.is_available() else 'cpu'), seed=scenario.model_seed)
    if cfg.seed != scenario.model_seed:
        raise ValueError('Model initialization seed must match the common scenario')
    directory = Path(out) if out is not None else output_dir(scenario, 'ckpts')
    logs = directory / 'logs' if out is not None else output_dir(scenario, 'logs')
    protect_training_outputs(directory, logs)
    directory.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kwargs or env_kwargs(scenario, protocol.train[0]))
    if kwargs.get('normalization_mode') != 'legacy':
        raise ValueError('The source GIN uses raw task4/VM3 inputs, not F19 feature scaling')
    if any(kwargs.get(key) != value for key, value in COMMON_REWARD.items()):
        raise ValueError('DRL-GNN must use the public global energy-tardiness coefficients')
    seeds = tuple(protocol.train[:scenario.episodes])
    envs = [DRLGNNEnv(**dict(kwargs, random_seed=seeds[0])) for _ in range(cfg.num_envs)]
    total_assignments = planned_assignments(envs[0], seeds)
    total_updates = math.ceil(total_assignments / cfg.batch_size)
    budget = dict(episodes=scenario.episodes, assignments=total_assignments,
        num_envs=cfg.num_envs, steps_per_env=cfg.num_steps, full_rollout=cfg.batch_size,
        planned_updates=total_updates, author_default_timesteps_unused=cfg.total_timesteps)
    seed_everything(scenario.model_seed)
    agent = PPOAgent(cfg, device=cfg.device)
    for path in (directory, logs):
        write_run_metadata(path, scenario)
    metadata = dict(**objective_metadata(), config=cfg.as_dict(), budget=budget,
        protocol=protocol.metadata(), parameters=sum(p.numel() for p in agent.network.parameters()),
        numerical_precision='float32', timing_scope='global nonoverlapping intervals between completed episodes; 4 environments share one policy')
    (logs / 'measurement_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    states = [None] * cfg.num_envs; slot_seeds = [None] * cfg.num_envs
    slot_returns = [0.] * cfg.num_envs; next_seed = 0; completed = 0; best = float('inf')
    records = []; updates = []; optimizer_steps = 0; sample_presentations = 0
    sync(agent); interval_clock = time.perf_counter()
    cumulative = dict(train=0., validation=0., checkpoint=0.)
    previous_counts = dict(assignments=0, optimizer_steps=0, sample_presentations=0)

    def reset_slot(slot):
        nonlocal next_seed
        if next_seed == len(seeds):
            states[slot] = None; slot_seeds[slot] = None
            return
        seed = seeds[next_seed]; next_seed += 1
        envs[slot].reset(seed=seed)
        states[slot] = envs[slot].graph_state(); slot_seeds[slot] = seed; slot_returns[slot] = 0.

    def update_rollout():
        nonlocal records, optimizer_steps, sample_presentations
        if not records:
            return
        learning_rate = cfg.learning_rate * (1 - agent.updates / total_updates) if cfg.anneal_lr else cfg.learning_rate
        sync(agent); started = time.perf_counter()
        result = agent.update(records, learning_rate=learning_rate)
        sync(agent); result['seconds'] = time.perf_counter() - started
        optimizer_steps += result['epochs'] * min(cfg.num_minibatches, len(records))
        sample_presentations += result['epochs'] * len(records)
        result.update(assignments=agent.steps, optimizer_steps=optimizer_steps,
                      sample_presentations=sample_presentations)
        append_json(logs / 'updates.jsonl', result); updates.append(result); records = []

    for slot in range(cfg.num_envs):
        reset_slot(slot)
    print(f'[DRL-GNN train] {scenario.task}{scenario.resource}/{scenario.deadline} '
          f'episodes={len(seeds)} assignments={total_assignments} device={agent.device}', flush=True)
    while completed < len(seeds):
        for _ in range(cfg.num_steps):
            for slot, env in enumerate(envs):
                state = states[slot]
                if state is None:
                    continue
                action, logp, value = agent.act(state)
                reward, done, _ = env.pair_step(action)
                next_state = None if done else env.graph_state()
                next_value = 0. if done else agent.value(next_state)
                records.append(Transition(state, action, logp, value, reward, done, next_value, slot))
                slot_returns[slot] += reward; states[slot] = next_state
                if len(records) == cfg.batch_size:
                    update_rollout()
                if not done:
                    continue
                completed += 1
                if completed == len(seeds):
                    update_rollout()
                sync(agent); now = time.perf_counter()
                train_seconds = now - interval_clock; cumulative['train'] += train_seconds
                metrics = episode_metrics(env)
                identity = instance_identity(env, scenario, slot_seeds[slot])
                validation_started = time.perf_counter()
                validation_rows, validation = evaluate(agent, scenario, protocol.validation,
                    split='validation', kwargs=kwargs)
                sync(agent); validation_seconds = time.perf_counter() - validation_started
                cumulative['validation'] += validation_seconds
                energy = validation['total_energy']['mean']
                checkpoint_started = time.perf_counter()
                if energy < best:
                    best = energy
                    save_checkpoint(directory / 'best.pth', agent, scenario, completed, best, kwargs, budget)
                if completed == len(seeds):
                    save_checkpoint(directory / 'final.pth', agent, scenario, completed, best, kwargs, budget)
                sync(agent); checkpoint_seconds = time.perf_counter() - checkpoint_started
                cumulative['checkpoint'] += checkpoint_seconds
                counts = dict(assignments=agent.steps, optimizer_steps=optimizer_steps,
                              sample_presentations=sample_presentations)
                row = dict(version=MEASUREMENT_VERSION, algorithm='drl_gnn', episode=completed - 1,
                    instance_seed=slot_seeds[slot], model_seed=scenario.model_seed, metrics=metrics,
                    validation=validation_rows, validation_summary=validation, instance=identity,
                    external_return=slot_returns[slot], episode_assignments=env.assignment_count,
                    assignment_calls=counts['assignments'] - previous_counts['assignments'],
                    optimizer_steps={'GIN-PPO.optimizer': counts['optimizer_steps'] - previous_counts['optimizer_steps']},
                    optimizer_sample_presentations={'GIN-PPO.optimizer': counts['sample_presentations'] - previous_counts['sample_presentations']},
                    train_seconds=train_seconds, validation_seconds=validation_seconds,
                    checkpoint_seconds=checkpoint_seconds, cumulative_train_seconds=cumulative['train'],
                    cumulative_validation_seconds=cumulative['validation'], cumulative_checkpoint_seconds=cumulative['checkpoint'],
                    cumulative_counts=counts, timing_scope=metadata['timing_scope'])
                append_json(logs / 'training_measurements.jsonl', row)
                previous_counts = counts
                print(f'[DRL-GNN {completed}/{len(seeds)}] seed={slot_seeds[slot]} '
                      f'energy={metrics["total_energy"]:.6f} validation={energy:.6f}', flush=True)
                interval_clock = time.perf_counter()
                reset_slot(slot)
            if completed == len(seeds):
                break
    if agent.steps != total_assignments or agent.updates != total_updates:
        raise RuntimeError('Actual training exposure did not match the fixed assignment budget')
    summary = dict(**metadata, completed_episodes=completed, counts=counts,
        timing_seconds=cumulative, best_validation_energy=best,
        seeds_started=list(seeds), final_rollout_retained=True)
    (logs / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return agent, directory


def train_main():
    train(parse_scenario('train', 'drl_gnn'))


def evaluate_main():
    scenario = parse_scenario('eval', 'drl_gnn')
    device = (scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = output_dir(scenario, 'ckpts')
    agent, _ = load_checkpoint(checkpoint / 'best.pth', scenario, device)
    directory = output_dir(scenario, 'run'); directory.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(agent, scenario, scenario.seeds, directory=directory / 'measurements',
                             checkpoint_directory=checkpoint)
    path = directory / 'evaluation.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (directory / 'summary.json').write_text(json.dumps(dict(**objective_metadata(), summary=summary,
        protocol=scenario.protocol().metadata(), source_checkpoint=str(checkpoint / 'best.pth')),
        indent=2), encoding='utf-8')
    print(f'DRL-GNN evaluation saved: {path}', flush=True)
