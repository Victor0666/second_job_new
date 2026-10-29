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
from common.scenarios import ROOT
from .scenario import parse_scenario
from algorithms.drl_gnn.runner import (seed_everything, sync, env_kwargs, checked_reference,
    append_json, planned_assignments)
from common.workflow_opt import ENERGY_ACCOUNTING, COMMUNICATION_MODEL
from .config import (COMMON_REWARD, DRLGNNConfig, INPUT_ENCODING, MODEL_VERSION,
                     REWARD_PROTOCOL, SOURCE_COMMIT, SOURCE_REPOSITORY,
                     TASK_FEATURES, VM_FEATURES, SCHEDULING_PROTOCOL, POWER_REFERENCE_W, ALGORITHM, DISPLAY_NAME, GRAPHS_PER_BACKWARD, PPO_UPDATE_EXECUTION)
from .environment import FastDRLGNNEnv as DRLGNNEnv
from .ppo import PPOAgent, Transition
from algorithms.drl_gnn.runtime import cuda_memory_usage
from .selection import selection_metadata, validation_selection, selection_key, validate_checkpoint_selection


def output_dir(scenario, kind):
    if kind == 'ckpts':
        return scenario.checkpoint_dir(ALGORITHM)
    if kind == 'logs':
        return scenario.log_dir(ALGORITHM)
    if kind == 'run':
        return scenario.evaluation_dir
    return ROOT / 'outputs' / kind / MODEL_VERSION / scenario.model_name(ALGORITHM) / (
        scenario.task + scenario.resource + '_' + scenario.deadline)


def objective_metadata():
    return dict(algorithm=ALGORITHM, display_name=DISPLAY_NAME, checkpoint_selection=selection_metadata(), source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT,
        model_version=MODEL_VERSION, policy_input=INPUT_ENCODING,
        observation_fields=dict(task=list(TASK_FEATURES), vm=list(VM_FEATURES)),
        observation_scaling=dict(
            task=['identity', 'identity', '/workload_mi_ref', '/T_ref', '/T_ref', '/T_ref',
                  '/input_bits_ref', '/output_bits_ref'],
            vm=['/T_ref', '*vm_pc_ref', '/(100W/host_pc_ref)', '/vm_bw_ref', '/T_ref',
                'identity', '/100W', '/100W', 'identity'],
            clipping=False, completion_time='absolute', deadline_remaining='signed'),
        reward_protocol=REWARD_PROTOCOL, reward_settings=COMMON_REWARD,
        scheduling_protocol=SCHEDULING_PROTOCOL,
        energy_accounting=ENERGY_ACCOUNTING, communication_model=COMMUNICATION_MODEL,
        evaluation='argmax with author single-graph train-mode BN on an isolated copy')


def selection_expectation(scenario):
    return dict(model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
        protocol=scenario.selection_protocol(), validation_seeds=list(scenario.protocol().validation),
        source_train_scene=scenario.normalization_kwargs()['source_train_scene'], model_seed=scenario.model_seed)


def save_checkpoint(path, agent, scenario, completed, best_energy, environment_config, budget,
                    *, kind, candidate, best_selection, validation_rows, best_validation_rows):
    evidence = dict(selection_expectation(scenario), checkpoint_kind=kind,
        episode=completed - 1, completed_episodes=completed, selected_episode=best_selection['episode'],
        validation_selection=deepcopy(candidate), best_validation_selection=deepcopy(best_selection),
        validation_rows=deepcopy(validation_rows), best_validation_rows=deepcopy(best_validation_rows))
    validate_checkpoint_selection(evidence, selection_expectation(scenario))
    obj = dict(**objective_metadata(), config=agent.cfg.as_dict(),
        execution=dict(ppo_update=PPO_UPDATE_EXECUTION, graphs_per_backward=agent.graphs_per_backward, cuda_memory=agent.network.cuda_memory_policy),
        network=agent.network.state_dict(), optimizer=agent.optimizer.state_dict(),
        selection_evidence=evidence, completed_episodes=int(completed), steps=agent.steps, updates=agent.updates,
        best_validation_energy=float(best_energy), protocol=scenario.selection_protocol(),
        shared_reference_normalization=checked_reference(scenario, environment_config),
        scenario=scenario.describe(ALGORITHM), environment=environment_config, budget=budget,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        resume_supported=False)
    # Best snapshots can be inside a rollout; they are evaluation checkpoints, not resumable runs.
    torch.save(obj, path)


def load_checkpoint(path, scenario, device='cpu', require_best=True):
    # Deserialize weights, optimizer and RNG on CPU before the budgeted CUDA model.
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key, value in objective_metadata().items():
        if obj.get(key) != value:
            raise ValueError('DRL-GNN-Fast checkpoint mismatch: ' + key)
    if obj.get('protocol') != scenario.selection_protocol():
        raise ValueError('DRL-GNN-Fast checkpoint data partition mismatch')
    source = obj['scenario']
    expected = (scenario.train_task if scenario.mode == 'eval' else scenario.task,
                (scenario.train_resource or scenario.resource) if scenario.mode == 'eval' else scenario.resource,
                (scenario.train_deadline or scenario.deadline) if scenario.mode == 'eval' else scenario.deadline)
    if (source['task'], source['resource'], source['deadline']) != expected:
        raise ValueError('DRL-GNN-Fast checkpoint source training scenario mismatch')
    if source['model_seed'] != scenario.model_seed:
        raise ValueError('DRL-GNN-Fast checkpoint model seed mismatch')
    reference = checked_reference(scenario)
    for key in ('normalization_version', 'normalization_config_hash', 'source_train_scene',
                'experiment_protocol_hash', 'observation_scales', 'power_reference_w'):
        if obj.get('shared_reference_normalization', {}).get(key) != reference.get(key):
            raise ValueError('DRL-GNN public experiment reference mismatch: ' + key)
    evidence = validate_checkpoint_selection(obj.get('selection_evidence'), selection_expectation(scenario))
    if evidence['completed_episodes'] != obj.get('completed_episodes'):
        raise ValueError('DRL-GNN-Fast selection episode mismatch')
    if require_best and evidence['checkpoint_kind'] != 'best':
        raise ValueError('DRL-GNN-Fast evaluation requires best checkpoint evidence')
    config = dict(obj['config']); config['device'] = str(device)
    agent = PPOAgent(DRLGNNConfig(**config), device=device, graphs_per_backward=obj['execution']['graphs_per_backward'])
    agent.network.load_state_dict(obj['network'])
    agent.optimizer.load_state_dict(obj['optimizer'])
    agent.steps = int(obj['steps']); agent.updates = int(obj['updates'])
    return agent, obj


def run_episode(env, agent, seed, evaluation=None):
    env.reset(seed=int(seed))
    recorder = EvaluationRecorder(evaluation[0], ALGORITHM, env, {'GIN-PPO': agent},
        evaluation[1], checkpoint_directory=str(evaluation[2])) if evaluation else None
    total_reward = 0.
    try:
        while not env.done_flag:
            state = env.graph_state()
            action = agent.select_action(state)
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
    kwargs = dict(kwargs or env_kwargs(scenario, seeds[0]))
    checked_reference(scenario, kwargs)
    frozen = deepcopy(agent)
    frozen.network.clear_cache()
    frozen._bootstrap.clear()
    snapshot = deepcopy(frozen.network.state_dict())
    rows = []
    for seed in seeds:
        frozen.network.load_state_dict(snapshot)
        env = DRLGNNEnv(**dict(kwargs, random_seed=int(seed)))
        evaluation = (scenario, directory, checkpoint_directory or output_dir(scenario, 'ckpts')) if directory else None
        result = run_episode(env, frozen, seed, evaluation)
        rows.append(dict(seed=int(seed), **result['metrics']))
    summary = {key: dict(mean=float(np.mean([r[key] for r in rows])),
        std=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.)
        for key in rows[0] if key != 'seed'}
    return rows, summary


def train(scenario, config=None, kwargs=None, out=None):
    protocol = scenario.protocol()
    if protocol is None or not 0 < scenario.episodes <= protocol.episodes:
        raise ValueError('DRL-GNN training requires the fixed common episode budget')
    checked_reference(scenario)
    cfg = config or DRLGNNConfig(device=(scenario.paper_options or {}).get('device') or (
        'cuda' if torch.cuda.is_available() else 'cpu'), seed=scenario.model_seed)
    if cfg.seed != scenario.model_seed:
        raise ValueError('Model initialization seed must match the common scenario')
    directory = Path(out) if out is not None else output_dir(scenario, 'ckpts')
    logs = directory / 'logs' if out is not None else output_dir(scenario, 'logs')
    protect_training_outputs(directory, logs)
    directory.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kwargs or env_kwargs(scenario, protocol.train[0]))
    reference = checked_reference(scenario, kwargs)
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
    agent = PPOAgent(cfg, device=cfg.device, graphs_per_backward=(scenario.paper_options or {}).get(
        'graphs_per_backward', GRAPHS_PER_BACKWARD))
    for path in (directory, logs):
        write_run_metadata(path, scenario)
    metadata = dict(**objective_metadata(), config=cfg.as_dict(), budget=budget,
        shared_reference_normalization=reference,
        execution=dict(ppo_update=PPO_UPDATE_EXECUTION, graphs_per_backward=agent.graphs_per_backward, cuda_memory=agent.network.cuda_memory_policy),
        protocol=scenario.selection_protocol(), parameters=sum(p.numel() for p in agent.network.parameters()),
        numerical_precision='float32', timing_scope='global nonoverlapping intervals between completed episodes; 4 environments share one policy')
    (logs / 'measurement_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    states = [None] * cfg.num_envs; slot_seeds = [None] * cfg.num_envs
    slot_returns = [0.] * cfg.num_envs; next_seed = 0; completed = 0; best = None; best_rows = None
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
        agent._bootstrap.pop(slot, None)
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
    print(f'[DRL-GNN-Fast train] {scenario.task}{scenario.resource}/{scenario.deadline} '
          f'episodes={len(seeds)} assignments={total_assignments} device={agent.device}', flush=True)
    while completed < len(seeds):
        for _ in range(cfg.num_steps):
            for slot, env in enumerate(envs):
                state = states[slot]
                if state is None:
                    continue
                action, logp, value = agent.act(state, slot=slot)
                reward, done, _ = env.pair_step(action)
                next_state = None if done else env.graph_state()
                next_value = 0. if done else agent.bootstrap_value(next_state, slot)
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
                candidate = validation_selection(validation_rows)
                energy = candidate['mean_total_energy']
                selected = best is None or selection_key(candidate) < selection_key(best)
                checkpoint_started = time.perf_counter()
                if selected:
                    best = dict(candidate, episode=completed - 1, completed_episodes=completed)
                    best_rows = deepcopy(validation_rows)
                    save_checkpoint(directory / 'best.pth', agent, scenario, completed,
                        best['mean_total_energy'], kwargs, budget, kind='best', candidate=candidate,
                        best_selection=best, validation_rows=validation_rows, best_validation_rows=best_rows)
                if completed == len(seeds):
                    save_checkpoint(directory / 'final.pth', agent, scenario, completed,
                        best['mean_total_energy'], kwargs, budget, kind='final', candidate=candidate,
                        best_selection=best, validation_rows=validation_rows, best_validation_rows=best_rows)
                append_json(logs / 'validation_selection.jsonl', dict(algorithm=ALGORITHM,
                    model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(),
                    episode=completed - 1, instance_seed=slot_seeds[slot], validation_rows=validation_rows,
                    validation_selection=candidate, best_validation_selection=best,
                    selected_as_best=selected, selected_episode=best['episode']))
                sync(agent); checkpoint_seconds = time.perf_counter() - checkpoint_started
                cumulative['checkpoint'] += checkpoint_seconds
                counts = dict(assignments=agent.steps, optimizer_steps=optimizer_steps,
                              sample_presentations=sample_presentations)
                row = dict(version=MEASUREMENT_VERSION, algorithm=ALGORITHM, episode=completed - 1,
                    instance_seed=slot_seeds[slot], model_seed=scenario.model_seed, metrics=metrics,
                    validation=validation_rows, validation_summary=validation, instance=identity,
                    validation_selection=candidate, selected_as_best=selected, selected_episode=best['episode'],
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
                print(f'[DRL-GNN-Fast {completed}/{len(seeds)}] seed={slot_seeds[slot]} '
                      f'energy={metrics["total_energy"]:.6f} validation={energy:.6f} '
                      f'SR={candidate["workflow_success_rate"]:.4f} feasible={candidate["feasible"]} selected={selected}', flush=True)
                interval_clock = time.perf_counter()
                reset_slot(slot)
            if completed == len(seeds):
                break
    if agent.steps != total_assignments or agent.updates != total_updates:
        raise RuntimeError('Actual training exposure did not match the fixed assignment budget')
    summary = dict(**metadata, completed_episodes=completed, counts=counts,
        cuda_memory_usage=cuda_memory_usage(agent.device),
        timing_seconds=cumulative, best_validation_energy=best['mean_total_energy'],
        selected_episode=best['episode'], best_validation_selection=best,
        final_validation_selection=candidate, feasible_model_found=best['feasible'], bootstrap_cache_hits=agent.bootstrap_hits,
        seeds_started=list(seeds), final_rollout_retained=True)
    (logs / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return agent, directory


def train_main():
    train(parse_scenario('train'))


def evaluate_main():
    scenario = parse_scenario('eval')
    device = (scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = output_dir(scenario, 'ckpts')
    agent, checkpoint_data = load_checkpoint(checkpoint / 'best.pth', scenario, device)
    directory = output_dir(scenario, 'run')
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Refusing to overwrite DRL-GNN-Fast evaluation output: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(agent, scenario, scenario.seeds, directory=directory / 'measurements',
                             checkpoint_directory=checkpoint)
    path = directory / 'evaluation.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (directory / 'summary.json').write_text(json.dumps(dict(**objective_metadata(), summary=summary,
        selection_evidence=checkpoint_data['selection_evidence'],
        shared_reference_normalization=checked_reference(scenario),
        execution=dict(ppo_update=PPO_UPDATE_EXECUTION, graphs_per_backward=agent.graphs_per_backward, cuda_memory=agent.network.cuda_memory_policy),
        cuda_memory_usage=cuda_memory_usage(agent.device),
        protocol=scenario.selection_protocol(), source_checkpoint=str(checkpoint / 'best.pth')),
        indent=2), encoding='utf-8')
    print(f'DRL-GNN-Fast evaluation saved: {path}', flush=True)
