"""Paired-instance sampling, isolated validation and versioned GAT checkpoints."""
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch_geometric

from common.experiment_measurements import EvaluationRecorder, episode_metrics, VERSION as MEASUREMENT_VERSION
from common.f11 import protect_training_outputs
from common.f18_state_normalization import check_checkpoint
from common.f22_instances import instance_identity, write_run_metadata
from common.read_xml_opt_Tsize import load_workflow_from_dax
from common.scenarios import ROOT, parse_scenario
from common.workflow_opt import ENERGY_ACCOUNTING, COMMUNICATION_MODEL
from .config import (COMMON_REWARD, INPUT_ENCODING, MODEL_VERSION, REWARD_PROTOCOL,
                     SCHEDULING_PROTOCOL, SOURCE_COMMIT, SOURCE_REPOSITORY, TwoStageGATConfig)
from .environment import TwoStageGATEnv
from .ppo import PPOAgent, Transition


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        deadline_alpha_probs=scenario.deadline_probs, energy_check_on_done=True,
        **COMMON_REWARD, **scenario.normalization_kwargs())


def output_dir(scenario, kind):
    if kind == 'ckpts':
        return scenario.checkpoint_dir('two_stage_gat')
    if kind == 'logs':
        return scenario.log_dir('two_stage_gat')
    return ROOT / 'outputs' / kind / MODEL_VERSION / scenario.model_name('two_stage_gat') / (
        scenario.task + scenario.resource + '_' + scenario.deadline)


def artifact_name(scenario, kind):
    count = f'_{len(scenario.seeds)}seeds' if kind != 'train' else ''
    source = (scenario.train_task, scenario.train_resource or scenario.resource,
              scenario.train_deadline or scenario.deadline)
    suffix = f'_from_task{source[0]}_res{source[1]}_{source[2]}' if kind != 'train' and source != (
        scenario.task, scenario.resource, scenario.deadline) else ''
    return f'{kind}_two_stage_gat_task{scenario.task}_res{scenario.resource}{count}_{scenario.deadline}_model{scenario.model_seed}{suffix}'


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def objective_metadata():
    return dict(source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT,
        model_version=MODEL_VERSION, policy_input=INPUT_ENCODING,
        reward_protocol=REWARD_PROTOCOL, reward_settings=COMMON_REWARD,
        scheduling_protocol=SCHEDULING_PROTOCOL, energy_accounting=ENERGY_ACCOUNTING,
        communication_model=COMMUNICATION_MODEL, pooling='real_arrived_tasks_and_real_VMs',
        evaluation='sequential task argmax then conditional VM argmax; paper/common protocol')


def save_checkpoint(path, agent, scenario, completed, best_energy, environment_config, budget):
    obj = dict(**objective_metadata(), config=agent.cfg.as_dict(),
        network=agent.network.state_dict(), optimizer=agent.optimizer.state_dict(),
        completed_episodes=int(completed), steps=agent.steps, updates=agent.updates,
        best_validation_energy=float(best_energy), protocol=scenario.protocol().metadata(),
        normalization=scenario.normalization_metadata(), scenario=scenario.describe('two_stage_gat'),
        environment=environment_config, budget=budget, versions=dict(torch=torch.__version__,
            pyg=torch_geometric.__version__, numpy=np.__version__), resume_supported=False)
    torch.save(obj, path)


def load_checkpoint(path, scenario, device='cpu'):
    obj = torch.load(path, map_location=device, weights_only=False)
    for key, value in objective_metadata().items():
        if obj.get(key) != value:
            raise ValueError('Two-stage GAT checkpoint mismatch: ' + key)
    if obj.get('protocol') != scenario.protocol().metadata():
        raise ValueError('Checkpoint data partition mismatch')
    source = obj['scenario']
    expected = (scenario.train_task if scenario.mode == 'eval' else scenario.task,
        (scenario.train_resource or scenario.resource) if scenario.mode == 'eval' else scenario.resource,
        (scenario.train_deadline or scenario.deadline) if scenario.mode == 'eval' else scenario.deadline)
    if (source['task'], source['resource'], source['deadline']) != expected or source['model_seed'] != scenario.model_seed:
        raise ValueError('Checkpoint source scenario/model seed mismatch')
    check_checkpoint(obj.get('normalization'), scenario.normalization_metadata())
    config = dict(obj['config']); config['device'] = str(device)
    agent = PPOAgent(TwoStageGATConfig(**config))
    agent.network.load_state_dict(obj['network']); agent.optimizer.load_state_dict(obj['optimizer'])
    agent.steps = int(obj['steps']); agent.updates = int(obj['updates'])
    return agent, obj


def run_episode(env, agent, seed, evaluation=None):
    env.reset(seed=int(seed))
    recorder = EvaluationRecorder(evaluation[0], 'two_stage_gat', env, {'GAT-PPO': agent},
        evaluation[1], checkpoint_directory=str(evaluation[2])) if evaluation else None
    total_reward = 0.
    try:
        while not env.done_flag:
            action, _, _ = agent.act(env.graph_state(), deterministic=True, count_step=False)
            reward, _, _ = env.pair_step(action)
            total_reward += reward
        if env.completed_workflows != env.workflows_per_episode or any(s != 'Finished' for s in env.task_state):
            raise RuntimeError('Episode ended before all reservations physically completed')
        metrics = episode_metrics(env)
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError('Nonfinite public metrics')
        return dict(metrics=metrics, external_return=float(total_reward),
                    assignments=env.assignment_count, queued_assignments=env.queued_assignments)
    finally:
        if recorder:
            recorder.finish()


def evaluate(agent, scenario, seeds, split='test', kwargs=None, directory=None, checkpoint_directory=None):
    protocol = scenario.protocol()
    if protocol is None or split not in ('test', 'validation') or not seeds:
        raise ValueError('Evaluation requires the frozen protocol and a nonempty partition')
    if not set(seeds) <= set(protocol.validation if split == 'validation' else protocol.test):
        raise ValueError('Evaluation attempted to cross the frozen partition')
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    training = agent.network.training
    rows = []
    try:
        agent.network.eval()
        for seed in seeds:
            env = TwoStageGATEnv(**dict(kwargs or env_kwargs(scenario, seed), random_seed=int(seed)))
            measurement = (scenario, directory, checkpoint_directory or output_dir(scenario, 'ckpts')) if directory else None
            result = run_episode(env, agent, seed, measurement)
            rows.append(dict(seed=int(seed), **result['metrics']))
    finally:
        agent.network.train(training)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    summary = {key: dict(mean=float(np.mean([r[key] for r in rows])),
        std=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.)
        for key in rows[0] if key != 'seed'}
    return rows, summary


def planned_assignments(env, seeds):
    """Budget metadata only; future templates never enter policy observations."""
    if env.deadline_mode != 'cache_fcfs':
        raise ValueError('Training budget requires paired cached instances')
    counts = {Path(p).name.casefold(): len(load_workflow_from_dax(p, seed=0)[1])
              for p in dict.fromkeys(env.dax_paths)}
    total = 0
    for seed in seeds:
        templates = env._deadline_cache[int(seed)]['wf_dax_names'][:env.workflows_per_episode]
        if len(templates) != env.workflows_per_episode:
            raise ValueError('Incomplete cached instance')
        total += sum(counts[Path(p).name.casefold()] for p in templates)
    return total


def train(scenario, config=None, kwargs=None, out=None):
    protocol = scenario.protocol()
    if protocol is None or not 0 < scenario.episodes <= protocol.episodes:
        raise ValueError('Training requires the fixed paired episode budget')
    scenario.normalization_metadata()
    cfg = config or TwoStageGATConfig(seed=scenario.model_seed,
        device=(scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if cfg.seed != scenario.model_seed:
        raise ValueError('Model seed must match the common protocol')
    kwargs = dict(kwargs or env_kwargs(scenario, protocol.train[0]))
    if any(kwargs.get(key) != value for key, value in COMMON_REWARD.items()) or kwargs.get('normalization_mode') == 'legacy':
        raise ValueError('Formal GAT uses public reward coefficients and frozen source input scales')
    directory = Path(out) if out is not None else output_dir(scenario, 'ckpts')
    logs = directory / 'logs' if out is not None else output_dir(scenario, 'logs')
    protect_training_outputs(directory, logs)
    for folder in (directory, logs):
        folder.mkdir(parents=True, exist_ok=True); write_run_metadata(folder, scenario)
    seeds = tuple(protocol.train[:scenario.episodes])
    queues = [iter(seeds[r::cfg.num_envs]) for r in range(cfg.num_envs)]
    envs = [TwoStageGATEnv(**dict(kwargs, random_seed=seeds[0])) for _ in queues]
    assignments_by_slot = [planned_assignments(envs[0], seeds[r::cfg.num_envs]) for r in range(cfg.num_envs)]
    total_assignments = sum(assignments_by_slot)
    total_updates = max(math.ceil(n / cfg.num_steps) for n in assignments_by_slot)
    budget = dict(episodes=len(seeds), assignments=total_assignments,
        assignments_by_slot=assignments_by_slot, num_envs=cfg.num_envs, steps_per_env=cfg.num_steps,
        full_rollout=cfg.batch_size, planned_updates=total_updates,
        original_200000_step_limit_replaced=True, seed_dispatch='slot r: seeds[r::num_envs]')
    seed_everything(scenario.model_seed)
    agent = PPOAgent(cfg)
    metadata = dict(**objective_metadata(), config=cfg.as_dict(), budget=budget,
        protocol=protocol.metadata(), normalization=scenario.normalization_metadata(),
        numerical_precision='float32', parameters=sum(p.numel() for p in agent.network.parameters()),
        versions=dict(torch=torch.__version__, pyg=torch_geometric.__version__, numpy=np.__version__),
        validation_timing='episode completion at current policy; normal full-rollout/final-budget update occurs first when coincident',
        timing_scope='nonoverlapping global intervals between completed episodes; all four environments share one policy')
    (logs / 'measurement_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    states = [None] * cfg.num_envs; slot_seeds = [None] * cfg.num_envs
    slot_returns = [0.] * cfg.num_envs; started_seeds = []
    completed = 0; best = float('inf'); records = []
    optimizer_steps = 0; sample_presentations = 0
    cumulative = dict(train=0., validation=0., checkpoint=0.)
    previous_counts = dict(assignments=0, optimizer_steps=0, sample_presentations=0)

    def reset_slot(slot):
        seed = next(queues[slot], None)
        slot_seeds[slot] = seed; slot_returns[slot] = 0.
        if seed is None:
            states[slot] = None
        else:
            started_seeds.append(seed); envs[slot].reset(seed=seed)
            states[slot] = envs[slot].graph_state()

    def update_rollout():
        nonlocal records, optimizer_steps, sample_presentations
        if not records:
            return
        lr = cfg.learning_rate * (1 - agent.updates / total_updates) if cfg.anneal_lr else cfg.learning_rate
        sync(agent); started = time.perf_counter()
        result = agent.update(records, learning_rate=lr)
        sync(agent); result['seconds'] = time.perf_counter() - started
        optimizer_steps += result['optimizer_steps']; sample_presentations += result['sample_presentations']
        result.update(assignments=agent.steps)
        append_json(logs / 'updates.jsonl', result); records = []

    for slot in range(cfg.num_envs):
        reset_slot(slot)
    sync(agent); interval_clock = time.perf_counter()
    print(f'[2SD-GAT train] {scenario.task}{scenario.resource}/{scenario.deadline} '
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
                if len(records) == cfg.batch_size or (done and completed + 1 == len(seeds)):
                    update_rollout()
                if not done:
                    continue
                completed += 1
                sync(agent); train_seconds = time.perf_counter() - interval_clock
                cumulative['train'] += train_seconds
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
                row = dict(version=MEASUREMENT_VERSION, algorithm='two_stage_gat', episode=completed - 1,
                    instance_seed=slot_seeds[slot], model_seed=scenario.model_seed, metrics=metrics,
                    validation=validation_rows, validation_summary=validation, instance=identity,
                    external_return=slot_returns[slot], episode_assignments=env.assignment_count,
                    queued_assignments=env.queued_assignments, policy_updates_at_validation=agent.updates,
                    assignment_calls=counts['assignments'] - previous_counts['assignments'],
                    optimizer_steps={'GAT-PPO.optimizer': counts['optimizer_steps'] - previous_counts['optimizer_steps']},
                    optimizer_sample_presentations={'GAT-PPO.optimizer': counts['sample_presentations'] - previous_counts['sample_presentations']},
                    train_seconds=train_seconds, validation_seconds=validation_seconds,
                    checkpoint_seconds=checkpoint_seconds, cumulative_train_seconds=cumulative['train'],
                    cumulative_validation_seconds=cumulative['validation'], cumulative_checkpoint_seconds=cumulative['checkpoint'],
                    cumulative_counts=counts, timing_scope=metadata['timing_scope'])
                append_json(logs / (artifact_name(scenario, 'train') + '.jsonl'), row)
                previous_counts = counts
                print(f'[2SD-GAT {completed}/{len(seeds)}] seed={slot_seeds[slot]} '
                    f'energy={metrics["total_energy"]:.6f} validation={energy:.6f}', flush=True)
                interval_clock = time.perf_counter(); reset_slot(slot)
            if completed == len(seeds):
                break
        update_rollout()
    if agent.steps != total_assignments or agent.updates != total_updates or sorted(started_seeds) != list(seeds):
        raise RuntimeError('Actual exposure did not match the frozen instance/assignment/update budget')
    summary = dict(**metadata, completed_episodes=completed, counts=counts, timing_seconds=cumulative,
        best_validation_energy=best, seeds_started=started_seeds, final_rollout_retained=True)
    (logs / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[2SD-GAT train completed] best={(directory / "best.pth").resolve()} '
          f'final={(directory / "final.pth").resolve()} logs={logs.resolve()}', flush=True)
    return agent, directory


def train_main():
    train(parse_scenario('train', 'two_stage_gat'))


def evaluate_main():
    scenario = parse_scenario('eval', 'two_stage_gat')
    device = (scenario.paper_options or {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = output_dir(scenario, 'ckpts')
    agent, _ = load_checkpoint(checkpoint / 'best.pth', scenario, device)
    directory = output_dir(scenario, 'run'); directory.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(agent, scenario, scenario.seeds, directory=directory / 'measurements',
        checkpoint_directory=checkpoint)
    path = directory / (artifact_name(scenario, 'eval') + '.csv')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary_path = directory / (artifact_name(scenario, 'summary') + '.json')
    summary_path.write_text(json.dumps(dict(**objective_metadata(), summary=summary,
        protocol=scenario.protocol().metadata(), source_checkpoint=str(checkpoint / 'best.pth')),
        indent=2), encoding='utf-8')
    print(f'[2SD-GAT eval completed] csv={path.resolve()} summary={summary_path.resolve()}', flush=True)
