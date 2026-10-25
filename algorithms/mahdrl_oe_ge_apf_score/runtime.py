"""Shared variant-only training/evaluation control path."""
import json
import csv
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from common.f11 import F11Logger, protect_training_outputs
from common.f18_state_normalization import normalization_diagnostics
from common.f22_cache import check_cache
from common.f22_instances import write_run_metadata, episode_identity
from common.experiment_measurements import TrainingRecorder, EvaluationRecorder, episode_metrics, collect_validation
from algorithms.mahdrl_oe.reward import finish_lower_phase, LowerRewardAudit
from environments.mahdrl_oe_ge_apf_score import CloudWorkflowEnv_VMAgents, MANAGER_ACTION_TABLE
from .agent import TAAgent, SetD3QNAgent
from .config import ALGORITHM, DISPLAY_NAME, TA_HIDDEN, learning_kwargs, variant_metadata
from .trace import SetDecisionTrace, snapshot_decision


FILES = {"TA": "manager", "HA": "host", "VA": "vm"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_kwargs(scene, seed):
    return dict(dax_paths=scene.dax_paths(), horizon=1e9, arrival_lambda=.03,
        random_seed=int(seed), max_ready_tasks="auto", normalize=True, workflows_per_episode=50,
        num_hosts=scene.num_hosts, vms_per_host=scene.vms_per_host,
        pc_tiers=(1., 2., 4., 6., 8.), bw_tiers=(1000., 2000., 4000., 6000., 8000.),
        deadline_mode="cache_fcfs", deadline_cache_path=str(scene.cache_path), deadline_cache_strict=True,
        deadline_alpha_small=2., deadline_alpha_large=3., deadline_alpha_small_prob=scene.small_probability,
        manager_alpha_delay=.75, manager_delay_mode="tardiness", **scene.normalization_kwargs())


def build_env(scene, seed):
    env = CloudWorkflowEnv_VMAgents(**build_env_kwargs(scene, seed))
    env.energy_reward_scale = .001
    env.task_baseline_norm = 300.
    env.energy_norm_per_mi_ref = 20.
    env.alpha_delay_host = env.alpha_delay_vm = .75
    return env


def validate_inputs(scene):
    scene.validate_calibration()
    if scene.protocol():
        check_cache(scene)
    else:
        # Legacy/debug runs still cannot borrow a cache from another topology.
        obj = json.loads(scene.cache_path.read_text(encoding="utf-8"))
        expected = dict(num_hosts=scene.num_hosts, vms_per_host=list(scene.vms_per_host),
                        task=scene.task, resource=scene.resource)
        for key, value in expected.items():
            if obj.get("meta", {}).get(key) != value:
                raise ValueError("Deadline cache configuration mismatch: "+key)


def build_agents(scene):
    device = torch.device(scene.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agents = {"TA": TAAgent(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN, **learning_kwargs("TA", device)),
              "HA": SetD3QNAgent(23, 10, **learning_kwargs("HA", device)),
              "VA": SetD3QNAgent(30, 7, **learning_kwargs("VA", device))}
    for role, agent in agents.items():
        agent.bind(scene, role)
    return agents


def save_agents(agents, directory, kind):
    group = uuid.uuid4().hex
    for role, agent in agents.items():
        agent.checkpoint_group = group
        name = f"best_{FILES[role]}.pth" if kind == "best" else f"{FILES[role]}_{kind}.pth"
        agent.save(str(Path(directory)/name))


def load_agents(scene):
    agents = build_agents(scene)
    for role, agent in agents.items():
        agent.load(str(scene.checkpoint_dir()/f"best_{FILES[role]}.pth"), require_learning_model=True)
    if len({a.loaded_group for a in agents.values()}) != 1:
        raise ValueError("TA/HA/VA checkpoints are from different saved groups")
    return agents


def run_episode(env, agents, *, training=False, warmup=.03, phase_log=None):
    traces = {r: SetDecisionTrace(agents[r], warmup) for r in ("HA", "VA")} if training else {}
    audit = LowerRewardAudit(traces.get("HA"), traces.get("VA"))
    phases, returns = 0, dict(TA=0., HA=0., VA=0.)
    state, mask = env.get_manager_state(), env.get_manager_action_mask()
    action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
    env.apply_manager_delta(MANAGER_ACTION_TABLE[action])
    while not env.done_flag:
        rewards = {"HA": [], "VA": []}
        while True:
            host, ok = env.get_host_state_for_next_assignment()
            if not ok:
                break
            host = snapshot_decision(host)
            if training:
                traces["HA"].advance(host["obs"], host["mask"])
            ah = agents["HA"].select_action(host["obs"], host["mask"], deterministic=not training,
                    count_step=training, performance_times=host["performance_times"])
            env.host_select(ah)
            vm, ok = env.get_vm_state_for_current_task()
            if not ok:
                raise RuntimeError("Selected APF host has no legal VM")
            vm = snapshot_decision(vm)
            if training:
                traces["VA"].advance(vm["obs"], vm["mask"])
            av = agents["VA"].select_action(vm["obs"], vm["mask"], deterministic=not training,
                    count_step=training, performance_times=vm["performance_times"])
            rh, rv, info = env.vm_assign(av)
            audit.assignment(info)
            rewards["HA"].append(float(rh)); rewards["VA"].append(float(rv))
            if training:
                traces["HA"].record(host["obs"], host["mask"], ah, rh, host_index=info["host_id"])
                traces["VA"].record(vm["obs"], vm["mask"], av, rv, host_index=info["host_id"])
        rm, info = env.finish_phase_and_advance()
        audit.phase(info, traces.get("VA"))
        if training:
            finish_lower_phase(env, traces["HA"], traces["VA"], info)
        unreturned = float(info.get("lower_assignment_energy_unemitted", 0.))
        if unreturned:
            rewards["HA"].append(-(1-env.alpha_delay_host)*unreturned*env.energy_reward_scale)
            rewards["VA"].append(-(1-env.alpha_delay_vm)*unreturned*env.energy_reward_scale)
        next_state, next_mask = env.get_manager_state(), env.get_manager_action_mask()
        if training and info.get("assign_cnt", 0) > 0:
            agents["TA"].f11.truncation_cutoffs += int(info.get("truncated", False))
            agents["TA"].remember(state, mask, action, rm, next_state, next_mask, float(env.done_flag))
            if len(agents["TA"].buffer) >= max(agents["TA"].batch_size, int(agents["TA"].buffer.maxlen*warmup)):
                agents["TA"].update()
        returns["TA"] += float(rm)
        for role in rewards:
            returns[role] += float(np.mean(rewards[role])) if rewards[role] else 0.
        phases += 1
        if phase_log:
            phase_log(dict(phase=phases, manager_reward=float(rm), **info))
        state, mask = next_state, next_mask
        if not env.done_flag:
            action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
            env.apply_manager_delta(MANAGER_ACTION_TABLE[action])
    if training:
        for trace in traces.values():
            trace.finish(truncated=not env._episode_should_end())
    return dict(phase_count=phases, rewards={r: v/max(1, phases) for r,v in returns.items()},
                reward_audit=audit.summary(env, traces.get("HA"), traces.get("VA")),
                **episode_metrics(env))


def apf_report(env):
    windows = {}
    for r in env._records:
        if r.server_id not in windows:
            windows[r.server_id] = [float(r.start_time), float(r.end_time)]
        else:
            windows[r.server_id][0] = min(windows[r.server_id][0], float(r.start_time))
            windows[r.server_id][1] = max(windows[r.server_id][1], float(r.end_time))
    return dict(variant=variant_metadata(), decisions=env.apf_decisions,
                host_windows={str(h): dict(start=w[0], end=w[1], duration=w[1]-w[0]) for h,w in windows.items()})


def train(scene):
    protect_training_outputs(scene.checkpoint_dir(), scene.log_dir())
    validate_inputs(scene)
    set_seed(scene.model_seed)
    env, agents = build_env(scene, scene.training_seed(0)), build_agents(scene)
    env.reset()
    scene.checkpoint_dir().mkdir(parents=True)
    scene.log_dir().mkdir(parents=True)
    for directory in (scene.checkpoint_dir(), scene.log_dir()):
        write_run_metadata(directory, scene)
        write_json(directory/"algorithm_variant.json", variant_metadata())
    recorder = TrainingRecorder(scene.log_dir(), scene, ALGORITHM, env, agents)
    f11 = F11Logger(scene.log_dir())
    best = float("inf")
    if agents["TA"].device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agents["TA"].device)
    try:
        for episode in range(scene.episodes):
            if episode:
                env.reset(seed=scene.training_seed(episode))
            result = run_episode(env, agents, training=True,
                phase_log=lambda row: append_json(scene.log_dir()/"phases.jsonl", dict(episode=episode, **row)))
            f11.log(episode, agents, normalization=normalization_diagnostics(env),
                    instance=episode_identity(env, scene), reward_audit=result["reward_audit"], pending=0)
            def validate():
                values = []
                for seed in scene.validation_seeds():
                    validation = build_env(scene, seed)
                    validation.reset()
                    values.append(run_episode(validation, agents))
                    collect_validation(validation)
                return float(np.mean([v["total_energy"] for v in values]))
            energy = recorder.validation(validate)
            append_json(scene.log_dir()/"episodes.jsonl", dict(episode=episode,
                        instance_seed=scene.training_seed(episode), validation_energy=energy, **result))
            write_json(scene.log_dir()/f"apf_episode{episode}.json", apf_report(env))
            if energy < best:
                best = energy
                save_agents(agents, scene.checkpoint_dir(), "best")
            print(f"[{DISPLAY_NAME}] episode={episode+1}/{scene.episodes} energy={result['total_energy']:.6f} "
                  f"validation_energy={energy:.6f}", flush=True)
        save_agents(agents, scene.checkpoint_dir(), "final")
        write_json(scene.log_dir()/"training_summary.json", dict(episodes=scene.episodes,
            best_validation_energy=best, peak_cuda_allocated_bytes=(
                torch.cuda.max_memory_allocated(agents["TA"].device) if agents["TA"].device.type == "cuda" else None),
            parameter_counts={r: sum(p.numel() for p in a.online.parameters()) for r,a in agents.items()}))
    finally:
        recorder.close()
    print(f"[MAHDRL-AS train completed] models={scene.checkpoint_dir().resolve()} logs={scene.log_dir().resolve()}")
    return agents


def evaluate(scene):
    directory = scene.evaluation_dir()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation artifacts: {directory}")
    validate_inputs(scene)
    set_seed(scene.model_seed)
    agents = load_agents(scene)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"algorithm_variant.json", variant_metadata())
    write_run_metadata(directory, scene)
    values = []
    for seed in scene.seeds:
        set_seed(scene.model_seed)
        env = build_env(scene, seed)
        env.reset()
        recorder = EvaluationRecorder(scene, ALGORITHM, env, agents,
            directory=directory/"measurements", checkpoint_directory=str(scene.checkpoint_dir()))
        try:
            result = run_episode(env, agents)
        finally:
            recorder.finish()
        write_json(directory/f"apf_seed{seed}.json", apf_report(env))
        append_json(directory/"evaluation.jsonl", dict(seed=seed, **result))
        csv_path = directory/"evaluation_mahdrl_as.csv"
        row = dict(seed=seed, model_seed=scene.model_seed, **episode_metrics(env))
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        values.append(result)
    summary = {key: dict(mean=float(np.mean([v[key] for v in values])),
                        std=float(np.std([v[key] for v in values], ddof=1)) if len(values)>1 else 0.)
               for key in ("total_energy", "wf_success_rate", "wf_avg_tardiness", "makespan", "wf_avg_response_time")}
    write_json(directory/"summary.json", dict(algorithm=ALGORITHM, seeds=list(scene.seeds), metrics=summary))
    print(f"[MAHDRL-AS eval completed] results={directory.resolve()}")
    return summary
