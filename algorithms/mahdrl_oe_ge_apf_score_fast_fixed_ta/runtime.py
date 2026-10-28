"""Fixed sorting with the unchanged AS-Fast v2 HA/VA training lifecycle."""
import csv
import json
import uuid
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from common.f11 import F11Logger, protect_training_outputs
from common.f18_state_normalization import normalization_diagnostics
from common.f22_instances import write_run_metadata, episode_identity
from common.experiment_measurements import TrainingRecorder, EvaluationRecorder, episode_metrics, collect_validation
from algorithms.mahdrl_oe.reward import finish_lower_phase, LowerRewardAudit
from algorithms.mahdrl_oe_ge_apf_score_fast.agent import TAAgent as InitializationTA
from algorithms.mahdrl_oe_ge_apf_score_fast.runtime import (
    build_env_kwargs, validate_inputs, set_seed, write_json, append_json, apf_report as base_apf_report,
)
from common.mahdrl_ablation import (
    selection_metadata, validation_selection, selection_key, validate_checkpoint_selection,
)
from algorithms.mahdrl_oe_ge_apf_score_fast.trace import SetDecisionTrace, snapshot_decision
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from environments.mahdrl_oe_ge_apf_score_fast_fixed_ta import CloudWorkflowEnv_VMAgents
from .agent import SetD3QNAgent
from .config import TA_HIDDEN, learning_kwargs


FILES = {"HA": "host", "VA": "vm"}


def build_env(scene, seed):
    env = CloudWorkflowEnv_VMAgents(**build_env_kwargs(scene, seed),
        ordering_rule=scene.ordering_rule, profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    env.energy_reward_scale = .001
    env.task_baseline_norm = 300.
    env.energy_norm_per_mi_ref = 20.
    env.alpha_delay_host = env.alpha_delay_vm = .75
    return env


def build_agents(scene):
    device = torch.device(scene.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    options = dict(diagnostics_every=scene.diagnostics_every, profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    # Preserve baseline HA/VA initial tensors and post-construction RNG states.
    initialization_ta = InitializationTA(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN,
        **options, **learning_kwargs("TA", device))
    agents = {"HA": SetD3QNAgent(23, 10, **options, **learning_kwargs("HA", device)),
              "VA": SetD3QNAgent(30, 7, **options, **learning_kwargs("VA", device))}
    del initialization_ta
    for role, agent in agents.items():
        agent.bind(scene, role)
    return agents


def save_agents(agents, directory, kind, *, episode, candidate, best, rows, best_rows):
    if set(agents) != set(FILES) or any(agent.role != role for role, agent in agents.items()):
        raise ValueError("Fixed-TA checkpoints require HA and VA in their correct roles")
    cfg = agents["HA"].fixed_config
    if any(agent.fixed_config != cfg for agent in agents.values()):
        raise ValueError("Cannot save mixed fixed-ordering rules")
    records = {role: dict(deepcopy(agent.selection_expectation), checkpoint_kind=kind,
        episode=episode, completed_episodes=episode+1, selected_episode=best["episode"],
        validation_selection=deepcopy(candidate), best_validation_selection=deepcopy(best),
        validation_rows=deepcopy(rows), best_validation_rows=deepcopy(best_rows))
        for role, agent in agents.items()}
    for role, agent in agents.items():
        validate_checkpoint_selection(records[role], agent.selection_expectation)
    write_json(Path(directory)/"ta_policy.json", cfg.ta_policy_metadata())
    group = uuid.uuid4().hex
    for role, agent in agents.items():
        agent.checkpoint_group = group
        agent.normalization_metadata[cfg.checkpoint_key] = records[role]
        name = f"best_{FILES[role]}.pth" if kind == "best" else f"{FILES[role]}_{kind}.pth"
        agent.save(str(Path(directory)/name))


def load_agents(scene):
    cfg = scene.fixed_config
    manifest = scene.checkpoint_dir()/"ta_policy.json"
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != cfg.ta_policy_metadata():
        raise ValueError("Missing or mismatched fixed-TA policy manifest")
    agents = build_agents(scene)
    for role, agent in agents.items():
        agent.load(str(scene.checkpoint_dir()/f"best_{FILES[role]}.pth"), require_learning_model=True)
    if len({agent.loaded_group for agent in agents.values()}) != 1:
        raise ValueError("HA/VA checkpoints are from different saved groups")
    records = [{key: value for key, value in agent.loaded_selection.items() if key != "learner_role"}
               for agent in agents.values()]
    if any(record["checkpoint_kind"] != "best" for record in records):
        raise ValueError("Evaluation requires best checkpoints")
    if any(record != records[0] for record in records[1:]):
        raise ValueError("HA/VA checkpoints have different selection evidence or episodes")
    return agents


def run_episode(env, agents, *, training=False, warmup=.03, phase_log=None):
    if set(agents) != set(FILES) or any(agent.fixed_config != env.fixed_config for agent in agents.values()):
        raise ValueError("Fixed task ordering requires matching HA/VA learners")
    traces = {r: SetDecisionTrace(agents[r], warmup) for r in ("HA", "VA")} if training else {}
    audit = LowerRewardAudit(traces.get("HA"), traces.get("VA"))
    phases, returns = 0, dict(TA=0., HA=0., VA=0.)
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
        returns["TA"] += float(rm)
        for role in rewards:
            returns[role] += float(np.mean(rewards[role])) if rewards[role] else 0.
        phases += 1
        if phase_log:
            phase_log(dict(phase=phases, manager_reward=float(rm),
                ordering_rule=env.fixed_config.rule, task_order=list(env.last_task_order), **info))
    if training:
        for trace in traces.values():
            trace.finish(truncated=not env._episode_should_end())
    return dict(phase_count=phases, fixed_ta=env.fixed_config.ta_policy_metadata(), sorting_calls=env.fixed_sort_calls,
                rewards={r: v/max(1, phases) for r,v in returns.items()},
                reward_audit=audit.summary(env, traces.get("HA"), traces.get("VA")),
                **episode_metrics(env))


def apf_report(env):
    result = base_apf_report(env)
    result["variant"] = env.fixed_config.variant_metadata()
    return result


def train(scene):
    cfg = scene.fixed_config
    ALGORITHM, DISPLAY_NAME, MODEL_VERSION = cfg.algorithm, cfg.display_name, cfg.model_version
    protect_training_outputs(scene.checkpoint_dir(), scene.log_dir())
    validate_inputs(scene)
    set_seed(scene.model_seed)
    env, agents = build_env(scene, scene.training_seed(0)), build_agents(scene)
    env.reset()
    scene.checkpoint_dir().mkdir(parents=True)
    scene.log_dir().mkdir(parents=True)
    for directory in (scene.checkpoint_dir(), scene.log_dir()):
        write_run_metadata(directory, scene)
        write_json(directory/"algorithm_variant.json", cfg.variant_metadata())
        write_json(directory/"implementation.json", cfg.implementation_metadata())
    recorder = TrainingRecorder(scene.log_dir(), scene, ALGORITHM, env, agents)
    f11 = F11Logger(scene.log_dir())
    best, best_rows = None, None
    if agents["HA"].device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agents["HA"].device)
    try:
        for episode in range(scene.episodes):
            if episode:
                env.reset(seed=scene.training_seed(episode))
            result = run_episode(env, agents, training=True,
                phase_log=lambda row: append_json(scene.log_dir()/"phases.jsonl", dict(episode=episode, **row)))
            append_json(scene.log_dir()/"stage_timings.jsonl", dict(episode=episode, scope="training",
                environment=env.profiler.report(), learners={r: a.profiler.report() for r, a in agents.items()}))
            f11.log(episode, agents, normalization=normalization_diagnostics(env),
                    instance=episode_identity(env, scene), reward_audit=result["reward_audit"], pending=0)
            def validate():
                values = []
                for seed in scene.validation_seeds():
                    validation = build_env(scene, seed)
                    validation.reset()
                    result = run_episode(validation, agents)
                    if result['wf_count'] != validation.workflows_per_episode:
                        raise RuntimeError('Validation ended before all workflows arrived')
                    values.append(dict(seed=int(seed), **{key: result[key] for key in
                        ('total_energy', 'wf_count', 'wf_success_count', 'wf_success_rate')}))
                    append_json(scene.log_dir()/"stage_timings.jsonl", dict(episode=episode, scope="validation", seed=seed,
                        environment=validation.profiler.report()))
                    collect_validation(validation)
                return values
            rows = recorder.validation(validate)
            candidate = validation_selection(rows)
            selected = best is None or selection_key(candidate) < selection_key(best)
            if selected:
                best = dict(candidate, episode=episode, completed_episodes=episode+1)
                best_rows = deepcopy(rows)
                save_agents(agents, scene.checkpoint_dir(), "best", episode=episode,
                            candidate=candidate, best=best, rows=rows, best_rows=best_rows)
            energy = candidate['mean_total_energy']
            append_json(scene.log_dir()/"episodes.jsonl", dict(episode=episode,
                        instance_seed=scene.training_seed(episode), validation_energy=energy,
                        validation_selection=candidate, selected_as_best=selected,
                        best_episode=best['episode'], **result))
            append_json(scene.log_dir()/"validation_selection.jsonl", dict(algorithm=ALGORITHM,
                model_version=MODEL_VERSION, checkpoint_selection=selection_metadata(), episode=episode,
                instance_seed=scene.training_seed(episode), validation_rows=rows,
                validation_selection=candidate, best_validation_selection=best,
                selected_as_best=selected, selected_episode=best['episode']))
            write_json(scene.log_dir()/f"candidates_episode{episode}.json", apf_report(env))
            print(f"[{DISPLAY_NAME}] episode={episode+1}/{scene.episodes} energy={result['total_energy']:.6f} "
                  f"validation_energy={energy:.6f} SR={candidate['workflow_success_rate']:.4f} "
                  f"feasible={candidate['feasible']} selected={selected} best_episode={best['episode']}", flush=True)
        save_agents(agents, scene.checkpoint_dir(), "final", episode=episode,
                    candidate=candidate, best=best, rows=rows, best_rows=best_rows)
        write_json(scene.log_dir()/"training_summary.json", dict(algorithm=ALGORITHM,
            model_version=MODEL_VERSION, episodes=scene.episodes, checkpoint_selection=selection_metadata(),
            protocol=scene.selection_protocol(), selected_episode=best['episode'],
            best_validation_selection=best, final_validation_selection=candidate,
            feasible_model_found=best['feasible'], resume_supported=False,
            best_validation_energy=best['mean_total_energy'], peak_cuda_allocated_bytes=(
                torch.cuda.max_memory_allocated(agents["HA"].device) if agents["HA"].device.type == "cuda" else None),
            parameter_counts={r: sum(p.numel() for p in a.online.parameters()) for r,a in agents.items()}))
    finally:
        recorder.close()
    print(f"[{DISPLAY_NAME} train completed] models={scene.checkpoint_dir().resolve()} logs={scene.log_dir().resolve()}")
    return agents


def evaluate(scene):
    cfg = scene.fixed_config
    csv_path, directory = scene.evaluation_csv(), scene.evaluation_details_dir()
    if csv_path.exists() or directory.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifacts: {csv_path}")
    validate_inputs(scene)
    set_seed(scene.model_seed)
    agents = load_agents(scene)
    # Reserve only this run; other scenarios may concurrently share the CSV parent.
    directory.mkdir(parents=True, exist_ok=False)
    values = []
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        write_json(directory/"algorithm_variant.json", cfg.variant_metadata())
        write_json(directory/"implementation.json", cfg.implementation_metadata())
        write_run_metadata(directory, scene)
        writer = None
        for seed in scene.seeds:
            set_seed(scene.model_seed)
            env = build_env(scene, seed)
            env.reset()
            recorder = EvaluationRecorder(scene, cfg.algorithm, env, agents,
                directory=directory/"measurements", checkpoint_directory=str(scene.checkpoint_dir()))
            try:
                result = run_episode(env, agents)
            finally:
                recorder.finish()
            append_json(directory/"stage_timings.jsonl", dict(seed=seed, scope="evaluation", environment=env.profiler.report()))
            write_json(directory/f"candidates_seed{seed}.json", apf_report(env))
            append_json(directory/"evaluation.jsonl", dict(seed=seed, **result))
            row = dict(algorithm=cfg.algorithm, model_version=cfg.model_version, ordering_rule=cfg.rule,
                host_apf_enabled=False, vm_apf_enabled=False, scene=scene.target_tag,
                task=scene.task, resource=scene.resource, deadline=scene.deadline,
                source_scene=scene.source_tag, train_task=scene.train_task,
                train_resource=scene.train_resource or scene.resource,
                train_deadline=scene.train_deadline or scene.deadline,
                model_seed=scene.model_seed, seed=seed,
                selected_episode=agents["HA"].loaded_selection["selected_episode"], **episode_metrics(env))
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            values.append(result)
    summary = {key: dict(mean=float(np.mean([v[key] for v in values])),
                        std=float(np.std([v[key] for v in values], ddof=1)) if len(values)>1 else 0.)
               for key in ("total_energy", "wf_success_rate", "wf_avg_tardiness", "makespan", "wf_avg_response_time")}
    write_json(directory/"summary.json", dict(algorithm=cfg.algorithm, model_version=cfg.model_version,
        ta_policy=cfg.ta_policy_metadata(), scene=scene.target_tag, source_scene=scene.source_tag,
        csv_file=str(csv_path), checkpoint_selection=selection_metadata(),
        selected_episode=agents["HA"].loaded_selection["selected_episode"],
        best_validation_selection=agents["HA"].loaded_selection["best_validation_selection"],
        seeds=list(scene.seeds), metrics=summary))
    print(f"[{cfg.display_name} eval completed] csv={csv_path.resolve()} details={directory.resolve()}")
    return summary
