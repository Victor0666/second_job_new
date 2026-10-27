"""Shared lifecycle for two isolated synchronization identities and output trees."""
import csv
import json
from copy import deepcopy
from pathlib import Path
import uuid
import numpy as np
import torch
from common.f11 import F11Logger, protect_training_outputs
from common.f18_state_normalization import normalization_diagnostics
from common.f22_instances import write_run_metadata, episode_identity
from common.experiment_measurements import (
    TrainingRecorder, EvaluationRecorder, episode_metrics, collect_validation,
)
from algorithms.mahdrl_oe_ge_apf_score_fast.runtime import (
    build_env_kwargs, validate_inputs, set_seed, write_json, append_json,
)
from algorithms.mahdrl_oe_ge_apf_score_fast.selection import (
    selection_metadata, validation_selection, selection_key, validate_checkpoint_selection,
)
from environments.mahdrl_oe_ge_apf_score_fast_sync import CloudWorkflowEnv_VMAgents, MANAGER_ACTION_TABLE
from .agent import TAAgent, SetD3QNAgent
from .config import TA_HIDDEN, learning_kwargs, get_config
from .rollout import run_episode
from .metrics import training_stability


FILES = {"TA": "manager", "HA": "host", "VA": "vm"}


def build_env(scene, seed):
    env = CloudWorkflowEnv_VMAgents(**build_env_kwargs(scene, seed), sync_mode=scene.sync_mode,
        profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    env.energy_reward_scale = .001
    env.task_baseline_norm = 300.
    env.energy_norm_per_mi_ref = 20.
    env.alpha_delay_host = env.alpha_delay_vm = .75
    return env


def build_agents(scene):
    device = torch.device(scene.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    options = dict(diagnostics_every=scene.diagnostics_every,
        profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    agents = {
        "TA": TAAgent(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN, **options, **learning_kwargs("TA", device)),
        "HA": SetD3QNAgent(23, 10, **options, **learning_kwargs("HA", device)),
        "VA": SetD3QNAgent(30, 7, **options, **learning_kwargs("VA", device)),
    }
    return {role: agent.bind(scene, role) for role, agent in agents.items()}


def save_agents(agents, directory, kind, *, episode, candidate, best, rows, best_rows):
    if set(agents) != set(FILES):
        raise ValueError("Sync checkpoints require TA, HA and VA")
    cfg = agents["TA"].sync_config
    if any(agent.sync_config != cfg for agent in agents.values()):
        raise ValueError("Cannot save mixed synchronization modes")
    records = {role: dict(deepcopy(agent.selection_expectation), checkpoint_kind=kind,
        episode=episode, completed_episodes=episode+1, selected_episode=best["episode"],
        validation_selection=deepcopy(candidate), best_validation_selection=deepcopy(best),
        validation_rows=deepcopy(rows), best_validation_rows=deepcopy(best_rows))
        for role, agent in agents.items()}
    for role, agent in agents.items():
        validate_checkpoint_selection(records[role], agent.selection_expectation)
    write_json(Path(directory)/"synchronization.json", cfg.synchronization_metadata())
    group = uuid.uuid4().hex
    for role, agent in agents.items():
        agent.checkpoint_group = group
        agent.normalization_metadata[cfg.checkpoint_key] = records[role]
        name = f"best_{FILES[role]}.pth" if kind == "best" else f"{FILES[role]}_{kind}.pth"
        agent.save(str(Path(directory)/name))


def load_agents(scene):
    cfg = scene.sync_config
    manifest = scene.checkpoint_dir()/"synchronization.json"
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != cfg.synchronization_metadata():
        raise ValueError("Missing or mismatched synchronization manifest")
    agents = build_agents(scene)
    for role, agent in agents.items():
        agent.load(str(scene.checkpoint_dir()/f"best_{FILES[role]}.pth"), require_learning_model=True)
    if len({a.loaded_group for a in agents.values()}) != 1:
        raise ValueError("TA/HA/VA checkpoints are from different saved groups")
    records = [{key: value for key, value in a.loaded_selection.items() if key != "learner_role"}
        for a in agents.values()]
    if any(record["checkpoint_kind"] != "best" for record in records):
        raise ValueError("Evaluation requires best checkpoints")
    if any(record != records[0] for record in records[1:]):
        raise ValueError("TA/HA/VA checkpoints have different selection evidence or episodes")
    return agents


def apf_report(env):
    windows = {}
    for record in env._records:
        h = record.server_id
        window = windows.setdefault(h, [float(record.start_time), float(record.end_time)])
        window[0] = min(window[0], float(record.start_time))
        window[1] = max(window[1], float(record.end_time))
    return dict(variant=env.sync_config.variant_metadata(), decisions=env.apf_decisions,
        host_windows={str(h): dict(start=w[0], end=w[1], duration=w[1]-w[0]) for h, w in windows.items()})


def train(scene):
    cfg = scene.sync_config
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
    curve_rows = []
    if agents["TA"].device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agents["TA"].device)
    try:
        for episode in range(scene.episodes):
            if episode:
                env.reset(seed=scene.training_seed(episode))
            result = run_episode(env, agents, training=True,
                phase_log=lambda row: append_json(scene.log_dir()/"phases.jsonl", dict(episode=episode, **row)),
                assignment_log=lambda row: append_json(scene.log_dir()/"assignments.jsonl", dict(episode=episode, **row)))
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
            curve = dict(episode=episode, instance_seed=scene.training_seed(episode),
                validation_success_rate=candidate['workflow_success_rate'], validation_energy=energy,
                training_success_rate=result['wf_success_rate'], training_energy=result['total_energy'],
                ta_decisions=result['decision_counts']['TA'], assignments=result['policy_assignments'],
                sorting_calls=result['sorting']['nonempty_calls'], sorting_seconds=result['sorting']['total_seconds'],
                **{role+'_optimizer_steps': result['learning_schedule']['optimizer_steps'][role] for role in FILES})
            with (scene.log_dir()/"learning_curves.csv").open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(curve))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(curve)
            curve_rows.append(curve)
            write_json(scene.log_dir()/f"apf_episode{episode}.json", apf_report(env))
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
                torch.cuda.max_memory_allocated(agents["TA"].device) if agents["TA"].device.type == "cuda" else None),
            optimizer_steps={r: a._updates for r, a in agents.items()},
            synchronization=cfg.synchronization_metadata(),
            training_stability=training_stability(curve_rows),
            parameter_counts={r: sum(p.numel() for p in a.online.parameters()) for r,a in agents.items()}))
    finally:
        recorder.close()
    print(f"[{DISPLAY_NAME} train completed] models={scene.checkpoint_dir().resolve()} logs={scene.log_dir().resolve()}")
    return agents


def evaluate(scene):
    cfg = scene.sync_config
    ALGORITHM, DISPLAY_NAME, MODEL_VERSION = cfg.algorithm, cfg.display_name, cfg.model_version
    directory = scene.evaluation_dir()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation artifacts: {directory}")
    validate_inputs(scene)
    set_seed(scene.model_seed)
    agents = load_agents(scene)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"algorithm_variant.json", cfg.variant_metadata())
    write_json(directory/"implementation.json", cfg.implementation_metadata())
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
        append_json(directory/"stage_timings.jsonl", dict(seed=seed, scope="evaluation", environment=env.profiler.report()))
        write_json(directory/f"apf_seed{seed}.json", apf_report(env))
        append_json(directory/"evaluation.jsonl", dict(seed=seed, **result))
        csv_path = directory/f"evaluation_mahdrl_as_fast_sync_{scene.sync_mode}.csv"
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
    sorting = {key: dict(mean=float(np.mean([v["sorting"][key] for v in values])),
                        std=float(np.std([v["sorting"][key] for v in values], ddof=1)) if len(values)>1 else 0.)
               for key in ("nonempty_calls", "nontrivial_calls", "total_seconds",
                           "calls_per_1000_assignments", "seconds_per_assignment")}
    write_json(directory/"summary.json", dict(algorithm=ALGORITHM, model_version=MODEL_VERSION,
        checkpoint_selection=selection_metadata(), selected_episode=agents['TA'].loaded_selection['selected_episode'],
        best_validation_selection=agents['TA'].loaded_selection['best_validation_selection'],
        seeds=list(scene.seeds), metrics=summary, sorting=sorting,
        synchronization=cfg.synchronization_metadata()))
    print(f"[{DISPLAY_NAME} eval completed] results={directory.resolve()}")
    return summary
