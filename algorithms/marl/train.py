"""marl: train. Select --task, --resource and --deadline; see --help."""

import os
import sys
import json
from contextlib import ExitStack
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from algorithms.marl.scenario import MARLScenario as Scenario, parse_scenario
from common.f18_state_normalization import bind_normalization, normalization_diagnostics
from common.f11 import F11Logger, protect_training_outputs, valid_decision

SCENARIO = Scenario(mode="train")
from common.metrics_logger import CSVLogger
from common.experiment_measurements import TrainingRecorder, collect_validation, episode_metrics
from algorithms.marl.runtime import isolated_evaluation, selection_record
from algorithms.marl.selection import selection_metadata, selection_key, validation_selection
from common.workflow_opt import energy_from_records
from algorithms.marl.q_agents import MODEL_VERSION, QAgent, QConfig, IndependentLocalAgents, Decision, QDecisionTrace, learner_map
from algorithms.marl.identity import DISPLAY_NAME
from environments.marl import CloudWorkflowEnv_MARL


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_lateness_stats(info2: dict, ep_wf_late: dict):
    late = float(info2.get("lateness", 0.0))
    wf_id = info2.get("wf_id", None)
    if wf_id is None:
        return (late, None)
    wf_id = int(wf_id)
    ep_wf_late[wf_id] = ep_wf_late.get(wf_id, 0.0) + late
    return (late, wf_id)


def _finalize_lateness_stats(ep_total_late: float, ep_task_cnt: int, ep_wf_late: dict):
    ep_avg_late = float(ep_total_late) / float(max(1, int(ep_task_cnt)))
    wf_avg_late = float(np.mean(list(ep_wf_late.values()))) if len(ep_wf_late) > 0 else 0.0
    return (float(ep_total_late), float(ep_avg_late), float(wf_avg_late))


def _load_cache_seeds(deadline_cache_path: str):
    with open(deadline_cache_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    seeds = sorted({int(rec["seed"]) for rec in obj.get("data", [])})
    if len(seeds) == 0:
        raise ValueError("deadline cache has empty data")
    return seeds


def evaluate_marl(
    env_kwargs,
    global_agent: QAgent,
    local_agent: IndependentLocalAgents,
    eval_seeds,
    episode_id_base: int = 0,
):
    rows = []
    eval_energy_list = []
    eval_ret_g_list = []
    eval_ret_l_list = []
    eval_total_late_list = []
    eval_avg_late_list = []
    eval_wf_avg_late_list = []
    for k, s in enumerate(eval_seeds):
        env = CloudWorkflowEnv_MARL(**env_kwargs)
        obs, _info = env.reset(
            seed=int(s),
            options={
                "episode_id": int(episode_id_base + k),
                "tag": "eval",
                "dump_deadlines": True,
                "print_deadlines": True,
            },
        )
        done = False
        ret_g = 0.0
        ret_l = 0.0
        ep_total_late = 0.0
        ep_task_cnt = 0
        ep_wf_late = {}
        while not done:
            gobs = obs["global_obs"]
            gmask = obs["global_mask"]
            a_host = global_agent.select_action(gobs, gmask, deterministic=True)
            host_id = env.host_ids[int(a_host)]
            host_index = int(a_host)
            lobs, lmask, vm_map = env.get_local_state(host_id)
            if vm_map == []:
                obs2, r_g, r_l, done, info2 = env.step(int(a_host), 0)
            else:
                a_vm = local_agent.select_action(
                    lobs, lmask, host_index=host_index, deterministic=True
                )
                obs2, r_g, r_l, done, info2 = env.step(int(a_host), int(a_vm))
            ret_g += float(r_g)
            ret_l += float(r_l)
            late, _wf = _update_lateness_stats(info2, ep_wf_late)
            ep_total_late += late
            ep_task_cnt += 1
            obs = obs2
        ep_total_late, ep_avg_late, ep_wf_avg_late = _finalize_lateness_stats(
            ep_total_late, ep_task_cnt, ep_wf_late
        )
        recs = env.records if hasattr(env, "records") else env._records
        E_re = float(energy_from_records(
            recs, env.hosts,
            start_time=float(env.arrival_times[0]) if len(env.arrival_times) else 0.0,
            end_time=float(env.current_time),
        ))
        collect_validation(env)
        metrics = episode_metrics(env)
        if metrics['wf_count'] != env.workflows_per_episode:
            raise RuntimeError('MARL validation ended before all workflows arrived')
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError('Nonfinite MARL validation metrics')
        rows.append(dict(seed=int(s), **metrics))
        E_acc = float(env.total_energy)
        diff = abs(E_re - E_acc)
        print(f"[EnergyCheck] E_acc={E_acc:.6f}  E_re={E_re:.6f}  diff={diff:.6f}")
        eval_energy_list.append(float(env.total_energy))
        eval_ret_g_list.append(ret_g)
        eval_ret_l_list.append(ret_l)
        eval_total_late_list.append(ep_total_late)
        eval_avg_late_list.append(ep_avg_late)
        eval_wf_avg_late_list.append(ep_wf_avg_late)
        env.close()
    return dict(rows=rows, ret_g=float(np.mean(eval_ret_g_list)),
        ret_l=float(np.mean(eval_ret_l_list)), energy=float(np.mean(eval_energy_list)),
        total_late=float(np.mean(eval_total_late_list)), avg_late=float(np.mean(eval_avg_late_list)),
        wf_avg_late=float(np.mean(eval_wf_avg_late_list)))


def train():
    with ExitStack() as resources:
        return _train(resources)


def _train(resources):
    print(f"[{DISPLAY_NAME}] algorithm={MODEL_VERSION}")
    random_seed = SCENARIO.first_training_seed()
    set_seed(SCENARIO.model_seed)
    dax_dir = os.path.join(ROOT_DIR, "data/dax")
    dax_list = SCENARIO.dax_paths(dax_dir)
    deadline_cache_path = str(SCENARIO.cache_path)
    cache_seeds = _load_cache_seeds(deadline_cache_path)
    required_seeds = set(range(random_seed, random_seed + SCENARIO.episodes))
    missing_seeds = sorted(required_seeds - set(cache_seeds))
    if missing_seeds:
        raise KeyError(f"Training seeds missing from deadline cache: {missing_seeds}")
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=1000000000.0,
        arrival_lambda=0.03,
        random_seed=random_seed,
        normalize=True,
        num_hosts=SCENARIO.num_hosts,
        vms_per_host=SCENARIO.vms_per_host,
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        workflows_per_episode=50,
        energy_reward_scale=0.001,
        task_baseline_norm=300.0,
        alpha_delay=0.75,
        max_ready_tasks=64,
        deadline_mode="cache_fcfs",
        deadline_cache_path=deadline_cache_path,
        alpha_deadline=1.5,
        alpha_deadline_choices=(2.0, 3.0),
        alpha_deadline_probs=SCENARIO.deadline_probs,
        strict_cache=True,
        dump_deadlines=True,
        print_deadlines=True,
        deadlines_out_dir=os.path.join(str(SCENARIO.log_dir("marl")), "deadlines"),
        deadlines_file_prefix=MODEL_VERSION,
    )
    env_kwargs.update(SCENARIO.normalization_kwargs())
    protect_training_outputs(SCENARIO.checkpoint_dir("marl"), SCENARIO.log_dir("marl"))
    env = CloudWorkflowEnv_MARL(**env_kwargs)
    resources.callback(env.close)
    from common.f22_instances import write_run_metadata, episode_identity
    write_run_metadata(SCENARIO.checkpoint_dir("marl"), SCENARIO)
    write_run_metadata(SCENARIO.log_dir("marl"), SCENARIO)
    obs0, _info0 = env.reset(
        seed=random_seed,
        options={"episode_id": 0, "tag": "probe", "dump_deadlines": True, "print_deadlines": True},
    )
    global_obs_dim = int(obs0["global_obs"].shape[0])
    global_act_dim = int(obs0["global_mask"].shape[0])
    valid_hosts = np.flatnonzero(obs0["global_mask"] > 0.5)
    hid = env.host_ids[int(valid_hosts[0])] if valid_hosts.size > 0 else env.host_ids[0]
    lobs0, lmask0, _ = env.get_local_state(hid)
    local_obs_dim = int(lobs0.shape[0])
    local_act_dim = int(lmask0.shape[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] global_obs_dim={global_obs_dim} global_act_dim={global_act_dim}")
    print(
        f"[env] local_obs_dim={local_obs_dim} local_act_dim={local_act_dim} max_vms_per_host={env.max_vms_per_host} num_hosts={env.num_hosts}"
    )
    print(f"[device] {device}")
    hid_host = (1024, 1024, 512, 512, 256)
    cfg_global = QConfig(
        lr=0.0003,
        gamma=0.95,
        ent_coef=0.0,
        grad_clip=0.0,
        device=device,
        hidden_dims=hid_host,
    )
    hid_vm = (1024, 1024, 512, 512, 256)
    cfg_local = QConfig(
        lr=0.0003,
        gamma=0.95,
        ent_coef=0.0,
        grad_clip=0.0,
        device=device,
        hidden_dims=hid_vm,
    )
    global_agent = QAgent(global_obs_dim, global_act_dim, cfg_global, peer_act_dim=local_act_dim)
    bind_normalization(global_agent, SCENARIO)
    local_agent = IndependentLocalAgents(
        local_obs_dim, local_act_dim, num_hosts=SCENARIO.num_hosts, cfg=cfg_local
    )
    bind_normalization(local_agent, SCENARIO)
    learners = learner_map(global_agent, local_agent)
    for agent in local_agent.agents:
        bind_normalization(agent, SCENARIO)

    save_dir = str(SCENARIO.checkpoint_dir("marl"))
    log_path = os.path.join(
        str(SCENARIO.log_dir("marl")), f"train_metrics_{MODEL_VERSION}.csv"
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "algorithm",
            "step",
            "episode",
            "type",
            "episode_seed",
            "env_time",
            "episode_energy",
            "completed_wf",
            "wf_target",
            "r_global",
            "r_local",
            "eval_ret_global",
            "eval_ret_local",
            "eval_energy",
            "eval_workflow_success_rate", "eval_feasible", "selected_as_best", "best_episode",
            "host_action",
            "vm_action",
            "lateness",
            "delay_norm",
            "task_deadline",
            "wf_deadline",
            "r_energy_total",
            "r_energy_host",
            "host_index",
            "episode_total_lateness",
            "episode_avg_lateness",
            "episode_workflow_avg_lateness",
            "eval_total_lateness",
            "eval_avg_lateness",
            "eval_workflow_avg_lateness",
        ],
    )
    resources.callback(logger.close)
    best = None
    best_global_path = os.path.join(save_dir, "best_global.pth")
    best_local_path = os.path.join(save_dir, "best_local_actor_critics.pth")
    max_episodes = SCENARIO.episodes
    eval_seeds = SCENARIO.validation_seeds()
    eval_interval_episode = 1
    global_step = 0
    episode_idx = 0
    f11_log = F11Logger(os.path.dirname(log_path))
    recorder = TrainingRecorder(os.path.dirname(log_path), SCENARIO, MODEL_VERSION, env, learners)
    resources.callback(recorder.close)
    save_locals = recorder.checkpoint(local_agent.save)
    while episode_idx < max_episodes:
        train_seed = SCENARIO.training_seed(episode_idx)
        obs, _info = env.reset(
            seed=train_seed,
            options={
                "episode_id": int(episode_idx),
                "tag": "train",
                "dump_deadlines": True,
                "print_deadlines": True,
            },
        )
        done = bool(env.done_flag)
        global_trace = QDecisionTrace(global_agent)
        local_traces = [QDecisionTrace(agent) for agent in local_agent.agents]
        ep_energy0 = float(env.total_energy)
        ep_ret_g = 0.0
        ep_ret_l = 0.0
        ep_total_late = 0.0
        ep_task_cnt = 0
        ep_wf_late = {}
        while not done:
            gobs = np.array(obs["global_obs"], copy=True)
            gmask = np.array(obs["global_mask"], copy=True)
            valid_decision(gobs, gmask)
            a_host = global_agent.select_action(gobs, gmask, deterministic=False)
            host_id = env.host_ids[int(a_host)]
            host_index = int(a_host)
            lobs, lmask, vm_map = env.get_local_state(host_id)
            if not vm_map or not np.any(lmask > 0.5):
                raise RuntimeError("Global selected a legal Host without a real Local VM decision")
            lobs, lmask = np.array(lobs, copy=True), np.array(lmask, copy=True)
            local_trace = local_traces[host_index]
            a_vm = local_agent.select_action(lobs, lmask, host_index=host_index, deterministic=False)
            global_decision = Decision(gobs, gmask, a_host, peer_action=a_vm)
            local_decision = Decision(lobs, lmask, a_vm)
            # Both targets read pre-update Q; the cached actions are executed once below.
            next_g = global_agent.successor_q(global_decision) if global_trace.pending else None
            next_l = local_trace.agent.successor_q(local_decision) if local_trace.pending else None
            global_trace.advance(global_decision, next_g)
            local_trace.advance(local_decision, next_l)
            obs2, r_g, r_l, done, info2 = env.step(int(a_host), int(a_vm))
            late, _wf = _update_lateness_stats(info2, ep_wf_late)
            ep_total_late += late
            ep_task_cnt += 1
            global_trace.record(global_decision, r_g)
            local_trace.record(local_decision, r_l)
            if done:
                for trace in (global_trace, *local_traces):
                    trace.finish(truncated=bool(info2.get("truncated", False)))
            ep_ret_g += float(r_g)
            ep_ret_l += float(r_l)
            logger.log(
                algorithm=MODEL_VERSION,
                step=global_step,
                episode=episode_idx,
                type="step",
                episode_seed=int(train_seed),
                env_time=info2.get("current_time", 0.0),
                episode_energy="",
                completed_wf=info2.get("completed_wf", 0),
                wf_target=info2.get("wf_target", ""),
                r_global=float(r_g),
                r_local=float(r_l),
                eval_ret_global="",
                eval_ret_local="",
                eval_energy="",
                host_action=int(a_host),
                vm_action=int(a_vm),
                lateness=info2.get("lateness", 0.0),
                delay_norm=info2.get("delay_norm", 0.0),
                task_deadline=info2.get("task_deadline", 0.0),
                wf_deadline=info2.get("wf_deadline", 0.0),
                r_energy_total=info2.get("r_energy_total", 0.0),
                r_energy_host=info2.get("r_energy_host", 0.0),
                host_index=int(host_index),
                episode_total_lateness="",
                episode_avg_lateness="",
                episode_workflow_avg_lateness="",
                eval_total_lateness="",
                eval_avg_lateness="",
                eval_workflow_avg_lateness="",
            )
            obs = obs2
            global_step += 1
        ep_energy = float(env.total_energy - ep_energy0)
        for trace in (global_trace, *local_traces):
            trace.finish()
        f11_log.log(episode_idx, learners, algorithm_version=MODEL_VERSION,
                    normalization=normalization_diagnostics(env), instance=episode_identity(env, SCENARIO),
                    pending=sum(t.pending is not None for t in (global_trace, *local_traces)),
                    global_reward_balance=global_trace.recorded_reward - global_trace.emitted_reward,
                    local_reward_balance=sum(t.recorded_reward - t.emitted_reward for t in local_traces))
        ep_total_late, ep_avg_late, ep_wf_avg_late = _finalize_lateness_stats(
            ep_total_late, ep_task_cnt, ep_wf_late
        )
        logger.log(
            algorithm=MODEL_VERSION,
            step=global_step,
            episode=episode_idx,
            type="episode",
            episode_seed=int(train_seed),
            env_time=getattr(env, "current_time", 0.0),
            episode_energy=ep_energy,
            completed_wf=getattr(env, "completed_workflows", 0),
            wf_target=getattr(env, "workflows_per_episode", ""),
            r_global=ep_ret_g,
            r_local=ep_ret_l,
            eval_ret_global="",
            eval_ret_local="",
            eval_energy="",
            host_action="",
            vm_action="",
            lateness="",
            delay_norm="",
            task_deadline="",
            wf_deadline="",
            r_energy_total="",
            r_energy_host="",
            host_index="",
            episode_total_lateness=ep_total_late,
            episode_avg_lateness=ep_avg_late,
            episode_workflow_avg_lateness=ep_wf_avg_late,
            eval_total_lateness="",
            eval_avg_lateness="",
            eval_workflow_avg_lateness="",
        )
        recs = env.records if hasattr(env, "records") else env._records
        E_re = float(energy_from_records(
            recs, env.hosts,
            start_time=float(env.arrival_times[0]) if len(env.arrival_times) else 0.0,
            end_time=float(env.current_time),
        ))
        E_acc = float(env.total_energy)
        diff = abs(E_re - E_acc)
        print(f"[EnergyCheck] E_acc={E_acc:.6f}  E_re={E_re:.6f}  diff={diff:.6f}")
        print(
            f"[episode={episode_idx}] seed={train_seed} steps={global_step} energy={ep_energy:.3f}J ret_g={ep_ret_g:.4f} ret_l={ep_ret_l:.4f} late_sum={ep_total_late:.3f} late_avg={ep_avg_late:.6f} wf_late_avg={ep_wf_avg_late:.6f} wf={getattr(env, 'completed_workflows', 0)}/{getattr(env, 'workflows_per_episode', '∞')}"
        )
        if episode_idx % eval_interval_episode == 0:
            with isolated_evaluation(global_agent, local_agent):
                validation = recorder.validation(evaluate_marl, env_kwargs, global_agent, local_agent,
                    eval_seeds=eval_seeds, episode_id_base=100000 + episode_idx * 10)
            candidate = validation_selection(validation['rows'])
            selected = best is None or selection_key(candidate) < selection_key(best)
            if selected:
                best = dict(candidate, episode=episode_idx, completed_episodes=episode_idx+1)
                record = selection_record(SCENARIO, episode_idx, candidate, best, validation['rows'], 'best')
                global_agent.training_selection = local_agent.training_selection = record
                global_agent.save(best_global_path)
                save_locals(best_local_path)
            with open(os.path.join(os.path.dirname(log_path), 'validation_selection.jsonl'), 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(episode=episode_idx, validation=validation['rows'],
                    validation_selection=candidate, best_validation_selection=best,
                    selected_as_best=selected), allow_nan=False) + '\n')
            eval_ret_g, eval_ret_l, eval_energy = validation['ret_g'], validation['ret_l'], validation['energy']
            eval_total_late, eval_avg_late, eval_wf_avg_late = validation['total_late'], validation['avg_late'], validation['wf_avg_late']
            logger.log(
                algorithm=MODEL_VERSION,
                step=global_step,
                episode=episode_idx,
                type="eval",
                episode_seed="",
                env_time="",
                episode_energy="",
                completed_wf="",
                wf_target="",
                r_global="",
                r_local="",
                eval_ret_global=eval_ret_g,
                eval_ret_local=eval_ret_l,
                eval_energy=eval_energy,
                eval_workflow_success_rate=candidate['workflow_success_rate'],
                eval_feasible=candidate['feasible'], selected_as_best=selected, best_episode=best['episode'],
                host_action="",
                vm_action="",
                lateness="",
                delay_norm="",
                task_deadline="",
                wf_deadline="",
                r_energy_total="",
                r_energy_host="",
                host_index="",
                episode_total_lateness="",
                episode_avg_lateness="",
                episode_workflow_avg_lateness="",
                eval_total_lateness=eval_total_late,
                eval_avg_lateness=eval_avg_late,
                eval_workflow_avg_lateness=eval_wf_avg_late,
            )
            print(
                f"  [eval] seeds={eval_seeds} ret_g={eval_ret_g:.4f} ret_l={eval_ret_l:.4f} eval_energy={eval_energy:.3f}J eval_late_sum={eval_total_late:.3f} eval_late_avg={eval_avg_late:.6f} eval_wf_late_avg={eval_wf_avg_late:.6f}"
            )
            print(f"  [selection] SR={candidate['workflow_success_rate']:.4f} feasible={candidate['feasible']} selected={selected}")
        episode_idx += 1
    global_final = os.path.join(save_dir, "global_final.pth")
    local_final = os.path.join(save_dir, "local_final.pth")
    record = selection_record(SCENARIO, episode_idx-1, candidate, best, validation['rows'], 'final')
    global_agent.training_selection = local_agent.training_selection = record
    global_agent.save(global_final)
    save_locals(local_final)
    with open(os.path.join(os.path.dirname(log_path), 'training_summary.json'), 'w', encoding='utf-8') as stream:
        json.dump(dict(algorithm=MODEL_VERSION, checkpoint_selection=selection_metadata(),
            protocol=SCENARIO.selection_protocol(), completed_episodes=episode_idx,
            best_validation_selection=best, feasible_model_found=best['feasible'],
            best_global=best_global_path, best_local=best_local_path), stream, indent=2, allow_nan=False)
    print("训练结束：")
    print(f"  total_episodes={episode_idx}")
    print(f"  total_steps   ={global_step}")
    print(f"  global_final  ={global_final}")
    print(f"  local_final   ={local_final}")
    print(f"  best_global   ={best_global_path}")
    print(f"  best_local    ={best_local_path}")
    print(f"  log_path      ={log_path}")



if __name__ == "__main__":
    SCENARIO = parse_scenario("train", "marl")
    train()
