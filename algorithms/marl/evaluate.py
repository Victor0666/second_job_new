"""marl: eval. Select --task, --resource and --deadline; see --help."""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Any, Dict, List
import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
from common.scenarios import Scenario, parse_scenario

SCENARIO = Scenario(mode="eval")


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _parse_seeds(s: str) -> List[int]:
    s = str(s).strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = (int(a.strip()), int(b.strip()))
        if b < a:
            a, b = (b, a)
        return list(range(a, b + 1))
    return [int(s)]


def _load_cache_seeds(deadline_cache_path: str) -> List[int]:
    with open(deadline_cache_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    seeds = sorted({int(rec["seed"]) for rec in obj.get("data", [])})
    if len(seeds) == 0:
        raise ValueError("deadline cache has empty data")
    return seeds


def _get_metric(
    metrics: Dict[str, Any], candidate_keys: List[str], default: float = float("nan")
) -> float:
    """
    从 metrics 中按候选 key 顺序取值，做 float 转换。
    兼容不同 env.get_episode_metrics() 的字段命名。
    """
    for k in candidate_keys:
        if k in metrics:
            try:
                v = float(metrics[k])
                return v
            except Exception:
                pass
    return float(default)


def _mean(xs: List[float]) -> float:
    xs2 = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs2)) if xs2 else float("nan")


from common.metrics_logger import CSVLogger
from common.workflow_opt import energy_from_records
from algorithms.marl.agents import A2CAgent, A2CConfig, LocalSharedActorIndependentCriticA2C
from environments.marl import CloudWorkflowEnv_MARL


def build_env_kwargs(deadline_cache_path: str) -> Dict[str, Any]:
    dax_dir = os.path.join(PROJECT_DIR, "data/dax")
    dax_list = SCENARIO.dax_paths(dax_dir)
    return dict(
        dax_paths=dax_list,
        horizon=1000000000.0,
        arrival_lambda=0.03,
        random_seed=0,
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
        deadlines_out_dir=os.path.join(PROJECT_DIR, "outputs", "run", "marl", "deadlines"),
        deadlines_file_prefix="marl",
    )


def build_agents_and_probe_dims(
    env_kwargs: Dict[str, Any], probe_seed: int, device: str, ckpt_global: str, ckpt_local: str
):
    env = CloudWorkflowEnv_MARL(**env_kwargs)
    obs0, _ = env.reset(
        seed=int(probe_seed),
        options={
            "episode_id": 0,
            "tag": "probe",
            "dump_deadlines": False,
            "print_deadlines": False,
        },
    )
    global_obs_dim = int(obs0["global_obs"].shape[0])
    global_act_dim = int(obs0["global_mask"].shape[0])
    valid_hosts = np.flatnonzero(obs0["global_mask"] > 0.5)
    hid = env.host_ids[int(valid_hosts[0])] if valid_hosts.size > 0 else env.host_ids[0]
    lobs0, lmask0, _ = env.get_local_state(hid)
    local_obs_dim = int(lobs0.shape[0])
    local_act_dim = int(lmask0.shape[0])
    num_hosts = SCENARIO.num_hosts
    if hasattr(env, "close"):
        env.close()
    del env
    hid_dims = (1024, 1024, 512, 512, 256)
    cfg_global = A2CConfig(
        lr=0.0003,
        gamma=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        grad_clip=10.0,
        device=device,
        hidden_dims=hid_dims,
    )
    cfg_local = A2CConfig(
        lr=0.0003,
        gamma=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        grad_clip=10.0,
        device=device,
        hidden_dims=hid_dims,
    )
    global_agent = A2CAgent(global_obs_dim, global_act_dim, cfg_global)
    local_agent = LocalSharedActorIndependentCriticA2C(
        local_obs_dim, local_act_dim, num_hosts=SCENARIO.num_hosts, cfg=cfg_local
    )
    print(f"[MARL load] global: {ckpt_global}")
    global_agent.load(ckpt_global, strict=True)
    print(f"[MARL load] local : {ckpt_local}")
    local_agent.load(ckpt_local, strict=True)
    dims = dict(
        global_obs_dim=global_obs_dim,
        global_act_dim=global_act_dim,
        local_obs_dim=local_obs_dim,
        local_act_dim=local_act_dim,
        num_hosts=SCENARIO.num_hosts,
    )
    return (global_agent, local_agent, dims)


def evaluate_one_seed(
    env_kwargs: Dict[str, Any],
    global_agent: A2CAgent,
    local_agent: LocalSharedActorIndependentCriticA2C,
    seed: int,
    episode_id: int,
):
    env = CloudWorkflowEnv_MARL(**env_kwargs)
    obs, info0 = env.reset(
        seed=int(seed),
        options={
            "episode_id": int(episode_id),
            "tag": "eval",
            "dump_deadlines": False,
            "print_deadlines": False,
        },
    )
    t_now = float(info0.get("current_time", getattr(env, "current_time", 0.0)))
    E0 = float(getattr(env, "total_energy", 0.0))
    arr = getattr(env, "arrival_times", [])
    t_first = float(arr[0]) if isinstance(arr, list) and len(arr) > 0 else float("nan")
    print(f"[ResetCheck] t_now={t_now} E0={E0} t_first_arr={t_first}")
    done = False
    ret_g = 0.0
    ret_l = 0.0
    steps = 0
    with torch.no_grad():
        while not done:
            gobs = obs["global_obs"]
            gmask = obs["global_mask"]
            a_host = global_agent.select_action(gobs, gmask, deterministic=True)
            host_id = env.host_ids[int(a_host)]
            host_index = int(a_host)
            lobs, lmask, vm_map = env.get_local_state(host_id)
            if vm_map == []:
                obs2, r_g, r_l, done, _info2 = env.step(int(a_host), 0)
            else:
                a_vm = local_agent.select_action(
                    lobs, lmask, host_index=host_index, deterministic=True
                )
                obs2, r_g, r_l, done, _info2 = env.step(int(a_host), int(a_vm))
            ret_g += float(r_g)
            ret_l += float(r_l)
            steps += 1
            obs = obs2
    m = env.get_episode_metrics()
    energy = _get_metric(m, ["total_energy_J", "total_energy"], getattr(env, "total_energy", 0.0))
    makespan = _get_metric(m, ["makespan"])
    task_avg_lateness = _get_metric(m, ["task_avg_lateness", "avg_lateness"])
    task_avg_tardiness = _get_metric(m, ["task_avg_tardiness", "avg_tardiness"])
    task_pct_tardy = _get_metric(m, ["task_pct_tardy", "pct_tardy"])
    if np.isfinite(task_pct_tardy):
        task_success_rate = 1.0 - task_pct_tardy
    else:
        task_success_rate = float("nan")
    wf_avg_lateness = _get_metric(
        m, ["wf_avg_lateness", "workflow_avg_lateness", "wf_lateness_mean"]
    )
    wf_avg_tardiness = _get_metric(
        m, ["wf_avg_tardiness", "workflow_avg_tardiness", "wf_tardiness_mean"]
    )
    wf_pct_tardy = _get_metric(m, ["wf_pct_tardy", "workflow_pct_tardy", "wf_tardy_ratio"])
    if np.isfinite(wf_pct_tardy):
        wf_success_rate = 1.0 - wf_pct_tardy
    else:
        wf_success_rate = float("nan")
    recs = env.records if hasattr(env, "records") else getattr(env, "_records", None)
    E_re = float(energy_from_records(
        recs, env.hosts,
        start_time=float(env.arrival_times[0]) if len(env.arrival_times) else 0.0,
        end_time=float(env.current_time),
    )) if recs is not None else float("nan")
    E_acc = float(getattr(env, "total_energy", energy))
    diff = float(E_acc - E_re) if np.isfinite(E_re) else float("nan")
    print(f"[EnergyCheck] E_acc={E_acc:.6f}  E_re={E_re:.6f}  diff={diff:.6f}")
    vm_r = float(ret_l)
    host_r = float(ret_g)
    mgr_r = 0.0
    print(
        f"[MARL0.75-MIX] vm={vm_r:.6f} host={host_r:.6f} mgr={mgr_r:.6f} energy={energy:.3f} J | task_avg_lateness={task_avg_lateness:.6f} task_avg_tardiness={task_avg_tardiness:.6f} task_success_rate={task_success_rate:.6f} task_pct_tardy={task_pct_tardy:.6f} | wf_avg_lateness={wf_avg_lateness:.6f} wf_avg_tardiness={wf_avg_tardiness:.6f} wf_success_rate={wf_success_rate:.6f} wf_pct_tardy={wf_pct_tardy:.6f} | makespan={makespan:.3f}"
    )
    if hasattr(env, "close"):
        env.close()
    del env
    return dict(
        steps=float(steps),
        vm=float(vm_r),
        host=float(host_r),
        mgr=float(mgr_r),
        energy=float(energy),
        task_avg_lateness=float(task_avg_lateness),
        task_avg_tardiness=float(task_avg_tardiness),
        task_success_rate=float(task_success_rate),
        task_pct_tardy=float(task_pct_tardy),
        wf_avg_lateness=float(wf_avg_lateness),
        wf_avg_tardiness=float(wf_avg_tardiness),
        wf_pct_tardy=float(wf_pct_tardy),
        wf_success_rate=float(wf_success_rate),
        makespan=float(makespan),
        energy_check_diff=float(diff),
    )


def main():
    default_cache = str(SCENARIO.cache_path)
    default_ckpt_dir = str(SCENARIO.checkpoint_dir("marl"))
    default_global = os.path.join(default_ckpt_dir, "best_global.pth")
    default_local = os.path.join(default_ckpt_dir, "best_local_actor_critics.pth")
    default_out = os.path.join(
        PROJECT_DIR,
        "outputs",
        "run",
        "marl_fcfs_mix_ddlAlpha",
        f"run_marl0p75_style_task{SCENARIO.task}_res{SCENARIO.resource}_ddlMix_{SCENARIO.deadline}.csv",
    )
    args = argparse.Namespace(
        deadline_cache=default_cache,
        ckpt_global=default_global,
        ckpt_local=default_local,
        seeds=SCENARIO.seed_spec,
        out=default_out,
        device="cuda",
    )
    for name, value in (SCENARIO.marl_options or {}).items():
        setattr(args, name, value)

    if args.deadline_cache is None or not os.path.exists(args.deadline_cache):
        raise FileNotFoundError(f"deadline cache not found: {args.deadline_cache}")
    ckpt_global = os.path.abspath(args.ckpt_global)
    ckpt_local = os.path.abspath(args.ckpt_local)
    if not os.path.exists(ckpt_global):
        raise FileNotFoundError(f"global ckpt not found: {ckpt_global}")
    if not os.path.exists(ckpt_local):
        raise FileNotFoundError(f"local ckpt not found: {ckpt_local}")
    seeds = _parse_seeds(args.seeds)
    if len(seeds) == 0:
        raise ValueError("seeds 为空")
    out_csv = os.path.abspath(args.out)
    _ensure_dir(os.path.dirname(out_csv))
    device = args.device.strip().lower()
    if device == "cuda" and (not torch.cuda.is_available()):
        device = "cpu"
    env_kwargs = build_env_kwargs(os.path.abspath(args.deadline_cache))
    cache_seeds = set(_load_cache_seeds(env_kwargs["deadline_cache_path"]))
    missing = [int(s) for s in seeds if int(s) not in cache_seeds]
    if missing:
        raise KeyError(f"strict_cache=True but these seeds not found in cache: {missing}")
    probe_seed = int(seeds[0])
    print(f"[MARL-MIX] PROJECT_DIR={PROJECT_DIR}")
    print(f"[MARL-MIX] deadline_cache={env_kwargs['deadline_cache_path']}")
    print(f"[MARL-MIX] ckpt_global={ckpt_global}")
    print(f"[MARL-MIX] ckpt_local ={ckpt_local}")
    print(f"[MARL-MIX] seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)}) device={device}")
    print(
        f"[MARL-MIX] ddlAlpha choices={env_kwargs['alpha_deadline_choices']} probs={env_kwargs['alpha_deadline_probs']}"
    )
    global_agent, local_agent, dims = build_agents_and_probe_dims(
        env_kwargs=env_kwargs,
        probe_seed=probe_seed,
        device=device,
        ckpt_global=ckpt_global,
        ckpt_local=ckpt_local,
    )
    print(f"[MARL-MIX probe] {dims}")
    logger = CSVLogger(
        filepath=out_csv,
        fieldnames=[
            "run_idx",
            "seed",
            "vm",
            "host",
            "mgr",
            "energy",
            "task_avg_lateness",
            "task_avg_tardiness",
            "task_success_rate",
            "task_pct_tardy",
            "wf_avg_lateness",
            "wf_avg_tardiness",
            "wf_pct_tardy",
            "wf_success_rate",
            "makespan",
            "steps",
            "energy_check_diff",
        ],
    )
    vm_list, host_list, mgr_list = ([], [], [])
    e_list, ms_list = ([], [])
    task_late_list, task_tard_list, task_succ_list, task_pct_list = ([], [], [], [])
    wf_late_list, wf_tard_list, wf_pct_list, wf_succ_list = ([], [], [], [])
    steps_list, diff_list = ([], [])
    for idx, sd in enumerate(seeds):
        print(f"\n===== Eval {idx}/{len(seeds) - 1} | seed={sd} =====")
        set_seed(int(sd))
        m = evaluate_one_seed(
            env_kwargs=env_kwargs,
            global_agent=global_agent,
            local_agent=local_agent,
            seed=int(sd),
            episode_id=100000 + idx,
        )
        logger.log(
            run_idx=int(idx),
            seed=int(sd),
            vm=float(m["vm"]),
            host=float(m["host"]),
            mgr=float(m["mgr"]),
            energy=float(m["energy"]),
            task_avg_lateness=float(m["task_avg_lateness"]),
            task_avg_tardiness=float(m["task_avg_tardiness"]),
            task_success_rate=float(m["task_success_rate"]),
            task_pct_tardy=float(m["task_pct_tardy"]),
            wf_avg_lateness=float(m["wf_avg_lateness"]),
            wf_avg_tardiness=float(m["wf_avg_tardiness"]),
            wf_pct_tardy=float(m["wf_pct_tardy"]),
            wf_success_rate=float(m["wf_success_rate"]),
            makespan=float(m["makespan"]),
            steps=float(m["steps"]),
            energy_check_diff=float(m["energy_check_diff"]),
        )
        vm_list.append(float(m["vm"]))
        host_list.append(float(m["host"]))
        mgr_list.append(float(m["mgr"]))
        e_list.append(float(m["energy"]))
        task_late_list.append(float(m["task_avg_lateness"]))
        task_tard_list.append(float(m["task_avg_tardiness"]))
        task_succ_list.append(float(m["task_success_rate"]))
        task_pct_list.append(float(m["task_pct_tardy"]))
        wf_late_list.append(float(m["wf_avg_lateness"]))
        wf_tard_list.append(float(m["wf_avg_tardiness"]))
        wf_pct_list.append(float(m["wf_pct_tardy"]))
        wf_succ_list.append(float(m["wf_success_rate"]))
        ms_list.append(float(m["makespan"]))
        steps_list.append(float(m["steps"]))
        diff_list.append(float(m["energy_check_diff"]))
    vm_m = _mean(vm_list)
    host_m = _mean(host_list)
    mgr_m = _mean(mgr_list)
    e_m = _mean(e_list)
    task_late_m = _mean(task_late_list)
    task_tard_m = _mean(task_tard_list)
    task_succ_m = _mean(task_succ_list)
    task_pct_m = _mean(task_pct_list)
    wf_late_m = _mean(wf_late_list)
    wf_tard_m = _mean(wf_tard_list)
    wf_pct_m = _mean(wf_pct_list)
    wf_succ_m = _mean(wf_succ_list)
    ms_m = _mean(ms_list)
    steps_m = _mean(steps_list)
    diff_m = _mean(diff_list)
    logger.log(
        run_idx="mean",
        seed="mean",
        vm=float(vm_m),
        host=float(host_m),
        mgr=float(mgr_m),
        energy=float(e_m),
        task_avg_lateness=float(task_late_m),
        task_avg_tardiness=float(task_tard_m),
        task_success_rate=float(task_succ_m),
        task_pct_tardy=float(task_pct_m),
        wf_avg_lateness=float(wf_late_m),
        wf_avg_tardiness=float(wf_tard_m),
        wf_pct_tardy=float(wf_pct_m),
        wf_success_rate=float(wf_succ_m),
        makespan=float(ms_m),
        steps=float(steps_m),
        energy_check_diff=float(diff_m),
    )
    print("\n================= MARL0.75-MIX（multi-seed）汇总 =================")
    print(
        f"[MARL0.75-MIX] vm={vm_m:.6f} host={host_m:.6f} mgr={mgr_m:.6f} energy={e_m:.3f} J | task_avg_lateness={task_late_m:.6f} task_avg_tardiness={task_tard_m:.6f} task_success_rate={task_succ_m:.6f} task_pct_tardy={task_pct_m:.6f} | wf_avg_lateness={wf_late_m:.6f} wf_avg_tardiness={wf_tard_m:.6f} wf_success_rate={wf_succ_m:.6f} wf_pct_tardy={wf_pct_m:.6f} | makespan={ms_m:.3f}"
    )
    diffs = [x for x in diff_list if np.isfinite(x)]
    if diffs:
        print(
            f"[EnergyCheckSummary] diff_mean={float(np.mean(diffs)):.6f} diff_max={float(np.max(np.abs(diffs))):.6f}"
        )
    else:
        print("[EnergyCheckSummary] diff_mean=nan (all skipped)")
    print("=================================================================")
    print(f"结果已写入 CSV: {out_csv}")


if __name__ == "__main__":
    SCENARIO = parse_scenario("eval", "marl")
    main()
