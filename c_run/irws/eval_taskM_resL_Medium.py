# -*- coding: utf-8 -*-
"""
run_irws_routeA_taskM_resL_ddlMix_Medium_rewardPD3QN.py

IRWS 评测脚本（PPO 选 VM + TaskSelector 选 Task）

对齐训练脚本：
    train_irws_ppo_taskM_resL.py（修改后版本）

对齐点：
1) 使用修改后的环境：
   b_irws_FCFS_ddlMix.env_irws.IRWSEnvRouteA
2) 资源规模对齐 resL：
   3 hosts, vms_per_host=(9,8,8)
3) deadline cache 对齐：
   ddl_fcfs_taskM_resL_seed0-1000.json
4) ddlAlpha mixture 对齐：
   20% -> 2.0, 80% -> 3.0
5) reward 参数对齐修改后的 env：
   alpha_delay_vm=0.75
   task_baseline_norm=300.0
   energy_reward_scale=1e-3
6) ckpt 目录对齐：
   irws_taskM_resL_mix_Medium_rewardPD3QN
7) deterministic=True，默认 seeds=2..31

说明：
- 评测时只用 PPO + TaskSelector，intrinsic net 不参与 action 选择
- metrics 统一从 env.get_episode_metrics() 读取
- 保留 EnergyCheck：env.total_energy vs energy_from_records(env.records, env.hosts)
"""

from __future__ import annotations

import os
import sys
import math
import random
import numpy as np
import torch


def _find_project_root(start_dir: str, max_up: int = 7) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        if (
            os.path.isdir(os.path.join(cur, "a_set"))
            and os.path.isdir(os.path.join(cur, "b_irws_FCFS"))
        ):
            return cur
        nxt = os.path.abspath(os.path.join(cur, ".."))
        if nxt == cur:
            break
        cur = nxt
    return os.path.abspath(start_dir)


PROJECT_DIR = _find_project_root(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from a_set.metrics_logger import CSVLogger
from a_set.workflow_opt import energy_from_records

from b_irws.env_irws import IRWSEnvRouteA
from b_irws_FCFS.gnn_task_ranker_FCFS import IRWSTaskSelector, TaskSelectorConfig
from b_irws_FCFS.ppo_vm_agent_FCFS import PPOVMAgent, PPOConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _fmt(x, digits=6):
    if _is_finite(x):
        return f"{float(x):.{digits}f}"
    return "nan"


def _first_valid(d: dict, keys, default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default


def _derive_success_rate(pct_tardy, total_cnt, tardy_cnt):
    if _is_finite(pct_tardy):
        return 1.0 - float(pct_tardy)
    if _is_finite(total_cnt) and float(total_cnt) > 0 and _is_finite(tardy_cnt):
        return 1.0 - float(tardy_cnt) / float(total_cnt)
    return float("nan")


def _canonicalize_episode_metrics(m_raw: dict) -> dict:
    out = {}

    out["total_energy"] = _first_valid(m_raw, ["total_energy", "energy"])
    out["makespan"] = _first_valid(m_raw, ["makespan", "episode_makespan", "total_time"])

    out["task_count"] = _first_valid(m_raw, [
        "task_count", "num_tasks", "n_tasks", "tasks",
        "total_tasks", "completed_tasks"
    ])
    out["task_tardy_count"] = _first_valid(m_raw, [
        "task_tardy_count", "num_tardy_tasks", "tardy_tasks",
        "task_num_tardy"
    ])

    out["task_total_lateness"] = _first_valid(m_raw, [
        "task_total_lateness", "total_task_lateness"
    ])
    out["task_total_tardiness"] = _first_valid(m_raw, [
        "task_total_tardiness", "total_task_tardiness"
    ])
    out["task_avg_lateness"] = _first_valid(m_raw, [
        "task_avg_lateness", "avg_lateness", "avg_task_lateness"
    ])
    out["task_avg_tardiness"] = _first_valid(m_raw, [
        "task_avg_tardiness", "avg_tardiness", "avg_task_tardiness"
    ])
    out["pct_tardy"] = _first_valid(m_raw, [
        "pct_tardy", "task_pct_tardy", "task_tardy_ratio"
    ])

    out["task_success_rate"] = _first_valid(m_raw, [
        "task_success_rate", "task_success_ratio"
    ])
    if not _is_finite(out["task_success_rate"]):
        out["task_success_rate"] = _derive_success_rate(
            out["pct_tardy"], out["task_count"], out["task_tardy_count"]
        )

    out["wf_count"] = _first_valid(m_raw, [
        "wf_count", "workflow_count", "num_workflows",
        "n_workflows", "workflows", "completed_workflows"
    ])
    out["wf_tardy_count"] = _first_valid(m_raw, [
        "wf_tardy_count", "workflow_tardy_count",
        "num_tardy_workflows", "tardy_workflows"
    ])

    out["wf_total_lateness"] = _first_valid(m_raw, [
        "wf_total_lateness", "workflow_total_lateness", "total_wf_lateness"
    ])
    out["wf_total_tardiness"] = _first_valid(m_raw, [
        "wf_total_tardiness", "workflow_total_tardiness", "total_wf_tardiness"
    ])
    out["wf_avg_lateness"] = _first_valid(m_raw, [
        "wf_avg_lateness", "workflow_avg_lateness", "avg_wf_lateness"
    ])
    out["wf_avg_tardiness"] = _first_valid(m_raw, [
        "wf_avg_tardiness", "workflow_avg_tardiness", "avg_wf_tardiness"
    ])
    out["wf_pct_tardy"] = _first_valid(m_raw, [
        "wf_pct_tardy", "workflow_pct_tardy", "wf_tardy_ratio"
    ])

    out["wf_success_rate"] = _first_valid(m_raw, [
        "wf_success_rate", "workflow_success_rate", "wf_success_ratio"
    ])
    if not _is_finite(out["wf_success_rate"]):
        out["wf_success_rate"] = _derive_success_rate(
            out["wf_pct_tardy"], out["wf_count"], out["wf_tardy_count"]
        )

    out["wf_avg_tardy"] = _first_valid(m_raw, ["wf_avg_tardy"])
    return out


def build_env_kwargs_irws(random_seed=1):
    dax_dir = os.path.join(PROJECT_DIR, "a_dax")
    dax_30 = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "Ligo_30.xml"),  # 与你 HRL 代码一致（Ligo_*)
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    dax_50 = [
        os.path.join(dax_dir, "CyberShake_50.xml"),
        os.path.join(dax_dir, "Epigenomics_47.xml"),
        os.path.join(dax_dir, "Ligo_50.xml"),  # 与你 HRL 代码一致（Ligo_*)
        os.path.join(dax_dir, "Montage_50.xml"),
        os.path.join(dax_dir, "Sipht_58.xml"),
    ]

    # 30% / 70%：重复实现（与 HRL 一致）
    dax_list = list(dax_30) * 3 + list(dax_50) * 7

    horizon = 1e9
    arrival_lambda = 0.03
    workflows_per_episode = 50
    max_ready_tasks = "auto"
    normalize_obs = True

    # resL
    num_hosts = 9
    vms_per_host = (9, 9, 9, 8, 8, 8, 8, 8, 8)

    pc_tiers = (1.0, 2.0, 4.0, 6.0, 8.0)
    bw_tiers = (1000.0, 2000.0, 4000.0, 6000.0, 8000.0)

    deadline_cache_path = os.path.join(
        PROJECT_DIR, "d_output", "deadlines",
        "ddl_fcfs_taskM_resL_seed0-1000.json",
    )
    if not os.path.exists(deadline_cache_path):
        raise FileNotFoundError(f"deadline cache 不存在：{deadline_cache_path}")

    deadline_alpha = 1.5
    DEADLINE_ALPHA_CHOICES = (2.0, 3.0)
    DEADLINE_ALPHA_PROBS = (0.5, 0.5)

    # 与修改后的训练代码一致
    alpha_delay_vm = 0.75
    task_baseline_norm = 300.0
    energy_reward_scale = 1e-3

    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=float(horizon),
        arrival_lambda=float(arrival_lambda),
        random_seed=int(random_seed),
        workflows_per_episode=int(workflows_per_episode),
        max_ready_tasks=max_ready_tasks,
        normalize=bool(normalize_obs),

        num_hosts=int(num_hosts),
        vms_per_host=tuple(int(x) for x in vms_per_host),
        pc_tiers=tuple(float(x) for x in pc_tiers),
        bw_tiers=tuple(float(x) for x in bw_tiers),

        deadline_mode="cache_fcfs",
        deadline_cache_path=deadline_cache_path,
        deadline_alpha=float(deadline_alpha),
        deadline_alpha_choices=DEADLINE_ALPHA_CHOICES,
        deadline_alpha_probs=DEADLINE_ALPHA_PROBS,
        deadline_cache_strict=True,

        alpha_delay_vm=float(alpha_delay_vm),
        task_baseline_norm=float(task_baseline_norm),
        energy_reward_scale=float(energy_reward_scale),

        energy_check_on_done=True,
    )
    return env_kwargs


def probe_obs_act_dims(env_kwargs: dict, seed: int) -> tuple[int, int]:
    kw = dict(env_kwargs)
    kw["random_seed"] = int(seed)
    env = IRWSEnvRouteA(**kw)
    env.reset()

    vm_st = None
    while not env.done_flag:
        cand = env.get_ready_task_candidates()
        gids = cand["gids"]
        if len(gids) == 0:
            break
        env.select_task(int(gids[0]))
        vm_st = env.get_vm_obs_and_mask()
        break

    if vm_st is None:
        raise RuntimeError("Probe failed：环境没有产生决策点，无法得到 obs_dim/act_dim")

    obs_dim = int(vm_st["obs"].shape[0])
    act_dim = int(vm_st["mask"].shape[0])
    return obs_dim, act_dim


def build_agents_and_load(env_kwargs: dict):
    obs_dim, act_dim = probe_obs_act_dims(env_kwargs, seed=int(env_kwargs["random_seed"]))
    print(f"[IRWS probe] obs_dim={obs_dim}, act_dim={act_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_selector = IRWSTaskSelector(TaskSelectorConfig(feat_dim=7, hidden=128, device=str(device)))

    ppo = PPOVMAgent(
        PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=10.0,
            update_epochs=4,
            minibatch_size=256,
            device=str(device),
        ),
        hidden=(1024, 1024, 512, 512, 256),
    )

    ckpt_dir_default = os.path.join(
        PROJECT_DIR, "d_output", "ckpts",
        "irws_taskS_resL_mix_Medium_rewardPD3QN_Medium",
    )

    ckpt_path = os.path.join(ckpt_dir_default, "best_joint.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"best_joint.pth 不存在：{ckpt_path}\n"
            f"请确认训练输出目录名是否一致。"
        )

    print(f"[IRWS load] ckpt_dir: {ckpt_dir_default}")
    print(f"[IRWS load] joint   : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ppo" not in ckpt or "task_selector" not in ckpt:
        raise ValueError("joint ckpt 缺少字段：ppo / task_selector")

    meta = dict(ckpt.get("meta", {}))
    if "obs_dim" in meta and int(meta["obs_dim"]) != obs_dim:
        raise RuntimeError(
            "❌ ckpt 与当前资源规模不匹配（obs_dim 不一致）。\n"
            f"ckpt.obs_dim={int(meta['obs_dim'])}, env.obs_dim={obs_dim}\n"
            "请换成同资源规模训练出来的 best_joint.pth。"
        )

    task_selector.net.load_state_dict(ckpt["task_selector"], strict=True)
    ppo.net.load_state_dict(ckpt["ppo"], strict=True)

    task_selector.net.eval()
    ppo.net.eval()

    return task_selector, ppo


def evaluate_one_seed(env_kwargs: dict, task_selector, ppo, seed: int) -> dict:
    kw = dict(env_kwargs)
    kw["random_seed"] = int(seed)
    env = IRWSEnvRouteA(**kw)
    env.reset()

    steps = 0
    with torch.no_grad():
        while not env.done_flag:
            cand = env.get_ready_task_candidates()
            gids, feats, mask = cand["gids"], cand["feats"], cand["mask"]

            idx, _ = task_selector.select(feats, mask, deterministic=True)
            env.select_task(int(gids[idx]))

            vm_st = env.get_vm_obs_and_mask()
            a_vm, _ = ppo.select_action(vm_st["obs"], vm_st["mask"], deterministic=True)
            env.step(int(a_vm))
            steps += 1

    m_raw = dict(env.get_episode_metrics())
    m = _canonicalize_episode_metrics(m_raw)
    m["steps"] = float(steps)

    try:
        E_acc = float(getattr(env, "total_energy", 0.0))
        recs = getattr(env, "records", None)
        hosts = getattr(env, "hosts", None)
        if recs is not None and hosts is not None:
            E_re = float(energy_from_records(recs, hosts))
            m["energy_check_diff"] = float(E_acc - E_re)
        else:
            m["energy_check_diff"] = float("nan")
    except Exception:
        m["energy_check_diff"] = float("nan")

    return m


def main():
    seeds = list(range(2, 32))

    env_kwargs_base = build_env_kwargs_irws(random_seed=seeds[0])

    result_dir = os.path.join(PROJECT_DIR, "d_output", "run", "irws_fcfs_wfInfo_Mix_Alpha_rn")
    _ensure_dir(result_dir)

    csv_path = os.path.join(
        result_dir,
        "eval_taskM_resL_Medium.csv"
    )

    logger = CSVLogger(
        filepath=csv_path,
        fieldnames=[
            "run_idx", "seed", "steps",
            "total_energy", "makespan",

            "task_count", "task_tardy_count",
            "task_total_lateness", "task_total_tardiness",
            "task_avg_lateness", "task_avg_tardiness",
            "pct_tardy", "task_success_rate",

            "wf_count", "wf_tardy_count",
            "wf_total_lateness", "wf_total_tardiness",
            "wf_avg_lateness", "wf_avg_tardiness",
            "wf_pct_tardy", "wf_success_rate",

            "wf_avg_tardy",
            "energy_check_diff",
        ],
    )

    print(">>> 构建 IRWS agents 并加载 best_joint.pth ...")
    task_selector, ppo = build_agents_and_load(env_kwargs_base)

    rows = []

    for idx, sd in enumerate(seeds):
        print(f"\n===== Eval {idx + 1}/{len(seeds)} | seed={sd} =====")
        set_seed(sd)

        kw_i = dict(env_kwargs_base)
        kw_i["random_seed"] = int(sd)

        m = evaluate_one_seed(kw_i, task_selector, ppo, seed=int(sd))

        row = dict(
            run_idx=int(idx),
            seed=int(sd),
            steps=float(m.get("steps", float("nan"))),
            total_energy=float(m.get("total_energy", float("nan"))),
            makespan=float(m.get("makespan", float("nan"))),

            task_count=float(m.get("task_count", float("nan"))),
            task_tardy_count=float(m.get("task_tardy_count", float("nan"))),
            task_total_lateness=float(m.get("task_total_lateness", float("nan"))),
            task_total_tardiness=float(m.get("task_total_tardiness", float("nan"))),
            task_avg_lateness=float(m.get("task_avg_lateness", float("nan"))),
            task_avg_tardiness=float(m.get("task_avg_tardiness", float("nan"))),
            pct_tardy=float(m.get("pct_tardy", float("nan"))),
            task_success_rate=float(m.get("task_success_rate", float("nan"))),

            wf_count=float(m.get("wf_count", float("nan"))),
            wf_tardy_count=float(m.get("wf_tardy_count", float("nan"))),
            wf_total_lateness=float(m.get("wf_total_lateness", float("nan"))),
            wf_total_tardiness=float(m.get("wf_total_tardiness", float("nan"))),
            wf_avg_lateness=float(m.get("wf_avg_lateness", float("nan"))),
            wf_avg_tardiness=float(m.get("wf_avg_tardiness", float("nan"))),
            wf_pct_tardy=float(m.get("wf_pct_tardy", float("nan"))),
            wf_success_rate=float(m.get("wf_success_rate", float("nan"))),

            wf_avg_tardy=float(m.get("wf_avg_tardy", float("nan"))),
            energy_check_diff=float(m.get("energy_check_diff", float("nan"))),
        )

        logger.log(**row)
        rows.append(row)

        task_cnt = row["task_count"]
        task_tardy_cnt = row["task_tardy_count"]
        wf_cnt = row["wf_count"]
        wf_tardy_cnt = row["wf_tardy_count"]

        task_cnt_str = "nan" if not _is_finite(task_cnt) else f"{int(round(task_cnt))}"
        task_tardy_cnt_str = "nan" if not _is_finite(task_tardy_cnt) else f"{int(round(task_tardy_cnt))}"
        wf_cnt_str = "nan" if not _is_finite(wf_cnt) else f"{int(round(wf_cnt))}"
        wf_tardy_cnt_str = "nan" if not _is_finite(wf_tardy_cnt) else f"{int(round(wf_tardy_cnt))}"

        print(
            f"[IRWS] "
            f"steps={int(row['steps']) if _is_finite(row['steps']) else 'nan'} | "
            f"energy={_fmt(row['total_energy'], 3)} J | "
            f"task={task_cnt_str} tardy={task_tardy_cnt_str} | "
            f"task_avg_lateness={_fmt(row['task_avg_lateness'])} "
            f"task_avg_tardiness={_fmt(row['task_avg_tardiness'])} "
            f"task_success_rate={_fmt(row['task_success_rate'])} | "
            f"wf={wf_cnt_str} tardy={wf_tardy_cnt_str} | "
            f"wf_avg_lateness={_fmt(row['wf_avg_lateness'])} "
            f"wf_avg_tardiness={_fmt(row['wf_avg_tardiness'])} "
            f"wf_success_rate={_fmt(row['wf_success_rate'])} | "
            f"wf_avg_tardy(old)={_fmt(row['wf_avg_tardy'])} | "
            f"makespan={_fmt(row['makespan'], 3)}"
        )

    def col_mean(key):
        vals = [float(r[key]) for r in rows if _is_finite(r.get(key, float("nan")))]
        if len(vals) == 0:
            return float("nan")
        return float(np.mean(vals))

    mean_row = dict(
        run_idx="AVG",
        seed="2-31",
        steps=col_mean("steps"),
        total_energy=col_mean("total_energy"),
        makespan=col_mean("makespan"),

        task_count=col_mean("task_count"),
        task_tardy_count=col_mean("task_tardy_count"),
        task_total_lateness=col_mean("task_total_lateness"),
        task_total_tardiness=col_mean("task_total_tardiness"),
        task_avg_lateness=col_mean("task_avg_lateness"),
        task_avg_tardiness=col_mean("task_avg_tardiness"),
        pct_tardy=col_mean("pct_tardy"),
        task_success_rate=col_mean("task_success_rate"),

        wf_count=col_mean("wf_count"),
        wf_tardy_count=col_mean("wf_tardy_count"),
        wf_total_lateness=col_mean("wf_total_lateness"),
        wf_total_tardiness=col_mean("wf_total_tardiness"),
        wf_avg_lateness=col_mean("wf_avg_lateness"),
        wf_avg_tardiness=col_mean("wf_avg_tardiness"),
        wf_pct_tardy=col_mean("wf_pct_tardy"),
        wf_success_rate=col_mean("wf_success_rate"),

        wf_avg_tardy=col_mean("wf_avg_tardy"),
        energy_check_diff=col_mean("energy_check_diff"),
    )

    logger.log(**mean_row)

    print("\n================= IRWS（30 seeds）汇总 =================")
    print(f"avg_steps              = {_fmt(mean_row['steps'], 3)}")
    print(f"avg_energy             = {_fmt(mean_row['total_energy'], 3)} J")
    print(f"avg_makespan           = {_fmt(mean_row['makespan'], 3)}")
    print(f"avg_task_count         = {_fmt(mean_row['task_count'], 3)}")
    print(f"avg_task_tardy_count   = {_fmt(mean_row['task_tardy_count'], 3)}")
    print(f"avg_task_lateness      = {_fmt(mean_row['task_avg_lateness'])}")
    print(f"avg_task_tardiness     = {_fmt(mean_row['task_avg_tardiness'])}")
    print(f"avg_task_pct_tardy     = {_fmt(mean_row['pct_tardy'])}")
    print(f"avg_task_success_rate  = {_fmt(mean_row['task_success_rate'])}")
    print(f"avg_wf_count           = {_fmt(mean_row['wf_count'], 3)}")
    print(f"avg_wf_tardy_count     = {_fmt(mean_row['wf_tardy_count'], 3)}")
    print(f"avg_wf_lateness        = {_fmt(mean_row['wf_avg_lateness'])}")
    print(f"avg_wf_tardiness       = {_fmt(mean_row['wf_avg_tardiness'])}")
    print(f"avg_wf_pct_tardy       = {_fmt(mean_row['wf_pct_tardy'])}")
    print(f"avg_wf_success_rate    = {_fmt(mean_row['wf_success_rate'])}")
    print(f"avg_wf_avg_tardy(old)  = {_fmt(mean_row['wf_avg_tardy'])}")
    print(f"energy_check_diff_mean = {_fmt(mean_row['energy_check_diff'])}")
    print("=========================================================")
    print(f"结果已写入 CSV: {csv_path}")


if __name__ == "__main__":
    main()