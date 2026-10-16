# -*- coding: utf-8 -*-
"""
run_pd3qn_taskL_resL_ddlAlphaMix_shallowAlignHRLVM.py

面向【修改后的 PD3QN 训练代码】的评测脚本
对齐点：
1) 使用修改后的 env：
   - obs: np.ndarray
   - mask: env.get_action_mask()
   - step(): (obs2, reward, terminated, truncated, info)
2) reward 参数口径与修改后的训练脚本一致：
   - alpha_delay_vm
   - task_baseline_norm
   - energy_reward_scale
3) deadline 口径与修改后的训练脚本一致：
   - deadline_alpha_choices=[2.0, 3.0]
   - deadline_alpha_probs=[0.2, 0.8]
4) 指标统一从 env.get_episode_metrics() 读取
5) 保留 EnergyCheck：env.total_energy vs energy_from_records(env.records, env.hosts)

输出：
- 控制台每个 seed 一行详细结果
- CSV：outputs/run/pd3qn_wfInfo/run_pd3qn_taskL_resL_ddlAlphaMix_shallowAlignHRLVM.csv
"""

import os
import sys
import inspect
import random
import numpy as np
import torch


# ---------------------------------------------------------------------
# 0) 自动定位项目根目录 & sys.path
# ---------------------------------------------------------------------


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

print(f"[PATH] project_root = {ROOT_DIR}")

from common.metrics_logger import CSVLogger
from common.workflow_opt import energy_from_records
from common.d3qn_agent_PER import D3QNAgent

# ---------------------------------------------------------------------
# 改成你修改后的 env 导入路径
# ---------------------------------------------------------------------
from algorithms.pd3qn.env_pd3qn import (
    CloudWorkflowEnv_PD3QN_Cost,
)

print(f"[PATH] env_file     = {inspect.getfile(CloudWorkflowEnv_PD3QN_Cost)}")


# ---------------------------------------------------------------------
# 1) 通用工具函数
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0):
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _pick_first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _derive_task_metrics(metrics: dict) -> dict:
    out = {}

    out["task_avg_lateness"] = _safe_float(
        _pick_first(metrics, ["task_avg_lateness", "avg_lateness_task", "avg_task_lateness"], np.nan)
    )
    out["task_avg_tardiness"] = _safe_float(
        _pick_first(metrics, ["task_avg_tardiness", "avg_tardiness_task", "avg_task_tardiness"], np.nan)
    )
    out["task_total_lateness"] = _safe_float(
        _pick_first(metrics, ["task_total_lateness", "task_lateness_sum", "sum_task_lateness"], np.nan)
    )
    out["task_total_tardiness"] = _safe_float(
        _pick_first(metrics, ["task_total_tardiness", "task_tardiness_sum", "sum_task_tardiness"], np.nan)
    )

    num_tasks = _pick_first(metrics, ["num_tasks", "task_count", "n_tasks"], None)
    num_tardy_tasks = _pick_first(metrics, ["num_tardy_tasks", "task_num_tardy", "tardy_tasks"], None)

    out["num_tasks"] = _safe_int(num_tasks, 0)
    out["num_tardy_tasks"] = _safe_int(num_tardy_tasks, 0)

    task_pct_tardy = _pick_first(metrics, ["task_pct_tardy", "pct_tardy"], None)
    if task_pct_tardy is None and num_tasks not in (None, 0) and num_tardy_tasks is not None:
        task_pct_tardy = float(num_tardy_tasks) / float(num_tasks)
    out["task_pct_tardy"] = _safe_float(task_pct_tardy, np.nan)

    task_success_rate = _pick_first(metrics, ["task_success_rate"], None)
    if task_success_rate is None and not np.isnan(out["task_pct_tardy"]):
        task_success_rate = 1.0 - out["task_pct_tardy"]
    out["task_success_rate"] = _safe_float(task_success_rate, np.nan)

    return out


def _derive_workflow_metrics(metrics: dict) -> dict:
    out = {}

    out["wf_avg_lateness"] = _safe_float(
        _pick_first(metrics, ["wf_avg_lateness", "workflow_avg_lateness", "avg_wf_lateness"], np.nan)
    )
    out["wf_avg_tardiness"] = _safe_float(
        _pick_first(metrics, ["wf_avg_tardiness", "workflow_avg_tardiness", "avg_wf_tardiness"], np.nan)
    )
    out["wf_lateness_sum"] = _safe_float(
        _pick_first(metrics, ["wf_lateness_sum", "workflow_lateness_sum", "wf_total_lateness"], np.nan)
    )
    out["wf_tardiness_sum"] = _safe_float(
        _pick_first(metrics, ["wf_tardiness_sum", "workflow_tardiness_sum", "wf_total_tardiness"], np.nan)
    )

    num_workflows = _pick_first(metrics, ["num_workflows", "wf_count", "n_workflows"], None)
    num_tardy_workflows = _pick_first(metrics, ["num_tardy_workflows", "wf_num_tardy", "tardy_workflows"], None)

    out["num_workflows"] = _safe_int(num_workflows, 0)
    out["num_tardy_workflows"] = _safe_int(num_tardy_workflows, 0)

    wf_pct_tardy = _pick_first(metrics, ["wf_pct_tardy", "workflow_pct_tardy"], None)
    if wf_pct_tardy is None and num_workflows not in (None, 0) and num_tardy_workflows is not None:
        wf_pct_tardy = float(num_tardy_workflows) / float(num_workflows)
    out["wf_pct_tardy"] = _safe_float(wf_pct_tardy, np.nan)

    wf_success_rate = _pick_first(metrics, ["wf_success_rate", "workflow_success_rate"], None)
    if wf_success_rate is None and not np.isnan(out["wf_pct_tardy"]):
        wf_success_rate = 1.0 - out["wf_pct_tardy"]
    out["wf_success_rate"] = _safe_float(wf_success_rate, np.nan)

    return out


# ---------------------------------------------------------------------
# 2) 单 episode 评测
# ---------------------------------------------------------------------
def evaluate_one_episode(env_cls, env_kwargs: dict, agent: D3QNAgent, deterministic: bool = True):
    env = env_cls(**env_kwargs)
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret

    # ===== ResetCheck =====
    t_now = getattr(env, "current_time", getattr(env, "time", getattr(env, "now", None)))
    E0 = getattr(env, "total_energy", None)
    arr = getattr(env, "arrival_times", None)
    t_first = float(np.min(arr)) if arr is not None and len(arr) > 0 else None
    print(f"[ResetCheck] t_now={t_now} E0={E0} t_first_arr={t_first}")

    ep_return = 0.0
    steps = 0
    last_info = {}

    done = False
    while not done:
        mask = env.get_action_mask()
        act = agent.select_action(obs, mask, deterministic=deterministic, count_step=False)
        obs, rew, terminated, truncated, info = env.step(int(act))
        done = bool(terminated or truncated)

        ep_return += float(rew)
        steps += 1
        if isinstance(info, dict) and info:
            last_info = info

        if steps > 10**9:
            break

    # ===== EnergyCheck =====
    E_acc = _safe_float(getattr(env, "total_energy", np.nan), np.nan)
    E_re = _safe_float(energy_from_records(env.records, env.hosts), np.nan)
    print(f"[EnergyCheck] E_acc={E_acc:.6f}  E_re={E_re:.6f}  diff={E_re - E_acc:.6f}")

    raw_metrics = env.get_episode_metrics()

    metrics = dict(raw_metrics)
    metrics["energy_acc"] = float(E_acc)
    metrics["energy_recalc"] = float(E_re)
    metrics["energy_recalc_diff"] = float(E_re - E_acc)

    metrics.update(_derive_task_metrics(metrics))
    metrics.update(_derive_workflow_metrics(metrics))

    metrics["total_energy"] = _safe_float(_pick_first(metrics, ["total_energy"], E_acc), E_acc)
    metrics["makespan"] = _safe_float(_pick_first(metrics, ["makespan"], np.nan), np.nan)

    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass
    del env

    return ep_return, steps, metrics, last_info


# ---------------------------------------------------------------------
# 3) 构建 env_kwargs + agent
# ---------------------------------------------------------------------
def build_agent_and_env_kwargs():
    dax_dir = os.path.join(ROOT_DIR, "data/dax")
    dax_list_30 = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "LIGO_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    # 50规模组
    dax_list_50 = [
        os.path.join(dax_dir, "CyberShake_50.xml"),
        os.path.join(dax_dir, "Epigenomics_47.xml"),
        os.path.join(dax_dir, "LIGO_50.xml"),
        os.path.join(dax_dir, "Montage_50.xml"),
        os.path.join(dax_dir, "Sipht_58.xml"),
    ]

    # 100规模组
    dax_list_100 = [
        os.path.join(dax_dir, "CyberShake_100.xml"),
        os.path.join(dax_dir, "Epigenomics_100.xml"),
        os.path.join(dax_dir, "LIGO_100.xml"),
        os.path.join(dax_dir, "Montage_100.xml"),
        os.path.join(dax_dir, "Sipht_97.xml"),
    ]

    # 通过重复实现 20%/30%/50% 加权均匀采样（2:3:5）
    # 2/(2+3+5)=0.2, 3/(2+3+5)=0.3, 5/(2+3+5)=0.5
    dax_list = dax_list_30 * 2 + dax_list_50 * 3 + dax_list_100 * 5

    DEADLINE_CACHE_PATH = os.path.join(
        ROOT_DIR, "data", "deadlines",
        "ddl_fcfs_taskL_resS_seed0-1000.json",
    )
    if not os.path.exists(DEADLINE_CACHE_PATH):
        raise FileNotFoundError(f"deadline cache 不存在：{DEADLINE_CACHE_PATH}")

    # -----------------------------------------------------------------
    # 改成与“修改后的训练代码”一致的新参数口径
    # -----------------------------------------------------------------
    env_kwargs_base = dict(
        dax_paths=dax_list,
        horizon=1e9,
        arrival_lambda=0.03,
        random_seed=1,
        workflows_per_episode=50,
        max_ready_tasks="auto",
        normalize=True,

        # resources
        num_hosts=9,
        vms_per_host=(9, 9, 9, 8, 8, 8, 8, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        # reward params aligned to modified training script
        alpha_delay_vm=0.75,
        task_baseline_norm=300.0,
        energy_reward_scale=1e-3,

        # compatibility placeholder
        energy_norm_per_mi_ref=20.0,

        # deadline cache aligned
        deadline_mode="cache_fcfs",
        deadline_cache_path=DEADLINE_CACHE_PATH,
        deadline_alpha=1.5,

        # 与修改后的训练代码保持一致：ddlAlpha 离散分布
        deadline_alpha_choices=[2.0, 3.0],
        deadline_alpha_probs=[0.5, 0.5],
        deadline_alpha_seed_offset=20260317,

        deadline_cache_strict=True,
    )

    tmp_env = CloudWorkflowEnv_PD3QN_Cost(**env_kwargs_base)
    ret = tmp_env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret
    obs_dim = int(np.asarray(obs, dtype=np.float32).reshape(-1).shape[0])
    act_dim = int(tmp_env.action_space.n)

    if hasattr(tmp_env, "close"):
        try:
            tmp_env.close()
        except Exception:
            pass
    del tmp_env

    print(f"[PD3QN-EVAL] obs_dim={obs_dim}, act_dim={act_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = D3QNAgent(
        input_dim=obs_dim,
        output_dim=act_dim,
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=120000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=80000,
        target_update_tau=0.005,
        grad_clip=10.0,
        hidden_dims=(1024, 1024, 512, 512, 256),
        device=device,
        use_per=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=80000,
    )

    # -----------------------------------------------------------------
    # 改成与修改后的训练脚本一致的 ckpt 目录
    # -----------------------------------------------------------------
    ckpt_path = os.path.join(
        ROOT_DIR, "outputs", "ckpts",
        "pd3qn_taskS_resL_warmup_mix_nr_Medium",
        "best_pd3qn.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt 不存在：{ckpt_path}")

    print(f"[PD3QN-EVAL] load ckpt: {ckpt_path}")
    agent.load(ckpt_path)

    return env_kwargs_base, agent


# ---------------------------------------------------------------------
# 4) main：30 seeds 评测 + CSV
# ---------------------------------------------------------------------
def main():
    base_seed = 2
    num_runs = 30
    seeds = [base_seed + i for i in range(num_runs)]

    result_dir = os.path.join(ROOT_DIR, "outputs", "run", "pd3qn_wfInfo_rn")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(
        result_dir,
        "run_pd3qn_taskL_resL_ddlAlphaMix_nr_Medium.csv"
    )

    logger = CSVLogger(
        filepath=csv_path,
        fieldnames=[
            "run_idx", "seed",
            "eval_return", "eval_steps",

            "eval_total_energy",
            "eval_energy_acc",
            "eval_energy_recalc",
            "eval_energy_recalc_diff",

            "eval_num_tasks",
            "eval_num_tardy_tasks",
            "eval_task_avg_lateness",
            "eval_task_avg_tardiness",
            "eval_task_total_lateness",
            "eval_task_total_tardiness",
            "eval_task_pct_tardy",
            "eval_task_success_rate",

            "eval_num_workflows",
            "eval_num_tardy_workflows",
            "eval_wf_avg_lateness",
            "eval_wf_avg_tardiness",
            "eval_wf_lateness_sum",
            "eval_wf_tardiness_sum",
            "eval_wf_pct_tardy",
            "eval_wf_success_rate",

            "eval_makespan",
        ],
    )

    print(">>> build agent & env_kwargs ...")
    env_kwargs_base, agent = build_agent_and_env_kwargs()

    ret_list, energy_list, makespan_list = [], [], []

    task_avg_late_list, task_avg_tardy_list, task_success_list = [], [], []
    wf_avg_late_list, wf_avg_tardy_list, wf_success_list = [], [], []

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n===== Eval {idx}/{num_runs} | seed={seed} =====")
        set_seed(seed)

        env_kwargs = dict(env_kwargs_base)
        env_kwargs["random_seed"] = int(seed)

        eval_ret, eval_steps, m, _ = evaluate_one_episode(
            CloudWorkflowEnv_PD3QN_Cost,
            env_kwargs,
            agent,
            deterministic=True,
        )

        logger.log(
            run_idx=idx,
            seed=seed,
            eval_return=_safe_float(eval_ret),
            eval_steps=_safe_int(eval_steps),

            eval_total_energy=_safe_float(m["total_energy"]),
            eval_energy_acc=_safe_float(m["energy_acc"]),
            eval_energy_recalc=_safe_float(m["energy_recalc"]),
            eval_energy_recalc_diff=_safe_float(m["energy_recalc_diff"]),

            eval_num_tasks=_safe_int(m["num_tasks"]),
            eval_num_tardy_tasks=_safe_int(m["num_tardy_tasks"]),
            eval_task_avg_lateness=_safe_float(m["task_avg_lateness"]),
            eval_task_avg_tardiness=_safe_float(m["task_avg_tardiness"]),
            eval_task_total_lateness=_safe_float(m["task_total_lateness"]),
            eval_task_total_tardiness=_safe_float(m["task_total_tardiness"]),
            eval_task_pct_tardy=_safe_float(m["task_pct_tardy"]),
            eval_task_success_rate=_safe_float(m["task_success_rate"]),

            eval_num_workflows=_safe_int(m["num_workflows"]),
            eval_num_tardy_workflows=_safe_int(m["num_tardy_workflows"]),
            eval_wf_avg_lateness=_safe_float(m["wf_avg_lateness"]),
            eval_wf_avg_tardiness=_safe_float(m["wf_avg_tardiness"]),
            eval_wf_lateness_sum=_safe_float(m["wf_lateness_sum"]),
            eval_wf_tardiness_sum=_safe_float(m["wf_tardiness_sum"]),
            eval_wf_pct_tardy=_safe_float(m["wf_pct_tardy"]),
            eval_wf_success_rate=_safe_float(m["wf_success_rate"]),

            eval_makespan=_safe_float(m["makespan"]),
        )

        ret_list.append(_safe_float(eval_ret))
        energy_list.append(_safe_float(m["total_energy"]))
        makespan_list.append(_safe_float(m["makespan"]))

        task_avg_late_list.append(_safe_float(m["task_avg_lateness"]))
        task_avg_tardy_list.append(_safe_float(m["task_avg_tardiness"]))
        task_success_list.append(_safe_float(m["task_success_rate"]))

        wf_avg_late_list.append(_safe_float(m["wf_avg_lateness"]))
        wf_avg_tardy_list.append(_safe_float(m["wf_avg_tardiness"]))
        wf_success_list.append(_safe_float(m["wf_success_rate"]))

        print(
            f"[PD3QN] return={_safe_float(eval_ret):.6f} steps={_safe_int(eval_steps)} | "
            f"energy={_safe_float(m['total_energy']):.3f} J | "
            f"task={_safe_int(m['num_tasks'])} tardy={_safe_int(m['num_tardy_tasks'])} | "
            f"task_avg_lateness={_safe_float(m['task_avg_lateness']):.6f} "
            f"task_avg_tardiness={_safe_float(m['task_avg_tardiness']):.6f} "
            f"task_success_rate={_safe_float(m['task_success_rate']):.6f} | "
            f"wf={_safe_int(m['num_workflows'])} tardy={_safe_int(m['num_tardy_workflows'])} | "
            f"wf_avg_lateness={_safe_float(m['wf_avg_lateness']):.6f} "
            f"wf_avg_tardiness={_safe_float(m['wf_avg_tardiness']):.6f} "
            f"wf_success_rate={_safe_float(m['wf_success_rate']):.6f} | "
            f"makespan={_safe_float(m['makespan']):.3f}"
        )

    print("\n================= PD3QN 汇总 =================")
    print(f"avg_return             = {float(np.nanmean(ret_list)):.6f}")
    print(f"avg_energy             = {float(np.nanmean(energy_list)):.3f} J")
    print(f"avg_task_avg_lateness  = {float(np.nanmean(task_avg_late_list)):.6f}")
    print(f"avg_task_avg_tardiness = {float(np.nanmean(task_avg_tardy_list)):.6f}")
    print(f"avg_task_success_rate  = {float(np.nanmean(task_success_list)):.6f}")
    print(f"avg_wf_avg_lateness    = {float(np.nanmean(wf_avg_late_list)):.6f}")
    print(f"avg_wf_avg_tardiness   = {float(np.nanmean(wf_avg_tardy_list)):.6f}")
    print(f"avg_wf_success_rate    = {float(np.nanmean(wf_success_list)):.6f}")
    print(f"avg_makespan           = {float(np.nanmean(makespan_list)):.3f}")
    print("===============================================")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()