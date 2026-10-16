# -*- coding: utf-8 -*-
"""
precompute_ddl_fcfs_ss.py

离线预计算：在“动态环境”中用基线策略跑完每个 episode（给定 seed），得到每个 workflow 的 makespan：
- 任务选择：FCFS（ready_time 最早；tie 用 task_id）
- VM 分配：先空闲先服务（idle VM 中按 (vm_available_at, vm_idx) 最小；tie 用 idx）

输出一个 JSON cache：
{
  "meta": {...环境关键参数...},
  "data": [
     {"seed": 1, "wf_makespans":[...], "wf_dax_names":[...], "arrival_times":[...]},
     ...
  ]
}
"""

import os
import json
import numpy as np

from b_hrl_legacy_fcfs.env_fcfs import (
    CloudWorkflowEnv_VMAgents,
)


def _pick_fcfs_task(env) -> int:
    # FCFS：ready_time 最小，tie 用 tid
    return int(min(env.ready_task_ids, key=lambda tid: (env.task_ready_time[tid], tid)))


def _pick_vm_first_idle(env) -> int:
    idle = np.where(env._vm_idle_mask())[0]
    # 先空闲先服务：按 (vm_available_at, vm_idx) 最小
    return int(min(idle, key=lambda j: (env.vm_available_at[j], j)))


def run_baseline_episode_fcfs_vmfirst(env: CloudWorkflowEnv_VMAgents):
    """
    在当前 env（已 reset）上跑 baseline，直到 done_flag=True
    """
    # 先推进到第一个 decision point（或结束）
    env._advance_until_decision_energy_only()

    while not bool(getattr(env, "done_flag", False)):
        # 在同一时刻尽可能分配（只要还有 ready 且有 idle VM）
        while env._has_decision_point():
            tid = _pick_fcfs_task(env)
            vm_idx = _pick_vm_first_idle(env)
            env._assign_task_to_specific_vm(tid, vm_idx)

        # 推进到下一次 decision / 或结束
        env._advance_until_decision_energy_only()

    # episode 完成后，统计 makespan
    wf_makespans = []
    wf_dax_names = []
    for wf_id in range(int(env.workflows_per_episode)):
        wf = env.workflows[wf_id]
        finish_t = float(env.wf_finish_time.get(wf_id, env.current_time))
        makespan = max(0.0, finish_t - float(wf.arrival_time))
        wf_makespans.append(float(makespan))
        wf_dax_names.append(os.path.basename(getattr(wf, "dax_path", "")))

    arrival_times = [float(x) for x in env.arrival_times]
    return wf_makespans, wf_dax_names, arrival_times


def main():
    # ========== 你只需要改这里：与训练 env_kwargs 完全一致 ==========
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    dax_dir = os.path.join(ROOT_DIR, "a_dax")
    dax_list = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "LIGO_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    # 与训练保持一致
    horizon = 1e9
    arrival_lambda = 0.03
    WORKFLOWS_PER_EPISODE = 50

    # resS：3 Host / 25 VM
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=0,  # 这里会在每个seed循环里覆盖
        max_ready_tasks="auto",
        normalize=True,
        workflows_per_episode=WORKFLOWS_PER_EPISODE,
        num_hosts=9,
        vms_per_host=(9, 9, 9, 8, 8, 8, 8, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        # 关键：预计算阶段不用 cache
        deadline_mode="none",
        deadline_cache_path=None,
        deadline_alpha=1.5,
        deadline_cache_strict=True,
    )

    # 预计算 seeds（按你需要改范围）
    # SEEDS = list(range(1, 31))  # 示例：先做 30 个 seed
    SEEDS = list(range(0, 1001))  # 示例：1000 个 seed

    out_dir = os.path.join(ROOT_DIR, "d_output", "deadlines")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ddl_fcfs_taskS_resL_seed0-1000.json")

    data = []

    for sd in SEEDS:
        print(f"[precompute] seed={sd} ...")

        env_kwargs["random_seed"] = int(sd)
        env = CloudWorkflowEnv_VMAgents(**env_kwargs)

        # reset 会自动把 self.rng 绑定到 random_seed（见 env reset override）
        env.reset()

        wf_makespans, wf_dax_names, arrival_times = run_baseline_episode_fcfs_vmfirst(env)

        data.append(
            {
                "seed": int(sd),
                "wf_makespans": wf_makespans,
                "wf_dax_names": wf_dax_names,
                "arrival_times": arrival_times,
            }
        )

        avg_ms = float(np.mean(wf_makespans)) if len(wf_makespans) > 0 else 0.0
        print(f"  done. avg_makespan={avg_ms:.6f}s")

    obj = {
        "meta": {
            "policy": "FCFS_task + VM_first_idle",
            "alpha": float(env_kwargs["deadline_alpha"]),
            "arrival_lambda": float(arrival_lambda),
            "workflows_per_episode": int(WORKFLOWS_PER_EPISODE),
            "num_hosts": int(env_kwargs["num_hosts"]),
            "vms_per_host": list(env_kwargs["vms_per_host"]),
            "pc_tiers": list(env_kwargs["pc_tiers"]),
            "bw_tiers": list(env_kwargs["bw_tiers"]),
            "dax_paths": [os.path.basename(p) for p in dax_list],
        },
        "data": data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cache: {out_path}")


if __name__ == "__main__":
    main()
