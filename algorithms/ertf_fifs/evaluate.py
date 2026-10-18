"""ertf_fifs: eval. Select --task, --resource and --deadline; see --help."""

import os
import sys
import math
import csv
import random
from typing import Dict, List, Tuple
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from common.scenarios import Scenario, parse_scenario

SCENARIO = Scenario(mode="eval")
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
from common.workflow_opt import energy_from_records


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_env_scales(env: CloudWorkflowEnv_VMAgents, env_kwargs: dict):
    for k in [
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    ]:
        if k in env_kwargs:
            setattr(env, k, env_kwargs[k])


def _force_fcfs_manager(env: CloudWorkflowEnv_VMAgents):
    """
    将上层固定为 FCFS 权重 [1,0,0,0,0]。
    为兼容不同版本环境，这里做多重回退。
    """
    fcfs_w = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if hasattr(env, "set_manager_combo") and callable(getattr(env, "set_manager_combo")):
        try:
            env.set_manager_combo(fcfs_w)
            return
        except Exception:
            pass
    if hasattr(env, "w_raw"):
        try:
            env.w_raw = fcfs_w.copy()
        except Exception:
            pass
    if hasattr(env, "combo_weights"):
        try:
            env.combo_weights = fcfs_w.copy()
        except Exception:
            pass
    for attr, value in [
        ("_phase_started", False),
        ("_phase_tasks", []),
        ("phase_tasks", []),
        ("_phase_task_ids", []),
        ("phase_task_ids", []),
    ]:
        if hasattr(env, attr):
            try:
                setattr(env, attr, value.copy() if isinstance(value, list) else value)
            except Exception:
                pass


def _host_id_from_action(env: CloudWorkflowEnv_VMAgents, host_action: int) -> int:
    host_ids = list(getattr(env, "host_ids", list(range(getattr(env, "num_hosts", 0)))))
    if 0 <= int(host_action) < len(host_ids):
        return int(host_ids[int(host_action)])
    return int(host_action)


def _global_vm_indices_of_host(env: CloudWorkflowEnv_VMAgents, host_id: int) -> List[int]:
    vm_ids = list(getattr(env, "vm_ids", []))
    vms = getattr(env, "vms", {})
    out = []
    for gidx, vid in enumerate(vm_ids):
        vm = vms[vid]
        if int(vm.host_id) == int(host_id):
            out.append(gidx)
    return out


def _pick_first_idle_host(env: CloudWorkflowEnv_VMAgents, host_mask: np.ndarray) -> int:
    """
    在合法 host 中选取“最早出现空闲 VM”的 host；若并列，选 action idx 更小者。
    """
    valid = np.where(np.asarray(host_mask, dtype=float) > 0.5)[0].astype(int)
    if valid.size == 0:
        return 0
    best_host_action = int(valid[0])
    best_time = float("inf")
    vm_available_at = np.asarray(getattr(env, "vm_available_at", []), dtype=float)
    for a_host in valid:
        host_id = _host_id_from_action(env, int(a_host))
        gvm_idxs = _global_vm_indices_of_host(env, host_id)
        if len(gvm_idxs) == 0:
            continue
        t0 = float(np.min(vm_available_at[gvm_idxs]))
        if t0 < best_time - 1e-12 or (
            abs(t0 - best_time) <= 1e-12 and int(a_host) < best_host_action
        ):
            best_time = t0
            best_host_action = int(a_host)
    return int(best_host_action)


def _pick_fifs_vm(
    env: CloudWorkflowEnv_VMAgents, vm_mask: np.ndarray, selected_host_action: int
) -> int:
    """Choose earliest-idle VM within the selected host; return its local slot."""
    host_action = int(selected_host_action)
    if not 0 <= host_action < len(env.host_ids):
        raise ValueError(f"Host action out of range: {host_action}")
    host_id = env.host_ids[host_action]
    # Use exactly the slot order consumed by env.vm_assign(), including on 8-VM hosts.
    mapping = env.host_to_vm_indices[host_id]
    mask = np.asarray(vm_mask, dtype=float)
    if mask.ndim != 1 or len(mask) != env.max_vms_per_host or not np.all(np.isfinite(mask)):
        raise ValueError("VM mask must be a finite vector padded to max_vms_per_host")
    if len(mapping) > len(mask):
        raise ValueError("Host VM mapping exceeds the padded mask length")
    valid = np.flatnonzero(mask > 0.5)
    if np.any(valid >= len(mapping)):
        raise ValueError("VM mask marks a padding slot as legal")
    if valid.size == 0:
        raise ValueError("Selected host has no legal VM action")
    available = np.asarray(env.vm_available_at, dtype=float)
    for slot in valid:
        gidx = int(mapping[int(slot)])
        if not 0 <= gidx < len(available) or not np.isfinite(available[gidx]):
            raise ValueError(f"Invalid VM availability mapping for host {host_id}, slot {slot}")
    return int(min(valid, key=lambda slot: (available[mapping[int(slot)]], int(slot))))


def _get_records(env: CloudWorkflowEnv_VMAgents):
    if hasattr(env, "records"):
        return getattr(env, "records")
    if hasattr(env, "_records"):
        return getattr(env, "_records")
    return []


def _compute_metrics(env: CloudWorkflowEnv_VMAgents) -> Dict[str, float]:
    """
    统一从 env 内部状态计算任务级 / 工作流级指标，避免依赖不同版本 env 的 get_episode_metrics。
    """
    task_state = list(getattr(env, "task_state", []))
    task_end_time = list(getattr(env, "task_end_time", []))
    task_baseline_finish = list(getattr(env, "task_baseline_finish", []))
    task_meta = list(getattr(env, "task_meta", []))
    workflows = list(getattr(env, "workflows", []))
    task_lateness_sum = 0.0
    task_tardiness_sum = 0.0
    task_finished = 0
    task_tardy_cnt = 0
    wf_finish_time: Dict[int, float] = {}
    for tid in range(len(task_state)):
        if task_state[tid] != "Finished":
            continue
        finish_t = float(task_end_time[tid]) if tid < len(task_end_time) else 0.0
        ddl_t = float(task_baseline_finish[tid]) if tid < len(task_baseline_finish) else finish_t
        late = finish_t - ddl_t
        tard = max(0.0, late)
        task_lateness_sum += late
        task_tardiness_sum += tard
        task_finished += 1
        if tard > 1e-12:
            task_tardy_cnt += 1
        if tid < len(task_meta):
            wf_id, _ = task_meta[tid]
            wf_id = int(wf_id)
            wf_finish_time[wf_id] = max(wf_finish_time.get(wf_id, 0.0), finish_t)
    task_avg_lateness = task_lateness_sum / max(task_finished, 1)
    task_avg_tardiness = task_tardiness_sum / max(task_finished, 1)
    task_pct_tardy = task_tardy_cnt / max(task_finished, 1)
    task_avg_success = 1.0 - task_pct_tardy
    wf_lateness_sum = 0.0
    wf_tardiness_sum = 0.0
    wf_cnt = 0
    wf_tardy_cnt = 0
    for wf_id, wf in enumerate(workflows):
        finish_t = float(wf_finish_time.get(wf_id, 0.0))
        ddl_t = float(getattr(wf, "deadline", finish_t))
        late = finish_t - ddl_t
        tard = max(0.0, late)
        wf_lateness_sum += late
        wf_tardiness_sum += tard
        wf_cnt += 1
        if tard > 1e-12:
            wf_tardy_cnt += 1
    wf_avg_lateness = wf_lateness_sum / max(wf_cnt, 1)
    wf_avg_tardiness = wf_tardiness_sum / max(wf_cnt, 1)
    wf_pct_tardy = wf_tardy_cnt / max(wf_cnt, 1)
    wf_avg_success = 1.0 - wf_pct_tardy
    makespan = float(getattr(env, "current_time", 0.0))
    total_energy = float(getattr(env, "total_energy", 0.0))
    recs = _get_records(env)
    if len(recs) > 0 and hasattr(env, "hosts"):
        try:
            energy_check = float(energy_from_records(
                recs, env.hosts,
                start_time=float(env.arrival_times[0]) if len(env.arrival_times) else 0.0,
                end_time=float(env.current_time),
            ))
        except Exception:
            energy_check = float("nan")
    else:
        energy_check = float("nan")
    return {
        "total_energy": total_energy,
        "energy_check": energy_check,
        "makespan": makespan,
        "task_avg_lateness": float(task_avg_lateness),
        "task_avg_tardiness": float(task_avg_tardiness),
        "task_pct_tardy": float(task_pct_tardy),
        "task_avg_success": float(task_avg_success),
        "wf_avg_lateness": float(wf_avg_lateness),
        "wf_avg_tardiness": float(wf_avg_tardiness),
        "wf_pct_tardy": float(wf_pct_tardy),
        "wf_avg_success": float(wf_avg_success),
        "task_finished": int(task_finished),
        "wf_finished": int(getattr(env, "completed_workflows", len(workflows))),
    }


def run_one_episode(env_cls, env_kwargs: dict, seed: int) -> Dict[str, float]:
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }
    ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}
    ctor_kwargs["random_seed"] = int(seed)
    env = env_cls(**ctor_kwargs)
    _ = env.reset()
    _sync_env_scales(env, env_kwargs)
    _force_fcfs_manager(env)
    done = bool(getattr(env, "done_flag", False))
    phase_cnt = 0
    vm_phase_reward_sum = 0.0
    host_phase_reward_sum = 0.0
    mgr_phase_reward_sum = 0.0
    while not done:
        host_rewards = []
        vm_rewards = []
        while True:
            st_host, has_next = env.get_host_state_for_next_assignment()
            if not has_next:
                break
            host_mask = np.asarray(st_host["mask"], dtype=np.float32)
            a_host = _pick_first_idle_host(env, host_mask)
            env.host_select(int(a_host))
            st_vm, ok_vm = env.get_vm_state_for_current_task()
            if not ok_vm:
                break
            vm_mask = np.asarray(st_vm["mask"], dtype=np.float32)
            a_vm = _pick_fifs_vm(env, vm_mask, selected_host_action=int(a_host))
            r_host, r_vm, _info_task = env.vm_assign(int(a_vm))
            host_rewards.append(float(r_host))
            vm_rewards.append(float(r_vm))
        r_manager_raw, _pinfo = env.finish_phase_and_advance()
        mgr_phase_reward_sum += float(r_manager_raw)
        host_phase_reward_sum += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
        vm_phase_reward_sum += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
        phase_cnt += 1
        done = bool(getattr(env, "done_flag", False))
        if done:
            break
        _force_fcfs_manager(env)
    metrics = _compute_metrics(env)
    metrics.update(
        {
            "avg_vm_reward": vm_phase_reward_sum / max(phase_cnt, 1),
            "avg_host_reward": host_phase_reward_sum / max(phase_cnt, 1),
            "avg_manager_reward": mgr_phase_reward_sum / max(phase_cnt, 1),
            "phase_count": int(phase_cnt),
            "seed": int(seed),
        }
    )
    return metrics


def _mean_std(rows: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [float(r[key]) for r in rows if key in r and (not math.isnan(float(r[key])))]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.mean(vals)), float(np.std(vals)))


def save_csv(rows: List[Dict[str, float]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if len(rows) == 0:
        return
    fieldnames = [
        "seed",
        "total_energy",
        "energy_check",
        "makespan",
        "avg_vm_reward",
        "avg_host_reward",
        "avg_manager_reward",
        "task_avg_lateness",
        "task_avg_tardiness",
        "task_pct_tardy",
        "task_avg_success",
        "wf_avg_lateness",
        "wf_avg_tardiness",
        "wf_pct_tardy",
        "wf_avg_success",
        "task_finished",
        "wf_finished",
        "phase_count",
    ]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    dax_dir = os.path.join(ROOT_DIR, "data/dax")
    dax_list = SCENARIO.dax_paths(dax_dir)
    horizon = 1000000000.0
    arrival_lambda = 0.03
    random_seed = SCENARIO.seeds[0]
    workflows_per_episode = 50
    max_ready_tasks = "auto"
    normalize_obs = True
    energy_reward_scale = 0.001
    task_baseline_norm = 300.0
    energy_norm_per_mi_ref = 20.0
    alpha_delay_host = 0.75
    alpha_delay_vm = 0.75
    deadline_alpha_small = 2.0
    deadline_alpha_large = 3.0
    deadline_alpha_small_prob = SCENARIO.small_probability
    deadline_cache_path = str(SCENARIO.cache_path)
    if not os.path.exists(deadline_cache_path):
        raise FileNotFoundError(
            f"deadline cache 不存在：{deadline_cache_path}\n请先确保 hrl新训练代码使用的 cache 文件已生成。"
        )
    save_csv_path = os.path.join(
        ROOT_DIR,
        "outputs",
        "run",
        SCENARIO.output_version,
        "fcfs_fifs_ddlMix",
        f"eval_task{SCENARIO.task}_res{SCENARIO.resource}_{SCENARIO.deadline}.csv",
    )
    eval_seeds = list(SCENARIO.seeds)
    set_seed(SCENARIO.model_seed)
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        max_ready_tasks=max_ready_tasks,
        normalize=normalize_obs,
        workflows_per_episode=workflows_per_episode,
        num_hosts=SCENARIO.num_hosts,
        vms_per_host=SCENARIO.vms_per_host,
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        deadline_mode="cache_fcfs",
        deadline_cache_path=deadline_cache_path,
        deadline_cache_strict=True,
        deadline_alpha_small=deadline_alpha_small,
        deadline_alpha_large=deadline_alpha_large,
        deadline_alpha_small_prob=SCENARIO.small_probability,
        manager_alpha_delay=0.75,
        manager_delay_mode="tardiness",
        energy_reward_scale=energy_reward_scale,
        task_baseline_norm=task_baseline_norm,
        energy_norm_per_mi_ref=energy_norm_per_mi_ref,
        alpha_delay_host=alpha_delay_host,
        alpha_delay_vm=alpha_delay_vm,
    )
    print("=" * 100)
    print("FCFS-FIFS baseline on hrl新训练代码 environment")
    print(f"eval_seeds = {eval_seeds[0]}..{eval_seeds[-1]}  (count={len(eval_seeds)})")
    print(f"deadline_mode = cache_fcfs")
    print(
        f"ddlAlpha mix: P({deadline_alpha_small})={deadline_alpha_small_prob:.2f}, P({deadline_alpha_large})={1.0 - deadline_alpha_small_prob:.2f}"
    )
    print("policy: task=FCFS, host=First-Idle Host, vm=FIFS")
    print("=" * 100)
    rows = []
    for sd in eval_seeds:
        metrics = run_one_episode(CloudWorkflowEnv_VMAgents, env_kwargs, sd)
        rows.append(metrics)
        e0 = metrics["total_energy"]
        e1 = metrics["energy_check"]
        delta_e = abs(e0 - e1) if not math.isnan(e1) else float("nan")
        print(
            f"[seed={sd:02d}] Energy={metrics['total_energy']:.3f} | EnergyCheck={metrics['energy_check']:.3f} | |Δ|={delta_e:.6f} | Makespan={metrics['makespan']:.3f} | TaskLate={metrics['task_avg_lateness']:.3f} | TaskTard={metrics['task_avg_tardiness']:.3f} | TaskSucc={metrics['task_avg_success']:.3f} | WFLate={metrics['wf_avg_lateness']:.3f} | WFTard={metrics['wf_avg_tardiness']:.3f} | WFSucc={metrics['wf_avg_success']:.3f}"
        )
    save_csv(rows, save_csv_path)
    print("\n" + "-" * 100)
    for k in [
        "total_energy",
        "makespan",
        "avg_vm_reward",
        "avg_host_reward",
        "avg_manager_reward",
        "task_avg_lateness",
        "task_avg_tardiness",
        "task_pct_tardy",
        "task_avg_success",
        "wf_avg_lateness",
        "wf_avg_tardiness",
        "wf_pct_tardy",
        "wf_avg_success",
    ]:
        mu, sd = _mean_std(rows, k)
        print(f"{k:>20s} : mean={mu:.6f}  std={sd:.6f}")
    print("-" * 100)
    print(f"CSV saved to: {save_csv_path}")


if __name__ == "__main__":
    SCENARIO = parse_scenario("eval", "ertf_fifs")
    main()
