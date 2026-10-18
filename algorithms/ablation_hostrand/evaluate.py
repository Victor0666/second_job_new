"""ablation_hostrand: eval. Select --task, --resource and --deadline; see --help."""

import os
import sys
import random
import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
from common.scenarios import Scenario, parse_scenario
from common.f18_state_normalization import bind_normalization, check_checkpoint
from common.f11 import LEARNING_MODEL
from common.ta_weights import TA_WEIGHT_MODEL

SCENARIO = Scenario(mode="eval")
from common.metrics_logger import CSVLogger
from common.workflow_opt import energy_from_records
from common.d3qn_agent_EXDehid import D3QNAgent
from environments.mahdrl_fcfs import (
    CloudWorkflowEnv_VMAgents as EnvHOSTRAND,
    MANAGER_ACTION_TABLE,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


NUM_OPTIONS = int(MANAGER_ACTION_TABLE.shape[0])


def manager_apply_action(env: EnvHOSTRAND, act_idx: int):
    delta = MANAGER_ACTION_TABLE[int(act_idx)]
    env.apply_manager_delta(delta)


def _sync_env_scales(env: EnvHOSTRAND, env_kwargs: dict):
    env.energy_reward_scale = float(env_kwargs["energy_reward_scale"])
    env.task_baseline_norm = float(env_kwargs["task_baseline_norm"])
    env.energy_norm_per_mi_ref = float(env_kwargs["energy_norm_per_mi_ref"])
    env.alpha_delay_host = float(env_kwargs["alpha_delay_host"])
    env.alpha_delay_vm = float(env_kwargs["alpha_delay_vm"])


class RandomHostSelector:
    """
    中层消融：不学习 host，只在合法 mask 内随机选一个 host。
    """

    def __init__(self, base_seed: int = 0):
        self.base_seed = int(base_seed)
        self.rs = np.random.RandomState(self.base_seed)

    def reset(self, episode_seed: int):
        local_seed = int(episode_seed) * 1000003 + 20260325 + self.base_seed * 97 & 4294967295
        self.rs = np.random.RandomState(local_seed)

    def select_action(self, mask, deterministic=True, count_step=False):
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        valid = np.flatnonzero(mask > 0.5)
        if valid.size == 0:
            return 0
        idx = int(self.rs.randint(valid.size))
        return int(valid[idx])


def build_env_kwargs_hostrand(random_seed=2):
    dax_dir = os.path.join(PROJECT_DIR, "data/dax")
    dax_list = SCENARIO.dax_paths(dax_dir)
    deadline_cache_path = str(SCENARIO.cache_path)
    if not os.path.exists(deadline_cache_path):
        alt = str(SCENARIO.cache_path)
        if alt is None:
            raise FileNotFoundError(
                f"deadline cache 不存在：{deadline_cache_path}\n也未在 data/deadlines/ 下找到可用的 deadline_cache_fcfs_vmfirst*.json"
            )
        deadline_cache_path = str(SCENARIO.cache_path)
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=1000000000.0,
        arrival_lambda=0.03,
        random_seed=int(random_seed),
        max_ready_tasks="auto",
        normalize=True,
        workflows_per_episode=50,
        num_hosts=SCENARIO.num_hosts,
        vms_per_host=SCENARIO.vms_per_host,
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        deadline_mode="cache_fcfs",
        deadline_cache_path=deadline_cache_path,
        deadline_cache_strict=True,
        deadline_alpha_small=2.0,
        deadline_alpha_large=3.0,
        deadline_alpha_small_prob=SCENARIO.small_probability,
        manager_alpha_delay=0.75,
        manager_delay_mode="tardiness",
        energy_reward_scale=0.001,
        task_baseline_norm=300.0,
        energy_norm_per_mi_ref=20.0,
        alpha_delay_host=0.75,
        alpha_delay_vm=0.75,
    )
    env_kwargs.update(SCENARIO.normalization_kwargs())
    return env_kwargs


def compute_task_workflow_metrics(env: EnvHOSTRAND):
    """
    task 级：
      lateness  = finish - deadline，可为负
      tardiness = max(lateness, 0)

    workflow 级：
      1) 保留原先按 task 聚合得到的 workflow lateness / tardiness
      2) 新增 wf_success_rate：
         若 workflow 的实际完成时间 <= workflow.deadline，则认为成功
    """
    task_total_lateness = 0.0
    task_total_tardiness = 0.0
    finished_tasks = 0
    tardy_tasks = 0
    wf_lateness_sum = {}
    wf_tardiness_sum = {}
    task_state = getattr(env, "task_state", [])
    baseline = getattr(env, "task_baseline_finish", [])
    task_end_time = getattr(env, "task_end_time", [])
    task_meta = getattr(env, "task_meta", [])
    n_all = len(task_state)
    for tid in range(n_all):
        if task_state[tid] != "Finished":
            continue
        if tid >= len(baseline):
            continue
        if tid >= len(task_end_time):
            continue
        finish_t = float(task_end_time[tid])
        deadline_t = float(baseline[tid])
        lateness = finish_t - deadline_t
        tardiness = max(lateness, 0.0)
        task_total_lateness += lateness
        task_total_tardiness += tardiness
        finished_tasks += 1
        if tardiness > 0.0:
            tardy_tasks += 1
        if tid < len(task_meta) and task_meta[tid] is not None:
            wf_id, _local_id = task_meta[tid]
        else:
            wf_id = -1
        wf_lateness_sum[wf_id] = wf_lateness_sum.get(wf_id, 0.0) + lateness
        wf_tardiness_sum[wf_id] = wf_tardiness_sum.get(wf_id, 0.0) + tardiness
    task_avg_lateness = task_total_lateness / max(finished_tasks, 1)
    task_avg_tardiness = task_total_tardiness / max(finished_tasks, 1)
    task_pct_tardy = float(tardy_tasks) / float(max(finished_tasks, 1))
    target_workflows = int(getattr(env, "workflows_per_episode", 0) or 0)
    completed_workflows = int(getattr(env, "completed_workflows", 0) or 0)
    if target_workflows > 0:
        wf_ids = list(range(target_workflows))
        wf_total_lateness = sum((wf_lateness_sum.get(wf_id, 0.0) for wf_id in wf_ids))
        wf_total_tardiness = sum((wf_tardiness_sum.get(wf_id, 0.0) for wf_id in wf_ids))
        tardy_wf = sum((1 for wf_id in wf_ids if wf_tardiness_sum.get(wf_id, 0.0) > 0.0))
        wf_avg_lateness = wf_total_lateness / max(target_workflows, 1)
        wf_avg_tardiness = wf_total_tardiness / max(target_workflows, 1)
        wf_pct_tardy = float(tardy_wf) / float(max(target_workflows, 1))
    else:
        wf_ids = sorted(wf_lateness_sum.keys())
        wf_cnt = len(wf_ids)
        wf_total_lateness = sum(wf_lateness_sum.values()) if wf_cnt > 0 else 0.0
        wf_total_tardiness = sum(wf_tardiness_sum.values()) if wf_cnt > 0 else 0.0
        tardy_wf = sum((1 for wf_id in wf_ids if wf_tardiness_sum.get(wf_id, 0.0) > 0.0))
        wf_avg_lateness = wf_total_lateness / max(wf_cnt, 1)
        wf_avg_tardiness = wf_total_tardiness / max(wf_cnt, 1)
        wf_pct_tardy = float(tardy_wf) / float(max(wf_cnt, 1))
    workflows = getattr(env, "workflows", [])
    wf_finish_time = getattr(env, "wf_finish_time", {})
    if target_workflows > 0:
        wf_success_count = 0
        for wf_id in range(target_workflows):
            if wf_id >= len(workflows):
                continue
            wf = workflows[wf_id]
            ddl = float(getattr(wf, "deadline", np.inf))
            finish_t = wf_finish_time.get(wf_id, None)
            if finish_t is not None and float(finish_t) <= ddl:
                wf_success_count += 1
        wf_success_rate = float(wf_success_count) / float(max(target_workflows, 1))
    else:
        wf_cnt = len(workflows)
        wf_success_count = 0
        for wf_id, wf in enumerate(workflows):
            ddl = float(getattr(wf, "deadline", np.inf))
            finish_t = wf_finish_time.get(wf_id, None)
            if finish_t is not None and float(finish_t) <= ddl:
                wf_success_count += 1
        wf_success_rate = float(wf_success_count) / float(max(wf_cnt, 1))
    makespan = float(getattr(env, "current_time", 0.0))
    return {
        "finished_tasks": int(finished_tasks),
        "completed_workflows": int(completed_workflows),
        "target_workflows": int(target_workflows),
        "task_total_lateness": float(task_total_lateness),
        "task_avg_lateness": float(task_avg_lateness),
        "task_total_tardiness": float(task_total_tardiness),
        "task_avg_tardiness": float(task_avg_tardiness),
        "task_pct_tardy": float(task_pct_tardy),
        "wf_total_lateness": float(wf_total_lateness),
        "wf_avg_lateness": float(wf_avg_lateness),
        "wf_total_tardiness": float(wf_total_tardiness),
        "wf_avg_tardiness": float(wf_avg_tardiness),
        "wf_pct_tardy": float(wf_pct_tardy),
        "wf_success_count": int(wf_success_count),
        "wf_success_rate": float(wf_success_rate),
        "makespan": float(makespan),
    }


def evaluate_hostrand_one_seed(env_cls, env_kwargs, vm_agent, manager_agent):
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }
    ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}
    env = env_cls(**ctor_kwargs)
    _sync_env_scales(env, env_kwargs)
    _obs0, info0 = env.reset()
    reset_t = float(getattr(env, "current_time", 0.0))
    reset_e = float(getattr(env, "total_energy", 0.0))
    first_arrival_t = float(
        getattr(env, "first_arrival_time", info0.get("first_arrival_time", 0.0))
    )
    host_selector = RandomHostSelector(base_seed=123456)
    host_selector.reset(int(env_kwargs.get("random_seed", 0)))
    sH = env.get_manager_state()
    m_mask = env.get_manager_action_mask()
    m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
    manager_apply_action(env, m_act)
    done = bool(getattr(env, "done_flag", False))
    phases = 0
    ret_mgr = 0.0
    ret_vm_phase_mean = 0.0
    ret_host_phase_mean = 0.0
    while not done:
        vm_rewards = []
        host_rewards = []
        while True:
            st_host, has_next = env.get_host_state_for_next_assignment()
            if not has_next:
                break
            a_host = host_selector.select_action(
                st_host["mask"], deterministic=True, count_step=False
            )
            env.host_select(int(a_host))
            st_vm, ok_vm = env.get_vm_state_for_current_task()
            if not ok_vm:
                break
            a_vm = vm_agent.select_action(
                st_vm["obs"], st_vm["mask"], deterministic=True, count_step=False
            )
            r_host, r_vm, _info_task = env.vm_assign(int(a_vm))
            host_rewards.append(float(r_host))
            vm_rewards.append(float(r_vm))
        r_manager_raw, _pinfo = env.finish_phase_and_advance()
        ret_mgr += float(r_manager_raw)
        ret_vm_phase_mean += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
        ret_host_phase_mean += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
        phases += 1
        done = bool(getattr(env, "done_flag", False))
        if done:
            break
        sH = env.get_manager_state()
        m_mask = env.get_manager_action_mask()
        m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
        manager_apply_action(env, m_act)
    avg_vm = ret_vm_phase_mean / max(phases, 1)
    avg_host = ret_host_phase_mean / max(phases, 1)
    avg_mgr = ret_mgr / max(phases, 1)
    eval_energy = float(env.total_energy)
    recs = getattr(env, "_records", getattr(env, "records", []))
    energy_recalc = float(energy_from_records(
        recs, env.hosts,
        start_time=float(env.arrival_times[0]) if len(env.arrival_times) else 0.0,
        end_time=float(env.current_time),
    ))
    energy_diff = float(eval_energy - energy_recalc)
    metrics = compute_task_workflow_metrics(env)
    metrics.update(
        {
            "eval_vm_reward": float(avg_vm),
            "eval_host_reward": float(avg_host),
            "eval_mgr_reward": float(avg_mgr),
            "eval_energy": float(eval_energy),
            "energy_recalc": float(energy_recalc),
            "energy_diff": float(energy_diff),
            "reset_t": float(reset_t),
            "reset_e": float(reset_e),
            "first_arrival_t": float(first_arrival_t),
        }
    )
    return metrics


def build_agents_and_load(env_kwargs: dict):
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }
    ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}
    tmp_env = EnvHOSTRAND(**ctor_kwargs)
    _sync_env_scales(tmp_env, env_kwargs)
    tmp_env.reset()
    st_host, ok = tmp_env.get_host_state_for_next_assignment()
    while not ok and (not tmp_env.done_flag):
        _rM, _ = tmp_env.finish_phase_and_advance()
        st_host, ok = tmp_env.get_host_state_for_next_assignment()
    host_act_dim = int(st_host["mask"].shape[0])
    probe_host_selector = RandomHostSelector(base_seed=999)
    probe_host_selector.reset(int(tmp_env.random_seed))
    a_host0 = probe_host_selector.select_action(
        st_host["mask"], deterministic=True, count_step=False
    )
    tmp_env.host_select(int(a_host0))
    st_vm, ok_vm = tmp_env.get_vm_state_for_current_task()
    if not ok_vm:
        raise RuntimeError(
            "Probe vm_state failed: get_vm_state_for_current_task() returned ok_vm=False"
        )
    vm_state_dim = int(st_vm["obs"].shape[0])
    vm_act_dim = int(st_vm["mask"].shape[0])
    mgr_state_dim = int(tmp_env.get_manager_state().shape[0])
    print(
        f"[HOSTRAND probe] host_act_dim={host_act_dim} | vm_obs_dim={vm_state_dim}, vm_act_dim={vm_act_dim} | mgr_obs_dim={mgr_state_dim}, mgr_act_dim={NUM_OPTIONS}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm_agent = D3QNAgent(
        input_dim=vm_state_dim,
        output_dim=vm_act_dim,
        lr=0.0003,
        gamma=0.95,
        batch_size=256,
        buffer_size=120000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=80000,
        target_update_tau=0.005,
        grad_clip=10.0,
        hidden_dims=(1024, 1024, 512, 512, 256),
        device=device,
    )
    bind_normalization(vm_agent, SCENARIO)
    manager_agent = D3QNAgent(
        input_dim=mgr_state_dim,
        output_dim=NUM_OPTIONS,
        lr=0.0003,
        gamma=0.95,
        batch_size=256,
        buffer_size=60000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=40000,
        target_update_tau=0.01,
        grad_clip=10.0,
        hidden_dims=(512, 512, 256, 128),
        device=device,
    )
    bind_normalization(manager_agent, SCENARIO)
    ckpt_dir_default = str(SCENARIO.checkpoint_dir("ablation_hostrand"))
    ckpt_vm = os.path.join(ckpt_dir_default, "best_vm.pth")
    ckpt_mgr = os.path.join(ckpt_dir_default, "best_manager.pth")
    if not (os.path.exists(ckpt_vm) and os.path.exists(ckpt_mgr)):
        raise FileNotFoundError(
            f"Missing model files under {SCENARIO.checkpoint_dir('ablation_hostrand')}; train this scenario or use --checkpoint-dir"
        )
    print(f"[HOSTRAND load] ckpt_dir: {ckpt_dir_default}")
    print(f"[HOSTRAND load] vm     : {ckpt_vm}")
    vm_agent.load(ckpt_vm, require_learning_model=True)
    print(f"[HOSTRAND load] manager: {ckpt_mgr}")
    manager_agent.load(ckpt_mgr, require_learning_model=True)
    vm_agent.online.eval()
    vm_agent.target.eval()
    manager_agent.online.eval()
    manager_agent.target.eval()
    return (vm_agent, manager_agent)


def main():
    seeds = list(SCENARIO.seeds)
    env_kwargs_base = build_env_kwargs_hostrand(random_seed=seeds[0])
    result_dir = os.path.join(PROJECT_DIR, "outputs", "run", SCENARIO.normalization_mode, LEARNING_MODEL, TA_WEIGHT_MODEL, "hrl_hostrand_ddlMix")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(
        result_dir,
        f"run_hrl_HOSTRAND_task{SCENARIO.task}_res{SCENARIO.resource}_{len(SCENARIO.seeds)}seeds_ddlMix_{SCENARIO.deadline}.csv",
    )
    logger = CSVLogger(
        filepath=csv_path,
        fieldnames=[
            "run_idx",
            "seed",
            "eval_vm_reward",
            "eval_host_reward",
            "eval_mgr_reward",
            "eval_energy",
            "energy_recalc",
            "energy_diff",
            "finished_tasks",
            "completed_workflows",
            "target_workflows",
            "task_total_lateness",
            "task_avg_lateness",
            "task_total_tardiness",
            "task_avg_tardiness",
            "task_pct_tardy",
            "wf_total_lateness",
            "wf_avg_lateness",
            "wf_total_tardiness",
            "wf_avg_tardiness",
            "wf_pct_tardy",
            "wf_success_count",
            "wf_success_rate",
            "makespan",
        ],
    )
    print(">>> 构建 agent 并加载 HOSTRAND best_ckpt ...")
    vm_agent, manager_agent = build_agents_and_load(env_kwargs_base)
    all_rows = []
    for idx, sd in enumerate(seeds):
        print(f"\n===== Eval {idx}/{len(seeds) - 1} | seed={sd} =====")
        set_seed(sd)
        kwargs_i = dict(env_kwargs_base)
        kwargs_i["random_seed"] = int(sd)
        row = evaluate_hostrand_one_seed(EnvHOSTRAND, kwargs_i, vm_agent, manager_agent)
        row.update({"run_idx": int(idx), "seed": int(sd)})
        all_rows.append(row)
        logger.log(**row)
        print(
            f"[ResetCheck] t_now={row['reset_t']} E0={row['reset_e']} t_first_arr={row['first_arrival_t']}"
        )
        print(
            f"[EnergyCheck] E_acc={row['eval_energy']:.6f}  E_re={row['energy_recalc']:.6f}  diff={row['energy_diff']:.6f}"
        )
        print(
            f"[HRL-HOSTRAND] vm={row['eval_vm_reward']:.6f} host={row['eval_host_reward']:.6f} mgr={row['eval_mgr_reward']:.6f} energy={row['eval_energy']:.3f} J | tasks={row['finished_tasks']} | task_avg_lateness={row['task_avg_lateness']:.6f} task_avg_tardiness={row['task_avg_tardiness']:.6f} task_pct_tardy={row['task_pct_tardy']:.6f} | wf={row['completed_workflows']}/{row['target_workflows']} wf_avg_lateness={row['wf_avg_lateness']:.6f} wf_avg_tardiness={row['wf_avg_tardiness']:.6f} wf_pct_tardy={row['wf_pct_tardy']:.6f} wf_success_rate={row['wf_success_rate']:.6f} | makespan={row['makespan']:.3f}"
        )
        print(
            f"[Detail] task_total_lateness={row['task_total_lateness']:.6f} task_total_tardiness={row['task_total_tardiness']:.6f} | wf_total_lateness={row['wf_total_lateness']:.6f} wf_total_tardiness={row['wf_total_tardiness']:.6f} | wf_success_count={row['wf_success_count']}"
        )

    def avg_of(key):
        return float(np.mean([r[key] for r in all_rows])) if len(all_rows) > 0 else 0.0

    print("\n================= HRL-HOSTRAND（30 seeds）汇总 =================")
    print(f"avg_vm_reward        = {avg_of('eval_vm_reward'):.6f}")
    print(f"avg_host_reward      = {avg_of('eval_host_reward'):.6f}")
    print(f"avg_mgr_reward       = {avg_of('eval_mgr_reward'):.6f}")
    print(f"avg_energy           = {avg_of('eval_energy'):.3f} J")
    print(f"avg_energy_recalc    = {avg_of('energy_recalc'):.3f} J")
    print(f"avg_energy_diff      = {avg_of('energy_diff'):.6f}")
    print(f"avg_finished_tasks   = {avg_of('finished_tasks'):.3f}")
    print(f"avg_completed_wf     = {avg_of('completed_workflows'):.3f}")
    print(f"avg_target_wf        = {avg_of('target_workflows'):.3f}")
    print(f"avg_task_lateness    = {avg_of('task_avg_lateness'):.6f}")
    print(f"avg_task_tardiness   = {avg_of('task_avg_tardiness'):.6f}")
    print(f"avg_task_pct_tardy   = {avg_of('task_pct_tardy'):.6f}")
    print(f"avg_wf_lateness      = {avg_of('wf_avg_lateness'):.6f}")
    print(f"avg_wf_tardiness     = {avg_of('wf_avg_tardiness'):.6f}")
    print(f"avg_wf_pct_tardy     = {avg_of('wf_pct_tardy'):.6f}")
    print(f"avg_wf_success_rate  = {avg_of('wf_success_rate'):.6f}")
    print(f"avg_makespan         = {avg_of('makespan'):.3f}")
    print("==============================================================")
    print(f"结果已写入 CSV: {csv_path}")


if __name__ == "__main__":
    SCENARIO = parse_scenario("eval", "ablation_hostrand")
    main()
