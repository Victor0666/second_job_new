# -*- coding: utf-8 -*-
"""
train_hrl_tri_003_alpha15_taskS_resS_seed5_ddlFCFS_alpha075_ddlMix_Loose_MGRRAND.py

三层 HRL 上层消融训练脚本：

- Manager：不再使用智能体学习
           每个 phase 开始时，在合法 mask 内“随机”选择一个 Δw 动作
- HostAgent：逐任务选 host（D3QN，保持不变）
- VMAgent：逐任务在选定 host 内选空闲 VM 槽位（D3QN，保持不变）

说明：
1) 上层 reward 仍由环境 finish_phase_and_advance() 返回；
   即“上层不学习”，但“上层 reward 仍统计、仍记录”。

2) 为了不污染 env.rng（环境内部还要用于 workflow 模板抽样等），
   Manager 的随机动作使用独立 RandomState，并由 episode_seed 派生，
   从而保证可复现与公平性。

3) 中层/下层训练逻辑、reward 口径、环境配置保持不变。
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from b_hrl.d3qn_agent_EXDehid import D3QNAgent

from b_mahdrl.env_fcfs import (
    CloudWorkflowEnv_VMAgents,
    MANAGER_ACTION_TABLE,
)

from a_set.metrics_logger import CSVLogger


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


NUM_OPTIONS = int(MANAGER_ACTION_TABLE.shape[0])


def manager_apply_action(env: CloudWorkflowEnv_VMAgents, act_idx: int):
    delta = MANAGER_ACTION_TABLE[int(act_idx)]
    env.apply_manager_delta(delta)


def make_manager_rng(episode_seed: int) -> np.random.RandomState:
    """
    为 Manager 构造独立随机数流，避免污染 env.rng。
    用 episode_seed 派生，保证每个 episode 可复现。
    """
    local_seed = (
        int(episode_seed) * 1000003
        + 20260325
        + 9176
    ) & 0xFFFFFFFF
    return np.random.RandomState(local_seed)


def random_manager_action(mask: np.ndarray, rng: np.random.RandomState) -> int:
    """
    在合法动作中等概率随机采样一个动作。
    若极端情况下没有合法动作，则回退为 0。
    """
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    valid = np.flatnonzero(mask > 0.5)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def compute_episode_task_lateness_metrics(env: CloudWorkflowEnv_VMAgents):
    total_late = 0.0
    task_cnt = 0
    wf_late_sum = {}

    n_tasks = len(getattr(env, "task_state", []))
    for tid in range(n_tasks):
        if env.task_state[tid] != "Finished":
            continue
        if tid >= len(env.task_baseline_finish):
            continue

        finish_t = float(env.task_end_time[tid])
        deadline_t = float(env.task_baseline_finish[tid])
        late = max(0.0, finish_t - deadline_t)

        total_late += late
        task_cnt += 1

        wf_id, _local_id = env.task_meta[tid]
        wf_late_sum[wf_id] = wf_late_sum.get(wf_id, 0.0) + late

    ep_avg_late = total_late / max(task_cnt, 1)

    if getattr(env, "workflows_per_episode", None) is not None:
        n_wf = int(env.workflows_per_episode)
        wf_total = 0.0
        for wf_id in range(n_wf):
            wf_total += wf_late_sum.get(wf_id, 0.0)
        wf_avg_late = wf_total / max(n_wf, 1)
    else:
        wf_avg_late = (sum(wf_late_sum.values()) / max(len(wf_late_sum), 1)) if len(wf_late_sum) > 0 else 0.0

    return float(total_late), float(ep_avg_late), float(wf_avg_late)


def _sync_env_scales(eval_env: CloudWorkflowEnv_VMAgents, env_kwargs: dict):
    for k in ["energy_reward_scale", "task_baseline_norm", "energy_norm_per_mi_ref", "alpha_delay_host", "alpha_delay_vm"]:
        if k in env_kwargs:
            setattr(eval_env, k, env_kwargs[k])


def evaluate_hrl_three_layer_multi_seed(env_cls, env_kwargs, vm_agent, host_agent, seeds):
    """
    评测时：
    - Manager：随机合法动作（由 seed 派生独立 rng）
    - Host / VM：仍为贪心 deterministic=True
    """
    ctor_block = {
        "energy_reward_scale", "task_baseline_norm",
        "energy_norm_per_mi_ref", "alpha_delay_host", "alpha_delay_vm"
    }
    base_ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}

    vm_list, host_list, mgr_list, energy_list = [], [], [], []

    for sd in seeds:
        ctor_kwargs = dict(base_ctor_kwargs)
        ctor_kwargs["random_seed"] = int(sd)

        eval_env = env_cls(**ctor_kwargs)
        _sync_env_scales(eval_env, env_kwargs)

        eval_env.reset()
        mgr_rng = make_manager_rng(int(sd))

        m_mask = eval_env.get_manager_action_mask()
        m_act = random_manager_action(m_mask, mgr_rng)
        manager_apply_action(eval_env, m_act)

        done = bool(getattr(eval_env, "done_flag", False))
        phases = 0
        ret_mgr = 0.0
        ret_vm_phase_mean = 0.0
        ret_host_phase_mean = 0.0

        while not done:
            vm_rewards = []
            host_rewards = []

            while True:
                st_host, has_next = eval_env.get_host_state_for_next_assignment()
                if not has_next:
                    break

                a_host = host_agent.select_action(
                    st_host["obs"], st_host["mask"],
                    deterministic=True, count_step=False
                )
                eval_env.host_select(int(a_host))

                st_vm, ok_vm = eval_env.get_vm_state_for_current_task()
                if not ok_vm:
                    break

                a_vm = vm_agent.select_action(
                    st_vm["obs"], st_vm["mask"],
                    deterministic=True, count_step=False
                )
                r_host, r_vm, _info_task = eval_env.vm_assign(int(a_vm))

                host_rewards.append(float(r_host))
                vm_rewards.append(float(r_vm))

            r_manager_raw, _pinfo = eval_env.finish_phase_and_advance()
            ret_mgr += float(r_manager_raw)
            ret_vm_phase_mean += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
            ret_host_phase_mean += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
            phases += 1

            done = bool(getattr(eval_env, "done_flag", False))
            if done:
                break

            m_mask = eval_env.get_manager_action_mask()
            m_act = random_manager_action(m_mask, mgr_rng)
            manager_apply_action(eval_env, m_act)

        avg_vm = ret_vm_phase_mean / max(phases, 1)
        avg_host = ret_host_phase_mean / max(phases, 1)
        avg_mgr = ret_mgr / max(phases, 1)
        total_energy = float(eval_env.total_energy)

        vm_list.append(avg_vm)
        host_list.append(avg_host)
        mgr_list.append(avg_mgr)
        energy_list.append(total_energy)

    return (
        float(np.mean(vm_list)),
        float(np.mean(host_list)),
        float(np.mean(mgr_list)),
        float(np.mean(energy_list)),
    )


def _warmup_ready(agent: D3QNAgent, frac: float = 0.1) -> bool:
    maxlen = agent.buffer.maxlen or 0
    need = int(maxlen * float(frac))
    need = max(need, int(agent.batch_size))
    return len(agent.buffer) >= need


def train():
    # ---------------- 基本超参 ----------------
    dax_dir = os.path.join(ROOT_DIR, "a_dax")
    dax_list = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "LIGO_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]
    horizon = 1e9
    arrival_lambda = 0.03
    random_seed = 1

    max_ready_tasks = "auto"
    normalize_obs = True
    WORKFLOWS_PER_EPISODE = 50

    # -------- 路线A参数 --------
    ENERGY_REWARD_SCALE = 1e-3
    TASK_BASELINE_NORM = 300.0
    ENERGY_NORM_PER_MI_REF = 20.0

    ALPHA_DELAY_HOST = 0.75
    ALPHA_DELAY_VM = 0.75

    # ===== 上层 reward：alpha * delay + (1-alpha) * energy =====
    MGR_ALPHA_DELAY = 0.75
    MGR_DELAY_MODE = "tardiness"

    # ===== workflow ddlAlpha 混合采样 =====
    DEADLINE_ALPHA_SMALL = 2.0
    DEADLINE_ALPHA_LARGE = 3.0
    DEADLINE_ALPHA_SMALL_PROB = 0.2

    # ---- VMAgent ----
    lr_vm = 3e-4
    gamma_vm = 0.95
    batch_vm = 256
    buf_vm = 120000
    eps_start_vm = 1.0
    eps_end_vm = 0.05
    eps_decay_vm = 80000
    tau_vm = 0.005
    grad_vm = 10.0
    hid_vm = (1024, 1024, 512, 512, 256)

    # ---- HostAgent ----
    lr_host = 3e-4
    gamma_host = 0.95
    batch_host = 256
    buf_host = 120000
    eps_start_host = 1.0
    eps_end_host = 0.05
    eps_decay_host = 80000
    tau_host = 0.005
    grad_host = 10.0
    hid_host = (1024, 1024, 512, 512, 256)

    MAX_EPISODES = 600
    save_interval = 2000

    save_dir = os.path.join(
        ROOT_DIR, "d_output", "ckpts",
        "mahdrl_taskS_resS_seed1_mgrrand_r075_ddlMix_hv_Loose",
    )
    log_path = os.path.join(
        ROOT_DIR, "d_output", "logs",
        "mahdrl_taskS_resS_seed1_mgrrand_r075_ddlMix_hv_Loose",
        "train.csv",
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    EVAL_SEEDS = [1]
    WARMUP_FRAC = 0.03

    DEADLINE_CACHE_PATH = os.path.join(
        ROOT_DIR, "d_output", "deadlines",
        "ddl_fcfs_taskS_resS_seed0-1000.json",
    )
    if not os.path.exists(DEADLINE_CACHE_PATH):
        raise FileNotFoundError(
            f"deadline cache 不存在：{DEADLINE_CACHE_PATH}\n"
            f"请先运行：b_hrl_legacy_ave/precompute_deadline_cache_fcfs_vmfirst_ss.py 生成 cache"
        )

    # ---------------- 初始化 ----------------
    set_seed(random_seed)

    EnvCls = CloudWorkflowEnv_VMAgents
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        max_ready_tasks=max_ready_tasks,
        normalize=normalize_obs,
        workflows_per_episode=WORKFLOWS_PER_EPISODE,

        num_hosts=3,
        vms_per_host=(9, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        deadline_mode="cache_fcfs",
        deadline_cache_path=DEADLINE_CACHE_PATH,
        deadline_cache_strict=True,
        deadline_alpha_small=DEADLINE_ALPHA_SMALL,
        deadline_alpha_large=DEADLINE_ALPHA_LARGE,
        deadline_alpha_small_prob=DEADLINE_ALPHA_SMALL_PROB,

        # 上层 reward 配置仍然保留
        manager_alpha_delay=MGR_ALPHA_DELAY,
        manager_delay_mode=MGR_DELAY_MODE,
    )

    env = EnvCls(**env_kwargs)
    env.reset()

    env.energy_reward_scale = ENERGY_REWARD_SCALE
    env.task_baseline_norm = TASK_BASELINE_NORM
    env.energy_norm_per_mi_ref = ENERGY_NORM_PER_MI_REF
    env.alpha_delay_host = ALPHA_DELAY_HOST
    env.alpha_delay_vm = ALPHA_DELAY_VM

    env_kwargs.update({
        "energy_reward_scale": env.energy_reward_scale,
        "task_baseline_norm": env.task_baseline_norm,
        "energy_norm_per_mi_ref": env.energy_norm_per_mi_ref,
        "alpha_delay_host": env.alpha_delay_host,
        "alpha_delay_vm": env.alpha_delay_vm,
    })

    st_host, ok = env.get_host_state_for_next_assignment()
    while (not ok) and (not env.done_flag):
        _rM, _ = env.finish_phase_and_advance()
        st_host, ok = env.get_host_state_for_next_assignment()

    host_state_dim = int(st_host["obs"].shape[0])
    host_act_dim = int(st_host["mask"].shape[0])

    a_host0 = int(np.argmax(st_host["mask"])) if np.sum(st_host["mask"]) > 0 else 0
    env.host_select(a_host0)
    st_vm, ok_vm = env.get_vm_state_for_current_task()
    vm_state_dim = int(st_vm["obs"].shape[0])
    vm_act_dim = int(st_vm["mask"].shape[0])

    env.reset()
    env.energy_reward_scale = env_kwargs["energy_reward_scale"]
    env.task_baseline_norm = env_kwargs["task_baseline_norm"]
    env.energy_norm_per_mi_ref = env_kwargs["energy_norm_per_mi_ref"]
    env.alpha_delay_host = env_kwargs["alpha_delay_host"]
    env.alpha_delay_vm = env_kwargs["alpha_delay_vm"]

    print(f"[env] host_obs_dim={host_state_dim} host_act_dim={host_act_dim} | vm_obs_dim={vm_state_dim} vm_act_dim={vm_act_dim}")
    print(f"[manager-ablation] action_dim={NUM_OPTIONS} (3^5), mode=random legal delta")
    print(
        f"[manager reward cfg] alpha_delay={MGR_ALPHA_DELAY:.3f} "
        f"alpha_energy={1.0-MGR_ALPHA_DELAY:.3f} delay_mode={MGR_DELAY_MODE}"
    )
    print(
        f"[ddl alpha cfg] P({DEADLINE_ALPHA_SMALL})={DEADLINE_ALPHA_SMALL_PROB:.2f}, "
        f"P({DEADLINE_ALPHA_LARGE})={1.0-DEADLINE_ALPHA_SMALL_PROB:.2f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vm_agent = D3QNAgent(
        input_dim=vm_state_dim,
        output_dim=vm_act_dim,
        lr=lr_vm,
        gamma=gamma_vm,
        batch_size=batch_vm,
        buffer_size=buf_vm,
        eps_start=eps_start_vm,
        eps_end=eps_end_vm,
        eps_decay_steps=eps_decay_vm,
        target_update_tau=tau_vm,
        grad_clip=grad_vm,
        hidden_dims=hid_vm,
        device=device,
    )

    host_agent = D3QNAgent(
        input_dim=host_state_dim,
        output_dim=host_act_dim,
        lr=lr_host,
        gamma=gamma_host,
        batch_size=batch_host,
        buffer_size=buf_host,
        eps_start=eps_start_host,
        eps_end=eps_end_host,
        eps_decay_steps=eps_decay_host,
        target_update_tau=tau_host,
        grad_clip=grad_host,
        hidden_dims=hid_host,
        device=device,
    )

    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "step", "episode", "type",
            "ep_length", "env_time",
            "episode_energy",
            "wf_completed", "wf_target",
            "eps_vm", "eps_host", "eps_mgr",
            "assign_cnt", "phase_size_sum_mi",
            "r_vm_phase_mean", "r_host_phase_mean",
            "r_manager_raw",
            "phase_energy", "phase_delay",
            "phase_energy_cost", "phase_delay_cost",
            "manager_reward_old", "manager_reward_new",
            "ep_total_lateness", "ep_avg_lateness", "wf_avg_lateness",
            "ep_wf_lateness_sum",
            "eval_vm", "eval_host", "eval_mgr", "eval_energy",
        ],
    )

    best_eval_energy = float("inf")
    best_ckpt_vm = os.path.join(save_dir, "best_vm.pth")
    best_ckpt_host = os.path.join(save_dir, "best_host.pth")

    global_step = 0
    episode_idx = 0
    episode_steps = 0
    ep_energy0 = float(env.total_energy)

    manager_rng = make_manager_rng(int(env.random_seed))
    m_mask = env.get_manager_action_mask()
    m_act = random_manager_action(m_mask, manager_rng)
    manager_apply_action(env, m_act)

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current_device:", torch.cuda.current_device())
        print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("vm_agent online device:", next(vm_agent.online.parameters()).device)
    print("host_agent online device:", next(host_agent.online.parameters()).device)

    HARD_MAX_STEPS = 10 ** 9
    while (episode_idx < MAX_EPISODES) and (global_step < HARD_MAX_STEPS):
        if env.done_flag:
            eval_vm, eval_host, eval_mgr, eval_energy = evaluate_hrl_three_layer_multi_seed(
                EnvCls, dict(env_kwargs), vm_agent, host_agent, seeds=EVAL_SEEDS
            )

            ep_energy = float(env.total_energy - ep_energy0)
            ep_total_late, ep_avg_late, wf_avg_late = compute_episode_task_lateness_metrics(env)
            ep_wf_late_sum = float(getattr(env, "ep_wf_lateness_sum", 0.0))

            logger.log(
                step=global_step, episode=episode_idx, type="episode",
                ep_length=episode_steps, env_time=getattr(env, "current_time", 0.0),
                episode_energy=ep_energy,
                wf_completed=getattr(env, "completed_workflows", 0),
                wf_target=(getattr(env, "workflows_per_episode", None) or ""),
                eps_vm=vm_agent.epsilon(),
                eps_host=host_agent.epsilon(),
                eps_mgr="",
                assign_cnt="",
                phase_size_sum_mi="",
                r_vm_phase_mean="",
                r_host_phase_mean="",
                r_manager_raw="",
                phase_energy="",
                phase_delay="",
                phase_energy_cost="",
                phase_delay_cost="",
                manager_reward_old="",
                manager_reward_new="",
                ep_total_lateness=ep_total_late,
                ep_avg_lateness=ep_avg_late,
                wf_avg_lateness=wf_avg_late,
                ep_wf_lateness_sum=ep_wf_late_sum,
                eval_vm=eval_vm, eval_host=eval_host, eval_mgr=eval_mgr, eval_energy=eval_energy,
            )

            print(
                f"[episode={episode_idx}] len={episode_steps} energy={ep_energy:.3f}J | "
                f"task_late_sum={ep_total_late:.3f}s task_late_avg={ep_avg_late:.6f}s | "
                f"wf_late_sum={ep_wf_late_sum:.3f}s | "
                f"eval_vm={eval_vm:.6f} eval_host={eval_host:.6f} eval_mgr={eval_mgr:.6f} eval_energy={eval_energy:.3f}J "
                f"(mean over seeds={EVAL_SEEDS}) | "
                f"wf={getattr(env,'completed_workflows',0)}/{getattr(env,'workflows_per_episode','∞')}"
            )

            if eval_energy < best_eval_energy:
                best_eval_energy = eval_energy
                vm_agent.save(best_ckpt_vm)
                host_agent.save(best_ckpt_host)
                print(f"⭐ best updated: eval_energy={best_eval_energy:.3f}J -> saved {best_ckpt_vm} / {best_ckpt_host}")

            if (global_step % save_interval) == 0:
                vm_agent.save(os.path.join(save_dir, f"vm_step{global_step}.pth"))
                host_agent.save(os.path.join(save_dir, f"host_step{global_step}.pth"))

            env.random_seed = getattr(env, "random_seed", 0) + 1
            env.reset()

            env.energy_reward_scale = env_kwargs["energy_reward_scale"]
            env.task_baseline_norm = env_kwargs["task_baseline_norm"]
            env.energy_norm_per_mi_ref = env_kwargs["energy_norm_per_mi_ref"]
            env.alpha_delay_host = env_kwargs["alpha_delay_host"]
            env.alpha_delay_vm = env_kwargs["alpha_delay_vm"]

            ep_energy0 = float(env.total_energy)
            episode_steps = 0
            episode_idx += 1

            manager_rng = make_manager_rng(int(env.random_seed))
            m_mask = env.get_manager_action_mask()
            m_act = random_manager_action(m_mask, manager_rng)
            manager_apply_action(env, m_act)
            continue

        phase_vm_rewards = []
        phase_host_rewards = []

        while True:
            st_host, has_next = env.get_host_state_for_next_assignment()
            if not has_next:
                break

            a_host = host_agent.select_action(
                st_host["obs"], st_host["mask"],
                deterministic=False, count_step=True
            )
            env.host_select(int(a_host))

            st_vm, ok_vm = env.get_vm_state_for_current_task()
            if not ok_vm:
                break

            a_vm = vm_agent.select_action(
                st_vm["obs"], st_vm["mask"],
                deterministic=False, count_step=True
            )
            r_host, r_vm, _info_task = env.vm_assign(int(a_vm))

            phase_host_rewards.append(float(r_host))
            phase_vm_rewards.append(float(r_vm))

            z_host_s = np.zeros_like(st_host["obs"])
            z_host_m = np.zeros_like(st_host["mask"])
            host_agent.remember(
                st_host["obs"], st_host["mask"], int(a_host),
                float(r_host), z_host_s, z_host_m, 1.0
            )
            if _warmup_ready(host_agent, WARMUP_FRAC):
                host_agent.update()

            z_vm_s = np.zeros_like(st_vm["obs"])
            z_vm_m = np.zeros_like(st_vm["mask"])
            vm_agent.remember(
                st_vm["obs"], st_vm["mask"], int(a_vm),
                float(r_vm), z_vm_s, z_vm_m, 1.0
            )
            if _warmup_ready(vm_agent, WARMUP_FRAC):
                vm_agent.update()

        r_manager_raw, pinfo = env.finish_phase_and_advance()
        assign_cnt = int(pinfo.get("assign_cnt", 0))
        phase_size_sum_mi = float(pinfo.get("phase_size_sum_mi", 0.0))

        r_vm_phase_mean = float(np.mean(phase_vm_rewards)) if len(phase_vm_rewards) > 0 else 0.0
        r_host_phase_mean = float(np.mean(phase_host_rewards)) if len(phase_host_rewards) > 0 else 0.0

        logger.log(
            step=global_step, episode=episode_idx, type="phase",
            ep_length="", env_time=pinfo.get("current_time", 0.0),
            episode_energy="",
            wf_completed="", wf_target="",
            eps_vm=vm_agent.epsilon(),
            eps_host=host_agent.epsilon(),
            eps_mgr="",
            assign_cnt=assign_cnt,
            phase_size_sum_mi=phase_size_sum_mi,
            r_vm_phase_mean=r_vm_phase_mean,
            r_host_phase_mean=r_host_phase_mean,
            r_manager_raw=float(r_manager_raw),
            phase_energy=float(pinfo.get("phase_energy", 0.0)),
            phase_delay=float(pinfo.get("phase_delay", 0.0)),
            phase_energy_cost=float(pinfo.get("phase_energy_cost", 0.0)),
            phase_delay_cost=float(pinfo.get("phase_delay_cost", 0.0)),
            manager_reward_old=float(pinfo.get("manager_reward_old", 0.0)),
            manager_reward_new=float(pinfo.get("manager_reward_new", 0.0)),
            ep_total_lateness="",
            ep_avg_lateness="",
            wf_avg_lateness="",
            ep_wf_lateness_sum="",
            eval_vm="", eval_host="", eval_mgr="", eval_energy="",
        )

        episode_steps += 1
        global_step += 1

        if not env.done_flag:
            m_mask = env.get_manager_action_mask()
            m_act = random_manager_action(m_mask, manager_rng)
            manager_apply_action(env, m_act)

    vm_final = os.path.join(save_dir, "vm_final.pth")
    host_final = os.path.join(save_dir, "host_final.pth")
    vm_agent.save(vm_final)
    host_agent.save(host_final)

    print(
        f"训练结束。\n"
        f"VM最终模型：{vm_final}\n"
        f"Host最终模型：{host_final}\n"
        f"最优评测模型：{best_ckpt_vm} / {best_ckpt_host}\n"
        f"日志：{log_path}"
    )


if __name__ == "__main__":
    train()