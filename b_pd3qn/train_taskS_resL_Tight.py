# -*- coding: utf-8 -*-
"""
train_taskS_resL_Tight.py  (SHALLOW ALIGN to HRL0.75 lower-layer VM reward)

本版修改点：
1) reward 口径浅对齐 HRL 下层 VMAgent：
       r_delay       = - clip( max(0, finish_time - task_deadline) / task_baseline_norm , 0, 1 )
       r_energy_host = - ΔE_host * energy_reward_scale
       reward        = (1-alpha_delay_vm) * r_energy_host + alpha_delay_vm * r_delay

2) 这里只改 reward 公式与参数语义：
   - 不改当前 PD3QN 的 ΔE 结算时机
   - 不改 bandit transition
   - 不改 warmup / 每次 assignment 更新一次的训练逻辑

3) 仍支持任意自定义 ddlAlpha 离散分布：
   deadline_alpha_choices / deadline_alpha_probs
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from a_set.metrics_logger import CSVLogger
from b_hrl.d3qn_agent_PER import D3QNAgent
from b_pd3qn.env_pd3qn import CloudWorkflowEnv_PD3QN_Cost


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warmup_ready(agent: D3QNAgent, warmup_frac: float = 0.03) -> bool:
    maxlen = int(getattr(agent, "buffer_size", 0) or 0)
    need = int(maxlen * float(warmup_frac))
    need = max(need, int(agent.batch_size))
    return len(agent.buffer) >= need


def evaluate_pd3qn(env_cls, env_kwargs: dict, agent: D3QNAgent, seeds):
    metrics_list = []
    for sd in seeds:
        ctor_kwargs = dict(env_kwargs)
        ctor_kwargs["random_seed"] = int(sd)

        env = env_cls(**ctor_kwargs)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done:
            mask = env.get_action_mask()
            act = agent.select_action(obs, mask, deterministic=True, count_step=False)
            obs, r, terminated, truncated, info = env.step(int(act))
            done = bool(terminated or truncated)
            steps += 1
            if steps > 10**9:
                break

        metrics_list.append(env.get_episode_metrics())

    keys = metrics_list[0].keys()
    out = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    return out


def train():
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
    WORKFLOWS_PER_EPISODE = 50
    random_seed = 1

    # ---------------- reward 参数：浅对齐 HRL 下层 VM reward ----------------
    ALPHA_DELAY_VM = 0.75
    TASK_BASELINE_NORM = 300.0
    ENERGY_REWARD_SCALE = 1e-3

    # 兼容保留，仅作占位打印；新 reward 中不再使用该量
    ENERGY_NORM_PER_MI_REF = 20.0

    DEADLINE_CACHE_PATH = os.path.join(
        ROOT_DIR, "d_output", "deadlines",
        "ddl_fcfs_taskS_resL_seed0-1000.json",
    )
    if not os.path.exists(DEADLINE_CACHE_PATH):
        raise FileNotFoundError(f"deadline cache 不存在：{DEADLINE_CACHE_PATH}")

    # ==========================================================
    # 任意自定义 ddlAlpha 离散分布
    # 默认示例：20% -> 2.0, 80% -> 3.0
    # ==========================================================
    DEADLINE_ALPHA_CHOICES = [2.0, 3.0]
    DEADLINE_ALPHA_PROBS = [0.8, 0.2]

    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        workflows_per_episode=WORKFLOWS_PER_EPISODE,
        max_ready_tasks="auto",
        normalize=True,

        num_hosts=9,
        vms_per_host=(9, 9, 9, 8, 8, 8, 8, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        # 新 reward 参数
        alpha_delay_vm=ALPHA_DELAY_VM,
        task_baseline_norm=TASK_BASELINE_NORM,
        energy_reward_scale=ENERGY_REWARD_SCALE,

        # 兼容占位：保留传入，但新 reward 不再使用
        energy_norm_per_mi_ref=ENERGY_NORM_PER_MI_REF,

        deadline_mode="cache_fcfs",
        deadline_cache_path=DEADLINE_CACHE_PATH,

        # 若没传离散分布，则回退固定 alpha
        deadline_alpha=1.5,

        # 任意离散分布
        deadline_alpha_choices=DEADLINE_ALPHA_CHOICES,
        deadline_alpha_probs=DEADLINE_ALPHA_PROBS,

        # alpha 独立采样的局部种子偏移量
        deadline_alpha_seed_offset=20260317,

        deadline_cache_strict=True,
    )

    # -------------- agent超参（PER版本）--------------
    lr = 3e-4
    gamma = 0.99
    batch_size = 256
    buffer_size = 120000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 80000

    target_update_tau = 0.005
    grad_clip = 10.0
    hidden_dims = (1024, 1024, 512, 512, 256)

    # PER 参数
    use_per = True
    per_alpha = 0.6
    per_beta_start = 0.4
    per_beta_end = 1.0
    per_beta_steps = 80000

    # warmup：buffer<比例只收集不更新
    WARMUP_FRAC = 0.03

    MAX_EPISODES = 600
    SAVE_INTERVAL = 2000
    EVAL_SEEDS = [1]

    save_dir = os.path.join(
        ROOT_DIR, "d_output", "ckpts",
        "pd3qn_taskS_resL_warmup_mix_nr_Tight",
    )
    log_path = os.path.join(
        ROOT_DIR, "d_output", "logs",
        "pd3qn_taskS_resL_warmup_mix_nr_Tight",
        "train_metrics_pd3qn_cost.csv",
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    set_seed(random_seed)

    env = CloudWorkflowEnv_PD3QN_Cost(**env_kwargs)
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = D3QNAgent(
        input_dim=obs_dim,
        output_dim=act_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
        target_update_tau=target_update_tau,
        grad_clip=grad_clip,
        hidden_dims=hidden_dims,
        device=device,

        use_per=use_per,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        per_beta_steps=per_beta_steps,
    )

    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "step", "episode", "type",
            "eps",
            "ep_length",
            "episode_energy",
            "task_avg_lateness", "task_avg_tardiness", "pct_tardy",
            "wf_avg_lateness", "wf_avg_tardiness",
            "wf_lateness_sum", "wf_tardiness_sum",
            "makespan",
            "eval_energy",
            "eval_task_avg_lateness", "eval_task_avg_tardiness", "eval_pct_tardy",
            "eval_wf_avg_lateness", "eval_wf_avg_tardiness",
            "eval_makespan",
        ],
    )

    best_eval_energy = float("inf")
    best_ckpt = os.path.join(save_dir, "best_pd3qn.pth")

    global_step = 0
    episode_idx = 0

    print(f"[env] obs_dim={obs_dim} act_dim={act_dim} | device={device}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("agent online device:", next(agent.online.parameters()).device)

    print(f"[deadline] cache={DEADLINE_CACHE_PATH}")
    print(f"[deadline] alpha_choices={DEADLINE_ALPHA_CHOICES}")
    print(f"[deadline] alpha_probs={DEADLINE_ALPHA_PROBS}")

    print(
        f"[reward shallow-align] "
        f"alpha_delay_vm={ALPHA_DELAY_VM:.3f}, "
        f"alpha_energy_vm={1.0-ALPHA_DELAY_VM:.3f}, "
        f"task_baseline_norm={TASK_BASELINE_NORM}, "
        f"energy_reward_scale={ENERGY_REWARD_SCALE}"
    )

    HARD_MAX_STEPS = 10**9
    while episode_idx < MAX_EPISODES and global_step < HARD_MAX_STEPS:
        done = False
        ep_steps = 0

        obs, _ = env.reset()

        # episode loop (一条经验=一次 task->VM 分配)
        while not done:
            mask = env.get_action_mask()
            act = agent.select_action(obs, mask, deterministic=False, count_step=True)

            obs2, rew, terminated, truncated, info = env.step(int(act))
            done = bool(terminated or truncated)

            # bandit transition：done=1, next_state/next_mask = 0
            z_s = np.zeros_like(obs, dtype=np.float32)
            z_m = np.zeros_like(mask, dtype=np.float32)

            agent.remember(
                obs, mask, int(act), float(rew),
                z_s, z_m, 1.0
            )

            if _warmup_ready(agent, WARMUP_FRAC):
                agent.update()

            obs = obs2
            ep_steps += 1
            global_step += 1

        m = env.get_episode_metrics()
        ep_energy = float(m["total_energy"])

        eval_m = evaluate_pd3qn(CloudWorkflowEnv_PD3QN_Cost, dict(env_kwargs), agent, seeds=EVAL_SEEDS)

        logger.log(
            step=global_step, episode=episode_idx, type="episode",
            eps=agent.epsilon(),
            ep_length=ep_steps,
            episode_energy=ep_energy,
            task_avg_lateness=float(m["task_avg_lateness"]),
            task_avg_tardiness=float(m["task_avg_tardiness"]),
            pct_tardy=float(m["pct_tardy"]),
            wf_avg_lateness=float(m["wf_avg_lateness"]),
            wf_avg_tardiness=float(m["wf_avg_tardiness"]),
            wf_lateness_sum=float(m["wf_lateness_sum"]),
            wf_tardiness_sum=float(m["wf_tardiness_sum"]),
            makespan=float(m["makespan"]),
            eval_energy=float(eval_m["total_energy"]),
            eval_task_avg_lateness=float(eval_m["task_avg_lateness"]),
            eval_task_avg_tardiness=float(eval_m["task_avg_tardiness"]),
            eval_pct_tardy=float(eval_m["pct_tardy"]),
            eval_wf_avg_lateness=float(eval_m["wf_avg_lateness"]),
            eval_wf_avg_tardiness=float(eval_m["wf_avg_tardiness"]),
            eval_makespan=float(eval_m["makespan"]),
        )

        print(
            f"[episode={episode_idx}] steps={ep_steps} eps={agent.epsilon():.4f} "
            f"energy={m['total_energy']:.3f}J | "
            f"task_avg_tardy={m['task_avg_tardiness']:.6f} pct_tardy={m['pct_tardy']:.6f} | "
            f"wf_avg_tardy={m['wf_avg_tardiness']:.6f} | "
            f"makespan={m['makespan']:.3f} || "
            f"eval_energy={eval_m['total_energy']:.3f}J (seeds={EVAL_SEEDS})"
        )

        if float(eval_m["total_energy"]) < best_eval_energy:
            best_eval_energy = float(eval_m["total_energy"])
            agent.save(best_ckpt)
            print(f"⭐ best updated: eval_energy={best_eval_energy:.3f}J -> saved {best_ckpt}")

        if (global_step % SAVE_INTERVAL) == 0:
            agent.save(os.path.join(save_dir, f"pd3qn_step{global_step}.pth"))

        # next episode seed progression
        env.random_seed = int(getattr(env, "random_seed", 0) + 1)
        env_kwargs["random_seed"] = int(env.random_seed)

        episode_idx += 1

    final_ckpt = os.path.join(save_dir, "pd3qn_final.pth")
    agent.save(final_ckpt)
    print(f"训练结束：final={final_ckpt} \n best={best_ckpt} \n log={log_path}")


if __name__ == "__main__":
    train()