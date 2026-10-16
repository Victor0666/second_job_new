# -*- coding: utf-8 -*-
"""
algorithms.irws/train_irws_ppo_taskS_resM.py

IRWS 训练脚本（PPO 选 VM + GNN 选 Task + optional intrinsic）

新增：
- 支持每个 workflow 的 ddlAlpha 采用离散分布采样
- 例子：20% -> 2.0, 80% -> 3.0
- 可通过 DEADLINE_ALPHA_CHOICES / DEADLINE_ALPHA_PROBS 自由修改

本版说明：
- 环境 reward 已改为与“修改后的 PD3QN reward”相同：
      r_delay       = - clip( max(0, finish_time - task_deadline) / task_baseline_norm , 0, 1 )
      r_energy_host = - ΔE_host * energy_reward_scale
      reward        = (1-alpha_delay_vm) * r_energy_host + alpha_delay_vm * r_delay

- 但当前训练仍使用：
      r_fused = r_ext + intrinsic_weight * r_int
  因此若 intrinsic_weight > 0，则 PPO 实际训练信号仍不完全等同于 PD3QN

- 若你想让“训练时实际 reward”也尽量和 PD3QN 一致，
  可以把：
      use_intrinsic = False
  或：
      intrinsic_weight = 0.0
"""

from __future__ import annotations

import os
import sys
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.metrics_logger import CSVLogger
from algorithms.irws.env_irws import IRWSEnvRouteA
from algorithms.irws.gnn_task_ranker import IRWSTaskSelector, TaskSelectorConfig
from algorithms.irws.intrinsic_reward_net import IntrinsicRewardNet, IntrinsicNetConfig
from algorithms.irws.ppo_vm_agent import PPOVMAgent, PPOConfig


@dataclass
class RNGSnapshot:
    py_state: Any
    np_state: Any
    torch_state: Any
    cuda_state: Any

    @staticmethod
    def capture() -> "RNGSnapshot":
        return RNGSnapshot(
            py_state=random.getstate(),
            np_state=np.random.get_state(),
            torch_state=torch.get_rng_state(),
            cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def restore(self):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available() and self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_irws_multi_seed(
    env_ctor_kwargs: dict,
    ppo: PPOVMAgent,
    task_selector: IRWSTaskSelector,
    seeds: List[int],
    use_intrinsic: bool = False,
    intrinsic_net: Optional[IntrinsicRewardNet] = None,
) -> Dict[str, float]:
    metrics_list = []

    for sd in seeds:
        kw = dict(env_ctor_kwargs)
        kw["random_seed"] = int(sd)
        env = IRWSEnvRouteA(**kw)
        env.reset()

        while not env.done_flag:
            cand = env.get_ready_task_candidates()
            gids = cand["gids"]
            feats = cand["feats"]
            mask = cand["mask"]

            idx, _ = task_selector.select(feats, mask, deterministic=True)
            task_gid = int(gids[idx])
            env.select_task(task_gid)

            vm_st = env.get_vm_obs_and_mask()
            a_vm, _ = ppo.select_action(vm_st["obs"], vm_st["mask"], deterministic=True)

            _obs2, _r, _term, _trunc, _info = env.step(int(a_vm))

        metrics_list.append(env.get_episode_metrics())

    keys = metrics_list[0].keys() if metrics_list else []
    out = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in metrics_list]))
    return out


def train():
    # ---------------- 环境配置 ----------------
    dax_dir = os.path.join(ROOT_DIR, "data/dax")
    dax_list = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "LIGO_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    horizon = 1e9
    arrival_lambda = 0.03
    workflows_per_episode = 50
    max_ready_tasks = "auto"
    normalize_obs = True

    DEADLINE_CACHE_PATH = os.path.join(
        ROOT_DIR, "data", "deadlines",
        "ddl_fcfs_taskS_resM_seed0-1000.json",
    )
    if not os.path.exists(DEADLINE_CACHE_PATH):
        raise FileNotFoundError(
            f"deadline cache 不存在：{DEADLINE_CACHE_PATH}\n"
            f"请先运行你 HRL 的 precompute_deadline_cache_fcfs_vmfirst.py 生成 cache"
        )

    # ---------------- reward 参数：与修改后的 PD3QN reward 一致 ----------------
    ALPHA_DELAY_VM = 0.75
    TASK_BASELINE_NORM = 300.0
    ENERGY_REWARD_SCALE = 1e-3

    # ---------------- workflow-level ddlAlpha mixture ----------------
    # 20% -> 2.0, 80% -> 3.0
    # 以后你只需要改这两行即可
    DEADLINE_ALPHA_CHOICES = (2.0, 3.0)
    DEADLINE_ALPHA_PROBS = (0.8, 0.2)

    # 若你想退回固定alpha，可改为：
    # DEADLINE_ALPHA_CHOICES = None
    # DEADLINE_ALPHA_PROBS = None
    # 并保留 deadline_alpha=1.5

    UPDATE_EVERY_ASSIGNMENTS = 2048
    EVAL_SEEDS = [1]

    random_seed = 1
    set_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        workflows_per_episode=workflows_per_episode,
        max_ready_tasks=max_ready_tasks,
        normalize=normalize_obs,

        num_hosts=6,
        vms_per_host=(9, 9, 8, 8, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        deadline_mode="cache_fcfs",
        deadline_cache_path=DEADLINE_CACHE_PATH,

        # 固定 alpha 兼容参数；当 choices/probs 不为 None 时会被忽略
        deadline_alpha=1.5,

        # 每个 workflow 独立采样 ddlAlpha
        deadline_alpha_choices=DEADLINE_ALPHA_CHOICES,
        deadline_alpha_probs=DEADLINE_ALPHA_PROBS,

        deadline_cache_strict=True,

        # 与修改后的 PD3QN reward 对齐
        alpha_delay_vm=ALPHA_DELAY_VM,
        task_baseline_norm=TASK_BASELINE_NORM,
        energy_reward_scale=ENERGY_REWARD_SCALE,
    )

    env = IRWSEnvRouteA(**env_kwargs)
    env.reset()

    # ---------------- 构造智能体 ----------------
    task_selector = IRWSTaskSelector(TaskSelectorConfig(feat_dim=7, hidden=128, device=device))

    vm_st = None
    while not env.done_flag:
        cand = env.get_ready_task_candidates()
        gids = cand["gids"]
        if len(gids) == 0:
            env.reset()
            continue
        env.select_task(int(gids[0]))
        vm_st = env.get_vm_obs_and_mask()
        break
    if vm_st is None:
        raise RuntimeError("无法探测 PPO 输入维度：环境没有产生决策点")

    obs_dim = int(vm_st["obs"].shape[0])
    act_dim = int(vm_st["mask"].shape[0])

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
            device=device,
        ),
        hidden=(1024, 1024, 512, 512, 256),
    )

    # 注意：
    # 环境 reward 已经与修改后的 PD3QN 一致，
    # 但这里只保留你原来的 intrinsic 机制，因此 PPO 实际训练 reward 仍为 r_ext + 0.1*r_int
    use_intrinsic = True
    intrinsic_weight = 0.1
    intrinsic = IntrinsicRewardNet(IntrinsicNetConfig(obs_dim=obs_dim, hidden_dim=128, device=device))

    # ---------------- 日志/ckpt ----------------
    mix_tag = "ddlMix_Tight_rewardPD3QN_Tight"
    save_dir = os.path.join(
        ROOT_DIR, "outputs", "ckpts",
        f"ckpts_irws_ppo_smallTask_medRes_FCFS_{mix_tag}"
    )
    log_dir = os.path.join(
        ROOT_DIR, "outputs", "logs",
        f"logs_irws_ppo_smallTask_medRes_FCFS_{mix_tag}"
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train_metrics_irws.csv")
    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "assign_step", "episode", "type",
            "episode_assignments",
            "total_energy", "makespan",
            "task_avg_lateness", "task_avg_tardiness", "pct_tardy", "wf_avg_tardy",
            "eval_energy", "eval_makespan", "eval_task_avg_tardiness", "eval_pct_tardy", "eval_wf_avg_tardy",
            "ppo_loss",
        ],
    )

    best_eval_energy = float("inf")
    best_path = os.path.join(save_dir, "best_joint.pth")
    final_path = os.path.join(save_dir, "final_joint.pth")

    def save_joint(path: str):
        torch.save(
            {
                "ppo": ppo.net.state_dict(),
                "task_selector": task_selector.net.state_dict(),
                "intrinsic": intrinsic.state_dict(),
                "rng": RNGSnapshot.capture().__dict__,
                "env_kwargs": env_kwargs,
                "meta": {
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "device": device,
                },
            },
            path,
        )

    print(
        f"[IRWS env reward cfg] alpha_delay_vm={ALPHA_DELAY_VM:.3f}, "
        f"alpha_energy_vm={1.0-ALPHA_DELAY_VM:.3f}, "
        f"task_baseline_norm={TASK_BASELINE_NORM}, "
        f"energy_reward_scale={ENERGY_REWARD_SCALE}"
    )
    print(
        f"[ddl alpha cfg] choices={DEADLINE_ALPHA_CHOICES}, probs={DEADLINE_ALPHA_PROBS}"
    )
    print(
        f"[intrinsic] use_intrinsic={use_intrinsic}, intrinsic_weight={intrinsic_weight}"
    )

    # ---------------- 训练循环 ----------------
    episode = 0
    MAX_EPISODES = 600
    total_assign = 0

    while episode < MAX_EPISODES:
        env.random_seed = int(env.random_seed)
        env.reset()

        episode_assign = 0
        ppo_losses = []
        h_int = None

        while not env.done_flag:
            cand = env.get_ready_task_candidates()
            gids = cand["gids"]
            feats = cand["feats"]
            mask = cand["mask"]

            idx, _ = task_selector.select(feats, mask, deterministic=False)
            task_gid = int(gids[idx])
            env.select_task(task_gid)

            vm_st = env.get_vm_obs_and_mask()
            a_vm, info_a = ppo.select_action(vm_st["obs"], vm_st["mask"], deterministic=False)

            _obs2, r_ext, term, trunc, info = env.step(int(a_vm))

            r_int = 0.0
            if use_intrinsic and intrinsic_weight > 0:
                obs_t = torch.tensor(vm_st["obs"], dtype=torch.float32, device=intrinsic.device).view(1, 1, -1)
                with torch.no_grad():
                    r_seq, h_int = intrinsic(obs_t, h_int)
                    r_int = float(r_seq.item())

            # 保留你原来的 fused reward 逻辑
            r_fused = float(r_ext + intrinsic_weight * r_int)

            ppo.remember(
                vm_st["obs"],
                vm_st["mask"],
                int(a_vm),
                float(info_a["logp"]),
                float(info_a["value"]),
                float(r_fused),
                float(1.0 if term else 0.0),
            )

            episode_assign += 1
            total_assign += 1

            if (total_assign % UPDATE_EVERY_ASSIGNMENTS) == 0:
                loss = ppo.update()
                if loss is not None:
                    ppo_losses.append(float(loss))

        loss = ppo.update()
        if loss is not None:
            ppo_losses.append(float(loss))

        m = env.get_episode_metrics()

        eval_m = evaluate_irws_multi_seed(
            env_ctor_kwargs=env_kwargs,
            ppo=ppo,
            task_selector=task_selector,
            seeds=EVAL_SEEDS,
            use_intrinsic=False,
            intrinsic_net=None,
        )

        logger.log(
            assign_step=total_assign,
            episode=episode,
            type="episode",
            episode_assignments=episode_assign,
            total_energy=m["total_energy"],
            makespan=m["makespan"],
            task_avg_lateness=m["task_avg_lateness"],
            task_avg_tardiness=m["task_avg_tardiness"],
            pct_tardy=m["pct_tardy"],
            wf_avg_tardy=m["wf_avg_tardy"],
            eval_energy=eval_m["total_energy"],
            eval_makespan=eval_m["makespan"],
            eval_task_avg_tardiness=eval_m["task_avg_tardiness"],
            eval_pct_tardy=eval_m["pct_tardy"],
            eval_wf_avg_tardy=eval_m["wf_avg_tardy"],
            ppo_loss=float(np.mean(ppo_losses)) if ppo_losses else "",
        )

        print(
            f"[IRWS][episode={episode}] assign={episode_assign} | "
            f"energy={m['total_energy']:.3f}J makespan={m['makespan']:.3f} | "
            f"task_avg_tardy={m['task_avg_tardiness']:.6f} pct_tardy={m['pct_tardy']:.6f} wf_avg_tardy={m['wf_avg_tardy']:.6f} | "
            f"eval_energy={eval_m['total_energy']:.3f}J (seeds={EVAL_SEEDS})"
        )

        if eval_m["total_energy"] < best_eval_energy:
            best_eval_energy = float(eval_m["total_energy"])
            save_joint(best_path)
            print(f"⭐ best updated: eval_energy={best_eval_energy:.3f}J -> saved {best_path}")

        env.random_seed = int(env.random_seed) + 1
        episode += 1

    save_joint(final_path)
    print(f"训练结束。\nbest={best_path}\nfinal={final_path}\nlog={log_path}")


if __name__ == "__main__":
    train()