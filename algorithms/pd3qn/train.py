"""pd3qn: train. Select --task, --resource and --deadline; see --help."""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from common.scenarios import Scenario, parse_scenario
from common.f18_state_normalization import bind_normalization, check_checkpoint, normalization_diagnostics
from common.f11 import pd3qn_transition, F11Logger, protect_training_outputs

SCENARIO = Scenario(mode="train")
from common.metrics_logger import CSVLogger
from common.d3qn_agent_PER import D3QNAgent
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


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
    dax_dir = os.path.join(ROOT_DIR, "data/dax")
    dax_list = SCENARIO.dax_paths(dax_dir)
    horizon = 1000000000.0
    arrival_lambda = 0.03
    WORKFLOWS_PER_EPISODE = 50
    random_seed = SCENARIO.first_training_seed()
    ALPHA_DELAY_VM = 0.75
    TASK_BASELINE_NORM = 300.0
    ENERGY_REWARD_SCALE = 0.001
    ENERGY_NORM_PER_MI_REF = 20.0
    DEADLINE_CACHE_PATH = str(SCENARIO.cache_path)
    if not os.path.exists(DEADLINE_CACHE_PATH):
        raise FileNotFoundError(f"deadline cache 不存在：{DEADLINE_CACHE_PATH}")
    DEADLINE_ALPHA_CHOICES = [2.0, 3.0]
    DEADLINE_ALPHA_PROBS = list(SCENARIO.deadline_probs)
    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=horizon,
        arrival_lambda=arrival_lambda,
        random_seed=random_seed,
        workflows_per_episode=WORKFLOWS_PER_EPISODE,
        max_ready_tasks="auto",
        normalize=True,
        num_hosts=SCENARIO.num_hosts,
        vms_per_host=SCENARIO.vms_per_host,
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        alpha_delay_vm=ALPHA_DELAY_VM,
        task_baseline_norm=TASK_BASELINE_NORM,
        energy_reward_scale=ENERGY_REWARD_SCALE,
        energy_norm_per_mi_ref=ENERGY_NORM_PER_MI_REF,
        deadline_mode="cache_fcfs",
        deadline_cache_path=DEADLINE_CACHE_PATH,
        deadline_alpha=1.5,
        deadline_alpha_choices=DEADLINE_ALPHA_CHOICES,
        deadline_alpha_probs=DEADLINE_ALPHA_PROBS,
        deadline_cache_strict=True,
    )
    env_kwargs.update(SCENARIO.normalization_kwargs())
    lr = 0.0003
    gamma = 0.99
    batch_size = 256
    buffer_size = 120000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 80000
    target_update_tau = 0.005
    grad_clip = 10.0
    hidden_dims = (1024, 1024, 512, 512, 256)
    use_per = True
    per_alpha = 0.6
    per_beta_start = 0.4
    per_beta_end = 1.0
    per_beta_steps = 80000
    WARMUP_FRAC = 0.03
    MAX_EPISODES = SCENARIO.episodes
    SAVE_INTERVAL = 2000
    EVAL_SEEDS = SCENARIO.validation_seeds()
    save_dir = str(SCENARIO.checkpoint_dir("pd3qn"))
    log_path = os.path.join(
        ROOT_DIR, "outputs", "logs", SCENARIO.model_name("pd3qn"), "train_metrics_pd3qn_cost.csv"
    )
    protect_training_outputs(save_dir, os.path.dirname(log_path))
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
    bind_normalization(agent, SCENARIO)
    logger = CSVLogger(
        filepath=log_path,
        fieldnames=[
            "step",
            "episode",
            "type",
            "eps",
            "ep_length",
            "episode_energy",
            "task_avg_lateness",
            "task_avg_tardiness",
            "pct_tardy",
            "wf_avg_lateness",
            "wf_avg_tardiness",
            "wf_lateness_sum",
            "wf_tardiness_sum",
            "makespan",
            "eval_energy",
            "eval_task_avg_lateness",
            "eval_task_avg_tardiness",
            "eval_pct_tardy",
            "eval_wf_avg_lateness",
            "eval_wf_avg_tardiness",
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
        f"[reward shallow-align] alpha_delay_vm={ALPHA_DELAY_VM:.3f}, alpha_energy_vm={1.0 - ALPHA_DELAY_VM:.3f}, task_baseline_norm={TASK_BASELINE_NORM}, energy_reward_scale={ENERGY_REWARD_SCALE}"
    )
    f11_log = F11Logger(os.path.dirname(log_path))
    HARD_MAX_STEPS = 10**9
    while episode_idx < MAX_EPISODES and global_step < HARD_MAX_STEPS:
        done = False
        ep_steps = 0
        obs, _ = env.reset()
        while not done:
            obs = np.array(obs, copy=True)
            mask = np.array(env.get_action_mask(), copy=True)
            act = agent.select_action(obs, mask, deterministic=False, count_step=True)
            obs2, rew, terminated, truncated, info = env.step(int(act))
            done = bool(terminated or truncated)
            next_mask = env.get_action_mask()
            pd3qn_transition(agent, obs, mask, int(act), float(rew), obs2, next_mask, terminated, truncated)
            if _warmup_ready(agent, WARMUP_FRAC):
                agent.update()
            obs = obs2
            ep_steps += 1
            global_step += 1
        f11_log.log(episode_idx, {"PD3QN": agent}, normalization=normalization_diagnostics(env), pending=0,
                    terminated=bool(terminated), truncated=bool(truncated))
        m = env.get_episode_metrics()
        ep_energy = float(m["total_energy"])
        eval_m = evaluate_pd3qn(
            CloudWorkflowEnv_PD3QN_Cost, dict(env_kwargs), agent, seeds=EVAL_SEEDS
        )
        logger.log(
            step=global_step,
            episode=episode_idx,
            type="episode",
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
            f"[episode={episode_idx}] steps={ep_steps} eps={agent.epsilon():.4f} energy={m['total_energy']:.3f}J | task_avg_tardy={m['task_avg_tardiness']:.6f} pct_tardy={m['pct_tardy']:.6f} | wf_avg_tardy={m['wf_avg_tardiness']:.6f} | makespan={m['makespan']:.3f} || eval_energy={eval_m['total_energy']:.3f}J (seeds={EVAL_SEEDS})"
        )
        if float(eval_m["total_energy"]) < best_eval_energy:
            best_eval_energy = float(eval_m["total_energy"])
            agent.save(best_ckpt)
            print(f"⭐ best updated: eval_energy={best_eval_energy:.3f}J -> saved {best_ckpt}")
        if global_step % SAVE_INTERVAL == 0:
            agent.save(os.path.join(save_dir, f"pd3qn_step{global_step}.pth"))
        env.random_seed = int(getattr(env, "random_seed", 0) + 1)
        env_kwargs["random_seed"] = int(env.random_seed)
        episode_idx += 1
    final_ckpt = os.path.join(save_dir, "pd3qn_final.pth")
    agent.save(final_ckpt)
    print(f"训练结束：final={final_ckpt} \n best={best_ckpt} \n log={log_path}")

    logger.close()


if __name__ == "__main__":
    SCENARIO = parse_scenario("train", "pd3qn")
    train()
