# -*- coding: utf-8 -*-
"""
algorithms.irws/ppo_vm_agent.py

带动作掩码 PPO（VM 选择）
公平性相关修改（6点）：
- deterministic=True：masked logits 后严格 argmax
- RolloutBuffer 存 r_step（融合后的 reward，由训练脚本统一提供）
- update 接受外部 fused rewards（保持你 IRWS 设定）

注意：这里实现的是“标准 PPO（GAE）”，用于 assignment step 粒度训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

from common.f11 import LearningDiagnostics, valid_decision, LEARNING_MODEL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 10.0
    update_epochs: int = 4
    minibatch_size: int = 256
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (1024, 1024, 512, 512, 256)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.shared = nn.Sequential(*layers)
        self.pi = nn.Linear(d, act_dim)
        self.v = nn.Linear(d, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.mask = []
        self.act = []
        self.logp = []
        self.val = []
        self.r_step = []
        self.done = []

    def clear(self):
        self.__init__()

    def add(self, obs, mask, act, logp, val, r_step, done):
        self.obs.append(obs)
        self.mask.append(mask)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.r_step.append(r_step)
        self.done.append(done)

    def size(self) -> int:
        return len(self.obs)


class PPOVMAgent:
    def __init__(self, cfg: PPOConfig, hidden: Tuple[int, ...] = (256, 256)):
        self.f11 = LearningDiagnostics()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = ActorCritic(cfg.obs_dim, cfg.act_dim, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.f11.numerics.inputs(self.net)
        self.buf = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray, mask_np: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, val = self.net(obs)  # [1,A], [1]
        logits = logits.masked_fill(mask <= 0.5, -1e9)

        if deterministic:
            act = int(torch.argmax(logits, dim=1).item())
            # logp for bookkeeping (not used in deterministic eval)
            logp = float(torch.log_softmax(logits, dim=1)[0, act].item())
            return act, {"logp": logp, "value": float(val.item()), "logits": logits.squeeze(0).cpu().numpy()}

        probs = torch.softmax(logits, dim=1)
        act_t = torch.multinomial(probs, 1)  # [1,1]
        act = int(act_t.item())
        logp = float(torch.log(probs[0, act] + 1e-12).item())
        return act, {"logp": logp, "value": float(val.item()), "probs": probs.squeeze(0).cpu().numpy()}

    def remember(self, obs_np, mask_np, act, logp, value, r_step, done):
        valid_decision(obs_np, mask_np)
        if mask_np[int(act)] <= 0.5 or not np.isfinite([logp, value, r_step, done]).all():
            raise ValueError("Invalid PPO transition")
        self.f11.transition(done)
        self.buf.add(
            obs=np.array(obs_np, dtype=np.float32),
            mask=np.array(mask_np, dtype=np.float32),
            act=int(act),
            logp=float(logp),
            val=float(value),
            r_step=float(r_step),
            done=float(done),
        )

    @torch.no_grad()
    def value(self, obs_np, mask_np):
        valid_decision(obs_np, mask_np)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.net(obs)[1].item())

    def update(self, last_value=None):
        if self.buf.size() == 0:
            return None
        if self.buf.done[-1]:
            last_value = 0.0
        elif last_value is None:
            raise ValueError("Non-terminal PPO rollout requires an actual next-decision value")
        if not np.isfinite(last_value):
            raise ValueError("Non-finite PPO bootstrap value")
        self.f11.last_rollout_value = float(last_value)
        if not self.buf.done[-1]:
            self.f11.nonterminal_rollouts += 1
            self.f11.last_nonterminal_rollout_value = float(last_value)

        obs = torch.tensor(np.stack(self.buf.obs), dtype=torch.float32, device=self.device)      # [T,obs]
        mask = torch.tensor(np.stack(self.buf.mask), dtype=torch.float32, device=self.device)    # [T,A]
        act = torch.tensor(np.array(self.buf.act), dtype=torch.int64, device=self.device)        # [T]
        old_logp = torch.tensor(np.array(self.buf.logp), dtype=torch.float32, device=self.device) # [T]
        old_val = torch.tensor(np.array(self.buf.val), dtype=torch.float32, device=self.device)   # [T]
        rew = torch.tensor(np.array(self.buf.r_step), dtype=torch.float32, device=self.device)    # [T]
        done = torch.tensor(np.array(self.buf.done), dtype=torch.float32, device=self.device)     # [T]

        # GAE advantage
        with torch.no_grad():
            # The last value comes from the selected next task, before any PPO update.
            T = obs.shape[0]
            adv = torch.zeros((T,), dtype=torch.float32, device=self.device)
            ret = torch.zeros((T,), dtype=torch.float32, device=self.device)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_non_terminal = 1.0 - done[t]
                next_value = last_value if (t == T - 1) else old_val[t + 1]
                delta = rew[t] + self.cfg.gamma * next_value * next_non_terminal - old_val[t]
                last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
                adv[t] = last_gae
                ret[t] = adv[t] + old_val[t]

        future = self.cfg.gamma * (1.0 - done) * torch.cat((old_val[1:], old_val.new_tensor([last_value])))
        self.f11.target(future, ret)
        self.f11.numerics.observe("return", ret, scalar=True)
        self.f11.numerics.observe("advantage_raw", adv, scalar=True)
        std = adv.std() if adv.numel() > 1 else adv.new_zeros(())
        adv = (adv - adv.mean()) / (std + 1e-8)

        # PPO update
        idxs = np.arange(obs.shape[0])
        losses = []

        for _ in range(int(self.cfg.update_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), int(self.cfg.minibatch_size)):
                mb = idxs[start:start + int(self.cfg.minibatch_size)]
                if len(mb) == 0:
                    continue

                logits, val = self.net(obs[mb])
                logits = logits.masked_fill(mask[mb] <= 0.5, -1e9)
                logp = torch.log_softmax(logits, dim=1).gather(1, act[mb].view(-1, 1)).squeeze(1)

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(val, ret[mb])

                # entropy
                probs = torch.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.f11.numerics.observe("value", val, scalar=True)
                self.f11.numerics.observe("advantage_used", adv[mb], scalar=True)
                loss_value = self.f11.loss(loss)
                self.opt.zero_grad()
                loss.backward()
                self.f11.numerics.gradients(self.net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.f11.updates += 1

                losses.append(loss_value)

        self.buf.clear()
        return float(np.mean(losses)) if losses else None

    def save(self, path: str):
        torch.save({"learning_model": LEARNING_MODEL, "state_dict": self.net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str, strict: bool = True, require_learning_model: bool = False):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        if require_learning_model and ckpt.get("learning_model") != LEARNING_MODEL:
            raise ValueError(f"Checkpoint is not an F11-trained model: {path}")
        self.net.load_state_dict(ckpt["state_dict"], strict=strict)
