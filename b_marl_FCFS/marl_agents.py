# -*- coding: utf-8 -*-
"""
b_marl_FCFS/marl_agents.py

A2C with action-mask support.

- GlobalAgent: 标准 A2C（actor+critic共享一套）
- LocalAgent : 方案1+2（用于“每Host一个局部智能体”的参数共享实现）
    * actor: 共享（一个 policy_head）
    * critic: 每个 Host 一个独立 value head（ModuleList）
    * local_obs 已在 env 侧追加 host_id one-hot（方案2）

新增：
- load_weights_flexible(): 支持“用 small ckpt 加载到 medium/large 网络”做泛化评测
  * 允许 obs_dim / act_dim / num_hosts 变化
  * 对线性层做裁剪拷贝：重叠部分复制，新增维度保持当前初始化
  * 评测只加载 net，不加载 optim（避免优化器 state 不匹配）
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Flexible load helpers
# =========================
def _torch_load_ckpt(path: str, device: torch.device):
    """
    兼容不同 torch 版本：
    - 新版本支持 weights_only=True 可减少 pickle 风险与 warning
    - 老版本没有 weights_only 参数则回退
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _copy_tensor_flexible(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    将 src 拷贝到 dst（允许尺寸不同）：
    - 1D: 拷贝前 min(n)
    - 2D: 拷贝 [:min(r), :min(c)]
    - 其他维度：仅当 shape 完全一致时拷贝
    返回是否完成拷贝（True=拷贝了重叠部分）
    """
    if dst.dim() == 1 and src.dim() == 1:
        n = min(dst.shape[0], src.shape[0])
        if n > 0:
            dst[:n].copy_(src[:n])
            return True
        return False

    if dst.dim() == 2 and src.dim() == 2:
        r = min(dst.shape[0], src.shape[0])
        c = min(dst.shape[1], src.shape[1])
        if r > 0 and c > 0:
            dst[:r, :c].copy_(src[:r, :c])
            return True
        return False

    if dst.shape == src.shape:
        dst.copy_(src)
        return True

    return False


def _load_state_dict_flexible(net: nn.Module, src_state_dict: dict):
    """
    以“裁剪/扩维”方式将 src_state_dict 加载到 net：
    - key 存在且 tensor 维度可重叠：拷贝重叠部分
    - key 不存在或不可拷贝：跳过（保持初始化）
    """
    dst_sd = net.state_dict()
    with torch.no_grad():
        for k, src_v in src_state_dict.items():
            if k not in dst_sd:
                continue
            dst_v = dst_sd[k]
            _copy_tensor_flexible(dst_v, src_v)
    net.load_state_dict(dst_sd, strict=False)


# =========================
# Config
# =========================
@dataclass
class A2CConfig:
    lr: float = 3e-4
    gamma: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip: float = 10.0
    device: str = "cpu"
    hidden_dims: Tuple[int, int] = (256, 128)


# ---------------- 基础：单critic A2C（用于 Global） ----------------
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        h = dims[-1]
        self.policy_head = nn.Linear(h, act_dim)
        self.value_head = nn.Linear(h, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        logits = self.policy_head(h)
        v = self.value_head(h).squeeze(-1)  # [B]
        return logits, v


class A2CAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: A2CConfig):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.net = ActorCriticNet(obs_dim, act_dim, hidden_dims=cfg.hidden_dims).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[mask <= 0.5] = -1e9
        all_invalid = (mask > 0.5).sum(dim=-1) == 0
        if all_invalid.any():
            masked[all_invalid] = logits[all_invalid]
        return masked

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _v = self.net(obs_t)
        logits = self._masked_logits(logits, mask_t)

        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())

        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def update_one_step(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        act: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        done: float,
    ) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor([int(act)], dtype=torch.int64, device=self.device)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_mask_t = torch.tensor(next_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        r_t = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
        d_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        logits, v = self.net(obs_t)
        logits = self._masked_logits(logits, mask_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        ent = dist.entropy()

        with torch.no_grad():
            _logits2, v2 = self.net(next_obs_t)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * v2

        adv = (target - v).detach()

        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(v, target)
        entropy_loss = -ent.mean()
        loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * entropy_loss

        self.optim.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "cfg": self.cfg.__dict__,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = _torch_load_ckpt(path, self.device)
        if isinstance(ckpt, dict) and "net" in ckpt:
            self.net.load_state_dict(ckpt["net"], strict=strict)
            # 训练继续跑才需要 optim；评测用不加载也行
            if "optim" in ckpt:
                try:
                    self.optim.load_state_dict(ckpt["optim"])
                except Exception:
                    pass
        else:
            self.net.load_state_dict(ckpt, strict=strict)

    def load_weights_flexible(self, path: str):
        """
        评测用：允许 obs_dim/act_dim 变化的“裁剪/扩维加载”。
        - 只加载 net 权重（不加载 optim）
        """
        ckpt = _torch_load_ckpt(path, self.device)
        src_sd = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        _load_state_dict_flexible(self.net, src_sd)


# ---------------- 方案1：共享actor + 独立critic（按Host）用于 Local ----------------
class SharedActorMultiCriticNet(nn.Module):
    """
    一个共享 policy（actor），多个 value head（critic），每个 Host 一个 value head。

    forward(obs, host_index) -> (logits, v)
    """

    def __init__(self, obs_dim: int, act_dim: int, num_hosts: int, hidden_dims=(256, 128)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_hosts = int(num_hosts)

        dims = [obs_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        h = dims[-1]
        self.policy_head = nn.Linear(h, act_dim)  # 共享 actor
        self.value_heads = nn.ModuleList([nn.Linear(h, 1) for _ in range(self.num_hosts)])  # 独立 critic

    def forward(self, obs: torch.Tensor, host_index: torch.Tensor):
        """
        obs: [B, obs_dim]
        host_index: [B]  0..num_hosts-1
        """
        feat = self.backbone(obs)
        logits = self.policy_head(feat)  # [B, act_dim]

        # 逐样本选择对应 host 的 value head
        vs = []
        for i in range(feat.shape[0]):
            hi = int(host_index[i].item())
            hi = max(0, min(self.num_hosts - 1, hi))
            vs.append(self.value_heads[hi](feat[i : i + 1]).squeeze(-1))  # [1] -> [1]
        v = torch.cat(vs, dim=0)  # [B]
        return logits, v


class LocalSharedActorIndependentCriticA2C:
    """
    Local 侧智能体（参数共享实现）：
    - actor 共享
    - critic 按 host 独立
    - select/update 需要传入 host_index（即 Global 选择的 host 的 index）
    """

    def __init__(self, obs_dim: int, act_dim: int, num_hosts: int, cfg: A2CConfig):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_hosts = int(num_hosts)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.net = SharedActorMultiCriticNet(obs_dim, act_dim, num_hosts, hidden_dims=cfg.hidden_dims).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[mask <= 0.5] = -1e9
        all_invalid = (mask > 0.5).sum(dim=-1) == 0
        if all_invalid.any():
            masked[all_invalid] = logits[all_invalid]
        return masked

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask: np.ndarray, host_index: int, deterministic: bool = False) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        hi_t = torch.tensor([int(host_index)], dtype=torch.int64, device=self.device)

        logits, _v = self.net(obs_t, hi_t)
        logits = self._masked_logits(logits, mask_t)

        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())

        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def update_one_step(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        act: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        done: float,
        host_index: int,
    ) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor([int(act)], dtype=torch.int64, device=self.device)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_mask_t = torch.tensor(next_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        r_t = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
        d_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        hi_t = torch.tensor([int(host_index)], dtype=torch.int64, device=self.device)

        logits, v = self.net(obs_t, hi_t)
        logits = self._masked_logits(logits, mask_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        ent = dist.entropy()

        with torch.no_grad():
            _logits2, v2 = self.net(next_obs_t, hi_t)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * v2

        adv = (target - v).detach()

        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(v, target)
        entropy_loss = -ent.mean()
        loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * entropy_loss

        self.optim.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "cfg": self.cfg.__dict__,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "num_hosts": self.num_hosts,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = _torch_load_ckpt(path, self.device)
        if isinstance(ckpt, dict) and "net" in ckpt:
            self.net.load_state_dict(ckpt["net"], strict=strict)
            if "optim" in ckpt:
                try:
                    self.optim.load_state_dict(ckpt["optim"])
                except Exception:
                    pass
        else:
            self.net.load_state_dict(ckpt, strict=strict)

    def load_weights_flexible(self, path: str):
        """
        评测用：允许 obs_dim / act_dim / num_hosts 变化的“裁剪/扩维加载”。
        - 只加载 net（不加载 optim）
        - value_heads 数量变化：只会拷贝重叠的 head；新增 head 保持初始化
        - backbone 第1层输入维度变化（host one-hot 变长）：按列拷贝重叠部分
        """
        ckpt = _torch_load_ckpt(path, self.device)
        src_sd = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        _load_state_dict_flexible(self.net, src_sd)
