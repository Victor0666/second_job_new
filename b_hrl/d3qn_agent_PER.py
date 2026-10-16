# -*- coding: utf-8 -*-
"""
d3qn_agent_PER.py
Dueling Double DQN with action-mask support + Prioritized Experience Replay (PER).

变更点：
- ε 衰减按「参数更新步数」推进（在 update() 成功执行一次就 +1），而不是按 select_action 次数。
- 保留 Polyak 软更新（target_update_tau>0）或硬同步（fallback）。
- 当 next_state 全非法时不做 bootstrap（q2=0）。
- 评测时 deterministic=True 且不会影响 ε。
- 新增：Prioritized Experience Replay（PER）
    * 经验池保存优先级 priority_i ≈ |TD error| + ε
    * 抽样概率 P(i) ∝ priority_i^α
    * 重要性采样权重 w_i ∝ (1 / (N·P(i)))^β，归一化到 [0,1]
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(512, 256)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.feature = nn.Sequential(*layers)
        last_h = dims[-1]
        self.adv = nn.Sequential(nn.Linear(last_h, last_h), nn.ReLU(), nn.Linear(last_h, output_dim))
        self.val = nn.Sequential(nn.Linear(last_h, last_h), nn.ReLU(), nn.Linear(last_h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        a = self.adv(h)
        v = self.val(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


def _polyak_update(target: nn.Module, online: nn.Module, tau: float):
    """target = (1 - tau) * target + tau * online"""
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


class D3QNAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 100000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 150000,  # 现在按“update 成功次数”线性退火
        target_update_freq: int = 2000,  # 仅当 tau<=0 时使用硬同步
        target_update_tau: float = 0.005,   # >0 启用 Polyak 软更新
        grad_clip: float = 10.0,
        hidden_dims: Tuple[int, int] = (512, 256),
        device: torch.device = torch.device("cpu"),
        # ====== 新增：PER 相关超参 ======
        use_per: bool = True,
        per_alpha: float = 0.6,        # 控制“多大程度按优先级”抽样（0=均匀）
        per_beta_start: float = 0.4,   # IS 权重初始 β
        per_beta_end: float = 1.0,     # 训练后期建议接近 1
        per_beta_steps: int = 150000,  # β 线性退火步数（按 update 次数）
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)
        self.device = device
        self.grad_clip = grad_clip

        # 经验池 & PER 优先级
        # buffer: List[(s, m, a, r, s2, m2, d)]
        # priorities: List[priority_i]
        self.buffer = []
        self.priorities = []

        self.online = DuelingQNet(input_dim, output_dim, hidden_dims).to(device)
        self.target = DuelingQNet(input_dim, output_dim, hidden_dims).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)

        # ε 线性衰减（按“成功 update 次数”推进）
        self._eps_start = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = 0                 # ✅ 每次成功 update() 后自增

        # 目标网络同步计数
        self._updates = 0
        self.target_update_freq = max(1, int(target_update_freq))
        self.target_update_tau = float(target_update_tau)

        # 仅用于统计（不影响 ε）
        self._action_calls = 0

        # ===== PER 配置 =====
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = max(1, int(per_beta_steps))
        self.per_beta_count = 0  # 按 update 次数推进 β

    # ---------- 实用 ----------
    def epsilon(self) -> float:
        t = min(self._eps_steps, self._eps_decay_steps)
        frac = 1.0 - (t / self._eps_decay_steps)
        return float(self._eps_end + (self._eps_start - self._eps_end) * frac)

    def _per_beta(self) -> float:
        """当前 IS 权重 β（线性退火）"""
        if not self.use_per:
            return 0.0
        t = min(self.per_beta_count, self.per_beta_steps)
        frac = t / self.per_beta_steps
        return float(self.per_beta_start + (self.per_beta_end - self.per_beta_start) * frac)

    def _valid_indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask > 0.5)
        if idx.size == 0:
            # 若全非法，退回到最后一个动作（常被当作 No-op）
            return np.array([self.output_dim - 1], dtype=np.int64)
        return idx

    # ---------- 交互 ----------
    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
        count_step: bool = True  # 兼容旧接口；现在对 ε 无效
    ) -> int:
        # 仅统计，不再用于 ε 计步
        if count_step:
            self._action_calls += 1

        eps = 0.0 if deterministic else self.epsilon()
        valid = self._valid_indices_from_mask(mask)

        # ε-greedy：只在合法动作中探索
        if (not deterministic) and (random.random() < eps):
            return int(np.random.choice(valid))

        # 贪心：对非法动作置极小
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, S]
            q = self.online(s).squeeze(0).clone()  # [A]
            m = torch.tensor(mask, dtype=torch.float32, device=self.device)
            q[m <= 0.5] = -1e9
            a = int(torch.argmax(q).item())
            return a

    def remember(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: float,
    ):
        """写入经验池，并为新样本赋最大优先级（鼓励近期样本被多采样几次）"""
        transition = (
            np.array(state, copy=False),
            np.array(mask, copy=False),
            int(action),
            float(reward),
            np.array(next_state, copy=False),
            np.array(next_mask, copy=False),
            float(done),
        )
        # 新样本优先级 = 当前池子中的最大优先级（或 1.0）
        max_prio = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            # FIFO：丢弃最老样本
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(max_prio)

    # ---------- 学习 ----------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        N = len(self.buffer)

        # ===== 按优先级抽样 =====
        if self.use_per:
            prios = np.asarray(self.priorities, dtype=np.float32)
            # 避免全 0
            prios = prios + 1e-6
            probs = prios ** self.per_alpha
            probs /= probs.sum()

            indices = np.random.choice(N, self.batch_size, p=probs)
            beta = self._per_beta()
            self.per_beta_count += 1

            # IS 权重：w_i ∝ (1 / (N * P(i)))^β，归一化到 [0,1]
            weights = (N * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            indices = np.random.choice(N, self.batch_size, replace=False)
            weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)

        batch = [self.buffer[i] for i in indices]
        s, m, a, r, s2, m2, d = map(np.array, zip(*batch))

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        m = torch.tensor(m, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        m2 = torch.tensor(m2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_all = self.online(s)                          # [B, A]
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        # Double DQN：online 选 a'，target 评估；两侧都 mask 非法
        with torch.no_grad():
            q2_online = self.online(s2).masked_fill(m2 <= 0.5, -1e9)
            a2 = q2_online.argmax(dim=1, keepdim=True)               # [B,1]

            q2_target = self.target(s2).masked_fill(m2 <= 0.5, -1e9)
            q2 = q2_target.gather(1, a2).squeeze(1)                  # [B]

            has_valid = (m2 > 0.5).any(dim=1)
            q2 = torch.where(has_valid, q2, torch.zeros_like(q2))

            target = r + (1.0 - d) * self.gamma * q2

        # 元素级 TD 误差
        td_errors = (target - q_sa).detach()  # [B]

        # PER：加权 Huber 损失
        loss_each = F.smooth_l1_loss(q_sa, target, reduction="none")  # [B]
        loss = (loss_each * weights).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        # === 成功更新后推进计数 ===
        self._updates += 1
        self._eps_steps += 1   # ✅ ε 的计步只在这里进行

        # 更新对应样本的优先级
        if self.use_per:
            new_prios = td_errors.abs().cpu().numpy() + 1e-6
            for idx, p in zip(indices, new_prios):
                self.priorities[int(idx)] = float(p)

        # 目标网络同步
        if self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        else:
            if (self._updates % self.target_update_freq) == 0:
                self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ---------- 存取 ----------
    def save(self, path: str):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optim": self.optim.state_dict(),
                "updates": self._updates,
                "eps_steps": self._eps_steps,
                "eps_start": self._eps_start,
                "eps_end": self._eps_end,
                "eps_decay_steps": self._eps_decay_steps,
                "tau": self.target_update_tau,
                "hard_freq": self.target_update_freq,
                # PER 超参
                "use_per": self.use_per,
                "per_alpha": self.per_alpha,
                "per_beta_start": self.per_beta_start,
                "per_beta_end": self.per_beta_end,
                "per_beta_steps": self.per_beta_steps,
                "per_beta_count": self.per_beta_count,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        # 兼容新版本 PyTorch 的安全加载（如可用）
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        self.online.load_state_dict(ckpt["online"], strict=strict)
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])

        self._updates = ckpt.get("updates", 0)
        self._eps_steps = ckpt.get("eps_steps", 0)
        self._eps_start = ckpt.get("eps_start", self._eps_start)
        self._eps_end = ckpt.get("eps_end", self._eps_end)
        self._eps_decay_steps = ckpt.get("eps_decay_steps", self._eps_decay_steps)
        self.target_update_tau = ckpt.get("tau", self.target_update_tau)
        self.target_update_freq = ckpt.get("hard_freq", self.target_update_freq)

        # 恢复 PER 配置（如果 ckpt 里没有，就沿用当前构造参数）
        self.use_per = ckpt.get("use_per", self.use_per)
        self.per_alpha = ckpt.get("per_alpha", self.per_alpha)
        self.per_beta_start = ckpt.get("per_beta_start", self.per_beta_start)
        self.per_beta_end = ckpt.get("per_beta_end", self.per_beta_end)
        self.per_beta_steps = ckpt.get("per_beta_steps", self.per_beta_steps)
        self.per_beta_count = ckpt.get("per_beta_count", self.per_beta_count)
