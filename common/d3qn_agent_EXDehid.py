# -*- coding: utf-8 -*-
"""
d3qn_agent_EXDehid.py
Dueling Double DQN with action-mask support.

变更点（对齐你新提供文件的行为）：
- ε 衰减按「参数更新步数」推进（在 update() 成功执行一次就 +1），而不是按 select_action 次数。
- 保留 Polyak 软更新（target_update_tau>0）或硬同步（fallback）。
- F11：非终止下一状态必须有合法动作；终止可使用空掩码，学习目标无未来项。
- 评测时 deterministic=True 且不会影响 ε。
"""
from common.f18_state_normalization import checkpoint_metadata, check_checkpoint

from common.f11 import LearningDiagnostics, validate_transition, LEARNING_MODEL

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(512, 256), head_hidden_dims=None):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.feature = nn.Sequential(*layers)

        last_h = dims[-1]

        # --- 新增：head 结构可配置 ---
        if head_hidden_dims is None:
            # 默认：比你原来更深一点（2个隐藏层）
            head_hidden_dims = (last_h, last_h)

        self.adv = self._mlp(last_h, head_hidden_dims, output_dim)
        self.val = self._mlp(last_h, head_hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        a = self.adv(h)
        v = self.val(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    @staticmethod
    def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)



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
        target_update_freq: int = 2000, # 仅当 tau<=0 时使用硬同步
        target_update_tau: float = 0.005,   # >0 启用 Polyak 软更新
        grad_clip: float = 10.0,
        hidden_dims: Tuple[int, int] = (512, 256),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f11 = LearningDiagnostics()
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.grad_clip = grad_clip

        # self.online = DuelingQNet(input_dim, output_dim, hidden_dims).to(device)
        # self.target = DuelingQNet(input_dim, output_dim, hidden_dims).to(device)

        head_hidden_dims = (hidden_dims[-1], hidden_dims[-1])  # 常用：和 last_h 一样宽

        self.online = DuelingQNet(input_dim, output_dim, hidden_dims, head_hidden_dims=head_hidden_dims).to(device)
        self.target = DuelingQNet(input_dim, output_dim, hidden_dims, head_hidden_dims=head_hidden_dims).to(device)

        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.f11.numerics.inputs(self.online, "online_input")
        self.f11.numerics.inputs(self.target, "target_input")

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

    # ---------- 实用 ----------
    def epsilon(self) -> float:
        t = min(self._eps_steps, self._eps_decay_steps)
        frac = 1.0 - (t / self._eps_decay_steps)
        return float(self._eps_end + (self._eps_start - self._eps_end) * frac)

    def _valid_indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask > 0.5)
        if idx.size == 0:
            return np.array([self.output_dim - 1], dtype=np.int64)  # 回退到 No-op
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
        validate_transition(state, mask, action, reward, next_state, next_mask, done)
        self.f11.transition(done)
        self.buffer.append((np.array(state, copy=True), np.array(mask, copy=True), int(action),
                            float(reward), np.array(next_state, copy=True),
                            np.array(next_mask, copy=True), float(done)))

    # ---------- 学习 ----------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
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

        self.f11.target((1.0 - d) * self.gamma * q2, target)

        loss = F.smooth_l1_loss(q_sa, target)

        self.f11.numerics.observe("Q_selected", q_sa, scalar=True)
        self.f11.numerics.observe("TD_target", target, scalar=True)
        self.f11.numerics.observe("TD_error", target-q_sa, scalar=True)
        loss_value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        # === 成功更新后推进计数 ===
        self._updates += 1
        self.f11.updates += 1
        self._eps_steps += 1   # ✅ ε 的计步只在这里进行

        # 目标网络同步
        if self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        else:
            if (self._updates % self.target_update_freq) == 0:
                self.target.load_state_dict(self.online.state_dict())

        return loss_value

    # ---------- 存取 ----------
    def save(self, path: str):
        torch.save(
            {
                "normalization": checkpoint_metadata(self),
                "learning_model": LEARNING_MODEL,
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
            },
            path,
        )

    def load(self, path: str, strict: bool = True, require_learning_model: bool = False):
        # 兼容新版本 PyTorch 的安全加载（如可用）
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        check_checkpoint(ckpt.get("normalization") if isinstance(ckpt, dict) else None, checkpoint_metadata(self))
        if require_learning_model and ckpt.get("learning_model") != LEARNING_MODEL:
            raise ValueError(f"Checkpoint is not an F11-trained model: {path}")
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
