# -*- coding: utf-8 -*-
"""
d3qn_agent_PER.py
Dueling DQN with action masks and Prioritized Experience Replay (PER).

paper_mode=True 按式(29)用 target 取最大值，执行无 IS 校正的 PER、MSE 和周期硬同步；
默认 False 保留历史训练行为及其 checkpoint 兼容性。

兼容机制：
- 历史模式的 ε 按参数更新步数衰减，paper 模式按训练决策步衰减。
- F11：非终止下一状态必须有合法动作；终止可使用空掩码，学习目标无未来项。
- 评测时 deterministic=True 且不会影响 ε。
- Prioritized Experience Replay（PER）
    * 经验池保存优先级 priority_i ≈ |TD error| + ε
    * 抽样概率 P(i) ∝ priority_i^α
    * 仅历史模式使用 IS 权重 w_i ∝ (1 / (N·P(i)))^β
"""
from common.f18_state_normalization import checkpoint_metadata, check_checkpoint

from common.f11 import LearningDiagnostics, validate_transition, LEARNING_MODEL
from algorithms.pd3qn_dag.identity import MODEL_VERSION

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_IMPLEMENTATION = MODEL_VERSION


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
        eps_decay_steps: int = 150000,  # paper 按训练决策；历史模式按成功 update 次数
        target_update_freq: int = 2000,  # 仅当 tau<=0 时使用硬同步
        target_update_tau: float = 0.005,   # >0 启用 Polyak 软更新
        grad_clip: float | None = 10.0,
        hidden_dims: Tuple[int, ...] = (512, 256),
        device: torch.device = torch.device("cpu"),
        # ====== 新增：PER 相关超参 ======
        use_per: bool = True,
        per_alpha: float = 0.6,        # 控制“多大程度按优先级”抽样（0=均匀）
        per_beta_start: float = 0.4,   # IS 权重初始 β
        per_beta_end: float = 1.0,     # 训练后期建议接近 1
        per_beta_steps: int = 150000,  # β 线性退火步数（按 update 次数）
        paper_mode: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f11 = LearningDiagnostics()
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)
        self.device = device
        self.paper_mode = bool(paper_mode)
        self.hidden_dims = tuple(hidden_dims)
        self.grad_clip = None if self.paper_mode else grad_clip

        # 经验池 & PER 优先级
        # buffer: List[(s, m, a, r, s2, m2, d)]
        # priorities: List[priority_i]
        self.buffer = []
        self.priorities = []

        self.online = DuelingQNet(input_dim, output_dim, self.hidden_dims).to(device)
        self.target = DuelingQNet(input_dim, output_dim, self.hidden_dims).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.f11.numerics.inputs(self.online, "online_input")
        self.f11.numerics.inputs(self.target, "target_input")

        # 论文模式按训练决策推进；历史模式按成功 update 次数推进。
        self._eps_start = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = 0

        # 目标网络同步计数
        self._updates = 0
        self.target_update_freq = max(1, int(target_update_freq))
        self.target_update_tau = 0.0 if self.paper_mode else float(target_update_tau)
        self._episode_steps = 0
        self._target_copy_due = False

        # 仅用于统计（不影响 ε）
        self._action_calls = 0

        # ===== PER 配置 =====
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = 0.0 if self.paper_mode else float(per_beta_start)
        self.per_beta_end = 0.0 if self.paper_mode else float(per_beta_end)
        self.per_beta_steps = max(1, int(per_beta_steps))
        self.per_beta_count = 0  # 按 update 次数推进 β

    # ---------- 实用 ----------
    def epsilon(self) -> float:
        t = min(self._eps_steps, self._eps_decay_steps)
        frac = 1.0 - (t / self._eps_decay_steps)
        return float(self._eps_end + (self._eps_start - self._eps_end) * frac)

    def _per_beta(self) -> float:
        """当前 IS 权重 β（线性退火）"""
        if self.paper_mode or not self.use_per:
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
    def begin_episode(self):
        """论文 Algorithm 1 的目标网络复制时钟在每个训练回合从 step=0 开始。"""
        self._episode_steps = 0
        self._target_copy_due = False

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
        count_step: bool = True,
    ) -> int:
        # 保留历史调用计数；评测不会推进论文模式的训练时钟。
        if count_step:
            self._action_calls += 1

        eps = 0.0 if deterministic else self.epsilon()
        if self.paper_mode and count_step and not deterministic:
            self._target_copy_due = (self._episode_steps % self.target_update_freq) == 0
            self._episode_steps += 1
            self._eps_steps += 1
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
        """论文模式按新经验自身 TD 误差入池；历史模式保留最大优先级。"""
        validate_transition(state, mask, action, reward, next_state, next_mask, done)
        self.f11.transition(done)
        transition = (
            np.array(state, copy=True),
            np.array(mask, copy=True),
            int(action),
            float(reward),
            np.array(next_state, copy=True),
            np.array(next_mask, copy=True),
            float(done),
        )
        if self.paper_mode and self.use_per:
            with torch.no_grad():
                s = torch.tensor(transition[0], dtype=torch.float32, device=self.device).unsqueeze(0)
                s2 = torch.tensor(transition[4], dtype=torch.float32, device=self.device).unsqueeze(0)
                m2 = torch.tensor(transition[5], dtype=torch.float32, device=self.device).unsqueeze(0)
                r = torch.tensor([transition[3]], dtype=torch.float32, device=self.device)
                d = torch.tensor([transition[6]], dtype=torch.float32, device=self.device)
                target, _ = self._td_target(r, s2, m2, d)
                q = self.online(s)[0, transition[2]]
                priority = float((target[0] - q).abs().item()) + 1e-6
                if not np.isfinite(priority):
                    raise FloatingPointError("Nonfinite PD3QN initial priority")
        else:
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            # FIFO：丢弃最老样本
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(priority)

    # ---------- 学习 ----------
    @torch.no_grad()
    def _td_target(self, rewards, next_states, next_masks, dones):
        if self.paper_mode:
            # Eq.(29): target selects its own maximum over legal next actions.
            q2 = self.target(next_states).masked_fill(next_masks <= 0.5, -torch.inf).max(dim=1).values
            bootstrap = (dones < 0.5) & (next_masks > 0.5).any(dim=1)
            q2 = torch.where(bootstrap, q2, torch.zeros_like(q2))
            return rewards + self.gamma * q2, q2
        # Historical Double DQN: online selects, target evaluates.
        q2_online = self.online(next_states).masked_fill(next_masks <= 0.5, -1e9)
        a2 = q2_online.argmax(dim=1, keepdim=True)
        q2_target = self.target(next_states).masked_fill(next_masks <= 0.5, -1e9)
        q2 = q2_target.gather(1, a2).squeeze(1)
        q2 = torch.where((next_masks > 0.5).any(dim=1), q2, torch.zeros_like(q2))
        return rewards + (1.0 - dones) * self.gamma * q2, q2

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        N = len(self.buffer)

        # ===== 按优先级抽样 =====
        if self.use_per:
            prios = np.asarray(self.priorities, dtype=np.float32)
            # 论文模式的 priority 已是 |TD error| + ζ，式(34)只加一次 ζ。
            if not self.paper_mode:
                prios = prios + 1e-6
            probs = prios ** self.per_alpha
            probs /= probs.sum()

            indices = np.random.choice(N, self.batch_size, p=probs)
            if self.paper_mode:
                weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
            else:
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

        target, q2 = self._td_target(r, s2, m2, d)

        # 元素级 TD 误差
        self.f11.target((1.0 - d) * self.gamma * q2, target)

        td_errors = (target - q_sa).detach()  # [B]

        if self.paper_mode:
            loss = F.mse_loss(q_sa, target)
        else:
            loss_each = F.smooth_l1_loss(q_sa, target, reduction="none")
            loss = (loss_each * weights).mean()

        self.f11.numerics.observe("Q_selected", q_sa, scalar=True)
        self.f11.numerics.observe("TD_target", target, scalar=True)
        self.f11.numerics.observe("TD_error", target-q_sa, scalar=True)
        loss_value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), None if self.paper_mode else self.grad_clip)
        if not self.paper_mode and self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        # === 成功更新后推进计数 ===
        self._updates += 1
        self.f11.updates += 1
        if not self.paper_mode:
            self._eps_steps += 1

        # 更新对应样本的优先级
        if self.use_per:
            new_prios = td_errors.abs().cpu().numpy() + 1e-6
            for idx, p in zip(indices, new_prios):
                self.priorities[int(idx)] = float(p)

        # 目标网络同步
        if self.paper_mode:
            # Algorithm 1 为零基 step：第 1、61、121…次决策更新后复制（freq=60）。
            if self._target_copy_due:
                self.target.load_state_dict(self.online.state_dict())
                self._target_copy_due = False
        elif self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        else:
            if (self._updates % self.target_update_freq) == 0:
                self.target.load_state_dict(self.online.state_dict())

        return loss_value

    # ---------- 存取 ----------
    def save(self, path: str):
        torch.save(
            {
                "pd3qn_implementation": PAPER_IMPLEMENTATION if self.paper_mode else "legacy",
                "network": {"input_dim": self.input_dim, "output_dim": self.output_dim,
                            "hidden_dims": list(self.hidden_dims)},
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
                "episode_steps": self._episode_steps,
                "target_copy_due": self._target_copy_due,
                "action_calls": self._action_calls,
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

    def load(self, path: str, strict: bool = True, require_learning_model: bool = False):
        # 兼容新版本 PyTorch 的安全加载（如可用）
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        expected = PAPER_IMPLEMENTATION if self.paper_mode else "legacy"
        actual = ckpt.get("pd3qn_implementation", "legacy")
        if actual != expected:
            raise ValueError(f"PD3QN implementation mismatch: checkpoint={actual}, agent={expected}. "
                             "Use the matching paper/legacy configuration; retraining is required to change algorithms.")
        network = {"input_dim": self.input_dim, "output_dim": self.output_dim,
                   "hidden_dims": list(self.hidden_dims)}
        saved_network = ckpt.get("network", None if self.paper_mode else network)
        if saved_network != network:
            raise ValueError(f"PD3QN network mismatch: checkpoint={saved_network}, agent={network}")
        check_checkpoint(ckpt.get("normalization"), checkpoint_metadata(self))
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
        self._episode_steps = ckpt.get("episode_steps", 0)
        self._target_copy_due = ckpt.get("target_copy_due", False)
        self._action_calls = ckpt.get("action_calls", 0)
        self.target_update_tau = 0.0 if self.paper_mode else ckpt.get("tau", self.target_update_tau)
        self.target_update_freq = ckpt.get("hard_freq", self.target_update_freq)

        # 恢复 PER 配置（如果 ckpt 里没有，就沿用当前构造参数）
        self.use_per = ckpt.get("use_per", self.use_per)
        self.per_alpha = ckpt.get("per_alpha", self.per_alpha)
        self.per_beta_start = 0.0 if self.paper_mode else ckpt.get("per_beta_start", self.per_beta_start)
        self.per_beta_end = 0.0 if self.paper_mode else ckpt.get("per_beta_end", self.per_beta_end)
        self.per_beta_steps = ckpt.get("per_beta_steps", self.per_beta_steps)
        self.per_beta_count = 0 if self.paper_mode else ckpt.get("per_beta_count", self.per_beta_count)
