# -*- coding: utf-8 -*-
"""
algorithms.irws/gnn_task_ranker.py

IRWS Task Selector（“GNN”风格的轻量实现，纯 PyTorch，无 torch_geometric 依赖）
- 输入：候选 task 的 node feats [N,F] 与 mask [N]
- 输出：每个候选的 score/logits
- 训练：可用 PPO/外部信号更新（本文件只提供网络与 select/save/load）

公平性相关修改（对应你6点）：
- deterministic=True 时：严格 argmax（mask 后）
- 候选集大小由 env 统一截断（max_ready_tasks 一致）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMessagePassing(nn.Module):
    """
    一个非常轻量的“消息传递”层（无显式图邻接时退化为 MLP）。
    这里先实现为 MLP（你真实 GNN 版本若有图邻接，可在 forward 里加邻接聚合）。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return h


class TaskRankNet(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.mp = SimpleMessagePassing(feat_dim, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [N,F]
        return logits: [N]
        """
        h = self.mp(feats)
        logits = self.head(h).squeeze(-1)
        return logits


@dataclass
class TaskSelectorConfig:
    feat_dim: int = 7
    hidden: int = 128
    device: str = "cpu"


class IRWSTaskSelector:
    def __init__(self, cfg: TaskSelectorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = TaskRankNet(cfg.feat_dim, cfg.hidden).to(self.device)
        from common.f19_numerics import NumericalDiagnostics
        self.numerics = NumericalDiagnostics()
        self.numerics.inputs(self.net)

    @torch.no_grad()
    def select(
        self,
        feats_np: np.ndarray,
        mask_np: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        返回：chosen_index_in_candidates, info
        """
        feats = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)

        logits = self.net(feats)  # [N]
        logits = logits.masked_fill(mask <= 0.5, -1e9)

        if deterministic:
            idx = int(torch.argmax(logits).item())
            return idx, {"logits": logits.detach().cpu().numpy()}

        probs = torch.softmax(logits, dim=0)
        idx = int(torch.multinomial(probs, 1).item())
        return idx, {"logits": logits.detach().cpu().numpy(), "probs": probs.detach().cpu().numpy()}

    def parameters(self):
        return self.net.parameters()

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"], strict=strict)
