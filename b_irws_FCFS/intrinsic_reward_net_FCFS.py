# -*- coding: utf-8 -*-
"""
b_irws/intrinsic_reward_net.py

GRU 内在奖励网络（IRWS 用于 intrinsic shaping）
公平性6点不要求修改 intrinsic 本身，但需要：
- deterministic 评测时可关闭 intrinsic（训练脚本里通过 weight=0 控制）
- save/load 供 joint checkpoint

这里提供一个稳定的实现：输入 state embedding（PPO obs）输出 r_int
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IntrinsicNetConfig:
    obs_dim: int
    hidden_dim: int = 128
    device: str = "cpu"


class IntrinsicRewardNet(nn.Module):
    def __init__(self, cfg: IntrinsicNetConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.fc_in = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.gru = nn.GRU(input_size=cfg.hidden_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(cfg.hidden_dim, 1)

        self.to(self.device)

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_seq: [B,T,obs_dim]
        return:
          r_int: [B,T]
          hT: [1,B,H]
        """
        x = F.relu(self.fc_in(obs_seq))
        out, hT = self.gru(x, h0)
        r = self.fc_out(out).squeeze(-1)
        return r, hT

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["state_dict"], strict=strict)
