"""Source-configured PPO for the adapted DRL-GNN joint policy."""
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .config import DRLGNNConfig
from .networks import GraphActorCritic


@dataclass(frozen=True)
class Transition:
    state: Any
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    next_value: float
    env_id: int = 0


def compute_gae(records: Sequence[Transition], gamma: float, gae_lambda: float):
    """Return advantages/returns in record order without mixing vector-env paths."""
    if not records:
        raise ValueError("PPO needs a nonempty rollout")
    scalars = np.asarray([
        (r.logp, r.value, r.reward, r.next_value) for r in records
    ], dtype=np.float64)
    if not np.isfinite(scalars).all():
        raise ValueError("PPO rollout contains nonfinite values")
    advantages = np.empty(len(records), dtype=np.float32)
    carry_by_env = {}
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        nonterminal = 1.0 - float(record.done)
        delta = record.reward + gamma * nonterminal * record.next_value - record.value
        carry = delta + gamma * gae_lambda * nonterminal * carry_by_env.get(record.env_id, 0.0)
        advantages[index] = carry
        carry_by_env[record.env_id] = carry
    returns = advantages + np.asarray([r.value for r in records], dtype=np.float32)
    return advantages, returns


class PPOAgent:
    def __init__(self, config=None, device=None):
        self.cfg = config or DRLGNNConfig()
        self.device = torch.device(device if device is not None else self.cfg.device)
        self.network = GraphActorCritic(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.cfg.learning_rate, eps=self.cfg.adam_eps,
        )
        self.steps = 0
        self.updates = 0

    @torch.no_grad()
    def act(self, state, deterministic=False):
        # The author uses single-graph BatchNorm in train mode during inference too.
        self.network.train()
        distribution, value = self.network.distribution_and_value(state)
        action = distribution.probs.argmax() if deterministic else distribution.sample()
        self.steps += 1
        return int(action), float(distribution.log_prob(action)), float(value)

    @torch.no_grad()
    def value(self, state):
        self.network.train()
        return float(self.network.value(state))

    def update(self, transitions: Sequence[Transition], learning_rate=None):
        records = tuple(transitions)
        advantages, returns = compute_gae(records, self.cfg.gamma, self.cfg.gae_lambda)
        if learning_rate is not None:
            if not np.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError("Invalid PPO learning rate")
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate)
        self.network.train()
        old_logp = torch.as_tensor([r.logp for r in records], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor([r.value for r in records], dtype=torch.float32, device=self.device)
        batch_advantages = torch.as_tensor(advantages, device=self.device)
        batch_returns = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(records))
        measurements = []
        epochs = 0
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for minibatch in np.array_split(indices, min(self.cfg.num_minibatches, len(records))):
                new_logp, entropy, new_values = [], [], []
                for index in minibatch:
                    record = records[int(index)]
                    mask = np.asarray(record.state.action_mask).reshape(-1)
                    if not 0 <= record.action < len(mask) or not mask[record.action]:
                        raise ValueError("PPO rollout contains an invalid joint action")
                    distribution, value = self.network.distribution_and_value(record.state)
                    action = torch.tensor(record.action, dtype=torch.long, device=self.device)
                    new_logp.append(distribution.log_prob(action))
                    entropy.append(distribution.entropy())
                    new_values.append(value)
                new_logp = torch.stack(new_logp).reshape(-1)
                new_values = torch.stack(new_values).reshape(-1)
                entropy_loss = torch.stack(entropy).mean()
                selection = torch.as_tensor(minibatch, dtype=torch.long, device=self.device)
                logratio = new_logp - old_logp[selection]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1).abs() > self.cfg.clip_coef).float().mean()
                mb_advantages = batch_advantages[selection]
                if self.cfg.norm_adv:
                    # The final partial rollout can contain a singleton minibatch.
                    std = mb_advantages.std() if len(minibatch) > 1 else mb_advantages.new_zeros(())
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                policy_loss = torch.maximum(
                    -mb_advantages * ratio,
                    -mb_advantages * ratio.clamp(1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef),
                ).mean()
                if self.cfg.clip_vloss:
                    unclipped = (new_values - batch_returns[selection]).square()
                    clipped_values = old_values[selection] + (
                        new_values - old_values[selection]
                    ).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
                    clipped = (clipped_values - batch_returns[selection]).square()
                    value_loss = 0.5 * torch.maximum(unclipped, clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - batch_returns[selection]).square().mean()
                loss = policy_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * value_loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True,
                )
                self.optimizer.step()
                measurements.append([
                    float(loss.detach()), float(policy_loss.detach()), float(value_loss.detach()),
                    float(entropy_loss.detach()), float(approx_kl), float(old_approx_kl),
                    float(clipfrac), float(gradient_norm),
                ])
            epochs += 1
            if self.cfg.target_kl is not None and float(approx_kl) > self.cfg.target_kl:
                break
        self.updates += 1
        keys = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "old_approx_kl", "clipfrac", "grad_norm")
        result = dict(zip(keys, np.mean(measurements, axis=0).tolist()))
        variance = float(np.var(returns))
        result.update(
            transitions=len(records), epochs=epochs, updates=self.updates,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            explained_variance=(None if variance == 0 else float(
                1 - np.var(returns - np.asarray([r.value for r in records])) / variance
            )),
        )
        return result
