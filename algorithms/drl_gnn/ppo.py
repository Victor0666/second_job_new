"""Source-configured PPO for the adapted DRL-GNN joint policy."""
from dataclasses import dataclass, replace
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
        self.network = GraphActorCritic(replace(self.cfg, device=str(self.device)))
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
                selection = torch.as_tensor(minibatch, dtype=torch.long, device=self.device)
                mb_advantages = batch_advantages[selection]
                if self.cfg.norm_adv:
                    # Normalize the whole minibatch, never individual graphs.
                    std = mb_advantages.std() if len(minibatch) > 1 else mb_advantages.new_zeros(())
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                mb_measurements = torch.zeros(7, device=self.device)
                self.optimizer.zero_grad()
                for position, index in enumerate(minibatch):
                    record = records[int(index)]
                    mask = np.asarray(record.state.action_mask).reshape(-1)
                    if not 0 <= record.action < len(mask) or not mask[record.action]:
                        raise ValueError("PPO rollout contains an invalid joint action")
                    distribution, value = self.network.distribution_and_value(record.state)
                    action = torch.tensor(record.action, dtype=torch.long, device=self.device)
                    new_logp = distribution.log_prob(action)
                    entropy_loss = distribution.entropy()
                    logratio = new_logp - old_logp[index]
                    ratio = logratio.exp()
                    advantage = mb_advantages[position]
                    policy_loss = torch.maximum(
                        -advantage * ratio,
                        -advantage * ratio.clamp(1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef),
                    )
                    if self.cfg.clip_vloss:
                        unclipped = (value - batch_returns[index]).square()
                        clipped_value = old_values[index] + (
                            value - old_values[index]
                        ).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
                        clipped = (clipped_value - batch_returns[index]).square()
                        value_loss = 0.5 * torch.maximum(unclipped, clipped)
                    else:
                        value_loss = 0.5 * (value - batch_returns[index]).square()
                    loss = policy_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * value_loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite PPO loss")
                    # Release each graph's saved activations before forwarding the next.
                    # Divide by the actual size, including singleton/tail minibatches.
                    (loss / len(minibatch)).backward()
                    with torch.no_grad():
                        mb_measurements += torch.stack((
                            loss, policy_loss, value_loss, entropy_loss,
                            (ratio - 1) - logratio, -logratio,
                            ((ratio - 1).abs() > self.cfg.clip_coef).float(),
                        ))
                    del distribution, value, new_logp, loss
                mb_measurements /= len(minibatch)
                approx_kl = mb_measurements[4]
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True,
                )
                self.optimizer.step()
                measurements.append(mb_measurements.cpu().tolist() + [float(gradient_norm)])
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
