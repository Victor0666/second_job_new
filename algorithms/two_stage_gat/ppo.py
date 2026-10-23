"""Joint task--VM clipped PPO, matching the frozen author's update equations."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import TwoStageGATConfig
from .networks import TwoStageActorCritic


@dataclass(frozen=True)
class Transition:
    state: object
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    next_value: float
    env_id: int = 0


def compute_gae(records, gamma, gae_lambda):
    if not records:
        raise ValueError('A rollout cannot be empty')
    scalars = np.asarray([(r.logp, r.value, r.reward, r.next_value) for r in records])
    if not np.isfinite(scalars).all():
        raise ValueError('Nonfinite rollout values')
    advantages = np.empty(len(records), dtype=np.float32)
    carries = {}
    for i in range(len(records) - 1, -1, -1):
        r = records[i]
        live = 1. - float(r.done)
        delta = r.reward + gamma * live * r.next_value - r.value
        carries[r.env_id] = delta + gamma * gae_lambda * live * carries.get(r.env_id, 0.)
        advantages[i] = carries[r.env_id]
    return advantages, advantages + np.asarray([r.value for r in records], dtype=np.float32)


class PPOAgent:
    def __init__(self, config=None, device=None):
        self.cfg = config or TwoStageGATConfig()
        self.device = torch.device(device or self.cfg.device)
        self.network = TwoStageActorCritic(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate,
                                          eps=self.cfg.adam_eps)
        self.steps = 0
        self.updates = 0

    @torch.no_grad()
    def act(self, state, deterministic=False, count_step=True):
        action, logp, _, value = self.network([state], deterministic=deterministic)
        if not torch.isfinite(logp).all() or not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite action/value')
        if count_step:
            self.steps += 1
        return int(action[0]), float(logp[0]), float(value[0])

    @torch.no_grad()
    def value(self, state):
        value = self.network.value([state])[0]
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite bootstrap value')
        return float(value)

    def update(self, records, learning_rate=None):
        records = tuple(records)
        advantages, returns = compute_gae(records, self.cfg.gamma, self.cfg.gae_lambda)
        if learning_rate is not None:
            if not np.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError('Invalid learning rate')
            for group in self.optimizer.param_groups:
                group['lr'] = float(learning_rate)
        self.network.train()
        old_logp = torch.tensor([r.logp for r in records], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([r.value for r in records], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, device=self.device)
        returns_tensor = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(records))
        measurements = []
        presentations = 0
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for minibatch in np.array_split(indices, min(self.cfg.num_minibatches, len(records))):
                selection = torch.as_tensor(minibatch, dtype=torch.long, device=self.device)
                rows = [records[int(i)] for i in minibatch]
                _, new_logp, entropy, new_values = self.network(
                    [r.state for r in rows], [r.action for r in rows])
                logratio = new_logp - old_logp[selection]
                ratio = logratio.exp()
                mb_adv = advantages[selection]
                if self.cfg.norm_adv:
                    std = mb_adv.std() if len(rows) > 1 else mb_adv.new_zeros(())
                    mb_adv = (mb_adv - mb_adv.mean()) / (std + 1e-8)
                policy_loss = torch.maximum(-mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)).mean()
                if self.cfg.clip_vloss:
                    clipped = old_values[selection] + (new_values - old_values[selection]).clamp(
                        -self.cfg.clip_coef, self.cfg.clip_coef)
                    value_loss = .5 * torch.maximum((new_values - returns_tensor[selection]).square(),
                        (clipped - returns_tensor[selection]).square()).mean()
                else:
                    value_loss = .5 * (new_values - returns_tensor[selection]).square().mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * value_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(),
                    self.cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1).abs() > self.cfg.clip_coef).float().mean()
                measurements.append([float(loss.detach()), float(policy_loss.detach()),
                    float(value_loss.detach()), float(entropy_loss.detach()), float(approx_kl),
                    float(clipfrac), float(grad_norm)])
                presentations += len(rows)
        self.updates += 1
        keys = ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clipfrac', 'grad_norm')
        result = dict(zip(keys, np.mean(measurements, axis=0).tolist()))
        variance = float(np.var(returns))
        result.update(transitions=len(records), epochs=self.cfg.update_epochs, updates=self.updates,
            optimizer_steps=len(measurements), sample_presentations=presentations,
            learning_rate=float(self.optimizer.param_groups[0]['lr']),
            explained_variance=None if variance == 0 else float(1 - np.var(
                returns - np.asarray([r.value for r in records])) / variance))
        return result
