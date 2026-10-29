"""Same PPO loss/update budget, bounded graph groups and versioned bootstrap reuse."""
from dataclasses import replace
from typing import Sequence
import numpy as np
import torch
from torch import nn
from algorithms.drl_gnn.ppo import PPOAgent as SourcePPOAgent, Transition, compute_gae
from .config import DRLGNNConfig, GRAPHS_PER_BACKWARD
from .networks import FastGraphActorCritic


class PPOAgent(SourcePPOAgent):
    def __init__(self, config=None, device=None, graphs_per_backward=GRAPHS_PER_BACKWARD):
        if type(graphs_per_backward) is not int or graphs_per_backward not in (1, 2, 4):
            raise ValueError('graphs_per_backward must be 1, 2 or 4')
        self.cfg = config or DRLGNNConfig()
        self.device = torch.device(device if device is not None else self.cfg.device)
        self.network = FastGraphActorCritic(replace(self.cfg, device=str(self.device)))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate, eps=self.cfg.adam_eps)
        self.steps = self.updates = 0
        self.graphs_per_backward = graphs_per_backward
        self._bootstrap = {}
        self.bootstrap_hits = 0

    @torch.no_grad()
    def bootstrap_value(self, state, slot):
        if type(slot) is not int or not 0 <= slot < self.cfg.num_envs:
            raise ValueError('Bootstrap slot must identify a configured sampling environment')
        self.network.train()
        statistics = []
        handles = []
        def capture(module, args):
            x = args[0]
            variance, mean = torch.var_mean(x, dim=0, correction=1)
            statistics.append((module, mean.detach(), variance.detach()))
        for module in self.network.critic.modules():
            if isinstance(module, nn.BatchNorm1d):
                handles.append(module.register_forward_pre_hook(capture))
        try:
            value = float(self.network.value(state))
        finally:
            for handle in handles:
                handle.remove()
        self._bootstrap[slot] = (state, (self.updates, self.network.weight_revision), value, statistics)
        return value

    @staticmethod
    @torch.no_grad()
    def _replay_bn(statistics):
        # Reuse the batch moments, not saved running buffers: other envs may have run since.
        for module, mean, variance in statistics:
            if module.track_running_stats:
                module.num_batches_tracked.add_(1)
                momentum = module.momentum if module.momentum is not None else 1. / float(module.num_batches_tracked)
                module.running_mean.mul_(1. - momentum).add_(mean, alpha=momentum)
                module.running_var.mul_(1. - momentum).add_(variance, alpha=momentum)

    @torch.no_grad()
    def act(self, state, deterministic=False, slot=None):
        self.network.train()
        cached = self._bootstrap.pop(slot, None) if slot is not None else None
        version = (self.updates, self.network.weight_revision)
        if not deterministic and cached is not None and cached[0] is state and cached[1] == version:
            distribution = self.network.actor_distribution(state)
            value = cached[2]
            self._replay_bn(cached[3])
            self.bootstrap_hits += 1
        else:
            distribution, value = self.network.distribution_and_value(state)
        action = distribution.probs.argmax() if deterministic else distribution.sample()
        self.steps += 1
        return int(action), float(distribution.log_prob(action)), float(value)

    @torch.no_grad()
    def select_action(self, state):
        self.network.train()
        distribution = self.network.actor_distribution(state)
        self.steps += 1
        return int(distribution.probs.argmax())

    def update(self, transitions: Sequence[Transition], learning_rate=None):
        self._bootstrap.clear()
        self.network.clear_cache()
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
                group_losses = []
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
                    group_losses.append(loss / len(minibatch))
                    if len(group_losses) == self.graphs_per_backward or position + 1 == len(minibatch):
                        group_loss = torch.stack(group_losses).sum()
                        if not torch.isfinite(group_loss):
                            raise ValueError("Nonfinite PPO loss")
                        group_loss.backward()
                        group_losses.clear()
                        del group_loss
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
        self.network.clear_cache()
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
