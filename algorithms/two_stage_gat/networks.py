"""Author GAT/MLP architecture; variable real-node graphs replace fixed padding."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT

from .config import TwoStageGATConfig


@dataclass(frozen=True)
class GraphState:
    task_features: np.ndarray
    vm_features: np.ndarray
    dependencies: np.ndarray
    task_mask: np.ndarray
    vm_mask: np.ndarray
    task_gids: tuple[int, ...]

    @property
    def action_mask(self):
        return self.task_mask[:, None] & self.vm_mask


def _mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, output_dim))


def _batch(states, device):
    if not states:
        raise ValueError('At least one graph is required')
    tasks, vm_features, edges, batches, masks, vm_masks = [], [], [], [], [], []
    offset = 0
    for i, state in enumerate(states):
        x = torch.as_tensor(state.task_features, dtype=torch.float32, device=device)
        vm = torch.as_tensor(state.vm_features, dtype=torch.float32, device=device)
        edge = torch.as_tensor(state.dependencies, dtype=torch.long, device=device)
        mask = torch.as_tensor(state.task_mask, dtype=torch.bool, device=device)
        vm_mask = torch.as_tensor(state.vm_mask, dtype=torch.bool, device=device)
        if x.ndim != 2 or x.shape[1] != 6 or len(x) < 1 or vm.ndim != 3 or vm.shape[0] != len(x) or vm.shape[2] != 4 or vm.shape[1] < 1:
            raise ValueError('Two-stage GAT requires task[T,6] and task-VM[T,V,4]')
        if mask.shape != (len(x),) or vm_mask.shape != vm.shape[:2] or len(state.task_gids) != len(x):
            raise ValueError('Graph masks/IDs do not match real task and VM rows')
        if edge.ndim != 2 or edge.shape[1] != 2 or (edge.numel() and (edge.min() < 0 or edge.max() >= len(x))):
            raise ValueError('Dependencies must be valid task-local edges [E,2]')
        if not torch.isfinite(x).all() or not torch.isfinite(vm).all():
            raise ValueError('Graph features must be finite')
        tasks.append(x); vm_features.append(vm); edges.append(edge.T + offset)
        batches.append(torch.full((len(x),), i, dtype=torch.long, device=device))
        masks.append(mask); vm_masks.append(vm_mask); offset += len(x)
    return (torch.cat(tasks), torch.cat(edges, dim=1), torch.cat(batches),
            vm_features, masks, vm_masks, [len(x) for x in tasks])


class Encoders(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.task_encoder = GAT(in_channels=6, hidden_channels=cfg.hidden_dim,
            num_layers=cfg.gat_layers, out_channels=cfg.embedding_dim,
            heads=cfg.heads, concat=False, dropout=0., act='relu', norm=None)
        self.vm_encoder = _mlp(4, cfg.hidden_dim, cfg.embedding_dim)

    def contexts(self, data):
        tasks, edges, batch, vm_features, _, _, sizes = data
        h = self.task_encoder(tasks, edge_index=edges)
        task_pool = global_mean_pool(h, batch, size=len(sizes))
        vm_sizes = [vm.shape[1] for vm in vm_features]
        avg_vm = torch.cat([vm.mean(dim=0) for vm in vm_features])
        vm_batch = torch.repeat_interleave(torch.arange(len(sizes), device=h.device),
            torch.tensor(vm_sizes, device=h.device))
        vm_pool = global_mean_pool(self.vm_encoder(avg_vm), vm_batch, size=len(sizes))
        return h, task_pool, vm_pool


class TaskVMActor(Encoders):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.task_decoder = _mlp(3 * cfg.embedding_dim, cfg.hidden_dim, 1)
        self.vm_decoder = _mlp(3 * cfg.embedding_dim, cfg.hidden_dim, 1)

    def forward(self, data, actions=None, deterministic=False):
        _, _, batch, vm_features, task_masks, vm_masks, sizes = data
        h, task_pool, avg_vm_pool = self.contexts(data)
        # Index by each node's graph, rather than repeating pools across samples.
        scores = self.task_decoder(torch.cat((h, task_pool[batch], avg_vm_pool[batch]), dim=-1)).flatten()
        task_scores = scores.split(sizes)
        if actions is not None and len(actions) != len(sizes):
            raise ValueError('One stored joint action is required per graph')
        chosen_tasks, task_logp, task_entropy = [], [], []
        for i, (logits, mask) in enumerate(zip(task_scores, task_masks)):
            if not mask.any():
                raise ValueError('Cannot select from an empty Ready task set')
            dist = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
            if actions is None:
                chosen = logits.masked_fill(~mask, -torch.inf).argmax() if deterministic else dist.sample()
            else:
                action = int(actions[i])
                if not 0 <= action < sizes[i] * vm_features[i].shape[1]:
                    raise ValueError('Stored joint action is out of range')
                chosen = torch.tensor(action // vm_features[i].shape[1], device=logits.device)
            if not mask[chosen]:
                raise ValueError('Stored action selects a non-Ready task')
            chosen_tasks.append(chosen); task_logp.append(dist.log_prob(chosen)); task_entropy.append(dist.entropy())
        selected = [vm[chosen] for vm, chosen in zip(vm_features, chosen_tasks)]
        vm_sizes = [len(vm) for vm in selected]
        encoded = self.vm_encoder(torch.cat(selected))
        vm_batch = torch.repeat_interleave(torch.arange(len(sizes), device=h.device),
            torch.tensor(vm_sizes, device=h.device))
        selected_pool = global_mean_pool(encoded, vm_batch, size=len(sizes))
        vm_scores = self.vm_decoder(torch.cat((encoded, task_pool[vm_batch], selected_pool[vm_batch]), dim=-1)).flatten().split(vm_sizes)
        joint_actions, logps, entropies = [], [], []
        for i, logits in enumerate(vm_scores):
            mask = vm_masks[i][chosen_tasks[i]]
            if not mask.any():
                raise ValueError('Selected task has no compatible VM')
            dist = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
            if actions is None:
                chosen = logits.masked_fill(~mask, -torch.inf).argmax() if deterministic else dist.sample()
            else:
                chosen = torch.tensor(int(actions[i]) % vm_sizes[i], device=h.device)
            if not mask[chosen]:
                raise ValueError('Stored action selects an incompatible VM')
            joint_actions.append(chosen_tasks[i] * vm_sizes[i] + chosen)
            logps.append(task_logp[i] + dist.log_prob(chosen))
            entropies.append(task_entropy[i] + dist.entropy())
        return torch.stack(joint_actions), torch.stack(logps), torch.stack(entropies)


class StateCritic(Encoders):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.network = _mlp(2 * cfg.embedding_dim, cfg.hidden_dim, 1)

    def forward(self, data):
        _, task_pool, vm_pool = self.contexts(data)
        return self.network(torch.cat((task_pool, vm_pool), dim=-1)).flatten()


class TwoStageActorCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or TwoStageGATConfig()
        self.actor = TaskVMActor(cfg)
        self.critic = StateCritic(cfg)
        self.to(cfg.device)

    def forward(self, states, actions=None, deterministic=False):
        data = _batch(states, next(self.parameters()).device)
        action, logp, entropy = self.actor(data, actions, deterministic)
        return action, logp, entropy, self.critic(data)

    def value(self, states):
        return self.critic(_batch(states, next(self.parameters()).device))
