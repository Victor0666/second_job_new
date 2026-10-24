"""Author-code GIN policy, implemented with torch without a PyG dependency.

Retains be67762c3dd22c05733ebb808083375de3bd7d62 / PyG 2.6.1 mechanisms.
The input layers accept public-state task8/VM9 features instead of task4/VM3.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import DRLGNNConfig, TASK_FEATURES, VM_FEATURES


@dataclass(frozen=True)
class GraphState:
    task_features: np.ndarray | torch.Tensor
    vm_features: np.ndarray | torch.Tensor
    dependencies: tuple[tuple[int, int], ...] | np.ndarray | torch.Tensor
    compatibility_mask: np.ndarray | torch.Tensor
    action_mask: np.ndarray | torch.Tensor
    task_gids: tuple[int, ...]


class _PyGLinear(nn.Linear):
    def reset_parameters(self):
        # PyG's default Kaiming bound is algebraically torch.Linear's default.
        bound = math.sqrt(6 / ((1 + math.sqrt(5) ** 2) * self.in_features))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))


class _GINMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lins = nn.ModuleList((_PyGLinear(in_channels, out_channels),
                                  _PyGLinear(out_channels, out_channels)))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lins:
            layer.reset_parameters()

    def forward(self, x):
        return self.lins[1](torch.relu(self.lins[0](x)))


class _GINConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = _GINMLP(in_channels, out_channels)
        self.register_buffer("eps", torch.zeros(1))
        # GINConv resets its MLP again after the MLP's own constructor reset.
        self.nn.reset_parameters()

    def forward(self, x, edge_index):
        neighbors = torch.zeros_like(x)
        neighbors.index_add_(0, edge_index[1], x[edge_index[0]])
        return self.nn(neighbors + (1 + self.eps) * x)


class _GIN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.convs = nn.ModuleList((_GINConv(embedding_dim, hidden_dim),
                                   _GINConv(hidden_dim, hidden_dim),
                                   _GINConv(hidden_dim, embedding_dim)))

    def forward(self, x, edge_index):
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = torch.relu(x)
        return x


def _encoder(input_dim, hidden_dim, embedding_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, embedding_dim))


class BaseGINNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.task_encoder = _encoder(len(TASK_FEATURES), config.hidden_dim, config.embedding_dim)
        self.vm_encoder = _encoder(len(VM_FEATURES), config.hidden_dim, config.embedding_dim)
        self.graph_network = _GIN(config.hidden_dim, config.embedding_dim)

    def forward(self, task_features, vm_features, edge_index):
        x = torch.cat((self.task_encoder(task_features), self.vm_encoder(vm_features)))
        embeddings = self.graph_network(x, edge_index)
        edge_embeddings = torch.cat((embeddings[edge_index[0]], embeddings[edge_index[1]]), dim=1)
        # PyG global_mean_pool uses a sum reduction followed by division.
        graph_embedding = embeddings.new_zeros((1, embeddings.shape[1]))
        graph_embedding.index_add_(0, torch.zeros(len(x), dtype=torch.long, device=x.device), embeddings)
        return edge_embeddings, graph_embedding / len(x)


class GINActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = BaseGINNetwork(config)
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * config.embedding_dim, 2 * config.hidden_dim),
            nn.BatchNorm1d(2 * config.hidden_dim), nn.ReLU(),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, task_features, vm_features, edge_index, compatibilities, action_mask):
        edge_embeddings, graph_embedding = self.network(task_features, vm_features, edge_index)
        features = torch.cat((edge_embeddings, graph_embedding.expand(len(edge_embeddings), -1)), dim=1)
        # Source BN sees all edges, including dependencies, before this slice.
        scores = self.edge_scorer(features).flatten()[:compatibilities.shape[1]]
        logits = scores.new_full((len(task_features), len(vm_features)), -1e8)
        logits[compatibilities[0], compatibilities[1]] = scores
        return logits.masked_fill(~action_mask, -1e8).flatten()


class GINCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = BaseGINNetwork(config)
        self.graph_scorer = nn.Sequential(nn.Linear(config.embedding_dim, config.hidden_dim), nn.ReLU(),
                                          nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                                          nn.Linear(config.hidden_dim, 1))

    def forward(self, task_features, vm_features, edge_index):
        _, graph_embedding = self.network(task_features, vm_features, edge_index)
        return self.graph_scorer(graph_embedding).reshape(())


class GraphActorCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DRLGNNConfig()
        self.actor = GINActor(self.config)
        self.critic = GINCritic(self.config)
        self.to(self.config.device)

    def _tensors(self, state):
        device = next(self.parameters()).device
        tasks = torch.as_tensor(state.task_features, dtype=torch.float32, device=device)
        vms = torch.as_tensor(state.vm_features, dtype=torch.float32, device=device)
        compatibility = torch.as_tensor(state.compatibility_mask, dtype=torch.bool, device=device)
        action_mask = torch.as_tensor(state.action_mask, dtype=torch.bool, device=device)
        dependencies = torch.as_tensor(state.dependencies, dtype=torch.long, device=device)
        if dependencies.numel() == 0:
            dependencies = dependencies.reshape(0, 2)
        if (tasks.ndim != 2 or tasks.shape[1] != len(TASK_FEATURES)
                or vms.ndim != 2 or vms.shape[1] != len(VM_FEATURES)):
            raise ValueError("The public-state policy needs task features [T,8] and VM features [V,9]")
        if len(tasks) < 2 or len(vms) < 2 or len(state.task_gids) != len(tasks):
            raise ValueError("Single-graph source BatchNorm needs at least two task and VM rows")
        if compatibility.shape != (len(tasks), len(vms)) or action_mask.shape != compatibility.shape:
            raise ValueError("Task-VM masks must have shape [T,V]")
        if (action_mask & ~compatibility).any() or not torch.isfinite(tasks).all() or not torch.isfinite(vms).all():
            raise ValueError("Illegal action mask or nonfinite raw graph features")
        if dependencies.ndim != 2 or dependencies.shape[1] != 2:
            raise ValueError("Dependencies must be task-local pairs [D,2]")
        if dependencies.numel() and (dependencies.min() < 0 or dependencies.max() >= len(tasks)):
            raise ValueError("Task dependency index is out of range")
        compatibilities = compatibility.nonzero(as_tuple=False).T.contiguous()
        task_vm_edges = compatibilities.clone()
        task_vm_edges[1] += len(tasks)
        edge_index = torch.cat((task_vm_edges, dependencies.T), dim=1)
        if edge_index.shape[1] < 2:
            raise ValueError("The source actor BatchNorm needs at least two graph edges")
        return tasks, vms, edge_index, compatibilities, action_mask

    def forward(self, state):
        tasks, vms, edges, compatibility, action_mask = self._tensors(state)
        return (self.actor(tasks, vms, edges, compatibility, action_mask),
                self.critic(tasks, vms, edges))

    def distribution_and_value(self, state, action=None):
        if not np.asarray(state.action_mask.detach().cpu() if isinstance(state.action_mask, torch.Tensor)
                          else state.action_mask).any():
            raise ValueError("Cannot sample a task-VM pair from an empty legal action set")
        logits, value = self(state)
        return Categorical(probs=torch.softmax(logits, dim=0)), value

    def value(self, state):
        tasks, vms, edges, _, _ = self._tensors(state)
        return self.critic(tasks, vms, edges)
