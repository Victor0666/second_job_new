"""Original parameters/BN semantics with owned CPU observations and bounded tensors."""
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from algorithms.drl_gnn.networks import (GraphState, GraphActorCritic, BaseGINNetwork,
    GINActor, GINCritic)
from algorithms.drl_gnn.runtime import configure_cuda_memory
from .config import DRLGNNConfig


def _array(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


@dataclass(frozen=True)
class PreparedGraphState(GraphState):
    edge_index: np.ndarray = field(init=False, repr=False)
    compatibilities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        tasks = _array(self.task_features, np.float32)
        vms = _array(self.vm_features, np.float32)
        compatibility = _array(self.compatibility_mask, np.float64)
        actions = _array(self.action_mask, np.float64)
        if not np.isin(compatibility, (0., 1.)).all() or not np.isin(actions, (0., 1.)).all():
            raise ValueError('Graph masks must be binary')
        compatibility = compatibility.astype(bool)
        actions = actions.astype(bool)
        deps = _array(self.dependencies, np.int64)
        if deps.size == 0:
            deps = deps.reshape(0, 2)
        if (tasks.ndim != 2 or tasks.shape[1] != 8 or len(tasks) < 2
                or vms.ndim != 2 or vms.shape[1] != 9 or len(vms) < 2
                or len(self.task_gids) != len(tasks)):
            raise ValueError('DRL-GNN-Fast needs task8/VM9 and at least two rows for single-graph BN')
        if (compatibility.shape != (len(tasks), len(vms)) or actions.shape != compatibility.shape
                or (actions & ~compatibility).any()
                or not np.isfinite(tasks).all() or not np.isfinite(vms).all()):
            raise ValueError('Invalid graph observation or masks')
        if (deps.ndim != 2 or deps.shape[1] != 2
                or (deps.size and (deps.min() < 0 or deps.max() >= len(tasks)))):
            raise ValueError('Invalid task dependencies')
        compat = np.stack(np.nonzero(compatibility))
        task_vm = compat.copy()
        task_vm[1] += len(tasks)
        edges = np.concatenate((task_vm, deps.T), axis=1)
        if edges.shape[1] < 2:
            raise ValueError('Actor single-graph BN requires two edges')
        for name, array in (('task_features', tasks), ('vm_features', vms),
                            ('compatibility_mask', compatibility), ('action_mask', actions),
                            ('dependencies', deps), ('compatibilities', compat), ('edge_index', edges)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, 'task_gids', tuple(self.task_gids))


def prepare(state):
    if isinstance(state, PreparedGraphState):
        return state
    return PreparedGraphState(state.task_features, state.vm_features, state.dependencies,
                              state.compatibility_mask, state.action_mask, state.task_gids)


class CriticGraphNetwork(BaseGINNetwork):
    def forward(self, task_features, vm_features, edge_index):
        x = torch.cat((self.task_encoder(task_features), self.vm_encoder(vm_features)))
        embeddings = self.graph_network(x, edge_index)
        graph = embeddings.new_zeros((1, embeddings.shape[1]))
        graph.index_add_(0, torch.zeros(len(x), dtype=torch.long, device=x.device), embeddings)
        return None, graph / len(x)


class FastCritic(GINCritic):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.network = CriticGraphNetwork(config)
        self.graph_scorer = nn.Sequential(nn.Linear(config.embedding_dim, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, 1))


class FastGraphActorCritic(GraphActorCritic):
    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.config = config or DRLGNNConfig()
        self.cuda_memory_policy = configure_cuda_memory(self.config.device)
        self.actor = GINActor(self.config)
        self.critic = FastCritic(self.config)
        self.to(self.config.device)
        self._tensor_cache = OrderedDict()

    def clear_cache(self):
        self._tensor_cache.clear()

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.clear_cache()
        self.weight_revision = getattr(self, 'weight_revision', 0) + 1
        return result

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        if hasattr(self, '_tensor_cache'):
            self.clear_cache()
        self.weight_revision = getattr(self, 'weight_revision', 0) + 1
        return result

    def _tensors(self, state):
        state = prepare(state)
        device = next(self.parameters()).device
        key = (id(state), str(device))
        if key in self._tensor_cache:
            self._tensor_cache.move_to_end(key)
            return self._tensor_cache[key][1]
        # Own CPU arrays are validated once; tensor() safely copies readonly storage.
        values = tuple(torch.tensor(a, device=device) for a in (
            state.task_features, state.vm_features, state.edge_index,
            state.compatibilities, state.action_mask))
        self._tensor_cache[key] = (state, values)
        # At most current/next observations for four sampling environments, never a rollout.
        while len(self._tensor_cache) > 8:
            self._tensor_cache.popitem(last=False)
        return values

    def distribution_and_value(self, state, action=None):
        state = prepare(state)
        if not state.action_mask.any():
            raise ValueError('Cannot sample an empty legal action set')
        logits, value = self(state)
        return torch.distributions.Categorical(probs=torch.softmax(logits, dim=0)), value

    def actor_distribution(self, state):
        state = prepare(state)
        if not state.action_mask.any():
            raise ValueError('Cannot sample an empty legal action set')
        logits = self.actor(*self._tensors(state))
        return torch.distributions.Categorical(probs=torch.softmax(logits, dim=0))
