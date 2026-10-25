"""Parameter shapes depend on feature dimensions, never on candidate counts."""
import numpy as np
import torch
from torch import nn
from .config import TRUNK, ENTITY


def mlp(dimensions):
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def copy_observation(obs):
    return {key: np.array(obs[key], dtype=np.float32, copy=True)
            for key in ("context", "candidates", "exists_mask")}


def validate_observation(obs, mask, *, decision=True):
    context, candidates, exists = (np.asarray(obs[k]) for k in
                                   ("context", "candidates", "exists_mask"))
    mask = np.asarray(mask)
    if (context.ndim != 1 or candidates.ndim != 2 or exists.ndim != 1
            or mask.shape != exists.shape or len(candidates) != len(exists)):
        raise ValueError("Invalid set observation/mask shapes")
    if not all(np.isfinite(x).all() for x in (context, candidates, exists, mask)):
        raise ValueError("Non-finite set observation")
    if not all(np.isin(x, (0, 1)).all() for x in (exists, mask)):
        raise ValueError("Set masks must be binary")
    if np.any(mask > exists):
        raise ValueError("Padding cannot be a legal action")
    if decision and not np.any(mask > .5):
        raise ValueError("No legal action at a non-terminal decision")


def collate(observations, masks, device):
    # At least one dummy row makes terminal-only batches well defined.
    n = max(1, max(len(x["exists_mask"]) for x in observations))
    b, f = len(observations), observations[0]["candidates"].shape[1]
    features = np.zeros((b, n, f), np.float32)
    exists = np.zeros((b, n), np.float32)
    legal = np.zeros_like(exists)
    for i, (obs, mask) in enumerate(zip(observations, masks)):
        validate_observation(obs, mask, decision=False)
        k = len(mask)
        features[i, :k], exists[i, :k], legal[i, :k] = obs["candidates"], obs["exists_mask"], mask
    return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in
                 (np.stack([x["context"] for x in observations]), features, exists, legal))


class CandidateDuelingQNet(nn.Module):
    def __init__(self, context_dim, candidate_dim, hidden_dims=TRUNK, entity_dims=ENTITY):
        super().__init__()
        self.context_dim, self.candidate_dim = context_dim, candidate_dim
        self.entity = mlp((candidate_dim, *entity_dims))
        self.entity_activation = nn.ReLU()
        e, h = entity_dims[-1], hidden_dims[-1]
        self.feature = mlp((context_dim + 2 * e + 2, *hidden_dims))
        self.feature_activation = nn.ReLU()
        self.val = mlp((h, h, h, 1))
        self.adv = mlp((h + e, h, h, 1))

    def forward(self, context, candidates, exists_mask, action_mask):
        if context.shape[-1] != self.context_dim or candidates.shape[-1] != self.candidate_dim:
            raise ValueError("Candidate feature schema mismatch")
        e = self.entity_activation(self.entity(candidates))
        exists, legal = exists_mask > .5, action_mask > .5
        n = exists.sum(1, keepdim=True)
        mean = (e * exists.unsqueeze(-1)).sum(1) / n.clamp_min(1)
        maximum = e.masked_fill(~exists.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(n > 0, maximum, torch.zeros_like(maximum))
        counts = torch.cat((n, legal.sum(1, keepdim=True)), 1).to(e.dtype).log1p()
        h = self.feature_activation(self.feature(torch.cat((context, mean, maximum, counts), 1)))
        a = self.adv(torch.cat((h.unsqueeze(1).expand(-1, e.shape[1], -1), e), -1)).squeeze(-1)
        center = (a * legal).sum(1, keepdim=True) / legal.sum(1, keepdim=True).clamp_min(1)
        return self.val(h) + a - center
