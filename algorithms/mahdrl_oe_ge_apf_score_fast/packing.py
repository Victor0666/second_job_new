"""Validated replay owns its arrays; sampling checks entire packed batches."""
from types import MappingProxyType
import numpy as np
import torch
from algorithms.mahdrl_oe_ge_apf_score.networks import validate_observation


def frozen_array(value, dtype=np.float32):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite replay array after dtype conversion")
    result.flags.writeable = False
    return result


def frozen_observation(obs):
    return MappingProxyType({k: frozen_array(obs[k]) for k in ("context", "candidates", "exists_mask")})


def pack(observations, masks):
    if not observations or len(observations) != len(masks):
        raise ValueError("Empty or mismatched replay batch")
    lengths = np.fromiter((len(x) for x in masks), dtype=np.int64, count=len(masks))
    context = np.stack([x["context"] for x in observations]).astype(np.float32, copy=False)
    n, b = max(1, int(lengths.max())), len(masks)
    f = observations[0]["candidates"].shape[1]
    if np.all(lengths == n):
        features = np.stack([x["candidates"] for x in observations]).astype(np.float32, copy=False)
        exists = np.stack([x["exists_mask"] for x in observations]).astype(np.float32, copy=False)
        legal = np.stack(masks).astype(np.float32, copy=False)
    else:
        features = np.zeros((b, n, f), np.float32)
        exists = np.zeros((b, n), np.float32)
        legal = np.zeros_like(exists)
        for k in np.unique(lengths):
            if not k:
                continue
            indices = np.flatnonzero(lengths == k)
            features[indices, :k] = np.stack([observations[i]["candidates"] for i in indices])
            exists[indices, :k] = np.stack([observations[i]["exists_mask"] for i in indices])
            legal[indices, :k] = np.stack([masks[i] for i in indices])
    if (context.ndim != 2 or features.shape != (b, n, f)
            or exists.shape != (b, n) or legal.shape != (b, n)):
        raise ValueError("Invalid packed observation shape")
    if not all(np.isfinite(x).all() for x in (context, features, exists, legal)):
        raise ValueError("Non-finite packed observation")
    if (np.any((exists != 0) & (exists != 1)) or np.any((legal != 0) & (legal != 1))
            or np.any(legal > exists)):
        raise ValueError("Invalid packed masks")
    return context, features, exists, legal


def to_device(arrays, device):
    return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays)


def collate(observations, masks, device):
    return to_device(pack(observations, masks), device)
