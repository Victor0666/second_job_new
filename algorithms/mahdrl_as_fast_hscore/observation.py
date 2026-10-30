"""Translate real local VM rows into stable, zero-padded fixed slots."""
import numpy as np
from algorithms.mahdrl_oe_ge_apf_score.networks import validate_observation
from .config import VM_CAPACITY, VM_CONTEXT_DIM, VM_FEATURE_DIM, VM_INPUT_DIM


def fixed_state(state):
    obs, mask = state["obs"], np.asarray(state["mask"], dtype=np.float32)
    validate_observation(obs, mask, decision=False)
    k = len(mask)
    if (obs["context"].shape != (VM_CONTEXT_DIM,)
            or obs["candidates"].shape != (k, VM_FEATURE_DIM) or k > VM_CAPACITY):
        raise ValueError("Fixed VA requires context30, VM7 and at most 9 local VMs")
    candidates = np.zeros((VM_CAPACITY, VM_FEATURE_DIM), np.float32)
    candidates[:k] = obs["candidates"]
    legal = np.zeros(VM_CAPACITY, np.float32)
    legal[:k] = mask
    result = dict(state, obs=np.concatenate((obs["context"], candidates.ravel())).astype(np.float32), mask=legal)
    if "performance_times" in state:
        times = np.zeros(VM_CAPACITY, np.float64)
        times[:k] = state["performance_times"]
        result["performance_times"] = times
    if "rows" in state:
        rows = np.zeros((VM_CAPACITY, 3), np.float64)
        rows[:k] = state["rows"]
        result["rows"] = rows
    result["candidate_ids"] = list(state.get("candidate_ids", [])) + [None] * (VM_CAPACITY - k)
    return result


def validate_fixed(obs, mask, *, decision=True):
    obs, mask = np.asarray(obs), np.asarray(mask)
    if obs.shape != (VM_INPUT_DIM,) or mask.shape != (VM_CAPACITY,):
        raise ValueError("Fixed VA requires a 93-vector and a 9-action mask")
    if not np.isfinite(obs).all() or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("Invalid fixed VA state or binary mask")
    if decision and not np.any(mask > .5):
        raise ValueError("No legal VM at a non-terminal decision")
