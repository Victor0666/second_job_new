"""Performance-only MAHDRL variants; the original learner and physics are reused."""
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch

from common.d3qn_agent_EXDehid import D3QNAgent


@dataclass(frozen=True)
class PerformancePolicy:
    scheme: str
    guided_probability: float = 0.0
    max_duration_ratio: float = 1.0


GUIDED_POLICY = PerformancePolicy("mahdrl_ge_v1", guided_probability=0.5)
FILTER_POLICY = PerformancePolicy("mahdrl_pf_v1", max_duration_ratio=1.25)


def candidate_durations(env, role, mask, task_id=None, host_id=None):
    """Predict completion delay of legal actions, including actual data locality."""
    durations = np.full(np.shape(mask), np.inf, dtype=np.float64)
    tid = env._cur_tid if task_id is None else task_id
    now = float(env.current_time)
    if role == "host":
        for hi in np.flatnonzero(np.asarray(mask) > 0.5):
            # Match the original HA idle tolerance and index-to-host mapping.
            candidates = [j for j in env.host_to_vm_indices[env.host_ids[hi]]
                          if env.vm_available_at[j] <= now + 1e-9]
            durations[hi] = min(max(0.0, float(env.vm_available_at[j]) - now)
                                + env._task_duration(tid, j) for j in candidates)
    elif role == "vm":
        candidates = env.host_to_vm_indices[env._cur_host_id if host_id is None else host_id]
        for slot in np.flatnonzero(np.asarray(mask) > 0.5):
            j = candidates[slot]
            durations[slot] = (max(0.0, float(env.vm_available_at[j]) - now)
                               + env._task_duration(tid, j))
    else:
        raise ValueError("Performance role must be host or vm")
    return durations


def _legal_durations(mask, durations):
    mask = np.asarray(mask)
    durations = np.asarray(durations, dtype=np.float64)
    if mask.ndim != 1 or durations.shape != mask.shape or not np.isfinite(mask).all():
        raise ValueError("Performance durations must match a finite one-dimensional mask")
    legal = mask > 0.5
    if np.any(~np.isfinite(durations[legal])) or np.any(durations[legal] < 0):
        raise ValueError("Legal candidates require finite nonnegative durations")
    return mask, durations, legal


def performance_mask(mask, durations, ratio=FILTER_POLICY.max_duration_ratio):
    """Retain all ties and at least one fastest action; never enable illegal actions."""
    if not np.isfinite(ratio) or ratio < 1.0:
        raise ValueError("Performance duration ratio must be finite and >= 1")
    mask, durations, legal = _legal_durations(mask, durations)
    result = np.array(mask, copy=True)
    if legal.any():
        limit = float(durations[legal].min()) * ratio
        result[legal & (durations > limit + 1e-9)] = 0
    return result


class _PerformanceAgent(D3QNAgent):
    policy = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The existing measurement recorder serializes dataclass configs.
        self.cfg = self.policy

    def save(self, path):
        self.normalization_metadata = dict(
            getattr(self, "normalization_metadata", {"normalization_version": "legacy"}),
            performance_policy=asdict(self.policy),
        )
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("normalization") if isinstance(checkpoint, dict) else None
        if not isinstance(metadata, dict) or metadata.get("performance_policy") != asdict(self.policy):
            raise ValueError("Checkpoint performance-policy mismatch; train the matching MAHDRL variant")
        del checkpoint
        super().load(path, strict=strict, require_learning_model=require_learning_model)


class GuidedExplorationAgent(_PerformanceAgent):
    policy = GUIDED_POLICY

    def select_action(self, state, mask, deterministic=False, count_step=True,
                      performance_times=None):
        # TA and every deterministic evaluation follow the original policy exactly.
        if deterministic or performance_times is None:
            return super().select_action(state, mask, deterministic, count_step)
        mask, durations, legal = _legal_durations(mask, performance_times)
        valid = np.flatnonzero(legal)
        if not valid.size:
            raise ValueError("Guided exploration requires a legal action")
        if count_step:
            self._action_calls += 1
        if random.random() < self.epsilon():
            if random.random() < self.policy.guided_probability:
                best = float(durations[valid].min())
                valid = valid[np.isclose(durations[valid], best, rtol=1e-9, atol=1e-9)]
            return int(np.random.choice(valid))
        return super().select_action(state, mask, deterministic=True, count_step=False)


class PerformanceFilterAgent(_PerformanceAgent):
    policy = FILTER_POLICY
