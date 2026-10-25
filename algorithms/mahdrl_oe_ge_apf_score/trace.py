"""Variant-local set snapshots with the original F11 decision boundaries."""
import numpy as np
from common.f11 import PendingDecision
from .networks import copy_observation, validate_observation


def snapshot_decision(state):
    return dict(state, obs=copy_observation(state["obs"]), mask=np.array(state["mask"], copy=True))


class SetDecisionTrace(PendingDecision):
    def __init__(self, agent, warmup_frac=.03):
        self.agent = agent
        def emit(p, obs, mask, done, next_host):
            agent.remember(p["obs"], p["mask"], p["action"], p["reward"], obs, mask, float(done))
            if len(agent.buffer) >= max(agent.batch_size, int(agent.buffer.maxlen*warmup_frac)):
                agent.update()
        super().__init__(emit)

    def advance(self, obs, mask, host_index=None):
        validate_observation(obs, mask)
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError("Previous decision has no actual successor yet")
        validate_observation(obs, mask)
        self.pending = dict(obs=copy_observation(obs), mask=np.array(mask, copy=True), action=int(action),
                            reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, copy_observation(obs), np.array(mask, copy=True), bool(done), next_host_index)
        self.emitted_reward += p["reward"]
        self.pending = None

    def finish(self, *, truncated=False):
        if self.pending is not None:
            self.cutoffs += int(truncated)
            self.agent.f11.truncation_cutoffs += int(truncated)
            p = self.pending["obs"]
            empty = dict(context=np.zeros_like(p["context"]),
                         candidates=np.zeros((0, p["candidates"].shape[1]), np.float32),
                         exists_mask=np.zeros(0, np.float32))
            self._emit(empty, np.zeros(0, np.float32), True, None)
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError("Unsettled set decision rewards")
