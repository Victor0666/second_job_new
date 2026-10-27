"""Only complete real transitions enter replay; optimizers run at explicit barriers."""
import numpy as np
from common.f11 import PendingDecision
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import frozen_array, frozen_observation


class UpdateClock:
    def __init__(self, agents, mode, *, training=False, warmup=.03):
        if mode not in ("task", "phase") or set(agents) != {"TA", "HA", "VA"}:
            raise ValueError("Invalid synchronization mode or learner roles")
        if not 0 <= warmup <= 1:
            raise ValueError("Warmup fraction must be in [0, 1]")
        self.agents, self.mode, self.training, self.warmup = agents, mode, training, warmup
        self.debt = dict.fromkeys(agents, 0)
        self.inserted = dict.fromkeys(agents, 0)
        self.requested = dict.fromkeys(agents, 0)
        self.executed = dict.fromkeys(agents, 0)
        self.barriers = 0

    def eligible(self, role):
        agent = self.agents[role]
        return len(agent.buffer) >= max(agent.batch_size, int(agent.buffer.maxlen*self.warmup))

    def remember(self, role, *transition):
        if not self.training:
            return
        self.agents[role].remember(*transition)
        self.inserted[role] += 1
        if self.mode == "phase" and self.eligible(role):
            self.debt[role] += 1
            self.requested[role] += 1

    def barrier(self):
        self.barriers += 1
        applied = dict.fromkeys(self.agents, 0)
        if not self.training:
            return applied
        for role, agent in self.agents.items():
            if self.mode == "task":
                count = int(self.eligible(role))
                self.requested[role] += count
            else:
                count = self.debt[role]
            for _ in range(count):
                before = agent._updates
                agent.update()
                if agent._updates != before+1:
                    raise RuntimeError("An eligible synchronous update did not execute")
                self.executed[role] += 1
                applied[role] += 1
            self.debt[role] = 0
        return applied

    def summary(self):
        if any(self.debt.values()) or self.executed != self.requested:
            raise RuntimeError("Unsettled synchronization update credits")
        return dict(mode=self.mode, barriers=self.barriers, transitions=self.inserted.copy(),
            update_requests=self.requested.copy(), optimizer_steps=self.executed.copy(),
            pending_credits=self.debt.copy())


class DecisionTrace(PendingDecision):
    def __init__(self, clock, role):
        self.clock, self.role = clock, role
        self.agent = clock.agents[role]
        def emit(p, obs, mask, done, next_host):
            clock.remember(role, p["obs"], p["mask"], p["action"], p["reward"], obs, mask, float(done))
        super().__init__(emit)

    def advance(self, obs, mask, host_index=None):
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError("Previous decision has no actual successor yet")
        copy = frozen_observation if self.agent.is_set else frozen_array
        self.pending = dict(obs=copy(obs), mask=frozen_array(mask), action=int(action),
            reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, obs, mask, bool(done), next_host_index)
        self.emitted_reward += p["reward"]
        self.pending = None

    def finish(self, *, truncated=False):
        if self.pending is not None:
            self.cutoffs += int(truncated)
            if self.clock.training:
                self.agent.f11.truncation_cutoffs += int(truncated)
            obs = self.pending["obs"]
            if self.agent.is_set:
                empty = dict(context=np.zeros_like(obs["context"]),
                    candidates=np.zeros((0, obs["candidates"].shape[1]), np.float32),
                    exists_mask=np.zeros(0, np.float32))
                mask = np.zeros(0, np.float32)
            else:
                empty, mask = np.zeros_like(obs), np.zeros_like(self.pending["mask"])
            self._emit(empty, mask, True, None)
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError("Unsettled synchronization decision rewards")
