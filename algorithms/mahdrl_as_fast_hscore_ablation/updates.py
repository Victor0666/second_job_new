"""Transition credits with optional deferred phase barriers and retained roles."""
from algorithms.mahdrl_as_fast_vaaware.updates import UpdateClock as BaseClock, DecisionTrace


class UpdateClock(BaseClock):
    def __init__(self, agents, *, training=False, warmup=.03):
        if not agents or not set(agents) <= {"TA", "HA", "VA"} or not 0 <= warmup <= 1:
            raise ValueError("Invalid learner roles or warmup")
        self.agents, self.training, self.warmup = agents, training, warmup
        self.debt = dict.fromkeys(agents, 0)
        self.inserted = dict.fromkeys(agents, 0)
        self.requested = dict.fromkeys(agents, 0)
        self.executed = dict.fromkeys(agents, 0)

    def summary(self):
        if any(self.debt.values()) or self.executed != self.requested:
            raise RuntimeError("Unsettled HScore ablation update credits")
        return dict(transitions=self.inserted.copy(), update_requests=self.requested.copy(),
            optimizer_steps=self.executed.copy(), pending_credits=self.debt.copy(),
            ha_eligible_steps=self.agents["HA"].eligible_steps if "HA" in self.agents else None)
