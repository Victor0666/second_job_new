"""Complete real transitions first; optimize VA only after assignment."""
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.updates import DecisionTrace


class UpdateClock:
    def __init__(self, agents, *, training=False, warmup=.03):
        if set(agents) != {"TA", "VA"} or not 0 <= warmup <= 1:
            raise ValueError("VAware TL requires only TA/VA and a valid warmup")
        self.agents, self.training, self.warmup = agents, training, warmup
        self.debt = dict.fromkeys(agents, 0)
        self.inserted = dict.fromkeys(agents, 0)
        self.requested = dict.fromkeys(agents, 0)
        self.executed = dict.fromkeys(agents, 0)

    def remember(self, role, *transition):
        if not self.training:
            return
        agent = self.agents[role]
        agent.remember(*transition)
        self.inserted[role] += 1
        if len(agent.buffer) >= max(agent.batch_size, int(agent.buffer.maxlen*self.warmup)):
            self.debt[role] += 1
            self.requested[role] += 1

    def barrier(self):
        for role, count in self.debt.items():
            for _ in range(count):
                agent = self.agents[role]
                before = agent._updates
                agent.update()
                if agent._updates != before+1:
                    raise RuntimeError("A scheduled TL update did not execute")
                self.executed[role] += 1
            self.debt[role] = 0

    def summary(self):
        if any(self.debt.values()) or self.executed != self.requested:
            raise RuntimeError("Unsettled TL update credits")
        return dict(transitions=self.inserted, update_requests=self.requested, optimizer_steps=self.executed)
