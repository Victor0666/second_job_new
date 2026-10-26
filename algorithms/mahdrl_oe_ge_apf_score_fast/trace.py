"""Same real-decision boundaries; complete transition validation lives at insertion."""
from algorithms.mahdrl_oe_ge_apf_score.trace import SetDecisionTrace as OriginalTrace
from .packing import frozen_array, frozen_observation


def snapshot_decision(state):
    # Public AS observations own their storage and are never mutated by the event engine.
    return state


class SetDecisionTrace(OriginalTrace):
    def advance(self, obs, mask, host_index=None):
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError("Previous decision has no actual successor yet")
        self.pending = dict(obs=frozen_observation(obs), mask=frozen_array(mask),
                            action=int(action), reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, obs, mask, bool(done), next_host_index)
        self.emitted_reward += p["reward"]
        self.pending = None
