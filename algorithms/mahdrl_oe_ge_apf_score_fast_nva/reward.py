"""Settle unreturned OE allocation energy on the retained HA learner only."""
from algorithms.mahdrl_oe.reward import LowerRewardAudit


def finish_lower_phase(env, host_trace, info):
    delta = float(info.get("lower_assignment_energy_unemitted", 0.))
    if delta:
        if host_trace is None or host_trace.pending is None:
            raise RuntimeError("Forced assignment energy has no preceding HA decision")
        host_trace.add_reward(-(1. - env.alpha_delay_host) * delta * env.energy_reward_scale)
    if env.done_flag and host_trace is not None:
        host_trace.finish(truncated=not env._episode_should_end())
