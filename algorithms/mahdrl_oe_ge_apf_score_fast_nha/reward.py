"""Settle unreturned OE allocation energy on the sole retained lower learner."""
from algorithms.mahdrl_oe.reward import LowerRewardAudit


def finish_lower_phase(env, vm_trace, info):
    delta = float(info.get("lower_assignment_energy_unemitted", 0.))
    if delta:
        if vm_trace is None or vm_trace.pending is None:
            raise RuntimeError("Forced assignment energy has no preceding VA decision")
        vm_trace.add_reward(-(1. - env.alpha_delay_vm) * delta * env.energy_reward_scale)
    if env.done_flag and vm_trace is not None:
        vm_trace.finish(truncated=not env._episode_should_end())
