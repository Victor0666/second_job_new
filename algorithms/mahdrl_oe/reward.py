"""Variant-only reward settlement; no changes to the original F11 helpers."""


def finish_lower_phase(env, host_trace, vm_trace, info):
    # Only a fuse assignment can add a record without returning vm_assign's reward.
    delta = float(info.get("lower_assignment_energy_unemitted", 0.))
    if delta:
        for trace, alpha in ((host_trace, env.alpha_delay_host), (vm_trace, env.alpha_delay_vm)):
            if trace is None or trace.pending is None:
                raise RuntimeError("Forced assignment energy has no preceding learner decision")
            trace.add_reward(-(1. - alpha) * delta * env.energy_reward_scale)
    if env.done_flag:
        truncated = not env._episode_should_end()
        for trace in (host_trace, vm_trace):
            if trace is not None:
                trace.finish(truncated=truncated)


class LowerRewardAudit:
    def __init__(self, host_trace, vm_trace):
        self.host_start = host_trace.recorded_reward if host_trace else 0.
        self.vm_start = vm_trace.recorded_reward if vm_trace else 0.
        self.energy = self.delay = self.phase_energy = 0.

    def assignment(self, info):
        self.energy += float(info.get("r_energy_total", 0.))
        self.delay += float(info.get("r_delay", 0.))

    def phase(self, info, vm_trace):
        self.phase_energy += float(info.get("phase_energy", 0.))
        # The factor is applied in summary, since env carries the actual run scale.
        self.unreturned_j = getattr(self, "unreturned_j", 0.) + float(info.get("lower_assignment_energy_unemitted", 0.))

    def summary(self, env, host_trace, vm_trace):
        energy = self.energy - getattr(self, "unreturned_j", 0.) * env.energy_reward_scale
        physical_gap = env.assignment_energy_total - env.total_energy
        result = dict(physical_energy_balance=self.phase_energy - env.total_energy,
                      lower_energy_balance=energy + env.assignment_energy_total * env.energy_reward_scale,
                      team_energy_reward=energy, selected_host_energy_reward=energy,
                      assignment_energy_total=env.assignment_energy_total,
                      assignment_energy_physical_gap=physical_gap,
                      complete_episode=bool(env._episode_should_end()))
        for name, trace, start, alpha in (("HA", host_trace, self.host_start, env.alpha_delay_host),
                                          ("VA", vm_trace, self.vm_start, env.alpha_delay_vm)):
            if trace is not None:
                result[name + "_reward_formula_balance"] = (trace.recorded_reward - start
                                                           - ((1. - alpha) * energy + alpha * self.delay))
        return result
