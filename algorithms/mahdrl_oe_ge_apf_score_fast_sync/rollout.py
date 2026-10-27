"""Three unchanged policies; task/phase barriers replace automatic trace updates."""
import numpy as np
from algorithms.mahdrl_oe.reward import finish_lower_phase, LowerRewardAudit
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from .updates import UpdateClock, DecisionTrace


def _assign(env, agents, traces, training):
    host, ok = env.get_host_state_for_next_assignment()
    if not ok:
        return None
    traces["HA"].advance(host["obs"], host["mask"])
    ah = agents["HA"].select_action(host["obs"], host["mask"], deterministic=not training,
        count_step=training, performance_times=host["performance_times"])
    env.host_select(ah)
    vm, ok = env.get_vm_state_for_current_task()
    if not ok:
        raise RuntimeError("Selected APF host has no legal VM")
    traces["VA"].advance(vm["obs"], vm["mask"])
    av = agents["VA"].select_action(vm["obs"], vm["mask"], deterministic=not training,
        count_step=training, performance_times=vm["performance_times"])
    decision_time = float(env.current_time)
    rh, rv, info = env.vm_assign(av)
    traces["HA"].record(host["obs"], host["mask"], ah, rh, host_index=info["host_id"])
    traces["VA"].record(vm["obs"], vm["mask"], av, rv, host_index=info["host_id"])
    row = dict(task_id=int(info["task_id"]), host_id=int(info["host_id"]),
        vm_id=int(env.vm_ids[info["vm_global_idx"]]), decision_time=decision_time,
        time_after_assignment=float(env.current_time), host_action=int(ah), vm_action=int(av),
        ready_at_last_sort=env.last_sort_ready_count,
        weights=[float(x) for x in env.combo_weights])
    return rh, rv, info, row


def run_episode(env, agents, *, training=False, warmup=.03, phase_log=None, assignment_log=None):
    from common.experiment_measurements import episode_metrics
    mode = env.sync_config.mode
    if any(agent.sync_config != env.sync_config for agent in agents.values()):
        raise ValueError("Environment and learners use different synchronization identities")
    clock = UpdateClock(agents, mode, training=training, warmup=warmup)
    traces = {role: DecisionTrace(clock, role) for role in agents}
    audit = LowerRewardAudit(traces["HA"], traces["VA"])
    totals = dict.fromkeys(agents, 0.)
    phase_energy = phase_delay = 0.
    phases = assignments = ta_decisions = 0
    state = mask = action = None
    if mode == "phase":
        state, mask = env.get_manager_state(), env.get_manager_action_mask()
        action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
        ta_decisions += 1
        env.apply_manager_delta(MANAGER_ACTION_TABLE[action])

    while not env.done_flag:
        deferred_row = None
        updates_before = clock.executed.copy()
        while True:
            if mode == "task":
                if not env.can_task_decide():
                    break
                state, mask = env.get_manager_state(), env.get_manager_action_mask()
                traces["TA"].advance(state, mask)
                action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
                traces["TA"].record(state, mask, action, 0.)
                ta_decisions += 1
                env.apply_task_manager_delta(MANAGER_ACTION_TABLE[action])
            result = _assign(env, agents, traces, training)
            if result is None:
                if mode == "task":
                    raise RuntimeError("Real task-level TA decision produced no legal host")
                break
            rh, rv, info, row = result
            audit.assignment(info)
            totals["HA"] += float(rh)
            totals["VA"] += float(rv)
            assignments += 1
            row.update(assignment=assignments, phase=phases, ta_action=int(action))
            if mode == "task":
                if env._phase_preadvanced or env.done_flag:
                    # Settle terminal rewards/experiences before this assignment's sole barrier.
                    deferred_row = row
                    break
                row["optimizer_steps"] = clock.barrier()
            else:
                row["optimizer_steps"] = dict.fromkeys(agents, 0)
            if assignment_log:
                assignment_log(row)

        rm, info = env.finish_phase_and_advance()
        totals["TA"] += float(rm)
        phase_energy += float(info.get("phase_energy", 0.))
        phase_delay += float(info.get("phase_delay", 0.))
        audit.phase(info, traces["VA"])
        finish_lower_phase(env, traces["HA"], traces["VA"], info)
        unreturned = float(info.get("lower_assignment_energy_unemitted", 0.))
        for role, alpha in (("HA", env.alpha_delay_host), ("VA", env.alpha_delay_vm)):
            totals[role] -= (1-alpha)*unreturned*env.energy_reward_scale
        if mode == "task":
            # All nonzero physical interval costs occur at original phase boundaries.
            # Within a same-clock allocation batch, intervening TA rewards are zero.
            traces["TA"].add_reward(float(rm))
            if env.done_flag:
                traces["TA"].finish(truncated=not env._episode_should_end())
            if deferred_row is not None:
                deferred_row["optimizer_steps"] = clock.barrier()
                if assignment_log:
                    assignment_log(deferred_row)
        else:
            next_state, next_mask = env.get_manager_state(), env.get_manager_action_mask()
            if info.get("assign_cnt", 0) > 0:
                if training:
                    agents["TA"].f11.truncation_cutoffs += int(info.get("truncated", False))
                clock.remember("TA", state, mask, action, rm, next_state, next_mask, float(env.done_flag))
            clock.barrier()
            state, mask = next_state, next_mask
        if phase_log:
            phase_log(dict(phase=phases, manager_reward=float(rm),
                optimizer_steps={r: clock.executed[r]-updates_before[r] for r in agents}, **info))
        phases += 1
        if mode == "phase" and not env.done_flag:
            action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
            ta_decisions += 1
            env.apply_manager_delta(MANAGER_ACTION_TABLE[action])

    for role in ("HA", "VA"):
        traces[role].finish(truncated=not env._episode_should_end())
    if mode == "task":
        traces["TA"].finish(truncated=not env._episode_should_end())
    if any(trace.pending is not None for trace in traces.values()):
        raise RuntimeError("Pending decisions survived the episode boundary")
    reward_audit = audit.summary(env, traces["HA"], traces["VA"])
    alpha = float(np.clip(env.manager_alpha_delay, 0., 1.))
    manager_formula = -(alpha*phase_delay/max(env.task_baseline_norm, 1e-12)
        + (1-alpha)*phase_energy*env.energy_reward_scale)
    reward_audit["TA_reward_formula_balance"] = totals["TA"]-manager_formula
    if mode == "task":
        reward_audit["TA_interval_reward_balance"] = traces["TA"].emitted_reward-totals["TA"]
    return dict(phase_count=phases, policy_assignments=assignments,
        forced_assignments=len(env._records)-assignments,
        decision_counts=dict(TA=ta_decisions, HA=assignments, VA=assignments),
        rewards=totals, reward_units="episode_total", reward_audit=reward_audit,
        learning_schedule=clock.summary(), sorting=env.sorting_report(len(env._records)),
        **episode_metrics(env))
