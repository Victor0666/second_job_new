"""Retained HScore learners, real successors and explicit assignment/phase barriers."""
import numpy as np
import torch
from common.experiment_measurements import episode_metrics
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore.agent import TAAgent, HAAgent, FixedVAAgent
from algorithms.mahdrl_as_fast_hscore.config import TA_HIDDEN, learning_kwargs
from algorithms.mahdrl_as_fast_hscore.preview import prepare_decision
from algorithms.mahdrl_oe.reward import LowerRewardAudit
from environments.mahdrl_as_fast_hscore_ablation import CloudWorkflowEnv_VMAgents
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from .updates import UpdateClock, DecisionTrace
from .preview import uniform_action, prepare_random_decision


def build_env(scene, seed):
    kwargs = base.build_env_kwargs(scene, seed)
    kwargs["manager_alpha_delay"] = scene.variant.alpha
    env = CloudWorkflowEnv_VMAgents(**kwargs, variant=scene.variant,
        policy_seed=scene.model_seed*1000003, mechanism=scene.mechanism,
        experiment_metadata=scene.variant_metadata(), profile_mode=scene.profile_mode,
        profile_every=scene.profile_every)
    env.energy_reward_scale, env.task_baseline_norm, env.energy_norm_per_mi_ref = .001, 300., 20.
    return env


def build_agents(scene):
    device = torch.device(scene.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    options = dict(diagnostics_every=scene.diagnostics_every, profile_mode=scene.profile_mode,
                   profile_every=scene.profile_every)
    constructors = dict(TA=lambda: TAAgent(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN,
                                         **options, **learning_kwargs("TA", device)),
        HA=lambda: HAAgent(23, scene.mechanism.host_features, **options, **learning_kwargs("HA", device)),
        VA=lambda: FixedVAAgent(**options, **learning_kwargs("VA", device)))
    return {role: constructors[role]().bind(scene, role) for role in scene.learned_roles}


def load_agents(scene):
    return base.load_agents(scene, agent_builder=build_agents)


def train(scene):
    return base.train(scene, env_builder=build_env, agent_builder=build_agents, rollout=run_episode)


def evaluate(scene):
    return base.evaluate(scene, env_builder=build_env, agent_builder=build_agents, rollout=run_episode)


def run_episode(env, agents, *, training=False, warmup=.03, phase_log=None, assignment_log=None):
    cfg = env.variant
    if set(agents) != set(cfg.roles) or any(a.mechanism != env.mechanism for a in agents.values()):
        raise ValueError("Mismatched HScore ablation roles or mechanisms")
    clock = UpdateClock(agents, training=training, warmup=warmup)
    traces = {r: DecisionTrace(clock, r) for r in agents if r != "TA" or cfg.suffix == "sync_task"}
    ht, vt = traces.get("HA"), traces.get("VA")
    audit = LowerRewardAudit(ht, vt)
    totals = dict.fromkeys(agents, 0.)
    decisions = dict(TA=0, HA=0, VA=0)
    phases = assignments = 0
    phase_energy = phase_delay = 0.
    state = mask = action = None

    def phase_ta():
        nonlocal state, mask, action
        if cfg.ordering or cfg.suffix == "sync_task":
            return
        state, mask = env.get_manager_state(), env.get_manager_action_mask()
        action = (agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
                  if "TA" in agents else uniform_action(mask, env.policy_rng))
        decisions["TA"] += 1
        env.apply_manager_delta(MANAGER_ACTION_TABLE[action])

    phase_ta()
    while not env.done_flag:
        before = clock.executed.copy()
        while True:
            if cfg.suffix == "sync_task":
                if not env.can_task_decide():
                    break
                state, mask = env.get_manager_state(), env.get_manager_action_mask()
                traces["TA"].advance(state, mask)
                action = agents["TA"].select_action(state, mask, deterministic=not training, count_step=training)
                traces["TA"].record(state, mask, action, 0.)
                decisions["TA"] += 1
                env.apply_task_manager_delta(MANAGER_ACTION_TABLE[action])
            host, ok = env.get_host_state_for_next_assignment()
            if not ok:
                if cfg.suffix == "sync_task":
                    raise RuntimeError("Task-level TA decision produced no legal host")
                break
            if "VA" in agents:
                host, previews, extra = prepare_decision(env, agents["VA"], host, training=training)
            else:
                host, previews, extra = prepare_random_decision(env, host)
            if ht:
                ht.advance(host["obs"], host["mask"])
                ah = agents["HA"].select_action(host["obs"], host["mask"], deterministic=not training,
                    count_step=training, performance_times=host["performance_times"])
            else:
                ah = uniform_action(host["mask"], env.policy_rng)
            hid = env.host_ids[ah]
            env.host_select(ah)
            vm = previews[hid]
            if vt:
                vt.advance(vm["obs"], vm["mask"])
                av = agents["VA"].select_preview_action(vm, training=training)
            else:
                av = uniform_action(vm["mask"], env.policy_rng)
            rh, rv, info = env.vm_assign(av)
            audit.assignment(info)
            for role, trace, obs, legal, chosen, reward in (
                ("HA", ht, host["obs"], host["mask"], ah, rh),
                ("VA", vt, vm["obs"], vm["mask"], av, rv)):
                decisions[role] += 1
                if trace:
                    trace.record(obs, legal, chosen, reward, host_index=hid)
                    totals[role] += float(reward)
            assignments += 1
            row = dict(assignment=assignments, phase=phases, task_id=int(info["task_id"]),
                host_id=int(hid), vm_id=int(env.vm_ids[info["vm_global_idx"]]), host_action=int(ah),
                vm_action=int(av), ta_action=None if action is None else int(action),
                va_version=vm["va_version"], va_epsilon=vm["epsilon"], preview_features=extra[ah].tolist(),
                assignment_energy=float(info["assignment_energy"]))
            if assignment_log:
                assignment_log(row)
            # A preview and both allocation actions always use one frozen VA version.
            at_end = cfg.suffix == "sync_task" and (env._phase_preadvanced or env.done_flag)
            if cfg.suffix != "sync_phase" and not at_end:
                clock.barrier()
            if at_end:
                break
        rm, info = env.finish_phase_and_advance()
        phase_energy += float(info.get("phase_energy", 0.))
        phase_delay += float(info.get("phase_delay", 0.))
        audit.phase(info, vt)
        unreturned = float(info.get("lower_assignment_energy_unemitted", 0.))
        for role, trace, alpha in (("HA", ht, env.alpha_delay_host), ("VA", vt, env.alpha_delay_vm)):
            if trace:
                if unreturned:
                    if trace.pending is None:
                        raise RuntimeError("Forced assignment energy has no preceding learner decision")
                    reward = -(1-alpha)*unreturned*env.energy_reward_scale
                    trace.add_reward(reward)
                    totals[role] += reward
                if env.done_flag:
                    trace.finish(truncated=not env._episode_should_end())
        if "TA" in agents:
            totals["TA"] += float(rm)
            if cfg.suffix == "sync_task":
                traces["TA"].add_reward(float(rm))
                if env.done_flag:
                    traces["TA"].finish(truncated=not env._episode_should_end())
            elif info.get("assign_cnt", 0) > 0:
                if training:
                    agents["TA"].f11.truncation_cutoffs += int(info.get("truncated", False))
                next_state, next_mask = env.get_manager_state(), env.get_manager_action_mask()
                if env.done_flag:
                    next_state, next_mask = np.zeros_like(next_state), np.zeros_like(next_mask)
                clock.remember("TA", state, mask, action, rm, next_state, next_mask, float(env.done_flag))
        clock.barrier()
        if phase_log:
            phase_log(dict(phase=phases, manager_reward=float(rm),
                optimizer_steps={r: clock.executed[r]-before[r] for r in agents}, **info))
        phases += 1
        if not env.done_flag:
            phase_ta()
    for trace in traces.values():
        trace.finish(truncated=not env._episode_should_end())
    clock.barrier()
    reward_audit = audit.summary(env, ht, vt)
    if "TA" in agents:
        formula = -(cfg.alpha*phase_delay/env.task_baseline_norm + (1-cfg.alpha)*phase_energy*env.energy_reward_scale)
        reward_audit["TA_reward_formula_balance"] = totals["TA"]-formula
        if cfg.suffix == "sync_task":
            reward_audit["TA_interval_reward_balance"] = traces["TA"].emitted_reward-totals["TA"]
    return dict(phase_count=phases, policy_assignments=assignments, decision_counts=decisions,
        rewards=totals, reward_units="episode_total", reward_audit=reward_audit,
        update_clock=clock.summary(), **episode_metrics(env))
