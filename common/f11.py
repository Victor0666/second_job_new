"""F11 decision-boundary transitions and optional learning diagnostics."""
import json
import os
from pathlib import Path

import numpy as np
from common.f19_numerics import NumericalDiagnostics

LEARNING_MODEL = 'f11_real_decisions_v1'


def valid_decision(obs, mask):
    if not np.isfinite(obs).all() or not np.isfinite(mask).all():
        raise ValueError('Non-finite decision state/mask')
    if not np.any(np.asarray(mask) > 0.5):
        raise ValueError('No legal action at a non-terminal decision')


def snapshot_decision(state):
    return {**state, 'obs': np.array(state['obs'], copy=True),
            'mask': np.array(state['mask'], copy=True)}


def validate_transition(state, mask, action, reward, next_state, next_mask, done):
    valid_decision(state, mask)
    if not 0 <= int(action) < len(mask) or mask[int(action)] <= 0.5:
        raise ValueError('Illegal recorded action')
    if not np.isfinite(reward) or not np.isfinite(next_state).all() or not np.isfinite(next_mask).all():
        raise ValueError('Non-finite transition')
    if done not in (0, 1, False, True):
        raise ValueError('Learning stop flag must be boolean')
    if not done:
        valid_decision(next_state, next_mask)


class LearningDiagnostics:
    def __init__(self):
        self.enabled = os.environ.get('F11_DIAGNOSTICS', '1') != '0'
        self.numerics = NumericalDiagnostics()
        self.transitions = self.nonterminal = self.updates = 0
        self.targets = self.bootstrap_nonzero = 0
        self.future_abs_sum = self.target_sum = 0.0
        self.last_loss = self.max_abs_loss = 0.0
        self.last_rollout_value = None
        self.nonterminal_rollouts = self.truncation_cutoffs = 0
        self.last_nonterminal_rollout_value = None

    def transition(self, done):
        if self.enabled:
            self.transitions += 1
            self.nonterminal += int(not done)

    def target(self, future, target):
        if self.enabled:
            f = future.detach().cpu().numpy().reshape(-1)
            t = target.detach().cpu().numpy().reshape(-1)
            self.targets += len(t)
            self.bootstrap_nonzero += int(np.count_nonzero(np.abs(f) > 1e-12))
            self.future_abs_sum += float(np.abs(f).sum())
            self.target_sum += float(t.sum())

    def loss(self, loss):
        self.numerics.observe("loss", loss, scalar=True)
        value = float(loss.detach().item())
        if not np.isfinite(value):
            raise FloatingPointError('Non-finite learning loss')
        if self.enabled:
            self.last_loss = value
            self.max_abs_loss = max(self.max_abs_loss, abs(value))
        return value

    def snapshot(self):
        return dict(transitions=self.transitions, nonterminal=self.nonterminal,
                    nonterminal_ratio=self.nonterminal / max(self.transitions, 1),
                    gradient_updates=self.updates, sampled_targets=self.targets,
                    last_loss=self.last_loss, max_abs_loss=self.max_abs_loss,
                    bootstrap_nonzero_ratio=self.bootstrap_nonzero / max(self.targets, 1),
                    future_abs_mean=self.future_abs_sum / max(self.targets, 1),
                    target_mean=self.target_sum / max(self.targets, 1),
                    last_rollout_value=self.last_rollout_value,
                    nonterminal_rollouts=self.nonterminal_rollouts,
                    last_nonterminal_rollout_value=self.last_nonterminal_rollout_value,
                    truncation_cutoffs=self.truncation_cutoffs,
                    terminal_transitions=self.transitions - self.nonterminal - self.truncation_cutoffs)


class F11Logger:
    def __init__(self, directory):
        self.path = Path(directory) / 'f11_diagnostics.jsonl'

    def log(self, episode, agents, **fields):
        if os.environ.get('F11_DIAGNOSTICS', '1') == '0':
            return
        record = dict(learning_model=LEARNING_MODEL, episode=episode,
                      learners={k: v.f11.snapshot() for k, v in agents.items()},
                      numerics={k: v.f11.numerics.snapshot() for k,v in agents.items()}, **fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')


def protect_training_outputs(save_dir, log_dir):
    """Versioned output names protect legacy artifacts; also refuse reusing a completed run."""
    for directory in (save_dir, log_dir):
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Refusing to overwrite training artifacts: {path}')


class PendingDecision:
    """One learner's consecutive actual decisions, independent of phases or host changes."""
    def __init__(self, emit):
        self.emit = emit
        self.pending = None
        self.recorded_reward = self.emitted_reward = 0.0
        self.cutoffs = 0

    def advance(self, obs, mask, host_index=None):
        valid_decision(obs, mask)
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError('Previous decision has no actual successor yet')
        valid_decision(obs, mask)
        self.pending = dict(obs=np.array(obs, copy=True), mask=np.array(mask, copy=True),
                            action=int(action), reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def add_reward(self, reward):
        if self.pending is None:
            if abs(reward) > 1e-12:
                raise RuntimeError('Reward without a preceding learner decision')
            return
        self.pending['reward'] += float(reward)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, np.array(obs, copy=True), np.array(mask, copy=True), bool(done), next_host_index)
        self.emitted_reward += p['reward']
        self.pending = None

    def finish(self, *, truncated=False):
        # No real successor is supplied by the stopped hierarchical env: explicit cutoff.
        if self.pending is not None:
            self.cutoffs += int(truncated)
            self._emit(np.zeros_like(self.pending['obs']), np.zeros_like(self.pending['mask']), True, None)
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError('Unsettled decision rewards')


class DQNDecisionTrace(PendingDecision):
    def __init__(self, agent, warmup_frac):
        self.agent = agent
        def emit(p, obs, mask, done, next_host):
            agent.remember(p['obs'], p['mask'], p['action'], p['reward'], obs, mask, float(done))
            capacity = getattr(agent.buffer, 'maxlen', None) or getattr(agent, 'buffer_size', 0)
            if len(agent.buffer) >= max(agent.batch_size, int(capacity * warmup_frac)):
                agent.update()
        super().__init__(emit)

    def finish(self, *, truncated=False):
        if truncated and self.pending is not None:
            self.agent.f11.truncation_cutoffs += 1
        super().finish(truncated=truncated)


class LocalDecisionTrace(PendingDecision):
    def __init__(self, agent):
        self.agent = agent
        def emit(p, obs, mask, done, next_host):
            agent.update_one_step(p['obs'], p['mask'], p['action'], p['reward'], obs, mask,
                                  float(done), host_index=p['host_index'], next_host_index=next_host)
        super().__init__(emit)

    def finish(self, *, truncated=False):
        if truncated and self.pending is not None:
            self.agent.f11.truncation_cutoffs += 1
        super().finish(truncated=truncated)


def pd3qn_transition(agent, obs, mask, action, reward, next_obs, next_mask, terminated, truncated):
    # A time limit with a valid returned decision bootstraps, but never connects episodes.
    stop = bool(terminated)
    if truncated and not terminated and not np.any(np.asarray(next_mask) > 0.5):
        stop = True
        agent.f11.truncation_cutoffs += 1
    agent.remember(obs, mask, action, reward, next_obs, next_mask, float(stop))


def finish_lower_phase(env, host_trace, vm_trace, info):
    """Only energy not already returned by vm_assign belongs to this extra advance."""
    if not info.get('lower_energy_already_emitted', True):
        energy = float(info.get('r_energy_phase', 0.0))
        if host_trace is not None:
            host_trace.add_reward((1 - env.alpha_delay_host) * energy)
        if vm_trace is not None and vm_trace.pending is not None:
            host = vm_trace.pending['host_index']
            local = info.get('r_energy_by_host', {}).get(host, 0.0)
            vm_energy = energy if getattr(env, 'vm_energy_scope', 'selected_host') == 'global' else local
            vm_trace.add_reward((1 - env.alpha_delay_vm) * vm_energy)
    if env.done_flag:
        truncated = not env._episode_should_end()
        for trace in (host_trace, vm_trace):
            if trace is not None:
                trace.finish(truncated=truncated)


class LowerRewardAudit:
    """Conservation checks for existing team HA and selected-host VA event rewards."""
    def __init__(self, host_trace, vm_trace):
        self.host_start = host_trace.recorded_reward if host_trace else 0.0
        self.vm_start = vm_trace.recorded_reward if vm_trace else 0.0
        self.energy = self.local_energy = self.delay = self.phase_energy = 0.0

    def assignment(self, info):
        self.energy += float(info.get('r_energy_total', 0.0))
        self.local_energy += float(info.get('r_energy_host', 0.0))
        self.delay += float(info.get('r_delay', 0.0))

    def phase(self, info, vm_trace):
        self.phase_energy += float(info.get('phase_energy', 0.0))
        if not info.get('lower_energy_already_emitted', True):
            self.energy += float(info.get('r_energy_phase', 0.0))
            if vm_trace and vm_trace.pending is not None:
                self.local_energy += float(info.get('r_energy_by_host', {}).get(vm_trace.pending['host_index'], 0.0))

    def summary(self, env, host_trace, vm_trace):
        result = dict(physical_energy_balance=self.phase_energy - float(env.total_energy),
                      lower_energy_balance=self.energy + float(env.total_energy) * env.energy_reward_scale,
                      team_energy_reward=self.energy, selected_host_energy_reward=self.local_energy)
        for name, trace, start, alpha, energy in (
            ('HA', host_trace, self.host_start, env.alpha_delay_host, self.energy),
            ('VA', vm_trace, self.vm_start, env.alpha_delay_vm,
             self.energy if getattr(env, 'vm_energy_scope', 'selected_host') == 'global' else self.local_energy)):
            if trace is not None:
                result[name + '_reward_formula_balance'] = (trace.recorded_reward - start
                                                            - ((1 - alpha) * energy + alpha * self.delay))
        return result
