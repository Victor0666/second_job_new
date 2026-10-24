"""Interval minibatch PPO with explicit, finite-sample local meta-gradients.

The public cost and Eq.(30) fusion are retained. Eq.(34)-(39) use observed
interval means and direct intrinsic-reward partials, not an exact lifetime
gradient. Batch updates and boundary expectations are declared adaptations.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .networks import TaskPolicy, HistoryReward, mlp, masked_distribution


def clipped_loss(logp, old_logp, adv, epsilon):
    ratio = torch.exp(logp-old_logp)
    return -torch.minimum(ratio*adv, ratio.clamp(1-epsilon, 1+epsilon)*adv).mean()


def clipping_gate(ratio, advantage, epsilon):
    return (torch.abs(ratio-1)<epsilon) | (torch.sign(ratio-1)!=torch.sign(advantage))


def fuse_reward(external, intrinsic, weight):
    if not np.isfinite(weight):
        raise ValueError('Nonfinite fusion weight')
    return weight*intrinsic+(1-weight)*external


def batch_meta_jacobian_scales(pre_scores, cost_score, reward_score, weights, config):
    """Contract each sample's Jacobian before averaging its local gradient."""
    cost_dot = sum((a.detach()*b.detach().unsqueeze(0)).flatten(1).sum(1)
                   for a, b in zip(pre_scores, cost_score))
    reward_dot = sum((a.detach()*b.detach().unsqueeze(0)).flatten(1).sum(1)
                     for a, b in zip(pre_scores, reward_score))
    return config.intrinsic_lr*weights*cost_dot, config.embedding_lr*weights*reward_dot


@dataclass
class Decision:
    graphs: tuple
    candidates: tuple
    gid: int
    resources: np.ndarray
    mask: np.ndarray
    obs: np.ndarray


@dataclass
class BoundaryState:
    """Read-only old-policy alternatives; none is committed as the next task."""
    decisions: tuple
    probabilities: np.ndarray
    value: float


class PaperLearner:
    def __init__(self, vm_count, config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.vm_count = vm_count
        self.obs_dim = config.embedding_dim+1+6*vm_count
        self.dtype = torch.float64
        self.task = TaskPolicy(config).to(device=self.device, dtype=self.dtype)
        self.actor = mlp(self.obs_dim, config.actor_hidden, vm_count).to(device=self.device, dtype=self.dtype)
        self.critic = mlp(self.obs_dim, config.critic_hidden, 1).to(device=self.device, dtype=self.dtype)
        self.intrinsic = HistoryReward(self.obs_dim, vm_count, config.rnn_hidden).to(device=self.device, dtype=self.dtype)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.intrinsic_opt = torch.optim.Adam(self.intrinsic.parameters(), lr=config.intrinsic_lr)
        self.task_opt = torch.optim.Adam(self.task.embedding.parameters(), lr=config.embedding_lr)
        from common.f19_numerics import NumericalDiagnostics
        self.numerics = NumericalDiagnostics()
        for name in ('actor', 'critic', 'intrinsic'):
            self.numerics.inputs(getattr(self, name), name+'_input')
        self.numerics.inputs(self.task.score, 'selector_embedding_input')
        self.batch_size = config.batch_size
        self.steps = 0
        self.updates = 0
        self.buffer = []
        self._pending_boundary = None
        self.reset_history()

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def reset_history(self):
        if self.buffer or self._pending_boundary is not None:
            raise RuntimeError('Close interval memory and resolve its actual successor before resetting history')
        self.hidden = torch.zeros(self.cfg.rnn_hidden, device=self.device, dtype=self.dtype)
        self._embedding_cache = {}

    def _ready_embeddings(self, env):
        graphs = env.graphs()
        candidates = tuple(env.get_ready_task_candidates()['gids'])
        needed = set(candidates)
        for graph in graphs:
            if not needed.isdisjoint(graph.gids) and graph.wf_id not in self._embedding_cache:
                self.numerics.observe('task_rank_input', graph.features)
                self._embedding_cache[graph.wf_id] = self.task.embedding(graph)
        embeddings = {gid: self._embedding_cache[g.wf_id][i] for g in graphs
                      if g.wf_id in self._embedding_cache for i, gid in enumerate(g.gids)}
        return graphs, candidates, embeddings

    def _decision(self, env, graphs, candidates, embeddings, gid):
        resources = env.resource_features(gid)
        mask = env.queue_mask()
        obs = torch.cat((embeddings[gid], self.tensor(resources))).cpu().numpy()
        return Decision(graphs, candidates, gid, resources.copy(), mask.copy(), obs.copy())

    @torch.no_grad()
    def prepare(self, env, deterministic=False):
        graphs, candidates, embeddings = self._ready_embeddings(env)
        dist = self.task.distribution(embeddings, candidates)
        chosen = dist.logits.argmax() if deterministic else dist.sample()
        gid = candidates[int(chosen)]
        decision = self._decision(env, graphs, candidates, embeddings, gid)
        env.select_task(gid)
        return decision

    @torch.no_grad()
    def boundary_snapshot(self, env):
        if env.current_task_gid is not None:
            raise RuntimeError('Boundary estimation must not preselect the next task')
        graphs, candidates, embeddings = self._ready_embeddings(env)
        dist = self.task.distribution(embeddings, candidates)
        decisions = tuple(self._decision(env, graphs, candidates, embeddings, g) for g in candidates)
        observations = torch.stack([self.tensor(d.obs) for d in decisions])
        value = (dist.probs*self.critic(observations).squeeze(-1)).sum()
        return BoundaryState(decisions, dist.probs.cpu().numpy().copy(), float(value))

    @torch.no_grad()
    def resolve_boundary(self, successor):
        """Continue real RNN history with the task actually selected after updating."""
        record = self._pending_boundary
        if record is None or successor.gid not in {d.gid for d in record['boundary'].decisions}:
            raise ValueError('Missing boundary or successor outside the observed Ready set')
        token = self.history_token(self.tensor(record['decision'].obs), record['action'],
                                   record['step_reward'], False, self.tensor(successor.obs))
        _, self.hidden = self.intrinsic(token, record['history_before'])
        self._pending_boundary = None

    @torch.no_grad()
    def act(self, decision, deterministic=False):
        obs = self.tensor(decision.obs)
        dist = masked_distribution(self.actor(obs), decision.mask)
        action = dist.logits.argmax() if deterministic else dist.sample()
        return int(action), float(dist.log_prob(action)), float(self.critic(obs).squeeze())

    @torch.no_grad()
    def value(self, decision):
        return float(self.critic(self.tensor(decision.obs)).squeeze())

    def history_token(self, obs, action, reward, done, next_obs):
        onehot = torch.nn.functional.one_hot(torch.tensor(action, device=self.device), self.vm_count).to(self.dtype)
        if next_obs.ndim == 2:
            count = len(next_obs)
            return torch.cat((obs.expand(count, -1), onehot.expand(count, -1),
                              self.tensor([reward, float(done)]).expand(count, -1), next_obs), dim=-1)
        return torch.cat((obs, onehot, self.tensor([reward, float(done)]), next_obs))

    def remember(self, decision, action, logp, value, reward, done, info, next_decision, boundary=None):
        if self._pending_boundary is not None:
            raise RuntimeError('Resolve the previous interval boundary before collecting again')
        if decision.mask[action]<=0 or not np.isfinite([reward, logp, value, info['outer_cost']]).all():
            raise ValueError('Invalid actual VM transition')
        if not np.isclose(info['outer_cost'], -reward, rtol=0, atol=1e-12):
            raise ValueError('Common cost mismatch')
        if boundary is not None and (done or next_decision is not None):
            raise ValueError('A nonterminal boundary uses alternatives, not a selected successor')
        if not done and next_decision is None and boundary is None:
            raise ValueError('Nonterminal transition needs a successor or boundary alternatives')
        before = self.hidden.detach().clone()
        with torch.no_grad():
            if boundary is None:
                nxt = self.tensor(next_decision.obs) if next_decision else torch.zeros_like(self.tensor(decision.obs))
                token = self.history_token(self.tensor(decision.obs), action, reward, done, nxt)
                intrinsic, self.hidden = self.intrinsic(token, self.hidden)
            else:
                nxt = torch.stack([self.tensor(d.obs) for d in boundary.decisions])
                token = self.history_token(self.tensor(decision.obs), action, reward, False, nxt)
                estimates, _ = self.intrinsic(token, before.expand(len(nxt), -1))
                intrinsic = (self.tensor(boundary.probabilities)*estimates).sum()
        record = dict(decision=decision, action=action, logp=logp, value=value, step_reward=reward,
                      done=done, info=dict(info), next_decision=next_decision, boundary=boundary,
                      history_before=before, behavior_intrinsic=float(intrinsic))
        self.buffer.append(record)
        if boundary is not None:
            self._pending_boundary = record
        self.steps += 1
        return float(intrinsic), float(fuse_reward(reward, intrinsic, info['fusion_weight']))

    def _replay(self):
        """Encode each required DAG once per batch; replay chronological RNN chunks."""
        decisions = [d for r in self.buffer for d in (r['decision'], r['next_decision']) if d is not None]
        decisions.extend(d for r in self.buffer if r['boundary'] is not None for d in r['boundary'].decisions)
        needed = {d.gid for d in decisions}
        graphs = {g.wf_id: g for d in decisions for g in d.graphs if not needed.isdisjoint(g.gids)}
        embeddings = self.task.embeddings(tuple(graphs.values()))
        def state(d):
            return torch.cat((embeddings[d.gid], self.tensor(d.resources)))
        observations = torch.stack([state(r['decision']) for r in self.buffer])
        intrinsics = []
        for i, r in enumerate(self.buffer):
            if i % self.cfg.trajectory_length == 0:
                hidden = r['history_before'].detach()
            if r['boundary'] is not None:
                if i != len(self.buffer)-1:
                    raise RuntimeError('Boundary alternatives must end the interval')
                boundary = r['boundary']
                nxt = torch.stack([state(d) for d in boundary.decisions])
                token = self.history_token(observations[i], r['action'], r['step_reward'], False, nxt)
                estimates, _ = self.intrinsic(token, hidden.expand(len(nxt), -1))
                intrinsic = (self.tensor(boundary.probabilities)*estimates).sum()
            else:
                nxt = state(r['next_decision']) if r['next_decision'] else torch.zeros_like(observations[i])
                token = self.history_token(observations[i], r['action'], r['step_reward'], r['done'], nxt)
                intrinsic, hidden = self.intrinsic(token, hidden)
            intrinsics.append(intrinsic)
        return observations, torch.stack(intrinsics)

    def _advantages(self, bootstrap):
        result = []
        for i, r in enumerate(self.buffer):
            next_value = (self.buffer[i+1]['value'] if i+1<len(self.buffer) else bootstrap) if not r['done'] else 0.
            reward = fuse_reward(r['step_reward'], r['behavior_intrinsic'], r['info']['fusion_weight'])
            result.append(reward+self.cfg.gamma*float(next_value)-r['value'])
        return result

    def _outer_scores(self, observations, advantages, old, masks, actions, costs):
        # The same re-encoded, detached observations are used by inner and outer Actor.
        logs = masked_distribution(self.actor(observations), masks).log_prob(actions)
        ratios = torch.exp(logs.detach()-old)
        gates = clipping_gate(ratios, advantages, self.cfg.clip)
        factors = gates.to(self.dtype)*ratios*advantages
        params = tuple(self.actor.parameters())
        reward_score = tuple(g.detach() for g in torch.autograd.grad((factors*logs).mean(), params, retain_graph=True))
        cost_score = tuple(g.detach() for g in torch.autograd.grad((costs*factors*logs).mean(), params))
        return reward_score, cost_score, ratios, gates

    def update(self, bootstrap=None):
        if not self.buffer:
            return None
        if not self.buffer[-1]['done'] and bootstrap is None:
            raise ValueError('Nonterminal interval needs an explicitly estimated successor value')
        if len({r['info']['interval'] for r in self.buffer}) != 1:
            raise ValueError('Interval memory must belong to one policy interval')
        bootstrap = float(bootstrap or 0.)
        if not np.isfinite(bootstrap):
            raise ValueError('Nonfinite bootstrap')
        count = len(self.buffer)
        advantages = self.tensor(self._advantages(bootstrap))
        old = self.tensor([r['logp'] for r in self.buffer])
        actions = torch.tensor([r['action'] for r in self.buffer], device=self.device)
        masks = torch.as_tensor(np.stack([r['decision'].mask for r in self.buffer]), device=self.device, dtype=torch.bool)
        costs = self.tensor([r['info']['outer_cost'] for r in self.buffer])
        weights = self.tensor([r['info']['fusion_weight'] for r in self.buffer])
        targets = self.tensor([r['value'] for r in self.buffer])+advantages
        actor_params = tuple(self.actor.parameters())
        phi_params = tuple(self.intrinsic.parameters())
        eta_params = tuple(self.task.embedding.parameters())
        samples = []
        for _ in range(self.cfg.update_rounds):
            indices = np.random.choice(count, size=min(count, self.cfg.batch_size), replace=False).tolist()
            samples.extend(indices)
            index = torch.tensor(indices, device=self.device)
            self.batch_size = len(indices)  # shared recorder counts the actual short batch
            observations, intrinsics = self._replay()
            observations = observations.detach()
            logs = masked_distribution(self.actor(observations[index]), masks[index]).log_prob(actions[index])
            # Batched score Jacobians avoid one backward/Actor update per sample.
            pre_scores = tuple(g.detach() for g in torch.autograd.grad(
                logs, actor_params, grad_outputs=torch.eye(len(indices), device=self.device, dtype=self.dtype),
                is_grads_batched=True, retain_graph=True))
            actor_loss = clipped_loss(logs, old[index], advantages[index], self.cfg.clip)
            critic_loss = (self.critic(observations[index]).squeeze(-1)-targets[index]).square().mean()
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            reward_score, cost_score, ratios, gates = self._outer_scores(observations, advantages, old, masks, actions, costs)
            phi_scale, eta_scale = batch_meta_jacobian_scales(pre_scores, cost_score, reward_score, weights[index], self.cfg)
            # Keep each sample's contraction and direct partial paired before mean.
            phi_loss = (phi_scale.detach()*intrinsics[index]).mean()
            eta_loss = -(eta_scale.detach()*intrinsics[index]).mean()
            phi_grad = torch.autograd.grad(phi_loss, phi_params, retain_graph=True, allow_unused=True)
            eta_grad = torch.autograd.grad(eta_loss, eta_params, allow_unused=True)
            for opt, params, grads in ((self.intrinsic_opt, phi_params, phi_grad), (self.task_opt, eta_params, eta_grad)):
                opt.zero_grad()
                for parameter, gradient in zip(params, grads):
                    parameter.grad = None if gradient is None else gradient.detach()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite local meta-gradient')
                opt.step()
            norms = {name: self.numerics.gradients(getattr(self, name).parameters(), name=name+'_gradient')
                     for name in ('actor', 'critic', 'intrinsic')}
            norms['embedding'] = self.numerics.gradients(eta_params, name='embedding_gradient')
            losses = dict(actor=float(actor_loss.detach()), critic=float(critic_loss.detach()),
                          intrinsic_meta=float(phi_loss.detach()), embedding_meta=float(eta_loss.detach()))
            if not np.isfinite(list(losses.values())).all():
                raise FloatingPointError('Nonfinite learning estimate')
            self.numerics.observe('advantage', advantages[index], scalar=True)
            for name, value in losses.items():
                self.numerics.observe(name+'_loss', value, scalar=True)
            self.updates += 1
            last = dict(losses=losses, gradient_norms=norms, clip_active_fraction=float(gates[index].to(self.dtype).mean()),
                        intrinsic=float(intrinsics[index].detach().mean()), phi_coefficient=float(phi_scale.mean()),
                        eta_coefficient=float(eta_scale.mean()))
        boundary = self.buffer[-1]['boundary']
        last.update(transitions=min(count, self.cfg.batch_size), collected_transitions=count,
                    sample_presentations=len(samples), outer_transition_presentations=count*self.cfg.update_rounds,
                    optimizer_steps_per_network=self.cfg.update_rounds, sampled_indices=samples,
                    bootstrap=bootstrap, terminal=bool(self.buffer[-1]['done']),
                    boundary_candidates=len(boundary.decisions) if boundary is not None else 0,
                    update_rule='interval_minibatch_mean_v2')
        self.buffer.clear()
        self._embedding_cache = {}
        return last
