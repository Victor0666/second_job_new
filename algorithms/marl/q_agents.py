"""Paper-style action-conditioned critics with independent Local policies.

The paper leaves the TD target unspecified. We use the next policy's expected Q,
then update Critic before Actor, without sampling a stale successor action.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common.f11 import LearningDiagnostics, LEARNING_MODEL, valid_decision
from common.f18_state_normalization import checkpoint_metadata, check_checkpoint
from algorithms.marl.identity import MODEL_VERSION


@dataclass(frozen=True)
class QConfig:
    lr: float = 3e-4
    gamma: float = 0.95
    hidden_dims: tuple = (1024, 1024, 512, 512, 256)
    device: str = 'cpu'
    ent_coef: float = 0.0
    grad_clip: float = 0.0

    def __post_init__(self):
        if self.ent_coef != 0 or self.grad_clip != 0:
            raise ValueError('MARL Q adaptation disables entropy and gradient clipping')


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class QAgent:
    def __init__(self, obs_dim, act_dim, cfg, peer_act_dim=0):
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.peer_act_dim = int(peer_act_dim)
        self.cfg, self.device = cfg, torch.device(cfg.device)
        self.actor = _mlp(self.obs_dim, cfg.hidden_dims, self.act_dim).to(self.device)
        self.critic = _mlp(self.obs_dim + self.act_dim + self.peer_act_dim,
                           cfg.hidden_dims, 1).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.f11 = LearningDiagnostics()
        self.f11.numerics.inputs(self.actor, 'actor_input')
        self.f11.numerics.inputs(self.critic, 'critic_input')

    def _obs(self, obs):
        array = np.asarray(obs, dtype=np.float32)
        if array.shape != (self.obs_dim,) or not np.isfinite(array).all():
            raise ValueError('Invalid MARL observation')
        return torch.as_tensor(array, device=self.device).unsqueeze(0)

    def _distribution(self, obs, mask):
        valid_decision(obs, mask)
        if np.shape(mask) != (self.act_dim,):
            raise ValueError('Invalid MARL action mask dimension')
        logits = self.actor(self._obs(obs))
        legal = torch.as_tensor(mask, device=self.device) > .5
        return torch.distributions.Categorical(logits=logits.masked_fill(~legal, -1e9))

    @torch.no_grad()
    def select_action(self, obs, mask, deterministic=False):
        dist = self._distribution(obs, mask)
        return int(dist.logits.argmax(-1).item() if deterministic else dist.sample().item())

    @torch.no_grad()
    def action_probabilities(self, obs, mask):
        return self._distribution(obs, mask).probs.squeeze(0).cpu().numpy().copy()

    @torch.no_grad()
    def expected_q(self, obs, mask, peer_probabilities=None):
        """Policy evaluation over legal actions, without drawing a behavior action."""
        probs = self._distribution(obs, mask).probs.squeeze(0)
        legal = torch.nonzero(torch.as_tensor(mask, device=self.device) > .5).flatten()
        if self.peer_act_dim:
            if peer_probabilities is None or set(peer_probabilities) != set(legal.tolist()):
                raise ValueError('Global successor needs a Local policy for every legal Host')
            actions, peers, weights = [], [], []
            for action in legal.tolist():
                peer = torch.as_tensor(peer_probabilities[action], dtype=torch.float32, device=self.device)
                if (peer.shape != (self.peer_act_dim,) or not torch.isfinite(peer).all()
                        or (peer < 0).any() or not torch.isclose(peer.sum(), peer.new_tensor(1.), atol=1e-6)):
                    raise ValueError('Invalid Local successor action probabilities')
                indices = torch.nonzero(peer > 0).flatten()
                actions.append(torch.full_like(indices, action))
                peers.append(indices)
                weights.append(probs[action] * peer[indices])
            actions, peers, weights = torch.cat(actions), torch.cat(peers), torch.cat(weights)
        else:
            if peer_probabilities is not None:
                raise ValueError('Local successor must not receive another agent policy')
            actions, weights = legal, probs[legal]
        parts = [self._obs(obs).expand(len(actions), -1), F.one_hot(actions, self.act_dim).float()]
        if self.peer_act_dim:
            parts.append(F.one_hot(peers, self.peer_act_dim).float())
        q = self.critic(torch.cat(parts, dim=-1)).squeeze(-1)
        return float(torch.dot(weights, q).item())

    def _q(self, decision):
        parts = [self._obs(decision.obs)]
        actions = [(decision.action, self.act_dim)]
        if self.peer_act_dim:
            actions.append((decision.peer_action, self.peer_act_dim))
        elif decision.peer_action is not None:
            raise ValueError('Local critic must not receive another agent action')
        for action, size in actions:
            if action is None or not 0 <= int(action) < size:
                raise ValueError('Missing or invalid MARL critic action')
            parts.append(F.one_hot(torch.tensor([int(action)], device=self.device), size).float())
        return self.critic(torch.cat(parts, dim=-1)).squeeze(-1)

    def update_actor(self, decision):
        # Eq. (12): no extra V baseline, advantage normalization or entropy term.
        dist = self._distribution(decision.obs, decision.mask)
        if decision.mask[decision.action] <= .5:
            raise ValueError('Illegal recorded MARL action')
        with torch.no_grad():
            q = self._q(decision)
        action = torch.tensor([decision.action], device=self.device)
        loss = -(dist.log_prob(action) * q).mean()
        self.f11.numerics.observe('Q_policy', q, scalar=True)
        value = self.f11.loss(loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.actor.parameters(), name='actor_gradient')
        self.actor_optim.step()
        return value

    def update_critic(self, decision, reward, next_value=None):
        done = next_value is None
        if not np.isfinite(reward) or (not done and not np.isfinite(next_value)):
            raise ValueError('Nonfinite MARL reward or successor Q')
        with torch.no_grad():
            future = torch.tensor([0. if done else self.cfg.gamma * next_value], device=self.device)
            target = torch.tensor([reward], dtype=torch.float32, device=self.device) + future
        q = self._q(decision)
        loss = F.mse_loss(q, target)
        self.f11.transition(done)
        self.f11.target(future, target)
        self.f11.numerics.observe('Q_selected', q, scalar=True)
        self.f11.numerics.observe('TD_target', target, scalar=True)
        value = self.f11.loss(loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.critic.parameters(), name='critic_gradient')
        self.critic_optim.step()
        self.f11.updates += 1
        return value

    def update(self, decision, reward, next_value=None):
        # Algorithms 1/2: the Actor must read Q after this transition's Critic step.
        critic_loss = self.update_critic(decision, reward, next_value)
        actor_loss = self.update_actor(decision)
        return dict(critic_loss=critic_loss, actor_loss=actor_loss)

    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    actor_optim=self.actor_optim.state_dict(), critic_optim=self.critic_optim.state_dict())

    def restore(self, state):
        self.actor.load_state_dict(state['actor'], strict=True)
        self.critic.load_state_dict(state['critic'], strict=True)
        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])

    def save(self, path):
        torch.save(dict(model_version=MODEL_VERSION, learning_model=LEARNING_MODEL,
                        normalization=checkpoint_metadata(self), cfg=asdict(self.cfg),
                        obs_dim=self.obs_dim, act_dim=self.act_dim, peer_act_dim=self.peer_act_dim,
                        state=self.state()), path)

    def load(self, path):
        ckpt = _load(path, self.device, checkpoint_metadata(self))
        if (ckpt['obs_dim'], ckpt['act_dim'], ckpt['peer_act_dim']) != (
                self.obs_dim, self.act_dim, self.peer_act_dim):
            raise ValueError('MARL Q checkpoint dimensions differ')
        # Device migration does not change the algorithm or network.
        if {k:v for k,v in ckpt['cfg'].items() if k != 'device'} != {
                k:v for k,v in asdict(self.cfg).items() if k != 'device'}:
            raise ValueError('MARL Q checkpoint configuration differs')
        self.restore(ckpt['state'])


def _load(path, device, normalization):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # The earlier long v2 name has the same learning rules and checkpoint schema.
    if not isinstance(ckpt, dict) or ckpt.get('model_version') not in (MODEL_VERSION, 'marl_independent_q_v2') or ckpt.get('learning_model') != LEARNING_MODEL:
        raise ValueError('Not an independent MARL Q checkpoint; retrain the new model')
    check_checkpoint(ckpt.get('normalization'), normalization)
    return ckpt


class IndependentLocalAgents:
    """Each list entry owns separate Actor/Critic parameters and Adam states."""
    def __init__(self, obs_dim, act_dim, num_hosts, cfg):
        self.obs_dim, self.act_dim, self.num_hosts = obs_dim, act_dim, num_hosts
        self.cfg, self.device = cfg, torch.device(cfg.device)
        self.agents = [QAgent(obs_dim, act_dim, cfg) for _ in range(num_hosts)]

    def select_action(self, obs, mask, host_index, deterministic=False):
        if not 0 <= host_index < self.num_hosts:
            raise ValueError('Invalid Local host index')
        return self.agents[host_index].select_action(obs, mask, deterministic)

    def save(self, path):
        torch.save(dict(model_version=MODEL_VERSION, learning_model=LEARNING_MODEL,
                        normalization=checkpoint_metadata(self), cfg=asdict(self.cfg),
                        obs_dim=self.obs_dim, act_dim=self.act_dim, num_hosts=self.num_hosts,
                        states=[a.state() for a in self.agents]), path)

    def load(self, path):
        ckpt = _load(path, self.device, checkpoint_metadata(self))
        if (ckpt['obs_dim'], ckpt['act_dim'], ckpt['num_hosts']) != (
                self.obs_dim, self.act_dim, self.num_hosts) or len(ckpt['states']) != self.num_hosts:
            raise ValueError('MARL Local checkpoint topology differs')
        if {k:v for k,v in ckpt['cfg'].items() if k != 'device'} != {
                k:v for k,v in asdict(self.cfg).items() if k != 'device'}:
            raise ValueError('MARL Local checkpoint configuration differs')
        for agent, state in zip(self.agents, ckpt['states']):
            agent.restore(state)


def learner_map(global_agent, locals):
    return {'Global': global_agent, **{f'Local_{i}': a for i, a in enumerate(locals.agents)}}


@dataclass(frozen=True)
class Decision:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    peer_action: int | None = None

    def __post_init__(self):
        object.__setattr__(self, 'obs', np.array(self.obs, dtype=np.float32, copy=True))
        object.__setattr__(self, 'mask', np.array(self.mask, dtype=np.float32, copy=True))
        valid_decision(self.obs, self.mask)
        if not 0 <= self.action < len(self.mask) or self.mask[self.action] <= .5:
            raise ValueError('Illegal MARL decision')


class QDecisionTrace:
    """One agent's decision chain. Update both networks before its next action draw."""
    def __init__(self, agent):
        self.agent, self.pending = agent, None
        self.recorded_reward = self.emitted_reward = 0.0

    def advance(self, obs, mask, peer_probabilities=None):
        if self.pending is not None:
            decision, reward = self.pending
            next_value = self.agent.expected_q(obs, mask, peer_probabilities)
            self.agent.update(decision, reward, next_value)
            self.emitted_reward += reward
            self.pending = None

    def record(self, decision, reward):
        if self.pending is not None:
            raise RuntimeError('Previous MARL decision has no successor')
        if not np.isfinite(reward):
            raise ValueError('Nonfinite MARL reward')
        self.pending = (decision, float(reward))
        self.recorded_reward += float(reward)

    def finish(self, truncated=False):
        if self.pending is not None:
            decision, reward = self.pending
            self.agent.f11.truncation_cutoffs += int(truncated)
            self.agent.update(decision, reward)
            self.emitted_reward += reward
            self.pending = None
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError('Unsettled MARL rewards')
