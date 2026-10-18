"""Differentiable one-step meta-PPO; approximations are documented, never hidden."""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from .networks import TaskPolicy,HistoryReward,mlp,masked_distribution


def gae(rewards,values,dones,bootstrap,gamma,lam):
    if not torch.isfinite(torch.as_tensor(bootstrap)).all():raise ValueError('Nonfinite bootstrap')
    if bool(dones[-1]):bootstrap=values.new_zeros(())
    advantages=[];carry=values.new_zeros(())
    for i in reversed(range(len(rewards))):
        next_value=bootstrap if i==len(rewards)-1 else values[i+1]
        delta=rewards[i]+gamma*(1-dones[i])*next_value-values[i]
        carry=delta+gamma*lam*(1-dones[i])*carry;advantages.append(carry)
    return torch.stack(list(reversed(advantages)))


def clipped_loss(logp,old_logp,adv,epsilon):
    ratio=torch.exp(logp-old_logp)
    return -torch.minimum(ratio*adv,ratio.clamp(1-epsilon,1+epsilon)*adv).mean()


@dataclass
class Decision:
    graphs: tuple
    candidates: tuple
    gid: int
    task_logp: float
    resources: np.ndarray
    mask: np.ndarray
    obs: np.ndarray


class PaperLearner:
    def __init__(self,vm_count,config):
        self.cfg=config;self.device=torch.device(config.device);self.vm_count=vm_count
        self.obs_dim=config.embedding_dim+1+6*vm_count
        self.dtype=torch.float64  # second-order meta-gradients can be below float32 parameter resolution
        self.task=TaskPolicy(config).to(device=self.device,dtype=self.dtype)
        self.actor=mlp(self.obs_dim,config.actor_hidden,vm_count).to(device=self.device,dtype=self.dtype)
        self.critic=mlp(self.obs_dim,config.critic_hidden,1).to(device=self.device,dtype=self.dtype)
        self.intrinsic=HistoryReward(self.obs_dim,vm_count,config.rnn_hidden).to(device=self.device,dtype=self.dtype)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=config.actor_lr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=config.critic_lr)
        self.intrinsic_opt=torch.optim.Adam(self.intrinsic.parameters(),lr=config.intrinsic_lr)
        self.task_opt=torch.optim.Adam(self.task.parameters(),lr=config.embedding_lr)
        self.updates=0;self.steps=0;self.buffer=[];self.reset_history()

    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)

    def reset_history(self):
        if self.buffer:raise RuntimeError('Close the rollout before resetting an independent episode')
        self.hidden=torch.zeros(self.cfg.rnn_hidden,device=self.device,dtype=self.dtype);self._embedding_cache={}

    @torch.no_grad()
    def prepare(self,env,deterministic=False):
        graphs=env.graphs();candidates=tuple(env.get_ready_task_candidates()['gids'])
        needed=set(candidates)
        for graph in graphs:
            if needed.isdisjoint(graph.gids):continue
            if graph.wf_id not in self._embedding_cache:
                self._embedding_cache[graph.wf_id]=self.task.embedding(graph)
        embeddings={gid:self._embedding_cache[g.wf_id][i] for g in graphs if g.wf_id in self._embedding_cache for i,gid in enumerate(g.gids)}
        dist=self.task.distribution(embeddings,candidates)
        chosen=dist.logits.argmax() if deterministic else dist.sample();gid=candidates[int(chosen)]
        env.select_task(gid);vm=env.get_vm_obs_and_mask()
        obs=torch.cat((embeddings[gid],self.tensor(vm['obs']))).cpu().numpy()
        return Decision(graphs,candidates,gid,float(dist.log_prob(chosen)),vm['obs'].copy(),vm['mask'].copy(),obs.copy())

    @torch.no_grad()
    def act(self,decision,deterministic=False):
        obs=self.tensor(decision.obs);dist=masked_distribution(self.actor(obs),decision.mask)
        action=dist.logits.argmax() if deterministic else dist.sample()
        return int(action),float(dist.log_prob(action)),float(self.critic(obs).squeeze())

    @torch.no_grad()
    def value(self,decision):return float(self.critic(self.tensor(decision.obs)).squeeze())

    def history_token(self,obs,action,reward,duration,energy,done,next_obs):
        onehot=torch.nn.functional.one_hot(torch.tensor(action,device=self.device),self.vm_count).to(dtype=self.dtype)
        return torch.cat((obs,onehot,self.tensor([reward,duration,energy,float(done)]),next_obs))

    def remember(self,decision,action,logp,value,reward,done,info,next_decision,time_ref=1.,energy_ref=1.):
        if decision.mask[action]<=0 or not np.isfinite([reward,logp,value,info['outer_cost']]).all():
            raise ValueError('Invalid real policy transition')
        # RNN feedback uses fixed source scales only; rewards/costs themselves are unscaled.
        duration=float(info['dur'])/time_ref;energy=float(info['delta_energy_total'])/energy_ref
        nxt=self.tensor(next_decision.obs) if next_decision else torch.zeros(self.obs_dim,device=self.device,dtype=self.dtype)
        token=self.history_token(self.tensor(decision.obs),action,reward,duration,energy,done,nxt)
        before=self.hidden.detach().clone()
        with torch.no_grad():intrinsic,self.hidden=self.intrinsic(token,self.hidden)
        self.buffer.append(dict(decision=decision,action=action,logp=logp,value=value,step_reward=reward,
            done=done,info=dict(info),next_decision=next_decision,history_before=before,
            duration=duration,energy=energy))
        self.steps+=1
        return float(intrinsic),float(info['fusion_weight']*intrinsic+(1-info['fusion_weight'])*reward)

    def losses(self,bootstrap):
        records=self.buffer
        if not records:raise ValueError('Empty rollout')
        if not records[-1]['done'] and bootstrap is None:raise ValueError('Nonterminal rollout needs the actual next-decision value')
        needed={gid for r in records for gid in r['decision'].candidates}
        needed.update(r['next_decision'].gid for r in records if r['next_decision'])
        all_graphs={g.wf_id:g for r in records for d in (r['decision'],r['next_decision']) if d for g in d.graphs
                    if not needed.isdisjoint(g.gids)}
        embeddings=self.task.embeddings(tuple(all_graphs.values()))
        def state(d):return torch.cat((embeddings[d.gid],self.tensor(d.resources)))
        observations=torch.stack([state(r['decision']) for r in records])
        hidden=records[0]['history_before'].detach();intrinsics=[];fused=[]
        for i,r in enumerate(records):
            if i%self.cfg.trajectory_length==0:hidden=hidden.detach()
            next_obs=state(r['next_decision']) if r['next_decision'] else torch.zeros_like(observations[i])
            token=self.history_token(observations[i],r['action'],r['step_reward'],r['duration'],r['energy'],r['done'],next_obs)
            ri,hidden=self.intrinsic(token,hidden);intrinsics.append(ri)
            weight=r['info']['fusion_weight'];fused.append(weight*ri+(1-weight)*r['step_reward'])
        fused=torch.stack(fused);values=self.tensor([r['value'] for r in records]);dones=self.tensor([r['done'] for r in records])
        advantage=gae(fused,values,dones,float(bootstrap or 0.),self.cfg.gamma,self.cfg.gae_lambda)
        old_logp=self.tensor([r['logp']+r['decision'].task_logp for r in records])
        logp=[];entropies=[]
        for i,r in enumerate(records):
            decision=r['decision'];td=self.task.distribution(embeddings,decision.candidates)
            vd=masked_distribution(self.actor(observations[i]),decision.mask)
            logp.append(td.log_prob(torch.tensor(decision.candidates.index(decision.gid),device=self.device))+
                        vd.log_prob(torch.tensor(r['action'],device=self.device)))
            entropies.append(td.entropy()+vd.entropy())
        actor_loss=clipped_loss(torch.stack(logp),old_logp,advantage,self.cfg.clip)-self.cfg.entropy_coef*torch.stack(entropies).mean()
        critic_loss=torch.nn.functional.mse_loss(self.critic(observations.detach()).squeeze(-1),(values+advantage).detach())
        params=OrderedDict(self.actor.named_parameters())
        inner_grad=torch.autograd.grad(actor_loss,tuple(params.values()),create_graph=True)
        # Eq.(36)/(39) linearized policy step, not a claim of differentiating Adam's full history.
        fast=OrderedDict((name,p-self.cfg.actor_lr*g) for (name,p),g in zip(params.items(),inner_grad))
        fast_logp=[]
        for i,r in enumerate(records):
            dist=masked_distribution(functional_call(self.actor,fast,(observations[i].detach(),)),r['decision'].mask)
            fast_logp.append(dist.log_prob(torch.tensor(r['action'],device=self.device)))
        fast_logp=torch.stack(fast_logp);old_vm=self.tensor([r['logp'] for r in records])
        # Outer phi objective: a likelihood-ratio estimate of the physical Eq.(25)-(27) cost.
        costs=self.tensor([r['info']['outer_cost'] for r in records])
        cost_to_go=gae(costs,torch.zeros_like(costs),dones,0.,self.cfg.gamma,1.)
        intrinsic_meta=(torch.exp(fast_logp-old_vm)*cost_to_go.detach()).mean()
        # Eq.(37)/(38): embedding meta-objective is the fused PPO surrogate.
        embedding_meta=clipped_loss(fast_logp,old_vm,advantage.detach(),self.cfg.clip)
        return dict(actor=actor_loss,critic=critic_loss,intrinsic_meta=intrinsic_meta,
                    embedding_meta=embedding_meta,inner_grad=inner_grad,intrinsics=torch.stack(intrinsics),fused=fused)

    def update(self,bootstrap=None):
        if not self.buffer:return None
        result=None
        for _ in range(self.cfg.update_rounds):
            losses=self.losses(bootstrap)
            intrinsic_params=tuple(self.intrinsic.parameters());task_params=tuple(self.task.parameters())
            phi=torch.autograd.grad(losses['intrinsic_meta'],intrinsic_params,retain_graph=True,allow_unused=True)
            eta=torch.autograd.grad(losses['embedding_meta'],task_params,retain_graph=True,allow_unused=True)
            groups=(('actor',tuple(self.actor.parameters()),losses['inner_grad']),('intrinsic',intrinsic_params,phi),('embedding_and_selector',task_params,eta))
            norms={}
            for name,params,grads in groups:
                for g in grads:
                    if g is not None and not torch.isfinite(g).all():raise FloatingPointError('Nonfinite '+name+' gradient')
                norms[name]=float(sum(g.detach().square().sum() for g in grads if g is not None).sqrt())
            if not all(torch.isfinite(losses[k]) for k in ('actor','critic','intrinsic_meta','embedding_meta')):
                raise FloatingPointError('Nonfinite learning objective')
            self.critic_opt.zero_grad();losses['critic'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.critic.parameters()):
                raise FloatingPointError('Nonfinite critic gradient')
            for opt,params,grads in ((self.actor_opt,tuple(self.actor.parameters()),losses['inner_grad']),
                    (self.intrinsic_opt,intrinsic_params,phi),(self.task_opt,task_params,eta)):
                opt.zero_grad()
                for p,g in zip(params,grads):p.grad=None if g is None else g.detach().clone()
            self.critic_opt.step();self.actor_opt.step();self.intrinsic_opt.step();self.task_opt.step();self.updates+=1
            result=dict(losses={k:float(losses[k].detach()) for k in ('actor','critic','intrinsic_meta','embedding_meta')},
                gradient_norms=norms,bootstrap=float(bootstrap or 0.),terminal=bool(self.buffer[-1]['done']),
                transitions=len(self.buffer),r_int_mean=float(losses['intrinsics'].detach().mean()))
        self.buffer.clear();self._embedding_cache={}
        return result