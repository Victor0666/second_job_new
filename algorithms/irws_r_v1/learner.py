"""VM-only PPO and explicit finite-sample Eq.(34)-(39) Jacobian contractions.

Reward/cost is the retained common problem. Pre-update intrinsic reward provides
local partials in Eq.(36)/(39); post-update VM policy supplies Eq.(35)/(38).
No surrogate virtual-SGD step, entropy bonus or task-policy likelihood is added.
"""
from dataclasses import dataclass, replace
import numpy as np
import torch
from .networks import TaskPolicy,HistoryReward,mlp,masked_distribution


def clipped_loss(logp,old_logp,adv,epsilon):
    ratio=torch.exp(logp-old_logp)
    return -torch.minimum(ratio*adv,ratio.clamp(1-epsilon,1+epsilon)*adv).mean()


def clipping_gate(ratio,advantage,epsilon):
    return (torch.abs(ratio-1)<epsilon) | (torch.sign(ratio-1)!=torch.sign(advantage))


def fuse_reward(external,intrinsic,weight):
    if not np.isfinite(weight): raise ValueError('Nonfinite fusion weight')
    return weight*intrinsic+(1-weight)*external


def meta_jacobian_scales(pre_score,cost_score,reward_score,weight,config):
    """Eq.(34)/(37): contract each printed outer gradient with its local Jacobian."""
    cost_dot=sum((a.detach()*b.detach()).sum() for a,b in zip(pre_score,cost_score))
    reward_dot=sum((a.detach()*b.detach()).sum() for a,b in zip(pre_score,reward_score))
    return config.intrinsic_lr*weight*cost_dot,config.embedding_lr*weight*reward_dot


@dataclass
class Decision:
    graphs: tuple
    candidates: tuple
    gid: int
    resources: np.ndarray
    mask: np.ndarray
    obs: np.ndarray


class PaperLearner:
    def __init__(self,vm_count,config):
        self.cfg=config;self.device=torch.device(config.device);self.vm_count=vm_count
        self.obs_dim=config.embedding_dim+1+6*vm_count;self.dtype=torch.float64
        self.task=TaskPolicy(config).to(device=self.device,dtype=self.dtype)
        self.actor=mlp(self.obs_dim,config.actor_hidden,vm_count).to(device=self.device,dtype=self.dtype)
        self.critic=mlp(self.obs_dim,config.critic_hidden,1).to(device=self.device,dtype=self.dtype)
        self.intrinsic=HistoryReward(self.obs_dim,vm_count,config.rnn_hidden).to(device=self.device,dtype=self.dtype)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=config.actor_lr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=config.critic_lr)
        self.intrinsic_opt=torch.optim.Adam(self.intrinsic.parameters(),lr=config.intrinsic_lr)
        self.task_opt=torch.optim.Adam(self.task.embedding.parameters(),lr=config.embedding_lr)
        from common.f19_numerics import NumericalDiagnostics
        self.numerics=NumericalDiagnostics()
        for name in ('actor','critic','intrinsic'): self.numerics.inputs(getattr(self,name),name+'_input')
        self.numerics.inputs(self.task.score,'selector_embedding_input')
        self.batch_size=1  # one sample per optimizer step, for the shared budget recorder
        self.steps=0;self.updates=0;self.buffer=[];self.reset_history()

    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)

    def reset_history(self):
        if self.buffer: raise RuntimeError('Close interval memory before resetting history')
        self.hidden=torch.zeros(self.cfg.rnn_hidden,device=self.device,dtype=self.dtype)
        self._embedding_cache={}

    @torch.no_grad()
    def prepare(self,env,deterministic=False):
        graphs=env.graphs();candidates=tuple(env.get_ready_task_candidates()['gids'])
        needed=set(candidates)
        for graph in graphs:
            if not needed.isdisjoint(graph.gids) and graph.wf_id not in self._embedding_cache:
                self.numerics.observe('task_rank_input',graph.features)
                self._embedding_cache[graph.wf_id]=self.task.embedding(graph)
        embeddings={gid:self._embedding_cache[g.wf_id][i] for g in graphs if g.wf_id in self._embedding_cache for i,gid in enumerate(g.gids)}
        dist=self.task.distribution(embeddings,candidates)
        chosen=dist.logits.argmax() if deterministic else dist.sample();gid=candidates[int(chosen)]
        env.select_task(gid);vm=env.get_vm_obs_and_mask()
        obs=torch.cat((embeddings[gid],self.tensor(vm['obs']))).cpu().numpy()
        return Decision(graphs,candidates,gid,vm['obs'].copy(),vm['mask'].copy(),obs.copy())

    @torch.no_grad()
    def refresh(self,decision):
        """Keep the real successor task; refresh its encoding after the interval update."""
        graph=next(g for g in decision.graphs if decision.gid in g.gids)
        embedding=self.task.embedding(graph)[graph.gids.index(decision.gid)]
        obs=torch.cat((embedding,self.tensor(decision.resources))).cpu().numpy()
        return replace(decision,obs=obs)

    @torch.no_grad()
    def act(self,decision,deterministic=False):
        obs=self.tensor(decision.obs);dist=masked_distribution(self.actor(obs),decision.mask)
        action=dist.logits.argmax() if deterministic else dist.sample()
        return int(action),float(dist.log_prob(action)),float(self.critic(obs).squeeze())

    @torch.no_grad()
    def value(self,decision):return float(self.critic(self.tensor(decision.obs)).squeeze())

    def history_token(self,obs,action,reward,done,next_obs):
        onehot=torch.nn.functional.one_hot(torch.tensor(action,device=self.device),self.vm_count).to(self.dtype)
        return torch.cat((obs,onehot,self.tensor([reward,float(done)]),next_obs))

    def remember(self,decision,action,logp,value,reward,done,info,next_decision):
        if decision.mask[action]<=0 or not np.isfinite([reward,logp,value,info['outer_cost']]).all():
            raise ValueError('Invalid actual VM transition')
        if not np.isclose(info['outer_cost'],-reward,rtol=0,atol=1e-12):raise ValueError('Common cost mismatch')
        nxt=self.tensor(next_decision.obs) if next_decision else torch.zeros(self.obs_dim,device=self.device,dtype=self.dtype)
        token=self.history_token(self.tensor(decision.obs),action,reward,done,nxt)
        before=self.hidden.detach().clone()
        with torch.no_grad():intrinsic,self.hidden=self.intrinsic(token,self.hidden)
        self.buffer.append(dict(decision=decision,action=action,logp=logp,value=value,step_reward=reward,
            done=done,info=dict(info),next_decision=next_decision,history_before=before,behavior_intrinsic=float(intrinsic)))
        self.steps+=1
        return float(intrinsic),float(fuse_reward(reward,intrinsic,info['fusion_weight']))

    def _sample(self,index,bootstrap):
        # Reconstruct the true chronological prefix, never a shuffled RNN history.
        start=(index//self.cfg.trajectory_length)*self.cfg.trajectory_length
        records=self.buffer[start:index+1]
        graphs={g.wf_id:g for r in records for d in (r['decision'],r['next_decision']) if d for g in d.graphs}
        needed={d.gid for r in records for d in (r['decision'],r['next_decision']) if d}
        graphs={k:g for k,g in graphs.items() if not needed.isdisjoint(g.gids)}
        embeddings=self.task.embeddings(tuple(graphs.values()))
        def state(d):return torch.cat((embeddings[d.gid],self.tensor(d.resources)))
        hidden=records[0]['history_before'].detach()
        for r in records:
            obs=state(r['decision'])
            nxt=state(r['next_decision']) if r['next_decision'] else torch.zeros_like(obs)
            token=self.history_token(obs,r['action'],r['step_reward'],r['done'],nxt)
            intrinsic,hidden=self.intrinsic(token,hidden)
        r=self.buffer[index]
        return obs,intrinsic,self.tensor(self._advantages(bootstrap)[index])

    def _advantages(self,bootstrap):
        # Algorithm 1 computes these from the collected behavior trajectory before updates.
        result=[]
        for i,r in enumerate(self.buffer):
            next_value=(self.buffer[i+1]['value'] if i+1<len(self.buffer) else bootstrap) if not r['done'] else 0.
            reward=fuse_reward(r['step_reward'],r['behavior_intrinsic'],r['info']['fusion_weight'])
            result.append(reward+self.cfg.gamma*float(next_value)-r['value'])
        return result

    def _outer_scores(self,bootstrap):
        observations=torch.stack([self.tensor(r['decision'].obs) for r in self.buffer])
        logs=[]
        for obs,r in zip(observations,self.buffer):
            dist=masked_distribution(self.actor(obs),r['decision'].mask)
            logs.append(dist.log_prob(torch.tensor(r['action'],device=self.device)))
        logs=torch.stack(logs);old=self.tensor([r['logp'] for r in self.buffer])
        advantages=self.tensor(self._advantages(bootstrap));ratios=torch.exp(logs.detach()-old)
        gates=clipping_gate(ratios,advantages,self.cfg.clip).to(self.dtype)
        # This scalar constructs the printed score-gradient factors, not a new reward loss.
        factors=gates*ratios*advantages
        costs=self.tensor([r['info']['outer_cost'] for r in self.buffer])
        params=tuple(self.actor.parameters())
        reward_score=tuple(g.detach() for g in torch.autograd.grad((factors*logs).sum(),params,retain_graph=True))
        cost_score=tuple(g.detach() for g in torch.autograd.grad((costs*factors*logs).sum(),params))
        return reward_score,cost_score,ratios,gates

    def update(self,bootstrap=None):
        if not self.buffer:return None
        if not self.buffer[-1]['done'] and bootstrap is None: raise ValueError('Nonterminal interval needs real successor value')
        intervals={r['info']['interval'] for r in self.buffer}
        if len(intervals)!=1: raise ValueError('Algorithm 1 memory must belong to one policy interval')
        count=len(self.buffer);samples=[];last=None
        # Finite observed estimate of the retained common lifetime cost; no invented future tail.
        actor_params=tuple(self.actor.parameters());phi_params=tuple(self.intrinsic.parameters())
        eta_params=tuple(self.task.embedding.parameters())
        for _ in range(self.cfg.update_rounds):
            indices=np.random.choice(count,size=min(count,self.cfg.batch_size),replace=False).tolist()
            for index in indices:
                samples.append(index);r=self.buffer[index]
                obs,intrinsic,advantage=self._sample(index,float(bootstrap or 0.))
                dist=masked_distribution(self.actor(obs.detach()),r['decision'].mask)
                logp=dist.log_prob(torch.tensor(r['action'],device=self.device))
                pre_score=tuple(g.detach() for g in torch.autograd.grad(logp,actor_params,retain_graph=True))
                actor_loss=clipped_loss(logp,self.tensor(r['logp']),advantage.detach(),self.cfg.clip)
                target=(self.tensor(r['value'])+advantage).detach()
                critic_loss=(self.critic(obs.detach()).squeeze()-target).square()
                self.critic_opt.zero_grad();critic_loss.backward();self.critic_opt.step()
                self.actor_opt.zero_grad();actor_loss.backward();self.actor_opt.step()
                # Actual post-Adam theta'; include the full observed interval in the outer sum.
                reward_score,cost_score,ratios,gates=self._outer_scores(float(bootstrap or 0.))
                phi_scale,eta_scale=meta_jacobian_scales(pre_score,cost_score,reward_score,
                    r['info']['fusion_weight'],self.cfg)
                phi_grad=torch.autograd.grad(intrinsic,phi_params,retain_graph=True,allow_unused=True)
                eta_grad=torch.autograd.grad(intrinsic,eta_params,allow_unused=True)
                for opt,params,grads,scale in ((self.intrinsic_opt,phi_params,phi_grad,phi_scale),
                                             (self.task_opt,eta_params,eta_grad,-eta_scale)):
                    opt.zero_grad()
                    for parameter,gradient in zip(params,grads):
                        parameter.grad=None if gradient is None else gradient.detach()*scale
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError('Nonfinite paper meta-gradient')
                    opt.step()
                norms={}
                for name in ('actor','critic','intrinsic'):
                    norms[name]=self.numerics.gradients(getattr(self,name).parameters(),name=name+'_gradient')
                norms['embedding']=self.numerics.gradients(eta_params,name='embedding_gradient')
                losses=dict(actor=float(actor_loss.detach()),critic=float(critic_loss.detach()),
                            intrinsic_meta=float(phi_scale*intrinsic.detach()),embedding_meta=float(-eta_scale*intrinsic.detach()))
                if not np.isfinite(list(losses.values())).all(): raise FloatingPointError('Nonfinite learning estimate')
                self.numerics.observe('advantage',advantage,scalar=True)
                for name,value in losses.items():self.numerics.observe(name+'_loss',value,scalar=True)
                self.updates+=1
                last=dict(losses=losses,gradient_norms=norms,clip_active=bool(gates[index]),
                          intrinsic=float(intrinsic.detach()),phi_coefficient=float(phi_scale),eta_coefficient=float(eta_scale))
        last.update(transitions=min(count,self.cfg.batch_size),collected_transitions=count,sample_presentations=len(samples),
                    sampled_indices=samples,bootstrap=float(bootstrap or 0.),terminal=bool(self.buffer[-1]['done']))
        self.buffer.clear();self._embedding_cache={}
        return last
