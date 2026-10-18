"""Training/evaluation/checkpoint orchestration, with no private physical model."""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from common.f18_state_normalization import VERSION,check_checkpoint
from common.f11 import protect_training_outputs
from common.scenarios import ROOT,parse_scenario
from .config import MODEL_VERSION,PaperConfig
from .environment import PaperCloudEnv
from .learner import PaperLearner


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def env_kwargs(scenario,seed):
    return dict(dax_paths=scenario.dax_paths(),horizon=1e9,arrival_lambda=.03,random_seed=seed,
        workflows_per_episode=50,num_hosts=scenario.num_hosts,vms_per_host=scenario.vms_per_host,
        pc_tiers=(1.,2.,4.,6.,8.),bw_tiers=(1000.,2000.,4000.,6000.,8000.),
        deadline_mode='cache_fcfs',deadline_cache_path=str(scenario.cache_path),deadline_cache_strict=True,
        deadline_alpha_choices=[2.,3.],deadline_alpha_probs=list(scenario.deadline_probs),
        max_ready_tasks='auto',**scenario.normalization_kwargs())


def output_dir(scenario,kind):
    source=scenario.normalization_kwargs()['source_train_scene']
    protocol=scenario.protocol()
    if protocol is None:raise ValueError('Formal paper IRWS requires the explicit disjoint common protocol')
    digest=hashlib.sha256(json.dumps(protocol.metadata(),sort_keys=True).encode()).hexdigest()[:12]
    norm=scenario.normalization_metadata()
    if norm['normalization_version']!=VERSION:raise ValueError('Formal IRWS must use the existing F18 shared configuration')
    return ROOT/'outputs'/kind/MODEL_VERSION/(source+'_'+digest+'_'+norm['normalization_config_hash'][:12])


def run_episode(env,learner,seed,training=False):
    env.reset(seed=seed);learner.reset_history();updates=[];weights=[];boundaries=0
    pending=None
    if not env.done_flag:
        decision=learner.prepare(env,deterministic=not training)
        pending=(decision,*learner.act(decision,deterministic=not training))
    while pending is not None:
        decision,action,logp,value=pending
        reward,done,info=env.paper_step(action,learner.cfg.alpha)
        weights.append(info['fusion_weight']);boundaries+=int(info['fusion_boundary'])
        if done:next_decision=None;pending=None
        else:
            next_decision=learner.prepare(env,deterministic=not training)
            pending=(next_decision,*learner.act(next_decision,deterministic=not training))
        if training:
            t_ref=env.state_normalizer.scales['T_ref'] if env.state_normalizer else 1.
            # A fixed source energy unit for RNN inputs, not a new reward coefficient.
            energy_ref=t_ref*sum(h.power(1.) for h in env.hosts.values())
            learner.remember(decision,action,logp,value,reward,done,info,next_decision,t_ref,energy_ref)
            if len(learner.buffer)>=learner.cfg.batch_size or done:
                bootstrap=0. if done else pending[3]  # selected actual successor, pre-update critic
                updates.append(learner.update(bootstrap))
        if len(weights)>100000:raise RuntimeError('Episode exceeded the decision guard')
    metrics=env.get_episode_metrics()
    if env.completed_workflows!=env.workflows_per_episode or any(s!='Finished' for s in env.task_state):
        raise RuntimeError('Paper IRWS ended before actual physical completion')
    if not np.isfinite(list(metrics.values())).all():raise FloatingPointError('Nonfinite common physical metrics')
    return dict(metrics=metrics,updates=updates,assignments=len(weights),fusion=dict(
        minimum=min(weights,default=0.),maximum=max(weights,default=0.),mean=float(np.mean(weights)) if weights else 0.,
        outside_unit_interval=sum(w<0 or w>1 for w in weights),zero_rank_boundaries=boundaries))


def git_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def save_checkpoint(path,learner,scenario,episode,best_energy,environment_config):
    if learner.buffer:raise RuntimeError('Checkpoints are written only at completed episode/rollout boundaries')
    modules={n:getattr(learner,n).state_dict() for n in ('task','actor','critic','intrinsic')}
    optimizers={n:getattr(learner,n+'_opt').state_dict() for n in ('task','actor','critic','intrinsic')}
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    obj=dict(model_version=MODEL_VERSION,modules=modules,optimizers=optimizers,paper_config=learner.cfg.as_dict(),
        episode=episode,steps=learner.steps,updates=learner.updates,best_validation_energy=best_energy,
        normalization=scenario.normalization_metadata(),protocol=scenario.protocol().metadata(),
        environment=environment_config,scenario=scenario.describe(),rng=rng,git_commit=git_commit(),
        recurrent_hidden=learner.hidden.detach().cpu(),vm_count=learner.vm_count,numerical_precision='float64')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,path)


def load_checkpoint(path,scenario,device='cpu',restore_training=False):
    # Locally produced training files contain RNG/optimizer state, not weights alone.
    obj=torch.load(path,map_location=device,weights_only=False)
    if obj.get('model_version')!=MODEL_VERSION:raise ValueError('Historical IRWS checkpoint is not a paper-model checkpoint')
    check_checkpoint(obj.get('normalization'),scenario.normalization_metadata())
    if obj.get('protocol')!=scenario.protocol().metadata():raise ValueError('Checkpoint experiment protocol mismatch')
    cfg=dict(obj['paper_config']);cfg['device']=device
    for k in ('actor_hidden','critic_hidden'):cfg[k]=tuple(cfg[k])
    learner=PaperLearner(obj['vm_count'],PaperConfig(**cfg))
    if learner.vm_count!=sum(scenario.vms_per_host):raise ValueError('Source/target VM dimensions are incompatible')
    for name,weights in obj['modules'].items():getattr(learner,name).load_state_dict(weights)
    learner.steps=obj['steps'];learner.updates=obj['updates']
    if restore_training:
        for name,state in obj['optimizers'].items():getattr(learner,name+'_opt').load_state_dict(state)
        random.setstate(obj['rng']['python']);np.random.set_state(obj['rng']['numpy']);torch.set_rng_state(obj['rng']['torch'].cpu())
        if torch.cuda.is_available() and obj['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(obj['rng']['cuda'])
    return learner,obj


def evaluate(learner,scenario,seeds,split='test',kwargs=None):
    protocol=scenario.protocol()
    allowed=protocol.validation if split=='validation' else protocol.test
    if not set(seeds)<=set(allowed):raise ValueError('Evaluation attempted to cross the frozen data partition')
    rows=[]
    for seed in seeds:
        env=PaperCloudEnv(**(dict(kwargs,random_seed=seed) if kwargs is not None else env_kwargs(scenario,seed)))
        result=run_episode(env,learner,seed,False);rows.append(dict(seed=seed,**result['metrics']))
    names=[k for k in rows[0] if k!='seed']
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else 0.) for k in names}
    return rows,summary


def train(scenario,config=None,kwargs=None,out=None,resume=None):
    protocol=scenario.protocol()
    if protocol is None or scenario.episodes>protocol.episodes:raise ValueError('A valid fixed experiment budget is required')
    if scenario.normalization_mode!=VERSION:raise ValueError('Paper training requires f18_shared_v1')
    directory=Path(out) if out is not None else output_dir(scenario,'ckpts')
    if resume:
        if Path(resume).name != 'final.pth':raise ValueError('Resume from final.pth to preserve the latest run state')
        if directory.exists() and directory.resolve()!=Path(resume).resolve().parent:
            raise ValueError('Resume must continue its own output directory')
    else:protect_training_outputs(directory,directory/'logs')
    kwargs=dict(kwargs or env_kwargs(scenario,protocol.train[0]));env=PaperCloudEnv(**kwargs)
    scenario.normalization_metadata();seed_everything(protocol.train[0])
    cfg=config or PaperConfig(device=(scenario.paper_options or {}).get('device','cpu'))
    start=0;best=float('inf')
    if resume:
        learner,obj=load_checkpoint(resume,scenario,cfg.device,True);start=obj['episode']+1;best=obj['best_validation_energy']
    else:learner=PaperLearner(env.num_vms,cfg)
    directory.mkdir(parents=True,exist_ok=True);log=directory/'training.jsonl'
    for episode in range(start,scenario.episodes):
        seed=protocol.train[episode]
        result=run_episode(env,learner,seed,True)
        _,validation=evaluate(learner,scenario,protocol.validation,'validation',kwargs)
        energy=validation['total_energy']['mean']
        result.update(episode=episode,seed=seed,validation=validation,model_version=MODEL_VERSION)
        with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
        if energy<best:
            best=energy;save_checkpoint(directory/'best.pth',learner,scenario,episode,best,kwargs)
        save_checkpoint(directory/'final.pth',learner,scenario,episode,best,kwargs)
        print(f'IRWS-paper episode={episode} seed={seed} energy={result["metrics"]["total_energy"]:.6f} validation_energy={energy:.6f}',flush=True)
    return learner,directory


def train_main():
    scenario=parse_scenario('train','irws_paper')
    train(scenario,resume=(scenario.paper_options or {}).get('resume_checkpoint'))


def evaluate_main():
    scenario=parse_scenario('eval','irws_paper');device=(scenario.paper_options or {}).get('device','cpu')
    directory=Path(scenario.checkpoint_override) if scenario.checkpoint_override else output_dir(scenario,'ckpts')
    learner,_=load_checkpoint(directory/'best.pth',scenario,device)
    rows,summary=evaluate(learner,scenario,scenario.seeds)
    out=output_dir(scenario,'run')/(scenario.task+scenario.resource+'_'+scenario.deadline)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'episodes.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,normalization=scenario.normalization_metadata(),
        protocol=scenario.protocol().metadata(),source_checkpoint=str(directory/'best.pth')),indent=2),encoding='utf-8')