"""Local component benchmark and bounded four-graph CUDA pressure checks."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch

from algorithms.drl_gnn.environment import DRLGNNEnv
from algorithms.drl_gnn.networks import GraphState
from algorithms.drl_gnn.ppo import PPOAgent as SourceAgent, Transition
from algorithms.drl_gnn.runner import env_kwargs
from algorithms.drl_gnn.runtime import cuda_memory_usage
from common.scenarios import Scenario
from .config import DRLGNNConfig, MODEL_VERSION
from .environment import FastDRLGNNEnv
from .networks import prepare
from .ppo import PPOAgent


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def median_ms(fn,device='cpu',repeats=10):
    for _ in range(3):fn()
    samples=[]
    for _ in range(repeats):
        sync(device);start=time.perf_counter();fn();sync(device)
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)


def components(device):
    kw=env_kwargs(Scenario(mode='train'),0)
    a,b=DRLGNNEnv(**kw),FastDRLGNNEnv(**kw)
    a.reset(seed=0);b.reset(seed=0)
    for i in range(900):
        action=(a.ready_task_ids[0]+1)*a.num_vms+i%a.num_vms
        a.pair_step(action);b.pair_step(action)
    x,y=a.graph_state(),b.graph_state()
    np.testing.assert_array_equal(x.task_features,y.task_features)
    torch.manual_seed(31);old=SourceAgent(DRLGNNConfig(device=device))
    torch.manual_seed(31);fast=PPOAgent(DRLGNNConfig(device=device))
    with torch.no_grad():
        source_logits,_=old.network(x)
        fast_logits,_=fast.network(y)
    # CUDA scatter reductions also vary between repeated calls of the original.
    torch.testing.assert_close(source_logits,fast_logits,rtol=2e-4,atol=5e-5)
    result=dict(tasks=len(x.task_features),vms=a.num_vms,
        initial_logit_max_error=float((source_logits-fast_logits).abs().max()),
        graph_source_ms=median_ms(a.graph_state),graph_fast_ms=median_ms(b.graph_state),
        source_action_ms=median_ms(lambda:old.act(x,True),device),
        fast_actor_only_ms=median_ms(lambda:fast.select_action(y),device))
    # Same mathematical update, exact default 512/128/4-epoch budget on a fixed graph.
    records=[]
    for i in range(512):
        action,logp,value=old.act(x)
        records.append(Transition(x,action,logp,value,-.1+.01*(i%11),True,0.))
    fast_records=[replace(r,state=y) for r in records]
    fast.network.load_state_dict(old.network.state_dict())
    for name,agent,transitions in (('source',old,records),('fast',fast,fast_records)):
        np.random.seed(77);sync(device);start=time.perf_counter();r=agent.update(transitions);sync(device)
        result[name+'_512_sample_update_seconds']=time.perf_counter()-start
        assert r['epochs']==4 and r['transitions']==512
    return result


def pressure():
    rows=[]
    cfg=DRLGNNConfig(device='cuda',num_envs=1,num_steps=128,num_minibatches=1,update_epochs=1)
    for task,count in (('S',1502),('M',2902),('L',5002)):
        for resource,vms in (('S',25),('M',50),('L',75)):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(31);agent=PPOAgent(cfg)
            rng=np.random.RandomState(31)
            deps=np.array([(i,i+d) for d in (1,2,vms) for i in range(count-d)],dtype=np.int64)
            mask=np.ones((count,vms),dtype=bool)
            state=prepare(GraphState(rng.uniform(.1,1.,(count,8)).astype(np.float32),
                rng.uniform(.1,1.,(vms,9)).astype(np.float32),deps,mask,mask,tuple(range(count))))
            action,logp,value=agent.act(state)
            records=[Transition(state,action,logp,value,-.1+.01*(i%11),True,0.) for i in range(128)]
            sync('cuda');start=time.perf_counter();agent.update(records);sync('cuda')
            usage=cuda_memory_usage('cuda');policy=agent.network.cuda_memory_policy
            assert usage['peak_reserved_bytes']<policy['allocator_limit_bytes']
            row=dict(scene=task+resource,tasks=count,vms=vms,minibatch=128,graphs_per_backward=4,
                epochs=1,seconds=time.perf_counter()-start,policy=policy,**usage)
            rows.append(row);print(json.dumps(row),flush=True)
            del records,state,agent
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--pressure',action='store_true')
    parser.add_argument('--out',default='outputs/analysis/drl_gnn_fast_benchmark.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    report=dict(model_version=MODEL_VERSION,device=args.device,torch=torch.__version__,cpu_threads=1,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        scope='Components on a fixed training-seed0 snapshot; single full default update. Not full training speedup. Pressure uses one epoch/128-sample minibatch; allocator excludes external CUDA.',
        components=components(args.device))
    print(json.dumps(report),flush=True)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.pressure:
        if args.device!='cuda':raise ValueError('Pressure checks require CUDA')
        report['pressure']=pressure()
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
