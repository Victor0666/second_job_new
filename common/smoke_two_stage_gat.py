"""Small real-DAX verification only; never trains the formal 600-episode model."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from algorithms.two_stage_gat.config import TwoStageGATConfig
from algorithms.two_stage_gat.environment import TwoStageGATEnv
from algorithms.two_stage_gat.networks import GraphState
from algorithms.two_stage_gat.ppo import PPOAgent, Transition
from algorithms.two_stage_gat.runner import (env_kwargs, evaluate, load_checkpoint,
    objective_metadata, seed_everything, train)
from common.scenarios import ROOT, Scenario
from common.workflow_opt import energy_from_records, exec_time_components


def vector(module):
    return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def check_author_network(path):
    """The paper MLP depth/shared decoder differ from the upstream architecture."""
    return dict(executed=False,
        upstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        reason='Paper-v4 uses VM4, two-hidden-layer MLPs and one shared decoder; '
               'author VM4/MLP3/independent-decoder weight parity does not apply')


def check_trace(kwargs):
    env=TwoStageGATEnv(**kwargs);env.reset(seed=0)
    reward_sum=0.;delay_sum=0.;increments=0.
    while not env.done_flag:
        gid=int(np.flatnonzero(env.graph_state().task_mask)[0])
        reward,_,info=env.pair_step(gid*env.num_vms)
        reward_sum+=reward;delay_sum+=info['r_delay'];increments+=info['delta_energy_total']
    max_gap=0.
    for gid,(vm_id,host_id) in env.task_placement.items():
        vm=env.vms[vm_id]
        duration=exec_time_components(env._task_obj_from_gid(gid),vm.pc,vm.bw,
            parents_loc=env._build_parents_loc(gid),child_vm_id=vm_id,child_host_id=host_id,
            zero_if_same_vm=True,zero_if_same_host=False)[-1]
        max_gap=max(max_gap,abs(env.task_end_time[gid]-env.task_start_time[gid]-duration))
        base=gid-env.task_meta[gid].local_id
        assert all(env.task_dispatch_time[gid]>=env.task_end_time[base+p]-1e-8 for p in env.task_parents[gid])
    for vm_id in env.vms:
        ordered=sorted((g for g,p in env.task_placement.items() if p[0]==vm_id),key=lambda g:env.task_start_time[g])
        assert all(env.task_end_time[a]<=env.task_start_time[b]+1e-8 for a,b in zip(ordered,ordered[1:]))
    reintegrated=energy_from_records(env.records,env.hosts)
    assert max_gap<1e-7 and abs(reintegrated-env.total_energy)<1e-7
    assert abs(increments-env.total_energy)<1e-7
    assert abs(reward_sum-(-.25*.001*env.total_energy+.75*delay_sum))<1e-7
    assert env.queued_assignments>0 and all(s=='Finished' for s in env.task_state)
    return dict(assignments=env.assignment_count,queued_assignments=env.queued_assignments,
        max_duration_gap=max_gap,energy_reintegration_gap=abs(reintegrated-env.total_energy),
        reward_telescope_gap=abs(reward_sum-(-.25*.001*env.total_energy+.75*delay_sum)),
        all_reserved_tasks_completed=True,parents_completed_before_dispatch=True,no_vm_overlap=True)


def cuda_check():
    if not torch.cuda.is_available():
        return dict(executed=False,reason='CUDA unavailable')
    rng=np.random.RandomState(3)
    state=GraphState(rng.uniform(.1,1,(4,6)).astype(np.float32),
        rng.uniform(.1,1,(4,3,4)).astype(np.float32),np.array([(0,2),(1,3)]),
        np.array([True,True,False,False]),np.ones((4,3),dtype=bool),tuple(range(4)))
    agent=PPOAgent(TwoStageGATConfig(device='cuda'))
    records=[]
    for i in range(8):
        a,lp,v=agent.act(state)
        records.append(Transition(state,a,lp,v,-.1*(i+1),i==7,0. if i==7 else v,0))
    result=agent.update(records)
    torch.cuda.synchronize()
    return dict(executed=True,device=torch.cuda.get_device_name(),loss=result['loss'],
        transitions=result['transitions'],optimizer_steps=result['optimizer_steps'])


def run():
    torch.set_num_threads(1)
    scene=Scenario(mode='train',episodes=1)
    kwargs=env_kwargs(scene,0);kwargs['workflows_per_episode']=2
    physics=check_trace(kwargs)
    cfg=TwoStageGATConfig(device='cpu')
    seed_everything(scene.model_seed);initial=PPOAgent(cfg)
    def groups(agent):
        return dict(task_gat=agent.network.actor.task_encoder,vm_encoder=agent.network.actor.vm_encoder,
            shared_decoder=agent.network.actor.decoder,
            critic_gat=agent.network.critic.task_encoder,critic_vm=agent.network.critic.vm_encoder,
            critic_value=agent.network.critic.network)
    before={n:vector(m).clone() for n,m in groups(initial).items()}
    with TemporaryDirectory(prefix='two_stage_gat_smoke_') as temporary:
        directory=Path(temporary)
        agent,out=train(scene,cfg,kwargs,directory/'models')
        changes={n:float((vector(m)-before[n]).abs().max()) for n,m in groups(agent).items()}
        assert all(v>0 for v in changes.values()),changes
        loaded,obj=load_checkpoint(out/'final.pth',scene)
        first,_=evaluate(agent,scene,[700],kwargs=kwargs)
        second,summary=evaluate(loaded,scene,[700],kwargs=kwargs,directory=directory/'measurements',checkpoint_directory=out)
        assert first==second
        training=json.loads((out/'logs/training_summary.json').read_text())
        assert agent.steps==training['budget']['assignments']
    return dict(**objective_metadata(),physics=physics,config=cfg.as_dict(),
        parameter_max_changes=changes,checkpoint_deterministic_roundtrip=True,
        test_rows=second,test_summary=summary,training_budget=training['budget'],
        normalization=obj['normalization'],cuda=cuda_check(),
        author_network_parity=check_author_network(ROOT/'tmp/two_stage_gat_source/models_gnn_agent.py'),
        verification_overrides=dict(episodes=1,workflows_per_episode=2,temporary_outputs=True,
            architecture='task6/VM4, paper MLP2/shared decoder, 4-head GAT, 8-d embeddings',four_logical_environments=True),
        formal_600_episode_training_executed=False,formal_30_seed_evaluation_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',default='docs/Two_stage_GAT_paper_v4_smoke.json')
    args=parser.parse_args()
    result=run()
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'2SD-GAT verification completed: {Path(args.out).resolve()}')
