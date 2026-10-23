"""Small real-DAX verification only; never trains the formal 600-episode model."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.two_stage_gat.config import TwoStageGATConfig
from algorithms.two_stage_gat.environment import TwoStageGATEnv
from algorithms.two_stage_gat.networks import GraphState, TwoStageActorCritic
from algorithms.two_stage_gat.ppo import PPOAgent, Transition
from algorithms.two_stage_gat.runner import (env_kwargs, evaluate, load_checkpoint,
    objective_metadata, seed_everything, train)
from common.scenarios import ROOT, Scenario
from common.workflow_opt import energy_from_records, exec_time_components


def vector(module):
    return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def check_author_network(path):
    """Optional parity against the locally inspected frozen upstream file."""
    if not path.is_file():
        return dict(executed=False, reason='Fixed upstream file not present; no network fetch performed')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != 'ce90a71a9ab960d72b7fb7afdaf8f492ea119e542cbbdb93d0a5a6aa85e96630':
        raise ValueError('Refusing to execute a different upstream network snapshot')
    source = path.read_text(encoding='utf-8')
    for statement in ('from constants import F_TASK, F_VM, N_TASK, N_VM',
                      'from env.observation import decode_env_obs_batched',
                      'from models.base_agent import BaseAgent'):
        source = source.replace(statement, '')
    # The sole correctness fix to upstream is the documented per-sample pool index.
    source = source.replace('.repeat(num_x, 1)', '.repeat_interleave(num_x, dim=0)')
    class BaseAgent(torch.nn.Module):
        def __init__(self, device): super().__init__()
    rng = np.random.RandomState(17)
    states = [GraphState(rng.uniform(.1,1,(4,6)).astype(np.float32),
        rng.uniform(.1,1,(4,3,4)).astype(np.float32),np.asarray([(0,2),(1,2),(2,3)]),
        np.array([True,True,False,False]),np.ones((4,3),dtype=bool),tuple(range(4))) for _ in range(2)]
    holder = {}
    namespace = dict(F_TASK=6,F_VM=4,N_TASK=4,N_VM=3,BaseAgent=BaseAgent,
                     decode_env_obs_batched=lambda _:holder['obs'])
    exec(compile(source,str(path),'exec'),namespace)
    seed_everything(1)
    reference = namespace['GnnAgent'](device=torch.device('cpu'))
    adapted = TwoStageActorCritic()
    mapped = {}
    for name,value in reference.state_dict().items():
        for component in ('task_encoder','vm_encoder','task_decoder','vm_decoder'):
            name=name.replace('.'+component+'.network.','.'+component+'.')
        mapped[name]=value
    adapted.load_state_dict(mapped)
    gaps=[]
    for batch in (states[:1],states):
        dependency=np.zeros((len(batch),4,4),dtype=np.int64)
        for i,state in enumerate(batch):
            dependency[i,state.dependencies[:,0],state.dependencies[:,1]]=1
        holder['obs']=SimpleNamespace(task_features=torch.tensor(np.stack([s.task_features for s in batch])),
            vm_features=torch.tensor(np.stack([s.vm_features for s in batch])),
            task_dependencies=torch.tensor(dependency),task_mask=torch.tensor(np.stack([s.task_mask for s in batch])),
            vm_mask=torch.tensor(np.stack([s.vm_mask for s in batch])))
        actions=torch.tensor([1,5][:len(batch)])
        with torch.no_grad():
            expected=reference.get_action_and_value(torch.empty(len(batch),1),actions)
            actual=adapted(batch,actions)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
            gaps.append(float((a-b).abs().max()))
    return dict(executed=True,upstream_sha256=digest,
        single_and_batched_outputs_match=True,max_absolute_gap=max(gaps),
        upstream_context_repeat_index_corrected=True,
        comparison='identical real task/VM counts, copied author weights, same 6+4 input tensors')


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
            task_decoder=agent.network.actor.task_decoder,vm_decoder=agent.network.actor.vm_decoder,
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
            architecture='unchanged author widths, 4-head GAT, 8-d embeddings',four_logical_environments=True),
        formal_600_episode_training_executed=False,formal_30_seed_evaluation_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',default='docs/Two_stage_GAT_smoke.json')
    args=parser.parse_args()
    result=run()
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'2SD-GAT verification completed: {Path(args.out).resolve()}')
