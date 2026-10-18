"""Single-batch finite-update probe with production network widths and real observations."""
import argparse
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from common.scenarios import Scenario
from common.d3qn_agent_EXDehid import D3QNAgent
from common.d3qn_agent_PER import D3QNAgent as PERAgent
from algorithms.marl.agents import A2CAgent, A2CConfig, LocalSharedActorIndependentCriticA2C
from algorithms.irws.vm_agent import PPOVMAgent, PPOConfig

HIDDEN = (1024, 1024, 512, 512, 256)


def probe():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    result = {'description': 'Production widths; real decision observations repeated only to form one update batch. Not a convergence run.', 'learners': {}}
    paths = Scenario().dax_paths()
    from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
    env = CloudWorkflowEnv_VMAgents(dax_paths=paths, workflows_per_episode=1, random_seed=2, deadline_mode='none')
    env.reset()
    state, ok = env.get_host_state_for_next_assignment()
    env.host_select(int(np.flatnonzero(state['mask'] > .5)[0]))
    state, ok = env.get_vm_state_for_current_task()
    action = int(np.flatnonzero(state['mask'] > .5)[0])
    _, reward, _ = env.vm_assign(action)
    next_host, ok = env.get_host_state_for_next_assignment()
    if not ok:
        env.finish_phase_and_advance()
        next_host, ok = env.get_host_state_for_next_assignment()
    assert ok
    env.host_select(int(np.flatnonzero(next_host['mask'] > .5)[0]))
    state2, ok = env.get_vm_state_for_current_task()
    assert ok
    def record(name, agent, loss):
        assert math.isfinite(loss)
        net = getattr(agent, 'online', getattr(agent, 'net', None))
        assert all(bool(torch.isfinite(p).all()) for p in net.parameters())
        result['learners'][name] = dict(loss=loss, parameters=sum(p.numel() for p in net.parameters()), diagnostics=agent.f11.snapshot())
    agent = D3QNAgent(len(state['obs']), len(state['mask']), hidden_dims=HIDDEN, gamma=.95, batch_size=256, device=torch.device('cpu'))
    for _ in range(256): agent.remember(state['obs'], state['mask'], action, reward, state2['obs'], state2['mask'], 0)
    record('MAHDRL_VA_D3QN', agent, agent.update()); del agent; gc.collect()

    from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost
    env = CloudWorkflowEnv_PD3QN_Cost(dax_paths=paths, workflows_per_episode=1, random_seed=2, deadline_mode='none')
    obs, _ = env.reset(); mask = env.get_action_mask(); action = int(np.flatnonzero(mask > .5)[0])
    obs2, reward, term, trunc, _ = env.step(action)
    agent = PERAgent(len(obs),len(mask),hidden_dims=HIDDEN,gamma=.99,batch_size=256,device=torch.device('cpu'))
    for _ in range(256): agent.remember(obs,mask,action,reward,obs2,env.get_action_mask(),term)
    record('PD3QN_PER',agent,agent.update());del agent;gc.collect()

    from environments.marl import CloudWorkflowEnv_MARL
    env = CloudWorkflowEnv_MARL(dax_paths=paths, workflows_per_episode=1, random_seed=2,
                               deadline_mode='poisson_heftlike', dump_deadlines=False, print_deadlines=False)
    obs,_=env.reset();host=int(np.flatnonzero(obs['global_mask']>.5)[0])
    local,mask,vm_map=env.get_local_state(env.host_ids[host]);act=int(np.flatnonzero(mask>.5)[0])
    obs2,rg,rl,done,_=env.step(host,act)
    next_host=int(np.flatnonzero(obs2['global_mask']>.5)[-1]);local2,mask2,_=env.get_local_state(env.host_ids[next_host])
    cfg=A2CConfig(hidden_dims=HIDDEN)
    agent=A2CAgent(len(obs['global_obs']),len(obs['global_mask']),cfg)
    record('MARL_Global',agent,agent.update_one_step(obs['global_obs'],obs['global_mask'],host,rg,obs2['global_obs'],obs2['global_mask'],done));del agent;gc.collect()
    agent=LocalSharedActorIndependentCriticA2C(len(local),len(mask),env.num_hosts,cfg)
    record('MARL_Local',agent,agent.update_one_step(local,mask,act,rl,local2,mask2,done,host_index=host,next_host_index=next_host));del agent;gc.collect()

    from environments.irws import IRWSEnvRouteA
    env=IRWSEnvRouteA(dax_paths=paths,workflows_per_episode=1,random_seed=2,deadline_mode='none');env.reset()
    env.select_task(env.get_ready_task_candidates()['gids'][0]);state=env.get_vm_obs_and_mask()
    agent=PPOVMAgent(PPOConfig(len(state['obs']),len(state['mask'])),hidden=HIDDEN)
    act,info=agent.select_action(state['obs'],state['mask']);_,reward,term,trunc,_=env.step(act)
    env.select_task(env.get_ready_task_candidates()['gids'][0]);state2=env.get_vm_obs_and_mask()
    tail=agent.value(state2['obs'],state2['mask'])
    for _ in range(256):agent.remember(state['obs'],state['mask'],act,info['logp'],info['value'],reward,0)
    record('IRWS_VM_PPO',agent,agent.update(last_value=tail))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    result=probe();Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Production-width finite-update probes passed:', ', '.join(result['learners']))
