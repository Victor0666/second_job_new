"""Small, isolated training demonstration; never a formal benchmark."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import csv
import importlib
import json
import time
from unittest.mock import patch
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common.scenarios import Scenario
from common.precompute_deadlines import main as precompute
from common.experiment_measurements import TrainingRecorder, EvaluationRecorder
from common.workflow_opt import energy_from_records
from common.f22_instances import instance_identity

OUT = Path(__file__).resolve().parent
RUN = OUT / 'artifacts'
EPISODES = 20
EDGES = [(0,2),(0,3),(1,3),(1,4),(2,5),(3,5),(3,6),(4,6),(5,7),(6,7)]

class SmallScenario(Scenario):
    @property
    def vms_per_host(self): return (2,2,2)
    @property
    def cache_path(self): return RUN / 'cache.json'
    def dax_paths(self, directory=None): return [str(RUN / 'Example_8.xml')]
    def model_name(self, algorithm): return str(RUN / algorithm / 'logs')
    def log_dir(self, algorithm): return RUN / algorithm / 'logs'
    def checkpoint_dir(self, algorithm): return RUN / algorithm / 'models'

def trace_environment(env):
    env.gantt = []
    def record(gid, vmid, records):
        meta=env.task_meta[gid]
        wid=int(meta.wf_id if hasattr(meta,'wf_id') else meta[0])
        tid=int(meta.local_id if hasattr(meta,'local_id') else meta[1])
        r=next(r for r in records if r.vm_pc>0)
        env.gantt.append(dict(gid=int(gid),workflow=wid,task=tid,vm=int(vmid),
            host=int(r.server_id),start=float(r.start_time),end=float(r.end_time)))
    recs=lambda: env.records if hasattr(env,'records') else env._records
    method=next((n for n in ('_assign_task_to_specific_vm','_assign_task_to_vm') if hasattr(env,n)),None)
    if method:
        original=getattr(env,method)
        def assign(gid,vm,*a,**kw):
            before=len(recs());result=original(gid,vm,*a,**kw)
            vmid=env.vm_ids[int(vm)] if method=='_assign_task_to_specific_vm' else int(vm)
            record(gid,vmid,recs()[before:]);return result
        setattr(env,method,assign)
    else:
        original=env.step
        def step(action):
            gid=env._cur_task_gid;before=len(recs());result=original(action)
            record(gid,env.task_assigned_vm_id[gid],recs()[before:]);return result
        env.step=step
    return env

def inspect_result(name, env, scene):
    start=float(env.arrival_times[0]);end=float(env.current_time)
    records=env.records if hasattr(env,'records') else env._records
    busy=[r for r in records if r.vm_pc>0]
    full=energy_from_records(busy,env.hosts,start_time=start,end_time=end)
    old=0.
    for hid,host in env.hosts.items():
        rr=[r for r in busy if r.server_id==hid]
        if rr:old+=energy_from_records(rr,{hid:host},start_time=min(r.start_time for r in rr),end_time=max(r.end_time for r in rr))
    base=sum(h.power(0) for h in env.hosts.values())*(end-start)
    assert env.done_flag and all(s=='Finished' for s in env.task_state)
    assert len(env.gantt)==len(env.task_state)==16
    assert abs(full-env.total_energy)<1e-6,(name,full,env.total_energy)
    for vm in env.vms:
        rows=sorted((r for r in env.gantt if r['vm']==vm),key=lambda r:r['start'])
        assert all(a['end']<=b['start']+1e-8 for a,b in zip(rows,rows[1:]))
    for wid in (0,1):
        rows={r['task']:r for r in env.gantt if r['workflow']==wid}
        assert all(rows[p]['end']<=rows[c]['start']+1e-8 for p,c in EDGES)
    for row in env.gantt:
        row['start']-=start;row['end']-=start
    result=dict(algorithm=name,origin=start,duration_s=end-start,total_J=full,
        idle_baseline_J=base,load_extra_J=full-base,old_window_J=old,
        omitted_idle_J=full-old,old_to_new_pct=100*(full/old-1),
        instance=instance_identity(env,scene,700),arrivals=[float(x)-start for x in env.arrival_times],
        gantt=env.gantt,host_idle_W={h:host.power(0) for h,host in env.hosts.items()})
    return result

def train_and_evaluate(name,scene):
    module=importlib.import_module('algorithms.'+name+'.train')
    envname={'mahdrl':'CloudWorkflowEnv_VMAgents','marl':'CloudWorkflowEnv_MARL',
        'pd3qn':'CloudWorkflowEnv_PD3QN_Cost','irws':'IRWSEnvRouteA'}[name]
    original_env=getattr(module,envname);kwargs_seen=[];captured=[]
    def factory(*a,**kw):
        kw['workflows_per_episode']=2
        if name=='marl':kw.update(dump_deadlines=False,print_deadlines=False)
        kwargs_seen.append(dict(kw));return original_env(*a,**kw)
    def recorder(*a,**kw):
        obj=TrainingRecorder(*a,**kw);captured.append(obj);return obj
    with ExitStack() as stack,(RUN/name/'console.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
        stack.enter_context(patch.object(module,'SCENARIO',scene))
        stack.enter_context(patch.object(module,envname,side_effect=factory))
        stack.enter_context(patch.object(module,'TrainingRecorder',side_effect=recorder))
        if hasattr(module,'D3QNAgent'):
            original=module.D3QNAgent
            def agent(*a,**kw):
                kw.update(hidden_dims=(32,16),batch_size=8,buffer_size=256,device=torch.device('cpu'))
                return original(*a,**kw)
            stack.enter_context(patch.object(module,'D3QNAgent',side_effect=agent))
        elif name=='marl':
            original=module.A2CConfig
            def cfg(*a,**kw):kw.update(hidden_dims=(32,16),device='cpu');return original(*a,**kw)
            stack.enter_context(patch.object(module,'A2CConfig',side_effect=cfg))
        else:
            original=module.PPOVMAgent
            def ppo(cfg,hidden):cfg.device='cpu';return original(cfg,hidden=(32,16))
            stack.enter_context(patch.object(module,'PPOVMAgent',side_effect=ppo))
        module.train()
    agents=captured[0].agents
    rows=[json.loads(x) for x in (scene.log_dir(name)/'training_measurements.jsonl').read_text().splitlines()]
    updates={k:sum(r['optimizer_steps'].get(k,0) for r in rows) for k in rows[-1]['optimizer_steps']}
    assert sum(updates.values())>0
    ev=importlib.import_module('algorithms.'+name+'.evaluate');evalscene=replace(scene,mode='eval',seed_spec='700')
    kw=kwargs_seen[0];kw['random_seed']=700
    for key in ('energy_reward_scale','task_baseline_norm','energy_norm_per_mi_ref','alpha_delay_host','alpha_delay_vm'):
        if hasattr(captured[0].env,key):kw[key]=getattr(captured[0].env,key)
    evaluated=[]
    def evalfactory(*a,**kwargs):
        # MARL supplies its scales through its constructor, unlike MAHDRL.
        if name=='marl':
            for k in ('energy_norm_per_mi_ref','alpha_delay_host','alpha_delay_vm'):kwargs.pop(k,None)
        env=trace_environment(factory(*a,**kwargs));evaluated.append(env);return env
    def evalrecorder(*a,**kwargs):kwargs['directory']=RUN/name/'evaluation';return EvaluationRecorder(*a,**kwargs)
    with ExitStack() as stack,(RUN/name/'evaluation.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
        stack.enter_context(patch.object(ev,'SCENARIO',evalscene))
        stack.enter_context(patch.object(ev,'EvaluationRecorder',side_effect=evalrecorder))
        if name=='mahdrl':ev.evaluate_hrl_mix_one_seed(evalfactory,kw,agents['VA'],agents['HA'],agents['TA'])
        elif name=='pd3qn':ev.evaluate_one_episode(evalfactory,kw,agents['PD3QN'])
        elif name=='marl':
            stack.enter_context(patch.object(ev,envname,side_effect=evalfactory))
            ev.evaluate_one_seed(kw,agents['Global'],agents['Local'],700,0)
        else:
            stack.enter_context(patch.object(ev,envname,side_effect=evalfactory))
            # IRWS constructor has the reward scales already in kwargs.
            for k in ('energy_norm_per_mi_ref','alpha_delay_host'):kw.pop(k,None)
            ev.evaluate_one_seed(kw,agents['Task'],agents['VM'],700)
    result=inspect_result(name,evaluated[-1],evalscene)
    result['optimizer_steps']=updates
    (RUN/name/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,round(result['total_J'],2),'J',flush=True)
    return result

def draw(results):
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    colors=['#0072B2','#D55E00'];maxend=max(r['duration_s'] for r in results)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained')
    for ax,r in zip(axes,results):
        for vm in range(6):ax.barh(vm,r['duration_s'],left=0,height=.74,color='#eeeeee',zorder=0)
        for row in r['gantt']:
            ax.barh(row['vm'],row['end']-row['start'],left=row['start'],height=.68,
                color=colors[row['workflow']],edgecolor='white',linewidth=.7,hatch='' if row['workflow']==0 else '//')
            ax.text((row['start']+row['end'])/2,row['vm'],str(row['task']),ha='center',va='center',fontsize=9,color='white',weight='bold')
        for t in r['arrivals']:ax.axvline(t,color='#555555',linestyle=':',lw=1)
        ax.axvline(r['duration_s'],color='#222222',ls='--',lw=1)
        ax.set_yticks(range(6),[f'H{v//2} / VM{v}' for v in range(6)])
        ax.invert_yaxis();ax.set_xlim(0,maxend*1.025);ax.grid(axis='x',alpha=.15)
        ax.set_title(f"{r['algorithm'].upper()}  |  duration {r['duration_s']:.1f} s  |  energy {r['total_J']/1000:.2f} kJ  |  idle baseline {100*r['idle_baseline_J']/r['total_J']:.1f}%",loc='left',fontsize=11)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Time since first workflow arrival (s)')
    fig.suptitle('Same small instance, four scheduling algorithms\n20 training episodes, small networks; illustrative run, not a performance benchmark',fontsize=13)
    fig.legend(handles=[Patch(facecolor=colors[0],label='Workflow 0'),Patch(facecolor=colors[1],hatch='//',label='Workflow 1'),Patch(facecolor='#eeeeee',label='Idle VM (host still consumes power)')],loc='outside lower center',ncol=3,frameon=False)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'gantt_four_algorithms.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');x=np.arange(4)
    base=np.array([r['idle_baseline_J']/1000 for r in results]);extra=np.array([r['load_extra_J']/1000 for r in results])
    axes[0].bar(x,base,color='#999999',label='All-host idle baseline')
    axes[0].bar(x,extra,bottom=base,color='#009E73',label='Additional power from load')
    axes[1].bar(x-.18,[r['old_window_J']/1000 for r in results],width=.36,color='#E69F00',label='Hypothetical task-only host windows')
    axes[1].bar(x+.18,base+extra,width=.36,color='#0072B2',label='Current full-cluster window')
    for ax,title in zip(axes,['Current energy decomposition','Same trajectory, different accounting']):
        ax.set_xticks(x,[r['algorithm'].upper() for r in results]);ax.set_ylabel('Energy (kJ)');ax.set_title(title)
        ax.set_ylim(0,max(base+extra)*1.24);ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Accounting comparison uses identical current-code schedules; no old-policy rerun',fontsize=11)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'energy_explanation.{ext}',dpi=180)
    plt.close(fig)

def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'summary.json').exists():
        draw(json.loads((RUN/'summary.json').read_text())['results']);return
    dax=RUN/'Example_8.xml'
    if not dax.exists():
        dax.write_text('<adag xmlns="http://pegasus.isi.edu/schema/DAX">'+''.join(f'<job id="T{i}"/>' for i in range(8))+''.join(f'<child ref="T{c}"><parent ref="T{p}"/></child>' for p,c in EDGES)+'</adag>',encoding='utf-8')
    scene=SmallScenario(mode='train',episodes=EPISODES)
    if not scene.cache_path.exists():
        with (RUN/'precompute.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
            precompute(replace(scene,mode='precompute',seed_spec=','.join(map(str,list(range(EPISODES))+[600,700])),cache_output=str(scene.cache_path)))
    results=[]
    for name in ('mahdrl','marl','pd3qn','irws'):
        (RUN/name).mkdir(exist_ok=True)
        saved=RUN/name/'result.json'
        results.append(json.loads(saved.read_text()) if saved.exists() else train_and_evaluate(name,scene))
    assert len({r['instance']['instance_hash'] for r in results})==1,'Physical instances differ'
    summary=dict(verification_only=True,episodes=EPISODES,policy='final trained parameters; deterministic evaluation, no test selection',
        train_seeds=list(range(EPISODES)),validation_seed=600,test_seed=700,workflows=2,tasks_per_workflow=8,
        resources='3 hosts x 2 VMs; each host PC=(1,2) MI/s and BW=(1000,2000) Mb/s',
        normalization='f19_shared_v1; existing SS/Tight training reference, not recalibrated on synthetic DAG',
        networks=[32,16],dqn_batch=8,dqn_buffer=256,irws_implementation='algorithms/irws (historical adapted implementation, not irws_paper)',results=results)
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'energy_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['algorithm','duration_s','total_J','idle_baseline_J','load_extra_J','old_window_J','omitted_idle_J','old_to_new_pct']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in results)
    draw(results)
    print('All checks passed; figures saved to',OUT,flush=True)

if __name__=='__main__':main()
