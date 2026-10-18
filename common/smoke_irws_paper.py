"""One small common-objective IRWS train/validation/save/load/test, no formal run."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import torch
import numpy as np
from common.scenarios import Scenario
from common.workflow_opt import energy_from_records, exec_time_components
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as CommonEnv
from common.precompute_deadlines import main as precompute
from common.f18_calibrate_normalization import sample_episode,summarize
from algorithms.irws_paper.config import PaperConfig,COMMON_REWARD
from algorithms.irws_paper.environment import PaperCloudEnv
from algorithms.irws_paper.learner import PaperLearner
from algorithms.irws_paper.runner import train,evaluate,env_kwargs,seed_everything,load_checkpoint


def parameters(module):return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def verify_common_trace(scenario,kwargs,seed=32):
    a=PaperCloudEnv(**dict(kwargs,random_seed=seed))
    kw=dict(kwargs,random_seed=seed)
    kw.pop('deadline_alpha_choices');kw.pop('deadline_alpha_probs')
    kw['deadline_alpha_small_prob']=scenario.small_probability
    for key in COMMON_REWARD:kw.pop(key,None)
    b=CommonEnv(**kw)
    for key,value in COMMON_REWARD.items():setattr(b,key,value)
    a.reset(seed=seed);b.reset(seed=seed)
    steps=0;maximum_energy_gap=0.;maximum_duration_gap=0.
    while not a.done_flag:
        assert abs(a.current_time-b.current_time)<1e-9
        assert a.ready_task_ids==b.ready_task_ids
        gid=min(a.ready_task_ids,key=lambda i:(a.task_ready_time[i],i));a.select_task(gid)
        action=int(np.flatnonzero(a.get_vm_obs_and_mask()['mask'])[0])
        vm=a.vms[action];wf_id,local_id=b.task_meta[gid]
        ca=exec_time_components(a._task_obj_from_gid(gid),vm.pc,vm.bw,parents_loc=a._build_parents_loc(gid),
            child_vm_id=action,child_host_id=vm.host_id,zero_if_same_vm=True,zero_if_same_host=False)
        cb=exec_time_components(b.workflows[wf_id].tasks[local_id],vm.pc,vm.bw,parents_loc=b.parents_loc_by_wf.get(wf_id,{}),
            child_vm_id=action,child_host_id=vm.host_id,zero_if_same_vm=True,zero_if_same_host=False)
        np.testing.assert_allclose(ca,cb,atol=1e-10);maximum_duration_gap=max(maximum_duration_gap,max(abs(x-y) for x,y in zip(ca,cb)))
        now=b.current_time;b._assign_task_to_specific_vm(gid,action)
        _,_,info=a.paper_step(action)
        assert abs(info['start']-now)<1e-9 and abs(info['end']-b.task_end_time[gid])<1e-9
        b._advance_until_decision_energy_only()
        assert a.task_state==b.task_state
        np.testing.assert_allclose(a.task_end_time,b.task_end_time,atol=1e-9)
        np.testing.assert_allclose(a.task_baseline_finish,b.task_baseline_finish,atol=1e-9)
        np.testing.assert_allclose([w.deadline for w in a.workflows],[w.deadline for w in b.workflows],atol=1e-9)
        for host in a.hosts:
            la=sum(v.pc for i,v in a.vms.items() if v.host_id==host and a.vm_available_at[i]>a.current_time+1e-9)
            lb=sum(v.pc for i,v in b.vms.items() if v.host_id==host and b.vm_available_at[i]>b.current_time+1e-9)
            assert abs(la-lb)<1e-9
        ea=energy_from_records(a.records,a.hosts,start_time=a.first_arrival_time,end_time=a.current_time)
        eb=energy_from_records(b._records,b.hosts,start_time=b.first_arrival_time,end_time=b.current_time)
        maximum_energy_gap=max(maximum_energy_gap,abs(ea-eb));steps+=1
    assert b.done_flag
    assert maximum_energy_gap<1e-7 and abs(a.total_energy-b.total_energy)<1e-7
    return dict(assignments=steps,max_execution_component_gap=maximum_duration_gap,max_energy_window_gap=maximum_energy_gap,
                final_energy_gap=abs(a.total_energy-b.total_energy),deadlines_identical=True,host_loads_identical=True)


def run():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='IRWS_paper_smoke_') as directory:
        root=Path(directory)
        class LocalScenario(Scenario):
            @property
            def cache_path(self):return root/'cache.json'
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute',seed_spec='2,32,101,632',cache_output=str(root/'cache.json')))
            scene=LocalScenario(mode='train')
            calibration=sample_episode(scene,101)
            (root/'episodes').mkdir();(root/'episodes/SS_T_101.json').write_text(json.dumps(calibration),encoding='utf-8')
            manifest=dict(scenes={'SS_T':dict(seeds=[101])},verification_only=True)
            assert summarize(root,manifest)==0
        protocol=root/'protocol.json';protocol.write_text(json.dumps(dict(version='paper_smoke_disjoint_v1',train=list(range(32,102)),validation=[632],
            test=[2],episodes=1,selection='minimum_validation_total_energy')),encoding='utf-8')
        scene=LocalScenario(mode='train',episodes=1,experiment_protocol=str(protocol),normalization_config=str(root/'f18_shared_normalization_v1.json'))
        kwargs=env_kwargs(scene,32);kwargs['workflows_per_episode']=2
        physical_trace=verify_common_trace(scene,kwargs)
        cfg=PaperConfig();seed_everything(scene.model_seed);initial=PaperLearner(sum(scene.vms_per_host),cfg)
        before={n:parameters(m) for n,m in [('embedding',initial.task.embedding),('selector',initial.task.score),
            ('actor',initial.actor),('critic',initial.critic),('intrinsic',initial.intrinsic)]}
        learner,out=train(scene,cfg,kwargs,root/'models')
        changes={n:float((before[n]-parameters(m)).abs().max()) for n,m in [('embedding',learner.task.embedding),('selector',learner.task.score),
            ('actor',learner.actor),('critic',learner.critic),('intrinsic',learner.intrinsic)]}
        assert all(v>0 for v in changes.values()),changes
        loaded,obj=load_checkpoint(out/'best.pth',scene)
        rows,summary=evaluate(loaded,scene,[2],kwargs=kwargs)
        rows2,_=evaluate(learner,scene,[2],kwargs=kwargs)
        assert rows==rows2
        result=json.loads((out/'training.jsonl').read_text(encoding='utf-8').splitlines()[-1])
        assert result['updates'] and all(np.isfinite(list(u['losses'].values())).all() for u in result['updates'])
        return dict(common_mahdrl_physical_trace=physical_trace,model_version=obj['model_version'],paper_config=cfg.as_dict(),training_result=result,
            parameter_max_changes=changes,checkpoint_deterministic_roundtrip=True,test_rows=rows,test_summary=summary,
            normalization=obj['normalization'],protocol=obj['protocol'],
            verification_overrides=dict(episodes=1,workflows_per_episode=2,normalization_calibration_seeds=[101],
                temporary_caches=True,network_dimensions='published hidden widths, current deadline context added'),
            formal_training_executed=False,ablations_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',default='docs/IRWS_paper_smoke.json');args=p.parse_args()
    result=run();Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Adapted IRWS: all five network groups changed; shared reward/physics, validation, checkpoint and test passed')