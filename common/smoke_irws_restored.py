"""One small common-objective IRWS train/validation/save/load/test, no formal run."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
import numpy as np
from common.scenarios import Scenario
from common.experiment_protocol import protocol_binding
from common.workflow_opt import energy_from_records, exec_time_components
from common.precompute_deadlines import main as precompute
from common.f18_calibrate_normalization import sample_episode,summarize
from algorithms.irws_restored.config import PaperConfig,COMMON_REWARD
from algorithms.irws_restored.environment import PaperCloudEnv
from algorithms.irws_restored.learner import PaperLearner
from algorithms.irws_restored.runner import train,evaluate,env_kwargs,seed_everything,load_checkpoint,artifact_name,run_episode


def parameters(module):return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def verify_execution(kwargs):
    env=PaperCloudEnv(**kwargs)
    learner=PaperLearner(env.num_vms,PaperConfig())
    result=run_episode(env,learner,32,False)
    max_duration_gap=0.;queued=0
    for gid,(vm_id,host_id) in env.task_placement.items():
        task=env._task_obj_from_gid(gid);vm=env.vms[vm_id]
        duration=exec_time_components(task,vm.pc,vm.bw,parents_loc=env._build_parents_loc(gid),
            child_vm_id=vm_id,child_host_id=host_id,zero_if_same_vm=True,zero_if_same_host=False)[-1]
        max_duration_gap=max(max_duration_gap,abs(env.task_end_time[gid]-env.task_start_time[gid]-duration))
        assert env.task_start_time[gid]>=env.task_dispatch_time[gid]-1e-8
        queued+=int(env.task_start_time[gid]>env.task_dispatch_time[gid]+1e-8)
        base=gid-env.task_meta[gid].local_id
        assert all(env.task_start_time[gid]>=env.task_end_time[base+p]-1e-8 for p in task.parents)
    for vm in range(env.num_vms):
        gids=sorted([g for g,p in env.task_placement.items() if p[0]==vm],key=lambda g:env.task_start_time[g])
        assert all(env.task_end_time[a]<=env.task_start_time[b]+1e-8 for a,b in zip(gids,gids[1:]))
    from algorithms.irws_paper.environment import PaperCloudEnv as PreviousEnv
    from algorithms.irws_paper.learner import PaperLearner as PreviousLearner
    from algorithms.irws_paper.config import PaperConfig as PreviousConfig
    from algorithms.irws_paper.runner import run_episode as previous_episode
    previous_kwargs={k:v for k,v in kwargs.items() if k not in ('queue_capacity','slot_duration')}
    previous=PreviousEnv(**previous_kwargs)
    previous_episode(previous,PreviousLearner(previous.num_vms,PreviousConfig()),32,False)
    assert len(env.workflows)==len(previous.workflows)
    for a,b in zip(env.workflows,previous.workflows):
        assert a.arrival_time==b.arrival_time and a.deadline==b.deadline
        assert list(a.graph.edges)==list(b.graph.edges)
        assert [(t.workload_mi,t.parent_in_bits,t.ext_in_bits,t.out_file_size_sum_bits) for t in a.tasks]==[(t.workload_mi,t.parent_in_bits,t.ext_in_bits,t.out_file_size_sum_bits) for t in b.tasks]
    np.testing.assert_array_equal(env.task_baseline_finish,previous.task_baseline_finish)
    actual=energy_from_records(env.records,env.hosts)
    assert max_duration_gap<1e-7 and abs(actual-env.total_energy)<1e-7
    return dict(assignments=result['assignments'],queued_assignments=queued,max_duration_gap=max_duration_gap,
                energy_reintegration_gap=abs(actual-env.total_energy),all_tasks_finished=True,
                parent_dependencies_respected=True,no_VM_overlap=True,paired_instances_identical_to_previous=True)


def run():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='IRWS_restored_smoke_') as directory, patch(
            'algorithms.irws_restored.runner.output_dir', side_effect=lambda _scene,kind:Path(directory)/'verification'/kind):
        root=Path(directory)
        class LocalScenario(Scenario):
            @property
            def cache_path(self):return root/'cache.json'
        protocol=root/'protocol.json';protocol.write_text(json.dumps(dict(version='restored_smoke_disjoint_v1',train=list(range(32,102)),validation=[632],
            test=[2],episodes=1,selection='minimum_validation_total_energy')),encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute',seed_spec='2,32,101,632',cache_output=str(root/'cache.json')))
            scene=LocalScenario(mode='train',episodes=1,experiment_protocol=str(protocol))
            calibration=sample_episode(scene,101)
            (root/'episodes').mkdir();(root/'episodes/SS_T_101.json').write_text(json.dumps(calibration),encoding='utf-8')
            manifest=dict(**protocol_binding(protocol),scenes={'SS_T':dict(seeds=[101])},verification_only=True)
            assert summarize(root,manifest,protocol)==0
        scene=LocalScenario(mode='train',episodes=1,experiment_protocol=str(protocol),normalization_config=str(root/'f18_shared_normalization_v1.json'))
        kwargs=env_kwargs(scene,32);kwargs['workflows_per_episode']=2
        physical_trace=verify_execution(kwargs)
        cfg=PaperConfig();seed_everything(scene.model_seed);initial=PaperLearner(sum(scene.vms_per_host),cfg)
        before={n:parameters(m) for n,m in [('embedding',initial.task.embedding),('selector',initial.task.score),
            ('actor',initial.actor),('critic',initial.critic),('intrinsic',initial.intrinsic)]}
        learner,out=train(scene,cfg,kwargs,root/'models')
        changes={n:float((before[n]-parameters(m)).abs().max()) for n,m in [('embedding',learner.task.embedding),('selector',learner.task.score),
            ('actor',learner.actor),('critic',learner.critic),('intrinsic',learner.intrinsic)]}
        assert changes['selector']==0 and all(v>0 for k,v in changes.items() if k!='selector'),changes
        loaded,obj=load_checkpoint(out/'best.pth',scene)
        rows,summary=evaluate(loaded,scene,[2],kwargs=kwargs)
        rows2,_=evaluate(learner,scene,[2],kwargs=kwargs)
        assert rows==rows2
        result=json.loads((out/artifact_name(scene,'train')).read_text(encoding='utf-8').splitlines()[-1])
        assert result['updates'] and all(np.isfinite(list(u['losses'].values())).all() for u in result['updates'])
        return dict(queued_common_physics_checks=physical_trace,model_version=obj['model_version'],paper_config=cfg.as_dict(),training_result=result,
            parameter_max_changes=changes,checkpoint_deterministic_roundtrip=True,fixed_selector=True,test_rows=rows,test_summary=summary,
            normalization=obj['normalization'],protocol=obj['protocol'],
            verification_overrides=dict(episodes=1,workflows_per_episode=2,normalization_calibration_seeds=[101],
                temporary_caches=True,network_dimensions='published hidden widths, graph-only Q, no explicit deadline or Ready-wait context'),
            formal_training_executed=False,ablations_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',default='docs/IRWS_restored_graph_only_smoke.json');args=p.parse_args()
    result=run();Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Restored IRWS: four learned groups changed, Q stayed fixed; queue physics, checkpoint and test passed')
