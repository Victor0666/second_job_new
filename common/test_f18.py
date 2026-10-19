"""F18 fixed-scale preprocessing, calibration and physical-regression tests."""
from contextlib import redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from common.f18_state_normalization import (VERSION, SCALE_UNITS, StateNormalizer, scene_id,
    queue_features, queue_metrics, counts, check_checkpoint, normalization_diagnostics)
from common.f18_calibrate_normalization import (nine_scenarios, percentiles, validate_seeds,
    cache_info, sample_episode, summarize, main as calibrate)
from common.scenarios import Scenario
from common.experiment_protocol import protocol_binding
from common.precompute_deadlines import main as precompute
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as MA
from environments.marl import CloudWorkflowEnv_MARL as MARL
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost as PD
from environments.irws import IRWSEnvRouteA as IRWS


def fixture_config(path):
    scales=dict(R_ref=3,T_ref=123.,workload_mi_ref=400.,input_bits_ref=2e9,output_bits_ref=3e9,
                up_rank_ref=200.,down_rank_ref=300.,vm_pc_ref=8.,vm_bw_ref=8000.,host_pc_ref=60.)
    obj=dict(normalization_version=VERSION,units=SCALE_UNITS,scenes={scene_id('S',r,d):dict(scales)
        for r in 'SML' for d in ('Loose','Medium','Tight')},
        generation=dict(**protocol_binding(),calibration_seeds={scene_id('S',r,d):[101] for r in 'SML' for d in ('Loose','Medium','Tight')}))
    path.write_text(json.dumps(obj),encoding='utf-8')
    return obj


class F18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(prefix='F18_tests_');cls.root=Path(cls.temp.name)
        cls.path=cls.root/'normalization.json';cls.config=fixture_config(cls.path)
        class LocalScenario(Scenario):
            @property
            def cache_path(self):return cls.root/('cache_'+self.resource+'.json')
        cls.Scenario=LocalScenario
        with redirect_stdout(io.StringIO()):
            for r in 'SML':
                precompute(Scenario(mode='precompute',resource=r,seed_spec='101',cache_output=str(cls.root/('cache_'+r+'.json'))))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def kwargs(self,kind,mode=VERSION):
        kw=dict(dax_paths=Scenario().dax_paths(),horizon=1e9,arrival_lambda=.03,random_seed=101,
            workflows_per_episode=50,num_hosts=3,vms_per_host=(9,8,8),normalize=True,
            deadline_cache_path=str(self.root/'cache_S.json'),deadline_mode='cache_fcfs',
            normalization_mode=mode,normalization_config=str(self.path),source_train_scene='SS_T')
        if kind is MARL:kw.update(dump_deadlines=False,print_deadlines=False,alpha_deadline_choices=(2.,3.),alpha_deadline_probs=(.8,.2))
        elif kind is MA:kw['deadline_alpha_small_prob']=.8
        else:kw.update(deadline_alpha_choices=[2.,3.],deadline_alpha_probs=[.8,.2])
        return kw

    def test_T01_nine_scenes(self):
        self.assertEqual([scene_id(s.task,s.resource,s.deadline) for s in nine_scenarios()],
            ['SS_L','SS_M','SS_T','SM_L','SM_M','SM_T','SL_L','SL_M','SL_T'])

    def test_T02_linear_percentiles(self):
        a=[1,2,8,9,100];p=percentiles(a)
        np.testing.assert_array_equal([p['P50'],p['P95'],p['P99']],np.percentile(a,[50,95,99],method='linear'))
        self.assertEqual(p['Max'],100)
        with self.assertRaises(ValueError):percentiles([])

    def test_T03_T07_real_reference_reproducible_and_cache_readonly(self):
        scene=self.Scenario(mode='train',seed_spec='101')
        before=(self.root/'cache_S.json').read_bytes()
        a=sample_episode(scene,101,True);b=sample_episode(scene,101,True)
        self.assertEqual(a,b);self.assertEqual(a['status'],'ok')
        self.assertEqual(len(a['deadlines']),50)
        self.assertEqual(before,(self.root/'cache_S.json').read_bytes())
        self.assertGreater(a['assignments'],1000)
        self.assertEqual(len(a['raw_records']),len({r['current_time'] for r in a['raw_records']}))
        self.assertTrue(all(r['ready_count']>0 for r in a['raw_records']))
        self.assertTrue(all(r['ready_count']<=r['unfinished_count'] for r in a['raw_records']))

    def test_T04_T05_ready_and_relative_deadlines(self):
        env=MA(**self.kwargs(MA));env.reset();m=queue_metrics(env)
        self.assertEqual(m['ready_count'],sum(s=='Ready' for s in env.task_state))
        self.assertEqual(m['running_count'],0)
        self.assertTrue(all(w.deadline-w.arrival_time>0 for w in env.workflows))
        self.assertEqual(m['unfinished_count'],len(env.task_state))
        env.max_ready=1
        self.assertEqual(queue_metrics(env)['ready_count'],m['ready_count'])

    def test_T06_cache_validation_and_seed_isolation(self):
        s=self.Scenario();info=cache_info(s,[101]);self.assertIn('sha256',info)
        with self.assertRaisesRegex(ValueError,'f22_cache'):cache_info(s,[102])
        validate_seeds(range(101,131))
        for seeds in ([600],[700],[729],[601]):
            with self.assertRaises(ValueError):validate_seeds(seeds)

    def test_T08_four_algorithms_same_parameters(self):
        metas=[]
        for kind in (MA,MARL,PD,IRWS):
            env=kind(**self.kwargs(kind));metas.append(env.state_normalizer.metadata)
        self.assertTrue(all(m==metas[0] for m in metas))

    def test_T09_rr_urr_not_complements(self):
        env=MA(**self.kwargs(MA));env.reset()
        q=queue_features(env);m=counts(env)
        self.assertEqual(q[0],min(1,m['ready_count']/3))
        self.assertEqual(q[1],m['blocked_count']/max(1,m['unfinished_count']))
        env.task_state=['Ready','unReady','Running','Finished'];env.ready_task_ids=[0,0]
        env.task_meta=[(0,0)]*4;env.task_ready_time=[0]*4
        with patch.object(env,'_task_duration',return_value=1.):
            q=queue_features(env)
        self.assertAlmostEqual(q[0],1/3);self.assertAlmostEqual(q[1],1/3)
        self.assertNotAlmostEqual(q[0]+q[1],1)
        env.task_state=['Finished']*4;env.ready_task_ids=[]
        self.assertEqual(queue_features(env)[1],0)

    def test_T10_T11_T14_manager_host_shared_time_slack_and_dimensions(self):
        for kind in (MA,MARL):
            env=kind(**self.kwargs(kind));env.reset()
            if kind is MA:
                ta=env.get_manager_state();ha=env._compute_block2()
                self.assertEqual(ta.shape,(15,));self.assertEqual(ha.shape,(10,))
                np.testing.assert_array_equal(ta[[0,1,2,3,4,8,9]],ha[[0,1,2,3,4,7,8]])
                tid=env.ready_task_ids[0];task=env._compute_block1(tid)[0]
                self.assertEqual(task.shape,(12,))
                self.assertAlmostEqual(task[0],env.task_mi[tid]/400.,places=5)
                env.horizon=9876543.
                np.testing.assert_array_equal(ta,env.get_manager_state())
                for wf in env.workflows:wf.deadline=-100.
                self.assertLess(env.get_manager_state()[8],0)
                obs,mask=env._build_vm_obs_for_task_host(tid,env.host_ids[0])
                block=obs[-env.max_vms_per_host*5:].reshape(-1,5)
                j=env.host_to_vm_indices[env.host_ids[0]][0]
                self.assertAlmostEqual(block[0,4],env._task_duration(tid,j)/123.,places=4)
            else:
                g=env._build_global_obs();self.assertEqual(g.shape,(10,))
                tid=env.ready_task_ids[0];maps=env.host_vm_indices[0]
                mapping=list(maps)+[-1]*(env.max_vms_per_host-len(maps))
                local=env._build_local_obs(tid,0,mapping)
                np.testing.assert_array_equal(g[[0,1,2,3,4,8,9]],local[12:22][[0,1,2,3,4,7,8]])
                self.assertEqual(local.shape,(env.local_obs_dim,))
                self.assertTrue(np.isfinite(g).all())
                for wf in env.workflows:wf.deadline=-100.
                self.assertLess(env._build_global_obs()[8],0)

    def test_T12_pd3qn_attribute_scaling_no_l2(self):
        env=PD(**self.kwargs(PD));obs,_=env.reset();gid=env._cur_task_gid
        self.assertAlmostEqual(obs[0],env.current_time/123.,places=5)
        self.assertAlmostEqual(obs[2],env.task_mi[gid]/400.,places=5)
        self.assertAlmostEqual(obs[3],env.task_in_bits[gid]/2e9,places=5)
        self.assertFalse(np.isclose(np.linalg.norm(obs),1.))
        self.assertEqual(len(obs),6+3*env.num_vms)

    def test_T13_irws_attribute_scaling_candidate_cap_unchanged(self):
        env=IRWS(**self.kwargs(IRWS));env.reset();gid=env.ready_task_ids[0]
        cap=env._max_ready_tasks_int;feat=env._build_task_features([gid])[0]
        self.assertAlmostEqual(feat[0],env.task_mi[gid]/400.,places=5)
        self.assertAlmostEqual(feat[3],env.task_up_rank[gid]/200.,places=5)
        vm=env._build_vm_obs_for_task(gid)
        self.assertEqual(vm.shape,(7+6*env.num_vms,));self.assertEqual(cap,env._max_ready_tasks_int)
        self.assertAlmostEqual(vm[9],env.vms[0].pc/8.,places=5)

    def test_T15_source_scene_checkpoint_metadata(self):
        s=Scenario(task='L',train_task='S',resource='M',deadline='Loose',train_resource='S',train_deadline='Tight',normalization_config=str(self.path))
        self.assertEqual(s.normalization_kwargs()['source_train_scene'],'SS_T')
        self.assertEqual(s.normalization_metadata()['R_ref'],3)
        source=Scenario(mode='train',normalization_config=str(self.path))
        self.assertEqual(s.model_name('pd3qn'),source.model_name('pd3qn'))
        meta=s.normalization_metadata();check_checkpoint(meta,meta)
        for key in ('source_train_scene','normalization_config_hash','normalization_version','R_ref','T_ref'):
            bad=dict(meta);bad[key]='wrong'
            with self.assertRaises(ValueError):check_checkpoint(bad,meta)
        with self.assertRaises(ValueError):check_checkpoint(None,meta)
        check_checkpoint(None,{'normalization_version':'legacy'})

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        from common.d3qn_agent_EXDehid import D3QNAgent
        from common.f18_state_normalization import bind_normalization
        source=Scenario(mode='train',normalization_config=str(self.path))
        a=D3QNAgent(3,2,hidden_dims=(4,),device=torch.device('cpu'))
        b=D3QNAgent(3,2,hidden_dims=(4,),device=torch.device('cpu'))
        bind_normalization(a,source);bind_normalization(b,source)
        path=self.root/'roundtrip.pth';a.save(str(path));b.load(str(path))
        for x,y in zip(a.online.parameters(),b.online.parameters()):torch.testing.assert_close(x,y)
        b.normalization_metadata['source_train_scene']='SM_T'
        with self.assertRaises(ValueError):b.load(str(path))
        b.normalization_metadata={'normalization_version':'legacy'}
        with self.assertRaises(ValueError):b.load(str(path))

    def test_pd3qn_per_checkpoint_metadata(self):
        from common.d3qn_agent_PER import D3QNAgent
        from common.f18_state_normalization import bind_normalization
        scenario=Scenario(mode='train',normalization_config=str(self.path))
        a=D3QNAgent(3,2,hidden_dims=(4,),device=torch.device('cpu'))
        b=D3QNAgent(3,2,hidden_dims=(4,),device=torch.device('cpu'))
        bind_normalization(a,scenario);bind_normalization(b,scenario)
        path=self.root/'per_roundtrip.pth';a.save(str(path));b.load(str(path))
        b.normalization_metadata['normalization_config_hash']='wrong'
        with self.assertRaises(ValueError):b.load(str(path))

    def test_summary_missing_resume_and_dry_run(self):
        out=self.root/'summary';out.mkdir(exist_ok=True);(out/'episodes').mkdir(exist_ok=True)
        manifest=dict(**protocol_binding(),scenes={'SS_T':dict(seeds=[101])})
        self.assertEqual(summarize(out,manifest),1)
        import csv
        with (out/'ready_percentiles_9scenarios.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),9);self.assertTrue(all(r['P99']=='' for r in rows))
        with redirect_stdout(io.StringIO()) as f:self.assertEqual(calibrate(['--all-nine','--dry-run']),0)
        self.assertEqual(len(json.loads(f.getvalue())['scenes']),9)

    def test_resume_skips_success_and_summary_only(self):
        out=self.root/'resume'
        episode=dict(status='ok',seed=101,ready=[1,2,4],deadlines=[10,20],
            physical={k:[1.,2.,3.] for k in ('workload_mi_ref','input_bits_ref','output_bits_ref','up_rank_ref','down_rank_ref')},
            empty_observations=1,decisions=3,assignments=2,vm_pc_ref=8.,vm_bw_ref=8000.,host_pc_ref=40.)
        argv=['--seeds','101','--out',str(out)]
        with patch('common.f18_calibrate_normalization.cache_info',return_value={'sha256':'fixture'}), patch(
                'common.f18_calibrate_normalization.sample_episode',return_value=episode) as sample, redirect_stdout(io.StringIO()):
            self.assertEqual(calibrate(argv),0);self.assertEqual(sample.call_count,1)
            self.assertEqual(calibrate(argv+['--resume']),0);self.assertEqual(sample.call_count,1)
            self.assertEqual(calibrate(['--summary-only','--out',str(out)]),0)
            self.assertEqual(calibrate(['--seeds','102','--out',str(out),'--resume']),1)

    def test_T16_invalid_configs_and_saturation_diagnostics(self):
        n=StateNormalizer(self.path,'SS_T');self.assertLess(n.scale(-123,'T_ref'),0)
        self.assertGreater(n.scale(246,'T_ref'),1)
        with self.assertRaises(ValueError):n.scale(np.inf,'T_ref')
        for value in (0,-1,float('nan'),float('inf')):
            obj=copy.deepcopy(self.config);obj['scenes']['SS_T']['T_ref']=value
            p=self.root/'bad.json';p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):StateNormalizer(p,'SS_T')
        obj=copy.deepcopy(self.config);obj['scenes']['SS_T']['T_ref']=1e-300
        p=self.root/'tiny.json';p.write_text(json.dumps(obj))
        kw=self.kwargs(MA);kw['normalization_config']=str(p)
        with np.errstate(over='ignore',invalid='ignore'), self.assertRaises(ValueError):
            tiny=MA(**kw);tiny.reset();tiny.get_manager_state()
        env=MA(**self.kwargs(MA));env.reset();queue_features(env);queue_features(env)
        d=normalization_diagnostics(env)
        self.assertEqual(d['rr_observations'],1)
        self.assertGreaterEqual(d['rr_saturation_ratio'],0)
        self.assertLessEqual(d['rr_saturation_ratio'],1)

    def test_T17_identical_physics_fixed_actions_all_environments(self):
        from algorithms.ertf_fifs.evaluate import _force_fcfs_manager,_pick_first_idle_host,_pick_fifs_vm
        def snapshot(env):
            return (float(env.current_time),tuple(env.task_state),tuple(env.vm_available_at),
                    tuple(float(w.deadline) for w in env.workflows),tuple(env.arrival_times))
        for kind in (MA,MARL,PD,IRWS):
            a=kind(**self.kwargs(kind));b=kind(**self.kwargs(kind,'legacy'));a.reset();b.reset()
            self.assertEqual(snapshot(a),snapshot(b))
            assignments=0
            while True:
                self.assertEqual(snapshot(a),snapshot(b))
                if kind is MA:
                    if a.done_flag:break
                    _force_fcfs_manager(a);_force_fcfs_manager(b)
                    while True:
                        x,ok=a.get_host_state_for_next_assignment();y,ok2=b.get_host_state_for_next_assignment()
                        self.assertEqual(ok,ok2)
                        if not ok:break
                        np.testing.assert_array_equal(x['mask'],y['mask']);h=_pick_first_idle_host(a,x['mask'])
                        a.host_select(h);b.host_select(h)
                        x,ok=a.get_vm_state_for_current_task();y,ok2=b.get_vm_state_for_current_task()
                        np.testing.assert_array_equal(x['mask'],y['mask']);v=_pick_fifs_vm(a,x['mask'],h)
                        r1=a.vm_assign(v);r2=b.vm_assign(v);self.assertEqual(r1,r2);assignments+=1
                    self.assertEqual(a.finish_phase_and_advance(),b.finish_phase_and_advance())
                elif kind is PD:
                    if a.done_flag:break
                    np.testing.assert_array_equal(a.get_action_mask(),b.get_action_mask())
                    act=int(np.flatnonzero(a.get_action_mask())[0]);x=a.step(act);y=b.step(act)
                    self.assertEqual(x[1:],y[1:]);assignments+=1
                elif kind is IRWS:
                    if a.done_flag:break
                    ca=a.get_ready_task_candidates();cb=b.get_ready_task_candidates()
                    # Use the same policy-independent earliest Ready task.
                    gid=min(a.ready_task_ids,key=lambda i:(a.task_ready_time[i],i))
                    a.select_task(gid);b.select_task(gid)
                    x=a.get_vm_obs_and_mask();y=b.get_vm_obs_and_mask()
                    np.testing.assert_array_equal(x['mask'],y['mask']);act=int(np.flatnonzero(x['mask'])[0])
                    r1=a.step(act);r2=b.step(act)
                    self.assertEqual(r1[1:],r2[1:]);assignments+=1
                else:
                    if a.done_flag:break
                    # MARL phase interface handled by its public two-step methods below.
                    x=a.get_global_state();y=b.get_global_state()
                    h=int(np.flatnonzero(x[1])[0]);np.testing.assert_array_equal(x[1],y[1])
                    x=a.get_local_state(h);y=b.get_local_state(h)
                    v=int(np.flatnonzero(x[1])[0]);np.testing.assert_array_equal(x[1],y[1])
                    r1=a.step(h,v);r2=b.step(h,v)
                    self.assertEqual(r1[1:],r2[1:]);assignments+=1
                self.assertLess(assignments,10000)
            self.assertGreater(assignments,1000)
            self.assertEqual(snapshot(a),snapshot(b))
            self.assertAlmostEqual(a.total_energy,b.total_energy,places=8)


if __name__=='__main__':unittest.main()