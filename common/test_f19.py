"""F19 dimensionless ratio, bounded telemetry, unchanged learning and physical traces."""
from contextlib import redirect_stdout
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.f18_state_normalization import VERSION, CURRENT_VERSION, StateNormalizer, check_checkpoint
from common.f19_numerics import NumericalDiagnostics
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute


class F19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(prefix='f19_tests_');cls.root=Path(cls.temp.name)
        cls.path=cls.root/'scales.json';fixture_config(cls.path)
        cls.cache=cls.root/'cache.json'
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute',seed_spec='0',cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def env(self, kind, mode):
        cls=getattr(importlib.import_module('environments.'+kind), 'CloudWorkflowEnv_MARL' if kind=='marl' else 'CloudWorkflowEnv_VMAgents')
        kwargs=dict(dax_paths=Scenario().dax_paths(),random_seed=0,workflows_per_episode=2,
                    horizon=1e9,arrival_lambda=.03,num_hosts=3,vms_per_host=(9,8,8),max_ready_tasks=100,
                    pc_tiers=(1.,2.,4.,6.,8.),bw_tiers=(1000.,2000.,4000.,6000.,8000.),
                    deadline_mode='cache_fcfs',deadline_cache_path=str(self.cache),
                    normalization_mode=mode,normalization_config=str(self.path),source_train_scene='SS_T')
        if kind=='marl':kwargs.update(dump_deadlines=False,print_deadlines=False,alpha_deadline_choices=(2.,3.),alpha_deadline_probs=(.8,.2))
        else:kwargs['deadline_alpha_small_prob']=.8
        env=cls(**kwargs)
        with redirect_stdout(io.StringIO()):env.reset()
        return env

    def test_ratio_units_and_legacy(self):
        old=StateNormalizer(self.path,'SS_T',VERSION);new=StateNormalizer(self.path,'SS_T',CURRENT_VERSION)
        raw=555883200.
        self.assertEqual(old.io_ratio(raw),raw)
        self.assertAlmostEqual(new.io_ratio(raw),raw/((2e9+3e9)/400.))
        self.assertEqual(new.io_ratio(0.),0.)
        self.assertEqual(new.metadata['io_ratio_encoding'],'bits_per_MI_divided_by_reference_v1')
        with self.assertRaises(ValueError):StateNormalizer(self.path,'SS_T','unknown')

    def test_actual_mahdrl_marl_task_fields(self):
        reference=[]
        for kind in ('mahdrl_fcfs','marl'):
            fields=[]
            for mode in (VERSION,CURRENT_VERSION):
                env=self.env(kind,mode);gid=env.ready_task_ids[0]
                if kind=='mahdrl_fcfs':task=env._compute_block1(gid)[0]
                else:
                    maps=env.host_vm_indices[0];mapping=list(maps)+[-1]*(env.max_vms_per_host-len(maps))
                    task=env._build_local_obs(gid,0,mapping)[:4]
                fields.append(task.copy())
            self.assertEqual(fields[0].shape,fields[1].shape)
            if kind == 'marl':
                # V4 has no I/O ratio; both encodings scale its four retained fields equally.
                self.assertEqual(fields[0].shape,(4,))
                np.testing.assert_array_equal(fields[0],fields[1])
                continue
            expected=fields[0][3]/((2e9+3e9)/400.)
            self.assertAlmostEqual(fields[1][3],expected,places=5)
            np.testing.assert_array_equal(np.delete(fields[0],3),np.delete(fields[1],3))
            reference.append(fields[1][3])
        self.assertEqual(len(reference),1)

    def test_cli_default_version_and_old_override(self):
        from common.f18_state_normalization import PD3QN_UNITS
        for name in ('mahdrl','pd3qn','marl','irws','irws_paper','ablation_nohost','ablation_hostrand','ablation_vmrand','ablation_mgrrand'):
            for mode in ('train','eval'):
                for encoding in (CURRENT_VERSION,VERSION):
                    flags=['entry','--dry-run']+(['--normalization-mode',VERSION] if encoding==VERSION else [])
                    with patch('sys.argv',flags),redirect_stdout(io.StringIO()) as stream:
                        with self.assertRaises(SystemExit):parse_scenario(mode,name)
                    data=json.loads(stream.getvalue())
                    expected = PD3QN_UNITS if name == 'pd3qn' and encoding == CURRENT_VERSION else encoding
                    self.assertEqual(data['normalization']['normalization_mode'],expected)
                    self.assertEqual(data['model_seed'],1)

    def test_versioned_paths_and_checkpoint_rejection(self):
        scene=Scenario(mode='train',normalization_config=str(self.path))
        old=replace(scene,normalization_mode=VERSION)
        self.assertNotEqual(scene.model_name('mahdrl'),old.model_name('mahdrl'))
        from algorithms.irws_paper.runner import output_dir
        self.assertNotEqual(output_dir(scene,'ckpts'),output_dir(old,'ckpts'))
        with self.assertRaises(ValueError):check_checkpoint(old.normalization_metadata(),scene.normalization_metadata())
        check_checkpoint(old.normalization_metadata(),old.normalization_metadata())
        meta=scene.normalization_metadata()
        for key in ('io_ratio_encoding','io_ratio_ref','scale_config_version'):
            bad=dict(meta);bad[key]='wrong'
            with self.assertRaises(ValueError):check_checkpoint(bad,meta)

    def test_bounded_per_field_quantiles_and_nonfinite(self):
        for capacity in (0,-1,1.5):
            with self.assertRaises(ValueError):NumericalDiagnostics(capacity)
        diag=NumericalDiagnostics(capacity=4)
        diag.observe('input',np.array([[1.,10.],[2.,20.],[3.,30.],[4.,40.],[5.,50.]]))
        stats=diag.snapshot()['profiles']['input']
        self.assertEqual(stats['rows_seen'],5);self.assertEqual(stats['window_rows'],4)
        self.assertEqual(stats['P50'],[3.5,35.]);self.assertEqual(stats['max_abs'],[5.,50.])
        diag.observe('input',[[np.nan,np.inf]])
        stats=diag.snapshot()['profiles']['input'];self.assertEqual(stats['nonfinite'],[1,1])
        json.dumps(diag.snapshot(),allow_nan=False)
        with self.assertRaises(ValueError):diag.observe('input',[[1,2,3]])

    def test_gradients_observation_and_clipping(self):
        diag=NumericalDiagnostics();p=torch.nn.Parameter(torch.zeros(2));p.grad=torch.tensor([3.,4.])
        before=p.grad.clone();self.assertEqual(diag.gradients([p],2.),5.)
        torch.testing.assert_close(before,p.grad,rtol=0,atol=0)
        self.assertEqual(diag.snapshot()['clipping']['gradient']['clipped_ratio'],1.)
        torch.nn.utils.clip_grad_norm_([p],2.)
        self.assertAlmostEqual(float(p.grad.norm()),2.,places=5)
        p.grad[0]=float('nan')
        with self.assertRaises(FloatingPointError):diag.gradients([p])

    def test_hooks_preserve_rng_outputs_and_weights(self):
        module=torch.nn.Linear(2,3);x=torch.tensor([[1.,2.]])
        before=module(x).detach().clone();weights={k:v.clone() for k,v in module.state_dict().items()}
        py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state().clone()
        diag=NumericalDiagnostics();diag.inputs(module);after=module(x)
        self.assertTrue(torch.equal(before,after));self.assertEqual(py,random.getstate())
        np.testing.assert_array_equal(npstate[1],np.random.get_state()[1]);self.assertTrue(torch.equal(ts,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,module.state_dict()[k]) for k,v in weights.items()))
        self.assertIn('input',diag.snapshot()['profiles'])

    def test_dqn_updates_identical_with_diagnostics_disabled(self):
        for module in ('common.d3qn_agent_EXDehid','common.d3qn_agent_PER'):
            cls=importlib.import_module(module).D3QNAgent;outcomes=[]
            for enabled in ('0','1'):
                with patch.dict(os.environ,F19_DIAGNOSTICS=enabled):
                    random.seed(3);np.random.seed(3);torch.manual_seed(3)
                    agent=cls(2,2,batch_size=2,hidden_dims=(8,4),device=torch.device('cpu'))
                    for i in range(3):agent.remember(np.array([i,1.]),np.ones(2),0,1.,np.array([i+1.,1.]),np.ones(2),False)
                    agent.update();outcomes.append({k:v.clone() for k,v in agent.online.state_dict().items()})
                    stats=agent.f11.numerics.snapshot()
                    if enabled=='1':
                        self.assertIn('online_input',stats['profiles']);self.assertIn('gradient',stats['clipping'])
                        self.assertIn('Q_selected',stats['profiles']);self.assertIn('TD_target',stats['profiles'])
                    else:self.assertFalse(stats['enabled'])
            self.assertTrue(all(torch.equal(v,outcomes[1][k]) for k,v in outcomes[0].items()))

    def test_actor_critic_updates_identical_and_profiles(self):
        from algorithms.marl.agents import A2CAgent,A2CConfig,LocalSharedActorIndependentCriticA2C
        from algorithms.irws.vm_agent import PPOVMAgent,PPOConfig
        for kind in ('global','local','ppo'):
            outcomes=[]
            for enabled in ('0','1'):
                with patch.dict(os.environ,F19_DIAGNOSTICS=enabled):
                    random.seed(4);np.random.seed(4);torch.manual_seed(4)
                    cfg=A2CConfig(hidden_dims=(8,4),device='cpu')
                    if kind=='global':agent=A2CAgent(2,2,cfg)
                    elif kind=='local':agent=LocalSharedActorIndependentCriticA2C(2,2,2,cfg)
                    else:agent=PPOVMAgent(PPOConfig(2,2,update_epochs=1,minibatch_size=2),hidden=(8,4))
                    obs=np.array([1.,2.],dtype=np.float32);mask=np.ones(2)
                    if kind=='ppo':
                        for done in (False,True):
                            action,info=agent.select_action(obs,mask)
                            agent.remember(obs,mask,action,info['logp'],info['value'],1.,done)
                        agent.update()
                    else:
                        extra=dict(host_index=0,next_host_index=1) if kind=='local' else {}
                        agent.update_one_step(obs,mask,0,1.,obs+1,mask,False,**extra)
                    outcomes.append({k:v.clone() for k,v in agent.net.state_dict().items()})
                    if enabled=='1':
                        stats=agent.f11.numerics.snapshot()
                        for name in ('input','value','return'):self.assertIn(name,stats['profiles'])
                        self.assertIn('gradient',stats['clipping'])
            self.assertTrue(all(torch.equal(v,outcomes[1][k]) for k,v in outcomes[0].items()))

    def test_paper_networks_and_meta_gradients_logged(self):
        from common.test_irws_paper import synthetic_env,fixture
        from algorithms.irws_paper.config import PaperConfig
        from algorithms.irws_paper.learner import PaperLearner
        from algorithms.irws_paper.runner import run_episode
        path,scene=fixture(self.root)
        outcomes=[]
        for enabled in ('0','1'):
            with patch.dict(os.environ,F19_DIAGNOSTICS=enabled),synthetic_env(path) as env:
                torch.manual_seed(1);learner=PaperLearner(3,PaperConfig())
                result=run_episode(env,learner,32,True)
                outcomes.append({name:{k:v.clone() for k,v in getattr(learner,name).state_dict().items()} for name in ('actor','critic','task','intrinsic')})
                if enabled=='1':
                    profiles=result['numerics']['profiles'];clipping=result['numerics']['clipping']
                    for key in ('actor_input','critic_input','intrinsic_input','selector_embedding_input','task_rank_input','advantage','return'):
                        self.assertIn(key,profiles)
                    for name in ('actor','critic','intrinsic','task'):self.assertIn(name+'_gradient',clipping)
        self.assertTrue(all(torch.equal(v,outcomes[1][name][k]) for name,state in outcomes[0].items() for k,v in state.items()))

    def test_fixed_trace_physical_equivalence(self):
        old=self.env('mahdrl_fcfs',VERSION);new=self.env('mahdrl_fcfs',CURRENT_VERSION);steps=0
        while not old.done_flag:
            self.assertEqual(old.ready_task_ids,new.ready_task_ids)
            gid=min(old.ready_task_ids,key=lambda i:(old.task_ready_time[i],i))
            action=int(np.flatnonzero(old._vm_idle_mask())[0])
            for env in (old,new):
                env._assign_task_to_specific_vm(gid,action);env._advance_until_decision_energy_only()
            self.assertEqual(old.task_end_time,new.task_end_time)
            self.assertEqual(old.total_energy,new.total_energy);steps+=1
        self.assertTrue(new.done_flag);self.assertGreater(steps,0)


if __name__=='__main__':unittest.main()
