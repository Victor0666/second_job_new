"""F22 split, entrypoints, real DAX identities, RNG isolation and cache regression."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import importlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.experiment_protocol import ExperimentProtocol, DEFAULT_PROTOCOL
from common.scenarios import Scenario, MODEL_NAMES, parse_scenario
from common.f22_instances import instance_identity
from common.f22_cache import inspect_cache, check_cache, supplement
from common.precompute_deadlines import main as precompute
from common.test_paired_instances import ALGORITHMS


class F22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='f22_tests_')
        cls.root = Path(cls.temp.name)
        cls.cache = cls.root/'cache.json'
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute', seed_spec='0,1,599,600,700', cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_T01_training_sequence(self):
        scene=Scenario(mode='train')
        self.assertEqual([scene.training_seed(i) for i in range(600)], list(range(600)))
        with self.assertRaises(ValueError): scene.training_seed(600)

    def test_T02_single_validation(self):
        self.assertEqual(Scenario().validation_seeds(), [600])
        with self.assertRaises(ValueError): ExperimentProtocol((0,), (600,601), (700,), 1)

    def test_T03_test_defaults_and_explicit_override(self):
        self.assertEqual(Scenario().seeds, tuple(range(700,730)))
        self.assertEqual(Scenario(seed_spec='800,801').seeds, (800,801))
        with self.assertRaises(ValueError): Scenario(seed_spec='599,700')
        with self.assertRaises(ValueError): Scenario(seed_spec='600')
        self.assertEqual(Scenario(legacy_split=True,seed_spec='2-31').seeds,tuple(range(2,32)))

    def test_T04_disjoint(self):
        p=ExperimentProtocol.load(DEFAULT_PROTOCOL)
        for a,b in ((p.train,p.validation),(p.train,p.test),(p.validation,p.test)):
            self.assertFalse(set(a)&set(b))
        with self.assertRaises(ValueError): ExperimentProtocol((0,1),(1,),(700,),2)

    def test_T05_cli_four_algorithms_and_old_irws(self):
        for algorithm in ('mahdrl','pd3qn','marl','irws','irws_paper','ertf_fifs'):
            for mode in (('eval',) if algorithm=='ertf_fifs' else ('train','eval')):
                with self.subTest(algorithm=algorithm,mode=mode), patch('sys.argv',['entry','--dry-run']), redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(SystemExit) as caught: parse_scenario(mode,algorithm)
                self.assertEqual(caught.exception.code,0)
                obj=json.loads(out.getvalue())
                self.assertEqual(obj['model_seed'],1)
                self.assertEqual(obj['experiment_protocol']['train'],list(range(600)))
                self.assertEqual(obj['experiment_protocol']['validation'],[600])
                self.assertEqual(obj['experiment_protocol']['test'],list(range(700,730)))

    def test_T06_ablation_cli(self):
        for algorithm in MODEL_NAMES:
            if not algorithm.startswith('ablation_'): continue
            for mode in ('train','eval'):
                with patch('sys.argv',['entry','--dry-run']),redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(SystemExit):parse_scenario(mode,algorithm)
                obj=json.loads(out.getvalue())
                self.assertEqual(obj['experiment_protocol']['validation'],[600])
                self.assertEqual(obj['model_seed'],1)

    def snapshot(self, algorithm, seed, model_seed=1, deadline='Tight'):
        if algorithm=='irws_paper':
            from algorithms.irws_paper.environment import PaperCloudEnv
            cls=PaperCloudEnv
        else:
            module,name=ALGORITHMS[algorithm];cls=getattr(importlib.import_module('environments.'+module),name)
        scene=Scenario(deadline=deadline,mode='train',normalization_mode='legacy',model_seed=model_seed)
        random.seed(model_seed);np.random.seed(model_seed);torch.manual_seed(model_seed)
        # Consume model random numbers before and between arrivals.
        kwargs=dict(dax_paths=scene.dax_paths(),random_seed=seed,workflows_per_episode=2,
                    horizon=1e9,arrival_lambda=.03,num_hosts=scene.num_hosts,vms_per_host=scene.vms_per_host,
                    pc_tiers=(1.,2.,4.,6.,8.),bw_tiers=(1000.,2000.,4000.,6000.,8000.),max_ready_tasks=100,
                    deadline_mode='cache_fcfs',deadline_cache_path=str(self.cache),normalization_mode='legacy')
        if algorithm=='marl':kwargs.update(alpha_deadline_choices=(2.,3.),alpha_deadline_probs=scene.deadline_probs,print_deadlines=False,dump_deadlines=False)
        elif algorithm in ('pd3qn','irws','irws_paper'):kwargs.update(deadline_alpha_choices=(2.,3.),deadline_alpha_probs=scene.deadline_probs)
        else:kwargs['deadline_alpha_small_prob']=scene.small_probability
        with redirect_stdout(io.StringIO()):
            env=cls(**kwargs);env.reset(seed=seed)
            random.random();np.random.rand(77);torch.rand(37)
            env.current_time=float(env.arrival_times[-1])
            method='_add_workflows_up_to_current_time' if algorithm in ('irws','irws_paper') else '_add_workflow_if_arrived'
            getattr(env,method)()
        return env,scene,instance_identity(env,scene,seed)

    def test_T07_actual_instances_across_algorithms(self):
        for seed,deadline in ((0,'Tight'),(600,'Medium'),(700,'Loose')):
            expected=self.snapshot('mahdrl',seed,deadline=deadline)[2]
            for name in (*ALGORITHMS,'irws_paper'):
                with self.subTest(algorithm=name,seed=seed):
                    self.assertEqual(self.snapshot(name,seed,deadline=deadline)[2],expected)

    def test_T08_different_seeds_and_actual_reset(self):
        self.assertNotEqual(self.snapshot('mahdrl',0)[2]['content_hash'],self.snapshot('mahdrl',1)[2]['content_hash'])
        for name in ('mahdrl','pd3qn','irws','irws_paper','marl'):
            env,scene,_=self.snapshot(name,0)
            with redirect_stdout(io.StringIO()):env.reset(seed=599)
            self.assertEqual(getattr(env,'episode_seed',env.random_seed),599)
            self.assertEqual(env.workflows[0].arrival_time,self.snapshot(name,599)[0].workflows[0].arrival_time)

    def test_T09_model_rng_isolation(self):
        for name in ('mahdrl','pd3qn','marl','irws','irws_paper'):
            with self.subTest(algorithm=name):
                self.assertEqual(self.snapshot(name,700,1)[2],self.snapshot(name,700,999)[2])

    def test_T10_cache_missing_incompatible_and_no_overwrite(self):
        scene=Scenario(mode='precompute')
        report,_=inspect_cache(scene,self.cache)
        self.assertIn(2,report['missing']);self.assertNotIn(600,report['missing']);self.assertNotIn(700,report['missing'])
        with self.assertRaises(ValueError):check_cache(scene,self.cache)
        with self.assertRaises(FileExistsError):supplement(scene,self.cache)
        obj=json.loads(self.cache.read_text());obj['meta']['resource']='M'
        invalid=self.root/'invalid.json';invalid.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'scenario mismatch'):inspect_cache(scene,invalid)
        # Full coverage via a tiny, explicitly disjoint fixture protocol; no other seed fallback.
        protocol=self.root/'subset.json';protocol.write_text(json.dumps(dict(version='test_subset',train=[0,1],validation=[600],test=[700],episodes=2,selection='minimum_validation_total_energy')))
        custom=replace(scene,experiment_protocol=str(protocol))
        self.assertEqual(check_cache(custom,self.cache)['status'],'ok')
        _,data=inspect_cache(custom,self.cache);data['data']=[r for r in data['data'] if r['seed']!=600]
        missing=self.root/'missing.json';missing.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'600'):check_cache(custom,missing)
        class PartialScenario(Scenario):
            @property
            def cache_path(self):return missing
        partial_scene=PartialScenario(mode='precompute',experiment_protocol=str(protocol))
        output=self.root/'supplemented.json'
        original_bytes=missing.read_bytes()
        def generate_only_missing(request):
            self.assertEqual(request.seeds,(600,))
            source=json.loads(self.cache.read_text())
            source['data']=[r for r in source['data'] if r['seed']==600]
            Path(request.cache_output).write_text(json.dumps(source))
        with patch('common.precompute_deadlines.main',side_effect=generate_only_missing):
            self.assertEqual(supplement(partial_scene,output)['status'],'ok')
        self.assertEqual(missing.read_bytes(),original_bytes)
        merged=json.loads(output.read_text())
        self.assertEqual(len({r['seed'] for r in merged['data']}),len(merged['data']))


    def test_T11_checkpoint_provenance(self):
        from types import SimpleNamespace
        from common.f18_state_normalization import bind_normalization,check_checkpoint
        scene=Scenario(mode='train',normalization_mode='legacy')
        agent=bind_normalization(SimpleNamespace(),scene)
        metadata=agent.normalization_metadata
        self.assertEqual(metadata['experiment_protocol']['version'],'f22_split_v1')
        self.assertEqual(metadata['reproducibility']['model_seed'],1)
        self.assertIn('git_commit',metadata['reproducibility'])
        with self.assertRaises(ValueError):check_checkpoint({'normalization_version':'legacy'},metadata)
        self.assertNotEqual(scene.model_name('mahdrl'),replace(scene,legacy_split=True).model_name('mahdrl'))

    def test_T12_validation_has_no_learning_calls(self):
        # Inspect actual validation function bodies, not strings in the training loop.
        import inspect
        names={'mahdrl':'evaluate_hrl_three_layer_multi_seed','pd3qn':'evaluate_pd3qn','marl':'evaluate_marl',
               'irws':'evaluate_irws_multi_seed','ablation_nohost':'evaluate_hrl_two_layer_multi_seed',
               'ablation_hostrand':'evaluate_hrl_three_layer_multi_seed','ablation_mgrrand':'evaluate_hrl_three_layer_multi_seed',
               'ablation_vmrand':'evaluate_hrl_three_layer_multi_seed_vm_random'}
        for name,function in names.items():
            module=importlib.import_module('algorithms.'+name+'.train')
            tree=ast.parse(inspect.getsource(getattr(module,function)))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('update','remember','push','backward','add_transition','store_transition'))
                    if node.func.attr in ('select_action','select') and getattr(node.func.value,'id',None) not in ('vm_selector','host_selector'):
                        keywords={k.arg:k.value for k in node.keywords}
                        self.assertIsInstance(keywords.get('deterministic'),ast.Constant)
                        self.assertIs(keywords['deterministic'].value,True)
        class Policy:
            def __init__(self, pair=False):
                self.parameter=torch.tensor([3.]);self.replay=['training sentinel'];self.calls=0;self.pair=pair
            def select_action(self, obs, mask, **kw):
                assert kw.get('deterministic') is True
                assert kw.get('count_step',False) is False
                self.calls+=1
                action=int(np.flatnonzero(mask)[0])
                return (action,0.) if self.pair else action
            def select(self,obs,mask,**kw):
                return self.select_action(obs,mask,**kw)
        for name,function in names.items():
            module=importlib.import_module('algorithms.'+name+'.train')
            env,scene,_=self.snapshot(name,600)
            kwargs=dict(dax_paths=scene.dax_paths(),random_seed=600,workflows_per_episode=2,horizon=1e9,
                        arrival_lambda=.03,num_hosts=scene.num_hosts,vms_per_host=scene.vms_per_host,
                        pc_tiers=(1.,2.,4.,6.,8.),bw_tiers=(1000.,2000.,4000.,6000.,8000.),
                        max_ready_tasks=100,deadline_mode='cache_fcfs',deadline_cache_path=str(self.cache))
            policies=[Policy(pair=name=='irws') for _ in range(3)]
            with redirect_stdout(io.StringIO()):
                fn=getattr(module,function)
                if name=='marl':
                    kwargs.update(alpha_deadline_choices=(2.,3.),alpha_deadline_probs=(.8,.2),
                                  deadlines_out_dir=str(self.root/'validation_logs'))
                    fn(kwargs,policies[0],policies[1],[600])
                else:
                    if name in ('irws','pd3qn'):kwargs.update(deadline_alpha_choices=(2.,3.),deadline_alpha_probs=(.8,.2))
                    else:kwargs['deadline_alpha_small_prob']=.8
                    if name=='irws':fn(kwargs,policies[0],policies[1],[600])
                    elif name=='mahdrl':fn(type(env),kwargs,*policies,[600])
                    elif name=='pd3qn':fn(type(env),kwargs,policies[0],[600])
                    else:fn(type(env),kwargs,policies[0],policies[1],[600])
            self.assertGreater(sum(p.calls for p in policies),0)
            for policy in policies:
                self.assertEqual(policy.replay,['training sentinel'])
                self.assertTrue(torch.equal(policy.parameter,torch.tensor([3.])))


    def test_T13_budget_and_legacy_optin(self):
        self.assertEqual(Scenario(mode='train').episodes,600)
        self.assertEqual(ExperimentProtocol.load(DEFAULT_PROTOCOL).episodes,600)
        with self.assertRaises(ValueError):Scenario(experiment_protocol=None)
        self.assertIsNone(Scenario(legacy_split=True).protocol())

    def test_T14_same_action_physical_smoke(self):
        a,_,_=self.snapshot('pd3qn',0,1);b,_,_=self.snapshot('pd3qn',0,99)
        action=int(np.flatnonzero(a.get_action_mask())[0])
        ra=a.step(action);rb=b.step(action)
        self.assertEqual(ra[1:4],rb[1:4]);self.assertEqual(a.task_end_time,b.task_end_time)
        self.assertEqual(a.total_energy,b.total_energy)


if __name__=='__main__':unittest.main()
