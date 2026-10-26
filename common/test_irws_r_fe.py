"""IRWS-R-FE selection, isolated validation, real training and resume behavior."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.irws_r import runner as original
from algorithms.irws_r.config import PaperConfig as OriginalConfig
from algorithms.irws_r_fe import runner, selection
from algorithms.irws_r_fe.config import (ALGORITHM, MODEL_VERSION, BEST_CHECKPOINT,
                                         FINAL_CHECKPOINT, PaperConfig)
from algorithms.irws_r_fe.scenario import FEScenario, parse_scenario
from algorithms.pd3qn_dag_fe import selection as reference_selection
from common.experiment_protocol import protocol_binding
from common.test_irws_paper import fixture, synthetic_env


def row(success=50, energy=100., count=50, seed=632):
    return dict(seed=seed, wf_count=count, wf_success_count=success,
                wf_success_rate=success/count, total_energy=energy)


class SmallScenario(FEScenario):
    @property
    def vms_per_host(self):return (2,1)


class IRWSRFETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='irws_r_fe_test_')
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.path,scene=fixture(self.root)
        protocol_path=Path(scene.experiment_protocol)
        protocol=json.loads(protocol_path.read_text())
        protocol.update(train=list(range(32,36)),episodes=4)
        protocol_path.write_text(json.dumps(protocol))
        calibration=json.loads(self.path.read_text())
        calibration['generation']=dict(**protocol_binding(protocol_path),calibration_seeds={'SS_T':list(range(32,36))})
        self.path.write_text(json.dumps(calibration))
        self.scene=SmallScenario(**dict(vars(scene),episodes=4))

    def assert_tree_equal(self,a,b):
        if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.assert_tree_equal(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_tree_equal(x,y)
        else:self.assertEqual(a,b)

    def test_single_aggregate_threshold_and_pd3qn_equivalence(self):
        cases=[[row(49)],[row(50)],[row(99,count=100)],
               [row(50),row(49)], [row(1,count=1),row(98,count=100)]]
        self.assertEqual(selection.selection_metadata(),reference_selection.selection_metadata())
        for rows in cases:
            actual=selection.validation_selection(rows)
            self.assertEqual(actual,reference_selection.validation_selection(rows))
            self.assertEqual(selection.selection_key(actual),reference_selection.selection_key(actual))
        self.assertFalse(selection.validation_selection(cases[0])['feasible'])
        self.assertTrue(selection.validation_selection(cases[1])['feasible'])
        self.assertTrue(selection.validation_selection(cases[2])['feasible'])
        self.assertTrue(selection.validation_selection(cases[3])['feasible'])
        self.assertAlmostEqual(selection.validation_selection(cases[4])['workflow_success_rate'],99/101)

    def test_feasible_priority_fallback_energy_and_ties(self):
        choose=lambda success,energy:selection.validation_selection([row(success,energy,count=100)])
        key=selection.selection_key
        self.assertLess(key(choose(99,1000)),key(choose(98,1)))
        self.assertLess(key(choose(99,10)),key(choose(100,11)))
        self.assertLess(key(choose(100,10)),key(choose(99,10)))
        self.assertLess(key(choose(98,100)),key(choose(97,1)))
        self.assertLess(key(choose(98,10)),key(choose(98,11)))
        self.assertEqual(key(choose(99,10)),key(choose(99,10)))
        for bad in ([],[dict(row(),wf_count=0)],[dict(row(),wf_success_count=51)],
                    [dict(row(),wf_success_rate=.99)],[dict(row(),total_energy=float('nan'))]):
            with self.assertRaises(ValueError):selection.validation_selection(bad)

    def test_v3_core_and_paper_learning_settings_are_unchanged(self):
        self.assertEqual(PaperConfig().as_dict(),OriginalConfig().as_dict())
        root=Path(__file__).resolve().parents[1]/'algorithms'
        for name in ('learner.py','networks.py','environment.py','numerics.py'):
            self.assertEqual((root/'irws_r_fe'/name).read_bytes(),(root/'irws_r'/name).read_bytes())
        self.assertEqual(runner.IMPLEMENTATION,original.IMPLEMENTATION)
        self.assertEqual(runner.COMMON_REWARD,original.COMMON_REWARD)
        self.assertEqual(runner.queue_settings(self.scene),original.queue_settings(self.scene))

    def test_learning_and_physics_match_v3_for_same_episode(self):
        results=[];states=[]
        for module in (original,runner):
            with synthetic_env(self.path,module.PaperCloudEnv) as env:
                module.seed_everything(23)
                learner=module.PaperLearner(env.num_vms,PaperConfig())
                results.append(module.run_episode(env,learner,32,True))
                states.append({name:getattr(learner,name).state_dict() for name in ('task','actor','critic','intrinsic')})
        self.assertEqual(results[0]['metrics'],results[1]['metrics'])
        self.assertEqual(results[0]['updates'],results[1]['updates'])
        self.assert_tree_equal(states[0],states[1])

    def test_validation_restores_rng_modes_history_caches_diagnostics_and_weights(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as template:
            runner.seed_everything(23)
            learner=runner.PaperLearner(3,PaperConfig())
            runner.run_episode(template,learner,32,True)
            learner.actor.eval()
            before={name:deepcopy(getattr(learner,name).state_dict()) for name in ('task','actor','critic','intrinsic')}
            opts={name:deepcopy(getattr(learner,name+'_opt').state_dict()) for name in before}
            hidden=learner.hidden.clone();counts=(learner.steps,learner.updates)
            diagnostics=deepcopy(learner.numerics.snapshot())
            caches=(learner._embedding_cache,learner._resource_tensors,learner.task.embedding._graphs)
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            with patch.object(runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)):
                rows,_=runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
                self.assertEqual(rows[0]['wf_count'],2)
                self.assertIn('wf_success_count',rows[0])
                with self.assertRaisesRegex(RuntimeError,'callback failed'):
                    runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32),
                                    on_result=lambda _:(_ for _ in ()).throw(RuntimeError('callback failed')))
            self.assertFalse(learner.actor.training);self.assertTrue(learner.critic.training)
            self.assert_tree_equal(rng,(random.getstate(),np.random.get_state(),torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
            self.assert_tree_equal(hidden,learner.hidden)
            self.assertEqual(counts,(learner.steps,learner.updates))
            self.assertEqual(diagnostics,learner.numerics.snapshot())
            for a,b in zip(caches,(learner._embedding_cache,learner._resource_tensors,learner.task.embedding._graphs)):
                self.assertIs(a,b)
            for name in before:
                self.assert_tree_equal(before[name],getattr(learner,name).state_dict())
                self.assert_tree_equal(opts[name],getattr(learner,name+'_opt').state_dict())

    def test_evaluation_partition_and_metrics_disabled(self):
        learner=runner.PaperLearner(3,PaperConfig())
        for seeds,split in (([], 'validation'),([632,632],'validation'),([2],'validation'),
                            ([632],'test'),([32],'test'),([632],'unknown')):
            with self.assertRaises(ValueError):runner.evaluate(learner,self.scene,seeds,split)
        with synthetic_env(self.path,runner.PaperCloudEnv) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),patch.dict(
                'os.environ',{'EXPERIMENT_METRICS':'0'}):
            rows,_=runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
            self.assertEqual(selection.validation_selection(rows)['workflow_count'],2)

    def test_paths_metadata_and_source_model_remain_isolated(self):
        for kind in ('ckpts','run'):
            path=runner.output_dir(self.scene,kind)
            self.assertIn(ALGORITHM,path.parts);self.assertIn(MODEL_VERSION,path.parts)
            self.assertIn(selection.SELECTION_TAG,str(path))
            self.assertNotEqual(path,original.output_dir(self.scene,kind))
            self.assertNotEqual(path,runner.output_dir(replace(self.scene,model_seed=2),kind))
        target=replace(self.scene,mode='eval',task='L',seed_spec='2')
        self.assertEqual(runner.output_dir(self.scene,'ckpts'),runner.output_dir(target,'ckpts'))
        metadata=self.scene.normalization_metadata()
        self.assertEqual(metadata['checkpoint_selection'],selection.selection_metadata())
        self.assertEqual(metadata['experiment_protocol']['selection'],selection.SELECTION_TAG)
        self.assertEqual(self.scene.protocol().selection,'minimum_validation_total_energy')
        self.assertEqual(metadata['experiment_protocol_hash'],protocol_binding(self.scene.experiment_protocol)['experiment_protocol_hash'])

    def test_cli_nine_sources_and_27_targets_match_source_checkpoints(self):
        destinations=set();sources=set()
        # Dry-run path tests need no production calibration or cache writes.
        norm=dict(normalization_version='f19_shared_v1',normalization_config_hash='test_calibration_hash')
        with patch.object(FEScenario,'normalization_metadata',return_value=norm):
            for resource in 'SML':
                for deadline in ('Tight','Medium','Loose'):
                    args=['--resource',resource,'--deadline',deadline,'--dry-run']
                    output=io.StringIO()
                    with redirect_stdout(output),self.assertRaises(SystemExit) as stopped:
                        parse_scenario('train',argv=args)
                    self.assertEqual(stopped.exception.code,0)
                    source=json.loads(output.getvalue());sources.add(source['checkpoint_dir'])
                    self.assertEqual(source['checkpoint_selection'],selection.selection_metadata())
                    self.assertEqual(source['learning'],json.loads(json.dumps(PaperConfig().as_dict())))
                    for task in 'SML':
                        output=io.StringIO()
                        with redirect_stdout(output),self.assertRaises(SystemExit):
                            parse_scenario('eval',argv=args+['--task',task,'--train-task','S',
                                '--train-resource',resource,'--train-deadline',deadline])
                        target=json.loads(output.getvalue())
                        self.assertEqual(source['checkpoint_dir'],target['checkpoint_dir'])
                        self.assertEqual(target['seeds'],list(range(700,730)))
                        destinations.add(target['evaluation_directory'])
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaises(SystemExit):
                parse_scenario('eval',argv=['--dry-run','--checkpoint-dir',str(self.root/'explicit')])
            self.assertEqual(json.loads(output.getvalue())['checkpoint_dir'],str((self.root/'explicit').resolve()))
        self.assertEqual(len(sources),9);self.assertEqual(len(destinations),27)

    def test_real_train_save_load_evaluate_and_fallback_summary(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as template:
            def output(scene,kind):return self.root/'outputs'/kind
            with patch.object(runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),patch.object(
                    runner,'output_dir',side_effect=output),redirect_stdout(io.StringIO()):
                scene=replace(self.scene,episodes=2)
                _,models=runner.train(scene,kwargs=runner.env_kwargs(scene,32),out=self.root/'models')
                _,best=runner.load_checkpoint(models/BEST_CHECKPOINT,scene)
                _,final=runner.load_checkpoint(models/FINAL_CHECKPOINT,scene,restore_training=True)
                records=[json.loads(s) for s in (models/'validation_selection.jsonl').read_text().splitlines()]
                expected=min(records,key=lambda r:selection.selection_key(r['validation_selection']))
                self.assertEqual(best['episode'],expected['episode'])
                self.assertEqual(final['best_validation_selection'],best['best_validation_selection'])
                summary=json.loads((models/'training_summary.json').read_text())
                self.assertEqual(summary['feasible_model_found'],best['validation_selection']['feasible'])
                measured=[json.loads(s) for s in (models/'logs/training_measurements.jsonl').read_text().splitlines()]
                self.assertEqual(len(measured),2)
                self.assertTrue(all(r['validation'] for r in measured))
                evaluation=replace(scene,mode='eval',seed_spec='2',checkpoint_override=str(models))
                with patch.object(runner,'parse_scenario',return_value=evaluation):
                    runner.evaluate_main()
                    with self.assertRaises(FileExistsError):runner.evaluate_main()
                result_dir=output(evaluation,'run')/'SS_Tight'
                with (result_dir/runner.artifact_name(evaluation,'eval')).open(newline='',encoding='utf-8') as f:
                    result=next(csv.DictReader(f))
                self.assertEqual(result['algorithm'],ALGORITHM)
                report=json.loads((result_dir/runner.artifact_name(evaluation,'summary')).read_text())
                self.assertEqual(report['selected_checkpoint_episode'],best['episode'])
                self.assertEqual(report['checkpoint_selection'],selection.selection_metadata())
                with self.assertRaises(FileExistsError):runner.train(scene,out=models)

    def test_resume_keeps_historical_feasible_best_and_matches_uninterrupted(self):
        candidates=[row(1,1,count=2),row(2,100,count=2),row(1,.5,count=2),row(2,100,count=2)]
        def validations(items):
            iterator=iter(items)
            def validate(*args,**kwargs):
                candidate=next(iterator)
                return [candidate],{'total_energy':{'mean':candidate['total_energy'],'std':0.}}
            return validate
        with synthetic_env(self.path,runner.PaperCloudEnv) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),redirect_stdout(io.StringIO()):
            kwargs=runner.env_kwargs(self.scene,32)
            with patch.object(runner,'evaluate',side_effect=validations(candidates)):
                whole,_=runner.train(self.scene,kwargs=kwargs,out=self.root/'whole')
            with patch.object(runner,'evaluate',side_effect=validations(candidates[:2])):
                runner.train(replace(self.scene,episodes=2),kwargs=kwargs,out=self.root/'split')
            with patch.object(runner,'evaluate',side_effect=validations(candidates[2:])):
                continued,_=runner.train(self.scene,kwargs=kwargs,out=self.root/'split',resume=self.root/'split'/FINAL_CHECKPOINT)
            for name in ('task','actor','critic','intrinsic'):
                self.assert_tree_equal(getattr(whole,name).state_dict(),getattr(continued,name).state_dict())
            _,best=runner.load_checkpoint(self.root/'split'/BEST_CHECKPOINT,self.scene)
            self.assertEqual(best['episode'],1)
            self.assertTrue(best['validation_selection']['feasible'])
            records=[json.loads(s) for s in (self.root/'split'/'validation_selection.jsonl').read_text().splitlines()]
            self.assertEqual([r['selected_as_best'] for r in records],[True,True,False,False])

    def test_checkpoint_rejects_old_identity_wrong_rule_role_and_metrics(self):
        learner=runner.PaperLearner(3,PaperConfig())
        rows=[row(49)];candidate=selection.validation_selection(rows)
        best=dict(candidate,episode=0,completed_episodes=1)
        path=self.root/BEST_CHECKPOINT
        runner.save_checkpoint(path,learner,self.scene,0,candidate,best,rows,runner.env_kwargs(self.scene,32),'best')
        _,obj=runner.load_checkpoint(path,self.scene)
        for field,value in (('algorithm','irws_r'),('model_version','irws_r_v3_rollout'),
                            ('checkpoint_selection',{}),('role','final'),('model_seed',2),
                            ('validation_selection',dict(candidate,feasible=True)),
                            ('validation',[dict(rows[0],seed=2)])):
            torch.save(dict(obj,**{field:value}),self.root/'bad.pth')
            with self.assertRaises(ValueError):runner.load_checkpoint(self.root/'bad.pth',self.scene)
        with self.assertRaises(ValueError):runner.load_checkpoint(path,replace(self.scene,paper_options={'slot_duration':1.}))
        with self.assertRaises(ValueError):original.load_checkpoint(path,self.scene)
        with self.assertRaisesRegex(ValueError,'historical best'):
            (self.root/'missing').mkdir()
            runner.train(self.scene,out=self.root/'missing',resume=self.root/'missing'/FINAL_CHECKPOINT)

    def test_all_infeasible_is_labelled_and_measurement_toggle_preserves_training(self):
        scene=replace(self.scene,episodes=2)
        states=[];bests=[]
        with synthetic_env(self.path,runner.PaperCloudEnv) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),redirect_stdout(io.StringIO()):
            for enabled in ('1','0'):
                rows=[row(1,10,count=2)]
                with patch.dict('os.environ',{'EXPERIMENT_METRICS':enabled}),patch.object(
                        runner,'evaluate',return_value=(rows,{'total_energy':{'mean':10.,'std':0.}})):
                    learner,models=runner.train(scene,kwargs=runner.env_kwargs(scene,32),out=self.root/enabled)
                states.append({name:getattr(learner,name).state_dict() for name in ('task','actor','critic','intrinsic')})
                summary=json.loads((models/'training_summary.json').read_text())
                self.assertFalse(summary['feasible_model_found'])
                self.assertEqual(summary['best_validation_selection']['episode'],0)
                bests.append(summary['best_validation_selection'])
        self.assertEqual(bests[0],bests[1]);self.assert_tree_equal(states[0],states[1])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_training_validation_and_rng_restore(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)):
            learner=runner.PaperLearner(3,PaperConfig(device='cuda'))
            result=runner.run_episode(template,learner,32,True)
            before=torch.cuda.get_rng_state_all()
            rows,_=runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
            self.assert_tree_equal(before,torch.cuda.get_rng_state_all())
            self.assertEqual(rows[0]['wf_count'],2)
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),12)


if __name__=='__main__':unittest.main()
