"""IRWS-R v2 batch gradients, boundary ordering, physics and artifact isolation."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.irws_r import runner
from algorithms.irws_r.config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, IMPLEMENTATION, BEST_CHECKPOINT, FINAL_CHECKPOINT, PaperConfig
from algorithms.irws_r.learner import batch_meta_jacobian_scales
from algorithms.irws_r.networks import TopologicalLayer
from algorithms.irws_restored.config import PaperConfig as SourceConfig, MODEL_VERSION as SOURCE_VERSION
from algorithms.irws_restored import runner as source_runner
from common.test_irws_paper import fixture, synthetic_env


class IRWSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='irws_r_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.scene = fixture(self.root)
        self.scene=replace(self.scene,paper_options=dict(precision='float64'))

    def test_source_algorithms_and_learning_settings_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'algorithms/irws_r/copied_from_sha256.json').read_text())
        for file, digest in manifest.items():
            if file == 'common/scenarios.py':
                # Algorithm registration may grow; shared physical settings must stay fixed.
                from common.scenarios import Scenario
                self.assertEqual(Scenario().vms_per_host, (9, 8, 8))
                self.assertEqual(Scenario(resource='M').vms_per_host, (9, 9, 8, 8, 8, 8))
                self.assertEqual(Scenario(resource='L').vms_per_host, (9, 9, 9, 8, 8, 8, 8, 8, 8))
                self.assertEqual(Scenario().deadline_probs, (.8, .2))
                self.assertEqual(Scenario().protocol().train, tuple(range(600)))
                self.assertEqual(Scenario().validation_seeds(), [600])
                self.assertEqual(Scenario().seeds, tuple(range(700, 730)))
                continue
            raw = (root/file).read_bytes()
            lf = raw.replace(b'\r\n', b'\n')
            self.assertIn(digest, [hashlib.sha256(data).hexdigest()
                                  for data in (raw, lf, lf.replace(b'\n', b'\r\n'))], file)
        snapshot = root/'algorithms/irws_r_v1'
        for file, digest in json.loads((snapshot/'snapshot_sha256.json').read_text()).items():
            raw=(snapshot/file).read_bytes()
            lf=raw.replace(b'\r\n',b'\n')
            self.assertIn(digest,[hashlib.sha256(b).hexdigest() for b in (raw,lf,lf.replace(b'\n',b'\r\n'))],file)
        self.assertEqual((root/'algorithms/irws_r_v2/environment.py').read_bytes(),
                         (root/'algorithms/irws_restored/environment.py').read_bytes())
        current=PaperConfig().as_dict()
        for key in ('precision','rollout_size'):current.pop(key)
        self.assertEqual(current,SourceConfig().as_dict())

    def test_output_names_and_model_seeds_are_separate(self):
        for kind in ('ckpts', 'run'):
            current = runner.output_dir(self.scene, kind)
            self.assertIn(ALGORITHM, current.parts)
            self.assertIn(MODEL_VERSION, current.parts)
            self.assertEqual(current.name, 'model1')
            self.assertNotEqual(current, source_runner.output_dir(self.scene, kind))
            from algorithms.irws_r_v1.runner import output_dir as v1_output
            self.assertNotEqual(current,v1_output(self.scene,kind))
            self.assertNotEqual(current, runner.output_dir(replace(self.scene, model_seed=2), kind))
        for kind in ('train', 'eval', 'summary'):
            self.assertIn('_irws_r_', runner.artifact_name(self.scene, kind))
        self.assertIn(ALGORITHM, BEST_CHECKPOINT)
        self.assertIn(ALGORITHM, FINAL_CHECKPOINT)

    def test_new_schedule_is_repeatable_and_preserves_common_physics(self):
        with synthetic_env(self.path, runner.PaperCloudEnv) as current, synthetic_env(
                self.path, runner.PaperCloudEnv) as previous:
            runner.seed_everything(31)
            learner = runner.PaperLearner(current.num_vms, PaperConfig(precision='float64'))
            runner.seed_everything(31)
            old_learner = runner.PaperLearner(previous.num_vms, PaperConfig(precision='float64'))
            actual = runner.run_episode(current, learner, 32, False)
            expected = runner.run_episode(previous, old_learner, 32, False)
            self.assertEqual(actual['metrics'], expected['metrics'])
            self.assertEqual(current.task_placement, previous.task_placement)
            np.testing.assert_array_equal(current.task_end_time, previous.task_end_time)
            from common.workflow_opt import energy_from_records
            self.assertAlmostEqual(current.total_energy,energy_from_records(current.records,current.hosts),places=8)
            for gid,(vm,_) in current.task_placement.items():
                base=gid-current.task_meta[gid].local_id
                for parent in current._task_obj_from_gid(gid).parents:
                    self.assertGreaterEqual(current.task_start_time[gid]+1e-8,current.task_end_time[base+parent])
            for vm in range(current.num_vms):
                gids=sorted([g for g,p in current.task_placement.items() if p[0]==vm],key=lambda g:current.task_start_time[g])
                self.assertTrue(all(current.task_end_time[a]<=current.task_start_time[b]+1e-8 for a,b in zip(gids,gids[1:])))

    def test_graph_self_branch_has_effect_and_gradient(self):
        torch.manual_seed(53)
        layer=TopologicalLayer(5).double()
        states=torch.randn(3,5,dtype=torch.float64)
        neighbors=((),(0,),(0,1))
        output=layer(states,neighbors,[0,1,2])
        output.square().sum().backward()
        self.assertGreater(float(layer.self_weight.weight.grad.abs().max()),1e-10)
        self.assertGreater(float(layer.neighbor_weight.weight.grad.abs().max()),1e-10)
        with torch.no_grad():
            layer.self_weight.weight[0,0].add_(2.)
            changed=layer(states,neighbors,[0,1,2])
        self.assertTrue(torch.isfinite(changed).all())
        self.assertGreater(float((changed-output.detach()).abs().max()),1e-8)

    def test_meta_contractions_keep_sample_factors_paired(self):
        cfg=PaperConfig(precision='float64')
        pre=(torch.tensor([[1.,2.],[-3.,4.]],dtype=torch.float64),)
        cost=(torch.tensor([.3,-.7],dtype=torch.float64),)
        reward=(torch.tensor([.9,.2],dtype=torch.float64),)
        weights=torch.tensor([1.7,.4],dtype=torch.float64)
        phi_scale,eta_scale=batch_meta_jacobian_scales(pre,cost,reward,weights,cfg)
        phi=torch.tensor([.2,-.5],dtype=torch.float64,requires_grad=True)
        eta=torch.tensor([.7,.9],dtype=torch.float64,requires_grad=True)
        def intrinsic(p,e):
            return torch.stack((p[0]*e[0]+2*p[1],p[1]*e[1]+3*p[0]))
        values=intrinsic(phi,eta)
        phi_gradient=torch.autograd.grad((phi_scale*values).mean(),phi,retain_graph=True)[0]
        eta_gradient=torch.autograd.grad(-(eta_scale*values).mean(),eta)[0]
        epsilon=1e-5
        for parameter,gradient,scale,sign in ((phi,phi_gradient,phi_scale,1.),(eta,eta_gradient,eta_scale,-1.)):
            for i in range(2):
                shift=torch.zeros(2,dtype=torch.float64);shift[i]=epsilon
                p_plus,p_minus=(phi+shift,phi-shift) if parameter is phi else (phi,phi)
                e_plus,e_minus=(eta+shift,eta-shift) if parameter is eta else (eta,eta)
                numeric=sign*((scale*intrinsic(p_plus,e_plus)).mean()-(scale*intrinsic(p_minus,e_minus)).mean())/(2*epsilon)
                self.assertAlmostEqual(float(gradient[i]),float(numeric),places=12)
        naive=cfg.intrinsic_lr*weights.mean()*(pre[0].mean(0)*cost[0]).sum()*intrinsic(phi,eta).mean()
        naive_gradient=torch.autograd.grad(naive,phi)[0]
        self.assertFalse(torch.allclose(phi_gradient,naive_gradient,atol=1e-9,rtol=1e-7))

    def test_boundary_estimation_is_read_only_and_uses_expected_values(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            env.reset(seed=32)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            before_rng=torch.get_rng_state().clone()
            ready=list(env.ready_task_ids)
            boundary=learner.boundary_snapshot(env)
            self.assertIsNone(env.current_task_gid)
            self.assertEqual(ready,env.ready_task_ids)
            self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
            self.assertGreater(len(boundary.decisions),1)
            self.assertAlmostEqual(float(boundary.probabilities.sum()),1.)
            expected=sum(float(p)*learner.value(d) for p,d in zip(boundary.probabilities,boundary.decisions))
            self.assertAlmostEqual(float(boundary.value),float(expected),places=12)

    def _collect_interval(self,env,learner):
        env.reset(seed=32);learner.reset_history()
        decision=learner.prepare(env)
        while True:
            interval=env.interval_index
            action,logp,value=learner.act(decision)
            reward,_,info=env.paper_step(action)
            if not env.can_dispatch():env.finish_phase_and_advance()
            if env.done_flag:
                learner.remember(decision,action,logp,value,reward,True,info,None)
                return 0.
            if env.interval_index!=interval or len(learner.buffer)+1>=learner.cfg.rollout_size:
                boundary=learner.boundary_snapshot(env)
                learner.remember(decision,action,logp,value,reward,False,info,None,boundary=boundary)
                return boundary.value
            successor=learner.prepare(env)
            learner.remember(decision,action,logp,value,reward,False,info,successor)
            decision=successor

    def test_replay_preserves_chronology_and_inner_outer_encoding(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            bootstrap=self._collect_interval(env,learner)
            stored=learner.tensor(np.stack([r['decision'].obs for r in learner.buffer])).clone()
            old_probabilities=[r['logp'] for r in learner.buffer]
            _,intrinsics=learner._replay()
            torch.testing.assert_close(intrinsics.detach(),learner.tensor([r['behavior_intrinsic'] for r in learner.buffer]))
            with torch.no_grad():learner.task.embedding.pool.bias[0].add_(.5)
            replay=learner._replay;outer=learner._outer_scores
            encodings=[];actor_inputs=[]
            hook=learner.actor.register_forward_pre_hook(lambda _m,args:actor_inputs.append(args[0].detach().clone()))
            def observe_replay():
                observations,rewards=replay();encodings.append(observations.detach().clone())
                self.assertFalse(torch.equal(observations.detach(),stored))
                return observations,rewards
            def observe_outer(observations,*args):
                torch.testing.assert_close(observations,encodings[-1],rtol=0,atol=0)
                self.assertEqual([r['logp'] for r in learner.buffer],old_probabilities)
                return outer(observations,*args)
            try:
                with patch.object(learner,'_replay',side_effect=observe_replay),patch.object(learner,'_outer_scores',side_effect=observe_outer):
                    result=learner.update(bootstrap)
            finally:hook.remove()
            self.assertEqual(len(encodings),1)
            torch.testing.assert_close(actor_inputs[0],encodings[0][result['sampled_indices']])
            torch.testing.assert_close(actor_inputs[1],encodings[0])

    def test_batch_steps_and_selection_follow_interval_updates(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            fixed=[p.detach().clone() for p in learner.task.score.parameters()]
            prepare=learner.prepare;update=learner.update
            events=[]
            def observe_prepare(*a,**kw):
                result=prepare(*a,**kw);events.append(('select',learner.updates));return result
            def observe_update(*a,**kw):
                self.assertIsNone(env.current_task_gid)
                result=update(*a,**kw)
                events.append(('terminal' if result['terminal'] else 'update',learner.updates))
                return result
            with patch.object(learner,'prepare',side_effect=observe_prepare),patch.object(learner,'update',side_effect=observe_update):
                result=runner.run_episode(env,learner,32,True)
            self.assertLess(learner.updates,result['assignments'])
            self.assertEqual(learner.updates,len(result['updates']))
            for i,event in enumerate(events):
                if event[0]=='update':self.assertEqual(events[i+1],('select',event[1]))
            for name in ('actor','critic','intrinsic','task'):
                steps={int(v['step']) for v in getattr(learner,name+'_opt').state.values()}
                self.assertEqual(steps,{learner.updates})
            for old,p in zip(fixed,learner.task.score.parameters()):torch.testing.assert_close(old,p,rtol=0,atol=0)
            self.assertIsNone(learner._pending_boundary)
            self.assertFalse(learner.buffer)
            self.assertTrue(all(u['terminal'] or u['boundary_candidates']>0 for u in result['updates']))

    def test_short_batch_and_sampling_cap_are_recorded(self):
        for cap in (1,64):
            with self.subTest(cap=cap),synthetic_env(self.path,runner.PaperCloudEnv) as env:
                learner=runner.PaperLearner(env.num_vms,replace(PaperConfig(precision='float64'),batch_size=cap,rollout_size=cap))
                bootstrap=self._collect_interval(env,learner)
                count=len(learner.buffer)
                result=learner.update(bootstrap)
                expected=min(cap,count)
                self.assertEqual(len(result['sampled_indices']),expected)
                self.assertEqual(len(set(result['sampled_indices'])),expected)
                self.assertEqual(learner.batch_size,expected)
                self.assertEqual(result['optimizer_steps_per_network'],1)

    def test_actor_takes_one_mean_loss_adam_step(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            bootstrap=self._collect_interval(env,learner)
            observations,_=learner._replay()
            reference=deepcopy(learner.actor)
            optimizer=torch.optim.Adam(reference.parameters(),lr=learner.cfg.actor_lr)
            records=list(learner.buffer)
            from algorithms.irws_r.networks import masked_distribution
            from algorithms.irws_r.learner import clipped_loss
            mask=np.stack([r['decision'].mask for r in records])
            actions=torch.tensor([r['action'] for r in records])
            logs=masked_distribution(reference(observations.detach()),mask).log_prob(actions)
            loss=clipped_loss(logs,learner.tensor([r['logp'] for r in records]),
                              learner.tensor(learner._advantages(bootstrap)),learner.cfg.clip)
            loss.backward();optimizer.step()
            learner.update(bootstrap)
            for actual,expected in zip(learner.actor.parameters(),reference.parameters()):
                torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)

    def test_boundary_reward_expectation_and_real_history_are_distinct(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            bootstrap=self._collect_interval(env,learner)
            record=learner.buffer[-1];boundary=record['boundary']
            self.assertIsNotNone(boundary)
            self.assertFalse(record['done'])
            self.assertIsNone(record['next_decision'])
            with torch.no_grad():
                expected=0.
                for probability,decision in zip(boundary.probabilities,boundary.decisions):
                    token=learner.history_token(learner.tensor(record['decision'].obs),record['action'],
                                                record['step_reward'],False,learner.tensor(decision.obs))
                    intrinsic,_=learner.intrinsic(token,record['history_before'])
                    expected+=float(probability)*float(intrinsic)
            self.assertAlmostEqual(record['behavior_intrinsic'],expected,places=12)
            learner.update(bootstrap)
            successor=learner.prepare(env)
            with torch.no_grad():
                token=learner.history_token(learner.tensor(record['decision'].obs),record['action'],
                                            record['step_reward'],False,learner.tensor(successor.obs))
                _,expected_hidden=learner.intrinsic(token,record['history_before'])
            learner.resolve_boundary(successor)
            torch.testing.assert_close(learner.hidden,expected_hidden,rtol=0,atol=0)
            self.assertIsNone(learner._pending_boundary)

    def test_large_interval_samples_64_and_replays_truncated_history(self):
        import networkx as nx
        from common.read_xml_opt_Tsize import Task
        graph=nx.empty_graph(80,create_using=nx.DiGraph)
        tasks=[Task(task_id=i,workload_mi=float(8+i),parents=[],children=[]) for i in range(80)]
        make=lambda **kw:runner.PaperCloudEnv(queue_capacity=100,**kw)
        with synthetic_env(self.path,make) as env,patch('environments.irws.load_workflow_from_dax',
                side_effect=lambda *a,**kw:(graph.copy(),deepcopy(tasks))):
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            bootstrap=self._collect_interval(env,learner)
            self.assertEqual(len(learner.buffer),64)
            _,intrinsics=learner._replay()
            torch.testing.assert_close(intrinsics.detach(),learner.tensor([r['behavior_intrinsic'] for r in learner.buffer]))
            result=learner.update(bootstrap)
            self.assertEqual(len(set(result['sampled_indices'])),64)
            self.assertEqual(result['sample_presentations'],64)
            self.assertEqual(result['collected_transitions'],64)
            self.assertEqual(learner.updates,1)

    def test_checkpoint_resume_matches_uninterrupted_next_episode(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig(precision='float64'))
            runner.run_episode(env,learner,32,True)
            path=self.root/'resume.pth'
            runner.save_checkpoint(path,learner,self.scene,0,100.,runner.env_kwargs(self.scene,32))
            uninterrupted=runner.run_episode(env,learner,33,True)
            resumed,_=runner.load_checkpoint(path,self.scene,restore_training=True)
            continued=runner.run_episode(env,resumed,33,True)
            self.assertEqual(uninterrupted['metrics'],continued['metrics'])
            self.assertEqual(uninterrupted['updates'],continued['updates'])
            for name in ('actor','critic','intrinsic','task'):
                for actual,expected in zip(getattr(learner,name).parameters(),getattr(resumed,name).parameters()):
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_existing_training_or_evaluation_output_is_preserved(self):
        existing = self.root/'existing'
        existing.mkdir()
        sentinel = existing/'old_result.txt'
        sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            runner.train(self.scene, out=existing)
        out_root = self.root/'run'
        target = out_root/(self.scene.task+self.scene.resource+'_'+self.scene.deadline)
        target.mkdir(parents=True)
        old_csv = target/runner.artifact_name(self.scene, 'eval')
        old_csv.write_text('old csv', encoding='utf-8')
        with patch.object(runner, 'parse_scenario', return_value=self.scene), patch.object(
                runner, 'output_dir', side_effect=lambda scene, kind:out_root if kind=='run' else self.root/'ckpts'), patch.object(
                runner, 'load_checkpoint') as load:
            with self.assertRaisesRegex(FileExistsError, 'IRWS-R evaluation artifacts'):
                runner.evaluate_main()
            load.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')
        self.assertEqual(old_csv.read_text(encoding='utf-8'), 'old csv')

    def test_train_load_evaluate_and_resume_use_irws_r_identity(self):
        with synthetic_env(self.path, runner.PaperCloudEnv) as template:
            def output(scene, kind):
                return self.root/'verification'/kind/f'model{scene.model_seed}'
            with patch.object(runner, 'PaperCloudEnv', side_effect=lambda **kw:deepcopy(template)), patch.object(
                    runner, 'output_dir', side_effect=output), redirect_stdout(io.StringIO()):
                runner.seed_everything(23)
                _, models = runner.train(self.scene, kwargs=runner.env_kwargs(self.scene,32), out=self.root/'models')
                _, obj = runner.load_checkpoint(models/BEST_CHECKPOINT, self.scene)
                self.assertEqual(obj['algorithm'], ALGORITHM)
                self.assertEqual(obj['display_name'], DISPLAY_NAME)
                self.assertEqual(obj['model_version'], MODEL_VERSION)
                self.assertEqual(obj['implementation'],IMPLEMENTATION)
                self.assertTrue((models/FINAL_CHECKPOINT).is_file())
                training = json.loads((models/runner.artifact_name(self.scene,'train')).read_text().splitlines()[0])
                self.assertEqual(training['algorithm'], ALGORITHM)
                measurement = json.loads((models/'logs/training_measurements.jsonl').read_text().splitlines()[0])
                self.assertEqual(measurement['algorithm'], ALGORITHM)
                self.assertEqual(set(measurement['optimizer_steps'].values()),{len(training['updates'])})
                self.assertEqual(set(measurement['optimizer_sample_presentations'].values()),{sum(u['sample_presentations'] for u in training['updates'])})
                for version in ('irws_r_v2_batch','irws_r_v1',SOURCE_VERSION, 'irws_restored_queue_v1', 'irws_cloud_energy_tardiness_v2'):
                    torch.save(dict(obj,model_version=version), self.root/'old.pth')
                    with self.assertRaisesRegex(ValueError,'algorithm/version mismatch'):
                        runner.load_checkpoint(self.root/'old.pth',self.scene)
                with self.assertRaisesRegex(ValueError,'model seed mismatch'):
                    runner.load_checkpoint(models/BEST_CHECKPOINT,replace(self.scene,model_seed=2))
                evaluation = replace(self.scene,mode='eval',seed_spec='2',checkpoint_override=str(models))
                with patch.object(runner,'parse_scenario',return_value=evaluation):
                    runner.evaluate_main()
                    with self.assertRaises(FileExistsError):runner.evaluate_main()
                result_dir = output(evaluation,'run')/'SS_Tight'
                with (result_dir/runner.artifact_name(evaluation,'eval')).open(encoding='utf-8',newline='') as f:
                    row = next(csv.DictReader(f))
                self.assertEqual(row['algorithm'],ALGORITHM)
                self.assertEqual(row['display_name'],DISPLAY_NAME)
                self.assertEqual(row['model_version'],MODEL_VERSION)
                summary = json.loads((result_dir/runner.artifact_name(evaluation,'summary')).read_text())
                self.assertEqual(summary['algorithm'],ALGORITHM)
                self.assertEqual(Path(summary['source_checkpoint']).name,BEST_CHECKPOINT)
                details = json.loads((result_dir/'measurements/episodes/seed2.json').read_text())
                self.assertEqual(details['algorithm'],ALGORITHM)
                self.assertTrue(details['checkpoints'])
                self.assertEqual(Path(details['checkpoints'][0]['path']).name,BEST_CHECKPOINT)
                with self.assertRaises(FileExistsError):runner.train(self.scene,out=models)
                runner.train(self.scene,out=models,resume=models/FINAL_CHECKPOINT)


if __name__ == '__main__':
    unittest.main()
