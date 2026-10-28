"""Periodic idle-only scheduling: no reservation, correct events and V3 rollouts."""
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from algorithms.irws_r_fe import runner as baseline
from algorithms.irws_r_fe_r090_idle import runner as r090_idle
from algorithms.irws_r_fe_idle import runner, selection
from algorithms.irws_r_fe_idle.config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION,
    SCHEDULING_PROTOCOL, COMMON_REWARD, BEST_CHECKPOINT, FINAL_CHECKPOINT, PaperConfig)
from algorithms.irws_r_fe_idle.scenario import FEScenario, parse_scenario
from algorithms.irws_r_fe_idle.networks import masked_distribution
from common.experiment_protocol import protocol_binding
from common.read_xml_opt_Tsize import Task
from common.test_irws_paper import fixture, synthetic_env
from common.workflow_opt import energy_from_records


class SmallScenario(FEScenario):
    @property
    def vms_per_host(self):return (2,1)


class IdleIRWSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='irws_idle_test_')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.path,scene=fixture(self.root)
        protocol=Path(scene.experiment_protocol)
        data=json.loads(protocol.read_text());data.update(train=[32,33],episodes=2)
        protocol.write_text(json.dumps(data))
        calibration=json.loads(self.path.read_text())
        calibration['generation']=dict(**protocol_binding(protocol),calibration_seeds={'SS_T':[32,33]})
        self.path.write_text(json.dumps(calibration))
        self.scene=SmallScenario(**dict(vars(scene),episodes=2))

    @contextmanager
    def environment(self, count=None, duration=12.3, edges=(), vms=(2,1), workflows=2, arrivals=None):
        def make(**kwargs):
            return runner.PaperCloudEnv(**dict(kwargs,**COMMON_REWARD,num_hosts=len(vms),vms_per_host=vms,
                workflows_per_episode=workflows,pc_tiers=(1.,),slot_duration=5.))
        with synthetic_env(self.path,make) as env,ExitStack() as stack:
            if count is not None:
                graph=nx.empty_graph(count,create_using=nx.DiGraph);graph.add_edges_from(edges)
                tasks=[Task(task_id=i,workload_mi=duration,parents=list(graph.predecessors(i)),
                            children=list(graph.successors(i))) for i in range(count)]
                stack.enter_context(patch('environments.irws.load_workflow_from_dax',
                    side_effect=lambda *a,**kw:(graph.copy(),deepcopy(tasks))))
            if arrivals is not None:
                stack.enter_context(patch('environments.irws.poisson_arrival_times',return_value=np.array(arrivals)))
            yield env

    def assert_parameters_equal(self,a,b):
        for name in ('task','actor','critic','intrinsic'):
            for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters()):
                torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_busy_mask_illegal_action_is_atomic_and_queue_feature_stays_zero(self):
        with self.environment(count=4,workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32)
            learner=runner.PaperLearner(3,PaperConfig())
            before=learner.prepare(env)
            np.testing.assert_array_equal(before.mask,[1,1,1])
            env.paper_step(0)
            after=learner.prepare(env)
            np.testing.assert_array_equal(after.mask,[0,1,1])
            probability=masked_distribution(torch.tensor([1e6,0.,0.]),after.mask.copy()).probs
            self.assertEqual(float(probability[0]),0.)
            np.testing.assert_array_equal(env.resource_features(after.gid)[1::6],np.zeros(3))
            self.assertGreater(env.resource_features(after.gid)[2],0.)
            state=(env.state_revision,env.total_energy,env.task_state.copy(),env.task_placement.copy(),
                   env.ready_task_ids.copy(),env.vm_running.copy(),len(env.records),env.current_task_gid)
            with self.assertRaisesRegex(ValueError,'requires an idle VM'):env.paper_step(0)
            self.assertEqual(state,(env.state_revision,env.total_energy,env.task_state.copy(),env.task_placement.copy(),
                env.ready_task_ids.copy(),env.vm_running.copy(),len(env.records),env.current_task_gid))
            self.assertFalse(any(env.vm_queues.values()))

    def test_all_busy_waits_in_ready_set_until_next_tick_not_completion_event(self):
        with self.environment(count=2,vms=(1,),workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32);env.select_task(0)
            _,_,info=env.paper_step(0)
            self.assertAlmostEqual(info['end'],12.3)
            np.testing.assert_array_equal(env.queue_mask(),[0])
            self.assertFalse(env.can_dispatch());self.assertEqual(env.task_state[1],'Ready')
            self.assertNotIn(1,env.task_placement)
            env.finish_phase_and_advance()
            self.assertEqual(env.current_time,15.)
            self.assertEqual(env.task_end_time[0],12.3)
            self.assertEqual(env.task_state[0],'Finished')
            self.assertTrue(env.can_dispatch());env.select_task(1)
            _,_,info=env.paper_step(0)
            self.assertEqual(info['start'],15.);self.assertFalse(info['queued'])
            env.finish_phase_and_advance()
            self.assertTrue(env.done_flag)
            self.assertAlmostEqual(env.current_time,27.3)
            self.assertAlmostEqual(env.total_energy,energy_from_records(env.records,env.hosts),places=8)

    def test_simultaneous_completions_release_children_and_all_idle_masks(self):
        with self.environment(count=4,duration=10.,edges=((0,3),(1,3)),workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32)
            for gid,vm in enumerate((0,1,2)):
                env.select_task(gid);env.paper_step(vm)
            self.assertFalse(env.can_dispatch())
            env.finish_phase_and_advance()
            self.assertEqual(env.current_time,10.)
            self.assertEqual(env.ready_task_ids,[3])
            np.testing.assert_array_equal(env.queue_mask(),[1,1,1])
            self.assertEqual(env.task_state[:3],['Finished']*3)
            env.select_task(3);env.paper_step(0);env.finish_phase_and_advance()
            self.assertTrue(env.done_flag)

    def test_zero_duration_releases_at_same_tick_and_finalizes_without_future_event(self):
        with self.environment(count=2,duration=0.,edges=((0,1),),vms=(1,),workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32);env.select_task(0);env.paper_step(0)
            self.assertEqual(env.current_time,0.)
            self.assertEqual(env.task_state,['Finished','Ready'])
            np.testing.assert_array_equal(env.queue_mask(),[1])
            env.select_task(1);env.paper_step(0);env.finish_phase_and_advance()
            self.assertTrue(env.done_flag);self.assertEqual(env.current_time,0.)
            self.assertFalse(any(env.vm_queues.values()))

    def test_future_arrivals_when_idle_and_empty_ready_use_the_next_tick(self):
        with self.environment(count=1,duration=1.,vms=(1,),workflows=2,arrivals=[0.,12.3]) as env:
            env.reset(seed=32);env.select_task(0);env.paper_step(0);env.finish_phase_and_advance()
            self.assertEqual(env.current_time,15.)
            self.assertEqual(env.task_state,['Finished','Ready'])
            self.assertTrue(env.can_dispatch())
            env.select_task(1);env.paper_step(0);env.finish_phase_and_advance()
            self.assertTrue(env.done_flag)

    def test_forbidden_vm_queues_and_positive_layout_capacity(self):
        with self.environment(count=2,workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32);env.vm_queues[0].append(1)
            with self.assertRaisesRegex(RuntimeError,'cannot contain'):env.queue_mask()
            with self.assertRaisesRegex(RuntimeError,'cannot start'):env._finish_and_start()
        with self.assertRaisesRegex(ValueError,'positive integer'):runner.PaperCloudEnv(queue_capacity=0)

    def test_v3_rollouts_64_64_32_only_real_idle_assignments_and_terminal_tail(self):
        with self.environment(count=80,duration=1.) as env:
            runner.seed_everything(23);learner=runner.PaperLearner(3,PaperConfig())
            recorded=[];assignments=[];update=learner.update;step=env.paper_step
            def audit_step(action):
                self.assertIsNone(env.vm_running[action])
                self.assertTrue(env.queue_mask()[action])
                self.assertFalse(any(env.vm_queues.values()))
                self.assertAlmostEqual(env.current_time/5.,round(env.current_time/5.))
                value=step(action);assignments.append(value[2])
                return value
            def audit_update(*args,**kwargs):
                recorded.extend(learner.buffer)
                return update(*args,**kwargs)
            with patch.object(env,'paper_step',side_effect=audit_step),patch.object(learner,'update',side_effect=audit_update):
                result=runner.run_episode(env,learner,32,True)
            self.assertEqual([u['collected_transitions'] for u in result['updates']],[64,64,32])
            self.assertEqual(len(assignments),160);self.assertEqual(learner.steps,160)
            self.assertEqual(len({r['decision'].gid for r in recorded}),160)
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),160)
            self.assertTrue(result['updates'][-1]['terminal'])
            self.assertTrue(any(u['policy_intervals']>1 for u in result['updates']))
            self.assertFalse(learner.buffer);self.assertIsNone(learner._pending_boundary)
            for r in recorded:
                self.assertTrue(r['decision'].mask[r['action']])
                self.assertFalse(r['info']['queued'])
                np.testing.assert_array_equal(r['decision'].resources.cpu().numpy()[1::6],np.zeros(3))
                if r['boundary'] is not None:
                    self.assertFalse(r['done'])
                    self.assertTrue(all(d.mask.any() for d in r['boundary'].decisions))
            for vm in range(3):
                gids=sorted((g for g,p in env.task_placement.items() if p[0]==vm),key=lambda g:env.task_start_time[g])
                self.assertTrue(all(env.task_end_time[a]<=env.task_start_time[b]+1e-12 for a,b in zip(gids,gids[1:])))
            self.assertAlmostEqual(env.total_energy,energy_from_records(env.records,env.hosts),places=8)

    def test_base_learning_reward_and_core_files_remain_unchanged(self):
        root=Path(__file__).resolve().parents[1]/'algorithms'
        for name in ('learner.py','networks.py','numerics.py'):
            self.assertEqual((root/'irws_r_fe'/name).read_bytes(),(root/'irws_r_fe_idle'/name).read_bytes())
        self.assertEqual(PaperConfig().as_dict(),baseline.PaperConfig().as_dict())
        self.assertEqual(COMMON_REWARD,baseline.COMMON_REWARD)
        self.assertEqual(COMMON_REWARD['alpha_delay_vm'],.75)
        self.assertNotEqual(COMMON_REWARD,r090_idle.COMMON_REWARD)
        expected=dict(baseline.selection_metadata(),version='select_sr092_minenergy_v1',workflow_success_threshold=.92)
        self.assertEqual(selection.selection_metadata(),expected)
        extra=dict(runner.IMPLEMENTATION);self.assertEqual(extra.pop('scheduling'),SCHEDULING_PROTOCOL)
        self.assertEqual(extra,baseline.IMPLEMENTATION)
        self.assertNotEqual(getattr(baseline.PaperCloudEnv,'scheduling_protocol',None),SCHEDULING_PROTOCOL)

    def test_sr092_boundary_aggregate_counts_and_energy_priority(self):
        def row(success,count=1000,energy=100.):
            return dict(wf_count=count,wf_success_count=success,wf_success_rate=success/count,total_energy=energy)
        choose=lambda success,energy:selection.validation_selection([row(success,energy=energy)])
        self.assertFalse(choose(919,1.)['feasible'])
        self.assertTrue(choose(920,100.)['feasible'])
        self.assertTrue(choose(921,100.)['feasible'])
        # Aggregate success counts: 1/1 and 91/99 together meet 92/100.
        aggregate=selection.validation_selection([row(1,1),row(91,99)])
        self.assertTrue(aggregate['feasible']);self.assertEqual(aggregate['workflow_success_rate'],.92)
        key=selection.selection_key
        self.assertLess(key(choose(920,100.)),key(choose(919,1.)))
        self.assertLess(key(choose(920,10.)),key(choose(1000,11.)))
        self.assertLess(key(choose(921,10.)),key(choose(920,10.)))
        self.assertLess(key(choose(919,100.)),key(choose(918,1.)))
        self.assertLess(key(choose(919,10.)),key(choose(919,11.)))
        self.assertEqual(key(choose(920,10.)),key(choose(920,10.)))

    def test_sr092_protocol_and_output_paths_are_isolated_from_sr099(self):
        current=(selection.selection_protocol(self.scene),runner.output_dir(self.scene,'ckpts'),
                 runner.artifact_name(replace(self.scene,mode='eval',seed_spec='2'),'eval'))
        with patch.object(selection,'SELECTION_TAG','select_sr099_minenergy_v1'),patch.object(
                selection,'WORKFLOW_SUCCESS_THRESHOLD',.99):
            previous=(selection.selection_protocol(self.scene),runner.output_dir(self.scene,'ckpts'),
                      runner.artifact_name(replace(self.scene,mode='eval',seed_spec='2'),'eval'))
        for new,old in zip(current,previous):self.assertNotEqual(new,old)

    def test_actual_dispatch_reward_keeps_original_fe_weights(self):
        with self.environment(count=1,vms=(1,),workflows=1,arrivals=[0.]) as env:
            env.reset(seed=32);env.select_task(0)
            reward,_,info=env.paper_step(0)
            self.assertEqual(env.alpha_delay_vm,.75)
            self.assertGreater(info['delta_energy_total'],0.)
            expected=.75*info['r_delay']-.25*.001*info['delta_energy_total']
            r090=.9*info['r_delay']-.1*.001*info['delta_energy_total']
            self.assertAlmostEqual(reward,expected)
            self.assertNotAlmostEqual(reward,r090)
            self.assertEqual(info['outer_cost'],-reward)

    def test_train_resume_eval_isolated_paths_and_old_checkpoints_rejected(self):
        with self.environment() as template,patch.object(runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),redirect_stdout(io.StringIO()):
            kwargs=runner.env_kwargs(self.scene,32)
            full,_=runner.train(self.scene,kwargs=kwargs,out=self.root/'full')
            _,models=runner.train(replace(self.scene,episodes=1),kwargs=kwargs,out=self.root/'split')
            resumed,resume_dir=runner.train(self.scene,kwargs=kwargs,resume=models/FINAL_CHECKPOINT)
            self.assertEqual(resume_dir,models)
            self.assert_parameters_equal(full,resumed)
            _,obj=runner.load_checkpoint(models/BEST_CHECKPOINT,self.scene)
            self.assertEqual(obj['scheduling_protocol'],SCHEDULING_PROTOCOL)
            self.assertEqual(obj['display_name'],DISPLAY_NAME)
            self.assertEqual(obj['reward_settings'],COMMON_REWARD)
            self.assertEqual(obj['checkpoint_selection']['workflow_success_threshold'],.92)
            wrong=self.root/'wrong.pth'
            with patch.object(selection,'SELECTION_TAG','select_sr099_minenergy_v1'),patch.object(
                    selection,'WORKFLOW_SUCCESS_THRESHOLD',.99):
                old=dict(obj,checkpoint_selection=selection.selection_metadata(),
                         protocol=selection.selection_protocol(self.scene),normalization=self.scene.normalization_metadata())
            torch.save(old,wrong)
            with self.assertRaisesRegex(ValueError,'protocol mismatch'):
                runner.load_checkpoint(wrong,self.scene)
            torch.save(dict(obj,scheduling_protocol='queue_periodic'),wrong)
            with self.assertRaisesRegex(ValueError,'scheduling protocol mismatch'):runner.load_checkpoint(wrong,self.scene)
            torch.save(dict(obj,algorithm=baseline.ALGORITHM,model_version=baseline.MODEL_VERSION),wrong)
            with self.assertRaisesRegex(ValueError,'algorithm/version mismatch'):runner.load_checkpoint(wrong,self.scene)
            with self.assertRaises(ValueError):baseline.load_checkpoint(models/BEST_CHECKPOINT,self.scene)
            with self.assertRaisesRegex(ValueError,'algorithm/version mismatch'):
                r090_idle.load_checkpoint(models/BEST_CHECKPOINT,self.scene)
            torch.save(dict(obj,reward_settings=dict(COMMON_REWARD,alpha_delay_vm=.9)),wrong)
            with self.assertRaisesRegex(ValueError,'reward objective mismatch'):
                runner.load_checkpoint(wrong,self.scene)
            with self.assertRaises(FileExistsError):runner.train(self.scene,out=models)
            evaluation=replace(self.scene,mode='eval',seed_spec='2',checkpoint_override=str(models))
            with patch.object(runner,'parse_scenario',return_value=evaluation),patch.object(runner,'output_dir',return_value=self.root/'eval'):
                runner.evaluate_main()
                with self.assertRaises(FileExistsError):runner.evaluate_main()
            report=json.loads((self.root/'eval'/runner.artifact_name(evaluation,'summary')).read_text())
            self.assertEqual(report['scheduling_protocol'],SCHEDULING_PROTOCOL)
            self.assertEqual(report['checkpoint_selection'],selection.selection_metadata())
            csv_path=self.root/'eval'/runner.artifact_name(evaluation,'eval')
            self.assertTrue(csv_path.is_file())
            self.assertEqual(Path(report['evaluation_csv']),csv_path)
            details=runner.evaluation_directory(evaluation)
            self.assertTrue((self.root/'eval'/'_details'/details.name/'measurements/episodes/seed2.json').is_file())

    def test_validation_is_read_only_and_same_source_model_covers_27_cli_targets(self):
        with self.environment() as template,patch.object(runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)):
            learner=runner.PaperLearner(3,PaperConfig());runner.run_episode(template,learner,32,True)
            hidden=learner.hidden.clone();before=(learner.steps,learner.updates)
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            caches=learner._embedding_cache
            params=deepcopy(learner.actor.state_dict())
            runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
            self.assertEqual(before,(learner.steps,learner.updates));self.assertIs(caches,learner._embedding_cache)
            torch.testing.assert_close(hidden,learner.hidden,rtol=0,atol=0)
            self.assertEqual(rng[0],random.getstate());np.testing.assert_equal(rng[1],np.random.get_state())
            torch.testing.assert_close(rng[2],torch.get_rng_state(),rtol=0,atol=0)
            for k,v in params.items():torch.testing.assert_close(v,learner.actor.state_dict()[k],rtol=0,atol=0)
        norm=dict(normalization_version='f19_shared_v1',normalization_config_hash='test_hash')
        sources=set();targets=set()
        with patch.object(FEScenario,'normalization_metadata',return_value=norm):
            for resource in 'SML':
                for deadline in ('Tight','Medium','Loose'):
                    args=['--resource',resource,'--deadline',deadline,'--dry-run']
                    output=io.StringIO()
                    with redirect_stdout(output),self.assertRaises(SystemExit):parse_scenario('train',argv=args)
                    source=json.loads(output.getvalue());sources.add(source['checkpoint_dir'])
                    self.assertEqual(source['scheduling_protocol'],SCHEDULING_PROTOCOL)
                    self.assertEqual(source['reward_settings']['alpha_delay_vm'],.75)
                    self.assertFalse(source['queue']['waiting_enabled'])
                    self.assertEqual(source['queue']['slot_seconds'],5.)
                    for task in 'SML':
                        output=io.StringIO()
                        with redirect_stdout(output),self.assertRaises(SystemExit):
                            parse_scenario('eval',argv=args+['--task',task,'--train-task','S',
                                '--train-resource',resource,'--train-deadline',deadline])
                        target=json.loads(output.getvalue())
                        self.assertEqual(source['checkpoint_dir'],target['checkpoint_dir'])
                        targets.add(target['evaluation_csv'])
        self.assertEqual(len(sources),9);self.assertEqual(len(targets),27)
        for kind in ('run','ckpts'):
            self.assertNotEqual(runner.output_dir(self.scene,kind),baseline.output_dir(self.scene,kind))
            self.assertNotEqual(runner.output_dir(self.scene,kind),r090_idle.output_dir(self.scene,kind))

    def test_short_training_paths_and_27_flat_downloadable_csv_names(self):
        files=[];summaries=[]
        for resource in 'SML':
            for deadline in ('Tight','Medium','Loose'):
                source=FEScenario(mode='train',resource=resource,deadline=deadline)
                checkpoint=runner.output_dir(source,'ckpts')
                self.assertLessEqual(len(checkpoint.name),30)
                self.assertLessEqual(len(str(checkpoint.relative_to(runner.ROOT))),80)
                for task in 'SML':
                    target=replace(source,mode='eval',task=task,train_task='S',train_resource=resource,train_deadline=deadline)
                    self.assertEqual(checkpoint,runner.output_dir(target,'ckpts'))
                    csv=runner.output_dir(target,'run')/runner.artifact_name(target,'eval')
                    self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_',csv.name)
                    self.assertIn('_m1_s700-729_c',csv.name)
                    self.assertEqual(csv.parent,runner.ROOT/'outputs/run'/ALGORITHM)
                    details=runner.evaluation_directory(target)
                    self.assertEqual(details.parent.name,'_details');self.assertEqual(len(details.name),16)
                    self.assertLess(len(str(details/'measurements/episodes/seed700.json')),260)
                    files.append(csv)
                    summaries.append(runner.output_dir(target,'run')/runner.artifact_name(target,'summary'))
        self.assertEqual(len({p.name for p in files}),27)
        self.assertEqual(len({p.parent for p in files}),1)
        self.assertEqual(len({p.name for p in summaries}),27)

    def test_configuration_source_model_and_exact_seeds_do_not_collide(self):
        target=replace(self.scene,mode='eval',seed_spec='2')
        variants=[target,replace(target,model_seed=2),replace(target,train_task='M'),
                  replace(target,paper_options={'slot_duration':1.}),
                  replace(target,paper_options={'rollout_size':32}),
                  replace(target,paper_options={'precision':'float64'})]
        self.assertEqual(len({runner.artifact_name(s,'eval') for s in variants}),len(variants))
        self.assertEqual(len({runner.output_dir(s,'ckpts') for s in variants}),len(variants))
        # Equal seed counts and ranges with gaps must have distinct downloaded names.
        scene=FEScenario(mode='eval')
        seeds=[scene,replace(scene,seed_spec='700-728'),replace(scene,seed_spec='701-729'),
               replace(scene,seed_spec='700,702,704'),replace(scene,seed_spec='700,703,704')]
        self.assertEqual(len({runner.artifact_name(s,'eval') for s in seeds}),len(seeds))
        moved=self.root/'moved.json';moved.write_bytes(self.path.read_bytes())
        self.assertEqual(runner.run_name(target),runner.run_name(replace(target,normalization_config=str(moved))))
        moved.write_text('{}')
        self.assertNotEqual(runner.run_name(target),runner.run_name(replace(target,normalization_config=str(moved))))

    def test_existing_flat_csv_blocks_evaluation_before_loading_checkpoint(self):
        target=replace(self.scene,mode='eval',seed_spec='2')
        root=self.root/'existing_results';root.mkdir()
        csv=root/runner.artifact_name(target,'eval');csv.write_text('keep existing result',encoding='utf-8')
        with patch.object(runner,'parse_scenario',return_value=target),patch.object(
                runner,'output_dir',return_value=root),patch.object(runner,'load_checkpoint') as load:
            with self.assertRaises(FileExistsError):runner.evaluate_main()
            load.assert_not_called()
        self.assertEqual(csv.read_text(encoding='utf-8'),'keep existing result')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_idle_training_and_validation(self):
        with self.environment() as template,patch.object(runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)):
            learner=runner.PaperLearner(3,PaperConfig(device='cuda'))
            result=runner.run_episode(template,learner,32,True)
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),12)
            rows,_=runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
            self.assertEqual(rows[0]['wf_count'],2)
            self.assertFalse(any(template.vm_queues.values()))
            self.assertTrue(all(torch.isfinite(p).all() for p in learner.actor.parameters()))


if __name__=='__main__':unittest.main()
