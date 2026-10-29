"""Execution equivalence, BN bootstrap replay, selection provenance and independent paths."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch

from algorithms.drl_gnn.environment import DRLGNNEnv
from algorithms.drl_gnn.networks import GraphState
from algorithms.drl_gnn.ppo import PPOAgent as SourceAgent
from algorithms.drl_gnn_fast.config import DRLGNNConfig, COMMON_REWARD, MODEL_VERSION
from algorithms.drl_gnn_fast.environment import FastDRLGNNEnv
from algorithms.drl_gnn_fast.networks import prepare
from algorithms.drl_gnn_fast.ppo import PPOAgent
from algorithms.drl_gnn_fast.scenario import FastScenario
from algorithms.drl_gnn_fast import runner, selection
from algorithms.mahdrl_oe_ge_apf_score_fast import selection as as_selection
from algorithms.pd3qn_dag_fe import selection as reference_selection
from common.scenarios import Scenario
from common.test_drl_gnn_memory import graph, rollout, gradients, bulk_reference, SavedStorageTracker
from common.workflow_opt import energy_from_records


def validation_row(success, energy, count=50, seed=600):
    return dict(seed=seed, wf_count=count, wf_success_count=success,
                wf_success_rate=success/count, total_energy=energy)


def plain(state):
    return GraphState(*(np.array(getattr(state,k),copy=True) for k in (
        'task_features','vm_features','dependencies','compatibility_mask','action_mask')),state.task_gids)


class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = TemporaryDirectory(prefix='drl_gnn_fast_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dax = self.root/'Fork_3.xml'
        self.dax.write_text('<adag xmlns="http://pegasus.isi.edu/schema/DAX">'
            '<job id="a"/><job id="b"/><job id="c"/>'
            '<child ref="c"><parent ref="a"/><parent ref="b"/></child></adag>', encoding='utf-8')

    def kwargs(self, **changes):
        result = dict(dax_paths=[str(self.dax)], horizon=1e9, workflows_per_episode=2, arrival_lambda=.03,
            num_hosts=2, vms_per_host=(2,1), pc_tiers=(1.,8.), bw_tiers=(1000.,2000.),
            deadline_mode='none', max_ready_tasks='auto', normalize=False, random_seed=0,
            **Scenario(mode='train').normalization_kwargs(), **COMMON_REWARD)
        result.update(changes)
        return result

    def assert_state(self, a, b):
        for key in ('task_features','vm_features','dependencies','compatibility_mask','action_mask'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key),err_msg=key)
        self.assertEqual(a.task_gids,b.task_gids)

    def test_real_dax_cached_observations_and_physics_are_identical(self):
        dax = Path(__file__).resolve().parents[1]/'data/dax/Montage_25.xml'
        kw = self.kwargs(dax_paths=[str(dax)])
        a,b = DRLGNNEnv(**kw),FastDRLGNNEnv(**kw)
        old_snapshot = None
        rewards = 0.
        for seed in (0,1):
            a.reset(seed=seed);b.reset(seed=seed)
            while not a.done_flag:
                x,y=a.graph_state(),b.graph_state();self.assert_state(x,y)
                if old_snapshot is None:
                    old_snapshot=deepcopy(y)
                    saved=y.task_features.copy()
                action=(a.ready_task_ids[0]+1)*a.num_vms+a.assignment_count%a.num_vms
                self.assertEqual(a.pair_step(action),b.pair_step(action))
                self.assertEqual(a.task_start_time,b.task_start_time)
                self.assertEqual(a.task_end_time,b.task_end_time)
                self.assertEqual(a.total_energy,b.total_energy)
            self.assert_state(a.graph_state(),b.graph_state())
            self.assertAlmostEqual(b.total_energy,energy_from_records(b.records,b.hosts,
                start_time=b.first_arrival_time,end_time=b.current_time),places=7)
        np.testing.assert_array_equal(old_snapshot.task_features,saved)

    def test_owned_snapshots_validate_and_tensor_cache_is_bounded(self):
        state=graph(1);prepared=prepare(state);before=prepared.task_features.copy()
        state.task_features[:]=0.
        np.testing.assert_array_equal(before,prepared.task_features)
        with self.assertRaises(ValueError):prepared.task_features[0,0]=2.
        with self.assertRaises(ValueError):replace(prepared,action_mask=np.full(prepared.action_mask.shape,np.nan))
        with self.assertRaises(ValueError):replace(prepared,dependencies=np.array([[0,999]]))
        agent=PPOAgent()
        for i in range(20):agent.network._tensors(prepare(graph(i)))
        self.assertLessEqual(len(agent.network._tensor_cache),8)
        agent.network.clear_cache();self.assertFalse(agent.network._tensor_cache)

    def test_original_initialization_logits_values_and_actor_only(self):
        torch.manual_seed(17);a=SourceAgent()
        torch.manual_seed(17);b=PPOAgent()
        for key,value in a.network.state_dict().items():
            torch.testing.assert_close(value,b.network.state_dict()[key],rtol=0,atol=0)
        state=prepare(graph(3))
        with torch.no_grad():
            al,av=a.network(plain(state));bl,bv=b.network(state)
        torch.testing.assert_close(al,bl,rtol=0,atol=0)
        torch.testing.assert_close(av,bv,rtol=0,atol=0)
        before=deepcopy(b.network.critic.state_dict())
        self.assertEqual(a.act(plain(state),True)[0],b.select_action(state))
        for key,value in before.items():
            torch.testing.assert_close(value,b.network.critic.state_dict()[key],rtol=0,atol=0)

    def check_bootstrap(self,device):
        torch.manual_seed(73);a=SourceAgent(DRLGNNConfig(device=device))
        torch.manual_seed(73);b=PPOAgent(DRLGNNConfig(device=device))
        states=[prepare(graph(i)) for i in range(4)]
        for cycle in range(4):
            for slot in range(4):
                rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None
                expected=a.act(plain(states[slot]))
                torch.set_rng_state(rng)
                if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                actual=b.act(states[slot],slot=slot)
                self.assertEqual(expected[0],actual[0])
                np.testing.assert_allclose(expected[1:],actual[1:],rtol=2e-5,atol=3e-6)
                state=prepare(graph(slot+cycle+1));states[slot]=state
                self.assertAlmostEqual(a.value(plain(state)),b.bootstrap_value(state,slot),places=5)
        self.assertEqual(b.bootstrap_hits,12)
        for key,value in a.network.state_dict().items():
            torch.testing.assert_close(value,b.network.state_dict()[key],rtol=2e-5,atol=3e-6)
        hits=b.bootstrap_hits;b.updates+=1
        b.act(states[0],slot=0);self.assertEqual(b.bootstrap_hits,hits)
        b.bootstrap_value(states[0],0)
        b.act(prepare(graph(99)),slot=0);self.assertEqual(b.bootstrap_hits,hits)
        b.bootstrap_value(states[0],0)
        b.network.load_state_dict(b.network.state_dict())
        b.act(states[0],slot=0);self.assertEqual(b.bootstrap_hits,hits)

    def test_bootstrap_reuse_replays_bn_across_four_envs_and_invalidates(self):
        self.check_bootstrap('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_bootstrap_reuse_and_bn(self):
        self.check_bootstrap('cuda')

    def test_grouped_backward_gradients_bn_metrics_and_step_counts(self):
        for count,cfg in ((512,DRLGNNConfig()),(9,DRLGNNConfig(clip_vloss=False,norm_adv=False)),
                          (1,DRLGNNConfig()),(9,DRLGNNConfig(target_kl=1e-12))):
            with self.subTest(count=count,clip=cfg.clip_vloss):
                torch.manual_seed(31);a=SourceAgent(cfg);records=rollout(a,count)
                b=PPOAgent(cfg);b.network.load_state_dict(a.network.state_dict())
                np.random.seed(77)
                with patch.object(a.optimizer,'step'):
                    expected,wanted=bulk_reference(a,records)
                captured=[];clip=torch.nn.utils.clip_grad_norm_
                def capture(parameters,*args,**kw):
                    parameters=list(parameters);captured.append(gradients(parameters))
                    return clip(parameters,*args,**kw)
                np.random.seed(77)
                with patch('algorithms.drl_gnn_fast.ppo.nn.utils.clip_grad_norm_',side_effect=capture), \
                     patch.object(b.optimizer,'step') as step:
                    actual=b.update(records)
                self.assertEqual(len(wanted),step.call_count)
                for x,y in zip(captured,wanted):torch.testing.assert_close(x,y,rtol=5e-4,atol=3e-6)
                for key,value in a.network.state_dict().items():
                    torch.testing.assert_close(value,b.network.state_dict()[key],rtol=5e-4,atol=3e-6)
                keys=('loss','policy_loss','value_loss','entropy','approx_kl','old_approx_kl','clipfrac','grad_norm')
                np.testing.assert_allclose([actual[k] for k in keys],expected,rtol=5e-4,atol=3e-6)
                self.assertFalse(b._bootstrap);self.assertFalse(b.network._tensor_cache)

    def test_real_adam_grouping_is_close_and_keeps_16_steps(self):
        torch.manual_seed(31);a=SourceAgent();records=rollout(a,512)
        b=PPOAgent();b.network.load_state_dict(a.network.state_dict())
        np.random.seed(77);a.update(records)
        np.random.seed(77)
        with patch.object(b.optimizer,'step',wraps=b.optimizer.step) as step:b.update(records)
        self.assertEqual(step.call_count,16)
        # Fixed-weight gradients are checked strictly above. Real Adam can amplify
        # reduction roundoff near zero BN bias gradients; bound policy drift separately.
        for key,value in a.network.named_parameters():
            torch.testing.assert_close(value,dict(b.network.named_parameters())[key],rtol=0,atol=1e-3)
        with torch.no_grad():
            for state in (graph(0),graph(1),graph(2)):
                da,va=a.network.distribution_and_value(state)
                db,vb=b.network.distribution_and_value(state)
                self.assertLess(float(torch.distributions.kl_divergence(da,db)),.005)
                self.assertLess(float((va-vb).abs()),.001)

    def test_selection_keeps_099_when_as_uses_092_and_boundary_fallback_ties(self):
        for rows in ([validation_row(49,1.)],[validation_row(50,100.)],
                     [validation_row(1,3.,2),validation_row(98,7.,98,601)]):
            self.assertEqual(selection.validation_selection(rows),reference_selection.validation_selection(rows))
        self.assertEqual(selection.selection_metadata(),reference_selection.selection_metadata())
        self.assertFalse(selection.validation_selection([validation_row(46,1.)])['feasible'])
        self.assertTrue(as_selection.validation_selection([validation_row(46,1.)])['feasible'])
        low=selection.validation_selection([validation_row(49,1.)])
        high=selection.validation_selection([validation_row(50,100.)])
        self.assertFalse(low['feasible']);self.assertTrue(high['feasible'])
        self.assertLess(selection.selection_key(high),selection.selection_key(low))
        fallback=selection.validation_selection([validation_row(48,.1)])
        self.assertLess(selection.selection_key(low),selection.selection_key(fallback))
        self.assertEqual(selection.selection_key(high),selection.selection_key(dict(high)))
        with self.assertRaises(ValueError):selection.validation_selection([dict(validation_row(49,1.),wf_success_rate=1.)])

    def test_paths_source_scales_and_legacy_separation(self):
        train=FastScenario(mode='train',episodes=1)
        target=FastScenario(mode='eval',task='L',resource='L',deadline='Loose',
            train_task='S',train_resource='S',train_deadline='Tight')
        self.assertEqual(train.checkpoint_dir(),target.checkpoint_dir())
        self.assertNotEqual(train.checkpoint_dir(),Scenario(mode='train').checkpoint_dir('drl_gnn'))
        self.assertIn(MODEL_VERSION,str(train.checkpoint_dir()))
        self.assertEqual(target.normalization_kwargs()['source_train_scene'],'SS_T')
        self.assertEqual(train.selection_protocol()['selection'],selection.SELECTION_TAG)
        self.assertEqual(train.protocol().metadata()['selection'],'minimum_validation_total_energy')
        self.assertEqual(runner.output_dir(target,'run'),target.evaluation_dir)

    def test_validation_is_actor_only_does_not_touch_training_or_rng(self):
        scene=FastScenario(mode='train',episodes=1)
        agent=PPOAgent()
        before=deepcopy(agent.network.state_dict())
        torch_rng=torch.get_rng_state().clone();np_rng=np.random.get_state()
        with patch.object(agent,'bootstrap_value',side_effect=AssertionError('validation bootstrap')):
            rows,_=runner.evaluate(agent,scene,[600],split='validation',kwargs=self.kwargs())
        self.assertEqual(rows[0]['wf_completed_count'],2)
        for key,value in before.items():
            torch.testing.assert_close(value,agent.network.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        for x,y in zip(np_rng,np.random.get_state()):np.testing.assert_array_equal(x,y)

    def test_measurement_switch_does_not_control_selection(self):
        scene=FastScenario(mode='train',episodes=2)
        schedule=[validation_row(49,1.),validation_row(50,100.)]
        for enabled in ('0','1'):
            rows=iter(schedule)
            def validate(*args,**kwargs):
                row=next(rows)
                return [row],{'total_energy':{'mean':row['total_energy'],'std':0.}}
            with patch.dict('os.environ',{'EXPERIMENT_METRICS':enabled}), \
                 patch.object(runner,'planned_assignments',return_value=12), \
                 patch.object(runner,'evaluate',side_effect=validate):
                agent,directory=runner.train(scene,DRLGNNConfig(),self.kwargs(),self.root/('metrics'+enabled))
            _,obj=runner.load_checkpoint(directory/'best.pth',scene)
            self.assertEqual(obj['selection_evidence']['selected_episode'],1)
            self.assertTrue(obj['selection_evidence']['validation_selection']['feasible'])

    def test_controlled_training_best_final_evidence_roundtrip_and_old_rejection(self):
        scene=FastScenario(mode='train',episodes=6)
        schedule=[(0,1.),(49,30.),(50,100.),(49,1.),(50,80.),(50,80.)]
        index=iter(schedule)
        def validation(*args,**kwargs):
            success,energy=next(index);row=validation_row(success,energy)
            return [row],{'total_energy':{'mean':energy,'std':0.}}
        with patch.object(runner,'planned_assignments',return_value=36), \
             patch.object(runner,'evaluate',side_effect=validation):
            agent,directory=runner.train(scene,DRLGNNConfig(),self.kwargs(),self.root/'models')
        self.assertEqual((agent.steps,agent.updates),(36,1))
        loaded,obj=runner.load_checkpoint(directory/'best.pth',scene)
        self.assertEqual(obj['selection_evidence']['selected_episode'],4)
        self.assertEqual(obj['completed_episodes'],5)
        self.assertEqual(obj['selection_evidence']['checkpoint_kind'],'best')
        _,final=runner.load_checkpoint(directory/'final.pth',scene,require_best=False)
        self.assertEqual(final['completed_episodes'],6)
        self.assertEqual(final['selection_evidence']['selected_episode'],4)
        with self.assertRaisesRegex(ValueError,'requires best'):
            runner.load_checkpoint(directory/'final.pth',scene)
        for field,value in (('model_version','drl_gnn_source_be67762_f19_inputs_v4'),
                            ('selection_evidence',None)):
            changed=deepcopy(obj);changed[field]=value
            torch.save(changed,self.root/'tampered.pth')
            with self.assertRaises(ValueError):runner.load_checkpoint(self.root/'tampered.pth',scene)
        changed=deepcopy(obj);changed['selection_evidence']['validation_rows'][0]['seed']=700
        torch.save(changed,self.root/'tampered.pth')
        with self.assertRaises(ValueError):runner.load_checkpoint(self.root/'tampered.pth',scene)
        a,_=runner.evaluate(loaded,scene,[700],kwargs=self.kwargs())
        b,_=runner.evaluate(loaded,scene,[700],kwargs=self.kwargs())
        self.assertEqual(a,b)
        summary=json.loads((directory/'logs/training_summary.json').read_text())
        self.assertTrue(summary['feasible_model_found']);self.assertEqual(summary['selected_episode'],4)
        self.assertEqual(len((directory/'logs/validation_selection.jsonl').read_text().splitlines()),6)
        with self.assertRaises(FileExistsError):runner.train(scene,out=directory)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_tiny_train_save_load_evaluate(self):
        scene=FastScenario(mode='train',episodes=1)
        with patch.object(runner,'planned_assignments',return_value=6):
            agent,directory=runner.train(scene,DRLGNNConfig(device='cuda'),self.kwargs(),self.root/'cuda')
        loaded,obj=runner.load_checkpoint(directory/'best.pth',scene,device='cuda')
        self.assertEqual(loaded.device.type,'cuda')
        self.assertEqual(obj['execution']['graphs_per_backward'],4)
        self.assertLessEqual(loaded.network.cuda_memory_policy['allocator_fraction'],.85)
        a,_=runner.evaluate(agent,scene,[700],kwargs=self.kwargs())
        b,_=runner.evaluate(loaded,scene,[700],kwargs=self.kwargs());self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
