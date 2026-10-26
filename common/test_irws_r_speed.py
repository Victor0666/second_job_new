"""Accelerated IRWS: math equivalence, cache lifetimes, rollouts and precision."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import networkx as nx
import numpy as np
import torch

from algorithms.irws_r import runner
from algorithms.irws_r.config import PaperConfig
from algorithms.irws_r.environment import GraphData
from algorithms.irws_r.networks import WorkflowEmbedding
from algorithms.irws_r.numerics import NumericalDiagnostics
from algorithms.irws_r_v2.networks import WorkflowEmbedding as SerialEmbedding
from common.f19_numerics import NumericalDiagnostics as ReferenceDiagnostics
from common.test_irws_paper import fixture,synthetic_env


class IRWSSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='irws_speed_test_')
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.path,self.scene=fixture(self.root)

    def graphs(self):
        return (GraphData(0,(0,1,2,3),((.1,.2),(.3,.4),(.2,.5),(.6,.7)),
                          ((),(),(0,1),(1,2)),(),(),()),
                GraphData(1,(4,5,6),((.2,.1),(.4,.3),(.5,.8)),((),(0,),(0,1)),(),(),()))

    def test_defaults_and_historical_snapshot_are_preserved(self):
        import hashlib,json
        cfg=PaperConfig()
        self.assertEqual((cfg.rollout_size,cfg.precision,cfg.batch_size),(64,'float32',64))
        self.assertEqual((cfg.actor_lr,cfg.critic_lr,cfg.intrinsic_lr,cfg.embedding_lr),(1e-5,1e-4,1e-5,1e-6))
        self.assertEqual(runner.queue_settings(self.scene),dict(queue_capacity=10,slot_duration=5.))
        snapshot=Path(__file__).resolve().parents[1]/'algorithms/irws_r_v2'
        for name,digest in json.loads((snapshot/'snapshot_sha256.json').read_text()).items():
            raw=(snapshot/name).read_bytes();lf=raw.replace(b'\r\n',b'\n')
            self.assertIn(digest,[hashlib.sha256(b).hexdigest() for b in (raw,lf,lf.replace(b'\n',b'\r\n'))],name)

    def test_frontier_batch_matches_serial_forward_gradients_and_adam(self):
        torch.manual_seed(19)
        batch=WorkflowEmbedding().double();serial=SerialEmbedding().double()
        serial.load_state_dict(batch.state_dict())
        a=torch.cat(batch.forward_many(self.graphs()))
        b=torch.cat([serial(g) for g in self.graphs()])
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)
        weights=torch.arange(a.numel(),dtype=a.dtype).reshape_as(a)/a.numel()
        (a*weights).sum().backward();(b*weights).sum().backward()
        for x,y in zip(batch.parameters(),serial.parameters()):
            torch.testing.assert_close(x.grad,y.grad,rtol=1e-10,atol=1e-12)
        for model in (batch,serial):torch.optim.Adam(model.parameters(),lr=1e-6).step()
        for x,y in zip(batch.parameters(),serial.parameters()):
            torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-12)

    def test_static_cache_survives_parameter_updates_but_not_dtype_change(self):
        model=WorkflowEmbedding().double();graphs=self.graphs()
        with patch('algorithms.irws_r.networks.nx.topological_sort',wraps=nx.topological_sort) as sort:
            before=torch.cat(model.forward_many(graphs)).detach()
            model.forward_many(graphs)
            with torch.no_grad():model.pool.bias.add_(1.)
            after=torch.cat(model.forward_many(graphs)).detach()
            self.assertEqual(sort.call_count,len(graphs))
        torch.testing.assert_close(after,before+1.)
        self.assertEqual(len(model._graphs),2)
        model.float()
        self.assertFalse(model._graphs)
        self.assertEqual(model.forward_many(graphs)[0].dtype,torch.float32)

    def test_resource_cache_reused_at_boundary_invalidated_at_equal_time_dispatch(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            env.reset(seed=32)
            learner=runner.PaperLearner(env.num_vms,PaperConfig())
            boundary=learner.boundary_snapshot(env)
            cached={d.gid:d.resources for d in boundary.decisions}
            with patch.object(env,'resource_features',wraps=env.resource_features) as features,patch.object(env,'queue_mask',wraps=env.queue_mask) as masks:
                learner._embedding_cache={}
                actual=learner.prepare(env)
                self.assertIs(actual.resources,cached[actual.gid])
                features.assert_not_called()
                masks.assert_not_called()
            gid=next(g for g in env.ready_task_ids if g!=actual.gid)
            old=env.resource_features(gid)
            revision=env.state_revision;time=env.current_time
            env.paper_step(0)
            self.assertEqual(env.current_time,time)
            self.assertGreater(env.state_revision,revision)
            self.assertFalse(np.array_equal(old,env.resource_features(gid)))
            env.reset(seed=32)
            self.assertGreater(env.state_revision,revision)

    def test_64_decisions_then_tail_no_dropped_samples_or_interval_flush(self):
        from common.read_xml_opt_Tsize import Task
        graph=nx.empty_graph(80,create_using=nx.DiGraph)
        tasks=[Task(task_id=i,workload_mi=float(8+i),parents=[],children=[]) for i in range(80)]
        make=lambda **kw:runner.PaperCloudEnv(queue_capacity=100,**kw)
        with synthetic_env(self.path,make) as env,patch('environments.irws.load_workflow_from_dax',
                side_effect=lambda *a,**kw:(graph.copy(),deepcopy(tasks))):
            learner=runner.PaperLearner(env.num_vms,PaperConfig())
            collected=[];update=learner.update
            def record(*args,**kwargs):
                collected.extend(r['decision'].gid for r in learner.buffer)
                return update(*args,**kwargs)
            with patch.object(learner,'update',side_effect=record):
                result=runner.run_episode(env,learner,32,True)
            self.assertEqual([u['collected_transitions'] for u in result['updates']],[64,64,32])
            self.assertEqual(len(collected),160)
            self.assertEqual(len(set(collected)),160)
            self.assertTrue(any(u['policy_intervals']>1 for u in result['updates']))
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),160)
            self.assertTrue(result['updates'][-1]['terminal'])
            self.assertFalse(learner.buffer)
            self.assertIsNone(learner._pending_boundary)

    def test_float32_four_learned_groups_update_fixed_Q_and_checkpoint_isolation(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            runner.seed_everything(23)
            learner=runner.PaperLearner(env.num_vms,PaperConfig())
            before={n:[p.detach().clone() for p in m.parameters()] for n,m in
                    [('actor',learner.actor),('critic',learner.critic),('intrinsic',learner.intrinsic),
                     ('embedding',learner.task.embedding),('Q',learner.task.score)]}
            result=runner.run_episode(env,learner,32,True)
            self.assertEqual(len(result['updates']),1)
            for name,module in [('actor',learner.actor),('critic',learner.critic),('intrinsic',learner.intrinsic),
                                ('embedding',learner.task.embedding),('Q',learner.task.score)]:
                changed=any(not torch.equal(a,b) for a,b in zip(before[name],module.parameters()))
                self.assertEqual(changed,name!='Q',name)
                self.assertTrue(all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in module.parameters()))
            path=self.root/'checkpoint.pth'
            runner.save_checkpoint(path,learner,self.scene,0,10.,runner.env_kwargs(self.scene,32))
            loaded,obj=runner.load_checkpoint(path,self.scene)
            self.assertEqual(obj['numerical_precision'],'float32')
            for options in (dict(precision='float64'),dict(rollout_size=32),dict(slot_duration=1.)):
                with self.assertRaisesRegex(ValueError,'settings mismatch'):
                    runner.load_checkpoint(path,replace(self.scene,paper_options=options))
            resumed_scene=replace(self.scene,paper_options=dict(precision='float32'))
            resumed,_=runner.load_checkpoint(path,resumed_scene,restore_training=True)
            a=runner.run_episode(env,resumed,33,True)
            # Restoring the same RNG and optimizer state must reproduce continuation.
            replay,_=runner.load_checkpoint(path,resumed_scene,restore_training=True)
            b=runner.run_episode(env,replay,33,True)
            self.assertEqual(a['updates'],b['updates'])
            for name in ('actor','critic','intrinsic','task'):
                for x,y in zip(getattr(resumed,name).parameters(),getattr(replay,name).parameters()):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_fixed_action_physics_matches_v2_at_same_five_second_ticks(self):
        from algorithms.irws_r_v2.environment import PaperCloudEnv as PreviousEnv
        from common.workflow_opt import energy_from_records
        make=lambda **kw:PreviousEnv(slot_duration=5.,**kw)
        with synthetic_env(self.path,runner.PaperCloudEnv) as actual,synthetic_env(self.path,make) as reference:
            for env in (actual,reference):
                env.reset(seed=32)
                while not env.done_flag:
                    env.select_task(env.get_ready_task_candidates()['gids'][0])
                    env.paper_step(int(np.flatnonzero(env.queue_mask())[0]))
                    if not env.can_dispatch():env.finish_phase_and_advance()
            self.assertEqual(actual.task_placement,reference.task_placement)
            np.testing.assert_array_equal(actual.task_start_time,reference.task_start_time)
            np.testing.assert_array_equal(actual.task_end_time,reference.task_end_time)
            self.assertEqual(actual.total_energy,reference.total_energy)
            self.assertAlmostEqual(actual.total_energy,energy_from_records(actual.records,actual.hosts),places=8)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_observations_stay_on_device_and_diagnostics_keep_counts(self):
        with synthetic_env(self.path,runner.PaperCloudEnv) as env:
            learner=runner.PaperLearner(env.num_vms,PaperConfig(device='cuda'))
            env.reset(seed=32);d=learner.prepare(env)
            self.assertEqual(d.obs.device.type,'cuda')
            self.assertEqual(d.resources.device.type,'cuda')
            result=runner.run_episode(env,learner,32,True)
            self.assertTrue(result['numerics']['enabled'])
        reference=ReferenceDiagnostics();deferred=NumericalDiagnostics()
        for start in range(0,400,23):
            values=torch.arange(start*3,(start+23)*3,dtype=torch.float32).reshape(23,3)
            if start==0:values[0,0]=torch.nan
            reference.observe('x',values)
            deferred.observe('x',values.cuda())
        self.assertEqual(reference.snapshot(),deferred.snapshot())
        self.assertEqual(reference.snapshot(),deferred.snapshot())


if __name__=='__main__':unittest.main()
