"""Paper IRWS correctness, true meta-gradient chains, and shared-physics tests."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import importlib
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import networkx as nx
import numpy as np
import torch
from common.read_xml_opt_Tsize import Task
from common.scenarios import Scenario
from common.experiment_protocol import ExperimentProtocol,DEFAULT_PROTOCOL
from common.f18_state_normalization import VERSION,SCALE_UNITS
from environments.irws import IRWSEnvRouteA
from algorithms.irws_paper.config import PaperConfig
from algorithms.irws_paper.environment import PaperCloudEnv,GraphData,paper_ranks,fusion_weight
from algorithms.irws_paper.learner import PaperLearner,gae,clipped_loss
from algorithms.irws_paper.networks import WorkflowEmbedding,masked_distribution
from algorithms.irws_paper.runner import run_episode,save_checkpoint,load_checkpoint


@contextmanager
def synthetic_env(path,kind=PaperCloudEnv):
    graph=nx.DiGraph();graph.add_nodes_from(range(6));graph.add_edges_from([(0,2),(1,2),(2,3),(2,4),(4,5)])
    tasks=[Task(task_id=i,workload_mi=float(8+i),parents=list(graph.predecessors(i)),children=list(graph.successors(i))) for i in range(6)]
    for t in tasks:
        t.parent_in_bits={p:2e6 for p in t.parents};t.from_parents_bits=sum(t.parent_in_bits.values());t.out_file_size_sum_bits=2e6
    kwargs=dict(dax_paths=['synthetic.xml'],workflows_per_episode=2,deadline_mode='none',max_ready_tasks=1,
        num_hosts=2,vms_per_host=(2,1),pc_tiers=(1.,2.),bw_tiers=(1.,2.),random_seed=32,horizon=1e9,
        normalization_mode=VERSION,normalization_config=str(path),source_train_scene='SS_T')
    with patch('environments.irws.poisson_arrival_times',return_value=np.array([0.,1.])),patch(
            'environments.irws.load_workflow_from_dax',side_effect=lambda *a,**kw:(graph.copy(),deepcopy(tasks))):
        yield kind(**kwargs)


def fixture(root):
    scales=dict(R_ref=5,T_ref=20.,workload_mi_ref=20.,input_bits_ref=1e7,output_bits_ref=1e7,
                up_rank_ref=20.,down_rank_ref=20.,vm_pc_ref=2.,vm_bw_ref=2.,host_pc_ref=3.)
    path=root/'normalization.json';path.write_text(json.dumps(dict(normalization_version=VERSION,units=SCALE_UNITS,scenes={'SS_T':scales})))
    protocol=root/'protocol.json';protocol.write_text(json.dumps(dict(version='test_disjoint',train=[32],validation=[632],test=[2],
        episodes=1,selection='minimum_validation_total_energy')))
    class FixtureScenario(Scenario):
        @property
        def vms_per_host(self):return (2,1)
    scenario=FixtureScenario(mode='train',episodes=1,experiment_protocol=str(protocol),normalization_config=str(path))
    return path,scenario


def collect(env,learner,limit=8):
    env.reset(seed=32);learner.reset_history();d=learner.prepare(env);pending=(d,*learner.act(d))
    for _ in range(limit):
        d,a,lp,val=pending;r,done,info=env.paper_step(a)
        if done:nxt=None;pending=None
        else:
            nxt=learner.prepare(env);pending=(nxt,*learner.act(nxt))
        learner.remember(d,a,lp,val,r,done,info,nxt,20.,1000.)
        if done:break
    return 0. if pending is None else pending[3]


class PaperIRWSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='IRWS_paper_tests_');self.root=Path(self.temp.name)
        self.path,self.scenario=fixture(self.root);torch.manual_seed(73)
    def tearDown(self):self.temp.cleanup()

    def test_table_III_and_rank_equations(self):
        cfg=PaperConfig();self.assertEqual((cfg.gamma,cfg.embedding_dim,cfg.trajectory_length),(.2,5,16))
        self.assertEqual(cfg.actor_hidden,(128,64));self.assertEqual(cfg.critic_hidden,(64,64,32))
        graph=nx.DiGraph([(0,1)])
        tasks=[Task(task_id=0,workload_mi=10.),Task(task_id=1,workload_mi=20.,parent_in_bits={0:8e6})]
        up,down,ref=paper_ranks(graph,tasks,2.,1.)
        np.testing.assert_allclose(up,[23,10]);np.testing.assert_allclose(down,[0,13]);np.testing.assert_allclose(ref,[5,18])

    def test_bidirectional_topology_and_task_probabilities(self):
        encoder=WorkflowEmbedding()
        a=GraphData(0,(0,1,2),((.1,.2),(.3,.4),(.5,.6)),((),(0,),(1,)),(),(),())
        b=replace(a,parents=((),(),(0,1)))
        self.assertFalse(torch.allclose(encoder(a),encoder(b)))
        # Source embeddings see changed descendants; sinks see changed ancestors.
        self.assertFalse(torch.allclose(encoder(a)[0],encoder(b)[0]))
        self.assertFalse(torch.allclose(encoder(a)[2],encoder(b)[2]))
        with synthetic_env(self.path) as env:
            env.reset();learner=PaperLearner(3,PaperConfig());c=env.get_ready_task_candidates()['gids']
            self.assertEqual(len(c),2) # configured cap=1 is intentionally not used
            dist=learner.task.distribution(learner.task.embeddings(env.graphs()),c)
            self.assertTrue(torch.all(dist.probs>0));self.assertAlmostEqual(float(dist.probs.sum()),1.,places=6)
            self.assertEqual(len(env.graphs()),1) # second workflow has not arrived
            with self.assertRaises(ValueError):env.select_task(2)

    def test_all_fifteen_real_dax_templates_variable_graph_sizes(self):
        from common.read_xml_opt_Tsize import load_workflow_from_dax
        encoder=WorkflowEmbedding().double();paths=sorted({p for size in 'SML' for p in Scenario(task=size).dax_paths()})
        self.assertEqual(len(paths),15);sizes=set()
        for path in paths:
            graph,tasks=load_workflow_from_dax(path,seed=32,randomize_payloads=True)
            up,down,ref=paper_ranks(graph,tasks,4.,4000.)
            data=GraphData(0,tuple(range(len(tasks))),tuple((u/1000.,d/1000.) for u,d in zip(up,down)),
                tuple(tuple(graph.predecessors(i)) for i in range(len(tasks))),tuple(up),tuple(down),tuple(ref))
            result=encoder(data);self.assertEqual(result.shape,(len(tasks),5));self.assertTrue(torch.isfinite(result).all())
            sizes.add(len(tasks))
        self.assertGreater(len(sizes),5)

    def test_dynamic_fusion_literal_and_boundaries(self):
        self.assertEqual(fusion_weight(2,4),(.5,False));self.assertEqual(fusion_weight(2,1),(2.,False))
        self.assertEqual(fusion_weight(2,0),(1.,True));self.assertEqual(fusion_weight(0,0),(0.,True))
        with self.assertRaises(ValueError):fusion_weight(np.nan,1)

    def test_gae_bootstrap_clip_mse_and_masks(self):
        rewards=torch.tensor([1.,2.]);values=torch.tensor([.2,.3]);done=torch.tensor([0.,0.])
        adv=gae(rewards,values,done,5.,.2,0.)
        torch.testing.assert_close(adv,torch.tensor([.86,2.7]))
        done[-1]=1;self.assertAlmostEqual(float(gae(rewards,values,done,5.,.2,0.)[-1]),1.7,places=5)
        adv2=gae(rewards,values,torch.tensor([1.,0.]),5.,.2,.9)
        self.assertAlmostEqual(float(adv2[0]),.8,places=5)
        loss=clipped_loss(torch.log(torch.tensor([1.5,.5])),torch.zeros(2),torch.tensor([1.,-1.]),.2)
        self.assertAlmostEqual(float(loss),-.2,places=6)
        dist=masked_distribution(torch.zeros(3),[1,0,1]);self.assertEqual(float(dist.probs[1]),0)
        with self.assertRaises(ValueError):masked_distribution(torch.zeros(2),[0,0])
        for n in (1,3,25):
            l=PaperLearner(n,PaperConfig());self.assertEqual(l.actor(l.tensor(np.zeros(l.obs_dim))).shape,(n,))

    def test_meta_chain_and_all_networks_update(self):
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig());tail=collect(env,learner)
            before={n:torch.cat([p.detach().flatten() for p in getattr(learner,n).parameters()]).clone()
                    for n in ('task','actor','critic','intrinsic')}
            losses=learner.losses(tail)
            grad=torch.autograd.grad(losses['intrinsic_meta'],tuple(learner.intrinsic.parameters()),retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in grad));self.assertGreater(sum(float(g.abs().sum()) for g in grad),0)
            eta=torch.autograd.grad(losses['embedding_meta'],tuple(learner.task.parameters()),allow_unused=True)
            self.assertGreater(sum(float(g.abs().sum()) for g in eta if g is not None),0)
            result=learner.update(tail)
            for n in before:
                after=torch.cat([p.detach().flatten() for p in getattr(learner,n).parameters()])
                self.assertGreater(float((before[n]-after).abs().max()),0,n)
            self.assertTrue(all(np.isfinite(v) for v in result['losses'].values()))
            self.assertGreater(result['gradient_norms']['intrinsic'],0)
            self.assertGreater(result['gradient_norms']['embedding_and_selector'],0)

    def test_intrinsic_meta_gradient_matches_finite_difference(self):
        with synthetic_env(self.path) as env:
            # Magnified virtual step only for a numerically resolvable derivative test.
            learner=PaperLearner(3,replace(PaperConfig(),actor_lr=.1));tail=collect(env,learner)
            loss=learner.losses(tail)['intrinsic_meta'];parameter=learner.intrinsic.head.bias
            analytic=float(torch.autograd.grad(loss,parameter)[0][0]);original=parameter.detach().clone();epsilon=.01
            def perturbed(sign):
                with torch.no_grad():parameter.copy_(original+sign*epsilon)
                return float(learner.losses(tail)['intrinsic_meta'].detach())
            numeric=(perturbed(1)-perturbed(-1))/(2*epsilon)
            with torch.no_grad():parameter.copy_(original)
            self.assertGreater(abs(analytic),1e-7)
            np.testing.assert_allclose(analytic,numeric,rtol=.04,atol=2e-5)

    def test_task_selector_parameters_really_update(self):
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig());tail=collect(env,learner)
            before=[p.detach().clone() for p in learner.task.score.parameters()]
            learner.update(tail)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,learner.task.score.parameters())))

    def test_history_reset_nonterminal_and_pure_reference(self):
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig());tail=collect(env,learner,2)
            self.assertTrue(torch.any(learner.hidden!=0))
            with self.assertRaises(ValueError):learner.update(None)
            segments=deepcopy(env._host_segments);energy=env.total_energy
            env.reference_cost(env.ready_task_ids[0])
            self.assertEqual(segments,env._host_segments);self.assertEqual(energy,env.total_energy)
            learner.update(tail);learner.reset_history();self.assertTrue(torch.all(learner.hidden==0))

    def test_busy_vm_never_forged_legal_and_invalid_action_no_mutation(self):
        with synthetic_env(self.path) as env:
            env.reset();gid=env.ready_task_ids[0];env.select_task(gid)
            before=(env.current_task_gid,env.total_energy,list(env.task_state))
            with self.assertRaises(ValueError):env.paper_step(99)
            self.assertEqual(before,(env.current_task_gid,env.total_energy,env.task_state))
            env.vm_available_at[:]=10
            with patch.object(env,'_advance_until_decision') as advance:
                state=env.get_vm_obs_and_mask();advance.assert_called_once();self.assertEqual(state['mask'].sum(),0)

    def test_physical_trace_matches_shared_environment(self):
        with synthetic_env(self.path) as a,synthetic_env(self.path,IRWSEnvRouteA) as b:
            a.reset();b.reset();steps=0
            while not a.done_flag:
                self.assertEqual(a.current_time,b.current_time)
                gid=min(a.ready_task_ids,key=lambda i:(a.task_ready_time[i],i));a.select_task(gid);b.select_task(gid)
                mask=a.get_vm_obs_and_mask()['mask'];action=int(np.flatnonzero(mask)[0])
                _,_,ia=a.paper_step(action);_,_,_,_,ib=b.step(action)
                for field in ('start','end','dur','ddl_task','delta_energy_total'):self.assertEqual(ia[field],ib[field])
                self.assertEqual(a.task_state,b.task_state);self.assertEqual(a.task_start_time,b.task_start_time)
                self.assertEqual(a.task_end_time,b.task_end_time);np.testing.assert_array_equal(a.vm_available_at,b.vm_available_at)
                self.assertEqual(a._host_segments,b._host_segments);self.assertEqual(a.total_energy,b.total_energy);steps+=1
            self.assertEqual(steps,12);self.assertEqual(a.get_episode_metrics(),b.get_episode_metrics())

    def test_protocol_isolation_and_frozen_test_instances(self):
        protocol=ExperimentProtocol.load(DEFAULT_PROTOCOL)
        self.assertEqual(protocol.test,tuple(range(700,730)));self.assertEqual(len(protocol.train),600)
        with self.assertRaises(ValueError):ExperimentProtocol((1,2),(2,),(3,),2)
        with self.assertRaises(ValueError):Scenario(mode='eval',seed_spec='32',experiment_protocol=str(DEFAULT_PROTOCOL))
        self.assertEqual(Scenario(experiment_protocol=str(DEFAULT_PROTOCOL)).first_training_seed(),0)
        self.assertEqual(Scenario(legacy_split=True).first_training_seed(),1)
        from types import SimpleNamespace
        from common.f18_state_normalization import bind_normalization,check_checkpoint
        agent=bind_normalization(SimpleNamespace(),self.scenario)
        self.assertIn('experiment_protocol_hash',agent.normalization_metadata)
        without=dict(agent.normalization_metadata);without.pop('experiment_protocol_hash')
        with self.assertRaises(ValueError):check_checkpoint(without,agent.normalization_metadata)

    def test_explicit_protocol_reaches_existing_training_entries(self):
        from common.test_scenarios import ENVIRONMENTS,CapturedConfiguration,capture_environment
        for algorithm in ('mahdrl','marl','pd3qn','ablation_hostrand','ablation_mgrrand','ablation_vmrand','ablation_nohost'):
            module=importlib.import_module('algorithms.'+algorithm+'.train')
            scenario=Scenario(mode='train',episodes=1,experiment_protocol=str(DEFAULT_PROTOCOL))
            context=patch.object(module,'_load_cache_seeds',return_value=list(range(1001))) if algorithm=='marl' else contextlib.nullcontext()
            with context,patch.object(module,'SCENARIO',scenario),patch.object(module,ENVIRONMENTS[algorithm],side_effect=capture_environment), patch('os.path.exists',return_value=True),patch('os.makedirs'),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(CapturedConfiguration) as captured:module.train()
            self.assertEqual(captured.exception.kwargs['random_seed'],0)
            self.assertEqual(module.SCENARIO.validation_seeds(),[600]) # restoration preserves the default F22 configuration

    def test_checkpoint_roundtrip_resume_and_no_evaluation_learning(self):
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig());result=run_episode(env,learner,32,True)
            save_checkpoint(self.root/'model.pth',learner,self.scenario,0,1.,{'test_fixture':True})
            # Source fixture has a custom 3-VM resource shape; preserve explicit shape for loader validation.
            scenario=replace(self.scenario,resource='S')
            loaded,obj=load_checkpoint(self.root/'model.pth',scenario,restore_training=True)
            a=run_episode(env,learner,2,False);weights={k:v.clone() for k,v in loaded.task.state_dict().items()}
            b=run_episode(env,loaded,2,False)
            self.assertEqual(a['metrics'],b['metrics']);self.assertEqual(a['fusion'],b['fusion'])
            self.assertTrue(all(torch.equal(v,loaded.task.state_dict()[k]) for k,v in weights.items()))
            self.assertTrue(all(obj['optimizers'][n]['state'] for n in ('task','actor','critic','intrinsic')))
            historical=dict(obj);historical['normalization']=dict(obj['normalization'])
            historical['normalization'].pop('experiment_protocol_hash',None)
            torch.save(historical,self.root/'historical.pth')
            _,restored=load_checkpoint(self.root/'historical.pth',scenario)
            self.assertEqual(restored['protocol'],obj['protocol'])
            torch.save({'model_version':'old_irws'},self.root/'old.pth')
            with self.assertRaises(ValueError):load_checkpoint(self.root/'old.pth',scenario)


if __name__=='__main__':unittest.main()