"""Independent queue, equation, history and checkpoint checks for the copied IRWS."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.test_irws_paper import fixture,synthetic_env
from common.workflow_opt import energy_from_records
from environments.irws import IRWSEnvRouteA
from algorithms.irws_restored.config import PaperConfig,MODEL_VERSION
from algorithms.irws_restored.environment import PaperCloudEnv,fusion_weight
from algorithms.irws_restored.learner import PaperLearner,clipped_loss,clipping_gate,fuse_reward,meta_jacobian_scales
from algorithms.irws_restored.runner import run_episode,save_checkpoint,load_checkpoint,train,evaluate,env_kwargs,output_dir,artifact_name


class RestoredIRWSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='irws_restored_test_');self.root=Path(self.temp.name)
        self.path,self.scene=fixture(self.root);torch.manual_seed(23);np.random.seed(23)
    def tearDown(self):self.temp.cleanup()

    def test_both_original_versions_and_shared_physics_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        manifest=json.loads((root/'algorithms/irws_restored/copied_from_sha256.json').read_text())
        for file,digest in manifest.items():
            if file == 'common/scenarios.py':
                # New independent algorithms extend the registry/CLI, not the physical defaults.
                from common.scenarios import Scenario
                self.assertEqual(Scenario().vms_per_host,(9,8,8))
                self.assertEqual(Scenario(resource='M').vms_per_host,(9,9,8,8,8,8))
                self.assertEqual(Scenario(resource='L').vms_per_host,(9,9,9,8,8,8,8,8,8))
                self.assertEqual(Scenario().deadline_probs,(.8,.2))
                self.assertEqual(Scenario(deadline='Medium').deadline_probs,(.5,.5))
                self.assertEqual(Scenario(deadline='Loose').deadline_probs,(.2,.8))
                self.assertEqual(Scenario().protocol().train,tuple(range(600)))
                self.assertEqual(Scenario().validation_seeds(),[600])
                self.assertEqual(Scenario().seeds,tuple(range(700,730)))
                continue
            raw=(root/file).read_bytes();lf=raw.replace(b'\r\n',b'\n')
            self.assertIn(digest,[hashlib.sha256(data).hexdigest()
                                 for data in (raw,lf,lf.replace(b'\n',b'\r\n'))],file)

    def test_literal_fusion_and_only_zero_rank_boundary(self):
        self.assertEqual(fusion_weight(2.,4.),(.5,False))
        self.assertEqual(fusion_weight(2.,1.),(2.,False))
        self.assertEqual(fusion_weight(2.,0.),(1.,True))
        self.assertEqual(fusion_weight(0.,0.),(0.,True))
        self.assertEqual(fuse_reward(-3.,5.,2.),13.)
        self.assertEqual(fuse_reward(-3.,5.,.5),1.)
        with self.assertRaises(ValueError):fusion_weight(1.,-1.)

    def test_deadlines_and_ready_wait_are_not_explicit_network_inputs(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            env.reset();env.deadline_mode='cache_fcfs'
            env.workflows[0].deadline=1e9
            env.task_baseline_finish[:]=[1e9]*len(env.task_baseline_finish)
            learner=PaperLearner(env.num_vms,PaperConfig())
            self.assertEqual(learner.task.score[0].in_features,learner.cfg.embedding_dim)
            self.assertEqual(learner.obs_dim,learner.cfg.embedding_dim+1+6*env.num_vms)
            first=learner.prepare(env,deterministic=True)
            baseline=deepcopy(env)
            env._advance_physics(.5)  # No arrival/completion; only Ready waiting increases.
            env.workflows[0].deadline=-1.
            env.task_baseline_finish[:]=[-1.]*len(env.task_baseline_finish)
            second=learner.prepare(env,deterministic=True)
            self.assertEqual(first.gid,second.gid)
            np.testing.assert_array_equal(first.obs,second.obs)
            np.testing.assert_array_equal(first.resources,second.resources)
            with torch.no_grad():
                for module in (learner.actor,learner.critic):
                    torch.testing.assert_close(module(learner.tensor(first.obs)),module(learner.tensor(second.obs)))
                tokens=[learner.history_token(learner.tensor(d.obs),0,0.,False,learner.tensor(d.obs))
                        for d in (first,second)]
                self.assertEqual(tokens[0].numel(),learner.intrinsic.cell.input_size)
                rewards=[learner.intrinsic(token,learner.hidden)[0] for token in tokens]
                torch.testing.assert_close(rewards[0],rewards[1])
            _,_,on_time=baseline.paper_step(0)
            _,_,late=env.paper_step(0)
            self.assertEqual(on_time['r_delay'],0.)
            self.assertLess(late['r_delay'],0.)  # Deadlines still govern external rewards.

    def test_busy_vm_accepts_fifo_until_waiting_queue_full(self):
        make=lambda **kw:PaperCloudEnv(queue_capacity=1,**kw)
        with synthetic_env(self.path,make) as env:
            env.reset();env.select_task(0);env.paper_step(0)
            env.select_task(1);env.paper_step(0)
            first=env.task_end_time[0];second=env.task_end_time[1]
            self.assertEqual(env.task_state[:2],['Running','Queued'])
            self.assertEqual(env.vm_queues[0],[1]);self.assertEqual(env.queue_mask()[0],0)
            self.assertEqual(env.current_time,0.)
            self.assertEqual(env.task_start_time[1],first)
            self.assertEqual(env.records[1].start_time,first)
            env.finish_phase_and_advance() # second workflow arrives at t=1
            env.select_task(6)
            before=(list(env.task_state),deepcopy(env.vm_queues),env.current_task_gid,len(env.records),env.total_energy)
            with self.assertRaises(ValueError):env.paper_step(0)
            self.assertEqual(before,(env.task_state,env.vm_queues,env.current_task_gid,len(env.records),env.total_energy))
            env._advance_physics(first)
            self.assertEqual(env.task_state[:3],['Finished','Running','unReady'])
            self.assertEqual(env.vm_queues[0],[])
            env._advance_physics(second)
            self.assertEqual(env.task_state[2],'Ready')
            self.assertEqual(env.task_ready_time[2],second)

    def test_policy_ticks_do_not_round_actual_completion_or_ready_time(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            env.reset();env.select_task(0);env.paper_step(1)
            env.select_task(1);env.paper_step(1)
            finish=env.task_end_time[1]
            self.assertEqual(finish,10.5)
            env._advance_physics(11.)
            self.assertEqual(env.current_time,11.)
            self.assertEqual(env.task_ready_time[2],10.5)
            self.assertEqual(env.task_end_time[1],10.5)

    def test_common_physics_replay_without_queueing(self):
        with synthetic_env(self.path,PaperCloudEnv) as new,synthetic_env(self.path,IRWSEnvRouteA) as old:
            new.reset();old.reset();count=0
            while not old.done_flag:
                self.assertEqual(new.current_time,old.current_time)
                self.assertEqual(new.ready_task_ids,old.ready_task_ids)
                gid=min(old.ready_task_ids,key=lambda i:(old.task_ready_time[i],i))
                action=int(np.flatnonzero(old.vm_available_at<=old.current_time+1e-12)[0])
                new.select_task(gid);old.select_task(gid)
                _,_,a=new.paper_step(action);_,_,_,_,b=old.step(action)
                for key in ('start','end','dur'):self.assertEqual(a[key],b[key])
                new._advance_physics(old.current_time)
                self.assertEqual(new.task_state,old.task_state)
                self.assertAlmostEqual(new.total_energy,old.total_energy,places=8);count+=1
            self.assertTrue(new.done_flag);self.assertEqual(count,12)
            self.assertEqual(new.get_episode_metrics(),old.get_episode_metrics())

    def test_vm_only_clipping_and_printed_meta_factors(self):
        cfg=PaperConfig();dtype=torch.float64
        pre=(torch.tensor([1.,2.],dtype=dtype),);post=(torch.tensor([3.,4.],dtype=dtype),)
        outer=tuple(g*1.1*2. for g in post)
        phi,eta=meta_jacobian_scales(pre,tuple(g*3 for g in outer),outer,.5,cfg)
        self.assertAlmostEqual(float(phi),cfg.intrinsic_lr*.5*3*1.1*2*11,places=14)
        self.assertAlmostEqual(float(eta),cfg.embedding_lr*.5*1.1*2*11,places=14)
        active=clipping_gate(torch.tensor(1.3),torch.tensor(2.),cfg.clip)
        phi,eta=meta_jacobian_scales(pre,tuple(g*active for g in post),tuple(g*active for g in post),.5,cfg)
        self.assertEqual(float(phi),0.);self.assertEqual(float(eta),0.)
        self.assertTrue(bool(clipping_gate(torch.tensor(.7),torch.tensor(2.),cfg.clip)))
        self.assertFalse(hasattr(cfg,'entropy_coef'))
        loss=clipped_loss(torch.log(torch.tensor([1.5,.5])),torch.zeros(2),torch.tensor([1.,-1.]),.2)
        self.assertAlmostEqual(float(loss),-.2,places=6)

    def test_interval_training_fixed_Q_and_chronological_history(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            learner=PaperLearner(3,PaperConfig())
            modules=dict(embedding=learner.task.embedding,selector=learner.task.score,actor=learner.actor,critic=learner.critic,intrinsic=learner.intrinsic)
            before={name:torch.cat([p.detach().flatten().clone() for p in module.parameters()]) for name,module in modules.items()}
            prefix_calls=[];original=learner.history_token
            def observe(*args):prefix_calls.append(float(args[2]));return original(*args)
            with patch.object(learner,'history_token',side_effect=observe):result=run_episode(env,learner,32,True)
            self.assertEqual(result['assignments'],12)
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),learner.updates)
            for update in result['updates']:
                n=update['collected_transitions']
                self.assertEqual(sorted(update['sampled_indices']),list(range(n)))
            for name,module in modules.items():
                change=float((torch.cat([p.detach().flatten() for p in module.parameters()])-before[name]).abs().max())
                self.assertEqual(change==0,name=='selector',name)
            self.assertTrue(prefix_calls)
            self.assertEqual(learner.buffer,[])
            self.assertTrue(all(s=='Finished' for s in env.task_state))
            self.assertAlmostEqual(env.total_energy,energy_from_records(env.records,env.hosts),places=8)
            self.assertGreater(result['fusion']['outside_unit_interval'],0)
            for vm in range(env.num_vms):
                gids=sorted([g for g,p in env.task_placement.items() if p[0]==vm],key=lambda g:env.task_start_time[g])
                for left,right in zip(gids,gids[1:]):self.assertLessEqual(env.task_end_time[left],env.task_start_time[right]+1e-10)

    def test_shuffled_loss_samples_reconstruct_ordered_prefixes(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            env.reset();learner=PaperLearner(3,PaperConfig())
            decision=learner.prepare(env)
            action,lp,value=learner.act(decision);reward,_,info=env.paper_step(action)
            successor=learner.prepare(env)
            learner.remember(decision,action,lp,value,reward,False,info,successor)
            action,lp,value=learner.act(successor);reward,_,info=env.paper_step(action)
            env.finish_phase_and_advance();nxt=learner.prepare(env)
            learner.remember(successor,action,lp,value,reward,False,info,nxt)
            seen=[];original=learner.history_token
            def observe(*a):seen.append(float(a[2]));return original(*a)
            with patch.object(learner,'history_token',side_effect=observe):learner._sample(1,learner.value(nxt))
            self.assertEqual(seen,[r['step_reward'] for r in learner.buffer])
            before=learner._sample(0,learner.value(nxt))[2].detach().clone()
            with torch.no_grad():
                for parameter in learner.task.score.parameters():parameter.add_(20.)
            after=learner._sample(0,learner.value(nxt))[2].detach()
            torch.testing.assert_close(before,after)

    def test_full_interval_outer_score_matches_directional_difference(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            env.reset();learner=PaperLearner(3,PaperConfig());decision=learner.prepare(env)
            for i in range(2):
                action,lp,value=learner.act(decision);reward,_,info=env.paper_step(action)
                if not env.can_dispatch():env.finish_phase_and_advance()
                nxt=learner.prepare(env)
                learner.remember(decision,action,lp,value,reward,False,info,nxt);decision=nxt
            bootstrap=learner.value(decision)
            score,cost_score,ratios,gates=learner._outer_scores(bootstrap)
            coefficients=(ratios*gates*learner.tensor(learner._advantages(bootstrap))).detach()
            parameter=list(learner.actor.parameters())[-1];original=parameter.detach().clone();epsilon=1e-5
            def objective(sign):
                with torch.no_grad():
                    parameter.copy_(original);parameter[0]+=sign*epsilon
                    logs=[]
                    for record in learner.buffer:
                        obs=learner.tensor(record['decision'].obs)
                        from algorithms.irws_restored.networks import masked_distribution
                        dist=masked_distribution(learner.actor(obs),record['decision'].mask)
                        logs.append(dist.log_prob(torch.tensor(record['action'])))
                    return float((coefficients*torch.stack(logs)).sum())
            numeric=(objective(1)-objective(-1))/(2*epsilon)
            with torch.no_grad():parameter.copy_(original)
            self.assertAlmostEqual(float(score[-1][0]),numeric,places=8)
            coefficients*=learner.tensor([r['info']['outer_cost'] for r in learner.buffer])
            numeric=(objective(1)-objective(-1))/(2*epsilon)
            with torch.no_grad():parameter.copy_(original)
            self.assertAlmostEqual(float(cost_score[-1][0]),numeric,places=8)

    def test_checkpoint_and_training_outputs_are_independent(self):
        with synthetic_env(self.path,PaperCloudEnv) as env:
            cfg=PaperConfig();kwargs=env_kwargs(self.scene,32)
            def construct(**_kwargs):
                return PaperCloudEnv(dax_paths=['synthetic.xml'],workflows_per_episode=2,deadline_mode='none',
                    max_ready_tasks=1,num_hosts=2,vms_per_host=(2,1),pc_tiers=(1.,2.),bw_tiers=(1.,2.),
                    random_seed=32,horizon=1e9,normalization_mode='f18_shared_v1',
                    normalization_config=str(self.path),source_train_scene='SS_T')
            with patch('algorithms.irws_restored.runner.PaperCloudEnv',side_effect=construct),redirect_stdout(io.StringIO()):
                learner,out=train(self.scene,cfg,kwargs,self.root/'models')
                loaded,obj=load_checkpoint(out/'best.pth',self.scene,restore_training=True)
                rows,_=evaluate(loaded,self.scene,[2],kwargs=kwargs)
                expected=run_episode(env,learner,2,False)['metrics']
            self.assertEqual(rows[0],dict(seed=2,**expected))
            self.assertEqual(obj['model_version'],MODEL_VERSION)
            self.assertTrue(all(obj['optimizers'][name]['state'] for name in ('actor','critic','intrinsic','task')))
            measurement=json.loads((out/'logs/training_measurements.jsonl').read_text().splitlines()[-1])
            self.assertEqual(set(measurement['optimizer_steps'].values()),{learner.updates})
            self.assertEqual(set(measurement['optimizer_sample_presentations'].values()),{learner.updates})
            self.assertIn('irws_restored',artifact_name(self.scene,'train'))
            self.assertIn(MODEL_VERSION,str(output_dir(self.scene,'ckpts')))
            for previous_version in ('irws_cloud_energy_tardiness_v2','irws_restored_queue_v1'):
                wrong=dict(obj,model_version=previous_version);torch.save(wrong,self.root/'wrong.pth')
                with self.assertRaisesRegex(ValueError,'model/observation version mismatch'):
                    load_checkpoint(self.root/'wrong.pth',self.scene)
            wrong_scene=replace(self.scene,paper_options=dict(queue_capacity=11,slot_duration=1.))
            with self.assertRaises(ValueError):load_checkpoint(out/'best.pth',wrong_scene)
            from algorithms.irws_paper.runner import load_checkpoint as load_previous
            with self.assertRaises(ValueError):load_previous(out/'best.pth',self.scene)


if __name__=='__main__':unittest.main()
