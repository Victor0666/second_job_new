"""Reservation physics, graph/gradient correctness and paired GAT-PPO integration."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.two_stage_gat.config import (ARCHITECTURE, INPUT_ENCODING, MODEL_VERSION,
    TwoStageGATConfig, SOURCE_COMMIT, SELECTION_TAG)
from algorithms.two_stage_gat.environment import TwoStageGATEnv
from algorithms.two_stage_gat.networks import GraphState, TwoStageActorCritic
from algorithms.two_stage_gat.ppo import PPOAgent, Transition, compute_gae
from algorithms.two_stage_gat.runner import (env_kwargs, evaluate, load_checkpoint,
    objective_metadata, run_episode, seed_everything, train, artifact_name, evaluate_main)
from algorithms.two_stage_gat.paths import evaluation_csv, evaluation_directory
from algorithms.two_stage_gat.selection import (selection_key, selection_metadata,
    selection_protocol, selection_scenario, validation_selection)
from common.experiment_measurements import episode_metrics
from common.f22_instances import instance_identity
from common.scenarios import Scenario
from common.workflow_opt import energy_from_records, exec_time_components
from environments.irws import IRWSEnvRouteA


def graph(count=4, vms=3):
    rng = np.random.RandomState(count + vms)
    task = rng.uniform(.05, 1., (count, 6)).astype(np.float32)
    vm = rng.uniform(.05, 1., (count, vms, 4)).astype(np.float32)
    return GraphState(task, vm, np.asarray([(i, i+1) for i in range(count-1)], dtype=np.int64),
        np.asarray([True, True] + [False] * (count-2)), np.ones((count, vms), dtype=bool), tuple(range(count)))


def vector(module):
    return torch.cat([p.detach().cpu().flatten() for p in module.parameters()])


def validation_fixture(successes, energy=100., count=50):
    return dict(seed=600, wf_success_count=successes, wf_count=count,
                wf_success_rate=successes/count, total_energy=energy)


class SelectionTests(unittest.TestCase):
    def test_paper_workflow_threshold_and_probability_target_are_distinct(self):
        self.assertFalse(validation_selection([validation_fixture(45)])['feasible'])
        boundary = validation_selection([validation_fixture(46)])
        self.assertTrue(boundary['feasible'])
        self.assertEqual(boundary['workflow_success_rate'], .92)
        self.assertEqual(boundary['validation_ESP'], 1.)
        rows = [validation_fixture(50)] * 99 + [validation_fixture(45)]
        self.assertTrue(validation_selection(rows)['feasible'])
        rows = [validation_fixture(50)] * 98 + [validation_fixture(45)] * 2
        result = validation_selection(rows)
        self.assertGreater(result['workflow_success_rate'], .92)
        self.assertEqual(result['validation_ESP'], .98)
        self.assertFalse(result['feasible'])

    def test_feasible_energy_order_and_infeasible_success_fallback(self):
        def candidate(success, energy):
            return validation_selection([validation_fixture(success, energy)])
        self.assertLess(selection_key(candidate(46,200)), selection_key(candidate(45,1)))
        self.assertLess(selection_key(candidate(46,200)), selection_key(candidate(50,250)))
        self.assertLess(selection_key(candidate(49,200)), selection_key(candidate(46,200)))
        self.assertLess(selection_key(candidate(45,200)), selection_key(candidate(44,1)))
        self.assertLess(selection_key(candidate(45,100)), selection_key(candidate(45,200)))
        self.assertEqual(selection_key(candidate(46,200)), selection_key(candidate(46,200)))

    def test_invalid_or_inconsistent_validation_is_rejected(self):
        for rows in ([], [validation_fixture(51)], [validation_fixture(-1)],
                     [validation_fixture(46,float('nan'))], [validation_fixture(46,-1)],
                     [dict(validation_fixture(46),wf_success_rate=1.)],
                     [dict(validation_fixture(46),wf_count=0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validation_selection(rows)

    def test_selector_is_algorithm_specific_and_paths_are_isolated(self):
        scene = Scenario()
        self.assertEqual(scene.protocol().selection, 'minimum_validation_total_energy')
        protocol = selection_protocol(scene)
        for key in ('train','validation','test','episodes'):
            self.assertEqual(protocol[key], scene.protocol().metadata()[key])
        self.assertEqual(protocol['selection'], SELECTION_TAG)
        self.assertRegex(scene.model_name('two_stage_gat'), r'^SS_T_m1_c[0-9a-f]{8}$')
        self.assertNotIn(SELECTION_TAG, scene.model_name('mahdrl'))
        self.assertEqual(scene.describe('two_stage_gat')['checkpoint_selection'], selection_metadata())
        adapted = selection_scenario(scene)
        self.assertEqual(adapted.reproducibility_metadata()['data_split']['selection'], SELECTION_TAG)
        self.assertEqual(scene.reproducibility_metadata()['data_split']['selection'],
                         'minimum_validation_total_energy')


class OutputPathTests(unittest.TestCase):
    def test_short_training_and_flat_unique_csv_paths_for_all_27_scenes(self):
        training_paths, csv_paths = set(), set()
        for resource in 'SML':
            for deadline in ('Tight','Medium','Loose'):
                training = Scenario(mode='train', resource=resource, deadline=deadline)
                checkpoint = training.checkpoint_dir('two_stage_gat')
                training_paths.add(checkpoint)
                self.assertEqual(len(checkpoint.name),17)
                self.assertEqual(checkpoint.parent.name,'2sd_gat_v4')
                self.assertEqual(training.log_dir('two_stage_gat').name,checkpoint.name)
                for task in 'SML':
                    scene = Scenario(mode='eval', task=task, resource=resource, deadline=deadline,
                        train_task='S',train_resource=resource,train_deadline=deadline)
                    self.assertEqual(scene.checkpoint_dir('two_stage_gat'),checkpoint)
                    path = evaluation_csv(scene)
                    csv_paths.add(path)
                    self.assertEqual(path.parent.name,'2sd_gat_v4')
                    self.assertEqual(path.parent.parent.name,'run')
                    self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_',path.name)
                    self.assertIn('_m1_s700-729_',path.name)
                    self.assertLess(len(path.name),120)
                    self.assertEqual(scene.describe('two_stage_gat')['evaluation_csv'],str(path))
        self.assertEqual(len(training_paths),9)
        self.assertEqual(len(csv_paths),27)
        self.assertEqual(len({p.parent for p in csv_paths}),1)

    def test_paths_distinguish_source_seed_set_model_seed_and_configuration(self):
        scene = Scenario(task='M',deadline='Loose')
        alternatives = [scene, replace(scene,train_task='M'),replace(scene,model_seed=2),
            replace(scene,seed_spec='700,702,704'),replace(scene,seed_spec='701,703,705'),
            replace(scene,normalization_mode='f18_shared_v1')]
        self.assertEqual(len({evaluation_csv(s) for s in alternatives}),len(alternatives))
        self.assertEqual(evaluation_csv(scene),evaluation_csv(replace(scene,paper_options={'device':'cpu'})))
        self.assertEqual(evaluation_csv(scene),evaluation_csv(replace(scene,paper_options={'device':'cuda'})))
        with TemporaryDirectory(prefix='two_stage_gat_override_') as directory:
            override = replace(scene,checkpoint_override=directory)
            self.assertEqual(override.checkpoint_dir('two_stage_gat'),Path(directory).resolve())

    def test_evaluation_entry_writes_scene_columns_and_refuses_overwrite(self):
        scene = Scenario(task='M',deadline='Loose',seed_spec='700')
        row = dict(validation_fixture(50,123.),seed=700)
        candidate = validation_selection([row])
        with TemporaryDirectory(prefix='two_stage_gat_flat_') as directory:
            with patch('algorithms.two_stage_gat.paths.ROOT',Path(directory)), \
                 patch('algorithms.two_stage_gat.runner.parse_scenario',return_value=scene), \
                 patch('algorithms.two_stage_gat.runner.load_checkpoint',
                       return_value=(object(),dict(validation_selection=candidate))), \
                 patch('algorithms.two_stage_gat.runner.evaluate',
                       return_value=([row],{'total_energy':{'mean':123.}})) as evaluator:
                evaluate_main()
                path = evaluation_csv(scene)
                with path.open(encoding='utf-8',newline='') as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(saved[0]['target_scene'],'MS_Loose')
                self.assertEqual(saved[0]['source_scene'],'SS_Loose')
                self.assertEqual(saved[0]['model_version'],MODEL_VERSION)
                self.assertEqual(saved[0]['seed'],'700')
                self.assertEqual(evaluator.call_args.kwargs['directory'],evaluation_directory(scene)/'measurements')
                self.assertTrue((path.parent/(artifact_name(scene,'summary')+'.json')).is_file())
                with self.assertRaises(FileExistsError): evaluate_main()
                self.assertEqual(evaluator.call_count,1)


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='two_stage_gat_physics_')
        self.addCleanup(self.temp.cleanup)
        self.dax = Path(self.temp.name) / 'Fork_4.xml'
        self.dax.write_text('<adag xmlns="http://pegasus.isi.edu/schema/DAX">'
            '<job id="a"/><job id="b"/><job id="c"/><job id="d"/>'
            '<child ref="c"><parent ref="a"/><parent ref="b"/></child></adag>', encoding='utf-8')

    def kwargs(self, **overrides):
        result = dict(dax_paths=[str(self.dax)], horizon=1e9, workflows_per_episode=1,
            arrival_lambda=.03, num_hosts=2, vms_per_host=(2, 1), pc_tiers=(1., 8.),
            bw_tiers=(1000., 2000.), deadline_mode='none', normalization_mode='legacy',
            max_ready_tasks=1, random_seed=2)
        result.update(overrides)
        return result

    def test_busy_vm_remains_legal_and_parent_completion_is_required(self):
        env = TwoStageGATEnv(**self.kwargs()); env.reset(seed=2)
        original = env.graph_state()
        self.assertTrue(original.action_mask[1, 0])
        self.assertFalse(original.task_mask[2])
        _, _, first = env.pair_step(0)
        self.assertFalse(first['queued'])
        _, _, second = env.pair_step(env.num_vms)
        self.assertTrue(second['queued'])
        self.assertEqual(env.task_state[1], 'Queued')
        self.assertAlmostEqual(env.task_start_time[1], env.task_end_time[0])
        self.assertEqual(env.current_time, first['dispatch_time'])
        self.assertFalse(env.graph_state().task_mask[2])
        self.assertIn((0, 1), set(map(tuple, env.graph_state().dependencies)))
        snapshot = (env.assignment_count, env.total_energy, list(env.task_state), len(env.records))
        with self.assertRaises(ValueError): env.pair_step(2 * env.num_vms)
        self.assertEqual(snapshot, (env.assignment_count, env.total_energy, list(env.task_state), len(env.records)))
        env.pair_step(3 * env.num_vms)
        self.assertEqual(env.task_state[0], 'Finished')
        self.assertEqual(env.task_state[1], 'Finished')
        self.assertTrue(env.graph_state().task_mask[2])
        self.assertGreaterEqual(env.current_time, env.task_end_time[1])
        _, done, final = env.pair_step(2 * env.num_vms)
        self.assertTrue(done)
        self.assertTrue(final['queued'])
        self.assertEqual(env.assignment_count, 4)
        self.assertTrue(all(s == 'Finished' for s in env.task_state))
        self.assertTrue(original.task_mask[0])
        for bad in (-1, 1000, 1.5):
            with self.assertRaises((ValueError, RuntimeError)): env.pair_step(bad)

    def test_arrival_during_future_reservations_is_observed_on_time(self):
        with patch('environments.irws.poisson_arrival_times', return_value=np.array([0., 1.])):
            env = TwoStageGATEnv(**self.kwargs(workflows_per_episode=2))
            env.reset(seed=2)
            first = env.graph_state()
            self.assertEqual(len(first.task_features), 4)
            for gid in (0, 1, 3): env.pair_step(gid * env.num_vms)
            self.assertEqual(env.current_time, 1.)
            self.assertEqual(len(env.workflows), 2)
            self.assertEqual(len(env.graph_state().task_features), 8)
            self.assertEqual(len(first.task_features), 4)
            _, _, info = env.pair_step(4 * env.num_vms)
            self.assertTrue(info['queued'])

    def test_public_energy_and_communication_on_full_reserved_trace(self):
        env = TwoStageGATEnv(**self.kwargs(workflows_per_episode=2)); env.reset(seed=2)
        rewards = 0.; delay = 0.; increments = 0.
        while not env.done_flag:
            gid = int(np.flatnonzero(env.graph_state().task_mask)[0])
            reward, _, info = env.pair_step(gid * env.num_vms)
            rewards += reward; delay += info['r_delay']; increments += info['delta_energy_total']
        self.assertGreater(env.queued_assignments, 0)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts), places=8)
        self.assertAlmostEqual(increments, env.total_energy, places=8)
        self.assertAlmostEqual(rewards, -.25 * .001 * env.total_energy + .75 * delay, places=9)
        for gid, (vm_id, host_id) in env.task_placement.items():
            vm = env.vms[vm_id]
            duration = exec_time_components(env._task_obj_from_gid(gid), vm.pc, vm.bw,
                parents_loc=env._build_parents_loc(gid), child_vm_id=vm_id, child_host_id=host_id,
                zero_if_same_vm=True, zero_if_same_host=False)[-1]
            self.assertAlmostEqual(env.task_end_time[gid]-env.task_start_time[gid], duration, places=8)
            base = gid - env.task_meta[gid].local_id
            self.assertTrue(all(env.task_dispatch_time[gid] >= env.task_end_time[base+p]-1e-8
                for p in env.task_parents[gid]))
        order = sorted(env.task_placement, key=lambda g:env.task_start_time[g])
        self.assertTrue(all(env.task_end_time[a] <= env.task_start_time[b]+1e-8 for a,b in zip(order,order[1:])))

    def test_idle_assignments_match_mahdrl_execution_and_energy(self):
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
        kwargs = self.kwargs()
        env = TwoStageGATEnv(**kwargs); ref = CloudWorkflowEnv_VMAgents(**kwargs)
        env.reset(seed=2); ref.reset(seed=2)
        for gid, vm_id in ((0, 0), (1, 1), (3, 2)):
            ref._assign_task_to_specific_vm(gid, vm_id)
            env.pair_step(gid * env.num_vms + vm_id)
        ref._advance_until_decision_energy_only()
        self.assertAlmostEqual(env.current_time, ref.current_time, places=8)
        self.assertIn(2, ref.ready_task_ids)
        vm_id = int(np.flatnonzero(ref.vm_available_at <= ref.current_time+1e-12)[0])
        ref._assign_task_to_specific_vm(2, vm_id)
        ref._advance_until_decision_energy_only()
        env.pair_step(2 * env.num_vms + vm_id)
        np.testing.assert_allclose(env.task_end_time, ref.task_end_time, atol=1e-8, rtol=0)
        self.assertAlmostEqual(env.total_energy, ref.total_energy, places=7)

    def test_original_proxy_and_all_vm_forecast_with_signed_deadline_slot(self):
        env = TwoStageGATEnv(**self.kwargs()); env.reset(seed=2)
        t_ref = 10.
        env.state_normalizer = SimpleNamespace(scales={'T_ref':t_ref})
        before = env.graph_state()
        env.vm_available_at[:] = env.current_time + np.array([1000., 1., 100.])
        state = env.graph_state()
        self.assertEqual(state.vm_features.shape, (4, 3, 4))
        np.testing.assert_array_equal(state.vm_features[:, :, 1:], before.vm_features[:, :, 1:])
        np.testing.assert_allclose(state.vm_features[:, :, 0],
            np.broadcast_to(env.vm_available_at/t_ref, (4,3)),rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(state.vm_features[:, :, 1], 1.)
        self.assertGreater(state.task_features[0, 2], before.task_features[0, 2])
        self.assertTrue(state.action_mask[0].all())
        expected = min(max(env.current_time, env.vm_available_at[j]) + exec_time_components(
            env._task_obj_from_gid(0), v.pc, v.bw, parents_loc={}, child_vm_id=j,
            child_host_id=v.host_id)[-1] for j,v in env.vms.items())
        self.assertAlmostEqual(state.task_features[0, 2], expected/t_ref, places=3)
        self.assertTrue((state.task_features[:, 5] == 1).all())
        vm = env.vms[0]; host = env.hosts[vm.host_id]
        expected_proxy = env.task_mi[0] * (host.power(1)-host.power(0)) / host.total_pc * .001
        self.assertAlmostEqual(state.vm_features[0, 0, 3], expected_proxy, places=5)
        expected_duration = exec_time_components(env._task_obj_from_gid(0), vm.pc, vm.bw,
            parents_loc={}, child_vm_id=0, child_host_id=vm.host_id)[-1]
        self.assertAlmostEqual(state.vm_features[0, 0, 2], expected_duration/t_ref, places=3)
        env.deadline_mode = 'cache_fcfs'  # Only exercise encoding with explicit finite fixture deadlines.
        env.task_baseline_finish[0] = expected + 10
        a = env.graph_state().task_features[0, 4]
        env.task_baseline_finish[0] = expected - 10
        b = env.graph_state().task_features[0, 4]
        self.assertAlmostEqual(a, 10./t_ref, places=4); self.assertAlmostEqual(b, -10./t_ref, places=4)


class NetworkPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self): torch.manual_seed(73)

    def test_batch_and_single_state_contexts_are_equivalent(self):
        model = TwoStageActorCritic()
        states = [graph(4, 3), graph(6, 2)]
        actions = [1, 3]
        with torch.no_grad():
            batched = model(states, actions)
            for i, state in enumerate(states):
                single = model([state], [actions[i]])
                for a,b in zip(batched, single):
                    torch.testing.assert_close(a[i], b[0], rtol=2e-5, atol=2e-6)
            reverse = model(states[::-1], actions[::-1])
            torch.testing.assert_close(batched[1], reverse[1].flip(0), rtol=2e-5, atol=2e-6)

    def test_topology_changes_representation_and_no_parameter_sharing_with_critic(self):
        model = TwoStageActorCritic(); state = graph()
        changed = replace(state, dependencies=np.asarray([(0,2),(1,3)], dtype=np.int64))
        a = model([state], [1]); b = model([changed], [1])
        self.assertFalse(torch.allclose(a[3], b[3], rtol=1e-6, atol=1e-8))
        actor_ids = {id(p) for p in model.actor.parameters()}
        self.assertFalse(actor_ids & {id(p) for p in model.critic.parameters()})
        for module, widths in ((model.actor.vm_encoder, (4,64,64,8)),
                               (model.critic.vm_encoder, (4,64,64,8)),
                               (model.actor.decoder, (24,64,64,1)),
                               (model.critic.network, (16,64,64,1))):
            layers = [m for m in module if isinstance(m, torch.nn.Linear)]
            self.assertEqual(tuple([layers[0].in_features] + [m.out_features for m in layers]), widths)
        self.assertNotIn('task_decoder', model.actor._modules)
        self.assertNotIn('vm_decoder', model.actor._modules)
        self.assertEqual(model.actor.task_encoder.convs[0].heads, 4)
        self.assertFalse(model.actor.task_encoder.convs[0].concat)

    def test_both_stages_use_one_decoder_and_backpropagate_through_it(self):
        model = TwoStageActorCritic(); outputs = []
        def capture(module, inputs, output):
            output.retain_grad()
            outputs.append(output)
        hook = model.actor.decoder.register_forward_hook(capture)
        try:
            _, logp, _, _ = model([graph()], [5])
            logp.sum().backward()
        finally:
            hook.remove()
        self.assertEqual([tuple(x.shape) for x in outputs], [(4,1),(3,1)])
        for output in outputs:
            self.assertTrue(torch.isfinite(output.grad).all())
            self.assertGreater(float(output.grad.abs().sum()), 0.)

    def test_joint_logprob_is_task_plus_selected_vm_and_masks_are_exact(self):
        from algorithms.two_stage_gat.networks import _batch
        model = TwoStageActorCritic(); state = graph()
        data = _batch([state], torch.device('cpu'))
        h, pool, avg_vm = model.actor.contexts(data)
        logits = model.actor.decoder(torch.cat((h, pool.expand(len(h),-1), avg_vm.expand(len(h),-1)),dim=1)).flatten()
        mask = torch.tensor(state.task_mask)
        task_dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask,-torch.inf))
        chosen_task = 1; chosen_vm = 2
        encoded = model.actor.vm_encoder(torch.tensor(state.vm_features[chosen_task]))
        vm_logits = model.actor.decoder(torch.cat((encoded,pool.expand(len(encoded),-1),
            encoded.mean(0,keepdim=True).expand(len(encoded),-1)),dim=1)).flatten()
        vm_dist = torch.distributions.Categorical(logits=vm_logits)
        _, logp, entropy, _ = model([state], [chosen_task*3+chosen_vm])
        torch.testing.assert_close(logp[0],task_dist.log_prob(torch.tensor(1))+vm_dist.log_prob(torch.tensor(2)))
        torch.testing.assert_close(entropy[0],task_dist.entropy()+vm_dist.entropy())
        bad = replace(state,task_mask=np.zeros(4,dtype=bool))
        with self.assertRaises(ValueError): model([bad])
        with self.assertRaises(ValueError): model([state],[6])
        old_vm3 = replace(state, vm_features=state.vm_features[:, :, 1:].copy())
        with self.assertRaises(ValueError): model([old_vm3])

    def test_gae_keeps_vector_paths_and_terminal_boundaries_separate(self):
        def r(reward, done, nxt, env): return Transition(None,0,0.,0.,reward,done,nxt,env)
        records = [r(1,False,2,0),r(10,True,99,1),r(3,True,99,0),r(20,False,4,1)]
        adv, ret = compute_gae(records,.5,1.)
        np.testing.assert_allclose(adv,[3.5,10,3,22]);np.testing.assert_array_equal(adv,ret)

    def test_ppo_updates_shared_decoder_encoders_and_critic_without_resampling_stored_actions(self):
        cfg = TwoStageGATConfig(num_steps=2)
        agent = PPOAgent(cfg); states = [graph(4,3),graph(5,2)]
        groups = dict(gat=agent.network.actor.task_encoder,vm=agent.network.actor.vm_encoder,
            decoder=agent.network.actor.decoder,critic=agent.network.critic)
        before = {n:vector(m).clone() for n,m in groups.items()}
        records = []
        for i in range(8):
            state = states[i%2]; action,logp,val=agent.act(state)
            records.append(Transition(state,action,logp,val,-.1*(i+1),i%4==3,.2,i%2))
        result = agent.update(records)
        self.assertTrue(np.isfinite([result[k] for k in ('loss','entropy','grad_norm')]).all())
        for name,module in groups.items(): self.assertGreater(float((vector(module)-before[name]).abs().max()),0.,name)
        self.assertEqual(result['optimizer_steps'],16)
        self.assertEqual(result['sample_presentations'],32)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        flags=patch.dict('os.environ',EXPERIMENT_METRICS='1')
        flags.start();self.addCleanup(flags.stop)

    def test_real_dax_train_save_load_partition_and_evaluation_isolation(self):
        with TemporaryDirectory(prefix='two_stage_gat_runner_') as tmp:
            scene = Scenario(mode='train',episodes=8)
            kwargs = env_kwargs(scene,0); kwargs['workflows_per_episode']=3
            cfg = TwoStageGATConfig(device='cpu')
            agent,directory = train(scene,cfg,kwargs,Path(tmp)/'models')
            summary = json.loads((directory/'logs/training_summary.json').read_text())
            self.assertEqual(sorted(summary['seeds_started']),list(range(8)))
            self.assertEqual(summary['completed_episodes'],8)
            self.assertGreater(summary['budget']['assignments'],cfg.batch_size)
            self.assertGreater(agent.updates,1)
            self.assertEqual(agent.steps,summary['budget']['assignments'])
            self.assertEqual(agent.updates,summary['budget']['planned_updates'])
            rows = [json.loads(s) for s in (directory/'logs'/(artifact_name(scene,'train')+'.jsonl')).read_text().splitlines()]
            self.assertEqual(len(rows),8)
            self.assertEqual(rows[0]['policy_updates_at_validation'],0)
            self.assertEqual(rows[-1]['policy_updates_at_validation'],agent.updates)
            loaded,obj = load_checkpoint(directory/'final.pth',scene)
            self.assertEqual(obj['source_commit'],SOURCE_COMMIT)
            self.assertEqual(obj['model_version'],MODEL_VERSION)
            self.assertEqual(obj['architecture'],ARCHITECTURE)
            self.assertEqual(obj['policy_input'],INPUT_ENCODING)
            self.assertEqual(obj['checkpoint_selection'], selection_metadata())
            self.assertEqual(obj['protocol']['selection'], SELECTION_TAG)
            self.assertEqual(summary['best_validation_selection'],
                min((dict(r['validation_selection'], completed_episodes=r['episode']+1)
                     for r in rows), key=selection_key))
            best_agent, best_obj = load_checkpoint(directory/'best.pth',scene)
            self.assertEqual(best_obj['validation_selection'],
                             rows[best_obj['completed_episodes']-1]['validation_selection'])
            before = deepcopy(agent.network.state_dict()); steps=agent.steps; updates=agent.updates
            rng = (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            a,_=evaluate(agent,scene,[700],kwargs=kwargs,directory=Path(tmp)/'measurements',checkpoint_directory=directory)
            b,_=evaluate(loaded,scene,[700],kwargs=kwargs)
            self.assertEqual(a,b)
            self.assertEqual((agent.steps,agent.updates),(steps,updates))
            self.assertEqual(random.getstate(),rng[0]); np.testing.assert_array_equal(np.random.get_state()[1],rng[1][1])
            torch.testing.assert_close(torch.get_rng_state(),rng[2],rtol=0,atol=0)
            for key,value in before.items(): torch.testing.assert_close(agent.network.state_dict()[key],value,rtol=0,atol=0)
            self.assertTrue((Path(tmp)/'measurements/episodes/seed700.json').is_file())
            measurement=json.loads((Path(tmp)/'measurements/episodes/seed700.json').read_text())
            self.assertEqual(measurement['provenance']['data_split']['selection'],SELECTION_TAG)
            with self.assertRaises(ValueError): evaluate(agent,scene,[0],kwargs=kwargs)
            with self.assertRaises(ValueError): load_checkpoint(directory/'final.pth',replace(scene,model_seed=2))
            corrupt=deepcopy(obj);corrupt['scheduling_protocol']='idle_only'
            torch.save(corrupt,directory/'wrong.pth')
            with self.assertRaises(ValueError): load_checkpoint(directory/'wrong.pth',scene)
            old=deepcopy(obj);old['model_version']='two_stage_gat_source_aae1397_v1'
            old.pop('checkpoint_selection')
            torch.save(old,directory/'old_energy_only.pth')
            with self.assertRaises(ValueError): load_checkpoint(directory/'old_energy_only.pth',scene)
            legacy = deepcopy(obj)
            legacy.update(model_version='two_stage_gat_source_aae1397_selection_v2',
                          policy_input='task6_vm4_signed_slack_shared_v1')
            legacy.pop('architecture')
            torch.save(legacy,directory/'old_vm4_independent_decoders.pth')
            with self.assertRaises(ValueError): load_checkpoint(directory/'old_vm4_independent_decoders.pth',scene)
            vm3 = deepcopy(obj)
            vm3.update(model_version='two_stage_gat_paper_v3',
                       policy_input='task6_vm3_signed_slack_shared_v2')
            torch.save(vm3,directory/'old_vm3_shared_decoder.pth')
            with self.assertRaises(ValueError): load_checkpoint(directory/'old_vm3_shared_decoder.pth',scene)
            with self.assertRaises(FileExistsError): train(scene,cfg,kwargs,directory)

    def test_checkpoint_updates_follow_feasibility_not_only_energy(self):
        candidates = [validation_fixture(s,e) for s,e in
                      ((45,100.),(46,200.),(44,50.),(47,180.),(50,250.))]
        evaluations = [([row], {'total_energy':{'mean':row['total_energy']}})
                       for row in candidates]
        with TemporaryDirectory(prefix='two_stage_gat_selection_') as temporary:
            scene=Scenario(mode='train',episodes=5)
            kwargs=env_kwargs(scene,0);kwargs['workflows_per_episode']=1
            with patch('algorithms.two_stage_gat.runner.evaluate',side_effect=evaluations):
                _, directory=train(scene,TwoStageGATConfig(device='cpu'),kwargs,Path(temporary)/'models')
            _, best=load_checkpoint(directory/'best.pth',scene)
            _, final=load_checkpoint(directory/'final.pth',scene)
            self.assertEqual(best['completed_episodes'],4)
            self.assertEqual(best['validation_selection']['mean_total_energy'],180.)
            self.assertEqual(best['validation_selection']['workflow_success_rate'],.94)
            self.assertTrue(best['validation_selection']['feasible'])
            self.assertEqual(final['validation_selection']['mean_total_energy'],250.)
            self.assertEqual(final['best_validation_selection'],best['best_validation_selection'])
            metadata=json.loads((directory/'data_split.json').read_text())
            self.assertEqual(metadata['data_split']['selection'],SELECTION_TAG)

    def test_paired_instance_matches_public_environment_despite_different_queues(self):
        scene=Scenario();kwargs=env_kwargs(scene,700);kwargs['workflows_per_episode']=2
        env=TwoStageGATEnv(**kwargs);ref=IRWSEnvRouteA(**kwargs)
        env.reset(seed=700);ref.reset(seed=700)
        while not env.done_flag:
            gid=int(np.flatnonzero(env.graph_state().task_mask)[0]);env.pair_step(gid*env.num_vms)
        while not ref.done_flag:
            ref.select_task(ref.ready_task_ids[0])
            ref.step(int(np.flatnonzero(ref.vm_available_at<=ref.current_time+1e-12)[0]))
        self.assertEqual(instance_identity(env,scene,700),instance_identity(ref,scene,700))
        np.testing.assert_allclose(env.task_baseline_finish,ref.task_baseline_finish,rtol=0,atol=0)
        self.assertGreater(env.queued_assignments,0)


if __name__ == '__main__': unittest.main()
