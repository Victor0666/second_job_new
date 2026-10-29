"""Idle dispatch, completed-parent release and independent 92% checkpoint evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from algorithms.drl_gnn_fast_idle import scenario as idle_scenario

from algorithms.drl_gnn_fast import config as fast_config, selection as fast_selection
from algorithms.drl_gnn_fast.environment import FastDRLGNNEnv
from algorithms.drl_gnn_fast.ppo import PPOAgent
from algorithms.drl_gnn_fast.scenario import FastScenario
from algorithms.drl_gnn_fast_idle import config, runner, selection
from algorithms.drl_gnn_fast_idle.environment import IdleDRLGNNEnv
from algorithms.drl_gnn_fast_idle.scenario import IdleScenario, parse_scenario
from algorithms.mahdrl_as_fast_vaaware import selection as va_selection
from common.scenarios import Scenario
from common.workflow_opt import energy_from_records


def validation_row(success, energy, count=50, seed=600):
    return dict(seed=seed, wf_count=count, wf_success_count=success,
                wf_success_rate=success/count, total_energy=energy)


class IdleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = TemporaryDirectory(prefix='drl_gnn_fast_idle_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dax = self.root/'Join_3.xml'
        self.dax.write_text('<adag xmlns="http://pegasus.isi.edu/schema/DAX">'
            '<job id="a" runtime="100"/><job id="b" runtime="300"/>'
            '<job id="c" runtime="100"/>'
            '<child ref="c"><parent ref="a"/><parent ref="b"/></child></adag>', encoding='utf-8')

    def kwargs(self, **changes):
        result = dict(dax_paths=[str(self.dax)], horizon=1e9, workflows_per_episode=1,
            arrival_lambda=.03, num_hosts=1, vms_per_host=(3,),
            pc_tiers=(1., 2., 8.), bw_tiers=(1000., 2000., 4000.),
            deadline_mode='none', max_ready_tasks='auto', normalize=False, random_seed=0,
            **Scenario(mode='train').normalization_kwargs(), **config.COMMON_REWARD)
        result.update(changes)
        return result

    def drive(self, env, agent=None):
        reward_sum = 0.
        while not env.done_flag:
            state = env.graph_state()
            self.assertTrue(state.action_mask.any())
            idle = env.vm_available_at <= env.current_time + 1e-12
            self.assertFalse(state.action_mask[:, ~idle].any())
            self.assertFalse(state.action_mask[[0, -1]].any())
            for row in np.flatnonzero(state.action_mask.any(axis=1)):
                gid = int(row)-1
                base = gid-env.task_meta[gid].local_id
                self.assertTrue(all(env.task_state[base+p] == 'Finished' for p in env.task_parents[gid]))
            action = agent.select_action(state) if agent else int(np.flatnonzero(state.action_mask)[-1])
            self.assertTrue(state.action_mask.flat[action])
            dispatched = env.current_time
            reward, _, info = env.pair_step(action)
            gid = info['task_gid']
            self.assertAlmostEqual(env.task_start_time[gid], dispatched, places=10)
            self.assertFalse(info['queued'])
            self.assertNotIn('Queued', env.task_state)
            self.assertAlmostEqual(reward, -.25*.001*info['delta_energy_total']+.75*info['r_delay'])
            reward_sum += reward
        self.assertEqual(env.assignment_count, len(env.task_meta))
        self.assertEqual(env.completed_workflows, env.workflows_per_episode)
        self.assertEqual(env.queued_assignments, 0)
        self.assertFalse(env.graph_state().action_mask.any())
        per_vm = {}
        for gid, (vm, _) in env.task_placement.items():
            per_vm.setdefault(vm, []).append((env.task_start_time[gid], env.task_end_time[gid]))
            base = gid-env.task_meta[gid].local_id
            for parent in env.task_parents[gid]:
                self.assertGreaterEqual(env.task_start_time[gid]+1e-12, env.task_end_time[base+parent])
        for intervals in per_vm.values():
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:]):
                self.assertGreaterEqual(current[0]+1e-12, previous[1])
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts,
            start_time=env.first_arrival_time, end_time=env.current_time), places=6)
        return reward_sum

    def test_busy_vms_have_zero_probability_and_rejected_actions_cannot_mutate(self):
        env = IdleDRLGNNEnv(**self.kwargs()); env.reset(seed=0)
        env.pair_step(env.num_vms)  # First root onto VM 0; the other root and VM 1/2 remain free.
        self.assertEqual(env.task_state[2], 'unReady')
        state = env.graph_state()
        self.assertFalse(state.action_mask[:, 0].any())
        self.assertTrue(state.action_mask[2, 1:].all())
        self.assertFalse(state.action_mask[3].any())
        agent = PPOAgent()
        with torch.no_grad():
            distribution = agent.network.actor_distribution(state)
        probabilities = distribution.probs.detach().cpu().numpy().reshape(state.action_mask.shape)
        self.assertTrue(np.all(probabilities[~state.action_mask] == 0.))
        self.assertTrue(np.all(probabilities[state.action_mask] > 0.))
        for _ in range(20):
            action, _, _ = agent.act(state)
            self.assertTrue(state.action_mask.flat[action])
        self.assertTrue(state.action_mask.flat[agent.select_action(state)])
        before = deepcopy((env.task_state, env.ready_task_ids, env.task_placement,
            env.task_start_time, env.task_end_time, env.total_energy, env.records, env.assignment_count))
        with self.assertRaisesRegex(ValueError, 'idle VM'):
            env.pair_step(2*env.num_vms)
        self.assertEqual(before, (env.task_state, env.ready_task_ids, env.task_placement,
            env.task_start_time, env.task_end_time, env.total_energy, env.records, env.assignment_count))
        with self.assertRaisesRegex(ValueError, 'real Ready task'):
            env.pair_step(3*env.num_vms+1)
        self.drive(env, agent)

    def test_join_waits_for_all_parents_even_with_an_idle_vm(self):
        env = IdleDRLGNNEnv(**self.kwargs()); env.reset(seed=0)
        env.pair_step(env.num_vms)
        events = []
        process = env._process_completions_at_time
        def observe(t):
            process(t)
            events.append((t, tuple(env.task_state), tuple(env.ready_task_ids)))
        with patch.object(env, '_process_completions_at_time', side_effect=observe):
            env.pair_step(2*env.num_vms+1)
        self.assertNotEqual(env.task_end_time[0], env.task_end_time[1])
        partial = [states for _, states, _ in events if states[:2].count('Finished') == 1]
        self.assertTrue(partial)
        self.assertTrue(all(states[2] == 'unReady' for states in partial))
        self.assertEqual(env.task_state[:2], ['Finished', 'Finished'])
        self.assertEqual(env.task_state[2], 'Ready')
        self.assertAlmostEqual(env.current_time, max(env.task_end_time[:2]))
        self.assertAlmostEqual(env.task_ready_time[2], env.current_time)
        self.drive(env)

    def test_all_busy_advances_and_simultaneous_arrival_completion_is_processed(self):
        env = IdleDRLGNNEnv(**self.kwargs(workflows_per_episode=2, vms_per_host=(2,),
            pc_tiers=(1.,), bw_tiers=(1000.,)))
        env.reset(seed=0)
        env.pair_step(env.num_vms)
        finish = env.task_end_time[0]
        env.arrival_times[1] = finish
        env.pair_step(2*env.num_vms+1)
        self.assertGreater(env.current_time, env.task_dispatch_time[1])
        self.assertEqual(env.next_arrival_idx, 2)
        self.assertTrue(env.graph_state().action_mask.any())
        self.drive(env)

    def test_illegal_types_and_stale_readiness_are_rejected_before_mutation(self):
        env = IdleDRLGNNEnv(**self.kwargs()); env.reset(seed=0)
        for action in (1.5, -1, 10_000, 0):
            with self.assertRaises(ValueError): env.pair_step(action)
        self.assertEqual(env.assignment_count, 0)
        # Even an internally corrupted Ready flag cannot bypass dependency checks.
        env.task_state[2] = 'Ready'; env.ready_task_ids.append(2)
        with self.assertRaisesRegex(ValueError, 'parents to be finished'):
            env.pair_step(3*env.num_vms)
        self.assertEqual(env.assignment_count, 0)

    def test_real_dax_seeds_resources_and_energy_reintegration(self):
        dax = Path(__file__).resolve().parents[1]/'data/dax/Montage_25.xml'
        for hosts, counts in ((1, (3,)), (2, (2, 1))):
            for seed in (0, 1):
                env = IdleDRLGNNEnv(**self.kwargs(dax_paths=[str(dax)], workflows_per_episode=3,
                    num_hosts=hosts, vms_per_host=counts))
                env.reset(seed=seed)
                self.drive(env)

    def test_cached_observation_fields_and_original_protocol_remain_unchanged(self):
        kw = self.kwargs()
        old, new = FastDRLGNNEnv(**kw), IdleDRLGNNEnv(**kw)
        old.reset(seed=0); new.reset(seed=0)
        old.pair_step(old.num_vms); new.pair_step(new.num_vms)
        a, b = old.graph_state(), new.graph_state()
        for field in ('task_features', 'vm_features', 'dependencies', 'compatibility_mask'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        self.assertTrue(a.action_mask[2, 0]); self.assertFalse(b.action_mask[2, 0])
        old.pair_step(2*old.num_vms+1)
        self.assertEqual(old.task_state[2], 'Ready')  # v5 still releases on parent dispatch.
        self.assertEqual(old.current_time, old.task_dispatch_time[1])
        self.assertIs(runner.PPOAgent, PPOAgent)
        self.assertIs(config.DRLGNNConfig, fast_config.DRLGNNConfig)
        self.assertEqual(config.COMMON_REWARD, fast_config.COMMON_REWARD)

    def test_selection_092_boundary_fallback_aggregate_and_ties(self):
        self.assertEqual(selection.selection_metadata(), va_selection.selection_metadata())
        self.assertFalse(selection.validation_selection([validation_row(45, 1.)])['feasible'])
        feasible = selection.validation_selection([validation_row(46, 100.)])
        self.assertTrue(feasible['feasible'])
        self.assertFalse(fast_selection.validation_selection([validation_row(46, 100.)])['feasible'])
        fallback = selection.validation_selection([validation_row(45, 1.)])
        self.assertLess(selection.selection_key(feasible), selection.selection_key(fallback))
        cheap = selection.validation_selection([validation_row(46, 80.)])
        higher = selection.validation_selection([validation_row(47, 80.)])
        self.assertLess(selection.selection_key(cheap), selection.selection_key(feasible))
        self.assertLess(selection.selection_key(higher), selection.selection_key(cheap))
        rows = [validation_row(0, 1., 1), validation_row(92, 3., 99, 601)]
        self.assertTrue(selection.validation_selection(rows)['feasible'])
        self.assertEqual(selection.validation_selection(rows), va_selection.validation_selection(rows))

    def test_paths_cli_and_metadata_are_isolated_from_v5(self):
        source = IdleScenario(mode='train', episodes=1)
        target = IdleScenario(mode='eval', task='L', resource='L', deadline='Loose',
            train_task='S', train_resource='S', train_deadline='Tight')
        self.assertEqual(source.checkpoint_dir(), target.checkpoint_dir())
        self.assertNotEqual(source.checkpoint_dir(), FastScenario(mode='train').checkpoint_dir())
        self.assertNotEqual(source.checkpoint_dir(), Scenario(mode='train').checkpoint_dir('drl_gnn'))
        self.assertIn('_v6_sr092_', source.model_name())
        self.assertEqual(source.checkpoint_dir().parent.name, config.OUTPUT_TAG)
        self.assertEqual(target.evaluation_dir.parent.name, '_details')
        self.assertEqual(target.evaluation_csv.parent.name, config.OUTPUT_TAG)
        self.assertEqual(target.normalization_kwargs()['source_train_scene'], 'SS_T')
        self.assertEqual(source.describe()['scheduling_protocol'], config.SCHEDULING_PROTOCOL)
        with patch('common.f22_cache.check_cache'), patch.object(IdleScenario, 'validate_calibration'):
            self.assertIsInstance(parse_scenario('train', ['--episodes', '1', '--device', 'cpu']), IdleScenario)

    def test_all_27_csv_names_are_unique_flat_and_training_names_are_short(self):
        csvs, dirs, checkpoints = set(), set(), set()
        for resource in 'SML':
            for deadline in ('Tight', 'Medium', 'Loose'):
                source = IdleScenario(mode='train', resource=resource, deadline=deadline)
                checkpoints.add(source.checkpoint_dir())
                self.assertLessEqual(len(source.model_name()), 32)
                for task in 'SML':
                    target = IdleScenario(mode='eval', task=task, resource=resource, deadline=deadline,
                        train_task='S', train_resource=resource, train_deadline=deadline)
                    self.assertEqual(target.checkpoint_dir(), source.checkpoint_dir())
                    self.assertEqual(target.evaluation_csv.parent, target.evaluation_root)
                    self.assertIn(f'_{target.target_tag}_from_{target.source_tag}_', target.evaluation_csv.name)
                    self.assertIn('_m1_s700-729_', target.evaluation_csv.name)
                    self.assertLessEqual(len(target.evaluation_csv.name), 100)
                    csvs.add(target.evaluation_csv); dirs.add(target.evaluation_dir)
        self.assertEqual(len(csvs), 27); self.assertEqual(len(dirs), 27)
        self.assertEqual(len(checkpoints), 9)
        a = IdleScenario(mode='eval', seed_spec='700,702,704')
        b = IdleScenario(mode='eval', seed_spec='700-704')
        c = IdleScenario(mode='eval', seed_spec='700,703,704')
        self.assertEqual(len({s.evaluation_csv for s in (a,b,c)}), 3)
        self.assertNotEqual(a.evaluation_csv, IdleScenario(mode='eval', model_seed=2).evaluation_csv)

    def test_evaluation_entry_writes_scene_csvs_together_without_overwriting(self):
        row = validation_row(46, 123., seed=700)
        agent = PPOAgent()
        with patch.object(idle_scenario, 'ROOT', self.root), \
             patch.object(runner, 'load_checkpoint', return_value=(agent, {'selection_evidence': {}})), \
             patch.object(runner, 'checked_reference', return_value={}), \
             patch.object(runner, 'evaluate', return_value=([row], {})) as evaluate:
            scenes = [IdleScenario(mode='eval', task=task, seed_spec='700') for task in ('S','M')]
            for scene in scenes:
                with patch.object(runner, 'parse_scenario', return_value=scene):
                    runner.evaluate_main()
                self.assertTrue(scene.evaluation_csv.is_file())
                self.assertTrue((scene.evaluation_dir/'summary.json').is_file())
                self.assertFalse((scene.evaluation_dir/'evaluation.csv').exists())
            self.assertEqual(len(list(scenes[0].evaluation_root.glob('*.csv'))), 2)
            before = scenes[0].evaluation_csv.read_bytes()
            with patch.object(runner, 'parse_scenario', return_value=scenes[0]), \
                 self.assertRaisesRegex(FileExistsError, 'evaluation CSV'):
                runner.evaluate_main()
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(scenes[0].evaluation_csv.read_bytes(), before)

    def test_training_checkpoint_evidence_roundtrip_and_v5_rejection(self):
        scene = IdleScenario(mode='train', episodes=4)
        schedule = iter([(45, 1.), (46, 100.), (47, 80.), (46, 80.)])
        def validate(*args, **kwargs):
            success, energy = next(schedule); row = validation_row(success, energy)
            return [row], {'total_energy': dict(mean=energy, std=0.)}
        kw = self.kwargs(workflows_per_episode=2)
        with patch.object(runner, 'planned_assignments', return_value=24), \
             patch.object(runner, 'evaluate', side_effect=validate):
            agent, directory = runner.train(scene, config.DRLGNNConfig(), kw, self.root/'models')
        self.assertEqual((agent.steps, agent.updates), (24, 1))
        loaded, obj = runner.load_checkpoint(directory/'best.pth', scene)
        self.assertEqual(obj['selection_evidence']['selected_episode'], 2)
        self.assertEqual(obj['scheduling_protocol'], config.SCHEDULING_PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'requires best'):
            runner.load_checkpoint(directory/'final.pth', scene)
        _, final = runner.load_checkpoint(directory/'final.pth', scene, require_best=False)
        self.assertEqual(final['completed_episodes'], 4)
        changed = deepcopy(obj); changed['model_version'] = fast_config.MODEL_VERSION
        torch.save(changed, self.root/'old.pth')
        with self.assertRaisesRegex(ValueError, 'model_version'):
            runner.load_checkpoint(self.root/'old.pth', scene)
        changed = deepcopy(obj); changed['scheduling_protocol'] = fast_config.SCHEDULING_PROTOCOL
        torch.save(changed, self.root/'old.pth')
        with self.assertRaisesRegex(ValueError, 'scheduling_protocol'):
            runner.load_checkpoint(self.root/'old.pth', scene)
        a, _ = runner.evaluate(loaded, scene, [700], kwargs=kw)
        b, _ = runner.evaluate(loaded, scene, [700], kwargs=kw)
        self.assertEqual(a, b)
        summary = json.loads((directory/'logs/training_summary.json').read_text())
        self.assertEqual(summary['selected_episode'], 2)
        self.assertEqual(summary['checkpoint_selection']['workflow_success_threshold'], .92)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_training_validation_and_loaded_evaluation(self):
        scene = IdleScenario(mode='train', episodes=1)
        kw = self.kwargs(workflows_per_episode=2)
        with patch.object(runner, 'planned_assignments', return_value=6):
            agent, directory = runner.train(scene, config.DRLGNNConfig(device='cuda'), kw, self.root/'cuda')
        loaded, obj = runner.load_checkpoint(directory/'best.pth', scene, device='cuda')
        self.assertEqual(obj['execution']['graphs_per_backward'], 4)
        self.assertLessEqual(loaded.network.cuda_memory_policy['allocator_fraction'], .85)
        a, _ = runner.evaluate(agent, scene, [700], kwargs=kw)
        b, _ = runner.evaluate(loaded, scene, [700], kwargs=kw)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
