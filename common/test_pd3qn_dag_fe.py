"""Compact DAG-FE observations, paper learning, aggregate-SR selection and real smoke."""
from contextlib import redirect_stdout
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.pd3qn.config import agent_kwargs as dag_agent_kwargs
from algorithms.pd3qn_dag.identity import MODEL_VERSION as LEARNER_VERSION
from algorithms.pd3qn_dag_fe import ALGORITHM, MODEL_VERSION, OBSERVATION_LAYOUT, HEAD_LAYOUT
from algorithms.pd3qn_dag_fe import runtime, selection
from algorithms.pd3qn_dag_fe.config import agent_kwargs
from algorithms.pd3qn_dag_fe.environment import CloudWorkflowEnv_DAG_FE
from algorithms.pd3qn_dag_fe.scenario import DAGFEScenario, parse_scenario
from algorithms.pd3qn_fe import runtime as legacy_runtime, selection as legacy_selection
from algorithms.pd3qn_fe.scenario import FEScenario
from common.f18_state_normalization import PD3QN_UNITS, CURRENT_VERSION
from common.scenarios import MODEL_NAMES, Scenario
from common.test_f18 import fixture_config
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


def row(success=50, energy=100., count=50):
    return dict(seed=600, wf_count=count, wf_success_count=success,
                wf_success_rate=success / count, total_energy=energy)


class SelectionTests(unittest.TestCase):
    def test_aggregate_99_boundary_is_independent_of_fe_092(self):
        rows = [row(100, count=100), row(98, count=100)]
        current = selection.validation_selection(rows)
        self.assertTrue(current['feasible'])
        self.assertEqual(current['workflow_success_rate'], .99)
        self.assertNotIn('validation_ESP', current)
        self.assertTrue(legacy_selection.validation_selection(rows)['feasible'])
        self.assertFalse(selection.validation_selection([row(49)])['feasible'])
        self.assertTrue(selection.validation_selection([row(50)])['feasible'])
        self.assertFalse(selection.validation_selection([row(98, count=100)])['feasible'])

    def test_counts_are_weighted_and_energy_is_instance_mean(self):
        result = selection.validation_selection([row(1, 10, 1), row(98, 30, 100)])
        self.assertEqual(result['workflow_success_rate'], 99 / 101)
        self.assertEqual(result['mean_total_energy'], 20.)
        self.assertFalse(result['feasible'])

    def test_feasibility_energy_success_and_earlier_ties(self):
        def candidate(success, energy):
            return selection.validation_selection([row(success, energy, count=100)])
        key = selection.selection_key
        self.assertLess(key(candidate(99, 1000)), key(candidate(98, 1)))
        self.assertLess(key(candidate(99, 10)), key(candidate(100, 11)))
        self.assertLess(key(candidate(100, 10)), key(candidate(99, 10)))
        self.assertLess(key(candidate(98, 100)), key(candidate(97, 1)))
        self.assertLess(key(candidate(98, 1)), key(candidate(98, 2)))
        # The old ESP-first fallback preferred the low-success first candidate.
        low_success = selection.validation_selection([row(50), row(0)])
        high_success = selection.validation_selection([row(49), row(49)])
        self.assertLess(key(high_success), key(low_success))
        candidates = [candidate(98, 1), candidate(99, 10), candidate(99, 10)]
        best = None
        for episode, item in enumerate(candidates):
            if best is None or key(item) < key(best):
                best, selected_episode = item, episode
        self.assertEqual(selected_episode, 1)

    def test_invalid_metrics_rejected(self):
        for rows in ([], [row(-1)], [row(51)], [row(1, float('nan'))],
                     [row(1, -1)], [dict(row(), wf_count=0)],
                     [dict(row(), wf_success_rate=.99)], [dict(row(), wf_success_count=1.5)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                selection.validation_selection(rows)


class IsolationTests(unittest.TestCase):
    def test_shared_calibration_does_not_switch_to_historical_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scales.json'
            fixture_config(path)
            scene = DAGFEScenario(normalization_config=str(path))
            metadata = scene.normalization_metadata()
            target = replace(scene, mode='eval', task='L')
            self.assertEqual(metadata['normalization_config_hash'],
                             target.normalization_metadata()['normalization_config_hash'])
            self.assertEqual(target.normalization_kwargs()['source_train_scene'], 'SS_T')
        self.assertEqual(scene.normalization_mode, CURRENT_VERSION)
        self.assertEqual(agent_kwargs(scene), dag_agent_kwargs(Scenario(normalization_mode=PD3QN_UNITS)))
        self.assertFalse(dag_agent_kwargs(scene)['paper_mode'])
        cfg = agent_kwargs(scene)
        self.assertTrue(cfg['paper_mode'])
        self.assertEqual((cfg['lr'], cfg['gamma'], cfg['batch_size'], cfg['buffer_size']),
                         (.01, .95, 128, 10000))
        self.assertEqual((cfg['target_update_tau'], cfg['target_update_freq'], cfg['grad_clip']), (0., 60, None))
        self.assertEqual(cfg['hidden_dims'], (1024, 1024, 512, 512, 256))
        self.assertEqual(metadata['observation_layout'], OBSERVATION_LAYOUT)
        self.assertEqual(metadata['head_layout'], HEAD_LAYOUT)
        self.assertEqual(metadata['model_version'], MODEL_VERSION)
        self.assertEqual(metadata['experiment_protocol']['selection'], selection.SELECTION_TAG)
        self.assertEqual(metadata['reproducibility']['algorithm'], ALGORITHM)
        for mode in ('legacy', 'f18_shared_v1', PD3QN_UNITS):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'f19_shared_v1'):
                DAGFEScenario(normalization_mode=mode)
        with self.assertRaises(FileNotFoundError):
            DAGFEScenario(normalization_config='missing_calibration_file.json').normalization_metadata()
        with self.assertRaisesRegex(ValueError, 'fixed, disjoint'):
            DAGFEScenario(legacy_split=True)

    def test_selector_is_local_and_fe_uses_single_092_gate(self):
        self.assertIsNot(selection.validation_selection, legacy_selection.validation_selection)
        self.assertNotEqual(selection.SELECTION_TAG, legacy_selection.SELECTION_TAG)
        self.assertIs(runtime.isolated_evaluation, legacy_runtime.isolated_evaluation)
        self.assertIs(runtime.env_kwargs, legacy_runtime.env_kwargs)
        feasible = selection.validation_selection([row(50, 1000)])
        fallback = selection.validation_selection([row(49, 1)])
        self.assertTrue(feasible['feasible'])
        self.assertFalse(fallback['feasible'])
        self.assertLess(selection.selection_key(feasible), selection.selection_key(fallback))
        self.assertEqual(selection.selection_metadata()['workflow_success_threshold'], .99)
        self.assertNotIn('satisfaction_probability_target', selection.selection_metadata())
        self.assertEqual(legacy_selection.selection_metadata()['workflow_success_threshold'], .92)
        self.assertNotIn('satisfaction_probability_target', legacy_selection.selection_metadata())

    def test_paths_and_original_algorithms_remain_isolated(self):
        table = dict(MODEL_NAMES)
        dag = Scenario(mode='train', normalization_mode=PD3QN_UNITS)
        fe = FEScenario(mode='train')
        scene = DAGFEScenario(mode='train')
        self.assertNotEqual(scene.checkpoint_dir(), dag.checkpoint_dir('pd3qn'))
        self.assertNotEqual(scene.checkpoint_dir(), fe.checkpoint_dir())
        self.assertTrue(scene.model_name().startswith(MODEL_VERSION + '_'))
        self.assertNotEqual(scene.checkpoint_dir(), replace(scene, model_seed=2).checkpoint_dir())
        target = replace(scene, mode='eval', task='L')
        self.assertEqual(scene.checkpoint_dir(), target.checkpoint_dir())
        self.assertNotEqual(scene.evaluation_dir, target.evaluation_dir)
        self.assertEqual(dag.reproducibility_metadata()['checkpoint_selection'], 'minimum_validation_total_energy')
        self.assertEqual(dag.protocol().selection, 'minimum_validation_total_energy')
        self.assertEqual(fe.normalization_mode, 'f19_shared_v1')
        self.assertEqual(MODEL_NAMES, table)

    def test_cli_nine_sources_and_27_targets_resolve_matching_models(self):
        destinations = set()
        filenames = set()
        for resource in 'SML':
            for deadline in ('Tight', 'Medium', 'Loose'):
                args = ['--resource', resource, '--deadline', deadline, '--dry-run']
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                    parse_scenario('train', args)
                self.assertEqual(stopped.exception.code, 0)
                source = json.loads(output.getvalue())
                self.assertEqual(source['algorithm'], ALGORITHM)
                self.assertTrue(source['agent']['paper_mode'])
                self.assertEqual(source['input_dim'], 3 + 3 * sum(source['vms_per_host']))
                self.assertEqual(source['normalization']['normalization_mode'], CURRENT_VERSION)
                for task in 'SML':
                    output = io.StringIO()
                    with redirect_stdout(output), self.assertRaises(SystemExit):
                        parse_scenario('eval', args + ['--task', task, '--train-task', 'S',
                            '--train-resource', resource, '--train-deadline', deadline])
                    target = json.loads(output.getvalue())
                    self.assertEqual(source['checkpoint_dir'], target['checkpoint_dir'])
                    self.assertEqual(target['seeds'], list(range(700, 730)))
                    destinations.add(target['evaluation_directory'])
                    filenames.add(Path(target['evaluation_csv']).name)
        self.assertEqual(len(destinations), 27)
        self.assertEqual(len(filenames), 27)
        self.assertIn('eval_pd3qn_dag_fe_taskS_resS_Loose_model1_seeds700-729.csv', filenames)
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            parse_scenario('train', ['--normalization-mode', PD3QN_UNITS, '--dry-run'])

    def test_partitions_checked_before_environment_creation(self):
        scene = DAGFEScenario()
        for seeds, split in (([700], 'validation'), ([600], 'test'),
                             ([600, 600], 'validation'), ([], 'validation')):
            with self.subTest(seeds=seeds, split=split), self.assertRaises(ValueError):
                runtime.evaluate_instances(None, scene, seeds, split)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.temp = tempfile.TemporaryDirectory(prefix='pd3qn_dag_fe_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        caches = {task: cls.root / f'cache_{task}.json' for task in 'SML'}
        from common.precompute_deadlines import main as precompute
        with redirect_stdout(io.StringIO()):
            for task in 'SML':
                precompute(Scenario(mode='precompute', task=task,
                    seed_spec='0,1,600,700' if task == 'S' else '700', cache_output=str(caches[task])))

        class LocalScenario(DAGFEScenario):
            @property
            def cache_path(self):
                return caches[self.task]

            def checkpoint_dir(self, algorithm=ALGORITHM):
                return cls.root / 'ckpts'

            def log_dir(self, algorithm=ALGORITHM):
                return cls.root / 'logs'

            @property
            def evaluation_dir(self):
                return cls.root / 'evaluation' / (self.task + self.resource + '_' + self.deadline)

        scales = cls.root / 'scales.json'
        fixture_config(scales)
        cls.scene = LocalScenario(mode='train', episodes=2, paper_options={'device': 'cpu'},
                                  normalization_config=str(scales))
        cls.eval_scene = replace(cls.scene, mode='eval', seed_spec='700')
        # Only shrink network width; retain DAG batch 128, PER, LR and target-copy clock.
        cls.settings = dict(agent_kwargs(cls.scene), hidden_dims=(16, 16))
        from algorithms.pd3qn_dag_fe.train import train
        with patch.object(runtime, 'agent_kwargs', return_value=cls.settings), redirect_stdout(io.StringIO()):
            cls.agent, cls.summary = train(cls.scene)

    def test_real_training_preserves_dag_updates_and_saves_selected_checkpoint(self):
        best = torch.load(self.scene.checkpoint_dir() / 'best_pd3qn_dag_fe.pth', weights_only=True)
        final = torch.load(self.scene.checkpoint_dir() / 'pd3qn_dag_fe_final.pth', weights_only=True)
        records = [json.loads(line) for line in
                   (self.scene.log_dir() / 'validation_selection.jsonl').read_text().splitlines()]
        with (self.scene.log_dir() / 'train_metrics_pd3qn_dag_fe_cost.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        expected = min(records, key=lambda record: selection.selection_key(record['validation_selection']))
        self.assertEqual(len(records), 2)
        self.assertEqual(final['pd3qn_implementation'], LEARNER_VERSION)
        self.assertEqual(final['network']['input_dim'], 78)
        self.assertEqual(final['normalization']['normalization_version'], CURRENT_VERSION)
        self.assertEqual(final['normalization']['model_version'], MODEL_VERSION)
        self.assertEqual(final['normalization']['observation_layout'], OBSERVATION_LAYOUT)
        self.assertNotIn('eval_ESP', csv_rows[0])
        self.assertEqual(final['eps_steps'], int(csv_rows[-1]['step']))
        self.assertEqual(final['updates'], final['eps_steps'] - 127)
        self.assertEqual(final['hard_freq'], 60)
        self.assertEqual(final['tau'], 0.)
        self.assertEqual(final['per_beta_start'], 0.)
        self.assertEqual(final['optim']['param_groups'][0]['lr'], .01)
        self.assertEqual(best['normalization'][ALGORITHM]['episode'], expected['episode'])
        self.assertEqual(best['eps_steps'], int(csv_rows[expected['episode']]['step']))
        self.assertEqual(self.summary['best_validation_selection']['episode'], expected['episode'])
        self.assertTrue(all(torch.isfinite(value).all() for value in final['online'].values()))
        self.assertFalse((self.scene.checkpoint_dir() / 'best_pd3qn_dag.pth').exists())
        runtime.load_best_checkpoint(self.agent, self.scene.checkpoint_dir() / 'best_pd3qn_dag_fe.pth', self.eval_scene)
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key])
                            for key, value in best['online'].items()))

    def test_full_size_linear_heads_optimizer_binding_and_gradient_update(self):
        from algorithms.pd3qn_dag_fe.agent import DAGFEAgent
        agent = DAGFEAgent(input_dim=78, output_dim=25, device='cpu', **agent_kwargs(self.scene))
        self.assertIsInstance(agent.online.val, torch.nn.Linear)
        self.assertIsInstance(agent.online.adv, torch.nn.Linear)
        self.assertEqual((agent.online.val.in_features, agent.online.val.out_features), (256, 1))
        self.assertEqual((agent.online.adv.in_features, agent.online.adv.out_features), (256, 25))
        self.assertEqual(sum(p.numel() for p in agent.online.parameters()), 2055962)
        self.assertEqual({id(p) for p in agent.online.parameters()},
                         {id(p) for group in agent.optim.param_groups for p in group['params']})
        self.assertTrue(all(torch.equal(p, q) for p, q in
                            zip(agent.online.parameters(), agent.target.parameters())))
        obs = np.ones(78, dtype=np.float32)
        mask = np.ones(25, dtype=np.float32)
        for _ in range(127):
            agent.remember(obs, mask, 0, -1., obs, mask, True)
        self.assertIsNone(agent.update())
        before = agent.online.val.weight.detach().clone()
        agent.remember(obs, mask, 0, -1., obs, mask, True)
        agent.begin_episode()
        agent.select_action(obs, mask)
        self.assertIsNotNone(agent.update())
        self.assertFalse(torch.equal(before, agent.online.val.weight))
        self.assertEqual(agent._eps_steps, 1)

    def test_compact_shared_observations_and_fixed_action_physics_match(self):
        kwargs = runtime.env_kwargs(self.scene)
        env = CloudWorkflowEnv_DAG_FE(**kwargs)
        original = CloudWorkflowEnv_PD3QN_Cost(**kwargs)
        obs, _ = env.reset()
        full, _ = original.reset()
        keep = [2, 3, 5, *range(6, len(full))]
        self.assertEqual(obs.shape, (78,))
        self.assertEqual(env.observation_space.shape, (78,))
        done = False
        while not done:
            # The base observation casts task values to float32 before scaling.
            np.testing.assert_allclose(obs, full[keep], rtol=2e-7, atol=1e-7)
            np.testing.assert_array_equal(env.get_action_mask(), original.get_action_mask())
            action = int(np.flatnonzero(env.get_action_mask())[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            full, old_reward, old_terminated, old_truncated, _ = original.step(action)
            self.assertEqual((reward, terminated, truncated), (old_reward, old_terminated, old_truncated))
            self.assertEqual(env.current_time, original.current_time)
            done = terminated or truncated
        np.testing.assert_array_equal(obs[:3], np.zeros(3))
        self.assertEqual(env.total_energy, original.total_energy)
        self.assertEqual(env.task_end_time, original.task_end_time)
        self.assertEqual(env.get_episode_metrics(), original.get_episode_metrics())
        self.assertTrue(any(env.task_out_bits))
        self.assertTrue(any(env._host_segments.values()))

    def test_removed_features_are_not_read_and_slack_is_still_relative(self):
        env = CloudWorkflowEnv_DAG_FE(**runtime.env_kwargs(self.scene))
        obs, _ = env.reset()
        gid = env._cur_task_gid
        env.task_baseline_finish[gid] = env.current_time - 2 * env.state_normalizer.scales['T_ref']
        self.assertEqual(env._build_obs()[2], -2.)
        env.task_baseline_finish[gid] += 10 * env.state_normalizer.scales['T_ref']
        before = env._build_obs()
        env.task_out_bits[gid] += 1e20
        env.ready_task_ids.append(gid)
        # Absolute time can change without changing the retained relative times.
        env.current_time += 1000
        env.task_baseline_finish[gid] += 1000
        env.vm_available_at += 1000
        with patch.object(env, '_parent_input_bits_by_vm', side_effect=AssertionError('parent input read')), \
             patch.object(env, '_host_history_features', side_effect=AssertionError('host history read')):
            np.testing.assert_allclose(env._build_obs(), before, rtol=0., atol=1e-6)
        self.assertGreater(before[2], 1.)

    def test_measurements_disabled_keeps_selection_weights_and_counters(self):
        counters = self.agent._eps_steps, self.agent._updates, self.agent._episode_steps, len(self.agent.buffer)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        rows = runtime.evaluate_instances(self.agent, self.scene, [600])
        with patch.dict('os.environ', {'EXPERIMENT_METRICS': '0'}):
            disabled = runtime.evaluate_instances(self.agent, self.scene, [600])
        self.assertEqual(rows, disabled)
        self.assertEqual(rows[0]['wf_completed_count'], 50)
        self.assertEqual(counters, (self.agent._eps_steps, self.agent._updates,
                                   self.agent._episode_steps, len(self.agent.buffer)))
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_wrong_algorithm_rule_and_baseline_checkpoints_rejected_before_weight_mutation(self):
        source = torch.load(self.scene.checkpoint_dir() / 'best_pd3qn_dag_fe.pth', weights_only=True)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        for field, value in (('algorithm', 'pd3qn_fe'), ('algorithm', 'pd3qn_dag'),
                             ('role', 'final'), ('checkpoint_selection', {}), ('model_seed', 99),
                             ('model_version', 'pd3qn_dag_fe_v1'),
                             ('head_layout', 'hidden256_dueling'),
                             ('observation_layout', 'task6_vm4_parentdata_host2_v3')):
            obj = dict(source)
            obj['normalization'] = dict(source['normalization'])
            obj['normalization'][ALGORITHM] = dict(source['normalization'][ALGORITHM], **{field: value})
            path = self.root / 'invalid.pth'
            torch.save(obj, path)
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                runtime.load_best_checkpoint(self.agent, path, self.eval_scene)
        baseline = dict(source)
        baseline['normalization'] = dict(source['normalization'])
        baseline['normalization'].pop(ALGORITHM)
        torch.save(baseline, self.root / 'baseline.pth')
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            runtime.load_best_checkpoint(self.agent, self.root / 'baseline.pth', self.eval_scene)
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_three_task_sizes_use_same_source_model_with_independent_results(self):
        from algorithms.pd3qn_dag_fe.evaluate import evaluate
        from algorithms.pd3qn_dag_fe.train import train
        paths = set()
        with patch.object(runtime, 'agent_kwargs', return_value=self.settings), redirect_stdout(io.StringIO()):
            for task in 'SML':
                target = replace(self.eval_scene, task=task)
                rows, summary = evaluate(target)
                self.assertEqual(rows[0]['seed'], 700)
                self.assertEqual(rows[0]['wf_count'], 50)
                self.assertEqual(rows[0]['wf_completed_count'], 50)
                self.assertEqual(summary['algorithm'], ALGORITHM)
                paths.add(summary['source_checkpoint'])
                self.assertTrue(target.evaluation_csv.is_file())
                self.assertTrue((target.evaluation_dir / 'measurements/episodes/seed700.json').is_file())
                with self.assertRaises(FileExistsError):
                    evaluate(target)
            with self.assertRaises(FileExistsError):
                train(self.scene)
        self.assertEqual(len(paths), 1)


if __name__ == '__main__':
    unittest.main()
