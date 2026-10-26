"""DAG-FE identity, original-DAG settings, checkpoint isolation and real smoke."""
from contextlib import redirect_stdout
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_dag.identity import MODEL_VERSION
from algorithms.pd3qn_dag_fe import ALGORITHM
from algorithms.pd3qn_dag_fe import runtime, selection
from algorithms.pd3qn_dag_fe.scenario import DAGFEScenario, parse_scenario
from algorithms.pd3qn_fe import runtime as legacy_runtime, selection as legacy_selection
from algorithms.pd3qn_fe.scenario import FEScenario
from common.f18_state_normalization import PD3QN_UNITS
from common.scenarios import MODEL_NAMES, Scenario


def row(success=50, energy=100.):
    return dict(seed=600, wf_count=50, wf_success_count=success,
                wf_success_rate=success / 50, total_energy=energy)


class IsolationTests(unittest.TestCase):
    def test_defaults_match_dag_without_calibration_and_reject_historical_modes(self):
        scene = DAGFEScenario(normalization_config='missing_calibration_file.json')
        self.assertEqual(scene.normalization_mode, PD3QN_UNITS)
        self.assertEqual(agent_kwargs(scene), agent_kwargs(Scenario(normalization_mode=PD3QN_UNITS)))
        cfg = agent_kwargs(scene)
        self.assertTrue(cfg['paper_mode'])
        self.assertEqual((cfg['lr'], cfg['gamma'], cfg['batch_size'], cfg['buffer_size']),
                         (.01, .95, 128, 10000))
        self.assertEqual((cfg['target_update_tau'], cfg['target_update_freq'], cfg['grad_clip']), (0., 60, None))
        self.assertEqual(cfg['hidden_dims'], (1024, 1024, 512, 512, 256))
        metadata = scene.normalization_metadata()
        self.assertEqual(metadata['observation_layout'], 'task6_vm4_parentdata_host2_v3')
        self.assertEqual(metadata['experiment_protocol']['selection'], selection.SELECTION_TAG)
        self.assertEqual(metadata['reproducibility']['algorithm'], ALGORITHM)
        for mode in ('legacy', 'f18_shared_v1', 'f19_shared_v1'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'fixed-unit'):
                DAGFEScenario(normalization_mode=mode)
        with self.assertRaisesRegex(ValueError, 'fixed, disjoint'):
            DAGFEScenario(legacy_split=True)

    def test_selection_is_exactly_the_existing_dual_99_rule(self):
        self.assertIs(selection.validation_selection, legacy_selection.validation_selection)
        self.assertIs(selection.selection_key, legacy_selection.selection_key)
        self.assertIs(runtime.isolated_evaluation, legacy_runtime.isolated_evaluation)
        self.assertIs(runtime.env_kwargs, legacy_runtime.env_kwargs)
        feasible = selection.validation_selection([row(50, 1000)])
        fallback = selection.validation_selection([row(49, 1)])
        self.assertTrue(feasible['feasible'])
        self.assertFalse(fallback['feasible'])
        self.assertLess(selection.selection_key(feasible), selection.selection_key(fallback))
        self.assertEqual(selection.selection_metadata()['workflow_success_threshold'], .99)
        self.assertEqual(selection.selection_metadata()['satisfaction_probability_target'], .99)

    def test_paths_and_original_algorithms_remain_isolated(self):
        table = dict(MODEL_NAMES)
        dag = Scenario(mode='train', normalization_mode=PD3QN_UNITS)
        fe = FEScenario(mode='train')
        scene = DAGFEScenario(mode='train')
        self.assertNotEqual(scene.checkpoint_dir(), dag.checkpoint_dir('pd3qn'))
        self.assertNotEqual(scene.checkpoint_dir(), fe.checkpoint_dir())
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
            parse_scenario('train', ['--normalization-mode', 'f19_shared_v1', '--dry-run'])

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

        cls.scene = LocalScenario(mode='train', episodes=2, paper_options={'device': 'cpu'},
                                  normalization_config='missing_calibration_file.json')
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
        self.assertEqual(final['pd3qn_implementation'], MODEL_VERSION)
        self.assertEqual(final['network']['input_dim'], 6 + 4 * 25 + 2 * 3)
        self.assertEqual(final['normalization']['normalization_version'], PD3QN_UNITS)
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
                             ('role', 'final'), ('checkpoint_selection', {}), ('model_seed', 99)):
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
