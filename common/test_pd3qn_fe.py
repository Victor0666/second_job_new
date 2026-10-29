"""PD3QN-FE selection boundaries, isolation and real train/save/load/test smoke."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_fe import ALGORITHM
from algorithms.pd3qn_fe.runtime import (env_kwargs, evaluate_instances, isolated_evaluation,
    load_best_checkpoint)
from algorithms.pd3qn_fe.scenario import FEScenario, parse_scenario
from algorithms.pd3qn_fe.selection import (SELECTION_TAG, selection_key,
    selection_metadata, validation_selection)
from common.d3qn_agent_PER import D3QNAgent
from common.f18_state_normalization import PD3QN_UNITS
from common.scenarios import MODEL_NAMES, Scenario


def row(success=50, energy=100., count=50, seed=600):
    return dict(seed=seed, wf_count=count, wf_success_count=success,
                wf_success_rate=success / count, total_energy=energy)


class SelectionTests(unittest.TestCase):
    def test_inclusive_92_percent_boundary_and_50_workflow_discreteness(self):
        self.assertFalse(validation_selection([row(45)])['feasible'])
        self.assertTrue(validation_selection([row(46)])['feasible'])
        self.assertTrue(validation_selection([row(92, count=100)])['feasible'])
        self.assertFalse(validation_selection([row(91, count=100)])['feasible'])
        self.assertEqual(SELECTION_TAG, 'select_sr092_minenergy_v1')
        self.assertNotIn('satisfaction_probability_target', selection_metadata())
        self.assertNotIn('validation_ESP', validation_selection([row(46)]))

    def test_feasible_always_wins_then_energy_then_success(self):
        def candidate(success, energy):
            return validation_selection([row(success, energy, count=100)])
        self.assertLess(selection_key(candidate(92, 1000)), selection_key(candidate(91, 1)))
        self.assertLess(selection_key(candidate(92, 10)), selection_key(candidate(100, 11)))
        self.assertLess(selection_key(candidate(100, 10)), selection_key(candidate(92, 10)))

    def test_fallback_success_then_energy_without_instance_pass_rate(self):
        one_perfect_instance = validation_selection([row(50, 1000), row(0, 1000)])
        high_success = validation_selection([row(45, 1), row(45, 1)])
        self.assertFalse(high_success['feasible'])
        self.assertLess(selection_key(high_success), selection_key(one_perfect_instance))
        self.assertLess(selection_key(high_success), selection_key(validation_selection([row(44, 0)])))
        self.assertLess(selection_key(validation_selection([row(45, 1)])),
                        selection_key(validation_selection([row(45, 2)])))

    def test_weighted_workflow_success_and_instance_mean_energy(self):
        result = validation_selection([row(1, 10, 1), row(98, 30, 100)])
        self.assertEqual(result['workflow_success_rate'], 99 / 101)
        self.assertEqual(result['mean_total_energy'], 20.)
        self.assertTrue(result['feasible'])
        self.assertNotIn('validation_ESP', result)
        self.assertTrue(validation_selection([row(2, count=2), row(90, count=98)])['feasible'])

    def test_strict_comparison_keeps_earlier_checkpoint(self):
        candidates = [validation_selection([row(45, 1)]),
                      validation_selection([row(46, 100)]),
                      validation_selection([row(46, 100)]),
                      validation_selection([row(45, 0)])]
        best, episode = None, None
        for index, candidate in enumerate(candidates):
            if best is None or selection_key(candidate) < selection_key(best):
                best, episode = candidate, index
        self.assertEqual(episode, 1)

    def test_invalid_rows_rejected(self):
        for rows in ([], [row(-1)], [row(51)], [row(1, float('nan'))],
                     [row(1, -1)], [dict(row(), wf_count=0)],
                     [dict(row(), wf_success_rate=.99)], [dict(row(), wf_success_count=1.5)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validation_selection(rows)


class IsolationTests(unittest.TestCase):
    def test_default_matches_old_pd3qn_and_dag_is_rejected(self):
        scene = FEScenario()
        self.assertEqual(scene.normalization_mode, 'f19_shared_v1')
        self.assertEqual(agent_kwargs(scene), agent_kwargs(Scenario(normalization_mode='f19_shared_v1')))
        self.assertFalse(agent_kwargs(scene)['paper_mode'])
        for mode in ('legacy', 'f18_shared_v1'):
            self.assertFalse(agent_kwargs(replace(scene, normalization_mode=mode))['paper_mode'])
        with self.assertRaisesRegex(ValueError, 'PD3QN-DAG'):
            FEScenario(normalization_mode=PD3QN_UNITS)
        with self.assertRaisesRegex(ValueError, 'fixed, disjoint'):
            FEScenario(legacy_split=True)

    def test_paths_source_scene_and_original_protocol_are_isolated(self):
        original = Scenario(mode='train')
        original_path = original.checkpoint_dir('pd3qn')
        names = dict(MODEL_NAMES)
        scene = FEScenario(mode='train')
        self.assertNotEqual(scene.checkpoint_dir(), original_path)
        self.assertNotEqual(scene.checkpoint_dir(), replace(scene, model_seed=2).checkpoint_dir())
        self.assertLessEqual(len(scene.model_name()), 32)
        self.assertEqual(scene.selection_protocol()['selection'], SELECTION_TAG)
        target = replace(scene, mode='eval', task='L', resource='M', deadline='Loose',
                         train_resource='S', train_deadline='Tight')
        self.assertEqual(scene.checkpoint_dir(), target.checkpoint_dir())
        self.assertIn('_LM_Loose_fromSS_Tight_', target.evaluation_csv.name)
        self.assertEqual(scene.protocol().selection, 'minimum_validation_total_energy')
        self.assertEqual(scene.reproducibility_metadata()['data_split']['selection'], SELECTION_TAG)
        self.assertEqual(original.checkpoint_dir('pd3qn'), original_path)
        self.assertEqual(MODEL_NAMES, names)
        self.assertEqual(original.protocol().selection, 'minimum_validation_total_energy')

    def test_cli_dry_run_is_variant_local(self):
        for mode in ('train', 'eval'):
            console = io.StringIO()
            with redirect_stdout(console), self.assertRaises(SystemExit) as stopped:
                parse_scenario(mode, ['--dry-run'])
            self.assertEqual(stopped.exception.code, 0)
            result = json.loads(console.getvalue())
            self.assertEqual(result['algorithm'], ALGORITHM)
            self.assertFalse(result['agent']['paper_mode'])
            self.assertEqual(result['checkpoint_selection'], selection_metadata())
            self.assertEqual(result['experiment_protocol']['validation'], [600])
            if mode == 'eval':
                self.assertTrue(Path(result['evaluation_csv']).name.startswith(
                    'eval_pd3qn_fe_SS_Tight_fromSS_Tight_m1_s700-729_c'))
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            parse_scenario('train', ['--normalization-mode', PD3QN_UNITS, '--dry-run'])

    def test_csv_filenames_distinguish_target_scenes_models_and_seeds(self):
        scene = FEScenario(mode='eval')
        names = {replace(scene, task=task, resource=resource, deadline=deadline).evaluation_csv.name
                 for task in 'SML' for resource in 'SML' for deadline in ('Tight', 'Medium', 'Loose')}
        self.assertEqual(len(names), 27)
        loose = replace(scene, deadline='Loose')
        self.assertIn('_SS_Loose_fromSS_Loose_m1_s700-729_c', loose.evaluation_csv.name)
        self.assertNotEqual(scene.evaluation_csv.name, replace(scene, model_seed=2).evaluation_csv.name)
        self.assertIn('_s700+702_', replace(scene, seed_spec='700,702').evaluation_csv.name)

    def test_all_checkpoint_provenance_reports_actual_selector(self):
        metadata = FEScenario(normalization_mode='legacy').normalization_metadata()
        self.assertEqual(metadata['experiment_protocol']['selection'], SELECTION_TAG)
        self.assertEqual(metadata['reproducibility']['data_split']['selection'], SELECTION_TAG)
        self.assertEqual(metadata['checkpoint_selection'], selection_metadata())
        self.assertEqual(Scenario().protocol().selection, 'minimum_validation_total_energy')

    def test_rng_and_network_modes_restored_even_on_failure(self):
        agent = D3QNAgent(input_dim=2, output_dim=2, hidden_dims=(4,), device='cpu')
        agent.online.train()
        agent.target.eval()
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with isolated_evaluation(agent):
                self.assertFalse(agent.online.training)
                random.random()
                np.random.random()
                torch.rand(1)
                if cuda_before is not None:
                    torch.rand(1, device='cuda')
                raise RuntimeError('probe')
        self.assertEqual(before[0], random.getstate())
        self.assertEqual(before[1][0], np.random.get_state()[0])
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertEqual(before[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        if cuda_before is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())))
        self.assertTrue(agent.online.training)
        self.assertFalse(agent.target.training)

    def test_partitions_checked_before_environment_creation(self):
        scene = FEScenario()
        for seeds, split in (([700], 'validation'), ([600], 'test'), ([600, 600], 'validation'),
                             ([], 'validation'), ([600], 'training')):
            with self.subTest(seeds=seeds, split=split), self.assertRaises(ValueError):
                evaluate_instances(None, scene, seeds, split)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='pd3qn_fe_test_')
        cls.root = Path(cls.temp.name)
        cache = cls.root / 'cache.json'
        from common.precompute_deadlines import main as precompute
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute', seed_spec='0,1,600,700', cache_output=str(cache)))

        class LocalScenario(FEScenario):
            @property
            def cache_path(self):
                return cache

            def checkpoint_dir(self, algorithm=ALGORITHM):
                return cls.root / 'ckpts'

            def log_dir(self, algorithm=ALGORITHM):
                return cls.root / 'logs'

            @property
            def evaluation_dir(self):
                return cls.root / 'evaluation'

            @property
            def evaluation_root(self):
                return cls.root / 'results'

        cls.scene = LocalScenario(mode='train', episodes=2, paper_options={'device': 'cpu'})
        cls.eval_scene = replace(cls.scene, mode='eval', seed_spec='700')
        # Exercise real environment/optimizer/save/load with a small test-only learner.
        cls.settings = dict(agent_kwargs(cls.scene), hidden_dims=(16, 16), batch_size=8, buffer_size=128)
        from algorithms.pd3qn_fe.train import train
        with patch('algorithms.pd3qn_fe.runtime.agent_kwargs', return_value=cls.settings), redirect_stdout(io.StringIO()):
            cls.agent, cls.summary = train(cls.scene)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_real_training_best_and_final_roundtrip(self):
        best_path = self.scene.checkpoint_dir() / 'best_pd3qn_fe.pth'
        best = torch.load(best_path, weights_only=True)
        final = torch.load(self.scene.checkpoint_dir() / 'pd3qn_fe_final.pth', weights_only=True)
        records = [json.loads(line) for line in
                   (self.scene.log_dir() / 'validation_selection.jsonl').read_text().splitlines()]
        expected = min(records, key=lambda record: selection_key(record['validation_selection']))
        self.assertEqual(len(records), 2)
        self.assertEqual(best['normalization']['pd3qn_fe']['episode'], expected['episode'])
        self.assertEqual(self.summary['best_validation_selection']['episode'], expected['episode'])
        self.assertEqual(best['normalization']['pd3qn_fe']['validation_selection'], expected['validation_selection'])
        self.assertNotIn('validation_ESP', expected['validation_selection'])
        with (self.scene.log_dir() / 'train_metrics_pd3qn_fe_cost.csv').open(newline='') as stream:
            self.assertNotIn('eval_ESP', csv.DictReader(stream).fieldnames)
        self.assertEqual(final['pd3qn_implementation'], 'legacy')
        self.assertGreater(final['updates'], 0)
        self.assertEqual(final['normalization']['normalization_version'], 'f19_shared_v1')
        self.assertFalse((self.scene.checkpoint_dir() / 'best_pd3qn.pth').exists())
        self.assertTrue(all(torch.isfinite(value).all() for value in final['online'].values()))
        metadata = load_best_checkpoint(self.agent, best_path, self.eval_scene)
        self.assertEqual(metadata['episode'], expected['episode'])
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key])
                            for key, value in best['online'].items()))

    def test_measurements_disabled_does_not_change_selection_or_agent(self):
        before = self.agent._eps_steps, self.agent._updates, self.agent._episode_steps, len(self.agent.buffer)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        rows = evaluate_instances(self.agent, self.scene, [600])
        with patch.dict('os.environ', {'EXPERIMENT_METRICS': '0'}):
            disabled = evaluate_instances(self.agent, self.scene, [600])
        self.assertEqual(rows, disabled)
        self.assertEqual(validation_selection(rows), validation_selection(disabled))
        self.assertEqual(rows[0]['wf_count'], 50)
        self.assertEqual(rows[0]['wf_completed_count'], 50)
        self.assertEqual(before, (self.agent._eps_steps, self.agent._updates,
                                  self.agent._episode_steps, len(self.agent.buffer)))
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_reject_wrong_variant_and_final_checkpoint_before_mutating_weights(self):
        source = torch.load(self.scene.checkpoint_dir() / 'best_pd3qn_fe.pth', weights_only=True)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        for field, value in (('algorithm', 'pd3qn'), ('role', 'final'),
                             ('checkpoint_selection', {}), ('model_seed', 99),
                             ('checkpoint_selection', dict(selection_metadata(),
                                 version='select_sr099_esp099_v1', workflow_success_threshold=.99,
                                 satisfaction_probability_target=.99))):
            obj = dict(source)
            obj['normalization'] = dict(source['normalization'])
            obj['normalization']['pd3qn_fe'] = dict(source['normalization']['pd3qn_fe'], **{field: value})
            path = self.root / 'invalid.pth'
            torch.save(obj, path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                load_best_checkpoint(self.agent, path, self.eval_scene)
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_real_evaluation_and_duplicate_outputs_rejected(self):
        from algorithms.pd3qn_fe.evaluate import evaluate
        from algorithms.pd3qn_fe.train import train
        with patch('algorithms.pd3qn_fe.runtime.agent_kwargs', return_value=self.settings), redirect_stdout(io.StringIO()):
            rows, summary = evaluate(self.eval_scene)
            with self.assertRaises(FileExistsError):
                evaluate(self.eval_scene)
            with self.assertRaises(FileExistsError):
                train(self.scene)
        self.assertEqual(rows[0]['seed'], 700)
        self.assertEqual(rows[0]['wf_count'], 50)
        self.assertEqual(summary['algorithm'], ALGORITHM)
        self.assertTrue(self.eval_scene.evaluation_csv.is_file())
        self.assertTrue((self.eval_scene.evaluation_dir / 'measurements/episodes/seed700.json').is_file())


if __name__ == '__main__':
    unittest.main()
