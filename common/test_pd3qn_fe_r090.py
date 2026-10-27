"""R090 reward isolation, aggregate success selection and real train/evaluate smoke."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_fe import runtime as original_runtime, selection as original_selection
from algorithms.pd3qn_fe.scenario import FEScenario
from algorithms.pd3qn_dag_fe import runtime as dag_runtime, selection as sr_selection
from algorithms.pd3qn_dag_fe.scenario import DAGFEScenario
from algorithms.pd3qn_fe_r090 import ALGORITHM, MODEL_VERSION, reward_metadata
from algorithms.pd3qn_fe_r090 import runtime, selection
from algorithms.pd3qn_fe_r090.scenario import R090Scenario, parse_scenario
from common.f18_state_normalization import PD3QN_UNITS
from common.scenarios import MODEL_NAMES, Scenario
from common.test_f18 import fixture_config
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


class ConfigurationTests(unittest.TestCase):
    def test_only_reward_weight_changes_and_other_fe_algorithms_keep_075(self):
        scene = R090Scenario()
        old = original_runtime.env_kwargs(scene)
        new = runtime.env_kwargs(scene)
        self.assertEqual({key for key in old if old[key] != new[key]}, {'alpha_delay_vm'})
        self.assertEqual(new['alpha_delay_vm'], .9)
        self.assertEqual(original_runtime.env_kwargs(FEScenario())['alpha_delay_vm'], .75)
        self.assertEqual(dag_runtime.env_kwargs(DAGFEScenario())['alpha_delay_vm'], .75)
        self.assertEqual(agent_kwargs(scene), agent_kwargs(FEScenario()))
        cfg = agent_kwargs(scene)
        self.assertFalse(cfg['paper_mode'])
        self.assertEqual((cfg['lr'], cfg['gamma'], cfg['batch_size'], cfg['buffer_size']),
                         (.0003, .99, 256, 120000))
        self.assertIs(runtime.isolated_evaluation, original_runtime.isolated_evaluation)
        self.assertIs(runtime.set_seed, original_runtime.set_seed)

    def test_single_aggregate_sr_reuses_pure_selector_without_changing_original_fe(self):
        rows = [dict(seed=600, wf_count=100, wf_success_count=success,
                     wf_success_rate=success / 100, total_energy=energy)
                for success, energy in ((100, 100.), (98, 200.))]
        self.assertIs(selection.validation_selection, sr_selection.validation_selection)
        self.assertIs(selection.selection_key, sr_selection.selection_key)
        result = selection.validation_selection(rows)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['workflow_success_rate'], .99)
        self.assertEqual(result['mean_total_energy'], 150.)
        self.assertNotIn('validation_ESP', result)
        self.assertFalse(original_selection.validation_selection(rows)['feasible'])
        self.assertEqual(original_selection.SATISFACTION_PROBABILITY_TARGET, .99)

    def test_nine_source_and_27_target_paths_include_reward_identity(self):
        names = dict(MODEL_NAMES)
        paths = set()
        for resource in 'SML':
            for deadline in ('Tight', 'Medium', 'Loose'):
                args = ['--resource', resource, '--deadline', deadline, '--dry-run']
                stream = io.StringIO()
                with redirect_stdout(stream), self.assertRaises(SystemExit) as stopped:
                    parse_scenario('train', args)
                self.assertEqual(stopped.exception.code, 0)
                source = json.loads(stream.getvalue())
                self.assertEqual(source['algorithm'], ALGORITHM)
                self.assertEqual(source['model_version'], MODEL_VERSION)
                self.assertEqual(source['reward']['delay_weight'], .9)
                self.assertAlmostEqual(source['reward']['energy_weight'], .1)
                self.assertNotIn('satisfaction_probability_target', source['checkpoint_selection'])
                for task in 'SML':
                    stream = io.StringIO()
                    with redirect_stdout(stream), self.assertRaises(SystemExit):
                        parse_scenario('eval', args + ['--task', task, '--train-task', 'S',
                            '--train-resource', resource, '--train-deadline', deadline])
                    target = json.loads(stream.getvalue())
                    self.assertEqual(source['checkpoint_dir'], target['checkpoint_dir'])
                    self.assertEqual(target['normalization']['source_train_scene'],
                                     source['normalization']['source_train_scene'])
                    paths.add(target['evaluation_csv'])
        self.assertEqual(len(paths), 27)
        scene = R090Scenario()
        self.assertNotEqual(scene.checkpoint_dir(), FEScenario().checkpoint_dir())
        self.assertNotEqual(scene.checkpoint_dir(), DAGFEScenario().checkpoint_dir())
        self.assertEqual(scene.evaluation_csv.name,
                         'eval_pd3qn_fe_r090_taskS_resS_Tight_model1_seeds700-729.csv')
        self.assertEqual(MODEL_NAMES, names)
        with self.assertRaisesRegex(ValueError, 'historical PD3QN'):
            R090Scenario(normalization_mode=PD3QN_UNITS)
        with self.assertRaisesRegex(ValueError, 'fixed, disjoint'):
            R090Scenario(legacy_split=True)

    def test_invalid_partitions_rejected_before_creating_environment(self):
        scene = R090Scenario()
        for seeds, split in (([700], 'validation'), ([600], 'test'),
                             ([600, 600], 'validation'), ([], 'validation')):
            with self.subTest(seeds=seeds, split=split), self.assertRaises(ValueError):
                runtime.evaluate_instances(None, scene, seeds, split)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        cls.temp = tempfile.TemporaryDirectory(prefix='pd3qn_fe_r090_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.scales = cls.root / 'scales.json'
        fixture_config(cls.scales)
        caches = {task: cls.root / f'cache_{task}.json' for task in 'SML'}
        from common.precompute_deadlines import main as precompute
        with redirect_stdout(io.StringIO()):
            for task in 'SML':
                precompute(Scenario(mode='precompute', task=task,
                    seed_spec='0,1,600,700' if task == 'S' else '0,700', cache_output=str(caches[task])))

        class LocalScenario(R090Scenario):
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

        cls.scene = LocalScenario(mode='train', episodes=2, normalization_config=str(cls.scales),
                                  paper_options={'device': 'cpu'})
        cls.eval_scene = replace(cls.scene, mode='eval', seed_spec='700')
        cls.settings = dict(agent_kwargs(cls.scene), hidden_dims=(16, 16), batch_size=8, buffer_size=128)
        cls.created_envs = []
        from algorithms.pd3qn_fe_r090 import train as train_module

        def record_env(**kwargs):
            cls.created_envs.append((kwargs['random_seed'], kwargs['alpha_delay_vm']))
            return CloudWorkflowEnv_PD3QN_Cost(**kwargs)

        with patch.object(runtime, 'agent_kwargs', return_value=cls.settings), \
             patch.object(runtime, 'CloudWorkflowEnv_PD3QN_Cost', side_effect=record_env), \
             patch.object(train_module, 'CloudWorkflowEnv_PD3QN_Cost', side_effect=record_env), \
             redirect_stdout(io.StringIO()):
            cls.agent, cls.summary = train_module.train(cls.scene)

    def test_real_training_weight_metadata_and_checkpoint_roundtrip(self):
        self.assertEqual(self.created_envs, [(0, .9), (600, .9), (600, .9)])
        final = torch.load(self.scene.checkpoint_dir() / 'pd3qn_fe_r090_final.pth', weights_only=True)
        records = [json.loads(line) for line in
                   (self.scene.log_dir() / 'validation_selection.jsonl').read_text().splitlines()]
        expected = min(records, key=lambda item: selection.selection_key(item['validation_selection']))
        best_path = self.scene.checkpoint_dir() / 'best_pd3qn_fe_r090.pth'
        best = torch.load(best_path, weights_only=True)
        self.assertEqual(final['pd3qn_implementation'], 'legacy')
        self.assertEqual(final['network']['input_dim'], 81)
        self.assertGreater(final['updates'], 0)
        self.assertEqual(final['eps_steps'], final['updates'])
        self.assertEqual(final['normalization']['reward'], reward_metadata())
        self.assertEqual(final['normalization'][ALGORITHM]['reward'], reward_metadata())
        self.assertEqual(final['normalization']['model_version'], MODEL_VERSION)
        self.assertEqual(best['normalization'][ALGORITHM]['episode'], expected['episode'])
        self.assertEqual(self.summary['reward'], reward_metadata())
        self.assertNotIn('validation_ESP', self.summary['best_validation_selection'])
        self.assertFalse((self.scene.checkpoint_dir() / 'best_pd3qn_fe.pth').exists())
        runtime.load_best_checkpoint(self.agent, best_path, self.eval_scene)
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key])
                            for key, value in best['online'].items()))
        self.assertTrue(all(torch.isfinite(value).all() for value in final['online'].values()))

    def test_fixed_actions_preserve_physics_but_change_reward_to_09_01(self):
        new = CloudWorkflowEnv_PD3QN_Cost(**runtime.env_kwargs(self.scene))
        old = CloudWorkflowEnv_PD3QN_Cost(**original_runtime.env_kwargs(self.scene))
        obs, _ = new.reset()
        old_obs, _ = old.reset()
        reward_changed = False
        done = False
        while not done:
            np.testing.assert_array_equal(obs, old_obs)
            np.testing.assert_array_equal(new.get_action_mask(), old.get_action_mask())
            action = int(np.flatnonzero(new.get_action_mask())[0])
            obs, reward, terminated, truncated, info = new.step(action)
            old_obs, old_reward, old_terminated, old_truncated, old_info = old.step(action)
            self.assertEqual((terminated, truncated), (old_terminated, old_truncated))
            self.assertAlmostEqual(reward, .9 * info['r_delay'] + .1 * info['r_energy_total'], places=10)
            self.assertAlmostEqual(old_reward, .75 * old_info['r_delay'] + .25 * old_info['r_energy_total'], places=10)
            self.assertEqual(new.current_time, old.current_time)
            reward_changed |= not np.isclose(reward, old_reward)
            done = terminated or truncated
        self.assertTrue(reward_changed)
        self.assertEqual(new.total_energy, old.total_energy)
        self.assertEqual(new.task_end_time, old.task_end_time)
        self.assertEqual(new.records, old.records)

    def test_validation_keeps_agent_weights_and_training_counters(self):
        before = self.agent._eps_steps, self.agent._updates, self.agent._episode_steps, len(self.agent.buffer)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        rows = runtime.evaluate_instances(self.agent, self.scene, [600])
        with patch.dict('os.environ', {'EXPERIMENT_METRICS': '0'}):
            disabled = runtime.evaluate_instances(self.agent, self.scene, [600])
        self.assertEqual(rows, disabled)
        self.assertEqual(rows[0]['wf_completed_count'], 50)
        self.assertEqual(before, (self.agent._eps_steps, self.agent._updates,
                                  self.agent._episode_steps, len(self.agent.buffer)))
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_wrong_weight_rule_and_original_checkpoints_rejected_before_weight_mutation(self):
        source = torch.load(self.scene.checkpoint_dir() / 'best_pd3qn_fe_r090.pth', weights_only=True)
        weights = {key: value.clone() for key, value in self.agent.online.state_dict().items()}
        for field, value in (('algorithm', 'pd3qn_fe'), ('model_version', 'old'),
                             ('reward', dict(reward_metadata(), delay_weight=.75, energy_weight=.25)),
                             ('checkpoint_selection', original_selection.selection_metadata()),
                             ('role', 'final'), ('model_seed', 99)):
            obj = deepcopy(source)
            obj['normalization'][ALGORITHM][field] = value
            path = self.root / 'invalid.pth'
            torch.save(obj, path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                runtime.load_best_checkpoint(self.agent, path, self.eval_scene)
        for key in ('reward', ALGORITHM):
            obj = deepcopy(source)
            obj['normalization'].pop(key)
            torch.save(obj, self.root / 'invalid.pth')
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                runtime.load_best_checkpoint(self.agent, self.root / 'invalid.pth', self.eval_scene)
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            original_runtime.load_best_checkpoint(self.agent,
                self.scene.checkpoint_dir() / 'best_pd3qn_fe_r090.pth', self.eval_scene)
        self.assertTrue(all(torch.equal(value, self.agent.online.state_dict()[key]) for key, value in weights.items()))

    def test_three_target_sizes_use_r090_in_all_evaluation_environments(self):
        from algorithms.pd3qn_fe_r090 import evaluate as evaluate_module
        from algorithms.pd3qn_fe_r090.train import train
        created = []

        def record_env(**kwargs):
            created.append((kwargs['random_seed'], kwargs['alpha_delay_vm']))
            return CloudWorkflowEnv_PD3QN_Cost(**kwargs)

        sources = set()
        with patch.object(runtime, 'agent_kwargs', return_value=self.settings), \
             patch.object(runtime, 'CloudWorkflowEnv_PD3QN_Cost', side_effect=record_env), \
             patch.object(evaluate_module, 'CloudWorkflowEnv_PD3QN_Cost', side_effect=record_env), \
             redirect_stdout(io.StringIO()):
            for task in 'SML':
                scene = replace(self.eval_scene, task=task)
                rows, summary = evaluate_module.evaluate(scene)
                self.assertEqual(rows[0]['wf_count'], 50)
                self.assertEqual(rows[0]['wf_completed_count'], 50)
                self.assertEqual(summary['algorithm'], ALGORITHM)
                self.assertEqual(summary['reward'], reward_metadata())
                self.assertTrue(scene.evaluation_csv.is_file())
                self.assertTrue((scene.evaluation_dir / 'measurements/episodes/seed700.json').is_file())
                sources.add(summary['source_checkpoint'])
                with self.assertRaises(FileExistsError):
                    evaluate_module.evaluate(scene)
            with self.assertRaises(FileExistsError):
                train(self.scene)
        self.assertEqual(len(sources), 1)
        self.assertEqual(created, [(0, .9), (700, .9)] * 3)


if __name__ == '__main__':
    unittest.main()
