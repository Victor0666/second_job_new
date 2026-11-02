"""Observe exact inputs, physics and parameter updates against Speed v1."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common import test_mahdrl_as_fast_hscore as fixtures
from common import test_mahdrl_as_fast_hscore_speed as speed_tests
from common.f18_state_normalization import normalization_diagnostics
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore_speed.scenario import ScoreScenario as SpeedScenario
from algorithms.mahdrl_as_fast_hscore_speed2 import runtime
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import ScoreScenario, parse_scenario


class Speed2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HScoreTests.setUpClass()
        cls.root = fixtures.HScoreTests.root
        cls.scales, cls.cache = fixtures.HScoreTests.scales, fixtures.HScoreTests.cache

    @classmethod
    def tearDownClass(cls):
        fixtures.HScoreTests.tearDownClass()

    def scene(self, **kwargs):
        return ScoreScenario(**dict(dict(mode='train', episodes=2, legacy_split=True, device='cpu',
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName)), **kwargs))

    def env(self, scene, seed, *, original=False):
        with patch.object(base, 'build_env_kwargs', fixtures.HScoreTests.small_kwargs(base.build_env_kwargs)):
            return (base.build_env if original else runtime.build_env)(scene, seed)

    def compare_rollout(self, device='cpu', preview=True):
        scene = self.scene(device=device, va_preview=preview)
        results, rows, observations, records, agents, normalizations = [], [], [], [], [], []
        for original in (True, False):
            base.set_seed(11)
            learners = speed_tests.tiny_agents(scene)
            env = self.env(scene, 0, original=original)
            captured = []
            for name in ('_build_host_obs_for_task', '_build_vm_obs_for_task_host'):
                fn = getattr(env, name)
                def observe(*args, fn=fn, tag=name, **kwargs):
                    obs, mask = fn(*args, **kwargs)
                    captured.append((tag, obs.copy(), mask.copy()))
                    return obs, mask
                setattr(env, name, observe)
            with redirect_stdout(io.StringIO()):
                env.reset()
                assignments = []
                result = base.run_episode(env, learners, training=True, warmup=0,
                                          assignment_log=assignments.append)
            results.append(result)
            rows.append(assignments)
            observations.append(captured)
            records.append(env._records)
            agents.append(learners)
            normalizations.append(normalization_diagnostics(env))
        self.assertEqual(results[0], results[1])
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(records[0], records[1])
        self.assertEqual(normalizations[0], normalizations[1])
        self.assertEqual(len(observations[0]), len(observations[1]))
        for a, b in zip(*observations):
            self.assertEqual(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])
        for role in agents[0]:
            for name in ('online', 'target'):
                for key, value in getattr(agents[0][role], name).state_dict().items():
                    self.assertTrue(torch.equal(value, getattr(agents[1][role], name).state_dict()[key]), role+key)

    def test_cpu_rollout_exact_parity_with_preview_on_and_off(self):
        self.compare_rollout()
        self.compare_rollout(preview=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_real_rollout_updates_exact_parity(self):
        self.compare_rollout('cuda')

    def test_normalized_and_legacy_queue_and_reset(self):
        scene = self.scene()
        for normalized in (True, False):
            old, new = self.env(scene, 0, original=True), self.env(scene, 0)
            for seed in (0, 1):
                with redirect_stdout(io.StringIO()):
                    old.reset(seed=seed)
                    new.reset(seed=seed)
                if not normalized:
                    old.state_normalizer = new.state_normalizer = None
                a, oka = old.get_host_state_for_next_assignment()
                b, okb = new.get_host_state_for_next_assignment()
                self.assertEqual(oka, okb)
                for key in a['obs']:
                    np.testing.assert_array_equal(a['obs'][key], b['obs'][key])

    def test_train_reload_evaluate_paths_identity_and_guards(self):
        scene = self.scene()
        with redirect_stdout(io.StringIO()):
            runtime.train(scene, env_builder=self.env, agent_builder=speed_tests.tiny_agents)
            evaluation = replace(scene, mode='eval', train_resource='S', train_deadline='Tight', seed_spec='700')
            loaded = base.load_agents(evaluation, agent_builder=speed_tests.tiny_agents)
            runtime.evaluate(evaluation, env_builder=self.env, agent_builder=speed_tests.tiny_agents)
        self.assertTrue(evaluation.evaluation_csv().is_file())
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        self.assertNotEqual(scene.checkpoint_dir(), SpeedScenario().checkpoint_dir())
        old_scene = SpeedScenario(mode='eval', legacy_split=True, device='cpu',
            normalization_config=str(self.scales), cache_file=str(self.cache),
            train_resource='S', train_deadline='Tight')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            speed_tests.tiny_agents(old_scene)['VA'].load(str(scene.checkpoint_dir()/'best_vm.pth'))
        with self.assertRaises(FileExistsError):
            runtime.train(scene, env_builder=self.env, agent_builder=speed_tests.tiny_agents)
        for mode in ('train', 'eval', 'precompute'):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                parse_scenario(mode, ['--dry-run'])
            self.assertEqual(caught.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
