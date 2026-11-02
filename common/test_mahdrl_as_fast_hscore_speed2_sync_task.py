"""Exact sync-task parity, real learning across cycles and isolated checkpoints."""
from contextlib import redirect_stdout
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch

from common import test_mahdrl_as_fast_hscore as fixtures
from common.experiment_protocol import protocol_binding
from common.f18_state_normalization import normalization_diagnostics
from algorithms.mahdrl_as_fast_hscore import runtime as base, agent as original_agents
from algorithms.mahdrl_as_fast_hscore_ablation import runtime as original
from algorithms.mahdrl_as_fast_hscore_sync_task.scenario import ScoreScenario as OriginalScenario
from algorithms.mahdrl_as_fast_hscore_speed import agent as speed_agents
from algorithms.mahdrl_as_fast_hscore_speed.replay import IndexedReplay
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import ScoreScenario as AsyncScenario
from algorithms.mahdrl_as_fast_hscore_speed2.environment import CloudWorkflowEnv_VMAgents as Speed2Env
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task import runtime
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task.scenario import ScoreScenario, parse_scenario


def tiny_agents(scene, *, old=False):
    module = original_agents if old else speed_agents
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8,
                   device=scene.device or 'cpu', profile_mode='off')
    agents = dict(TA=module.TAAgent(15, 243, **options),
        HA=module.HAAgent(23, scene.mechanism.host_features, entity_dims=(8, 8), **options),
        VA=module.FixedVAAgent(**options))
    return {role: agent.bind(scene, role) for role, agent in agents.items()}


def learning_state(agents):
    return {role: (id(a), id(a.buffer), id(a.optim), a._updates, a._eps_steps,
                   a.eligible_steps, len(a.buffer), a.epsilon()) for role, a in agents.items()}


class SyncTaskTests(unittest.TestCase):
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

    def env(self, scene, seed, *, old=False):
        with patch.object(base, 'build_env_kwargs', fixtures.HScoreTests.small_kwargs(base.build_env_kwargs)):
            return (original.build_env if old else runtime.build_env)(scene, seed)

    def assert_nested_equal(self, a, b):
        if isinstance(a, Mapping):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        elif isinstance(a, (np.ndarray, torch.Tensor)):
            np.testing.assert_array_equal(a.cpu().numpy() if isinstance(a, torch.Tensor) else a,
                                          b.cpu().numpy() if isinstance(b, torch.Tensor) else b)
        else:
            self.assertEqual(a, b)

    def compare_sync_rollout(self, device='cpu', preview=True):
        scene = self.scene(device=device, va_preview=preview)
        observed, learners = [], []
        for old in (True, False):
            base.set_seed(17)
            agents, env = tiny_agents(scene, old=old), self.env(scene, 0, old=old)
            assignments, phases, updates, observations = [], [], [], []
            for name in ('_build_host_obs_for_task', '_build_vm_obs_for_task_host', 'get_manager_state'):
                fn = getattr(env, name)
                def observe(*args, fn=fn, name=name, **kwargs):
                    value = fn(*args, **kwargs)
                    observations.append((name, deepcopy(value)))
                    return value
                setattr(env, name, observe)
            for role, agent in agents.items():
                fn = agent.update
                def update(fn=fn, role=role):
                    updates.append((role, len(assignments), len(phases)))
                    return fn()
                agent.update = update

            def assigned(row):
                self.assertEqual(row['va_version'], agents['VA']._updates)
                assignments.append(row)

            with redirect_stdout(io.StringIO()):
                env.reset()
                result = runtime.run_episode(env, agents, training=True, warmup=0,
                    assignment_log=assigned, phase_log=phases.append)
            self.assertEqual(result['decision_counts']['TA'], len(assignments))
            self.assertEqual(result['update_clock']['transitions']['TA'], len(assignments))
            self.assertEqual(result['update_clock']['update_requests'], result['update_clock']['optimizer_steps'])
            self.assertEqual(agents['HA']._updates, agents['HA'].eligible_steps//scene.ha_update_period)
            self.assertTrue(all(a._updates > 0 for a in agents.values()))
            self.assertTrue(all(count > 0 for role, count, phase in updates))
            for key, value in result['reward_audit'].items():
                if key.endswith('balance'):
                    self.assertAlmostEqual(value, 0., places=6, msg=key)
            observed.append((result, assignments, phases, updates, observations, env._records,
                             normalization_diagnostics(env), random.getstate(), np.random.get_state(), torch.get_rng_state()))
            learners.append(agents)
            if not old:
                self.assertIsInstance(env, Speed2Env)
                self.assertTrue(hasattr(env, '_physical_prefix'))
        self.assert_nested_equal(observed[0], observed[1])
        for role in learners[0]:
            a, b = learners[0][role], learners[1][role]
            self.assert_nested_equal(list(a.buffer), list(b.buffer))
            self.assert_nested_equal(a.online.state_dict(), b.online.state_dict())
            self.assert_nested_equal(a.target.state_dict(), b.target.state_dict())
            self.assert_nested_equal(a.optim.state_dict(), b.optim.state_dict())
            self.assertEqual((a._updates, a._eps_steps, a._action_calls, a.eligible_steps),
                             (b._updates, b._eps_steps, b._action_calls, b.eligible_steps))

    def test_cpu_exact_sync_task_parity_with_preview_on_and_off(self):
        self.compare_sync_rollout()
        self.compare_sync_rollout(preview=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_exact_sync_task_learning_parity(self):
        self.compare_sync_rollout('cuda')

    def test_default_builders_identity_and_parser(self):
        scene = self.scene()
        agents = runtime.build_agents(scene)
        self.assertTrue(all(isinstance(a.buffer, IndexedReplay) for a in agents.values()))
        env = self.env(scene, 0)
        self.assertIs(env._compute_block2.__func__, Speed2Env._compute_block2)
        self.assertIs(env._energy_reward_to_current_time_with_breakdown.__func__,
                      Speed2Env._energy_reward_to_current_time_with_breakdown)
        self.assertIs(runtime.run_episode, original.run_episode)
        metadata, old = scene.variant_metadata(), OriginalScenario().variant_metadata()
        for key in old:
            if key not in ('algorithm', 'scheme', 'model_version', 'base_implementation'):
                self.assertEqual(metadata[key], old[key], key)
        self.assertEqual(metadata['update_schedule'], 'TA_per_allocation_barrier_after_assignment')
        for mode in ('train', 'eval', 'precompute'):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as caught:
                parse_scenario(mode, ['--dry-run'])
            self.assertEqual(caught.exception.code, 0)
            self.assertEqual(json.loads(out.getvalue())['algorithm'], scene.algorithm)

    def test_real_cyclic_train_load_frozen_evaluation_and_identity_guards(self):
        protocol = self.root/'sync_two_seeds.json'
        protocol.write_text(json.dumps(dict(version='test_sync_cycles', train=[0, 1], validation=[600],
            test=[700], episodes=2, selection='minimum_validation_total_energy')), encoding='utf8')
        calibration = json.loads(self.scales.read_text(encoding='utf8'))
        calibration['generation'] = dict(**protocol_binding(protocol), calibration_seeds={'SS_T': [0, 1]})
        scales = self.root/'sync_cycle_scales.json'
        scales.write_text(json.dumps(calibration), encoding='utf8')
        scene = self.scene(episodes=4, train_seed_cycles=2, legacy_split=False,
                           experiment_protocol=str(protocol), normalization_config=str(scales))
        seeds, starts, ends = [], [], []

        def rollout(env, agents, **kwargs):
            if kwargs.get('training'):
                seeds.append(env.random_seed)
                starts.append(learning_state(agents))
            result = runtime.run_episode(env, agents, warmup=0, **kwargs)
            if kwargs.get('training'):
                ends.append(learning_state(agents))
            return result

        with redirect_stdout(io.StringIO()):
            runtime.train(scene, env_builder=self.env, agent_builder=tiny_agents, rollout=rollout)
            target = replace(scene, mode='eval', seed_spec='700', train_resource='S', train_deadline='Tight')
            loaded = runtime.load_agents(target, agent_builder=tiny_agents)
            before = deepcopy({r: a.online.state_dict() for r, a in loaded.items()}), learning_state(loaded)
            env = self.env(target, 700)
            env.reset()
            runtime.run_episode(env, loaded)
            self.assert_nested_equal(before, ({r: a.online.state_dict() for r, a in loaded.items()}, learning_state(loaded)))
            runtime.evaluate(target, env_builder=self.env, agent_builder=tiny_agents)
            final = tiny_agents(target)['TA']
            final.load(str(scene.checkpoint_dir()/'manager_final.pth'))
        self.assertEqual(seeds, [0, 1, 0, 1])
        self.assertEqual(starts[1:], ends[:-1])
        self.assertTrue(all(ends[-1][r][3] > ends[1][r][3] for r in loaded))
        self.assertEqual(final.loaded_selection['completed_episodes'], 4)
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        with target.evaluation_csv().open(encoding='utf8', newline='') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual((row['training_episodes'], row['train_seed_cycles']), ('4', '2'))
        self.assertEqual((row['source_scene'], row['target_scene']), ('SS_T', 'SS_T'))
        self.assertEqual(target.evaluation_csv().parent, target.evaluation_root)
        for cls in (OriginalScenario, AsyncScenario):
            other = cls(mode='eval', legacy_split=True, device='cpu', normalization_config=str(scales))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                tiny_agents(other)['VA'].load(str(scene.checkpoint_dir()/'best_vm.pth'))
        with self.assertRaises(FileExistsError):
            runtime.train(scene, env_builder=self.env, agent_builder=tiny_agents)
        with self.assertRaises(FileExistsError):
            runtime.evaluate(target, env_builder=self.env, agent_builder=tiny_agents)


if __name__ == '__main__':
    unittest.main()
