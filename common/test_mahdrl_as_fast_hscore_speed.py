"""Differential checks against the unmodified HScore v2 implementation."""
from collections import deque
from contextlib import redirect_stdout
from dataclasses import replace
import io
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from algorithms.mahdrl_as_fast_hscore import agent as original, runtime as base
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as OriginalScenario
from algorithms.mahdrl_as_fast_hscore.observation import validate_fixed
from algorithms.mahdrl_as_fast_hscore_speed import agent, runtime
from algorithms.mahdrl_as_fast_hscore_speed.replay import IndexedReplay
from algorithms.mahdrl_as_fast_hscore_speed.scenario import ScoreScenario, parse_scenario
from common import test_mahdrl_as_fast_hscore as fixtures


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=4, buffer_size=11,
                   device=scene.device or "cpu", profile_mode="off")
    learners = dict(TA=agent.TAAgent(15, 243, **options),
        HA=agent.HAAgent(23, scene.mechanism.host_features, entity_dims=(8, 8), **options),
        VA=agent.FixedVAAgent(**options))
    return {role: learner.bind(scene, role) for role, learner in learners.items()}


class ReplayTests(unittest.TestCase):
    def test_order_eviction_and_exact_sample_rng(self):
        for capacity in (1, 7, 1000):
            old, new = deque(maxlen=capacity), IndexedReplay(capacity)
            for i in range(capacity*3+2):
                old.append(i)
                new.append(i)
                if i in (0, capacity-1, capacity, capacity*3+1):
                    self.assertEqual(list(old), list(new))
                    self.assertEqual(old[-1], new[-1])
                    self.assertEqual(old[0], new[0])
                    for k in sorted({1, min(256, len(old)), len(old)}):
                        random.seed(71)
                        expected = random.sample(old, k)
                        rng = random.getstate()
                        random.seed(71)
                        self.assertEqual(expected, random.sample(new, k))
                        self.assertEqual(rng, random.getstate())
            for index in (len(new), -len(new)-1):
                with self.assertRaises(IndexError):
                    new[index]

    def test_batch_rejects_same_invalid_states_as_original(self):
        state, mask = np.zeros(93, np.float32), np.ones(9, np.float32)
        for field, value in ((0, np.full(93, np.nan)), (1, np.ones(8)),
                             (1, np.full(9, .5)), (1, np.zeros(9)),
                             (4, np.full(93, np.inf)), (5, np.zeros(9))):
            row = [state.copy(), mask.copy(), 0, -.1, state.copy(), mask.copy(), False]
            row[field] = value
            with self.subTest(field=field, shape=value.shape):
                with self.assertRaises(ValueError):
                    validate_fixed(row[0], row[1])
                    validate_fixed(row[4], row[5])
                with self.assertRaises(ValueError):
                    agent.pack_fixed_batch([row, row])
        terminal = (state, mask, 0, -.1, state, np.zeros(9), True)
        packed = agent.pack_fixed_batch([terminal])
        self.assertEqual(packed[4].shape, (0, 93))


class DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixtures.HScoreTests.setUpClass()
        cls.root = fixtures.HScoreTests.root
        cls.scales, cls.cache = fixtures.HScoreTests.scales, fixtures.HScoreTests.cache

    @classmethod
    def tearDownClass(cls):
        fixtures.HScoreTests.tearDownClass()

    def scene(self, **kwargs):
        return ScoreScenario(**dict(dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName)), **kwargs))

    def env(self, scene, seed):
        with patch.object(base, "build_env_kwargs", fixtures.HScoreTests.small_kwargs(base.build_env_kwargs)):
            return base.build_env(scene, seed)

    def compare_update(self, device, *, production=False, terminal=False, tau=.005):
        rng = np.random.default_rng(42)
        options = dict(device=device, batch_size=256 if production else 4,
                       buffer_size=271 if production else 11, profile_mode="off",
                       target_update_tau=tau, target_update_freq=2)
        if not production:
            options["hidden_dims"] = (16, 8)
        base.set_seed(1)
        old = original.FixedVAAgent(**options)
        base.set_seed(1)
        new = agent.FixedVAAgent(**options)
        for i in range(options["buffer_size"]*2+3):
            s, n = rng.normal(size=(2, 93)).astype(np.float32)
            m, nm = rng.integers(0, 2, size=(2, 9)).astype(np.float32)
            m[0] = 1
            done = terminal or i % 5 == 0
            if done:
                nm[:] = 0
            else:
                nm[1] = 1
            for learner in (old, new):
                learner.remember(s, m, 0, -.1, n, nm, done)
        calls = []
        hook = new.target.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
        try:
            for seed in (2, 3, 4):
                base.set_seed(seed)
                expected = old.update()
                rng_state = random.getstate()
                base.set_seed(seed)
                self.assertEqual(expected, new.update())
                self.assertEqual(rng_state, random.getstate())
                for name in ("online", "target"):
                    for key, value in getattr(old, name).state_dict().items():
                        self.assertTrue(torch.equal(value, getattr(new, name).state_dict()[key]), key)
                self.assertEqual((old._updates, old._eps_steps, old.epsilon()),
                                 (new._updates, new._eps_steps, new.epsilon()))
                for a, b in zip(old.optim.state.values(), new.optim.state.values()):
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual(old.f11.snapshot(), new.f11.snapshot())
            if terminal:
                self.assertEqual(calls, [])
        finally:
            hook.remove()

    def test_cpu_update_exact_parity_and_hard_target(self):
        self.compare_update("cpu")
        self.compare_update("cpu", terminal=True, tau=0.)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_production_width_and_batch_exact_parity(self):
        self.compare_update("cuda", production=True)
        self.compare_update("cuda", terminal=True)

    def test_training_rollout_actions_physics_and_all_learners_match(self):
        scene = self.scene()
        old_options = dict(hidden_dims=(16, 8), batch_size=4, buffer_size=11, device="cpu", profile_mode="off")
        base.set_seed(7)
        old = dict(TA=original.TAAgent(15, 243, **old_options),
            HA=original.HAAgent(23, 14, entity_dims=(8, 8), **old_options),
            VA=original.FixedVAAgent(**old_options))
        for role, learner in old.items():
            learner.bind(scene, role)
        base.set_seed(7)
        new = tiny_agents(scene)
        results, rows, records = [], [], []
        for learners in (old, new):
            base.set_seed(9)
            env = self.env(scene, 0)
            with redirect_stdout(io.StringIO()):
                env.reset()
                assignments = []
                result = base.run_episode(env, learners, training=True, warmup=0,
                                          assignment_log=assignments.append)
            results.append(result)
            rows.append(assignments)
            records.append(env._records)
        self.assertEqual(results[0], results[1])
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(records[0], records[1])
        for role in old:
            for key, value in old[role].online.state_dict().items():
                self.assertTrue(torch.equal(value, new[role].online.state_dict()[key]), role+key)
            for transition in new[role].buffer:
                self.assertFalse(transition[1].flags.writeable)

    def test_train_reload_eval_identity_and_overwrite_guard(self):
        scene = self.scene()
        with redirect_stdout(io.StringIO()):
            runtime.train(scene, env_builder=self.env, agent_builder=tiny_agents)
            evaluation = replace(scene, mode="eval", train_resource="S", train_deadline="Tight", seed_spec="700")
            loaded = base.load_agents(evaluation, agent_builder=tiny_agents)
            result = runtime.evaluate(evaluation, env_builder=self.env, agent_builder=tiny_agents)
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        self.assertTrue(evaluation.evaluation_csv().is_file())
        self.assertTrue(np.isfinite(result["total_energy"]["mean"]))
        with self.assertRaises(FileExistsError):
            runtime.train(scene, env_builder=self.env, agent_builder=tiny_agents)
        with self.assertRaises(FileExistsError):
            runtime.evaluate(evaluation, env_builder=self.env, agent_builder=tiny_agents)
        old_scene = OriginalScenario(mode="eval", legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            train_resource="S", train_deadline="Tight")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fixtures.tiny_agents(old_scene)["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"))

    def test_parser_paths_metadata_and_original_defaults(self):
        current, old = parse_scenario("train", []), OriginalScenario()
        self.assertNotEqual(current.checkpoint_dir(), old.checkpoint_dir())
        self.assertNotEqual(current.evaluation_root, old.evaluation_root)
        self.assertEqual(current.mechanism, old.mechanism)
        self.assertEqual(current.validation_seeds(), old.validation_seeds())
        scientific = current.variant_metadata()
        for key, value in old.variant_metadata().items():
            if key not in ("algorithm", "scheme", "model_version", "base_implementation"):
                self.assertEqual(scientific[key], value, key)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            parse_scenario("train", ["--dry-run"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
