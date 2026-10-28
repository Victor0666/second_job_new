"""NTA policy isolation, retained learners, provenance and real train/eval flow."""
from common.test_mahdrl_ablation_settings import raw_physics, assert_flat_csv
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.experiment_protocol import protocol_binding
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, MODEL_NAMES
from common.test_f18 import fixture_config
from common.test_mahdrl_as_fast import tiny_agents as base_tiny_agents, validation_row
from common.test_mahdrl_oe_ge_apf_score import parameter_hash, observation
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as base_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nta import runtime, agent, config, selection
from algorithms.mahdrl_oe_ge_apf_score_fast_nta.policy import RandomWeightPolicy, manager_seed
from algorithms.mahdrl_oe_ge_apf_score_fast_nta.scenario import ScoreScenario, parse_scenario


def tiny_agents(scene):
    return {role: agent.SetD3QNAgent(c, f, hidden_dims=(16, 8), entity_dims=(8, 8),
                batch_size=2, buffer_size=8, device="cpu").bind(scene, role)
            for role, c, f in (("HA", 23, 10), ("VA", 30, 7))}


class NTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="as_nta_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=ScoreScenario, **kwargs):
        values = dict(mode="train", episodes=1, legacy_split=True,
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/"run"), device="cpu")
        return klass(**dict(values, **kwargs))

    def small_kwargs(self, original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene=None, seed=0):
        with patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(runtime.build_env_kwargs)):
            env = runtime.build_env(scene or self.scene(), seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_random_policy_is_uniform_legal_reproducible_and_rng_isolated(self):
        py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        mask = np.zeros(243); mask[[7, 45, 121]] = 1
        a, b = RandomWeightPolicy(700), RandomWeightPolicy(700)
        actions = [a.select_action(mask) for _ in range(6000)]
        self.assertEqual(actions, [b.select_action(mask) for _ in range(6000)])
        self.assertTrue(set(actions) <= {7, 45, 121})
        for value in (7, 45, 121):
            self.assertTrue(1800 < actions.count(value) < 2200)
        self.assertEqual(a.summary()["decisions"], 6000)
        self.assertEqual(a.seed, (700*1000003+20260325+9176) & 0xFFFFFFFF)
        self.assertNotEqual(a.seed, RandomWeightPolicy(701).seed)
        self.assertEqual(py_before, random.getstate())
        current = np.random.get_state()
        self.assertEqual(np_before[0], current[0]); np.testing.assert_array_equal(np_before[1], current[1])
        self.assertEqual(np_before[2:], current[2:])
        torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
        for bad in (np.zeros(243), np.ones(242), np.full(243, np.nan), np.full(243, 2.)):
            with self.assertRaises(ValueError):
                a.select_action(bad)

    def test_real_weight_walk_preserves_constraints_and_invalidates_cache(self):
        env = self.env()
        before_rng = repr(env.rng.get_state())
        policy = RandomWeightPolicy(0)
        changed = 0
        for _ in range(300):
            before = env.combo_weights.copy()
            mask = env.get_manager_action_mask()
            action = policy.select_action(mask)
            self.assertEqual(mask[action], 1.)
            env._fast_cache["sentinel"] = 1
            env.apply_manager_delta(runtime.MANAGER_ACTION_TABLE[action])
            self.assertEqual(env._fast_cache, {})
            self.assertTrue(np.isfinite(env.combo_weights).all())
            self.assertTrue((env.combo_weights >= 0).all())
            self.assertAlmostEqual(float(env.combo_weights.sum()), 1., places=6)
            changed += int(not np.array_equal(before, env.combo_weights))
        self.assertGreater(changed, 0)
        self.assertEqual(before_rng, repr(env.rng.get_state()))

    def test_formal_initial_HA_VA_tensors_and_rng_states_match_baseline(self):
        scene, original = self.scene(), self.scene(BaseScenario)
        base_runtime.set_seed(31)
        baseline = base_runtime.build_agents(original)
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        runtime.set_seed(31)
        retained = runtime.build_agents(scene)
        self.assertEqual(set(retained), {"HA", "VA"})
        for role in retained:
            self.assertIs(type(retained[role])._update, type(baseline[role])._update)
            self.assertEqual(config.learning_kwargs(role, "cpu"), base_runtime.learning_kwargs(role, "cpu"))
            for network in ("online", "target"):
                for left, right in zip(getattr(retained[role], network).parameters(), getattr(baseline[role], network).parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(py_rng, random.getstate())
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
        self.assertEqual(np_rng[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_real_evaluation_recreates_policy_and_preserves_parameters(self):
        scene = self.scene()
        learners = tiny_agents(scene)
        weights = parameter_hash(learners)
        results, logs = [], []
        for _ in range(2):
            env = self.env(scene, 700)
            phases = []
            with redirect_stdout(io.StringIO()):
                result = runtime.run_episode(env, learners, phase_log=phases.append)
            self.assertEqual(result["random_ta"]["decisions"], result["phase_count"])
            self.assertEqual(result["random_ta"]["seed"], manager_seed(700))
            self.assertEqual(result["wf_count"], 2)
            self.assertTrue(all(getattr(env, k) == .75 for k in ("manager_alpha_delay", "alpha_delay_host", "alpha_delay_vm")))
            self.assertTrue(all(p["ta_legal_actions"] > 0 for p in phases))
            self.assertTrue(all(a._updates == 0 for a in learners.values()))
            results.append(result); logs.append(phases)
        self.assertEqual(results[0], results[1])
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(parameter_hash(learners), weights)

    def test_SR092_selector_is_strict_and_does_not_change_full_model(self):
        self.assertIsNot(selection.validation_selection, base_runtime.validation_selection)
        self.assertFalse(selection.validation_selection([validation_row(46)])["feasible"])
        self.assertTrue(selection.validation_selection([validation_row(47)])["feasible"])
        self.assertFalse(base_runtime.validation_selection([validation_row(47)])["feasible"])
        low = selection.validation_selection([validation_row(46, 1.)])
        high = selection.validation_selection([validation_row(47, 100.)])
        self.assertLess(selection.selection_key(high), selection.selection_key(low))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_retained_cuda_learners_perform_real_updates(self):
        for role, context, features in (("HA", 23, 10), ("VA", 30, 7)):
            learner = agent.SetD3QNAgent(context, features, hidden_dims=(16, 8), entity_dims=(8, 8),
                batch_size=2, buffer_size=8, device="cuda").bind(self.scene(), role)
            before = parameter_hash({role: learner})
            for terminal in (False, True):
                state, successor = observation(3, context, features), observation(0 if terminal else 4, context, features)
                learner.remember(state, np.ones(3), 0, -.2, successor, np.ones(0 if terminal else 4), terminal)
            self.assertTrue(np.isfinite(learner.update()))
            self.assertEqual(learner._updates, 1)
            self.assertNotEqual(parameter_hash({role: learner}), before)

    def test_checkpoint_policy_identity_roles_groups_and_final_rejection(self):
        scene, old_scene = self.scene(), self.scene(BaseScenario)
        learners, baseline = tiny_agents(scene), base_tiny_agents(old_scene)
        rows = [validation_row(seed=scene.validation_seeds()[0])]
        candidate = selection.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        args = dict(episode=0, candidate=candidate, best=best, rows=rows, best_rows=rows)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory); new_dir, old_dir = root/"nta", root/"baseline"
            new_dir.mkdir(); old_dir.mkdir()
            runtime.save_agents(learners, new_dir, "best", **args)
            base_runtime.save_agents(baseline, old_dir, "best", **args)
            target = replace(scene, mode="eval", checkpoint_override=str(new_dir))
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                loaded = runtime.load_agents(target)
            self.assertEqual(parameter_hash(learners), parameter_hash(loaded))
            self.assertFalse((new_dir/"best_manager.pth").exists())
            for role, name in runtime.FILES.items():
                with self.assertRaises(ValueError): learners[role].load(str(old_dir/f"best_{name}.pth"))
                with self.assertRaises(ValueError): baseline[role].load(str(new_dir/f"best_{name}.pth"))
            with self.assertRaises(ValueError): learners["HA"].load(str(new_dir/"best_vm.pth"))
            with self.assertRaisesRegex(ValueError, "strict"): learners["HA"].load(str(new_dir/"best_host.pth"), strict=False)
            checkpoint = torch.load(new_dir/"best_host.pth", weights_only=True)
            for section in (None, "algorithm_variant", "implementation", config.CHECKPOINT_METADATA_KEY):
                tampered = deepcopy(checkpoint)
                meta = tampered["normalization"] if section is None else tampered["normalization"][section]
                meta["ta_policy"]["version"] = "foreign_policy"
                path = root/"tampered.pth"; torch.save(tampered, path)
                with self.subTest(section=section), self.assertRaises(ValueError):
                    learners["HA"].load(str(path))
            manifest = new_dir/"ta_policy.json"
            original = manifest.read_text()
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "policy manifest"):
                runtime.load_agents(target)
            manifest.write_text(original)
            learners["HA"].checkpoint_group = "foreign_group"
            learners["HA"].save(str(new_dir/"best_host.pth"))
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                with self.assertRaisesRegex(ValueError, "different saved groups"):
                    runtime.load_agents(target)
            runtime.save_agents(learners, new_dir, "final", episode=1, candidate=candidate,
                                best=best, rows=rows, best_rows=rows)
            for name in runtime.FILES.values():
                (new_dir/f"best_{name}.pth").write_bytes((new_dir/f"{name}_final.pth").read_bytes())
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                with self.assertRaisesRegex(ValueError, "requires best"):
                    runtime.load_agents(target)

    def test_real_training_selection_measurement_flags_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory); protocol, scales = root/"protocol.json", root/"scales.json"
            protocol.write_text(json.dumps(dict(version="as_nta_smoke_v1", train=[0, 1], validation=[600],
                test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
            scale_config = fixture_config(scales)
            scale_config["generation"] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in scale_config["scenes"]})
            scales.write_text(json.dumps(scale_config), encoding="utf-8")
            original_kwargs, original_episode = runtime.build_env_kwargs, runtime.run_episode
            runs = []
            for disabled in (False, True):
                scene = self.scene(output_root=str(root/("disabled" if disabled else "enabled")), episodes=2,
                    legacy_split=False, experiment_protocol=str(protocol), normalization_config=str(scales))
                snapshots, evaluations = [], 0
                def rollout(env, agents, **kwargs):
                    nonlocal evaluations
                    value = original_episode(env, agents, **kwargs)
                    if kwargs.get("training", False):
                        snapshots.append(parameter_hash(agents))
                    else:
                        # Episode 1 has lower energy but fails the workflow-SR gate.
                        success, energy = [(2, 100.), (1, 1.)][evaluations]
                        evaluations += 1
                        value.update(total_energy=energy, wf_success_count=success, wf_success_rate=success/2)
                    return value
                flags = {key: "0" if disabled else "1" for key in ("EXPERIMENT_METRICS", "F11_DIAGNOSTICS", "F19_DIAGNOSTICS")}
                with patch.dict(os.environ, flags), patch.object(runtime, "build_agents", side_effect=tiny_agents), \
                     patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(original_kwargs)):
                    with redirect_stdout(io.StringIO()), patch.object(runtime, "run_episode", side_effect=rollout):
                        trained = runtime.train(scene)
                    target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                    loaded = runtime.load_agents(target)
                    self.assertEqual(parameter_hash(loaded), snapshots[0])
                    self.assertEqual(parameter_hash(trained), snapshots[1])
                    self.assertTrue(all(a._updates > 0 for a in trained.values()))
                    before = parameter_hash(loaded)
                    with redirect_stdout(io.StringIO()), patch.object(runtime, "load_agents", return_value=loaded):
                        summary = runtime.evaluate(target)
                    self.assertEqual(parameter_hash(loaded), before)
                    self.assertIn("total_energy", summary)
                    with self.assertRaises(FileExistsError): runtime.train(scene)
                    assert_flat_csv(self, target)
                    with self.assertRaises(FileExistsError): runtime.evaluate(target)
                selection_log = [json.loads(line) for line in (scene.log_dir()/"validation_selection.jsonl").read_text().splitlines()]
                training = json.loads((scene.log_dir()/"training_summary.json").read_text())
                self.assertEqual(training["selected_episode"], 0)
                self.assertTrue(training["feasible_model_found"])
                self.assertEqual(set(training["parameter_counts"]), {"HA", "VA"})
                self.assertEqual({a.loaded_selection["episode"] for a in loaded.values()}, {0})
                self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
                self.assertEqual([r["selected_as_best"] for r in selection_log], [True, False])
                self.assertEqual(sorted(p.name for p in scene.checkpoint_dir().glob("*.pth")),
                                 ["best_host.pth", "best_vm.pth", "host_final.pth", "vm_final.pth"])
                episode_rows = [json.loads(line) for line in (scene.log_dir()/"episodes.jsonl").read_text().splitlines()]
                self.assertEqual([r["random_ta"]["seed"] for r in episode_rows], [manager_seed(0), manager_seed(1)])
                self.assertTrue(all(r["random_ta"]["decisions"] == r["phase_count"] for r in episode_rows))
                for name in runtime.FILES.values():
                    saved = torch.load(scene.checkpoint_dir()/f"{name}_final.pth", weights_only=True)
                    self.assertEqual(saved["normalization"][config.CHECKPOINT_METADATA_KEY]["episode"], 1)
                if not disabled:
                    measurement = json.loads((scene.log_dir()/"training_measurements.jsonl").read_text().splitlines()[0])
                    self.assertTrue(all(v > 0 for v in measurement["optimizer_steps"].values()))
                    self.assertEqual(set(measurement["policy_calls"]), {"HA", "VA"})
                    evaluation = json.loads((target.evaluation_dir()/"measurements/episodes/seed700.json").read_text())
                    self.assertEqual(set(evaluation["timing"]["policy_call_counts"]), {"HA", "VA", "RandomTA"})
                    self.assertGreater(evaluation["timing"]["policy_call_counts"]["RandomTA"], 0)
                runs.append((snapshots, selection_log, {r: a._updates for r, a in trained.items()}))
            self.assertEqual(runs[0], runs[1])

    def test_cli_paths_and_source_metadata_are_independent(self):
        registry = dict(MODEL_NAMES)
        for mode in ("train", "eval", "precompute"):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
                parse_scenario(mode, ["--dry-run", "--profile-mode", "off"])
            self.assertEqual(stop.exception.code, 0)
            row = json.loads(out.getvalue())
            self.assertEqual(row["algorithm"], config.ALGORITHM)
            self.assertEqual(row["model_version"], config.MODEL_VERSION)
            self.assertEqual(row["ta_policy"], config.ta_policy_metadata())
            self.assertEqual(row["checkpoint_selection"], selection.selection_metadata())
        self.assertEqual(registry, MODEL_NAMES)
        baseline, nta = BaseScenario(), ScoreScenario()
        self.assertNotEqual(baseline.checkpoint_dir(), nta.checkpoint_dir())
        self.assertNotEqual(baseline.evaluation_dir(), nta.evaluation_dir())
        self.assertEqual(baseline.normalization_kwargs(), nta.normalization_kwargs())
        self.assertEqual(baseline.cache_path, nta.cache_path)
        self.assertLess(len(str(nta.evaluation_dir()/"measurements/episodes/seed700.json")), 260)
        settings = config.ta_policy_metadata(); settings["learned_roles"].append("TA")
        self.assertEqual(config.ta_policy_metadata()["learned_roles"], ["HA", "VA"])


if __name__ == "__main__":
    unittest.main()
