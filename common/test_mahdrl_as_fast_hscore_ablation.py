"""Real retained-role training, strict selection, barriers and flat export names."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import importlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_hscore import agent, runtime as base
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as FullScene
from algorithms.mahdrl_as_fast_hscore_tl.scenario import ScoreScenario as TLScene
from algorithms.mahdrl_as_fast_hscore.selection import validation_selection
from algorithms.mahdrl_as_fast_hscore_ablation import runtime
from algorithms.mahdrl_as_fast_hscore_ablation.config import SUFFIXES
from algorithms.mahdrl_as_fast_hscore_ablation.preview import prepare_random_decision, uniform_action
from algorithms.mahdrl_as_fast_hscore_ablation.updates import UpdateClock
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.policy import ordered_ready_tasks


def scene_class(suffix):
    return importlib.import_module("algorithms.mahdrl_as_fast_hscore_"+suffix+".scenario").ScoreScenario


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=scene.device or "cpu")
    constructors = dict(TA=lambda: agent.TAAgent(15, 243, **options),
        HA=lambda: agent.HAAgent(23, scene.mechanism.host_features, entity_dims=(8, 8), **options),
        VA=lambda: agent.FixedVAAgent(**options))
    return {role: constructors[role]().bind(scene, role) for role in scene.learned_roles}


class HScoreAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="hsa_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,2,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, suffix, **kwargs):
        return scene_class(suffix)(**dict(dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName/suffix)), **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene, seed=0):
        with patch.object(base, "build_env_kwargs", self.small_kwargs(base.build_env_kwargs)):
            env = runtime.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_strict_selector_full_tl_variants_and_old_identity_rejected(self):
        for successes, feasible in ((46, False), (47, True)):
            result = validation_selection([dict(wf_count=50, wf_success_count=successes,
                wf_success_rate=successes/50, total_energy=1.)])
            self.assertEqual(result["feasible"], feasible)
        for cls in [FullScene, TLScene]+[scene_class(s) for s in SUFFIXES]:
            scene = cls()
            meta = scene.normalization_metadata()
            self.assertEqual(meta["checkpoint_selection"]["workflow_success_comparison"], "strictly_greater")
            self.assertEqual(scene.selection_protocol()["selection"], "select_srgt092_minenergy_v1")
            self.assertIn("_srgt092_", scene.evaluation_csv().name)
            self.assertNotIn("select_sr092_", json.dumps(scene.describe()))
        scene = self.scene("nva")
        learner = tiny_agents(scene)["HA"]
        old = deepcopy(learner.normalization_metadata)
        old["model_version"] = "mahdrl_hscore_fixedva_v1"
        with self.assertRaisesRegex(ValueError, "model_version"):
            learner._check_metadata(old)

    def test_all_public_parsers(self):
        for suffix in SUFFIXES:
            module = importlib.import_module("algorithms.mahdrl_as_fast_hscore_"+suffix+".scenario")
            for mode in ("train", "eval", "precompute"):
                capture = io.StringIO()
                with redirect_stdout(capture), self.assertRaises(SystemExit) as caught:
                    module.parse_scenario(mode, ["--dry-run"])
                self.assertEqual(caught.exception.code, 0)
                result = json.loads(capture.getvalue())
                self.assertEqual(result["algorithm"], "mahdrl_as_fast_hscore_"+suffix)
                self.assertEqual(result["checkpoint_selection"]["workflow_success_comparison"], "strictly_greater")

    def test_uniform_policy_legal_and_rng_isolation(self):
        before = random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone()
        rng = np.random.default_rng(117)
        mask = np.array([0, 1, 0, 1, 1, 0.])
        counts = np.bincount([uniform_action(mask, rng) for _ in range(15000)], minlength=6)
        np.testing.assert_array_equal(counts[[0, 2, 5]], 0)
        self.assertTrue(np.all(abs(counts[[1, 3, 4]]-5000) < 250))
        self.assertEqual(before[0], random.getstate())
        self.assertEqual(before[1], repr(np.random.get_state()))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        with self.assertRaises(ValueError):
            uniform_action(np.zeros(9), rng)

    def test_random_preview_is_pure_and_matches_execution_policy(self):
        env = self.env(self.scene("nva"))
        host, _ = env.get_host_state_for_next_assignment()
        before = repr(env.policy_rng.bit_generator.state), deepcopy(env._records), env.current_time
        enhanced, states, extra = prepare_random_decision(env, host)
        self.assertEqual(before, (repr(env.policy_rng.bit_generator.state), env._records, env.current_time))
        self.assertEqual(enhanced["obs"]["candidates"].shape, (3, 14))
        np.testing.assert_array_equal(extra[:, 3], 0)
        for hid, state in states.items():
            legal = state["mask"] > .5
            np.testing.assert_array_equal(state["probabilities"][~legal], 0)
            np.testing.assert_allclose(state["probabilities"][legal], 1/legal.sum())
            self.assertAlmostEqual(extra[env.host_ids.index(hid), 1],
                state["rows"][legal, 1].mean()/(100*env._obs_ref("T_ref", 300.)), places=6)
            self.assertEqual(state["obs"].shape, (93,))
            self.assertFalse(state["obs"].flags.writeable)
        hid, state = next(iter(states.items()))
        env.host_select(env.host_ids.index(hid))
        actual, _ = env.get_vm_state_for_current_task()
        np.testing.assert_array_equal(actual["obs"], state["obs"])
        np.testing.assert_array_equal(actual["mask"], state["mask"])

    def test_exact_fixed_sort_keys_and_no_TA(self):
        from types import SimpleNamespace
        env = SimpleNamespace(ready_task_ids=[3, 1, 2, 0, 4], task_state=["Ready"]*4+["Done"],
            task_ready_time=[1., 1., 0., 2., -1.], task_baseline_finish=[5., 2., 5., 1., 0.],
            task_up_rank=[3., 3., 2., 1., 9.])
        self.assertEqual(ordered_ready_tasks(env, "ertf"), [2, 0, 1, 3])
        self.assertEqual(ordered_ready_tasks(env, "edf"), [3, 1, 2, 0])
        self.assertEqual(ordered_ready_tasks(env, "cp"), [0, 1, 2, 3])
        for rule in ("ertf", "edf", "cp"):
            scene = self.scene(rule)
            with patch.object(runtime, "TAAgent", side_effect=AssertionError("ghost TA")), \
                    patch.object(runtime, "HAAgent", side_effect=lambda *a, **kw: tiny_agents(scene)["HA"]), \
                    patch.object(runtime, "FixedVAAgent", side_effect=lambda **kw: tiny_agents(scene)["VA"]):
                self.assertEqual(set(runtime.build_agents(scene)), {"HA", "VA"})
            with self.assertRaises(RuntimeError):
                self.env(scene).apply_manager_delta(np.zeros(5))

    def test_all_variants_real_rollout_rewards_successors_and_barriers(self):
        for suffix in SUFFIXES:
            with self.subTest(suffix=suffix):
                scene, rows, phases = self.scene(suffix), [], []
                learners = tiny_agents(scene)
                env = self.env(scene)
                events = []
                for role, learner in learners.items():
                    original = learner.update
                    def update(fn=original, role=role):
                        events.append((role, len(rows), len(phases)))
                        return fn()
                    learner.update = update
                with redirect_stdout(io.StringIO()):
                    result = runtime.run_episode(env, learners, training=True, warmup=0,
                        assignment_log=rows.append, phase_log=phases.append)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(result["task_count"], len(rows))
                for key, value in result["reward_audit"].items():
                    if key.endswith("balance"):
                        self.assertAlmostEqual(value, 0., places=6, msg=suffix+":"+key)
                for role, learner in learners.items():
                    self.assertGreater(learner._updates, 0)
                    self.assertTrue(learner.buffer[-1][-1])
                    for obs, mask, action, reward, nxt, nm, done in learner.buffer:
                        self.assertEqual(mask[action], 1)
                        self.assertFalse(mask.flags.writeable)
                        if done:
                            self.assertFalse(nm.any())
                        else:
                            self.assertTrue(nm.any())
                        if role == "VA":
                            self.assertEqual((obs.shape, mask.shape), ((93,), (9,)))
                if "HA" in learners:
                    self.assertEqual(learners["HA"]._updates, learners["HA"].eligible_steps//4)
                self.assertEqual(result["update_clock"]["update_requests"], result["update_clock"]["optimizer_steps"])
                if suffix == "sync_phase":
                    ends = set()
                    for phase in phases:
                        ends.add(sum(row["phase"] <= phase["phase"] for row in rows))
                    self.assertTrue(all(count in ends for role, count, p in events))
                if suffix == "sync_task":
                    self.assertEqual(result["decision_counts"]["TA"], len(rows))
                    self.assertEqual(result["update_clock"]["transitions"]["TA"], len(rows))
                if scene.variant.ordering:
                    self.assertEqual(result["decision_counts"]["TA"], 0)
                self.assertTrue(all(not row["host_apf_enabled"] and not row["vm_apf_enabled"] for row in env.apf_decisions))
                self.assertEqual((env.manager_alpha_delay, env.alpha_delay_host, env.alpha_delay_vm), (scene.variant.alpha,)*3)

    def test_all_variants_train_reload_frozen_eval_and_role_isolation(self):
        spec, cache = self.root/"six_hosts.json", self.root/"six_hosts.cache.json"
        spec.write_text('{"vms_per_host": [9,8,7,6,5,4]}', encoding="utf-8")
        preparation = self.scene("nta", mode="precompute", resource_config=str(spec),
            cache_output=str(cache), seed_spec="700")
        with redirect_stdout(io.StringIO()):
            precompute(preparation)
        for suffix in SUFFIXES:
            with self.subTest(suffix=suffix), \
                    patch.object(base, "build_env_kwargs", self.small_kwargs(base.build_env_kwargs)), \
                    patch.object(runtime, "build_agents", side_effect=tiny_agents), redirect_stdout(io.StringIO()):
                scene = self.scene(suffix)
                runtime.train(scene)
                stems = {base.FILES[r] for r in scene.learned_roles}
                self.assertEqual({p.stem for p in scene.checkpoint_dir().glob("best_*.pth")}, {"best_"+s for s in stems})
                self.assertEqual({p.stem for p in scene.checkpoint_dir().glob("*_final.pth")}, {s+"_final" for s in stems})
                target = replace(scene, mode="eval", train_resource="S", train_deadline="Tight", seed_spec="700")
                loaded = runtime.load_agents(target)
                initial = parameter_hash(loaded)
                counters = {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps) for r, a in loaded.items()}
                runtime.run_episode(self.env(target, 700), loaded)
                self.assertEqual(initial, parameter_hash(loaded))
                self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps) for r, a in loaded.items()})
                runtime.evaluate(target)
                self.assertTrue(target.evaluation_csv().is_file())
                larger = replace(target, resource_config=str(spec), cache_file=str(cache))
                transferred = runtime.load_agents(larger)
                self.assertEqual(initial, parameter_hash(transferred))
                result = runtime.run_episode(self.env(larger, 700), transferred)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(initial, parameter_hash(transferred))
                self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
                with self.assertRaises(FileExistsError):
                    runtime.evaluate(target)
                other = self.scene("nva" if suffix != "nva" else "nta")
                role = next(iter(set(other.learned_roles) & set(scene.learned_roles)))
                with self.assertRaises(ValueError):
                    tiny_agents(other)[role].load(str(scene.checkpoint_dir()/f"best_{base.FILES[role]}.pth"))

    def test_removed_agents_are_never_constructed(self):
        for suffix, removed in (("nta", "TA"), ("nha", "HA"), ("nva", "VA")):
            scene = self.scene(suffix)
            with patch.object(runtime, "TAAgent", side_effect=lambda *a, **kw: tiny_agents(scene)["TA"]) as ta, \
                    patch.object(runtime, "HAAgent", side_effect=lambda *a, **kw: tiny_agents(scene)["HA"]) as ha, \
                    patch.object(runtime, "FixedVAAgent", side_effect=lambda **kw: tiny_agents(scene)["VA"]) as va:
                absent = dict(TA=ta, HA=ha, VA=va)[removed]
                absent.side_effect = AssertionError("removed network was constructed")
                self.assertEqual(set(runtime.build_agents(scene)), set(scene.learned_roles))
                absent.assert_not_called()

    def test_paths_transfer_separation_and_workload(self):
        paths, names = set(), set()
        for suffix in SUFFIXES:
            for resource in "SML":
                for deadline in ("Tight", "Medium", "Loose"):
                    scene = self.scene(suffix, resource=resource, deadline=deadline)
                    self.assertEqual(len(scene.model_name()), 17)
                    self.assertLess(len(scene.checkpoint_dir().relative_to(scene.output_root).as_posix()), 64)
                    paths.add(str(scene.checkpoint_dir()))
                    for task in "SML":
                        target = replace(scene, mode="eval", task=task, train_resource=resource, train_deadline=deadline)
                        self.assertEqual(scene.checkpoint_dir(), target.checkpoint_dir())
                        self.assertEqual(target.evaluation_csv().parent, target.evaluation_root)
                        self.assertIn(target.target_tag+"_from_"+target.source_tag, target.evaluation_csv().name)
                        names.add(target.evaluation_csv().name)
            stress = self.scene(suffix, workload_profile="stress_a1_l006_ref003", legacy_split=False)
            self.assertEqual(base.build_env_kwargs(stress, 700)["arrival_lambda"], .06)
        self.assertEqual(len(paths), 90)
        self.assertEqual(len(names), 270)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_retained_learners(self):
        for suffix in SUFFIXES:
            scene = self.scene(suffix, device="cuda")
            learners = tiny_agents(scene)
            with redirect_stdout(io.StringIO()):
                runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
            self.assertTrue(all(a._updates > 0 for a in learners.values()))


if __name__ == "__main__":
    unittest.main()
