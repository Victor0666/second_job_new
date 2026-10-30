"""Fixed-slot masking, VAware parity, real learning and cross-host checkpoints."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_hscore import agent, runtime
from algorithms.mahdrl_as_fast_hscore.config import MODEL_VERSION
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_as_fast_hscore.preview import prepare_decision
from algorithms.mahdrl_as_fast_hscore.updates import UpdateClock
from algorithms.mahdrl_as_fast_vaaware import runtime as original
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario as OriginalScenario


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=scene.device or "cpu")
    learners = dict(TA=agent.TAAgent(15, 243, **options),
        HA=agent.HAAgent(23, scene.mechanism.host_features, entity_dims=(8, 8), **options),
        VA=agent.FixedVAAgent(**options))
    return {role: a.bind(scene, role) for role, a in learners.items()}


class HScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="hsfv_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,2,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, **kwargs):
        values = dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName))
        return ScoreScenario(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original_kwargs):
        return lambda scene, seed: dict(original_kwargs(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene=None, seed=0):
        with patch.object(runtime, "build_env_kwargs", self.small_kwargs(runtime.build_env_kwargs)):
            env = runtime.build_env(scene or self.scene(), seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def custom_scene(self, counts, source=None):
        spec = self.root/("topology_" + "_".join(map(str, counts)) + ".json")
        cache = spec.with_suffix(".cache.json")
        spec.write_text(json.dumps(dict(vms_per_host=counts)), encoding="utf-8")
        preparation = self.scene(mode="precompute", resource_config=str(spec), cache_output=str(cache), seed_spec="700")
        with redirect_stdout(io.StringIO()):
            if not cache.exists():
                precompute(preparation)
        return replace(source or self.scene(), mode="eval", resource_config=str(spec), cache_file=str(cache),
                       train_resource="S", train_deadline="Tight", seed_spec="700")

    def test_preview_is_pure_and_matches_actual_fixed_local_state(self):
        scene, env = self.scene(), self.env()
        learners = tiny_agents(scene)
        host, ok = env.get_host_state_for_next_assignment()
        self.assertTrue(ok)
        before = (env.current_time, deepcopy(env._records), deepcopy(env.assignment_ledger.segments),
                  env._cur_tid, env._cur_host_id, repr(env.rng.get_state()))
        rng = random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone()
        counters = learners["VA"]._updates, learners["VA"]._eps_steps, learners["VA"]._action_calls
        enhanced, states, extra = prepare_decision(env, learners["VA"], host, training=True)
        self.assertEqual(before, (env.current_time, env._records, env.assignment_ledger.segments,
            env._cur_tid, env._cur_host_id, repr(env.rng.get_state())))
        self.assertEqual(random.getstate(), rng[0])
        self.assertEqual(repr(np.random.get_state()), rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))
        self.assertEqual(counters, (learners["VA"]._updates, learners["VA"]._eps_steps, learners["VA"]._action_calls))
        self.assertEqual(enhanced["obs"]["candidates"].shape, (3, 14))
        for hid, state in states.items():
            self.assertEqual(state["obs"].shape, (93,))
            self.assertEqual(state["mask"].shape, (9,))
            self.assertFalse(state["obs"].flags.writeable)
            k = len(env.host_to_vm_indices[hid])
            np.testing.assert_array_equal(state["obs"][30:].reshape(9, 7)[k:], 0)
            np.testing.assert_array_equal(state["mask"][k:], 0)
            self.assertAlmostEqual(state["probabilities"].sum(), 1)
            self.assertTrue((state["probabilities"][state["mask"] == 0] == 0).all())
            self.assertAlmostEqual(extra[env.host_ids.index(hid), 1],
                state["probabilities"] @ state["rows"][:, 1]/(100*env._obs_ref("T_ref", 300.)), places=6)
        for hid in states:
            env.host_select(env.host_ids.index(hid))
            real, ok = env.get_vm_state_for_current_task()
            self.assertTrue(ok)
            np.testing.assert_array_equal(real["obs"], states[hid]["obs"])
            np.testing.assert_array_equal(real["mask"], states[hid]["mask"])
            break

    def test_busy_padding_masks_and_preview_version_guards(self):
        env, learners = self.env(), tiny_agents(self.scene())
        env.vm_available_at[env.host_to_vm_indices[env.host_ids[0]]] = env.current_time+100
        remaining = env.host_to_vm_indices[env.host_ids[1]]
        env.vm_available_at[remaining[0]] = env.current_time+100
        host, _ = env.get_host_state_for_next_assignment()
        _, states, _ = prepare_decision(env, learners["VA"], host, training=False)
        self.assertNotIn(env.host_ids[0], states)
        state = states[env.host_ids[1]]
        self.assertEqual(state["mask"][0], 0)
        self.assertEqual(state["mask"][8], 0)
        for _ in range(50):
            action = learners["VA"].select_action(state["obs"], state["mask"], performance_times=state["performance_times"])
            self.assertEqual(state["mask"][action], 1)
        learners["VA"]._updates += 1
        with self.assertRaisesRegex(RuntimeError, "VA changed"):
            learners["VA"].select_preview_action(state, training=False)
        learners["VA"]._updates -= 1
        state["epsilon"] = .5
        with self.assertRaisesRegex(RuntimeError, "epsilon changed"):
            learners["VA"].select_preview_action(state, training=False)

    def test_masked_dueling_and_saved_next_mask_double_dqn_terminal_target(self):
        va = agent.FixedVAAgent(hidden_dims=(8,), batch_size=2, buffer_size=8)
        with torch.no_grad():
            for net in (va.online, va.target):
                for p in net.parameters():
                    p.zero_()
            va.online.adv[-1].bias.copy_(torch.tensor([1., 3.] + [1000.]*7))
            va.target.adv[-1].bias.copy_(torch.tensor([5., 9.] + [2000.]*7))
        obs, current, successor = np.zeros(93, np.float32), np.zeros(9, np.float32), np.zeros(9, np.float32)
        current[0] = 1
        successor[:2] = 1
        q = va.online(torch.tensor(obs[None]), torch.tensor(successor[None]))
        np.testing.assert_allclose(q.detach().numpy()[0, :2], [-1., 1.])
        self.assertEqual(va.select_action(obs, successor, deterministic=True), 1)
        va.remember(obs, current, 0, -.1, obs, successor, False)
        va.remember(obs, current, 0, -.1, obs, np.zeros(9), True)
        before = deepcopy(va.online.state_dict())
        loss = va.update()
        self.assertTrue(np.isfinite(loss))
        va.f11.snapshot()  # FastDiagnostics transfers deferred totals at snapshot time.
        self.assertAlmostEqual(va.f11.future_abs_sum, .95*2, places=6)
        self.assertAlmostEqual(va.f11.target_sum, .95*2-.2, places=6)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in va.online.state_dict().items()))
        self.assertFalse(va.buffer[0][0].flags.writeable)
        with self.assertRaises(ValueError):
            va.remember(obs, current, 1, 0., obs, successor, False)
        with self.assertRaises(ValueError):
            va.remember(obs, current, 0, 0., obs, np.zeros(9), False)
        with self.assertRaises(ValueError):
            va.select_action(obs[:-1], current, deterministic=True)

    def test_capacity_validation_and_no_output_creation(self):
        spec = self.root/"too_many.json"
        spec.write_text('{"vms_per_host": [9, 10]}', encoding="utf-8")
        for mode in ("eval", "precompute"):
            with self.assertRaisesRegex(ValueError, "capacity"):
                self.scene(mode=mode, resource_config=str(spec), train_resource="S")
        self.assertFalse(Path(self.scene().output_root).exists())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            parse_scenario("train", ["--dry-run"])
        self.assertEqual(caught.exception.code, 0)

    def test_fixed_actions_preserve_original_vaaware_physics_and_rewards(self):
        scene, env = self.scene(), self.env()
        with patch.object(original, "build_env_kwargs", self.small_kwargs(original.build_env_kwargs)):
            old = original.build_env(scene, 0)
        with redirect_stdout(io.StringIO()):
            old.reset()
            while not old.done_flag:
                while True:
                    host, oka = old.get_host_state_for_next_assignment()
                    raw, okb = env.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    ah = int(np.flatnonzero(host["mask"])[0])
                    old.host_select(ah)
                    env.host_select(ah)
                    vm, _ = old.get_vm_state_for_current_task()
                    fixed, _ = env.get_vm_state_for_current_task()
                    av = int(np.flatnonzero(vm["mask"])[0])
                    self.assertEqual(fixed["mask"][av], 1)
                    self.assertEqual(old.vm_assign(av), env.vm_assign(av))
                    self.assertEqual(old._records, env._records)
                self.assertEqual(old.finish_phase_and_advance(), env.finish_phase_and_advance())
                if not old.done_flag:
                    old.apply_manager_delta(np.zeros(5))
                    env.apply_manager_delta(np.zeros(5))
        self.assertEqual(old.total_energy, env.total_energy)

    def test_real_rollout_atomicity_reward_balance_and_frozen_replay(self):
        scene = self.scene()
        learners, rows = tiny_agents(scene), []
        runtime.set_seed(7)
        env = self.env(scene)
        with redirect_stdout(io.StringIO()):
            result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append)
        self.assertGreater(learners["VA"]._updates, 10)
        self.assertEqual(learners["HA"]._updates, learners["HA"].eligible_steps//4)
        self.assertEqual(result["task_count"], len(rows))
        self.assertEqual(result["wf_completed_count"], 2)
        for key in ("physical_energy_balance", "lower_energy_balance", "HA_reward_formula_balance", "VA_reward_formula_balance"):
            self.assertAlmostEqual(result["reward_audit"][key], 0., places=6)
        for s, m, a, r, n, nm, done in learners["VA"].buffer:
            self.assertEqual((s.shape, n.shape, m.shape, nm.shape), ((93,), (93,), (9,), (9,)))
            self.assertFalse(s.flags.writeable)
            self.assertFalse(m.flags.writeable)
        self.assertTrue(learners["VA"].buffer[-1][-1])
        self.assertTrue(learners["HA"].buffer[-1][-1])
        self.assertTrue(all(not d["host_apf_enabled"] and not d["vm_apf_enabled"] for d in env.apf_decisions))

    def test_real_train_reload_evaluate_and_cross_host_frozen_transfer(self):
        scene = self.scene()
        with patch.object(runtime, "build_env_kwargs", self.small_kwargs(runtime.build_env_kwargs)), \
                patch.object(runtime, "build_agents", side_effect=tiny_agents), redirect_stdout(io.StringIO()):
            runtime.train(scene)
            evaluation = replace(scene, mode="eval", train_resource="S", train_deadline="Tight", seed_spec="700")
            loaded = runtime.load_agents(evaluation)
            fingerprint = parameter_hash(loaded)
            counters = {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps) for r, a in loaded.items()}
            summary = runtime.evaluate(evaluation)
            for deadline in ("Medium", "Loose"):
                other = replace(evaluation, deadline=deadline)
                runtime.evaluate(other)
                self.assertEqual(other.evaluation_csv().parent, evaluation.evaluation_csv().parent)
                self.assertNotEqual(other.evaluation_csv().name, evaluation.evaluation_csv().name)
                self.assertTrue(other.evaluation_csv().is_file())
            self.assertEqual(len(list(evaluation.evaluation_root.glob("*.csv"))), 3)
            source_shapes = {r: {k: tuple(v.shape) for k, v in a.online.state_dict().items()} for r, a in loaded.items()}
            for counts in ([8]*6, [9, 8]*6, [9, 8]*9):
                target = self.custom_scene(counts, scene)
                runtime.validate_inputs(target)
                transferred = runtime.load_agents(target)
                self.assertEqual(parameter_hash(transferred), fingerprint)
                env = runtime.build_env(target, 700)
                env.reset()
                host, _ = env.get_host_state_for_next_assignment()
                enhanced, states, _ = prepare_decision(env, transferred["VA"], host, training=False)
                self.assertEqual(enhanced["obs"]["candidates"].shape, (len(counts), 14))
                self.assertTrue(all(s["obs"].shape == (93,) and s["mask"].shape == (9,) for s in states.values()))
                # Obtain a fresh environment: querying the first task above consumes it.
                env = runtime.build_env(target, 700)
                env.reset()
                result = runtime.run_episode(env, transferred)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(parameter_hash(transferred), fingerprint)
                self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps) for r, a in transferred.items()})
                self.assertEqual(source_shapes, {r: {k: tuple(v.shape) for k, v in a.online.state_dict().items()} for r, a in transferred.items()})
        self.assertTrue(evaluation.evaluation_csv().is_file())
        with evaluation.evaluation_csv().open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["model_version"], MODEL_VERSION)
        self.assertEqual(row["vm_capacity"], "9")
        self.assertEqual(row["vm_selection_scope"], "host_local_fixed_slots")
        self.assertTrue(np.isfinite(summary["total_energy"]["mean"]))
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        wrong_role = tiny_agents(evaluation)
        with self.assertRaises(ValueError):
            wrong_role["VA"].load(str(scene.checkpoint_dir()/"best_host.pth"))
        from common.test_mahdrl_as_fast_vaaware import tiny_agents as original_tiny
        old_scene = OriginalScenario(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/"old_identity"))
        old_learners = original_tiny(old_scene)
        with self.assertRaises(ValueError):
            old_learners["TA"].load(str(scene.checkpoint_dir()/"best_manager.pth"))
        old_path = self.root/"old_manager.pth"
        evidence = loaded["TA"].loaded_selection
        original.save_agents(old_learners, self.root, "best", episode=evidence["episode"],
            candidate=evidence["validation_selection"], best=evidence["best_validation_selection"],
            rows=evidence["validation_rows"], best_rows=evidence["best_validation_rows"])
        shutil.move(str(self.root/"best_manager.pth"), old_path)
        with self.assertRaises(ValueError):
            wrong_role["TA"].load(str(old_path))
        with self.assertRaises(ValueError):
            tiny_agents(replace(evaluation, ha_update_period=1))["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"))
        renamed = self.root/"renamed_final"
        renamed.mkdir()
        for stem in ("manager", "host", "vm"):
            shutil.copy2(scene.checkpoint_dir()/f"{stem}_final.pth", renamed/f"best_{stem}.pth")
        with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "best checkpoints"):
            runtime.load_agents(replace(evaluation, checkpoint_override=str(renamed)))
        payload_path = scene.checkpoint_dir()/"best_vm.pth"
        payload = torch.load(payload_path, weights_only=True)
        payload["normalization"]["checkpoint_group"] = "wrong-group"
        torch.save(payload, payload_path)
        with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "different saved groups"):
            runtime.load_agents(evaluation)
        with self.assertRaises(FileExistsError):
            runtime.evaluate(evaluation)

    def test_paths_and_existing_vaaware_remain_separate(self):
        old = OriginalScenario()
        new = ScoreScenario()
        self.assertNotEqual(new.checkpoint_dir(), old.checkpoint_dir())
        self.assertNotEqual(new.log_dir(), old.log_dir())
        self.assertNotEqual(new.evaluation_root, old.evaluation_root)
        paths, csv_names = set(), set()
        for enabled in (False, True):
            for period in (1, 4):
                for resource in "SML":
                    for deadline in ("Tight", "Medium", "Loose"):
                        scene = self.scene(resource=resource, deadline=deadline,
                            va_preview=enabled, ha_update_period=period)
                        self.assertEqual(scene.checkpoint_dir().parent.name, "mahdrl_hscore")
                        self.assertEqual(len(scene.model_name()), 17)
                        self.assertLessEqual(len(scene.checkpoint_dir().relative_to(scene.output_root).as_posix()), 64)
                        paths.add(scene.checkpoint_dir())
                        for task in "SML":
                            target = replace(scene, mode="eval", task=task,
                                train_resource=resource, train_deadline=deadline)
                            self.assertEqual(scene.checkpoint_dir(), target.checkpoint_dir())
                            csv_path = target.evaluation_csv()
                            self.assertEqual(csv_path.parent, Path(scene.output_root)/"outputs/run/mahdrl_hscore")
                            self.assertIn(f"_{target.target_tag}_from_{scene.source_tag}_", csv_path.name)
                            self.assertIn(f"_p{int(enabled)}h{period}_", csv_path.name)
                            self.assertIn("_m1_s700-729_", csv_path.name)
                            self.assertEqual(target.evaluation_dir().parent, csv_path.parent/"_details")
                            csv_names.add(csv_path.name)
        self.assertEqual(len(paths), 36)
        self.assertEqual(len(csv_names), 108)
        first = self.scene(mode="eval", seed_spec="700,702,703")
        self.assertNotEqual(first.evaluation_csv().name, replace(first, seed_spec="700,701,703").evaluation_csv().name)
        self.assertNotEqual(first.evaluation_csv().name, replace(first, model_seed=2).evaluation_csv().name)
        stress = parse_scenario("eval", ["--workload-profile", "stress_a1_l006_ref003"])
        self.assertEqual(runtime.build_env_kwargs(stress, 700)["arrival_lambda"], .06)

    def test_preview_controls_and_HA_update_clock(self):
        for enabled in (True, False):
            for period in (1, 4):
                scene = self.scene(va_preview=enabled, ha_update_period=period)
                learners, env = tiny_agents(scene), self.env(scene)
                host, _ = env.get_host_state_for_next_assignment()
                enhanced, _, _ = prepare_decision(env, learners["VA"], host, training=True)
                self.assertEqual(enhanced["obs"]["candidates"].shape, (3, 14 if enabled else 10))
                clock = UpdateClock(learners, training=True, warmup=0)
                for _ in range(9):
                    clock.remember("HA", enhanced["obs"], host["mask"], int(np.flatnonzero(host["mask"])[0]),
                        -.1, enhanced["obs"], host["mask"], False)
                    clock.barrier()
                self.assertEqual(learners["HA"].eligible_steps, 8)
                self.assertEqual(learners["HA"]._updates, 8//period)
                clock.summary()

    def test_production_width_and_real_updates(self):
        scene, env = self.scene(), self.env()
        options = runtime.learning_kwargs
        with patch.object(runtime, "learning_kwargs", side_effect=lambda role, device: dict(options(role, device), batch_size=2, buffer_size=8)):
            learners = runtime.build_agents(scene)
        self.assertEqual(tuple(learners["VA"].online.feature[0].weight.shape), (1024, 93))
        self.assertEqual(tuple(learners["VA"].online.adv[-1].weight.shape), (9, 256))
        host, _ = env.get_host_state_for_next_assignment()
        enhanced, states, _ = prepare_decision(env, learners["VA"], host, training=False)
        state, mask = env.get_manager_state(), env.get_manager_action_mask()
        vm = next(iter(states.values()))
        observations = dict(TA=(state, mask), HA=(enhanced["obs"], enhanced["mask"]), VA=(vm["obs"], vm["mask"]))
        clock = UpdateClock(learners, training=True, warmup=0)
        for _ in range(5):
            for role, (obs, legal) in observations.items():
                clock.remember(role, obs, legal, int(np.flatnonzero(legal)[0]), -.1, obs, legal, False)
            clock.barrier()
        self.assertEqual(learners["HA"]._updates, 1)
        self.assertEqual(learners["VA"]._updates, 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_CUDA_preview_and_real_updates(self):
        scene = self.scene(device="cuda")
        learners = tiny_agents(scene)
        with redirect_stdout(io.StringIO()):
            result = runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
        self.assertGreater(learners["VA"]._updates, 0)
        self.assertGreater(learners["HA"]._updates, 0)
        self.assertEqual(result["wf_completed_count"], 2)


if __name__ == "__main__":
    unittest.main()
