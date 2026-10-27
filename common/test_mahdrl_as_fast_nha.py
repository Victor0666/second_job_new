"""All-idle-host routing, retained VM APF, OE settlement and model lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.experiment_protocol import protocol_binding
from common.f11 import PendingDecision
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, MODEL_NAMES
from common.test_f18 import fixture_config
from common.test_mahdrl_as_fast import tiny_agents as base_tiny_agents, validation_row
from common.test_mahdrl_as_fast_nta import tiny_agents as nta_tiny_agents
from common.test_mahdrl_oe_ge_apf_score import parameter_hash, observation
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as base_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nta.scenario import ScoreScenario as NTAScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nta import runtime as nta_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast_nha import runtime, agent, config, selection
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.policy import RandomHostPolicy, host_seed
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.reward import finish_lower_phase
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.scenario import ScoreScenario, parse_scenario
from environments.mahdrl_oe_ge_apf_score_fast import apf_candidates as base_apf
from environments.mahdrl_oe_ge_apf_score_fast_nha import vm_apf_candidates


def tiny_agents(scene):
    return dict(TA=agent.TAAgent(15, 243, hidden_dims=(16, 8), batch_size=2, buffer_size=8,
                              device="cpu").bind(scene, "TA"),
                VA=agent.SetD3QNAgent(30, 7, hidden_dims=(16, 8), entity_dims=(8, 8),
                              batch_size=2, buffer_size=8, device="cpu").bind(scene, "VA"))


class NHATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="as_nha_")
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

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, module=runtime, seed=0):
        scene = self.scene(BaseScenario if module is base_runtime else ScoreScenario)
        with patch.object(module, "build_env_kwargs", side_effect=self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_uniform_policy_variable_host_counts_reproducibility_and_rng_isolation(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        a, b = RandomHostPolicy(700), RandomHostPolicy(700)
        mask = np.array([0, 1, 0, 1, 1, 0])
        actions = [a.select_action(mask) for _ in range(6000)]
        self.assertEqual(actions, [b.select_action(mask) for _ in range(6000)])
        self.assertEqual(set(actions), {1, 3, 4})
        self.assertTrue(all(1800 < actions.count(i) < 2200 for i in (1, 3, 4)))
        self.assertEqual(a.seed, (700*1000003+20260325+123456*97) & 0xFFFFFFFF)
        self.assertNotEqual(a.seed, RandomHostPolicy(701).seed)
        self.assertEqual(a.select_action(np.array([0, 0, 1])), 2)
        self.assertEqual(a.summary()["multiple_choice_decisions"], 6000)
        self.assertEqual(before[0], random.getstate())
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1][1], after[1]); self.assertEqual(before[1][2:], after[2:])
        torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)
        for bad in (np.zeros(3), np.zeros(0), np.ones((2, 1)), np.array([np.nan]), np.array([2])):
            with self.assertRaises(ValueError): a.select_action(bad)

    def test_VM_APF_matches_baseline_for_all_hosts_but_host_APF_is_absent(self):
        rng = np.random.RandomState(19)
        ownership = {11: [0, 1, 2], 25: [3, 4], 90: [5, 6, 7], 121: [8]}
        fixtures = [{0: (1., 1., 0.), 3: (10., 10., 0.), 5: (20., 20., 0.)},
                    {0: (1., 5., 0.), 1: (2., 1., 0.), 2: (2., 1., 0.), 5: (20., 20., 0.)}]
        for _ in range(100):
            fixtures.append({j: (float(rng.randint(1, 50)), float(rng.randint(1, 20)), 0.)
                             for j in range(9) if rng.rand() < .8})
        for rows in fixtures:
            for deadline in (0., 3., 100.):
                old = base_apf(rows, ownership, 4, 0., deadline)
                new = vm_apf_candidates(rows, ownership, 4, 0., deadline)
                self.assertEqual(new["hosts"], {h for h, ids in ownership.items() if set(ids).intersection(rows)})
                self.assertEqual(new["vms"], old["vms"])
                self.assertEqual(new["energy_vms"], old["energy_vms"])
                for key in ("g", "u", "r_V", "raw_count"): self.assertEqual(new[key], old[key])
                self.assertIsNone(new["r_H"])
                self.assertEqual(new["energy_hosts"], set())
                self.assertTrue(all(new["vms"][h] for h in new["hosts"]))
        old = base_apf(fixtures[0], ownership, 4, 0., 100.)
        new = vm_apf_candidates(fixtures[0], ownership, 4, 0., 100.)
        self.assertEqual(old["hosts"], {11})
        self.assertEqual(new["hosts"], {11, 25, 90})
        self.assertEqual(vm_apf_candidates(fixtures[1], ownership, 4, 0., 100.)["vms"][11], {0, 1, 2})
        with self.assertRaises(ValueError): vm_apf_candidates({0: (np.nan, 1., 0.)}, ownership, 4, 0., 100.)

    def test_formerly_excluded_slow_host_is_selectable_with_nonempty_VM_APF(self):
        environments = [self.env(base_runtime), self.env()]
        masks = []
        for env in environments:
            owners = {j: hi for hi, h in enumerate(env.host_ids) for j in env.host_to_vm_indices[h]}
            with patch.object(env, "_task_duration", side_effect=lambda tid, j: [1., 20., 100.][owners[j]]*(j%3+1)), \
                 patch.object(env.assignment_ledger, "preview", side_effect=lambda hid, start, end, pc: end-start):
                host, ok = env.get_host_state_for_next_assignment()
                self.assertTrue(ok)
                self.assertTrue(np.isfinite(host["obs"]["candidates"]).all())
                masks.append(host["mask"].copy())
                if env is environments[1]:
                    env.host_select(2)
                    vm, ok = env.get_vm_state_for_current_task()
                    self.assertTrue(ok); self.assertGreater(vm["mask"].sum(), 0)
                    self.assertLess(vm["mask"].sum(), len(vm["mask"]))
                    env.vm_assign(int(np.flatnonzero(vm["mask"])[0]))
                    row = env.apf_decisions[-1]
                    self.assertFalse(row["host_apf_enabled"])
                    self.assertEqual(row["raw_host_count"], row["retained_host_count"])
                    self.assertIsNone(row["r_H"])
                    self.assertIsNone(row["selected_energy_only_host"])
                    self.assertIsNone(env._fast_key)
        np.testing.assert_array_equal(masks[0], [1., 0., 0.])
        np.testing.assert_array_equal(masks[1], [1., 1., 1.])

    def test_busy_hosts_and_idle_tolerance_cannot_produce_empty_VM_decision(self):
        for delay in (100., 5e-10):
            env = self.env()
            env.vm_available_at[env.host_to_vm_indices[env.host_ids[0]]] = env.current_time+delay
            host, ok = env.get_host_state_for_next_assignment()
            self.assertTrue(ok)
            self.assertEqual(host["mask"][0], 0.)
            with self.assertRaisesRegex(ValueError, "no legal idle VM"): env.host_select(0)
            for hi in np.flatnonzero(host["mask"]):
                env.host_select(int(hi))
                vm, legal = env.get_vm_state_for_current_task()
                self.assertTrue(legal); self.assertGreater(vm["mask"].sum(), 0)

    def test_same_actions_preserve_VM_observations_physics_rewards_and_ledger(self):
        old, new = self.env(base_runtime), self.env()
        assignments = 0
        with redirect_stdout(io.StringIO()):
            while not old.done_flag:
                while True:
                    ha, oka = old.get_host_state_for_next_assignment()
                    hb, okb = new.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka: break
                    self.assertTrue((ha["mask"] <= hb["mask"]).all())
                    ah = int(np.flatnonzero(ha["mask"])[0])
                    old.host_select(ah); new.host_select(ah)
                    va, _ = old.get_vm_state_for_current_task()
                    vb, _ = new.get_vm_state_for_current_task()
                    np.testing.assert_array_equal(va["mask"], vb["mask"])
                    np.testing.assert_array_equal(va["performance_times"], vb["performance_times"])
                    for key in va["obs"]: np.testing.assert_array_equal(va["obs"][key], vb["obs"][key])
                    av = int(np.flatnonzero(va["mask"])[0])
                    self.assertEqual(old.vm_assign(av), new.vm_assign(av))
                    self.assertEqual(old._records, new._records)
                    self.assertEqual(old.assignment_ledger.segments, new.assignment_ledger.segments)
                    assignments += 1
                self.assertEqual(old.finish_phase_and_advance(), new.finish_phase_and_advance())
                np.testing.assert_array_equal(old.get_manager_state(), new.get_manager_state())
                self.assertEqual(old.done_flag, new.done_flag)
                if not old.done_flag:
                    old.apply_manager_delta(np.zeros(5)); new.apply_manager_delta(np.zeros(5))
        self.assertGreater(assignments, 20)
        self.assertEqual(old.total_energy, new.total_energy)
        self.assertEqual(repr(old.rng.get_state()), repr(new.rng.get_state()))

    def test_formal_initial_TA_VA_tensors_and_rng_match_baseline(self):
        base_runtime.set_seed(31)
        old = base_runtime.build_agents(self.scene(BaseScenario))
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        runtime.set_seed(31)
        new = runtime.build_agents(self.scene())
        self.assertEqual(set(new), {"TA", "VA"})
        for role in new:
            self.assertIs(type(old[role])._update, type(new[role])._update)
            self.assertEqual(config.learning_kwargs(role, "cpu"), base_runtime.learning_kwargs(role, "cpu"))
            for network in ("online", "target"):
                for left, right in zip(getattr(old[role], network).parameters(), getattr(new[role], network).parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertEqual(before[1][2:], np.random.get_state()[2:])
        torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_unreturned_energy_settles_only_VA_and_detects_missing_pending(self):
        emitted = []
        trace = PendingDecision(lambda *args: emitted.append(args))
        trace.record(np.ones(1), np.ones(1), 0, -.2)
        env = SimpleNamespace(alpha_delay_vm=.75, energy_reward_scale=.001, done_flag=False,
                              _episode_should_end=lambda: True)
        finish_lower_phase(env, trace, {"lower_assignment_energy_unemitted": 100.})
        self.assertAlmostEqual(trace.pending["reward"], -.225)
        finish_lower_phase(env, trace, {"lower_assignment_energy_unemitted": 0.})
        self.assertAlmostEqual(trace.pending["reward"], -.225)
        env.done_flag = True
        finish_lower_phase(env, trace, {})
        self.assertEqual(len(emitted), 1)
        self.assertIsNone(trace.pending)
        with self.assertRaisesRegex(RuntimeError, "no preceding VA"):
            finish_lower_phase(env, trace, {"lower_assignment_energy_unemitted": 100.})

    def test_evaluation_is_reproducible_and_uses_raw_idle_host_masks(self):
        learners = tiny_agents(self.scene())
        before = parameter_hash(learners)
        results, choices = [], []
        for _ in range(2):
            env, rows = self.env(seed=700), []
            with redirect_stdout(io.StringIO()): result = runtime.run_episode(env, learners, assignment_log=rows.append)
            self.assertEqual(result["random_ha"]["decisions"], result["task_count"])
            self.assertEqual(result["random_ha"]["seed"], host_seed(700))
            self.assertGreater(result["random_ha"]["multiple_choice_decisions"], 0)
            self.assertTrue(all(r["legal_host_count"] >= 1 for r in rows))
            self.assertTrue(all(not r["host_apf_enabled"] for r in env.apf_decisions))
            self.assertEqual(result["wf_count"], 2)
            results.append(result); choices.append(rows)
        self.assertEqual(results[0], results[1]); self.assertEqual(choices[0], choices[1])
        self.assertEqual(parameter_hash(learners), before)
        self.assertTrue(all(a._updates == 0 for a in learners.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_TA_VA_CUDA_updates_reuse_baseline_math(self):
        options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device="cuda")
        learners = dict(TA=agent.TAAgent(15, 243, **options).bind(self.scene(), "TA"),
                        VA=agent.SetD3QNAgent(30, 7, entity_dims=(8, 8), **options).bind(self.scene(), "VA"))
        for role, learner in learners.items():
            before = parameter_hash({role: learner})
            for done in (False, True):
                if role == "TA":
                    state, mask, nxt, nm = np.ones(15), np.ones(243), np.zeros(15), np.zeros(243) if done else np.ones(243)
                else:
                    state, mask, nxt, nm = observation(3, 30, 7), np.ones(3), observation(0 if done else 4, 30, 7), np.ones(0 if done else 4)
                learner.remember(state, mask, 0, -.2, nxt, nm, done)
            self.assertTrue(np.isfinite(learner.update()))
            self.assertEqual(learner._updates, 1)
            self.assertNotEqual(parameter_hash({role: learner}), before)

    def test_checkpoint_identity_policy_roles_groups_and_final_rejection(self):
        scene = self.scene()
        learners = tiny_agents(scene)
        baseline = base_tiny_agents(self.scene(BaseScenario))
        nta = nta_tiny_agents(self.scene(NTAScenario))
        rows = [validation_row(seed=scene.validation_seeds()[0])]
        candidate = selection.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        args = dict(episode=0, candidate=candidate, best=best, rows=rows, best_rows=rows)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            new_dir, old_dir, nta_dir = root/"nha", root/"baseline", root/"nta"
            for folder in (new_dir, old_dir, nta_dir): folder.mkdir()
            runtime.save_agents(learners, new_dir, "best", **args)
            base_runtime.save_agents(baseline, old_dir, "best", **args)
            nta_runtime.save_agents(nta, nta_dir, "best", **args)
            target = replace(scene, mode="eval", checkpoint_override=str(new_dir))
            with patch.object(runtime, "build_agents", side_effect=tiny_agents): loaded = runtime.load_agents(target)
            self.assertEqual(parameter_hash(learners), parameter_hash(loaded))
            self.assertFalse((new_dir/"best_host.pth").exists())
            for role, name in runtime.FILES.items():
                with self.assertRaises(ValueError): learners[role].load(str(old_dir/f"best_{name}.pth"))
                with self.assertRaises(ValueError): baseline[role].load(str(new_dir/f"best_{name}.pth"))
            with self.assertRaises(ValueError): learners["VA"].load(str(nta_dir/"best_vm.pth"))
            with self.assertRaises(ValueError): nta["VA"].load(str(new_dir/"best_vm.pth"))
            with self.assertRaises(ValueError): learners["TA"].load(str(new_dir/"best_vm.pth"))
            with self.assertRaisesRegex(ValueError, "strict"): learners["TA"].load(str(new_dir/"best_manager.pth"), strict=False)
            checkpoint = torch.load(new_dir/"best_manager.pth", weights_only=True)
            for section in (None, "algorithm_variant", "implementation", config.CHECKPOINT_METADATA_KEY):
                tampered = deepcopy(checkpoint)
                meta = tampered["normalization"] if section is None else tampered["normalization"][section]
                meta["ha_policy"]["host_apf"] = "enabled"
                path = root/"tampered.pth"; torch.save(tampered, path)
                with self.subTest(section=section), self.assertRaises(ValueError): learners["TA"].load(str(path))
            manifest = new_dir/"ha_policy.json"; original = manifest.read_text(); manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "policy manifest"): runtime.load_agents(target)
            manifest.write_text(original)
            learners["TA"].checkpoint_group = "foreign_group"
            learners["TA"].save(str(new_dir/"best_manager.pth"))
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                with self.assertRaisesRegex(ValueError, "different saved groups"): runtime.load_agents(target)
            runtime.save_agents(learners, new_dir, "final", episode=1, candidate=candidate, best=best, rows=rows, best_rows=rows)
            for name in runtime.FILES.values():
                (new_dir/f"best_{name}.pth").write_bytes((new_dir/f"{name}_final.pth").read_bytes())
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                with self.assertRaisesRegex(ValueError, "requires best"): runtime.load_agents(target)

    def test_real_train_best_selection_measurement_flags_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory); protocol, scales = root/"protocol.json", root/"scales.json"
            protocol.write_text(json.dumps(dict(version="as_nha_smoke_v1", train=[0, 1], validation=[600],
                test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
            scales_config = fixture_config(scales)
            scales_config["generation"] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in scales_config["scenes"]})
            scales.write_text(json.dumps(scales_config), encoding="utf-8")
            original_kwargs, original_episode = runtime.build_env_kwargs, runtime.run_episode
            runs = []
            for disabled in (False, True):
                scene = self.scene(output_root=str(root/("disabled" if disabled else "enabled")), episodes=2,
                    legacy_split=False, experiment_protocol=str(protocol), normalization_config=str(scales))
                snapshots, validations = [], 0
                def rollout(env, learners, **kwargs):
                    nonlocal validations
                    value = original_episode(env, learners, **kwargs)
                    if kwargs.get("training", False): snapshots.append(parameter_hash(learners))
                    else:
                        success, energy = [(2, 100.), (1, 1.)][validations]; validations += 1
                        value.update(total_energy=energy, wf_success_count=success, wf_success_rate=success/2)
                    return value
                flags = {key: "0" if disabled else "1" for key in ("EXPERIMENT_METRICS", "F11_DIAGNOSTICS", "F19_DIAGNOSTICS")}
                with patch.dict(os.environ, flags), patch.object(runtime, "build_agents", side_effect=tiny_agents), \
                     patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(original_kwargs)):
                    with redirect_stdout(io.StringIO()), patch.object(runtime, "run_episode", side_effect=rollout):
                        trained = runtime.train(scene)
                    target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                    loaded = runtime.load_agents(target)
                    self.assertEqual(parameter_hash(loaded), snapshots[0]); self.assertEqual(parameter_hash(trained), snapshots[1])
                    self.assertTrue(all(a._updates > 0 for a in trained.values()))
                    self.assertGreater(trained["VA"].f11.nonterminal, 0)
                    for transition in trained["VA"].buffer:
                        if not transition[-1]: self.assertGreater(transition[5].sum(), 0)
                    before = parameter_hash(loaded)
                    with redirect_stdout(io.StringIO()), patch.object(runtime, "load_agents", return_value=loaded): runtime.evaluate(target)
                    self.assertEqual(parameter_hash(loaded), before)
                    with self.assertRaises(FileExistsError): runtime.train(scene)
                    with self.assertRaises(FileExistsError): runtime.evaluate(target)
                summary = json.loads((scene.log_dir()/"training_summary.json").read_text())
                self.assertEqual(summary["selected_episode"], 0); self.assertTrue(summary["feasible_model_found"])
                self.assertEqual(set(summary["parameter_counts"]), {"TA", "VA"})
                self.assertEqual({a.loaded_selection["episode"] for a in loaded.values()}, {0})
                self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
                selections = [json.loads(x) for x in (scene.log_dir()/"validation_selection.jsonl").read_text().splitlines()]
                self.assertEqual([r["selected_as_best"] for r in selections], [True, False])
                self.assertEqual(sorted(p.name for p in scene.checkpoint_dir().glob("*.pth")),
                                 ["best_manager.pth", "best_vm.pth", "manager_final.pth", "vm_final.pth"])
                episodes = [json.loads(x) for x in (scene.log_dir()/"episodes.jsonl").read_text().splitlines()]
                self.assertEqual([r["random_ha"]["seed"] for r in episodes], [host_seed(0), host_seed(1)])
                for row in episodes:
                    self.assertAlmostEqual(row["reward_audit"]["VA_reward_formula_balance"], 0., places=7)
                    self.assertAlmostEqual(row["reward_audit"]["lower_energy_balance"], 0., places=7)
                    self.assertAlmostEqual(row["reward_audit"]["physical_energy_balance"], 0., places=7)
                    self.assertNotIn("HA_reward_formula_balance", row["reward_audit"])
                host_rows = [json.loads(x) for x in (scene.log_dir()/"hosts.jsonl").read_text().splitlines()]
                self.assertEqual(len(host_rows), sum(r["random_ha"]["decisions"] for r in episodes))
                if not disabled:
                    measurement = json.loads((scene.log_dir()/"training_measurements.jsonl").read_text().splitlines()[0])
                    self.assertEqual(set(measurement["policy_calls"]), {"TA", "VA"})
                    self.assertTrue(all(v > 0 for v in measurement["optimizer_steps"].values()))
                    evaluation = json.loads((target.evaluation_dir()/"measurements/episodes/seed700.json").read_text())
                    self.assertEqual(set(evaluation["timing"]["policy_call_counts"]), {"TA", "VA", "RandomHA"})
                runs.append((snapshots, selections, {r: a._updates for r, a in trained.items()}))
            self.assertEqual(runs[0], runs[1])

    def test_CLI_policy_paths_and_custom_topologies_are_isolated(self):
        registry = dict(MODEL_NAMES)
        for mode in ("train", "eval", "precompute"):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stopped:
                parse_scenario(mode, ["--dry-run", "--profile-mode", "off"])
            self.assertEqual(stopped.exception.code, 0)
            row = json.loads(out.getvalue())
            self.assertEqual(row["algorithm"], config.ALGORITHM)
            self.assertEqual(row["model_version"], config.MODEL_VERSION)
            self.assertEqual(row["ha_policy"]["host_apf"], "disabled")
            self.assertEqual(row["ha_policy"]["vm_apf"], "enabled")
            self.assertEqual(row["checkpoint_selection"], selection.selection_metadata())
        self.assertIs(selection.selection_key, base_runtime.selection_key)
        self.assertIs(selection.validation_selection, base_runtime.validation_selection)
        self.assertFalse(selection.validation_selection([validation_row(49)])["feasible"])
        self.assertTrue(selection.validation_selection([validation_row(50)])["feasible"])
        self.assertEqual(registry, MODEL_NAMES)
        baseline, nha = BaseScenario(), ScoreScenario()
        self.assertEqual(baseline.normalization_kwargs(), nha.normalization_kwargs())
        self.assertEqual(baseline.cache_path, nha.cache_path)
        self.assertEqual(len({baseline.checkpoint_dir(), nha.checkpoint_dir(), NTAScenario().checkpoint_dir()}), 3)
        self.assertLess(len(str(nha.evaluation_dir()/"measurements/episodes/seed700.json")), 260)
        topology = self.root/"custom.json"; topology.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])))
        custom = ScoreScenario(mode="eval", resource_config=str(topology), train_resource="S")
        self.assertEqual(custom.vms_per_host, (1, 2, 1, 3))
        self.assertEqual(custom.num_hosts, 4)


if __name__ == "__main__":
    unittest.main()
