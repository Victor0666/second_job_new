"""Global learned VM routing without HA/host APF, physics and model lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
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
from algorithms.mahdrl_oe_ge_apf_score_fast_tl import runtime, agent, config, selection
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.reward import finish_lower_phase
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nha import runtime as nha_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast_nha.scenario import ScoreScenario as NHAScenario
from common.test_mahdrl_as_fast_nha import tiny_agents as nha_tiny_agents


def tiny_agents(scene):
    return dict(TA=agent.TAAgent(15, 243, hidden_dims=(16, 8), batch_size=2, buffer_size=8,
                              device="cpu").bind(scene, "TA"),
                VA=agent.SetD3QNAgent(23, 14, hidden_dims=(16, 8), entity_dims=(8, 8),
                              batch_size=2, buffer_size=8, device="cpu").bind(scene, "VA"))


class TLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="as_tl_")
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

    def test_global_mask_retains_slow_hosts_and_keeps_per_host_VM_APF(self):
        old, env = self.env(base_runtime), self.env()
        owners = {j: hi for hi, h in enumerate(env.host_ids) for j in env.host_to_vm_indices[h]}
        duration = lambda tid, j: [1., 20., 100.][owners[j]] * (j % 3 + 1)
        for current in (old, env):
            with patch.object(current, "_task_duration", side_effect=duration), \
                 patch.object(current.assignment_ledger, "preview", side_effect=lambda hid, start, end, pc: end-start):
                if current is old:
                    host, ok = current.get_host_state_for_next_assignment()
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(host["mask"], [1., 0., 0.])
                    continue
                vm, ok = current.get_vm_state_for_next_assignment()
                self.assertTrue(ok)
                self.assertEqual(vm["obs"]["context"].shape, (23,))
                self.assertEqual(vm["obs"]["candidates"].shape, (env.num_vms, 14))
                self.assertEqual(vm["candidate_ids"], list(env.vm_ids))
                self.assertTrue(env.observation_space.contains(dict(vm_obs=vm["obs"], vm_mask=vm["mask"])))
                snapshot = env._apf_snapshot(env._cur_tid)
                expected = set().union(*snapshot["vms"].values())
                self.assertEqual(set(np.flatnonzero(vm["mask"])), expected)
                self.assertEqual(snapshot["hosts"], set(env.host_ids))
                self.assertIsNone(snapshot["r_H"])
                self.assertEqual(snapshot["energy_hosts"], set())
                self.assertLess(len(expected), env.num_vms)
                for hid in env.host_ids:
                    self.assertGreater(vm["mask"][env.host_to_vm_indices[hid]].sum(), 0)
                self.assertIs(vm, env._build_global_vm_state_for_task(env._cur_tid))
                for value in (*vm["obs"].values(), vm["mask"], vm["performance_times"]):
                    self.assertFalse(value.flags.writeable)
                j = next(j for j in env.host_to_vm_indices[env.host_ids[-1]] if vm["mask"][j])
                _, info = env.vm_assign_global(j)
                self.assertEqual(info["vm_global_idx"], j)
                self.assertEqual(info["host_id"], env.host_ids[-1])
                row = env.apf_decisions[-1]
                self.assertFalse(row["host_apf_enabled"])
                self.assertTrue(row["vm_apf_enabled"])
                self.assertEqual(row["raw_host_count"], row["retained_host_count"])
                self.assertEqual(row["global_retained_vm_count"], len(expected))
                self.assertEqual(row["vm_selection_scope"], "global")
                self.assertIsNone(env._fast_key)

    def test_same_actions_preserve_all_VA_information_physics_and_OE_rewards(self):
        old, env = self.env(base_runtime), self.env()
        assignments = 0
        with redirect_stdout(io.StringIO()):
            while not old.done_flag:
                while True:
                    host, oka = old.get_host_state_for_next_assignment()
                    global_vm, okb = env.get_vm_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    ah = int(np.flatnonzero(host["mask"])[0])
                    old.host_select(ah)
                    local_vm, legal = old.get_vm_state_for_current_task()
                    self.assertTrue(legal)
                    hid = old.host_ids[ah]
                    indices = old.host_to_vm_indices[hid]
                    x = local_vm["obs"]["context"]
                    np.testing.assert_array_equal(global_vm["obs"]["context"], np.concatenate((x[:22], x[27:28])))
                    np.testing.assert_array_equal(global_vm["obs"]["candidates"][indices, :7], local_vm["obs"]["candidates"])
                    np.testing.assert_array_equal(global_vm["obs"]["candidates"][indices, 7:12], np.broadcast_to(x[22:27], (len(indices), 5)))
                    np.testing.assert_array_equal(global_vm["obs"]["candidates"][indices, 12:14], np.broadcast_to(x[28:30], (len(indices), 2)))
                    np.testing.assert_array_equal(global_vm["mask"][indices], local_vm["mask"])
                    np.testing.assert_array_equal(global_vm["performance_times"][indices], local_vm["performance_times"])
                    saved = deepcopy(global_vm["obs"])
                    av = int(np.flatnonzero(local_vm["mask"])[0])
                    _, reward, info = old.vm_assign(av)
                    new_reward, new_info = env.vm_assign_global(indices[av])
                    self.assertEqual(reward, new_reward)
                    self.assertEqual(info, {key: new_info[key] for key in info})
                    self.assertEqual(old._records, env._records)
                    self.assertEqual(old.assignment_ledger.segments, env.assignment_ledger.segments)
                    for key in saved:
                        np.testing.assert_array_equal(saved[key], global_vm["obs"][key])
                    assignments += 1
                self.assertEqual(old.finish_phase_and_advance(), env.finish_phase_and_advance())
                np.testing.assert_array_equal(old.get_manager_state(), env.get_manager_state())
                self.assertEqual(old.done_flag, env.done_flag)
                if not old.done_flag:
                    old.apply_manager_delta(np.zeros(5)); env.apply_manager_delta(np.zeros(5))
        self.assertGreater(assignments, 20)
        self.assertEqual(old.total_energy, env.total_energy)
        self.assertEqual(repr(old.rng.get_state()), repr(env.rng.get_state()))

    def test_busy_tolerance_invalid_actions_and_empty_decisions_preserve_task(self):
        env = self.env()
        busy = env.host_to_vm_indices[env.host_ids[0]]
        env.vm_available_at[busy] = env.current_time + 5e-10
        vm, ok = env.get_vm_state_for_next_assignment()
        self.assertTrue(ok)
        self.assertTrue((vm["mask"][busy] == 0).all())
        tid, count = env._cur_tid, len(env._records)
        for action in (-1, env.num_vms, busy[0], .5, True, "0"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.vm_assign_global(action)
            self.assertEqual(env._cur_tid, tid)
            self.assertEqual(len(env._records), count)
        with self.assertRaisesRegex(RuntimeError, "no host selection"):
            env.host_select(0)
        env.vm_assign_global(np.int64(np.flatnonzero(vm["mask"])[0]))
        self.assertIsNone(env._fast_key)
        with self.assertRaises(RuntimeError):
            env.vm_assign_global(0)
        env = self.env()
        original = env._build_global_vm_state_for_task
        def unavailable(tid):
            state = dict(original(tid))
            state["mask"] = np.zeros(env.num_vms, np.float32)
            return state
        with patch.object(env, "_build_global_vm_state_for_task", side_effect=unavailable):
            empty, ok = env.get_vm_state_for_next_assignment()
        self.assertFalse(ok)
        self.assertEqual(empty["obs"]["candidates"].shape, (0, 14))
        tid = env._phase_tasks[0]
        self.assertIsNone(env._cur_tid)
        _, ok = env.get_vm_state_for_next_assignment()
        self.assertTrue(ok)
        self.assertEqual(env._cur_tid, tid)
        env = self.env()
        env.vm_available_at[:] = env.current_time + 100.
        _, ok = env.get_vm_state_for_next_assignment()
        self.assertFalse(ok)
        self.assertEqual(len(env._records), 0)

    def test_rollout_is_reproducible_and_never_invokes_host_decisions(self):
        learners = tiny_agents(self.scene())
        before = parameter_hash(learners)
        results, choices = [], []
        for _ in range(2):
            env, rows = self.env(seed=700), []
            with redirect_stdout(io.StringIO()), \
                 patch.object(env, "get_host_state_for_next_assignment", side_effect=AssertionError("HA state called")), \
                 patch.object(env, "host_select", side_effect=AssertionError("HA action called")):
                result = runtime.run_episode(env, learners, assignment_log=rows.append)
            self.assertEqual(set(result["rewards"]), {"TA", "VA"})
            self.assertNotIn("random_ha", result)
            self.assertEqual(len(rows), result["task_count"])
            self.assertEqual(result["wf_completed_count"], 2)
            self.assertTrue(all(r["legal_vm_count"] >= 1 for r in rows))
            self.assertTrue(all(not r["host_apf_enabled"] and r["vm_apf_enabled"] for r in env.apf_decisions))
            results.append(result); choices.append(rows)
        self.assertEqual(results[0], results[1]); self.assertEqual(choices[0], choices[1])
        self.assertEqual(parameter_hash(learners), before)
        self.assertTrue(all(a._updates == 0 for a in learners.values()))

    def test_formal_build_has_only_two_learners_and_unchanged_learning_math(self):
        runtime.set_seed(31)
        with patch.object(runtime, "SetD3QNAgent", wraps=runtime.SetD3QNAgent) as constructor:
            learners = runtime.build_agents(self.scene())
        self.assertEqual(set(learners), {"TA", "VA"})
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(constructor.call_args.args, (23, 14))
        for role, learner in learners.items():
            base_class = getattr(base_runtime, "TAAgent" if role == "TA" else "SetD3QNAgent")
            self.assertIs(type(learner)._update, base_class._update)
            self.assertEqual(config.learning_kwargs(role, "cpu"), base_runtime.learning_kwargs(role, "cpu"))
        with self.assertRaisesRegex(ValueError, "23 context"):
            agent.SetD3QNAgent(30, 7, hidden_dims=(8,), entity_dims=(8,), device="cpu").bind(self.scene(), "VA")
        with self.assertRaisesRegex(ValueError, "only TA and VA"):
            learners["VA"].bind(self.scene(), "HA")


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


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_TA_VA_CUDA_updates_reuse_baseline_math(self):
        options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device="cuda")
        learners = dict(TA=agent.TAAgent(15, 243, **options).bind(self.scene(), "TA"),
                        VA=agent.SetD3QNAgent(23, 14, entity_dims=(8, 8), **options).bind(self.scene(), "VA"))
        for role, learner in learners.items():
            before = parameter_hash({role: learner})
            for done in (False, True):
                if role == "TA":
                    state, mask, nxt, nm = np.ones(15), np.ones(243), np.zeros(15), np.zeros(243) if done else np.ones(243)
                else:
                    state, mask, nxt, nm = observation(3, 23, 14), np.ones(3), observation(0 if done else 4, 23, 14), np.ones(0 if done else 4)
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
            new_dir, old_dir, nta_dir = root/"tl", root/"baseline", root/"nta"
            for folder in (new_dir, old_dir, nta_dir): folder.mkdir()
            runtime.save_agents(learners, new_dir, "best", **args)
            base_runtime.save_agents(baseline, old_dir, "best", **args)
            nta_runtime.save_agents(nta, nta_dir, "best", **args)
            nha = nha_tiny_agents(self.scene(NHAScenario))
            nha_dir = root/"nha"; nha_dir.mkdir()
            nha_runtime.save_agents(nha, nha_dir, "best", **args)
            target = replace(scene, mode="eval", checkpoint_override=str(new_dir))
            with patch.object(runtime, "build_agents", side_effect=tiny_agents): loaded = runtime.load_agents(target)
            self.assertEqual(parameter_hash(learners), parameter_hash(loaded))
            self.assertFalse((new_dir/"best_host.pth").exists())
            for role, name in runtime.FILES.items():
                with self.assertRaises(ValueError): learners[role].load(str(old_dir/f"best_{name}.pth"))
                with self.assertRaises(ValueError): baseline[role].load(str(new_dir/f"best_{name}.pth"))
                with self.assertRaises(ValueError): learners[role].load(str(nha_dir/f"best_{name}.pth"))
                with self.assertRaises(ValueError): nha[role].load(str(new_dir/f"best_{name}.pth"))
            with self.assertRaises(ValueError): learners["VA"].load(str(nta_dir/"best_vm.pth"))
            with self.assertRaises(ValueError): nta["VA"].load(str(new_dir/"best_vm.pth"))
            with self.assertRaises(ValueError): learners["TA"].load(str(new_dir/"best_vm.pth"))
            with self.assertRaisesRegex(ValueError, "strict"): learners["TA"].load(str(new_dir/"best_manager.pth"), strict=False)
            checkpoint = torch.load(new_dir/"best_manager.pth", weights_only=True)
            for section in (None, "algorithm_variant", "implementation", config.CHECKPOINT_METADATA_KEY):
                tampered = deepcopy(checkpoint)
                meta = tampered["normalization"] if section is None else tampered["normalization"][section]
                meta["tl_policy"]["host_apf"] = "enabled"
                path = root/"tampered.pth"; torch.save(tampered, path)
                with self.subTest(section=section), self.assertRaises(ValueError): learners["TA"].load(str(path))
            manifest = new_dir/"tl_policy.json"; original = manifest.read_text(); manifest.write_text("{}")
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
            protocol.write_text(json.dumps(dict(version="as_tl_smoke_v1", train=[0, 1], validation=[600],
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
                for row in episodes:
                    self.assertAlmostEqual(row["reward_audit"]["VA_reward_formula_balance"], 0., places=7)
                    self.assertAlmostEqual(row["reward_audit"]["lower_energy_balance"], 0., places=7)
                    self.assertAlmostEqual(row["reward_audit"]["physical_energy_balance"], 0., places=7)
                    self.assertNotIn("HA_reward_formula_balance", row["reward_audit"])
                host_rows = [json.loads(x) for x in (scene.log_dir()/"assignments.jsonl").read_text().splitlines()]
                self.assertEqual(len(host_rows), sum(r["task_count"] for r in episodes))
                if not disabled:
                    measurement = json.loads((scene.log_dir()/"training_measurements.jsonl").read_text().splitlines()[0])
                    self.assertEqual(set(measurement["policy_calls"]), {"TA", "VA"})
                    self.assertTrue(all(v > 0 for v in measurement["optimizer_steps"].values()))
                    evaluation = json.loads((target.evaluation_dir()/"measurements/episodes/seed700.json").read_text())
                    self.assertEqual(set(evaluation["timing"]["policy_call_counts"]), {"TA", "VA"})
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
            self.assertEqual(row["tl_policy"]["host_apf"], "disabled")
            self.assertEqual(row["tl_policy"]["vm_apf"], "enabled")
            self.assertEqual(row["checkpoint_selection"], selection.selection_metadata())
        self.assertIs(selection.selection_key, base_runtime.selection_key)
        self.assertIs(selection.validation_selection, base_runtime.validation_selection)
        self.assertFalse(selection.validation_selection([validation_row(49)])["feasible"])
        self.assertTrue(selection.validation_selection([validation_row(50)])["feasible"])
        self.assertEqual(registry, MODEL_NAMES)
        baseline, tl = BaseScenario(), ScoreScenario()
        self.assertEqual(baseline.normalization_kwargs(), tl.normalization_kwargs())
        self.assertEqual(baseline.cache_path, tl.cache_path)
        self.assertEqual(len({baseline.checkpoint_dir(), tl.checkpoint_dir(), NTAScenario().checkpoint_dir()}), 3)
        self.assertLess(len(str(tl.evaluation_dir()/"measurements/episodes/seed700.json")), 260)
        topology = self.root/"custom.json"; topology.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])))
        custom = ScoreScenario(mode="eval", resource_config=str(topology), train_resource="S")
        self.assertEqual(custom.vms_per_host, (1, 2, 1, 3))
        self.assertEqual(custom.num_hosts, 4)

    def test_saved_source_transfers_to_unequal_VM_counts_on_unseen_topologies(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            source = self.scene(output_root=str(root/"source"))
            learners = tiny_agents(source)
            initial = parameter_hash(learners)
            rows = [validation_row(seed=source.validation_seeds()[0])]
            candidate = selection.validation_selection(rows)
            best = dict(candidate, episode=0, completed_episodes=1)
            source.checkpoint_dir().mkdir(parents=True)
            runtime.save_agents(learners, source.checkpoint_dir(), "best", episode=0,
                candidate=candidate, best=best, rows=rows, best_rows=rows)
            for counts in ([1, 2, 1, 3], [2, 1, 3, 1, 2]):
                spec, cache = root/f"topology{len(counts)}.json", root/f"cache{len(counts)}.json"
                spec.write_text(json.dumps(dict(vms_per_host=counts)), encoding="utf-8")
                with redirect_stdout(io.StringIO()):
                    precompute(ScoreScenario(mode="precompute", seed_spec="700", resource_config=str(spec), cache_output=str(cache)))
                target = self.scene(mode="eval", seed_spec="700", resource_config=str(spec),
                    train_resource="S", train_deadline="Tight", cache_file=str(cache),
                    checkpoint_override=str(source.checkpoint_dir()))
                runtime.validate_inputs(target)
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    loaded = runtime.load_agents(target)
                with patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(runtime.build_env_kwargs)):
                    env = runtime.build_env(target, 700)
                with redirect_stdout(io.StringIO()):
                    env.reset()
                    state, ok = env.get_vm_state_for_next_assignment()
                    self.assertTrue(ok)
                    self.assertEqual(state["obs"]["candidates"].shape, (sum(counts), 14))
                    env.reset(seed=700)
                    result = runtime.run_episode(env, loaded)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(parameter_hash(loaded), initial)



if __name__ == "__main__":
    unittest.main()
