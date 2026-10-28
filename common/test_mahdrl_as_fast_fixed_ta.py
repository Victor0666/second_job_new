"""Fixed ordering, retained learning, strict models and flat scenario CSV output."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import importlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import networkx as nx
import numpy as np
import torch
from common.experiment_protocol import protocol_binding
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, MODEL_NAMES
from common.test_f18 import fixture_config
from common.test_mahdrl_as_fast import validation_row
from common.test_mahdrl_oe_ge_apf_score import observation, parameter_hash
from common.workflow_opt import Workflow
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OEEnv
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as baseline
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nta.scenario import ScoreScenario as NTAScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nta.agent import SetD3QNAgent as NTAAgent
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta import runtime, agent
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.config import get_config
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.policy import ordered_ready_tasks
from algorithms.mahdrl_oe_ge_apf_score_fast_ertf.scenario import ScoreScenario as ERTF
from algorithms.mahdrl_oe_ge_apf_score_fast_edf.scenario import ScoreScenario as EDF
from algorithms.mahdrl_oe_ge_apf_score_fast_cp.scenario import ScoreScenario as CP


SCENES = (ERTF, EDF, CP)


def tiny_agents(scene, device="cpu"):
    return {role: agent.SetD3QNAgent(c, f, hidden_dims=(16, 8), entity_dims=(8, 8),
        batch_size=2, buffer_size=8, device=device).bind(scene, role)
        for role, c, f in (("HA", 23, 10), ("VA", 30, 7))}


class FixedTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="fixed_ta_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=ERTF, **kwargs):
        values = dict(mode="train", episodes=1, legacy_split=True, normalization_config=str(self.scales),
            cache_file=str(self.cache), output_root=str(self.root/"run"), device="cpu")
        return klass(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, klass=ERTF, seed=0):
        module = baseline if klass is BaseScenario else runtime
        with patch.object(module, "build_env_kwargs", side_effect=self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(self.scene(klass), seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_exact_ordering_ties_and_nonready_exclusion(self):
        env = SimpleNamespace(ready_task_ids=[4, 2, 1, 3, 0, 5],
            task_state=["Ready"]*5+["Running"], task_ready_time=[10., 5., 5., 2., 5., 0.],
            task_baseline_finish=[100., 50., 50., 80., 40., 0.], task_up_rank=[7., 10., 10., 8., 9., 100.])
        for rule, expected in (("ertf", [3, 1, 2, 4, 0]), ("edf", [4, 1, 2, 3, 0]), ("cp", [1, 2, 4, 3, 0])):
            self.assertEqual(ordered_ready_tasks(env, rule), expected)
        env.task_baseline_finish[0] = -1.
        self.assertEqual(ordered_ready_tasks(env, "edf")[0], 0)
        env.task_ready_time = [1.]*6
        env.task_baseline_finish = [2.]*6
        env.task_up_rank = [3.]*6
        for rule in ("ertf", "edf", "cp"):
            self.assertEqual(ordered_ready_tasks(env, rule), list(range(5)))

    def test_empty_invalid_and_nonfinite_ordering(self):
        env = SimpleNamespace(ready_task_ids=[], task_state=[], task_ready_time=[],
                              task_baseline_finish=[], task_up_rank=[])
        for rule in ("ertf", "edf", "cp"):
            self.assertEqual(ordered_ready_tasks(env, rule), [])
        with self.assertRaises(ValueError): ordered_ready_tasks(env, "unknown")
        for rule, attribute in (("ertf", "task_ready_time"), ("edf", "task_baseline_finish"), ("cp", "task_up_rank")):
            env = SimpleNamespace(ready_task_ids=[0], task_state=["Ready"], task_ready_time=[1.],
                                  task_baseline_finish=[2.], task_up_rank=[3.])
            setattr(env, attribute, [float("nan")])
            with self.assertRaises(ValueError): ordered_ready_tasks(env, rule)
        env.ready_task_ids = [0, 0]
        with self.assertRaisesRegex(ValueError, "unique"): ordered_ready_tasks(env, "ertf")

    def test_CP_uses_remaining_weighted_path_including_communication(self):
        tasks = [SimpleNamespace(workload_mi=w, ext_in_bits=0., from_parents_bits=0.,
            out_file_size_sum_bits=0., upward_rank=None) for w in (1., 100., 50.)]
        graph = nx.DiGraph(); graph.add_nodes_from(range(3)); graph.add_edge(0, 1)
        workflow = Workflow(0, graph, tasks, 0., 0)
        workflow.compute_upward_ranks(1., 1.)
        env = SimpleNamespace(ready_task_ids=[2, 0, 1], task_state=["Ready", "unReady", "Ready"],
            task_ready_time=[0.]*3, task_up_rank=[task.upward_rank for task in tasks])
        self.assertEqual(env.task_up_rank, [101., 100., 50.])
        self.assertEqual(ordered_ready_tasks(env, "cp"), [0, 2])
        tasks[2].ext_in_bits = 100e6
        workflow.compute_upward_ranks(1., 1.)
        env.task_up_rank = [task.upward_rank for task in tasks]
        self.assertEqual(ordered_ready_tasks(env, "cp"), [2, 0])

    def test_formal_HA_VA_initial_parameters_and_rng_match_main_experiment(self):
        baseline.set_seed(1)
        original = baseline.build_agents(self.scene(BaseScenario))
        expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        for klass in SCENES:
            runtime.set_seed(1)
            learners = runtime.build_agents(self.scene(klass))
            self.assertEqual(set(learners), {"HA", "VA"})
            for role in learners:
                self.assertIs(type(learners[role])._update, type(original[role])._update)
                self.assertEqual(runtime.learning_kwargs(role, "cpu"), baseline.learning_kwargs(role, "cpu"))
                for name in ("online", "target"):
                    for key, value in getattr(original[role], name).state_dict().items():
                        torch.testing.assert_close(getattr(learners[role], name).state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(random.getstate(), expected_rng[0])
            np.testing.assert_array_equal(np.random.get_state()[1], expected_rng[1][1])
            self.assertEqual(np.random.get_state()[2:], expected_rng[1][2:])
            torch.testing.assert_close(torch.get_rng_state(), expected_rng[2], rtol=0, atol=0)

    def test_identical_actions_preserve_OE_observations_rewards_and_physics(self):
        for klass in SCENES:
            with self.subTest(rule=klass.ordering_rule), redirect_stdout(io.StringIO()):
                env = self.env(klass)
                old = OEEnv(**self.small_kwargs(runtime.build_env_kwargs)(self.scene(klass), 0),
                            combo_weights=env.fixed_config.weights)
                for name in ("energy_reward_scale", "task_baseline_norm", "energy_norm_per_mi_ref",
                             "alpha_delay_host", "alpha_delay_vm"):
                    setattr(old, name, getattr(env, name))
                old.reset()
                count, same_time = 0, False
                while not old.done_flag:
                    while True:
                        host_a, ok_a = old.get_host_state_for_next_assignment()
                        host_b, ok_b = env.get_host_state_for_next_assignment()
                        self.assertEqual(ok_a, ok_b)
                        if not ok_a: break
                        self.assertEqual(old._cur_tid, env._cur_tid)
                        x, b = host_a["obs"], old.base_host_obs_dim
                        np.testing.assert_array_equal(host_b["obs"]["context"], np.concatenate((x[:22], x[b:b+1])))
                        np.testing.assert_array_equal(host_b["obs"]["candidates"][:, :5], x[22:b].reshape(old.num_hosts, 5))
                        expected_host_mask = [any(old.vm_available_at[j] <= old.current_time+1e-12
                            for j in old.host_to_vm_indices[hid]) for hid in old.host_ids]
                        np.testing.assert_array_equal(host_b["mask"], expected_host_mask)
                        raw = old._candidate_rows(old._cur_tid)
                        for hi, hid in enumerate(old.host_ids):
                            expected = min((raw[j][0] for j in old.host_to_vm_indices[hid]
                                if j in raw and old.vm_available_at[j] <= old.current_time+1e-12), default=np.inf)
                            self.assertAlmostEqual(host_b["performance_times"][hi], expected, places=9)
                        ah = int(np.flatnonzero(host_b["mask"])[0])
                        old.host_select(ah); env.host_select(ah)
                        vm_a, ok_a = old.get_vm_state_for_current_task()
                        vm_b, ok_b = env.get_vm_state_for_current_task()
                        self.assertEqual(ok_a, ok_b)
                        x, b = vm_a["obs"], old.base_vm_obs_dim
                        k = len(old.host_to_vm_indices[old.host_ids[ah]])
                        expected_context = np.concatenate((x[:27], x[b:b+1], x[b+1+old.num_hosts:b+3+old.num_hosts]))
                        expected_candidates = np.concatenate((x[27:b].reshape(old.max_vms_per_host, 5)[:k],
                            x[b+3+old.num_hosts:].reshape(old.max_vms_per_host, 2)[:k]), axis=1)
                        expected_candidates[vm_a["mask"][:k] <= .5, 5:] = 0.
                        np.testing.assert_array_equal(vm_b["obs"]["context"], expected_context)
                        np.testing.assert_array_equal(vm_b["obs"]["candidates"], expected_candidates)
                        np.testing.assert_array_equal(vm_a["mask"][:k], vm_b["mask"])
                        av = int(np.flatnonzero(vm_a["mask"])[0])
                        now = env.current_time
                        self.assertEqual(old.vm_assign(av), env.vm_assign(av))
                        same_time |= env.current_time == now
                        self.assertEqual(old._records, env._records)
                        self.assertEqual(old.assignment_ledger.segments, env.assignment_ledger.segments)
                        count += 1
                    self.assertEqual(old.finish_phase_and_advance(), env.finish_phase_and_advance())
                    self.assertEqual(old.done_flag, env.done_flag)
                self.assertGreater(count, 20)
                self.assertTrue(same_time)
                self.assertEqual(old.total_energy, env.total_energy)
                self.assertEqual(repr(old.rng.get_state()), repr(env.rng.get_state()))
                self.assertEqual(len(env.apf_decisions), count)
                self.assertTrue(all(not row["host_apf_enabled"] and not row["vm_apf_enabled"] for row in env.apf_decisions))

    def test_no_APF_retains_slow_hosts_slow_VMs_and_overdue_actions(self):
        for klass in SCENES:
            old, env = self.env(BaseScenario), self.env(klass)
            old.set_manager_combo(env.fixed_config.weights)
            owners = {j: hi for hi, hid in enumerate(env.host_ids) for j in env.host_to_vm_indices[hid]}
            duration = lambda tid, j: (1., 20., 100.)[owners[j]] * (j % 3 + 1)
            for current in (old, env):
                current.task_baseline_finish = [current.current_time-1.] * len(current.task_baseline_finish)
            with patch.object(old, "_task_duration", side_effect=duration), \
                 patch.object(old.assignment_ledger, "preview", side_effect=lambda hid, start, end, pc: end-start):
                pruned, ok = old.get_host_state_for_next_assignment()
                self.assertTrue(ok)
                np.testing.assert_array_equal(pruned["mask"], [1., 0., 0.])
                old.host_select(0)
                old_vm, ok = old.get_vm_state_for_current_task()
                self.assertTrue(ok)
                self.assertLess(old_vm["mask"].sum(), len(old_vm["mask"]))
            with patch.object(env, "_task_duration", side_effect=duration), \
                 patch.object(env.assignment_ledger, "preview", side_effect=lambda hid, start, end, pc: end-start), \
                 patch("environments.mahdrl_oe_ge_apf_score_fast.apf_candidates", side_effect=AssertionError("APF executed")):
                host, ok = env.get_host_state_for_next_assignment()
                self.assertTrue(ok)
                np.testing.assert_array_equal(host["mask"], np.ones(env.num_hosts))
                snapshot = env._apf_snapshot(env._cur_tid)
                self.assertEqual(set(snapshot["rows"]), set(range(env.num_vms)))
                self.assertEqual(snapshot["hosts"], set(env.host_ids))
                self.assertIsNone(snapshot["r_H"]); self.assertIsNone(snapshot["r_V"])
                env.host_select(env.num_hosts-1)
                vm, ok = env.get_vm_state_for_current_task()
                self.assertTrue(ok)
                np.testing.assert_array_equal(vm["mask"], np.ones(len(vm["mask"])))
                av = int(np.argmax(vm["performance_times"]))
                _, _, info = env.vm_assign(av)
                self.assertEqual(info["host_id"], env.host_ids[-1])
                self.assertFalse(env.apf_decisions[-1]["host_apf_enabled"])
                self.assertFalse(env.apf_decisions[-1]["vm_apf_enabled"])

    def test_no_APF_still_rejects_busy_and_padding_resources(self):
        for klass in SCENES:
            env = self.env(klass)
            busy = env.host_to_vm_indices[env.host_ids[0]]
            env.vm_available_at[busy] = env.current_time + 5e-10
            host, ok = env.get_host_state_for_next_assignment()
            self.assertTrue(ok)
            self.assertEqual(host["mask"][0], 0.)
            with self.assertRaises(ValueError): env.host_select(0)
            env.host_select(1)
            hid = env.host_ids[1]
            j = env.host_to_vm_indices[hid][0]
            env.vm_available_at[j] = env.current_time+1.
            vm, ok = env.get_vm_state_for_current_task()
            self.assertTrue(ok)
            self.assertEqual(vm["mask"][0], 0.)
            with self.assertRaises(ValueError): env.vm_assign(0)
            with self.assertRaises(ValueError): env.vm_assign(len(env.host_to_vm_indices[hid]))
            env.vm_assign(int(np.flatnonzero(vm["mask"])[0]))
            env = self.env(klass)
            ready = list(env.ready_task_ids)
            env.vm_available_at[:] = env.current_time+1.
            _, ok = env.get_host_state_for_next_assignment()
            self.assertFalse(ok)
            self.assertEqual(env.ready_task_ids, ready)

    def test_GE_remains_guided_and_uniform_exploration_on_raw_legal_actions(self):
        for klass in SCENES:
            learners = tiny_agents(self.scene(klass))
            self.assertEqual(get_config(klass.ordering_rule).ta_policy_metadata()["guided_probability"], .5)
            for role, learner in learners.items():
                c, f = (23, 10) if role == "HA" else (30, 7)
                obs, mask, times = observation(3, c, f), np.ones(3), np.array([100., 1., 10.])
                with patch("algorithms.mahdrl_oe_ge_apf_score_fast.agent.random.random", side_effect=[0., 0.]):
                    self.assertEqual(learner.select_action(obs, mask, performance_times=times), 1)
                with patch("algorithms.mahdrl_oe_ge_apf_score_fast.agent.random.random", side_effect=[0., .9]), \
                     patch("algorithms.mahdrl_oe_ge_apf_score_fast.agent.np.random.choice", return_value=0) as choice:
                    self.assertEqual(learner.select_action(obs, mask, performance_times=times), 0)
                    np.testing.assert_array_equal(choice.call_args.args[0], [0, 1, 2])

    def test_sorting_only_at_phase_start_fixed_weights_and_cache_reset(self):
        for klass in SCENES:
            env = self.env(klass)
            expected = np.array(env.fixed_config.weights)
            with self.assertRaises(RuntimeError): env.apply_manager_delta(np.zeros(5))
            with self.assertRaises(ValueError): env.set_manager_combo([.2]*5)
            with patch.object(env, "get_manager_state", side_effect=AssertionError("TA state called")), \
                 patch.object(env, "get_manager_action_mask", side_effect=AssertionError("TA action called")), \
                 redirect_stdout(io.StringIO()):
                result = runtime.run_episode(env, tiny_agents(self.scene(klass)))
            self.assertEqual(result["sorting_calls"], result["phase_count"])
            np.testing.assert_array_equal(env.combo_weights, expected)
            env._fast_cache["sentinel"] = True
            env.reset(seed=1)
            self.assertEqual(env.fixed_sort_calls, 0)
            np.testing.assert_array_equal(env.combo_weights, expected)
            self.assertNotIn("sentinel", env._fast_cache)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_CUDA_updates_for_both_roles_all_rules(self):
        for klass in SCENES:
            for role, learner in tiny_agents(self.scene(klass), "cuda").items():
                c, f = (23, 10) if role == "HA" else (30, 7)
                before = parameter_hash({role: learner})
                for done in (False, True):
                    learner.remember(observation(3, c, f), np.ones(3), 0, -.2,
                        observation(0 if done else 4, c, f), np.ones(0 if done else 4), done)
                self.assertTrue(np.isfinite(learner.update()))
                self.assertNotEqual(before, parameter_hash({role: learner}))
                self.assertEqual(learner._updates, 1)

    def test_checkpoint_rules_roles_groups_final_and_tampering_rejected(self):
        rows = [validation_row(seed=1)]
        candidate = runtime.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        for klass in SCENES:
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                root = Path(directory)
                scene = self.scene(klass, output_root=str(root), episodes=2)
                learners = tiny_agents(scene)
                ckpts = scene.checkpoint_dir(); ckpts.mkdir(parents=True)
                runtime.save_agents(learners, ckpts, "best", episode=0, candidate=candidate, best=best, rows=rows, best_rows=rows)
                target = replace(scene, mode="eval", checkpoint_override=str(ckpts))
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    runtime.load_agents(target)
                    for other in SCENES:
                        if other is klass: continue
                        with self.assertRaisesRegex(ValueError, "policy manifest"):
                            runtime.load_agents(self.scene(other, mode="eval", checkpoint_override=str(ckpts)))
                with self.assertRaises(ValueError): learners["HA"].load(str(ckpts/"best_vm.pth"))
                with self.assertRaisesRegex(ValueError, "strict"):
                    learners["HA"].load(str(ckpts/"best_host.pth"), strict=False)
                saved = torch.load(ckpts/"best_host.pth", weights_only=True)
                obsolete = deepcopy(saved)
                obsolete["normalization"]["model_version"] = "mahdrl_as_fast_v2_"+klass.ordering_rule+"_v1"
                obsolete_path = root/"old_apf_v1.pth"; torch.save(obsolete, obsolete_path)
                with self.assertRaisesRegex(ValueError, "model_version"):
                    learners["HA"].load(str(obsolete_path))
                for key in (None, "algorithm_variant", "implementation", scene.fixed_config.checkpoint_key):
                    modified = deepcopy(saved)
                    section = modified["normalization"] if key is None else modified["normalization"][key]
                    section["ta_policy"]["clock"] = "assignment"
                    path = root/"tampered.pth"; torch.save(modified, path)
                    with self.assertRaises(ValueError): learners["HA"].load(str(path))
                for other, learner_class in ((BaseScenario, baseline.SetD3QNAgent), (NTAScenario, NTAAgent)):
                    foreign = learner_class(23, 10, hidden_dims=(16, 8), entity_dims=(8, 8),
                        batch_size=2, buffer_size=8, device="cpu").bind(self.scene(other), "HA")
                    with self.assertRaises(ValueError): foreign.load(str(ckpts/"best_host.pth"))
                learners["HA"].checkpoint_group = "foreign_group"
                learners["HA"].save(str(ckpts/"best_host.pth"))
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    with self.assertRaisesRegex(ValueError, "different saved groups"): runtime.load_agents(target)
                runtime.save_agents(learners, ckpts, "final", episode=1, candidate=candidate, best=best, rows=rows, best_rows=rows)
                for name in runtime.FILES.values():
                    (ckpts/f"best_{name}.pth").write_bytes((ckpts/f"{name}_final.pth").read_bytes())
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    with self.assertRaisesRegex(ValueError, "requires best"): runtime.load_agents(target)

    def test_real_training_best_selection_frozen_evaluation_and_flat_CSV(self):
        for klass in SCENES:
            with self.subTest(rule=klass.ordering_rule), tempfile.TemporaryDirectory(dir=self.root) as directory:
                root = Path(directory); protocol, scales = root/"protocol.json", root/"scales.json"
                protocol.write_text(json.dumps(dict(version="fixed_ta_smoke_v1", train=[0, 1], validation=[600],
                    test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
                config = fixture_config(scales)
                config["generation"] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in config["scenes"]})
                scales.write_text(json.dumps(config), encoding="utf-8")
                scene = self.scene(klass, output_root=str(root), episodes=2, legacy_split=False,
                    experiment_protocol=str(protocol), normalization_config=str(scales))
                original_kwargs, original_episode = runtime.build_env_kwargs, runtime.run_episode
                snapshots, calls = [], []
                def rollout(env, agents, **kwargs):
                    result = original_episode(env, agents, **kwargs)
                    if kwargs.get("training", False):
                        snapshots.append(parameter_hash(agents))
                    else:
                        successes, energy = ((2, 100.), (1, 1.))[len(calls)]
                        calls.append(True)
                        result.update(total_energy=energy, wf_success_count=successes, wf_success_rate=successes/2)
                    return result
                with patch.object(runtime, "build_agents", side_effect=tiny_agents), \
                     patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(original_kwargs)), \
                     patch.dict(os.environ, {"EXPERIMENT_METRICS": "1"}):
                    with redirect_stdout(io.StringIO()), patch.object(runtime, "run_episode", side_effect=rollout):
                        trained = runtime.train(scene)
                    target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                    loaded = runtime.load_agents(target)
                    self.assertEqual(parameter_hash(loaded), snapshots[0])
                    self.assertEqual(parameter_hash(trained), snapshots[1])
                    for learner in trained.values():
                        self.assertGreater(learner._updates, 0)
                        self.assertTrue(learner.buffer[-1][-1])
                    before = parameter_hash(loaded)
                    before_counters = {role: (learner._updates, learner._eps_steps, len(learner.buffer))
                                       for role, learner in loaded.items()}
                    for deadline in ("Tight", "Medium"):
                        evaluation = replace(target, deadline=deadline)
                        with redirect_stdout(io.StringIO()), patch.object(runtime, "load_agents", return_value=loaded):
                            runtime.evaluate(evaluation)
                        self.assertEqual(parameter_hash(loaded), before)
                        self.assertEqual(before_counters, {role: (learner._updates, learner._eps_steps, len(learner.buffer))
                                                          for role, learner in loaded.items()})
                        self.assertEqual(evaluation.evaluation_csv().parent, target.evaluation_dir())
                        self.assertIn(evaluation.target_tag, evaluation.evaluation_csv().name)
                        self.assertNotIn(evaluation.target_tag, evaluation.evaluation_csv().parent.parts)
                        with evaluation.evaluation_csv().open(encoding="utf-8", newline="") as stream:
                            records = list(csv.DictReader(stream))
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]["scene"], evaluation.target_tag)
                        self.assertEqual(records[0]["source_scene"], "SS_T")
                        self.assertEqual(records[0]["ordering_rule"], klass.ordering_rule)
                        self.assertEqual(records[0]["host_apf_enabled"], "False")
                        self.assertEqual(records[0]["vm_apf_enabled"], "False")
                        self.assertIn("_noapf_", evaluation.evaluation_csv().name)
                        self.assertEqual(records[0]["model_seed"], "1")
                        self.assertEqual(records[0]["seed"], "700")
                        self.assertIn("total_energy", records[0])
                        with self.assertRaises(FileExistsError): runtime.evaluate(evaluation)
                        measured = json.loads((evaluation.evaluation_details_dir()/"measurements/episodes/seed700.json").read_text())
                        self.assertEqual(set(measured["timing"]["policy_call_counts"]), {"HA", "VA"})
                    self.assertEqual(len(list(target.evaluation_dir().glob("*.csv"))), 2)
                    with self.assertRaises(FileExistsError): runtime.train(scene)
                self.assertEqual(sorted(path.name for path in scene.checkpoint_dir().glob("*.pth")),
                    ["best_host.pth", "best_vm.pth", "host_final.pth", "vm_final.pth"])
                summary = json.loads((scene.log_dir()/"training_summary.json").read_text())
                self.assertEqual(summary["selected_episode"], 0)
                self.assertEqual(set(summary["parameter_counts"]), {"HA", "VA"})
                episodes = [json.loads(line) for line in (scene.log_dir()/"episodes.jsonl").read_text().splitlines()]
                self.assertEqual([row["instance_seed"] for row in episodes], [0, 1])
                self.assertTrue(all(row["fixed_ta"]["rule"] == klass.ordering_rule for row in episodes))

    def test_all_scene_CSV_names_are_unique_and_parent_is_scene_independent(self):
        paths = []
        for klass in SCENES:
            parents = set()
            for task in "SML":
                for resource in "SML":
                    for deadline in ("Tight", "Medium", "Loose"):
                        scene = klass(mode="eval", task=task, resource=resource, deadline=deadline,
                            train_task="S", train_resource=resource, train_deadline=deadline)
                        paths.append(scene.evaluation_csv())
                        parents.add(scene.evaluation_csv().parent)
                        self.assertEqual(scene.model_seed, 1)
                        self.assertLess(len(str(scene.evaluation_csv())), 260)
            self.assertEqual(len(parents), 1)
        self.assertEqual(len(paths), 81)
        self.assertEqual(len({path.name for path in paths}), 81)
        target = ERTF(mode="eval", task="M", resource="S", deadline="Loose", train_resource="S", train_deadline="Tight")
        self.assertIn("MS_L_from_SS_T", target.evaluation_csv().name)
        for changed in (replace(target, model_seed=2), replace(target, seed_spec="700-728"),
                        replace(target, seed_spec="700,702,729"), replace(target, train_deadline="Medium")):
            self.assertNotEqual(target.evaluation_csv(), changed.evaluation_csv())
        self.assertNotEqual(replace(target, seed_spec="700,702,729").evaluation_csv(),
                            replace(target, seed_spec="700,703,729").evaluation_csv())

    def test_measurement_flags_preserve_learning_for_all_rules(self):
        for klass in SCENES:
            results = []
            for enabled in ("0", "1"):
                with tempfile.TemporaryDirectory(dir=self.root) as directory, \
                     patch.dict(os.environ, {key: enabled for key in ("EXPERIMENT_METRICS", "F11_DIAGNOSTICS", "F19_DIAGNOSTICS")}):
                    scene = self.scene(klass)
                    runtime.set_seed(1)
                    learners = tiny_agents(scene)
                    env = self.env(klass)
                    with redirect_stdout(io.StringIO()):
                        recorder = runtime.TrainingRecorder(Path(directory), scene, scene.fixed_config.algorithm, env, learners)
                        try:
                            value = runtime.run_episode(env, learners, training=True)
                        finally:
                            recorder.close()
                    results.append((parameter_hash(learners), value,
                        {role: (learner._updates, learner._eps_steps, len(learner.buffer)) for role, learner in learners.items()}))
            self.assertEqual(results[0], results[1])

    def test_models_transfer_to_unequal_VM_topology_with_named_CSV(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            spec, cache = root/"topology.json", root/"topology_cache.json"
            spec.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                precompute(ERTF(mode="precompute", seed_spec="700", resource_config=str(spec), cache_output=str(cache)))
            for klass in SCENES:
                scene = self.scene(klass, output_root=str(root/klass.ordering_rule))
                learners = tiny_agents(scene)
                expected = parameter_hash(learners)
                rows = [validation_row(seed=seed) for seed in scene.validation_seeds()]
                candidate = runtime.validation_selection(rows)
                scene.checkpoint_dir().mkdir(parents=True)
                runtime.save_agents(learners, scene.checkpoint_dir(), "best", episode=0,
                    candidate=candidate, best=dict(candidate, episode=0, completed_episodes=1), rows=rows, best_rows=rows)
                target = self.scene(klass, mode="eval", seed_spec="700", resource_config=str(spec),
                    train_resource="S", train_deadline="Tight", cache_file=str(cache),
                    output_root=str(root/klass.ordering_rule), checkpoint_override=str(scene.checkpoint_dir()))
                with patch.object(runtime, "build_agents", side_effect=tiny_agents), \
                     patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(runtime.build_env_kwargs)), \
                     redirect_stdout(io.StringIO()):
                    loaded = runtime.load_agents(target)
                    with patch.object(runtime, "load_agents", return_value=loaded):
                        runtime.evaluate(target)
                self.assertEqual(parameter_hash(loaded), expected)
                self.assertTrue(target.evaluation_csv().is_file())
                self.assertIn(target.target_tag, target.evaluation_csv().name)
                self.assertEqual(target.evaluation_csv().parent, scene.evaluation_dir())
                self.assertLess(len(str(target.evaluation_csv())), 260)

    def test_cli_protocol_and_source_normalization_match_main(self):
        registry = dict(MODEL_NAMES)
        for klass in SCENES:
            parser = importlib.import_module("algorithms.mahdrl_oe_ge_apf_score_fast_"+klass.ordering_rule+".scenario").parse_scenario
            for mode in ("train", "eval", "precompute"):
                with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as stop:
                    parser(mode, ["--dry-run", "--profile-mode", "off"])
                self.assertEqual(stop.exception.code, 0)
                info = json.loads(output.getvalue())
                self.assertEqual(info["ta_policy"]["rule"], klass.ordering_rule)
                self.assertEqual(info["model_seed"], 1)
                self.assertEqual(info["ta_policy"]["host_apf"], "disabled")
                self.assertEqual(info["ta_policy"]["vm_apf"], "disabled")
                self.assertIn("_noapf_srgt092_v3", info["model_version"])
                self.assertEqual(info["checkpoint_selection"], runtime.selection_metadata())
            fixed, main = klass(mode="train"), BaseScenario(mode="train")
            self.assertEqual(fixed.normalization_kwargs(), main.normalization_kwargs())
            self.assertEqual(fixed.cache_path, main.cache_path)
            self.assertEqual([fixed.training_seed(i) for i in range(600)], list(range(600)))
            self.assertEqual(fixed.validation_seeds(), [600])
            self.assertNotEqual(fixed.checkpoint_dir(), main.checkpoint_dir())
            self.assertIsNot(runtime.validation_selection, baseline.validation_selection)
        self.assertEqual(registry, MODEL_NAMES)


if __name__ == "__main__":
    unittest.main()
