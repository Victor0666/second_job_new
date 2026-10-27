"""Real-decision clocks, physical invariants, update budgets and isolated models."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
import os
from pathlib import Path
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
from common.test_mahdrl_oe_ge_apf_score import observation, parameter_hash
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as base_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_sync import runtime, agent
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.updates import UpdateClock, DecisionTrace
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.metrics import training_stability
from algorithms.mahdrl_oe_ge_apf_score_fast_sync_task.scenario import ScoreScenario as TaskScenario, parse_scenario as task_parser
from algorithms.mahdrl_oe_ge_apf_score_fast_sync_phase.scenario import ScoreScenario as PhaseScenario, parse_scenario as phase_parser
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE


ZERO_ACTION = next(i for i, delta in enumerate(MANAGER_ACTION_TABLE) if not np.any(delta))


def tiny_agents(scene, device="cpu"):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=device)
    learners = dict(TA=agent.TAAgent(15, 243, **options),
        HA=agent.SetD3QNAgent(23, 10, entity_dims=(8, 8), **options),
        VA=agent.SetD3QNAgent(30, 7, entity_dims=(8, 8), **options))
    return {role: value.bind(scene, role) for role, value in learners.items()}


class SyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="as_sync_tests_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=TaskScenario, **kwargs):
        values = dict(mode="train", episodes=1, legacy_split=True,
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/"run"), device="cpu")
        return klass(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, klass=TaskScenario, seed=0):
        scene = self.scene(klass)
        module = base_runtime if klass is BaseScenario else runtime
        with patch.object(module, "build_env_kwargs", side_effect=self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def transition(self, role, *, done=False, size=3):
        if role == "TA":
            return np.ones(15), np.ones(243), 0, -.2, np.zeros(15), np.zeros(243) if done else np.ones(243), done
        c, f = (23, 10) if role == "HA" else (30, 7)
        return observation(size, c, f), np.ones(size), 0, -.2, observation(0 if done else size+1, c, f), np.ones(0 if done else size+1), done

    def test_phase_credits_preserve_every_eligible_update_and_flush_once(self):
        learners = tiny_agents(self.scene(PhaseScenario))
        clock = UpdateClock(learners, "phase", training=True, warmup=0)
        for role in learners:
            for _ in range(3):
                clock.remember(role, *self.transition(role))
        self.assertEqual(clock.debt, dict(TA=2, HA=2, VA=2))
        self.assertTrue(all(value._updates == 0 for value in learners.values()))
        self.assertEqual(clock.barrier(), dict(TA=2, HA=2, VA=2))
        self.assertEqual(clock.barrier(), dict(TA=0, HA=0, VA=0))
        self.assertEqual(clock.summary()["optimizer_steps"], dict(TA=2, HA=2, VA=2))
        self.assertTrue(all(value._updates == 2 for value in learners.values()))

    def test_pending_experience_uses_actual_successor_and_terminal_cutoff(self):
        learners = tiny_agents(self.scene(PhaseScenario))
        clock = UpdateClock(learners, "phase", training=True, warmup=0)
        for role in learners:
            trace = DecisionTrace(clock, role)
            transition = self.transition(role)
            state, mask = transition[:2]
            trace.record(state, mask, 0, -.2)
            self.assertEqual(len(learners[role].buffer), 0)
            trace.add_reward(-.3)
            nxt, nm = transition[4:6]
            trace.advance(nxt, nm)
            saved = learners[role].buffer[-1]
            self.assertAlmostEqual(saved[3], -.5)
            self.assertFalse(saved[-1])
            if role == "TA":
                np.testing.assert_array_equal(saved[4], nxt)
            else:
                for key in nxt:
                    np.testing.assert_array_equal(saved[4][key], nxt[key])
            np.testing.assert_array_equal(saved[5], nm)
            trace.record(nxt, nm, 0, -.1)
            trace.finish(truncated=True)
            self.assertTrue(learners[role].buffer[-1][-1])
            self.assertFalse(learners[role].buffer[-1][5].any())
            self.assertEqual(learners[role].f11.truncation_cutoffs, 1)
            self.assertIsNone(trace.pending)
            self.assertAlmostEqual(trace.recorded_reward, trace.emitted_reward)
        self.assertTrue(all(value._updates == 0 for value in learners.values()))
        clock.barrier()
        self.assertTrue(all(value._updates == 1 for value in learners.values()))

    def test_task_resequencing_preserves_time_and_active_physical_phase(self):
        env = self.env()
        found = False
        with redirect_stdout(io.StringIO()):
            for _ in range(200):
                while env.can_task_decide():
                    env.apply_task_manager_delta(MANAGER_ACTION_TABLE[ZERO_ACTION])
                    host, ok = env.get_host_state_for_next_assignment()
                    self.assertTrue(ok)
                    env.host_select(int(np.flatnonzero(host["mask"])[0]))
                    vm, ok = env.get_vm_state_for_current_task()
                    self.assertTrue(ok)
                    env.vm_assign(int(np.flatnonzero(vm["mask"])[0]))
                    if env.can_task_decide():
                        before = (env.current_time, env._phase_assign_cnt, list(env._phase_assigned_tids),
                            env._phase_size_sum_mi, len(env._records), env.sort_calls)
                        env.apply_task_manager_delta(MANAGER_ACTION_TABLE[ZERO_ACTION])
                        after = (env.current_time, env._phase_assign_cnt, list(env._phase_assigned_tids),
                            env._phase_size_sum_mi, len(env._records), env.sort_calls-1)
                        self.assertEqual(before, after)
                        self.assertTrue(env._phase_started)
                        self.assertFalse(env._phase_preadvanced)
                        self.assertEqual(set(env._phase_tasks), {tid for tid in env.ready_task_ids if env.task_state[tid] == "Ready"})
                        found = True
                        break
                if found:
                    break
                env.finish_phase_and_advance()
        self.assertTrue(found, "Fixture needs a same-clock batch with at least two assignments")

    def test_same_actions_preserve_baseline_physics_APFs_and_manager_reward(self):
        records, apfs, results = [], [], []
        for klass in (BaseScenario, TaskScenario, PhaseScenario):
            env = self.env(klass)
            env.set_manager_combo([1., 0., 0., 0., 0.])
            learners = base_tiny_agents(self.scene(klass)) if klass is BaseScenario else tiny_agents(self.scene(klass))
            module = base_runtime if klass is BaseScenario else runtime
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                for role, learner in learners.items():
                    choose = (lambda state, mask, **kwargs: ZERO_ACTION) if role == "TA" else (lambda state, mask, **kwargs: int(np.flatnonzero(mask)[0]))
                    stack.enter_context(patch.object(learner, "select_action", side_effect=choose))
                if klass is TaskScenario:
                    original_finish = env.finish_phase_and_advance
                    def finish():
                        self.assertFalse(env.can_task_decide(), "Must not advance while another same-clock allocation is legal")
                        return original_finish()
                    stack.enter_context(patch.object(env, "finish_phase_and_advance", side_effect=finish))
                result = module.run_episode(env, learners)
            records.append(env._records)
            apfs.append(env.apf_decisions)
            results.append(result)
            self.assertEqual(result["total_energy"], results[0]["total_energy"])
            self.assertEqual(result["makespan"], results[0]["makespan"])
            self.assertEqual(env._records, records[0])
            self.assertEqual(env.apf_decisions, apfs[0])
        expected_reward = results[0]["rewards"]["TA"]*results[0]["phase_count"]
        self.assertAlmostEqual(results[1]["rewards"]["TA"], expected_reward)
        self.assertAlmostEqual(results[2]["rewards"]["TA"], expected_reward)
        self.assertEqual(results[1]["decision_counts"]["TA"], len(records[1]))
        self.assertEqual(results[2]["decision_counts"]["TA"], results[0]["phase_count"])
        self.assertGreater(results[1]["sorting"]["nonempty_calls"], results[2]["sorting"]["nonempty_calls"])

    def test_task_barriers_match_assignments_and_phase_barriers_flush_full_budget(self):
        for klass in (TaskScenario, PhaseScenario):
            env, learners, rows, updates = self.env(klass), tiny_agents(self.scene(klass)), [], []
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                for role, learner in learners.items():
                    original = learner.update
                    def update(original=original, role=role):
                        updates.append((role, len(env._records), env._phase_started))
                        return original()
                    stack.enter_context(patch.object(learner, "update", side_effect=update))
                result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append)
            self.assertEqual(result["wf_completed_count"], 2)
            self.assertEqual(result["reward_units"], "episode_total")
            self.assertEqual(len(rows), result["policy_assignments"])
            self.assertEqual(sum(sum(row["optimizer_steps"].values()) for row in rows), len(updates) if klass is TaskScenario else 0)
            for key, value in result["reward_audit"].items():
                if key.endswith("balance") or key.endswith("gap"):
                    self.assertAlmostEqual(value, 0., places=7, msg=key)
            for role, learner in learners.items():
                self.assertGreater(learner._updates, 0)
                self.assertEqual(result["learning_schedule"]["optimizer_steps"][role], learner._updates)
                self.assertEqual(result["learning_schedule"]["pending_credits"][role], 0)
                self.assertEqual(learner.f11.transitions-learner.f11.nonterminal, 1)
                for transition in learner.buffer:
                    if not transition[-1]:
                        self.assertGreater(transition[5].sum(), 0)
            if klass is TaskScenario:
                self.assertEqual(result["learning_schedule"]["barriers"], result["policy_assignments"])
                self.assertEqual(result["decision_counts"]["TA"], result["policy_assignments"])
                self.assertEqual(result["sorting"]["nonempty_calls"], result["policy_assignments"])
                self.assertTrue(any(row["decision_time"] == nxt["decision_time"] for row, nxt in zip(rows, rows[1:])))
                self.assertTrue(any(flag for _, _, flag in updates))
            else:
                self.assertTrue(all(not flag for _, _, flag in updates))
                self.assertEqual(result["learning_schedule"]["barriers"], result["phase_count"])
                for role in learners:
                    self.assertEqual(learners[role]._updates, result["learning_schedule"]["transitions"][role]-1)
                self.assertGreater(learners["HA"]._updates, result["phase_count"])

    def test_formal_initial_parameters_and_learning_math_match_baseline(self):
        runtime.set_seed(31)
        baseline = base_runtime.build_agents(self.scene(BaseScenario))
        for klass in (TaskScenario, PhaseScenario):
            runtime.set_seed(31)
            learners = runtime.build_agents(self.scene(klass))
            self.assertEqual(parameter_hash(learners), parameter_hash(baseline))
            for role in learners:
                self.assertIs(type(learners[role])._update, type(baseline[role])._update)

    def test_task_TA_transitions_use_actual_next_decisions_and_partition_tardy_cost(self):
        env, learners = self.env(), tiny_agents(self.scene())
        env.task_baseline_finish[:] = [env.current_time-1.] * len(env.task_baseline_finish)
        states, transitions, phases = [], [], []
        original_select, original_remember = learners["TA"].select_action, learners["TA"].remember
        def select(state, mask, **kwargs):
            states.append(np.array(state, copy=True))
            return original_select(state, mask, **kwargs)
        def remember(*transition):
            transitions.append(transition)
            return original_remember(*transition)
        with redirect_stdout(io.StringIO()), patch.object(learners["TA"], "select_action", side_effect=select), \
             patch.object(learners["TA"], "remember", side_effect=remember):
            result = runtime.run_episode(env, learners, training=True, warmup=0, phase_log=phases.append)
        self.assertEqual(len(transitions), len(states))
        self.assertGreater(sum(row["phase_delay"] for row in phases), 0.)
        self.assertTrue(any(transition[3] == 0 for transition in transitions))
        self.assertTrue(any(transition[3] < 0 for transition in transitions))
        self.assertAlmostEqual(sum(transition[3] for transition in transitions), sum(row["manager_reward"] for row in phases))
        for i, transition in enumerate(transitions):
            np.testing.assert_array_equal(transition[0], states[i])
            if i+1 < len(states):
                self.assertFalse(transition[-1])
                np.testing.assert_array_equal(transition[4], states[i+1])
            else:
                self.assertTrue(transition[-1])
                self.assertFalse(transition[5].any())
        self.assertAlmostEqual(result["reward_audit"]["TA_interval_reward_balance"], 0., places=7)

    def test_real_phase_cutoff_closes_all_roles_without_bootstrapping(self):
        for klass in (TaskScenario, PhaseScenario):
            env, learners = self.env(klass), tiny_agents(self.scene(klass))
            original = env.finish_phase_and_advance
            def stop():
                reward, info = original()
                env.done_flag = True
                info.update(terminated=False, truncated=True)
                return reward, info
            with redirect_stdout(io.StringIO()), patch.object(env, "finish_phase_and_advance", side_effect=stop):
                result = runtime.run_episode(env, learners, training=True, warmup=0)
            self.assertFalse(result["reward_audit"]["complete_episode"])
            self.assertEqual(result["learning_schedule"]["pending_credits"], dict(TA=0, HA=0, VA=0))
            for learner in learners.values():
                self.assertEqual(learner.f11.truncation_cutoffs, 1)
                self.assertTrue(learner.buffer[-1][-1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_CUDA_real_updates_for_all_roles_and_both_clocks(self):
        for klass in (TaskScenario, PhaseScenario):
            learners = tiny_agents(self.scene(klass), device="cuda")
            clock = UpdateClock(learners, klass.sync_mode, training=True, warmup=0)
            before = parameter_hash(learners)
            for role in learners:
                clock.remember(role, *self.transition(role, size=3))
                clock.remember(role, *self.transition(role, done=True, size=5))
            self.assertEqual(clock.barrier(), dict(TA=1, HA=1, VA=1))
            self.assertTrue(all(value._updates == 1 for value in learners.values()))
            self.assertNotEqual(parameter_hash(learners), before)

    def test_CLI_metadata_source_paths_topologies_and_registry_are_isolated(self):
        registry = dict(MODEL_NAMES)
        for parser, klass in ((task_parser, TaskScenario), (phase_parser, PhaseScenario)):
            for mode in ("train", "eval", "precompute"):
                with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stopped:
                    parser(mode, ["--dry-run", "--profile-mode", "off"])
                self.assertEqual(stopped.exception.code, 0)
                value = json.loads(out.getvalue())
                self.assertEqual(value["synchronization"]["mode"], klass.sync_mode)
                self.assertEqual(value["synchronization"]["host_apf"], "enabled")
                self.assertEqual(value["synchronization"]["vm_apf"], "enabled")
            scene = klass()
            self.assertEqual(scene.normalization_kwargs(), BaseScenario().normalization_kwargs())
            self.assertEqual(scene.cache_path, BaseScenario().cache_path)
            self.assertLess(len(str(scene.evaluation_dir()/"measurements/episodes/seed700.json")), 260)
            for resource in "SML":
                for deadline in ("Tight", "Medium", "Loose"):
                    source = klass(resource=resource, deadline=deadline)
                    for task in "SML":
                        target = klass(mode="eval", task=task, resource=resource, deadline=deadline,
                            train_resource=resource, train_deadline=deadline)
                        self.assertEqual(target.checkpoint_dir(), source.checkpoint_dir())
        self.assertEqual(len({BaseScenario().checkpoint_dir(), TaskScenario().checkpoint_dir(), PhaseScenario().checkpoint_dir()}), 3)
        self.assertEqual(registry, MODEL_NAMES)

    def test_stability_uses_raw_validation_metrics_and_counts_drop_events(self):
        success = [.98, 1., 1., 1., 1., 1., .98, .96, 1., .98]
        rows = [dict(episode=i, validation_success_rate=sr, validation_energy=100.+i) for i, sr in enumerate(success)]
        stats = training_stability(rows, tail_window=3)
        self.assertEqual(stats["first_sustained_feasible_completed_episode"], 6)
        self.assertEqual(stats["post_sustained_feasibility_drop_count"], 2)
        self.assertEqual(stats["tail_validation_energy_mean"], 108.)
        self.assertEqual(stats["tail_mean_absolute_energy_change"], 1.)
        self.assertIsNone(training_stability(rows[:3])["first_sustained_feasible_completed_episode"])
        with self.assertRaises(ValueError):
            training_stability([dict(episode=0, validation_success_rate=1., validation_energy=float("nan"))])

    def test_checkpoints_reject_other_clocks_baseline_tampering_groups_and_final(self):
        rows = [validation_row(seed=seed) for seed in self.scene().validation_seeds()]
        candidate = runtime.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        args = dict(episode=0, candidate=candidate, best=best, rows=rows, best_rows=rows)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            groups = {}
            for klass in (TaskScenario, PhaseScenario, BaseScenario):
                scene = self.scene(klass)
                module = base_runtime if klass is BaseScenario else runtime
                learners = base_tiny_agents(scene) if klass is BaseScenario else tiny_agents(scene)
                folder = root/klass.__module__.split(".")[-2]
                folder.mkdir()
                module.save_agents(learners, folder, "best", **args)
                groups[klass] = (scene, learners, folder)
            for klass in (TaskScenario, PhaseScenario):
                scene, learners, folder = groups[klass]
                target = replace(scene, mode="eval", checkpoint_override=str(folder))
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    loaded = runtime.load_agents(target)
                self.assertEqual(parameter_hash(loaded), parameter_hash(learners))
                for other in (BaseScenario, PhaseScenario if klass is TaskScenario else TaskScenario):
                    _, foreign, other_folder = groups[other]
                    for role, name in runtime.FILES.items():
                        with self.assertRaises(ValueError):
                            learners[role].load(str(other_folder/f"best_{name}.pth"))
                        with self.assertRaises(ValueError):
                            foreign[role].load(str(folder/f"best_{name}.pth"))
                with self.assertRaises(ValueError):
                    learners["TA"].load(str(folder/"best_host.pth"))
                with self.assertRaisesRegex(ValueError, "strict"):
                    learners["TA"].load(str(folder/"best_manager.pth"), strict=False)
                original = torch.load(folder/"best_manager.pth", weights_only=True)
                for section in (None, "algorithm_variant", "implementation", scene.sync_config.checkpoint_key):
                    tampered = deepcopy(original)
                    meta = tampered["normalization"] if section is None else tampered["normalization"][section]
                    meta["synchronization"]["mode"] = "foreign"
                    path = root/"tampered.pth"; torch.save(tampered, path)
                    with self.assertRaises(ValueError):
                        learners["TA"].load(str(path))
                manifest = folder/"synchronization.json"
                saved_manifest = manifest.read_text(encoding="utf-8")
                manifest.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    runtime.load_agents(target)
                manifest.write_text(saved_manifest, encoding="utf-8")
                learners["HA"].checkpoint_group = "foreign_group"
                learners["HA"].save(str(folder/"best_host.pth"))
                with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "different saved groups"):
                    runtime.load_agents(target)
                runtime.save_agents(learners, folder, "final", episode=1, candidate=candidate,
                    best=best, rows=rows, best_rows=rows)
                for name in runtime.FILES.values():
                    (folder/f"best_{name}.pth").write_bytes((folder/f"{name}_final.pth").read_bytes())
                with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "requires best"):
                    runtime.load_agents(target)
                # Restore the source before the other clock checks cross-loading.
                runtime.save_agents(learners, folder, "best", **args)

    def test_real_training_best_selection_curves_frozen_eval_and_measurement_flags(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            protocol, scales = root/"protocol.json", root/"scales.json"
            protocol.write_text(json.dumps(dict(version="as_sync_smoke_v1", train=[0, 1],
                validation=[600], test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
            scales_config = fixture_config(scales)
            scales_config["generation"] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in scales_config["scenes"]})
            scales.write_text(json.dumps(scales_config), encoding="utf-8")
            original_kwargs, original_episode = runtime.build_env_kwargs, runtime.run_episode
            for klass in (TaskScenario, PhaseScenario):
                runs = []
                for disabled in (False, True):
                    scene = self.scene(klass, output_root=str(root/(klass.sync_mode+str(disabled))),
                        episodes=2, legacy_split=False, experiment_protocol=str(protocol),
                        normalization_config=str(scales), profile_mode="off" if disabled else "sampled")
                    snapshots, validations = [], 0
                    def rollout(env, learners, **kwargs):
                        nonlocal validations
                        result = original_episode(env, learners, **kwargs)
                        if kwargs.get("training", False):
                            snapshots.append(parameter_hash(learners))
                        else:
                            success, energy = [(2, 100.), (1, 1.)][validations]
                            validations += 1
                            result.update(total_energy=energy, wf_success_count=success, wf_success_rate=success/2)
                        return result
                    flags = {key: "0" if disabled else "1" for key in ("EXPERIMENT_METRICS", "F11_DIAGNOSTICS", "F19_DIAGNOSTICS")}
                    with patch.dict(os.environ, flags), patch.object(runtime, "build_agents", side_effect=tiny_agents), \
                         patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(original_kwargs)):
                        with redirect_stdout(io.StringIO()), patch.object(runtime, "run_episode", side_effect=rollout):
                            trained = runtime.train(scene)
                        target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                        loaded = runtime.load_agents(target)
                        self.assertEqual(parameter_hash(loaded), snapshots[0])
                        self.assertEqual(parameter_hash(trained), snapshots[1])
                        before = (parameter_hash(loaded), {r: (a._updates, a._eps_steps, a._action_calls,
                            len(a.buffer), a.f11.transitions, a.f11.truncation_cutoffs) for r, a in loaded.items()})
                        with redirect_stdout(io.StringIO()), patch.object(runtime, "load_agents", return_value=loaded):
                            runtime.evaluate(target)
                        after = (parameter_hash(loaded), {r: (a._updates, a._eps_steps, a._action_calls,
                            len(a.buffer), a.f11.transitions, a.f11.truncation_cutoffs) for r, a in loaded.items()})
                        self.assertEqual(before, after)
                        with self.assertRaises(FileExistsError):
                            runtime.train(scene)
                        with self.assertRaises(FileExistsError):
                            runtime.evaluate(target)
                    summary = json.loads((scene.log_dir()/"training_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["selected_episode"], 0)
                    self.assertTrue(summary["feasible_model_found"])
                    self.assertEqual(summary["synchronization"]["mode"], klass.sync_mode)
                    self.assertEqual(summary["training_stability"]["observations"], 2)
                    self.assertTrue(all(a._updates > 0 for a in trained.values()))
                    with (scene.log_dir()/"learning_curves.csv").open(encoding="utf-8", newline="") as f:
                        curves = list(csv.DictReader(f))
                    self.assertEqual([float(row["validation_success_rate"]) for row in curves], [1., .5])
                    self.assertTrue(all(int(row["sorting_calls"]) > 0 for row in curves))
                    self.assertEqual(sorted(p.name for p in scene.checkpoint_dir().glob("*.pth")),
                        ["best_host.pth", "best_manager.pth", "best_vm.pth", "host_final.pth", "manager_final.pth", "vm_final.pth"])
                    selections = [json.loads(line) for line in (scene.log_dir()/"validation_selection.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual([row["selected_as_best"] for row in selections], [True, False])
                    if not disabled:
                        measured = json.loads((scene.log_dir()/"training_measurements.jsonl").read_text(encoding="utf-8").splitlines()[0])
                        self.assertEqual(set(measured["policy_calls"]), {"TA", "HA", "VA"})
                        self.assertTrue(all(value > 0 for value in measured["optimizer_steps"].values()))
                        evaluation = json.loads((target.evaluation_dir()/"measurements/episodes/seed700.json").read_text(encoding="utf-8"))
                        self.assertEqual(set(evaluation["timing"]["policy_call_counts"]), {"TA", "HA", "VA"})
                    runs.append((snapshots, selections, {r: a._updates for r, a in trained.items()}))
                self.assertEqual(runs[0], runs[1])

    def test_saved_three_role_models_transfer_to_unequal_VM_count_topology(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            spec, cache = root/"topology.json", root/"topology_cache.json"
            spec.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                precompute(TaskScenario(mode="precompute", seed_spec="700", resource_config=str(spec), cache_output=str(cache)))
            for klass in (TaskScenario, PhaseScenario):
                scene = self.scene(klass, output_root=str(root/klass.sync_mode))
                learners = tiny_agents(scene)
                original = parameter_hash(learners)
                rows = [validation_row(seed=seed) for seed in scene.validation_seeds()]
                candidate = runtime.validation_selection(rows)
                scene.checkpoint_dir().mkdir(parents=True)
                runtime.save_agents(learners, scene.checkpoint_dir(), "best", episode=0,
                    candidate=candidate, best=dict(candidate, episode=0, completed_episodes=1), rows=rows, best_rows=rows)
                target = self.scene(klass, mode="eval", seed_spec="700", resource_config=str(spec),
                    train_resource="S", train_deadline="Tight", cache_file=str(cache), checkpoint_override=str(scene.checkpoint_dir()))
                runtime.validate_inputs(target)
                with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                    loaded = runtime.load_agents(target)
                with patch.object(runtime, "build_env_kwargs", side_effect=self.small_kwargs(runtime.build_env_kwargs)):
                    env = runtime.build_env(target, 700)
                with redirect_stdout(io.StringIO()):
                    env.reset()
                    result = runtime.run_episode(env, loaded)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(parameter_hash(loaded), original)


if __name__ == "__main__":
    unittest.main()
