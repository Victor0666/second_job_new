"""Global host-batch actions/targets over the unchanged fixed93-to-9 VA."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
import random
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_hscore_tl import agent, runtime
from algorithms.mahdrl_as_fast_hscore_tl.config import MODEL_VERSION
from algorithms.mahdrl_as_fast_hscore_tl.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_as_fast_hscore_tl.updates import UpdateClock, DecisionTrace
from algorithms.mahdrl_as_fast_hscore import runtime as full_runtime
from algorithms.mahdrl_as_fast_hscore.agent import FixedVAAgent
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as FullScenario
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import ScoreScenario as OtherTLScenario


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=scene.device or "cpu")
    learners = dict(TA=agent.TAAgent(15, 243, **options), VA=agent.GroupedVAAgent(**options))
    return {role: a.bind(scene, role) for role, a in learners.items()}


class Table(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = scale

    def forward(self, obs, mask):
        return obs[:, :9]*self.scale + self.bias


class HScoreTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="hs_tl_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,2,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=ScoreScenario, **kwargs):
        values = dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName))
        return klass(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene=None, module=runtime, seed=0):
        with patch.object(module, "build_env_kwargs", self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(scene or self.scene(), seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def custom_scene(self, counts, source):
        spec = self.root/("topology_"+"_".join(map(str, counts))+".json")
        cache = spec.with_suffix(".cache.json")
        spec.write_text(json.dumps(dict(vms_per_host=counts)), encoding="utf-8")
        preparation = self.scene(mode="precompute", resource_config=str(spec), cache_output=str(cache), seed_spec="700")
        with redirect_stdout(io.StringIO()):
            precompute(preparation)
        return replace(source, mode="eval", resource_config=str(spec), cache_file=str(cache),
            train_resource="S", train_deadline="Tight", seed_spec="700")

    def test_global_rows_match_fixed_local_VA_and_physical_masks(self):
        env = self.env()
        busy = env.host_to_vm_indices[env.host_ids[0]][0]
        env.vm_available_at[busy] = env.current_time+100
        inactive = env.host_to_vm_indices[env.host_ids[2]]
        env.vm_available_at[inactive] = env.current_time+100
        with patch("environments.mahdrl_oe_ge_apf_score_fast.apf_candidates", side_effect=AssertionError("APF called")):
            state, ok = env.get_vm_state_for_next_assignment()
        self.assertTrue(ok)
        self.assertEqual(state["obs"].shape, (3, 93))
        self.assertEqual(state["mask"].shape, (27,))
        local = env.preview_vm_states()
        for row, hid in enumerate(env.host_ids):
            mask = state["mask"].reshape(-1, 9)[row]
            if hid in local:
                np.testing.assert_array_equal(state["obs"][row], local[hid]["obs"])
                np.testing.assert_array_equal(mask, local[hid]["mask"])
            else:
                np.testing.assert_array_equal(state["obs"][row], 0)
                np.testing.assert_array_equal(mask, 0)
            for slot, j in enumerate(env.host_to_vm_indices[hid]):
                self.assertEqual(state["candidate_ids"][row*9+slot], env.vm_ids[j])
                self.assertEqual(mask[slot], float(env.vm_available_at[j] <= env.current_time+1e-12))
        self.assertFalse(state["obs"].flags.writeable)
        self.assertTrue(env.observation_space["vm_obs"].contains(state["obs"]))
        for invalid in (0, 17, 18, 26, -1, 27, True, .5):
            with self.assertRaises(ValueError):
                env.vm_assign_global(invalid)
        with self.assertRaises(RuntimeError):
            env.host_select(0)

    def test_global_observation_is_pure_and_overdue_slow_VMs_remain_legal(self):
        env = self.env()
        _, ok = env.get_host_state_for_next_assignment()
        self.assertTrue(ok)
        env.task_baseline_finish[env._cur_tid] = env.current_time-10
        env._invalidate_decision()
        before = (env.current_time, deepcopy(env._records), deepcopy(env.assignment_ledger.segments),
            tuple(env.vm_available_at), env._cur_tid, env._cur_host_id, repr(env.rng.get_state()))
        python_rng, numpy_rng, torch_rng = random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone()
        state = env._build_global_vm_state_for_task(env._cur_tid)
        self.assertEqual(state["mask"].sum(), env.num_vms)
        self.assertEqual(before, (env.current_time, env._records, env.assignment_ledger.segments,
            tuple(env.vm_available_at), env._cur_tid, env._cur_host_id, repr(env.rng.get_state())))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(repr(np.random.get_state()), numpy_rng)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_global_greedy_and_GE_mix_across_unequal_host_candidate_counts(self):
        va = agent.GroupedVAAgent(hidden_dims=(8,), eps_start=.4)
        va.online = Table(1.)
        obs, mask, times = np.zeros((2, 93), np.float32), np.zeros(18, np.float32), np.full(18, 10.)
        mask[[0, 9, 10, 11]] = 1
        times[0] = 1
        obs[1, 2], obs[0, 8] = 9, 999
        self.assertEqual(va.select_action(obs, mask, deterministic=True, count_step=False, performance_times=times), 11)
        with patch("numpy.random.choice", return_value=11) as choose:
            va.select_action(obs, mask, performance_times=times)
        probabilities = choose.call_args.kwargs["p"]
        np.testing.assert_allclose(probabilities[[0, 9, 10, 11]], [.25, .05, .05, .65])
        np.testing.assert_array_equal(probabilities[mask == 0], 0)
        # Row permutation changes the action row, but retains the winning physical group/slot.
        swapped = va.select_action(obs[::-1].copy(), mask.reshape(2, 9)[::-1].ravel(),
            deterministic=True, count_step=False, performance_times=times.reshape(2, 9)[::-1].ravel())
        self.assertEqual(swapped, 2)

    def test_global_TD_uses_other_host_saved_masks_mixed_sizes_and_skips_terminal(self):
        va = agent.GroupedVAAgent(hidden_dims=(8,), batch_size=3, buffer_size=8)
        va.online, va.target = Table(1.), Table(10.)
        va.optim = torch.optim.Adam(va.online.parameters())
        current, mask = np.zeros((2, 93), np.float32), np.zeros(18, np.float32)
        current[1, 2], mask[11] = 3, 1
        nxt, next_mask = np.zeros((2, 93), np.float32), np.zeros(18, np.float32)
        nxt[0, 0], nxt[0, 1], nxt[1, 0], nxt[1, 1], nxt[1, 2] = 1000, 2, 4, 7, 5000
        next_mask[[1, 9, 10]] = 1
        va.remember(current, mask, 11, -1., nxt, next_mask, False)
        terminal = np.zeros((0, 93), np.float32)
        va.remember(current, mask, 11, -3., terminal, np.zeros(0), True)
        other, other_mask = np.zeros((3, 93), np.float32), np.zeros(27, np.float32)
        other[2, 8], other_mask[26] = 9, 1
        va.remember(current, mask, 11, -.2, other, other_mask, False)
        captured = []
        original = va.f11.target
        def capture(future, target):
            captured.extend(target.tolist())
            original(future, target)
        with patch.object(va.f11, "target", side_effect=capture), patch.object(va.target, "forward", wraps=va.target.forward) as target_forward:
            self.assertTrue(np.isfinite(va.update()))
        np.testing.assert_allclose(sorted(captured), [-3., 65.5, 85.3], rtol=1e-6)
        self.assertEqual(target_forward.call_count, 1)
        self.assertEqual(target_forward.call_args.args[0].shape, (2, 93))
        self.assertFalse(va.buffer[0][0].flags.writeable)
        self.assertFalse(va.buffer[0][5].flags.writeable)
        with self.assertRaises(ValueError):
            va.remember(current, mask, 0, 0., nxt, next_mask, False)
        with self.assertRaises(ValueError):
            va.remember(current, mask, 11, 0., terminal, np.zeros(0), False)
        with self.assertRaises(ValueError):
            va.remember(current[:, :-1], mask, 11, 0., nxt, next_mask, False)

    def test_network_is_the_same_fixed_slot_VA_and_production_updates_work(self):
        with torch.random.fork_rng():
            torch.manual_seed(77)
            full = FixedVAAgent(hidden_dims=(16, 8))
            torch.manual_seed(77)
            grouped = agent.GroupedVAAgent(hidden_dims=(16, 8))
        for key, value in full.online.state_dict().items():
            torch.testing.assert_close(value, grouped.online.state_dict()[key])
        env = self.env()
        options = runtime.learning_kwargs
        vm, _ = env.get_vm_state_for_next_assignment()
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            with patch.object(runtime, "learning_kwargs", side_effect=lambda role, device: dict(options(role, device), batch_size=2, buffer_size=8)):
                learners = runtime.build_agents(self.scene(device=device))
            self.assertEqual(set(learners), {"TA", "VA"})
            self.assertEqual(sum(p.numel() for p in learners["VA"].online.parameters()), 2330378)
            self.assertEqual(tuple(learners["VA"].online.feature[0].weight.shape), (1024, 93))
            va = learners["VA"]
            before = parameter_hash(learners)
            for _ in range(2):
                va.remember(vm["obs"], vm["mask"], int(np.flatnonzero(vm["mask"])[0]), -.1, vm["obs"], vm["mask"], False)
            self.assertTrue(np.isfinite(va.update()))
            self.assertNotEqual(before, parameter_hash(learners))

    def test_same_global_actions_preserve_full_hscore_physics_and_rewards(self):
        env = self.env()
        full = self.env(self.scene(FullScenario), module=full_runtime)
        with redirect_stdout(io.StringIO()):
            while not env.done_flag:
                while True:
                    vm, oka = env.get_vm_state_for_next_assignment()
                    _, okb = full.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    self.assertEqual(env._cur_tid, full._cur_tid)
                    action = int(np.flatnonzero(vm["mask"])[-1])
                    row, slot = divmod(action, 9)
                    full.host_select(row)
                    _, expected_reward, expected_info = full.vm_assign(slot)
                    reward, info = env.vm_assign_global(action)
                    self.assertEqual(reward, expected_reward)
                    self.assertEqual(info["vm_id"], vm["candidate_ids"][action])
                    for key in ("assignment_energy", "duration", "host_id", "vm_global_idx"):
                        if key in expected_info:
                            self.assertEqual(info[key], expected_info[key])
                    self.assertEqual(env._records, full._records)
                self.assertEqual(env.finish_phase_and_advance(), full.finish_phase_and_advance())
                if not env.done_flag:
                    env.apply_manager_delta(np.zeros(5))
                    full.apply_manager_delta(np.zeros(5))
        self.assertEqual(env.total_energy, full.total_energy)

    def test_real_updates_after_assignment_frozen_global_replay_and_reward_balance(self):
        scene, rows = self.scene(), []
        env, learners = self.env(scene), tiny_agents(scene)
        original = env.vm_assign_global
        def assign(action):
            before = learners["VA"]._updates
            result = original(action)
            self.assertEqual(learners["VA"]._updates, before)
            return result
        with patch.object(env, "vm_assign_global", side_effect=assign), redirect_stdout(io.StringIO()):
            result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append)
        self.assertGreater(learners["VA"]._updates, 10)
        self.assertEqual(result["wf_completed_count"], 2)
        self.assertEqual(result["task_count"], len(rows))
        self.assertEqual(set(result["update_clock"]["optimizer_steps"]), {"TA", "VA"})
        for key in ("physical_energy_balance", "lower_energy_balance", "VA_reward_formula_balance"):
            self.assertAlmostEqual(result["reward_audit"][key], 0., places=6)
        for s, m, a, r, n, nm, done in learners["VA"].buffer:
            self.assertEqual(s.shape, (3, 93))
            self.assertEqual(n.shape, (3, 93))
            self.assertEqual((m.shape, nm.shape), ((27,), (27,)))
            self.assertFalse(s.flags.writeable)
            self.assertFalse(n.flags.writeable)
        self.assertTrue(learners["VA"].buffer[-1][-1])
        self.assertTrue(all(not d["host_apf_enabled"] and not d["vm_apf_enabled"] for d in env.apf_decisions))

    def test_terminal_settlement_once_and_no_update_before_barrier(self):
        learners, env = tiny_agents(self.scene()), self.env()
        vm, _ = env.get_vm_state_for_next_assignment()
        clock = UpdateClock(learners, training=True, warmup=0)
        trace = DecisionTrace(clock, "VA")
        action = int(np.flatnonzero(vm["mask"])[0])
        trace.record(vm["obs"], vm["mask"], action, -.1)
        trace.advance(vm["obs"], vm["mask"])
        trace.record(vm["obs"], vm["mask"], action, -.2)
        trace.add_reward(-.3)
        trace.finish()
        trace.finish()
        self.assertEqual(len(learners["VA"].buffer), 2)
        self.assertEqual(learners["VA"]._updates, 0)
        self.assertAlmostEqual(learners["VA"].buffer[-1][3], -.5)
        clock.barrier()
        self.assertEqual(learners["VA"]._updates, 1)
        self.assertEqual(learners["VA"].buffer[-1][5].sum(), 0)
        clock.summary()

    def test_real_train_reload_flat_CSVs_and_cross_host_frozen_transfer(self):
        scene = self.scene()
        with patch.object(runtime, "build_env_kwargs", self.small_kwargs(runtime.build_env_kwargs)), \
                patch.object(runtime, "build_agents", side_effect=tiny_agents), redirect_stdout(io.StringIO()):
            runtime.train(scene)
            evaluation = replace(scene, mode="eval", train_resource="S", train_deadline="Tight", seed_spec="700")
            loaded = runtime.load_agents(evaluation)
            fingerprint = parameter_hash(loaded)
            counters = {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r, a in loaded.items()}
            for deadline in ("Tight", "Medium", "Loose"):
                target = replace(evaluation, deadline=deadline)
                runtime.evaluate(target)
                self.assertTrue(target.evaluation_csv().is_file())
                self.assertEqual(target.evaluation_csv().parent, evaluation.evaluation_root)
            for counts in ([8]*6, [9, 8]*6, [9, 8]*9, [1, 2, 1, 3]):
                target = self.custom_scene(counts, scene)
                runtime.validate_inputs(target)
                transferred = runtime.load_agents(target)
                result = runtime.run_episode(self.env(target, seed=700), transferred)
                self.assertEqual(result["wf_completed_count"], 2)
                self.assertEqual(parameter_hash(transferred), fingerprint)
                self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r, a in transferred.items()})
        self.assertEqual(len(list(evaluation.evaluation_root.glob("*.csv"))), 3)
        self.assertFalse((scene.checkpoint_dir()/"best_host.pth").exists())
        self.assertEqual({p.name for p in scene.checkpoint_dir().glob("*.pth")},
            {"best_manager.pth", "best_vm.pth", "manager_final.pth", "vm_final.pth"})
        with evaluation.evaluation_csv().open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["model_version"], MODEL_VERSION)
        self.assertEqual(row["vm_selection_scope"], "global_grouped_fixed9")
        self.assertEqual(row["vm_network_input_dim"], "93")
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        with self.assertRaises(FileExistsError):
            runtime.evaluate(evaluation)
        with self.assertRaises(FileExistsError):
            runtime.train(scene)
        # Shape-compatible HScore weights must still reject the different global policy identity.
        from common.test_mahdrl_as_fast_hscore import tiny_agents as full_tiny
        full = full_tiny(self.scene(FullScenario))
        with self.assertRaises(ValueError):
            full["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"))
        with self.assertRaises(ValueError):
            loaded["VA"].load(str(scene.checkpoint_dir()/"best_manager.pth"))
        with self.assertRaises(ValueError):
            loaded["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"), strict=False)
        from common.test_mahdrl_as_fast_vaaware_tl import tiny_agents as other_tiny
        other = other_tiny(self.scene(OtherTLScenario))
        with self.assertRaises(ValueError):
            other["TA"].load(str(scene.checkpoint_dir()/"best_manager.pth"))
        renamed = self.root/"renamed_final"
        renamed.mkdir()
        for stem in ("manager", "vm"):
            shutil.copy2(scene.checkpoint_dir()/f"{stem}_final.pth", renamed/f"best_{stem}.pth")
        with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "best checkpoints"):
            runtime.load_agents(replace(evaluation, checkpoint_override=str(renamed)))
        path = scene.checkpoint_dir()/"best_vm.pth"
        payload = torch.load(path, weights_only=True)
        payload["normalization"]["checkpoint_group"] = "wrong-group"
        torch.save(payload, path)
        with patch.object(runtime, "build_agents", side_effect=tiny_agents), self.assertRaisesRegex(ValueError, "different saved groups"):
            runtime.load_agents(evaluation)

    def test_paths_capacity_and_parser_have_no_HA_controls(self):
        paths, names = set(), set()
        for resource in "SML":
            for deadline in ("Tight", "Medium", "Loose"):
                source = self.scene(resource=resource, deadline=deadline)
                self.assertEqual(len(source.model_name()), 17)
                self.assertEqual(source.checkpoint_dir().parent.name, "mahdrl_hscore_tl")
                paths.add(source.checkpoint_dir())
                for task in "SML":
                    target = replace(source, mode="eval", task=task, train_resource=resource, train_deadline=deadline)
                    self.assertEqual(source.checkpoint_dir(), target.checkpoint_dir())
                    path = target.evaluation_csv()
                    self.assertEqual(path.parent, Path(source.output_root)/"outputs/run/mahdrl_hscore_tl")
                    self.assertIn(f"_{target.target_tag}_from_{source.source_tag}_", path.name)
                    self.assertNotIn("p1h4", path.name)
                    names.add(path.name)
        self.assertEqual((len(paths), len(names)), (9, 27))
        self.assertNotEqual(ScoreScenario().checkpoint_dir(), FullScenario().checkpoint_dir())
        self.assertNotEqual(ScoreScenario().checkpoint_dir(), OtherTLScenario().checkpoint_dir())
        bad = self.root/"capacity10.json"
        bad.write_text('{"vms_per_host": [10, 8]}', encoding="utf-8")
        for mode in ("eval", "precompute"):
            with self.assertRaisesRegex(ValueError, "capacity"):
                self.scene(mode=mode, train_resource="S", resource_config=str(bad))
        for args in (["--ha-update-period", "4"], ["--va-preview"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                parse_scenario("train", args)
            self.assertEqual(caught.exception.code, 2)
        stress = parse_scenario("eval", ["--workload-profile", "stress_a1_l006_ref003"])
        self.assertEqual(runtime.build_env_kwargs(stress, 700)["arrival_lambda"], .06)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_CUDA_global_host_batch_updates(self):
        scene = self.scene(device="cuda")
        learners = tiny_agents(scene)
        with redirect_stdout(io.StringIO()):
            result = runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
        self.assertGreater(learners["VA"]._updates, 0)
        self.assertEqual(result["wf_completed_count"], 2)


if __name__ == "__main__":
    unittest.main()
