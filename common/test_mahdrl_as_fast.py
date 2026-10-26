"""AS-Fast regression: numerical equivalence, cache lifetime, guards and real CLI flow."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.test_mahdrl_oe_ge_apf_score import observation, parameter_hash
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.experiment_protocol import protocol_binding
from environments.mahdrl_oe_ge_apf_score import CloudWorkflowEnv_VMAgents as OriginalEnv, apf_candidates as original_apf
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as FastEnv, apf_candidates
from algorithms.mahdrl_oe_ge_apf_score import agent as old, runtime as old_runtime
from algorithms.mahdrl_oe_ge_apf_score.networks import collate as old_collate
from algorithms.mahdrl_oe_ge_apf_score_fast import agent as fast, runtime
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import collate
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario, parse_scenario


def tiny_agents(scene, module=fast):
    agents = dict(TA=module.TAAgent(15, 243, hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=torch.device("cpu")),
                  HA=module.SetD3QNAgent(23, 10, hidden_dims=(16, 8), entity_dims=(8, 8), batch_size=2, buffer_size=8, device="cpu"),
                  VA=module.SetD3QNAgent(30, 7, hidden_dims=(16, 8), entity_dims=(8, 8), batch_size=2, buffer_size=8, device="cpu"))
    return {role: a.bind(scene, role) for role, a in agents.items()}


class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="as_fast_tests_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, **kwargs):
        args = dict(mode="train", episodes=1, legacy_split=True, normalization_config=str(self.scales),
                    cache_file=str(self.cache), output_root=str(self.root/"run"), device="cpu")
        return ScoreScenario(**dict(args, **kwargs))

    def env(self, cls=FastEnv, seed=0):
        args = runtime.build_env_kwargs(self.scene(), seed)
        args.update(workflows_per_episode=2, max_ready_tasks=100)
        env = cls(**args)
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    def test_packing_matches_all_lengths_padding_and_order(self):
        for sizes in ((6,)*256, (0, 0), (3, 18, 0, 6, 3)):
            obs = [observation(n) for n in sizes]
            masks = [np.ones(n, np.float32) for n in sizes]
            for a, b in zip(old_collate(obs, masks, "cpu"), collate(obs, masks, "cpu")):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        obs[0]["candidates"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            collate(obs, masks, "cpu")
        obs[0]["candidates"][0, 0] = 0.
        masks[0][0] = 2
        with self.assertRaises(ValueError):
            collate(obs, masks, "cpu")

    def test_insertion_checks_both_schemas_and_owns_readonly_arrays(self):
        a = fast.SetD3QNAgent(23, 10, hidden_dims=(8,), entity_dims=(8,), device="cpu")
        obs, mask = observation(3), np.ones(3)
        a.remember(obs, mask, 0, -.1, obs, mask, False)
        obs["context"][0] = 999
        mask[:] = 0
        self.assertNotEqual(a.buffer[0][0]["context"][0], 999)
        self.assertTrue(a.buffer[0][1].all())
        with self.assertRaises(ValueError):
            a.buffer[0][0]["context"][0] = 3
        with self.assertRaises(TypeError):
            a.buffer[0][0]["context"] = np.zeros(23)
        for nxt, nm, stop in ((observation(3, 24), np.ones(3), False),
                              (observation(0, 23, 11), np.zeros(0), True),
                              (observation(3), np.zeros(3), False)):
            with self.assertRaises(ValueError):
                a.remember(observation(3), np.ones(3), 0, 0, nxt, nm, stop)

    def test_numpy_apf_matches_scalar_random_ties_and_deadlines(self):
        rng = np.random.default_rng(19)
        for n in (0, 1, 5, 50, 180):
            for _ in range(30):
                rows = {i: (float(rng.choice([1., 1.25, 2., 4.])), float(rng.choice([1., 1.+1e-10, 8.])), 0.) for i in range(n)}
                hosts = {10+h*2: list(range(h*10, min(n, h*10+10))) for h in range(max(1, (n+9)//10))}
                ready, now, deadline = int(rng.integers(1, 80)), 3., float(rng.choice([3., 4., 5., 50.]))
                a, b = original_apf(rows, hosts, ready, now, deadline), apf_candidates(rows, hosts, ready, now, deadline)
                self.assertEqual(a, {k: v for k, v in b.items() if k != "arrays"})

    def test_episode_observations_masks_rewards_ledger_and_rng_equal(self):
        a, b = self.env(OriginalEnv), self.env()
        count = 0
        with redirect_stdout(io.StringIO()):
            while not a.done_flag:
                while True:
                    ha, oka = a.get_host_state_for_next_assignment()
                    hb, okb = b.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    for key in ha["obs"]:
                        np.testing.assert_array_equal(ha["obs"][key], hb["obs"][key])
                    np.testing.assert_array_equal(ha["mask"], hb["mask"])
                    np.testing.assert_array_equal(ha["performance_times"], hb["performance_times"])
                    ah = int(np.flatnonzero(ha["mask"])[-1])
                    a.host_select(ah); b.host_select(ah)
                    va, _ = a.get_vm_state_for_current_task()
                    vb, _ = b.get_vm_state_for_current_task()
                    for key in va["obs"]:
                        np.testing.assert_array_equal(va["obs"][key], vb["obs"][key])
                    np.testing.assert_array_equal(va["mask"], vb["mask"])
                    np.testing.assert_array_equal(va["performance_times"], vb["performance_times"])
                    av = int(np.flatnonzero(va["mask"])[-1])
                    self.assertEqual(a.vm_assign(av), b.vm_assign(av))
                    self.assertEqual(a.apf_decisions[-1], b.apf_decisions[-1])
                    self.assertIsNone(b._fast_key)
                    count += 1
                self.assertEqual(a.finish_phase_and_advance(), b.finish_phase_and_advance())
                np.testing.assert_array_equal(a.get_manager_state(), b.get_manager_state())
                if not a.done_flag:
                    a.apply_manager_delta(np.zeros(5)); b.apply_manager_delta(np.zeros(5))
        self.assertGreater(count, 20)
        self.assertEqual(a.total_energy, b.total_energy)
        self.assertEqual(a.assignment_ledger.segments, b.assignment_ledger.segments)
        self.assertEqual(repr(a.rng.get_state()), repr(b.rng.get_state()))

    def test_cache_reuse_readonly_preview_and_deadline_invalidation(self):
        env = self.env()
        host, ok = env.get_host_state_for_next_assignment()
        self.assertTrue(ok)
        before = (repr(env.rng.get_state()), repr(env._records), env.current_time,
                  repr(env.assignment_ledger.segments), repr(np.random.get_state()))
        with patch.object(env, "_compute_block1", wraps=env._compute_block1) as block:
            env.host_select(int(np.flatnonzero(host["mask"])[0]))
            self.assertEqual(block.call_count, 0)
        vm, _ = env.get_vm_state_for_current_task()
        cached = env._fast_cache[("vm_obs", env._cur_host_id)]
        self.assertIs(cached, env._build_vm_obs_for_task_host(env._cur_tid, env._cur_host_id))
        after = (repr(env.rng.get_state()), repr(env._records), env.current_time,
                 repr(env.assignment_ledger.segments), repr(np.random.get_state()))
        self.assertEqual(before, after)
        prev = env._apf_snapshot(env._cur_tid)
        env.task_baseline_finish[env._cur_tid] += 1
        self.assertIsNot(prev, env._apf_snapshot(env._cur_tid))
        env.reset()
        self.assertIsNone(env._fast_key)

    def compare_updates(self, device="cpu", full=False):
        for clsname, args in (("TAAgent", (15, 243)), ("SetD3QNAgent", (23, 10))):
            options = dict(batch_size=4, buffer_size=16, device=torch.device(device))
            if not full:
                options["hidden_dims"] = (16, 8)
                if clsname == "SetD3QNAgent":
                    options["entity_dims"] = (8, 8)
            runtime.set_seed(33)
            a = getattr(old, clsname)(*args, **options)
            runtime.set_seed(33)
            b = getattr(fast, clsname)(*args, **options)
            for i in range(8):
                if clsname == "TAAgent":
                    s, m, nxt, nm = np.arange(15)/15, np.ones(243), np.arange(15)/13, np.ones(243)
                    if i % 3 == 0: nm[:] = 0
                else:
                    s, m = observation(3+i%2), np.ones(3+i%2)
                    nxt, nm = observation(0 if i%3 == 0 else 6), np.ones(0 if i%3 == 0 else 6)
                for agent in (a, b): agent.remember(s, m, 0, -.1*i, nxt, nm, i%3 == 0)
            for step in range(3):
                random.seed(900+step)
                la = a.update()
                rng = random.getstate()
                random.seed(900+step)
                lb = b.update()
                self.assertEqual(random.getstate(), rng)
                self.assertEqual(la, lb)
                for x, y in zip(a.online.parameters(), b.online.parameters()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                for x, y in zip(a.target.parameters(), b.target.parameters()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(a._eps_steps, b._eps_steps)
            self.assertEqual(a.f11.snapshot()["sampled_targets"], b.f11.snapshot()["sampled_targets"])

    def test_same_replay_rng_loss_weights_targets_cpu(self):
        self.compare_updates()

    def test_complete_training_rollout_matches_original(self):
        results, agents, rngs = [], [], []
        for cls, module, runner in ((OriginalEnv, old, old_runtime), (FastEnv, fast, runtime)):
            runtime.set_seed(77)
            env = self.env(cls)
            learners = tiny_agents(self.scene(), module)
            with redirect_stdout(io.StringIO()):
                results.append(runner.run_episode(env, learners, training=True))
            agents.append(learners)
            rngs.append((repr(random.getstate()), repr(np.random.get_state()), torch.get_rng_state()))
        self.assertEqual(results[0], results[1])
        self.assertEqual(parameter_hash(agents[0]), parameter_hash(agents[1]))
        self.assertEqual(rngs[0][:2], rngs[1][:2])
        torch.testing.assert_close(rngs[0][2], rngs[1][2], rtol=0, atol=0)
        for role in agents[0]:
            self.assertEqual(agents[0][role]._updates, agents[1][role]._updates)

    def test_formal_network_width_update_cpu(self):
        self.compare_updates(full=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_same_replay_rng_loss_weights_targets_cuda(self):
        self.compare_updates(device="cuda", full=True)

    def test_nonfinite_gradient_guard_even_when_diagnostics_disabled(self):
        a = fast.SetD3QNAgent(23, 10, hidden_dims=(8,), entity_dims=(8,), batch_size=2, device="cpu", profile_mode="off")
        a.f11.numerics.enabled = False
        for _ in range(2): a.remember(observation(3), np.ones(3), 0, -.1, observation(3), np.ones(3), False)
        handle = next(a.online.parameters()).register_hook(lambda g: torch.full_like(g, float("nan")))
        with self.assertRaises(FloatingPointError):
            a.update()
        handle.remove()
        self.assertEqual(a._updates, 0)

    def test_saved_successor_mask_and_terminal_no_bootstrap(self):
        class Table(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))
                self.scale = scale
            def forward(self, context, features, exists, mask):
                return self.scale*features[:, :, 0] + self.bias
        a = fast.SetD3QNAgent(23, 10, hidden_dims=(8,), entity_dims=(8,), batch_size=2, device="cpu")
        a.online, a.target = Table(1.), Table(10.)
        a.optim = torch.optim.Adam(a.online.parameters())
        nxt = observation(2)
        nxt["candidates"][:, 0] = [1000., 2.]
        a.remember(observation(1), np.ones(1), 0, -1., nxt, np.array([0., 1.]), False)
        a.remember(observation(1), np.ones(1), 0, -3., observation(0), np.zeros(0), True)
        captured = []
        original_target = a.f11.target
        def capture(future, target):
            captured.extend(target.tolist())
            original_target(future, target)
        with patch.object(a.f11, "target", side_effect=capture):
            a.update()
        np.testing.assert_allclose(sorted(captured), [-3., 18.])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_stage_timings_flush_and_no_policy_rng_changes(self):
        a = fast.SetD3QNAgent(23, 10, hidden_dims=(8,), entity_dims=(8,), batch_size=2, device="cuda", profile_every=1)
        for _ in range(2): a.remember(observation(3), np.ones(3), 0, -.1, observation(6), np.ones(6), False)
        a.update()
        before = (repr(random.getstate()), repr(np.random.get_state()), torch.get_rng_state())
        report = a.profiler.report()
        self.assertGreater(report["cuda_elapsed"]["online_current_forward"]["seconds"], 0)
        self.assertGreater(report["cpu_exclusive"]["replay_pack"]["seconds"], 0)
        self.assertFalse(a.profiler.events)
        self.assertEqual(before[:2], (repr(random.getstate()), repr(np.random.get_state())))
        torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_diagnostics_and_profile_settings_do_not_change_training(self):
        agents = []
        for every, mode in ((1, "sampled"), (100, "off")):
            runtime.set_seed(222)
            a = fast.SetD3QNAgent(23, 10, hidden_dims=(8,), entity_dims=(8,), batch_size=2, device="cpu",
                                    diagnostics_every=every, profile_mode=mode)
            for _ in range(3): a.remember(observation(3), np.ones(3), 0, -.1, observation(6), np.ones(6), False)
            random.seed(333)
            for _ in range(3): a.update()
            agents.append(a)
        for x, y in zip(agents[0].online.parameters(), agents[1].online.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        before = agents[0].f11.numerics.snapshot()
        agents[0].select_action(observation(3), np.ones(3), deterministic=True, count_step=False)
        self.assertEqual(before, agents[0].f11.numerics.snapshot())

    def test_old_checkpoint_compatibility_strict_group_and_source(self):
        scene = self.scene()
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            agents = tiny_agents(scene, old)
            old_runtime.save_agents(agents, directory, "best")
            target = replace(scene, mode="eval", train_resource="S", checkpoint_override=directory)
            with patch.object(runtime, "build_agents", side_effect=tiny_agents):
                loaded = runtime.load_agents(target)
                self.assertEqual(parameter_hash(agents), parameter_hash(loaded))
                agents["TA"].checkpoint_group = "different"
                agents["TA"].save(str(Path(directory)/"best_manager.pth"))
                with self.assertRaisesRegex(ValueError, "different saved groups"):
                    runtime.load_agents(target)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                loaded["HA"].load(str(Path(directory)/"best_vm.pth"))
            wrong = tiny_agents(self.scene(resource="M"))["HA"]
            with self.assertRaisesRegex(ValueError, "source_train_scene"):
                wrong.load(str(Path(directory)/"best_host.pth"))

    def test_real_train_save_load_evaluate_and_measurements(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            protocol = Path(directory)/"protocol.json"
            protocol.write_text(json.dumps(dict(version="as_fast_smoke_v1", train=[0, 1], validation=[600],
                test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
            scales = Path(directory)/"scales.json"
            config = fixture_config(scales)
            config["generation"] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in config["scenes"]})
            scales.write_text(json.dumps(config), encoding="utf-8")
            scene = self.scene(output_root=directory, legacy_split=False, experiment_protocol=str(protocol), normalization_config=str(scales))
            original_kwargs = runtime.build_env_kwargs
            def small(scene, seed):
                return dict(original_kwargs(scene, seed), workflows_per_episode=2, max_ready_tasks=100)
            with patch.object(runtime, "build_agents", side_effect=tiny_agents), patch.object(runtime, "build_env_kwargs", side_effect=small):
                with redirect_stdout(io.StringIO()):
                    agents = runtime.train(scene)
                    self.assertTrue(all(a._updates > 0 for a in agents.values()))
                    target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                    loaded = runtime.load_agents(target)
                    before = parameter_hash(loaded)
                    with patch.object(runtime, "load_agents", return_value=loaded): runtime.evaluate(target)
                    self.assertEqual(parameter_hash(loaded), before)
                    with self.assertRaises(FileExistsError): runtime.train(scene)
                    with self.assertRaises(FileExistsError): runtime.evaluate(target)
            rows = [json.loads(x) for x in (scene.log_dir()/"stage_timings.jsonl").read_text().splitlines()]
            self.assertEqual([r["scope"] for r in rows], ["training", "validation"])
            self.assertIn("replay_pack", rows[0]["learners"]["HA"]["cpu_exclusive"])
            self.assertIn("online_current_forward", rows[0]["learners"]["VA"]["cpu_exclusive"])
            measurements = json.loads((scene.log_dir()/"training_measurements.jsonl").read_text().splitlines()[0])
            self.assertTrue(all(v > 0 for v in measurements["optimizer_steps"].values()))
            self.assertTrue((target.evaluation_dir()/"measurements/episodes/seed700.json").exists())

    def test_cli_modes_options_and_output_isolation(self):
        from algorithms.mahdrl_oe_ge_apf_score.scenario import ScoreScenario as OldScene
        from common.scenarios import MODEL_NAMES
        before = dict(MODEL_NAMES)
        for mode in ("train", "eval", "precompute"):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
                parse_scenario(mode, ["--dry-run", "--diagnostics-every", "200", "--profile-mode", "off"])
            self.assertEqual(stop.exception.code, 0)
            row = json.loads(out.getvalue())
            self.assertEqual(row["algorithm"], "mahdrl_oe_ge_apf_score_fast")
            self.assertEqual(row["diagnostics_every"], 200)
        self.assertEqual(before, MODEL_NAMES)
        self.assertNotEqual(OldScene().checkpoint_dir(), ScoreScenario().checkpoint_dir())
        self.assertLess(len(str(ScoreScenario(mode="eval").evaluation_dir()/"measurements/episodes/seed700.json")), 260)


if __name__ == "__main__":
    unittest.main()
