"""MAHDRL-AS: masks, reward parity, set learning and isolated real CLI lifecycle."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.f18_state_normalization import CURRENT_VERSION
from common.experiment_protocol import protocol_binding
from environments.mahdrl_oe_ge_pf import CloudWorkflowEnv_VMAgents as FixedPF
from environments.mahdrl_oe_ge_apf_score import CloudWorkflowEnv_VMAgents as AS, apf_candidates
from algorithms.mahdrl_oe_ge_apf_score.agent import TAAgent, SetD3QNAgent
from algorithms.mahdrl_oe_ge_apf_score.networks import CandidateDuelingQNet, collate
from algorithms.mahdrl_oe_ge_apf_score.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_oe_ge_apf_score import runtime


def observation(n, c=23, f=10):
    return dict(context=np.arange(c, dtype=np.float32)/c,
                candidates=np.arange(n*f, dtype=np.float32).reshape(n, f)/max(1,n*f),
                exists_mask=np.ones(n, np.float32))


def parameter_hash(agents):
    h = hashlib.sha256()
    for agent in agents.values():
        for tensor in agent.online.state_dict().values():
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class APFScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="mahdrl_as_tests_")
        cls.root = Path(cls.temp.name)
        cls.scales = cls.root/"scales.json"
        fixture_config(cls.scales)
        cls.cache = cls.root/"cache.json"
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, **kwargs):
        values = dict(mode="train", episodes=1, legacy_split=True, normalization_mode=CURRENT_VERSION,
                      normalization_config=str(self.scales), cache_file=str(self.cache),
                      output_root=str(self.root/"run"), device="cpu")
        values.update(kwargs)
        return ScoreScenario(**values)

    def env(self, cls=AS, seed=0, **kwargs):
        values = runtime.build_env_kwargs(self.scene(), seed)
        values.update(workflows_per_episode=2, max_ready_tasks=100, **kwargs)
        env = cls(**values)
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    @staticmethod
    def tiny_agents(scene):
        agents = dict(TA=TAAgent(15, 243, hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=torch.device("cpu")),
                      HA=SetD3QNAgent(23, 10, hidden_dims=(16, 8), entity_dims=(8, 8), batch_size=2, buffer_size=8, device="cpu"),
                      VA=SetD3QNAgent(30, 7, hidden_dims=(16, 8), entity_dims=(8, 8), batch_size=2, buffer_size=8, device="cpu"))
        return {role: a.bind(scene, role) for role,a in agents.items()}

    def test_energy_channel_reaches_both_layers_and_ontime_gate(self):
        rows = {0: (1., 20., 0.), 1: (4., 1., 0.), 2: (1.1, 15., 0.), 3: (3., 1., 0.)}
        hosts = {10: [0], 20: [1, 3], 30: [2], 40: [999]}
        result = apf_candidates(rows, hosts, 2, 0., 10.)
        self.assertIn(20, result["hosts"])
        self.assertEqual(result["energy_vms"][20], {1, 3})
        self.assertTrue({1, 3} <= result["vms"][20])
        self.assertNotIn(40, result["hosts"])
        late = apf_candidates(rows, hosts, 2, 0., 2.)
        self.assertNotIn(20, late["hosts"])
        self.assertFalse(late["energy_vms"][20])
        self.assertIn(10, late["hosts"])
        empty = apf_candidates({}, hosts, 1, 0., 2.)
        self.assertFalse(empty["hosts"])

    def test_apf_contains_fixed_pf_for_same_state(self):
        rng = np.random.default_rng(123)
        for _ in range(100):
            rows = {i: (float(rng.uniform(.1, 20)), float(rng.uniform(1, 100)), 0.) for i in range(18)}
            hosts = {h: list(range(3*h, 3*h+3)) for h in range(6)}
            s = apf_candidates(rows, hosts, int(rng.integers(1, 50)), 0., float(rng.uniform(.1, 30)))
            fastest = min(x[0] for x in rows.values())
            for h, indices in hosts.items():
                th = min(rows[j][0] for j in indices)
                if th <= 1.25*fastest:
                    self.assertIn(h, s["hosts"])
                self.assertTrue({j for j in indices if rows[j][0] <= 1.25*th} <= s["vms"][h])
            self.assertLessEqual(s["r_V"], 1.75)
            self.assertGreaterEqual(s["r_H"], s["r_V"])

    def test_network_permutation_padding_and_empty_state(self):
        net = CandidateDuelingQNet(23, 10, (16, 8), (8, 8))
        obs, mask = observation(5), np.array([1, 0, 1, 1, 0], np.float32)
        q = net(*collate([obs], [mask], "cpu"))[0]
        permutation = np.array([3, 0, 4, 2, 1])
        shuffled = dict(obs, candidates=obs["candidates"][permutation], exists_mask=obs["exists_mask"][permutation])
        other = net(*collate([shuffled], [mask[permutation]], "cpu"))[0]
        torch.testing.assert_close(other, q[permutation])
        padded = dict(obs, candidates=np.vstack((obs["candidates"], np.full((4, 10), 999., np.float32))),
                      exists_mask=np.r_[obs["exists_mask"], np.zeros(4, np.float32)])
        qp = net(*collate([padded], [np.r_[mask, np.zeros(4, np.float32)]], "cpu"))[0]
        torch.testing.assert_close(qp[:5], q)
        empty = observation(0)
        self.assertTrue(torch.isfinite(net(*collate([empty], [np.zeros(0, np.float32)], "cpu"))).all())

    def test_default_widths_train_on_mixed_candidate_counts(self):
        agent = SetD3QNAgent(23, 10, batch_size=2, device="cpu")
        for n, k in ((3, 12), (18, 6)):
            current, successor = observation(n), observation(k)
            agent.remember(current, np.ones(n), n-1, -.2, successor, np.ones(k), False)
        self.assertTrue(np.isfinite(agent.update()))
        self.assertEqual(tuple(agent.online.entity[0].weight.shape), (128, 10))
        self.assertEqual(tuple(agent.online.feature[0].weight.shape), (1024, 281))
        self.assertEqual(agent._eps_steps, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_shared_scoring_update(self):
        agent = SetD3QNAgent(30, 7, batch_size=2, device="cuda")
        for n, k in ((4, 12), (12, 8)):
            agent.remember(observation(n, 30, 7), np.ones(n), 0, -.1,
                           observation(k, 30, 7), np.ones(k), False)
        self.assertTrue(np.isfinite(agent.update()))
        self.assertEqual(next(agent.online.parameters()).device.type, "cuda")

    def test_td_uses_saved_next_mask_and_skips_terminal_bootstrap(self):
        class Table(nn.Module):
            def __init__(self, scale):
                super().__init__(); self.bias = nn.Parameter(torch.zeros(())); self.scale = scale
            def forward(self, c, x, exists, mask):
                return self.scale*x[:, :, 0]+self.bias
        agent = SetD3QNAgent(23, 10, batch_size=2, hidden_dims=(8,), entity_dims=(8,), device="cpu")
        agent.online, agent.target = Table(1.), Table(10.)
        agent.optim = torch.optim.Adam(agent.online.parameters())
        current, nxt = observation(1), observation(2)
        nxt["candidates"][:, 0] = [1000., 2.]
        agent.remember(current, np.ones(1), 0, -1., nxt, np.array([0., 1.]), False)
        agent.remember(current, np.ones(1), 0, -3., observation(0), np.zeros(0), True)
        captured = []
        original = agent.f11.target
        def capture(future, target):
            captured.extend(target.tolist()); original(future, target)
        with patch.object(agent.f11, "target", side_effect=capture):
            self.assertTrue(np.isfinite(agent.update()))
        np.testing.assert_allclose(sorted(captured), [-3., 18.])
        self.assertEqual(agent._updates, 1)

    def test_previews_are_readonly_and_public_observations_have_fixed_features(self):
        env = self.env()
        def physical():
            return (env.current_time, tuple(env.task_state), tuple(env.vm_available_at),
                    tuple(env.ready_task_ids), repr(env.event_heap), repr(env._records),
                    repr(env.assignment_ledger.segments), env.assignment_energy_total,
                    repr(env.rng.get_state()), repr(np.random.get_state()))
        before = physical()
        tid = env.ready_task_ids[0]
        env._build_host_obs_for_task(tid)
        for h in env.host_ids:
            env._build_vm_obs_for_task_host(tid, h)
        self.assertEqual(before, physical())
        reset, _ = env.reset()
        self.assertTrue(env.observation_space.contains(reset))
        host, ok = env.get_host_state_for_next_assignment()
        self.assertTrue(ok)
        self.assertEqual(host["obs"]["context"].shape, (23,))
        self.assertEqual(host["obs"]["candidates"].shape, (3, 10))
        env.host_select(int(np.flatnonzero(host["mask"])[0]))
        vm, ok = env.get_vm_state_for_current_task()
        self.assertTrue(ok)
        self.assertEqual(vm["obs"]["context"].shape, (30,))
        self.assertEqual(vm["obs"]["candidates"].shape[1], 7)
        np.testing.assert_array_equal(vm["obs"]["candidates"][vm["mask"] <= .5, 5:], 0.)

    def test_three_rewards_and_physics_equal_under_same_actions(self):
        base, env = self.env(FixedPF), self.env()
        count = 0
        with redirect_stdout(io.StringIO()):
            while not base.done_flag:
                while True:
                    old, old_ok = base.get_host_state_for_next_assignment()
                    new, new_ok = env.get_host_state_for_next_assignment()
                    self.assertEqual(old_ok, new_ok)
                    if not old_ok:
                        break
                    self.assertTrue(np.all(new["mask"] >= old["mask"]))
                    ah = int(np.flatnonzero(old["mask"])[0])
                    base.host_select(ah); env.host_select(ah)
                    ov, _ = base.get_vm_state_for_current_task()
                    nv, _ = env.get_vm_state_for_current_task()
                    self.assertTrue(np.all(nv["mask"] >= ov["mask"][:len(nv["mask"])]))
                    av = int(np.flatnonzero(ov["mask"])[0])
                    a, b = base.vm_assign(av), env.vm_assign(av)
                    np.testing.assert_allclose(a[:2], b[:2], rtol=0, atol=1e-10)
                    self.assertEqual(a[2], b[2])
                    count += 1
                ta, tb = base.finish_phase_and_advance(), env.finish_phase_and_advance()
                self.assertAlmostEqual(ta[0], tb[0], places=10)
                self.assertEqual(ta[1], tb[1])
                if not base.done_flag:
                    base.apply_manager_delta(np.zeros(5)); env.apply_manager_delta(np.zeros(5))
        self.assertGreater(count, 20)
        self.assertAlmostEqual(base.total_energy, env.total_energy, places=8)
        self.assertAlmostEqual(env.assignment_energy_total, env.total_energy, places=7)

    def test_checkpoint_roles_architecture_source_and_group_isolation(self):
        scene = self.scene()
        agents = self.tiny_agents(scene)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            runtime.save_agents(agents, directory, "best")
            for role, a in self.tiny_agents(scene).items():
                a.load(str(Path(directory)/f"best_{runtime.FILES[role]}.pth"), require_learning_model=True)
                self.assertEqual(a.loaded_group, agents[role].checkpoint_group)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                agents["HA"].load(str(Path(directory)/"best_vm.pth"))
            wrong_source = self.tiny_agents(self.scene(resource="M"))["HA"]
            with self.assertRaisesRegex(ValueError, "source_train_scene"):
                wrong_source.load(str(Path(directory)/"best_host.pth"))
            checkpoint = torch.load(Path(directory)/"best_host.pth", weights_only=True)
            checkpoint["normalization"]["algorithm_variant"]["apf"]["host_base"] = 9.
            torch.save(checkpoint, Path(directory)/"foreign.pth")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                agents["HA"].load(str(Path(directory)/"foreign.pth"))
            agents["TA"].checkpoint_group = "different_training_snapshot"
            agents["TA"].save(str(Path(directory)/"best_manager.pth"))
            target = replace(scene, mode="eval", train_resource="S", checkpoint_override=directory)
            with patch.object(runtime, "build_agents", side_effect=self.tiny_agents):
                with self.assertRaisesRegex(ValueError, "different saved groups"):
                    runtime.load_agents(target)

    def test_real_train_save_load_eval_and_cross_count_transfer(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            protocol = Path(directory)/"protocol.json"
            protocol.write_text(json.dumps(dict(version="mahdrl_as_smoke_split_v1", train=[0,1],
                validation=[600], test=[700], episodes=2, selection="minimum_validation_total_energy")), encoding="utf-8")
            scales = Path(directory)/"scales.json"
            config = fixture_config(scales)
            config["generation"] = dict(**protocol_binding(protocol),
                calibration_seeds={key: [0] for key in config["scenes"]})
            scales.write_text(json.dumps(config), encoding="utf-8")
            scene = self.scene(output_root=directory, legacy_split=False,
                               experiment_protocol=str(protocol), normalization_config=str(scales))
            original_env_kwargs = runtime.build_env_kwargs
            def small(scene, seed):
                return dict(original_env_kwargs(scene, seed), workflows_per_episode=2, max_ready_tasks=100)
            with patch.object(runtime, "build_agents", side_effect=self.tiny_agents), patch.object(runtime, "build_env_kwargs", side_effect=small):
                with redirect_stdout(io.StringIO()):
                    agents = runtime.train(scene)
                    self.assertTrue(all(a._updates > 0 for a in agents.values()))
                    target = replace(scene, mode="eval", seed_spec="700", train_resource="S", train_deadline="Tight")
                    frozen = runtime.load_agents(target)
                    initial = parameter_hash(frozen)
                    with patch.object(runtime, "load_agents", return_value=frozen):
                        runtime.evaluate(target)
                    self.assertEqual(parameter_hash(frozen), initial)
                    with self.assertRaises(FileExistsError):
                        runtime.evaluate(target)
                    with self.assertRaises(FileExistsError):
                        runtime.train(scene)
                    # One saved source model, two unseen host/VM counts, matching reference caches.
                    for counts in ([12]*6, [4]*12):
                        spec = Path(directory)/f"topology{len(counts)}.json"
                        spec.write_text(json.dumps(dict(vms_per_host=counts)), encoding="utf-8")
                        cache = Path(directory)/f"cache{len(counts)}.json"
                        precompute(ScoreScenario(mode="precompute", seed_spec="0,1,600,700", resource_config=str(spec), cache_output=str(cache)))
                        transferred = replace(target, resource_config=str(spec), cache_file=str(cache))
                        runtime.validate_inputs(transferred)
                        loaded = runtime.load_agents(transferred)
                        self.assertEqual(parameter_hash(loaded), initial)
                        e = runtime.build_env(transferred, 700); e.reset()
                        result = runtime.run_episode(e, loaded)
                        self.assertEqual(result["wf_completed_count"], 2)
                        self.assertEqual(parameter_hash(loaded), initial)
                        self.assertTrue(all(a._updates == frozen[r]._updates for r,a in loaded.items()))
            rows = [json.loads(x) for x in (scene.log_dir()/"training_measurements.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(all(v > 0 for v in rows[0]["optimizer_steps"].values()))
            self.assertTrue((target.evaluation_dir()/"measurements/episodes/seed700.json").is_file())

    def test_custom_topology_validation_and_model_seed_paths(self):
        preview = ScoreScenario(mode="eval", resource="L", train_resource="M")
        self.assertLess(len(str(preview.evaluation_dir()/"measurements/episodes/seed700.json")), 260)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            spec = Path(directory)/"resource.json"
            spec.write_text('{"vms_per_host": [true, 4]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ScoreScenario(mode="eval", resource_config=str(spec), train_resource="S")
            spec.write_text('{"vms_per_host": [4, 4, 4, 4]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-resource"):
                ScoreScenario(mode="eval", resource_config=str(spec))
            target = ScoreScenario(mode="eval", resource_config=str(spec), train_resource="S", normalization_mode="legacy")
            self.assertEqual((target.num_hosts, target.vms_per_host), (4, (4,4,4,4)))
            self.assertNotEqual(target.resource, "S")
            self.assertNotEqual(self.scene(model_seed=1).checkpoint_dir(), self.scene(model_seed=2).checkpoint_dir())

    def test_dry_runs_need_no_cache_or_calibration_and_no_registry_mutation(self):
        from common.scenarios import MODEL_NAMES
        before = dict(MODEL_NAMES)
        for mode in ("train", "eval", "precompute"):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as exit:
                parse_scenario(mode, ["--dry-run"])
            self.assertEqual(exit.exception.code, 0)
            self.assertEqual(json.loads(out.getvalue())["algorithm"], "mahdrl_oe_ge_apf_score")
        self.assertEqual(before, MODEL_NAMES)


if __name__ == "__main__":
    unittest.main()
