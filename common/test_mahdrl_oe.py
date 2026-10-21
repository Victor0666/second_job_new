"""Observation/assignment-energy variant: physics isolation and reward accounting."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common.f11 import PendingDecision
from common.f18_state_normalization import VERSION
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from common.workflow_opt import LoadRecord, energy_from_records
from algorithms.mahdrl_oe.reward import finish_lower_phase
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as Baseline
from environments.mahdrl_oe import AssignmentEnergyLedger, CloudWorkflowEnv_VMAgents as Variant


class MAHDRLOETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="mahdrl_oe_tests_")
        cls.root = Path(cls.temp.name)
        cls.scales = cls.root / "scales.json"
        fixture_config(cls.scales)
        cls.cache = cls.root / "cache.json"
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def kwargs(self):
        return dict(dax_paths=Scenario().dax_paths(), random_seed=0, workflows_per_episode=2,
                    horizon=1e9, arrival_lambda=.03, num_hosts=3, vms_per_host=(9, 8, 8),
                    max_ready_tasks=100, pc_tiers=(1., 2., 4., 6., 8.),
                    bw_tiers=(1000., 2000., 4000., 6000., 8000.), deadline_mode="cache_fcfs",
                    deadline_cache_path=str(self.cache), deadline_alpha_small_prob=.8,
                    normalization_mode=VERSION, normalization_config=str(self.scales),
                    source_train_scene="SS_T")

    def environment(self, cls=Variant):
        env = cls(**self.kwargs())
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    @staticmethod
    def snapshot(env):
        return (env.current_time, tuple(env.task_state), tuple(env.task_end_time),
                tuple(env.vm_available_at), tuple(env.ready_task_ids), tuple(env.event_heap),
                tuple(vars(r).items() for r in env._records), env.total_energy,
                env.assignment_energy_total, env.lower_energy_emitted,
                repr(env.parents_loc_by_wf), repr(env.rng.get_state()))

    def test_deadline_feature_and_host_power_information(self):
        env, base = self.environment(), self.environment(Baseline)
        tid = env.ready_task_ids[0]
        host = env.host_ids[1]
        hobs, hmask = env._build_host_obs_for_task(tid)
        vobs, vmask = env._build_vm_obs_for_task_host(tid, host)
        self.assertEqual(len(hobs), base.host_obs_dim + 1 + 5 * env.num_hosts)
        self.assertEqual(len(vobs), base.vm_obs_dim + 1 + env.num_hosts + 2 + 2 * env.max_vms_per_host)
        np.testing.assert_array_equal(hmask, base._build_host_obs_for_task(tid)[1])
        np.testing.assert_array_equal(vmask, base._build_vm_obs_for_task_host(tid, host)[1])
        tref = env._obs_ref("T_ref", env.horizon)
        remaining_index_h, remaining_index_v = base.host_obs_dim, base.vm_obs_dim
        self.assertAlmostEqual(hobs[remaining_index_h], (env.task_baseline_finish[tid] - env.current_time) / tref, places=5)
        self.assertAlmostEqual(vobs[remaining_index_v], hobs[remaining_index_h], places=5)
        # Changing only the task sub-deadline must now change HA and VA inputs.
        env.task_baseline_finish[tid] = env.current_time - 20.
        changed_h = env._build_host_obs_for_task(tid)[0]
        changed_v = env._build_vm_obs_for_task_host(tid, host)[0]
        self.assertLess(changed_h[remaining_index_h], 0.)
        self.assertLess(changed_v[remaining_index_v], 0.)
        self.assertFalse(np.array_equal(hobs, changed_h))
        self.assertFalse(np.array_equal(vobs, changed_v))
        # A shared VA can distinguish equal-capacity hosts with different power curves.
        start = base.vm_obs_dim + 1
        vectors = []
        for hi, hid in enumerate(env.host_ids):
            obs, _ = env._build_vm_obs_for_task_host(tid, hid)
            np.testing.assert_array_equal(obs[start:start + env.num_hosts], np.eye(env.num_hosts)[hi])
            np.testing.assert_allclose(obs[start + env.num_hosts:start + env.num_hosts + 2],
                                       [env.hosts[hid].power(0.) / 100.] * 2)
            vectors.append(obs)
        self.assertFalse(np.array_equal(vectors[1], vectors[2]))

    def test_candidate_summaries_masks_and_queries_are_readonly(self):
        env, base = self.environment(), self.environment(Baseline)
        tid = env.ready_task_ids[0]
        env.task_baseline_finish[tid] = env.current_time + 1.
        tref, eref = env._obs_ref("T_ref", env.horizon), 100. * env._obs_ref("T_ref", env.horizon)
        before = self.snapshot(env)
        obs, mask = env._build_host_obs_for_task(tid)
        summaries = obs[base.host_obs_dim + 1:].reshape(env.num_hosts, 5)
        for hi, hid in enumerate(env.host_ids):
            candidates = []
            for j in env.host_to_vm_indices[hid]:
                if env.vm_available_at[j] <= env.current_time + 1e-12:
                    duration = env._task_duration(tid, j)
                    energy = env.estimate_assignment_energy(tid, j)
                    candidates.append((duration, energy, max(0., env.current_time + duration - env.task_baseline_finish[tid])))
            fastest = min(candidates, key=lambda item: item[0])
            cheapest = min(candidates, key=lambda item: item[1])
            expected = [fastest[0] / tref, fastest[1] / eref, min(c[2] for c in candidates) / tref,
                        cheapest[1] / eref, cheapest[0] / tref]
            np.testing.assert_allclose(summaries[hi], expected, rtol=1e-5, atol=1e-7)
            va, _ = env._build_vm_obs_for_task_host(tid, hid)
            tail = va[base.vm_obs_dim + 1 + env.num_hosts + 2:].reshape(-1, 2)
            for slot, j in enumerate(env.host_to_vm_indices[hid]):
                vm = env.vms[env.vm_ids[j]]
                dp = env.hosts[hid].power(min(1., vm.pc / env.hosts[hid].total_pc)) - env.hosts[hid].power(0.)
                tardiness = max(0., env.current_time + env._task_duration(tid, j) - env.task_baseline_finish[tid])
                np.testing.assert_allclose(tail[slot], [tardiness / tref, dp / 100.], rtol=1e-5)
        self.assertEqual(before, self.snapshot(env))
        # Even an already-late task keeps the original physical action mask.
        np.testing.assert_array_equal(mask, base._build_host_obs_for_task(tid)[1])
        for j in env.host_to_vm_indices[env.host_ids[0]]:
            selected = env.ready_task_ids[0]
            env._assign_task_to_specific_vm(selected, j)
            base._assign_task_to_specific_vm(selected, j)
        obs, mask = env._build_host_obs_for_task(tid)
        np.testing.assert_array_equal(mask, base._build_host_obs_for_task(tid)[1])
        self.assertEqual(mask[0], 0.)
        np.testing.assert_array_equal(obs[base.host_obs_dim + 1:base.host_obs_dim + 6], np.zeros(5))

    def test_exact_assignment_increment_includes_overlap_and_internal_gap(self):
        env = self.environment()
        vm = env.vms[env.vm_ids[0]]
        hid, host = vm.host_id, env.hosts[vm.host_id]
        other = env.vms[env.vm_ids[env.host_to_vm_indices[hid][1]]]
        for now, records, expected in (
            (20., [LoadRecord(0., 2., hid, vm.pc)], host.power(0.) * 18. + host.power(vm.pc / host.total_pc) * 2.),
            (0., [LoadRecord(0., 10., hid, other.pc)],
             (host.power((other.pc + vm.pc) / host.total_pc) - host.power(other.pc / host.total_pc)) * 2.),
        ):
            ledger = AssignmentEnergyLedger(env.hosts)
            for record in records:
                ledger.add(record.server_id, record.start_time, record.end_time, record.vm_pc)
            before = (ledger.total_energy, repr(ledger.segments))
            increment = ledger.preview(hid, now, now + 2., vm.pc)
            self.assertAlmostEqual(increment, expected, places=7)
            prospective = records + [LoadRecord(now, now + 2., hid, vm.pc)]
            self.assertAlmostEqual(increment, energy_from_records(prospective, env.hosts) - energy_from_records(records, env.hosts), places=7)
            self.assertEqual(before, (ledger.total_energy, repr(ledger.segments)))
            self.assertAlmostEqual(ledger.add(hid, now, now + 2., vm.pc), increment, places=7)
            self.assertAlmostEqual(ledger.total_energy, energy_from_records(prospective, env.hosts), places=7)

    def test_same_actions_same_physics_ta_and_conserved_lower_energy(self):
        original, env = self.environment(Baseline), self.environment()
        traces = [PendingDecision(lambda *args: None) for _ in range(2)]
        energy_rewards, assignments, phases = 0., 0, 0
        with redirect_stdout(io.StringIO()):
            for _ in range(10000):
                while True:
                    states = [e.get_host_state_for_next_assignment() for e in (original, env)]
                    self.assertEqual(states[0][1], states[1][1])
                    if not states[0][1]:
                        break
                    np.testing.assert_array_equal(states[0][0]["mask"], states[1][0]["mask"])
                    action_h = int(np.flatnonzero(states[0][0]["mask"])[assignments % int(sum(states[0][0]["mask"]))])
                    for e in (original, env):
                        e.host_select(action_h)
                    vm_states = [e.get_vm_state_for_current_task()[0] for e in (original, env)]
                    np.testing.assert_array_equal(vm_states[0]["mask"], vm_states[1]["mask"])
                    action_v = int(np.flatnonzero(vm_states[0]["mask"])[0])
                    for trace, state in zip(traces, (states[1][0], vm_states[1])):
                        trace.advance(state["obs"], state["mask"])
                    old = original.vm_assign(action_v)
                    rh, rv, info = env.vm_assign(action_v)
                    self.assertEqual(old[2]["r_delay"], info["r_delay"])
                    self.assertAlmostEqual(rh, (1-env.alpha_delay_host)*info["r_energy_total"] + env.alpha_delay_host*info["r_delay"])
                    self.assertAlmostEqual(rv, (1-env.alpha_delay_vm)*info["r_energy_host"] + env.alpha_delay_vm*info["r_delay"])
                    self.assertAlmostEqual(info["r_energy_total"], info["r_energy_host"])
                    energy_rewards += info["r_energy_total"]
                    for trace, state, action, reward in zip(traces, (states[1][0], vm_states[1]), (action_h, action_v), (rh, rv)):
                        trace.record(state["obs"], state["mask"], action, reward, host_index=info["host_id"])
                    assignments += 1
                old_reward, old_info = original.finish_phase_and_advance()
                reward, info = env.finish_phase_and_advance()
                self.assertEqual(reward, old_reward)
                self.assertEqual(info["phase_energy"], old_info["phase_energy"])
                self.assertTrue(info["lower_energy_already_emitted"])
                self.assertAlmostEqual(info["lower_assignment_energy_unemitted"], 0.)
                before = tuple(t.recorded_reward for t in traces)
                finish_lower_phase(env, *traces, info)
                self.assertEqual(before, tuple(t.recorded_reward for t in traces))
                self.assertEqual(original.current_time, env.current_time)
                np.testing.assert_array_equal(original.task_end_time, env.task_end_time)
                self.assertEqual(original._records, env._records)
                self.assertEqual(original.done_flag, env.done_flag)
                phases += 1
                if env.done_flag:
                    break
                for e in (original, env):
                    e.apply_manager_delta(np.zeros(5))
            else:
                self.fail("Episode did not terminate")
        self.assertGreater(assignments, 0)
        self.assertGreater(phases, 1)
        self.assertEqual(original.wf_timeout, env.wf_timeout)
        self.assertEqual(original.total_energy, env.total_energy)
        self.assertAlmostEqual(env.assignment_energy_total, env.total_energy, places=6)
        self.assertAlmostEqual(env.lower_energy_emitted, env.total_energy, places=6)
        self.assertAlmostEqual(energy_rewards, -env.energy_reward_scale * env.total_energy, places=7)
        self.assertTrue(all(t.pending is None for t in traces))

    def test_forced_assignment_energy_is_emitted_once(self):
        env = self.environment()
        env._fuse_zero_assign_limit = 1
        traces = [PendingDecision(lambda *args: None) for _ in range(2)]
        for trace in traces:
            trace.record(np.ones(2), np.ones(2), 0, 0., host_index=0)
        with redirect_stdout(io.StringIO()):
            _, info = env.finish_phase_and_advance()
            self.assertGreater(info["lower_assignment_energy_unemitted"], 0.)
            self.assertTrue(info["lower_energy_already_emitted"])
            delta = info["lower_assignment_energy_unemitted"]
            finish_lower_phase(env, *traces, info)
            self.assertAlmostEqual(traces[0].recorded_reward, -(1-env.alpha_delay_host)*delta*env.energy_reward_scale)
            self.assertAlmostEqual(traces[1].recorded_reward, -(1-env.alpha_delay_vm)*delta*env.energy_reward_scale)
            _, following = env.finish_phase_and_advance()
        # The next fallback may place another task, but cannot re-emit the first.
        self.assertAlmostEqual(env.lower_energy_emitted, env.assignment_energy_total, places=7)
        self.assertGreaterEqual(following["lower_assignment_energy_unemitted"], 0.)

    def test_variant_checkpoint_metadata_rejects_original_and_wrong_role(self):
        from algorithms.mahdrl_oe.agent import D3QNAgent, bind_variant
        from common.d3qn_agent_EXDehid import D3QNAgent as OriginalAgent
        from common.f18_state_normalization import bind_normalization
        env = self.environment()
        scenario = Scenario(mode="train", normalization_config=str(self.scales))
        metadata = env.variant_metadata()
        self.assertEqual(metadata["changes"], [1, 2, 3, 5])
        self.assertFalse(metadata["deadline_action_filter"])
        self.assertEqual(metadata["time_ref_s"], 123.)
        self.assertEqual(metadata["power_ref_w"], 100.)
        self.assertEqual(metadata["energy_ref_j"], 12300.)
        agent = D3QNAgent(3, 2, hidden_dims=(4,), device=torch.device("cpu"))
        bind_normalization(agent, scenario)
        bind_variant(agent, metadata, "TA")
        new_path, old_path = self.root / "new.pth", self.root / "old.pth"
        agent.save(str(new_path))
        agent.load(str(new_path))
        original = OriginalAgent(3, 2, hidden_dims=(4,), device=torch.device("cpu"))
        bind_normalization(original, scenario)
        original.save(str(old_path))
        with self.assertRaisesRegex(ValueError, "variant/role mismatch"):
            agent.load(str(old_path))
        bind_variant(agent, metadata, "HA")
        with self.assertRaisesRegex(ValueError, "variant/role mismatch"):
            agent.load(str(new_path))

    def test_isolated_training_save_load_and_evaluation(self):
        from algorithms.mahdrl_oe import train as trainer, evaluate as evaluator
        from algorithms.mahdrl_oe.agent import D3QNAgent
        root, cache = self.root / "smoke", self.cache
        class LocalScenario(Scenario):
            def dax_paths(self, directory=None):
                return Scenario.dax_paths(self)
            @property
            def cache_path(self):
                return cache
            def checkpoint_dir(self, algorithm):
                return root / "ckpt" / algorithm
            def log_dir(self, algorithm):
                return root / "logs" / algorithm
        class TinyEpisode(Variant):
            def __init__(self, **kwargs):
                kwargs["workflows_per_episode"] = 2
                super().__init__(**kwargs)
        def small_agent(*args, **kwargs):
            kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
            return D3QNAgent(*args, **kwargs)
        scenario = LocalScenario(mode="train", episodes=1, normalization_config=str(self.scales))
        with (patch.object(trainer, "D3QNAgent", side_effect=small_agent),
              patch.object(trainer, "SCENARIO", scenario),
              patch.object(trainer, "CloudWorkflowEnv_VMAgents", TinyEpisode),
              patch.object(trainer, "ROOT_DIR", str(root)), redirect_stdout(io.StringIO())):
            trainer.train()
        directory = scenario.checkpoint_dir("mahdrl_oe")
        self.assertNotEqual(directory, scenario.checkpoint_dir("mahdrl"))
        self.assertFalse(scenario.checkpoint_dir("mahdrl").exists())
        for role in ("vm", "host", "manager"):
            ckpt = torch.load(directory / ("best_" + role + ".pth"), map_location="cpu", weights_only=True)
            self.assertGreater(ckpt["updates"], 0)
            self.assertEqual(ckpt["normalization"]["algorithm_variant"]["changes"], [1, 2, 3, 5])
        self.assertTrue((directory / "algorithm_variant.json").exists())
        diagnostics = list(root.rglob("f11_diagnostics.jsonl"))
        self.assertTrue(diagnostics)
        records = [json.loads(line) for line in diagnostics[0].read_text().splitlines()]
        for key in ("physical_energy_balance", "lower_energy_balance", "HA_reward_formula_balance", "VA_reward_formula_balance"):
            self.assertAlmostEqual(records[-1]["reward_audit"][key], 0., places=7)
        with (patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
              patch.object(evaluator, "EnvHRLMix", TinyEpisode),
              patch.object(evaluator, "SCENARIO", replace(scenario, mode="eval", seed_spec="700")),
              redirect_stdout(io.StringIO())):
            kwargs = evaluator.build_env_kwargs_hrl_mix(random_seed=700)
            va, ha, ta = evaluator.build_three_layer_agents_and_load(kwargs)
            metrics = evaluator.evaluate_hrl_mix_one_seed(TinyEpisode, kwargs, va, ha, ta)
        self.assertGreater(metrics["eval_energy"], 0.)
        self.assertTrue(np.isfinite(metrics["eval_energy"]))
        for mode in ("train", "eval"):
            with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result:
                    parse_scenario(mode, "mahdrl_oe")
            self.assertEqual(result.exception.code, 0)
            self.assertIn("mahdrl_oe", json.loads(out.getvalue())["checkpoint_dir"])


if __name__ == "__main__":
    unittest.main()
