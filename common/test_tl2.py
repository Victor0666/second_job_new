"""TL_2 reward isolation, event accounting and shared runner integration."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute
from common.f18_state_normalization import VERSION
from common.f11 import PendingDecision, LowerRewardAudit, finish_lower_phase
from common.tl_reward_protocol import (write_reward_protocol, check_reward_protocol,
                                       deferred_vm_phase_mean)
from environments.ablation_nohost import CloudWorkflowEnv_VMAgents as TL
from environments.ablation_tl2 import CloudWorkflowEnv_VMAgents as TL2


class TL2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="tl2_tests_")
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
                    horizon=1e9, arrival_lambda=.03, num_hosts=3, vms_per_host=(9,8,8),
                    max_ready_tasks=100, pc_tiers=(1.,2.,4.,6.,8.),
                    bw_tiers=(1000.,2000.,4000.,6000.,8000.), deadline_mode="cache_fcfs",
                    deadline_cache_path=str(self.cache), deadline_alpha_small_prob=.8,
                    normalization_mode=VERSION, normalization_config=str(self.scales),
                    source_train_scene="SS_T")

    def test_same_actions_same_physics_and_ta_only_va_reward_differs(self):
        envs = [cls(**self.kwargs()) for cls in (TL, TL2)]
        traces = [PendingDecision(lambda *args: None) for _ in envs]
        audits = [LowerRewardAudit(None, t) for t in traces]
        different = False
        with redirect_stdout(io.StringIO()):
            for env in envs:
                env.reset(); env.apply_manager_delta(np.zeros(5))
            for phase in range(10000):
                while True:
                    states = [env.get_vm_state_for_next_assignment() for env in envs]
                    self.assertEqual(states[0][1], states[1][1])
                    if not states[0][1]: break
                    left, right = [s[0] for s in states]
                    np.testing.assert_array_equal(left["obs"], right["obs"])
                    np.testing.assert_array_equal(left["mask"], right["mask"])
                    action = int(np.flatnonzero(left["mask"] > .5)[0])
                    for env, trace, audit, state in zip(envs, traces, audits, (left,right)):
                        trace.advance(state["obs"], state["mask"])
                        reward, info = env.vm_assign_global(action)
                        energy = info["r_energy_host"] if env is envs[0] else info["r_energy_total"]
                        self.assertAlmostEqual(reward, (1-env.alpha_delay_vm)*energy + env.alpha_delay_vm*info["r_delay"])
                        different |= abs(info["r_energy_total"]-info["r_energy_host"]) > 1e-8
                        audit.assignment(info)
                        trace.record(state["obs"], state["mask"], action, reward, host_index=info["host_id"])
                rewards = []
                for env, trace, audit in zip(envs, traces, audits):
                    reward, info = env.finish_phase_and_advance(); rewards.append(reward)
                    audit.phase(info, trace); finish_lower_phase(env, None, trace, info)
                self.assertEqual(rewards[0], rewards[1])
                self.assertEqual(envs[0].done_flag, envs[1].done_flag)
                if envs[0].done_flag: break
                for env in envs: env.apply_manager_delta(np.zeros(5))
            else: self.fail("Episode did not terminate")
        self.assertTrue(different)
        self.assertEqual(envs[0].total_energy, envs[1].total_energy)
        np.testing.assert_array_equal(envs[0].task_end_time, envs[1].task_end_time)
        for env, trace, audit in zip(envs, traces, audits):
            self.assertTrue(env._episode_should_end())
            self.assertIsNone(trace.pending)
            summary = audit.summary(env, None, trace)
            for key in ("physical_energy_balance", "lower_energy_balance", "VA_reward_formula_balance"):
                self.assertAlmostEqual(summary[key], 0., places=7)

    def test_deferred_global_tail_emitted_once_and_metric_matches(self):
        for scope, expected in (("selected_host", -1.), ("global", -2.5)):
            env = SimpleNamespace(vm_energy_scope=scope, alpha_delay_vm=.75,
                                  done_flag=False, _episode_should_end=lambda: True)
            trace = PendingDecision(lambda *args: None)
            trace.record(np.ones(2), np.ones(2), 0, 0., host_index=3)
            info = dict(lower_energy_already_emitted=True, r_energy_phase=-10., r_energy_by_host={3:-4.})
            finish_lower_phase(env, None, trace, info)
            self.assertEqual(trace.recorded_reward, 0.)
            info["lower_energy_already_emitted"] = False
            env.done_flag = True
            finish_lower_phase(env, None, trace, info)
            self.assertEqual(trace.emitted_reward, expected)
            self.assertEqual(deferred_vm_phase_mean(env, info, 2), -1.25 if scope=="global" else 0.)
            info["lower_energy_already_emitted"] = True
            self.assertEqual(deferred_vm_phase_mean(env, info, 2), 0.)

    def test_paths_generalization_and_checkpoint_scope(self):
        train = Scenario(mode="train", task="S", resource="M")
        for task in "SML":
            evaluation = replace(train, mode="eval", task=task)
            self.assertEqual(train.checkpoint_dir("ablation_tl2"), evaluation.checkpoint_dir("ablation_tl2"))
        self.assertNotEqual(train.checkpoint_dir("ablation_tl2"), train.checkpoint_dir("ablation_nohost"))
        directory = self.root / "scope"; directory.mkdir(exist_ok=True)
        check_reward_protocol(directory, "selected_host")
        with self.assertRaises(ValueError): check_reward_protocol(directory, "global")
        write_reward_protocol(directory, "selected_host")
        with self.assertRaises(ValueError): check_reward_protocol(directory, "global")
        write_reward_protocol(directory, "global"); check_reward_protocol(directory, "global")
        with self.assertRaises(ValueError): check_reward_protocol(directory, "selected_host")
        for mode in ("train", "eval"):
            with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result: parse_scenario(mode, "ablation_tl2")
            self.assertEqual(result.exception.code, 0)
            self.assertIn("tl2_", json.loads(out.getvalue())["checkpoint_dir"])

    def test_isolated_training_save_load_and_evaluation(self):
        from algorithms.ablation_nohost import train as trainer, evaluate as evaluator
        from common.d3qn_agent_EXDehid import D3QNAgent
        root, cache = self.root, self.cache
        class LocalScenario(Scenario):
            @property
            def cache_path(self): return cache
            def checkpoint_dir(self, algorithm): return root / "ckpt" / algorithm
            def log_dir(self, algorithm): return root / "logs" / algorithm
        class TinyEpisode(TL2):
            def __init__(self, **kwargs):
                kwargs["workflows_per_episode"] = 2
                super().__init__(**kwargs)
        def small_agent(*args, **kwargs):
            kwargs.update(hidden_dims=(16,8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
            return D3QNAgent(*args, **kwargs)
        scenario = LocalScenario(mode="train", episodes=1, normalization_config=str(self.scales))
        with (patch.object(trainer, "D3QNAgent", side_effect=small_agent),
              patch.object(trainer, "SCENARIO", scenario), redirect_stdout(io.StringIO())):
            trainer.train(scenario, algorithm="ablation_tl2", env_cls=TinyEpisode)
        directory = scenario.checkpoint_dir("ablation_tl2")
        self.assertTrue((directory / "best_vm.pth").exists())
        check_reward_protocol(directory, "global")
        records = [json.loads(line) for line in (scenario.log_dir("ablation_tl2") / "f11_diagnostics.jsonl").read_text().splitlines()]
        self.assertGreater(records[-1]["learners"]["VA"]["gradient_updates"], 0)
        self.assertAlmostEqual(records[-1]["reward_audit"]["VA_reward_formula_balance"], 0., places=7)
        with (patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
              patch.object(evaluator, "SCENARIO", replace(scenario, mode="eval", seed_spec="700")),
              redirect_stdout(io.StringIO())):
            kwargs = evaluator.build_env_kwargs_nohost(random_seed=700)
            va, ta = evaluator.build_two_layer_agents_and_load(kwargs, TinyEpisode, "ablation_tl2")
            metrics = evaluator.evaluate_nohost_one_seed(TinyEpisode, kwargs, va, ta)
        self.assertGreater(metrics["eval_energy"], 0.)
        self.assertGreater(metrics["task_n_finished"], 0)
        self.assertTrue(np.isfinite(metrics["eval_vm_reward"]))


if __name__ == "__main__":
    unittest.main()
