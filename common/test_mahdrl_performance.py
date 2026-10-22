"""Performance variants: decisions, unchanged physics, real updates and isolated outputs."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.mahdrl import train as baseline
from common.d3qn_agent_EXDehid import D3QNAgent
from common.experiment_measurements import EvaluationRecorder
from common.f18_state_normalization import VERSION
from common.mahdrl_performance import (
    FILTER_POLICY, GUIDED_POLICY, GuidedExplorationAgent, PerformanceFilterAgent,
    candidate_durations, performance_mask,
)
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as OriginalEnv
from environments.mahdrl_pf import CloudWorkflowEnv_VMAgents as FilterEnv


class PerformanceDecisionTests(unittest.TestCase):
    def agent(self, cls=GuidedExplorationAgent):
        return cls(input_dim=2, output_dim=4, hidden_dims=(8,), batch_size=2,
                   eps_start=1., eps_end=1., device=torch.device("cpu"))

    def test_guidance_only_inside_exploration_and_legal_ties(self):
        agent = self.agent()
        state, mask = np.zeros(2), np.array([1., 1., 1., 0.])
        times = np.array([10., 1., 1., 0.])
        with patch("common.mahdrl_performance.random.random", side_effect=[.1, .1]):
            self.assertIn(agent.select_action(state, mask, performance_times=times), (1, 2))
        # The other half of exploration still includes the slow legal candidate.
        with (patch("common.mahdrl_performance.random.random", side_effect=[.1, .9]),
              patch("common.mahdrl_performance.np.random.choice", return_value=0) as choose):
            self.assertEqual(agent.select_action(state, mask, performance_times=times), 0)
            np.testing.assert_array_equal(choose.call_args.args[0], [0, 1, 2])
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.adv[-1].bias[0] = 20.
        # Both greedy paths can still select a slow VM; PF was not mixed into GE.
        self.assertEqual(agent.select_action(state, mask, deterministic=True,
                                            performance_times=times), 0)
        agent._eps_start = agent._eps_end = 0.
        self.assertEqual(agent.select_action(state, mask, performance_times=times), 0)
        self.assertEqual(agent._eps_steps, 0)
        self.assertEqual(agent._action_calls, 4)
        calls = agent._action_calls
        agent.select_action(state, mask, deterministic=True, count_step=False)
        self.assertEqual(agent._action_calls, calls)

    def test_deterministic_guided_policy_matches_original(self):
        guided, original = self.agent(), self.agent(D3QNAgent)
        original.online.load_state_dict(guided.online.state_dict())
        for mask in (np.array([1, 0, 1, 0]), np.array([0, 1, 1, 1])):
            state = np.array([.5, -.25])
            self.assertEqual(guided.select_action(state, mask, True, False),
                             original.select_action(state, mask, True, False))

    def test_filter_boundaries_padding_and_no_legal_action(self):
        mask = np.array([1., 1., 0., 1., 1.], dtype=np.float32)
        times = np.array([4., 5., 0., 5.01, 4.])
        np.testing.assert_array_equal(performance_mask(mask, times), [1, 1, 0, 0, 1])
        np.testing.assert_array_equal(mask, [1, 1, 0, 1, 1])
        np.testing.assert_array_equal(performance_mask([1, 1], [0, 0]), [1, 1])
        np.testing.assert_array_equal(performance_mask([0, 1], [np.inf, 20]), [0, 1])
        np.testing.assert_array_equal(performance_mask([0, 0], [np.inf, np.inf]), [0, 0])
        for ratio in (.9, np.inf, np.nan):
            with self.assertRaises(ValueError):
                performance_mask(mask, times, ratio)
        for bad in ([1, 2], [4, np.nan, 0, 5, 4], [4, -1, 0, 5, 4]):
            with self.assertRaises(ValueError):
                performance_mask(mask, bad)

    def test_host_and_vm_mapping_use_current_task_duration(self):
        calls = []

        def duration(tid, j):
            calls.append((tid, j))
            return {0: 20., 1: 3., 2: 100., 3: 1.}[j]

        env = SimpleNamespace(_cur_tid=7, _cur_host_id=42, current_time=10.,
                              host_ids=[42, 99], host_to_vm_indices={42: [2, 0], 99: [3, 1]},
                              vm_available_at=[0., 0., 11., 0.], _task_duration=duration)
        np.testing.assert_array_equal(candidate_durations(env, "host", [1, 1]), [20, 1])
        np.testing.assert_array_equal(performance_mask([1, 1], [20, 1]), [0, 1])
        np.testing.assert_array_equal(candidate_durations(env, "vm", [0, 1, 0]), [np.inf, 20, np.inf])
        self.assertTrue(all(tid == 7 for tid, _ in calls))
        self.assertNotIn((7, 2), calls)  # busy VM and padded slots never become candidates

    def test_variant_checkpoints_reject_original_and_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = {}
            for cls in (D3QNAgent, GuidedExplorationAgent, PerformanceFilterAgent):
                agent = self.agent(cls)
                filename = Path(temporary) / (cls.__name__ + ".pth")
                agent.save(filename)
                files[cls] = filename
            for cls in (GuidedExplorationAgent, PerformanceFilterAgent):
                restored = self.agent(cls)
                restored.load(files[cls])
                for other, filename in files.items():
                    if other is not cls:
                        with self.assertRaisesRegex(ValueError, "performance-policy mismatch"):
                            restored.load(filename)


class PerformanceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        switches = patch.dict("os.environ", F11_DIAGNOSTICS="1", F19_DIAGNOSTICS="1", EXPERIMENT_METRICS="1")
        switches.start()
        cls.addClassCleanup(switches.stop)
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="mahdrl_performance_")
        cls.root = Path(cls.temporary.name)
        cls.scales, cls.cache = cls.root / "scales.json", cls.root / "cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def kwargs(self):
        return dict(dax_paths=Scenario().dax_paths(), random_seed=0, workflows_per_episode=2,
                    horizon=1e9, arrival_lambda=.03, num_hosts=3, vms_per_host=(9, 8, 8),
                    max_ready_tasks=100, pc_tiers=(1., 2., 4., 6., 8.),
                    bw_tiers=(1000., 2000., 4000., 6000., 8000.), deadline_mode="cache_fcfs",
                    deadline_cache_path=str(self.cache), deadline_alpha_small_prob=.8,
                    normalization_mode=VERSION, normalization_config=str(self.scales),
                    source_train_scene="SS_T")

    @staticmethod
    def physical_state(env):
        return (env.current_time, tuple(env.task_state), tuple(env.task_end_time),
                tuple(env.vm_available_at), tuple(env.ready_task_ids), tuple(env.event_heap),
                repr([vars(record) for record in env._records]), env.total_energy,
                repr(env.parents_loc_by_wf), repr(env.rng.get_state()), tuple(env.combo_weights))

    def test_filter_preserves_physics_and_rewards_on_same_actions(self):
        original, filtered = OriginalEnv(**self.kwargs()), FilterEnv(**self.kwargs())
        with redirect_stdout(io.StringIO()):
            original.reset()
            filtered.reset()
        for env in (original, filtered):
            env.apply_manager_delta(np.zeros(5))
        exclusions = phases = 0
        while not filtered.done_flag:
            while True:
                old, old_ok = original.get_host_state_for_next_assignment()
                new, new_ok = filtered.get_host_state_for_next_assignment()
                self.assertEqual(old_ok, new_ok)
                if not new_ok:
                    break
                np.testing.assert_array_equal(old["obs"], new["obs"])
                expected = performance_mask(old["mask"], candidate_durations(original, "host", old["mask"]))
                np.testing.assert_array_equal(new["mask"], expected)
                rejected = np.flatnonzero((old["mask"] > .5) & (new["mask"] <= .5))
                if rejected.size:
                    with self.assertRaisesRegex(ValueError, "Host excluded"):
                        filtered.host_select(rejected[0])
                host = int(np.flatnonzero(new["mask"] > .5)[0])
                original.host_select(host)
                filtered.host_select(host)
                old_vm, _ = original.get_vm_state_for_current_task()
                new_vm, _ = filtered.get_vm_state_for_current_task()
                np.testing.assert_array_equal(old_vm["obs"], new_vm["obs"])
                expected_vm = performance_mask(old_vm["mask"], candidate_durations(original, "vm", old_vm["mask"]))
                np.testing.assert_array_equal(new_vm["mask"], expected_vm)
                rejected = np.flatnonzero((old_vm["mask"] > .5) & (new_vm["mask"] <= .5))
                exclusions += len(rejected)
                if rejected.size:
                    with self.assertRaisesRegex(ValueError, "VM excluded"):
                        filtered.vm_assign(rejected[0])
                vm = int(np.flatnonzero(new_vm["mask"] > .5)[0])
                self.assertEqual(original.vm_assign(vm), filtered.vm_assign(vm))
                self.assertEqual(self.physical_state(original), self.physical_state(filtered))
            reward, _ = original.finish_phase_and_advance()
            filtered_reward, _ = filtered.finish_phase_and_advance()
            self.assertEqual(reward, filtered_reward)
            self.assertEqual(self.physical_state(original), self.physical_state(filtered))
            phases += 1
            self.assertLess(phases, 1000)
        self.assertGreater(exclusions, 0)
        self.assertEqual(filtered.completed_workflows, 2)

    def test_cli_namespaces(self):
        for algorithm in ("mahdrl_ge", "mahdrl_pf"):
            for mode in ("train", "eval"):
                with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(SystemExit) as result:
                        parse_scenario(mode, algorithm)
                self.assertEqual(result.exception.code, 0)
                self.assertIn(algorithm, json.loads(out.getvalue())["checkpoint_dir"])
        for resource in "SML":
            scene = Scenario(resource=resource)
            paths = [scene.checkpoint_dir(name) for name in ("mahdrl", "mahdrl_ge", "mahdrl_pf")]
            self.assertEqual(len(set(paths)), 3)

    def test_real_training_roundtrip_evaluation_and_original_outputs(self):
        root, cache, scales = self.root, self.cache, self.scales

        class LocalScenario(Scenario):
            def dax_paths(self, directory=None):
                return Scenario.dax_paths(self)

            @property
            def cache_path(self):
                return cache

            def checkpoint_dir(self, algorithm):
                if self.mode == "eval" and self.checkpoint_override:
                    return Path(self.checkpoint_override)
                return root / "outputs/ckpts" / self.model_name(algorithm)

            def log_dir(self, algorithm):
                return root / "outputs/logs" / self.model_name(algorithm)

        scene = LocalScenario(mode="train", episodes=1, normalization_mode=VERSION,
                              normalization_config=str(scales))
        captures = {}
        for algorithm in ("mahdrl", "mahdrl_ge", "mahdrl_pf"):
            trainer = importlib.import_module("algorithms." + algorithm + ".train")
            env_class, agent_class = trainer.CloudWorkflowEnv_VMAgents, trainer.D3QNAgent

            class TinyEpisode(env_class):
                def __init__(self, **kwargs):
                    kwargs["workflows_per_episode"] = 2
                    super().__init__(**kwargs)

            configs, agents = [], []

            def small_agent(*args, **kwargs):
                configs.append(dict(kwargs))
                kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
                agent = agent_class(*args, **kwargs)
                agents.append(agent)
                return agent

            with (patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)),
                  patch.object(trainer, "CloudWorkflowEnv_VMAgents", TinyEpisode),
                  patch.object(trainer, "D3QNAgent", side_effect=small_agent), redirect_stdout(io.StringIO())):
                trainer.train()
            captures[algorithm] = list(configs)
            for agent in agents:
                self.assertGreater(agent._updates, 0)
                self.assertTrue(any(row[-1] for row in agent.buffer))
                for row in agent.buffer:
                    self.assertGreater(row[1][row[2]], .5)
                    if not row[-1]:
                        self.assertTrue(np.any(row[5] > .5))
            if algorithm == "mahdrl":
                protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in (scene.checkpoint_dir(algorithm), scene.log_dir(algorithm))
                             for p in folder.rglob("*") if p.is_file()}
                continue
            self.assertEqual(captures[algorithm], captures["mahdrl"])
            for filename, digest in protected.items():
                self.assertEqual(hashlib.sha256(filename.read_bytes()).hexdigest(), digest)
            policy = GUIDED_POLICY if algorithm == "mahdrl_ge" else FILTER_POLICY
            metadata = json.loads((scene.log_dir(algorithm) / "measurement_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm"], algorithm)
            for role in ("TA", "HA", "VA"):
                self.assertEqual(metadata["agents"][role]["config"], asdict(policy))
            for role in ("vm", "host", "manager"):
                for prefix in ("best_", ""):
                    name = prefix + role + ".pth" if prefix else role + "_final.pth"
                    checkpoint = torch.load(scene.checkpoint_dir(algorithm) / name, weights_only=True)
                    self.assertGreater(checkpoint["updates"], 0)
                    self.assertEqual(checkpoint["normalization"]["performance_policy"], asdict(policy))

            evaluator = importlib.import_module("algorithms." + algorithm + ".evaluate")

            def recorder(*args, **kwargs):
                return EvaluationRecorder(*args, **kwargs, directory=root / (algorithm + "_measurements"))

            evaluation_scene = replace(scene, mode="eval", seed_spec="700")
            with (patch.object(evaluator, "SCENARIO", evaluation_scene),
                  patch.object(evaluator, "PROJECT_DIR", str(root)),
                  patch.object(evaluator, "EnvHRLMix", TinyEpisode),
                  patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
                  patch.object(evaluator, "EvaluationRecorder", side_effect=recorder), redirect_stdout(io.StringIO())):
                evaluator.main()
                episode = json.loads((root / (algorithm + "_measurements/episodes/seed700.json")).read_text(encoding="utf-8"))
                self.assertEqual(episode["algorithm"], algorithm)
                self.assertEqual(episode["metrics"]["wf_completed_count"], 2)
                self.assertGreater(episode["metrics"]["total_energy"], 0)
                self.assertEqual(len(list((root / "outputs/run").rglob("eval_" + algorithm + "_*.csv"))), 1)
                old = replace(evaluation_scene, checkpoint_override=str(scene.checkpoint_dir("mahdrl")))
                with patch.object(evaluator, "SCENARIO", old):
                    with self.assertRaisesRegex(ValueError, "performance-policy mismatch"):
                        evaluator.build_three_layer_agents_and_load(evaluator.build_env_kwargs_hrl_mix(700))
            with patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
