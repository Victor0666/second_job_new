"""OE/GE combinations: eligible summaries, reward conservation and isolated training."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.mahdrl_oe.agent import D3QNAgent as OEAgent, bind_variant as bind_oe
from algorithms.mahdrl_oe.reward import finish_lower_phase
from common.d3qn_agent_EXDehid import D3QNAgent as OriginalAgent
from common.experiment_measurements import EvaluationRecorder
from common.f11 import PendingDecision
from common.f18_state_normalization import VERSION
from common.mahdrl_combinations import (
    OE_GE_POLICY, OE_GE_PF_POLICY, OEGuidedAgent, OEGuidedFilterAgent, bind_variant,
)
from common.mahdrl_performance import (
    GuidedExplorationAgent, PerformanceFilterAgent, candidate_durations, performance_mask,
)
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OEEnv
from environments.mahdrl_oe_ge import CloudWorkflowEnv_VMAgents as OEGEEnv
from environments.mahdrl_oe_ge_pf import CloudWorkflowEnv_VMAgents as TripleEnv
from environments.mahdrl_pf import CloudWorkflowEnv_VMAgents as PFEnv


def physical_state(env):
    return (env.current_time, tuple(env.task_state), tuple(env.task_end_time),
            tuple(env.vm_available_at), tuple(env.ready_task_ids), tuple(env.event_heap),
            repr([vars(r) for r in env._records]), env.total_energy,
            repr(env.parents_loc_by_wf), repr(env.rng.get_state()), tuple(env.combo_weights))


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flags = patch.dict("os.environ", F11_DIAGNOSTICS="1", F19_DIAGNOSTICS="1", EXPERIMENT_METRICS="1")
        flags.start()
        cls.addClassCleanup(flags.stop)
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="mahdrl_combinations_")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.scales, cls.cache = cls.root / "scales.json", cls.root / "cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,600,700", cache_output=str(cls.cache)))

    def kwargs(self):
        return dict(dax_paths=Scenario().dax_paths(), random_seed=0, workflows_per_episode=2,
                    horizon=1e9, arrival_lambda=.03, num_hosts=3, vms_per_host=(9, 8, 8),
                    max_ready_tasks=100, pc_tiers=(1., 2., 4., 6., 8.),
                    bw_tiers=(1000., 2000., 4000., 6000., 8000.), deadline_mode="cache_fcfs",
                    deadline_cache_path=str(self.cache), deadline_alpha_small_prob=.8,
                    normalization_mode=VERSION, normalization_config=str(self.scales),
                    source_train_scene="SS_T")

    def env(self, cls):
        env = cls(**self.kwargs())
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    def test_same_actions_preserve_oe_rewards_and_original_pf_masks(self):
        oe, ge, triple, pf = [self.env(cls) for cls in (OEEnv, OEGEEnv, TripleEnv, PFEnv)]
        envs = (oe, ge, triple, pf)
        energy_rewards = phases = exclusions = 0
        with redirect_stdout(io.StringIO()):
            while not triple.done_flag:
                while True:
                    states = [env.get_host_state_for_next_assignment() for env in envs]
                    self.assertEqual(len({ok for _, ok in states}), 1)
                    if not states[0][1]:
                        break
                    np.testing.assert_array_equal(states[0][0]["obs"], states[1][0]["obs"])
                    np.testing.assert_array_equal(states[0][0]["mask"], states[1][0]["mask"])
                    np.testing.assert_array_equal(states[2][0]["mask"], states[3][0]["mask"])
                    host_mask = states[2][0]["mask"]
                    rejected = np.flatnonzero((states[0][0]["mask"] > .5) & (host_mask <= .5))
                    if rejected.size:
                        with self.assertRaisesRegex(ValueError, "Host excluded"):
                            triple.host_select(rejected[0])
                    host = int(np.flatnonzero(host_mask > .5)[0])
                    for env in envs:
                        env.host_select(host)
                    vm_states = [env.get_vm_state_for_current_task()[0] for env in envs]
                    np.testing.assert_array_equal(vm_states[0]["obs"], vm_states[1]["obs"])
                    np.testing.assert_array_equal(vm_states[2]["mask"], vm_states[3]["mask"])
                    mask = vm_states[2]["mask"]
                    rejected = np.flatnonzero((vm_states[0]["mask"] > .5) & (mask <= .5))
                    exclusions += len(rejected)
                    if rejected.size:
                        with self.assertRaisesRegex(ValueError, "VM excluded"):
                            triple.vm_assign(rejected[0])
                    vm = int(np.flatnonzero(mask > .5)[0])
                    results = [env.vm_assign(vm) for env in envs]
                    for result in results[1:3]:
                        self.assertEqual(result[:2], results[0][:2])
                        self.assertEqual(result[2]["assignment_energy"], results[0][2]["assignment_energy"])
                    energy_rewards += results[2][2]["r_energy_total"]
                    for env in envs[1:]:
                        self.assertEqual(physical_state(env), physical_state(oe))
                results = [env.finish_phase_and_advance() for env in envs]
                self.assertTrue(all(result[0] == results[0][0] for result in results))
                for result in results[:3]:
                    self.assertAlmostEqual(result[1]["lower_assignment_energy_unemitted"], 0.)
                for env in envs[1:]:
                    self.assertEqual(physical_state(env), physical_state(oe))
                phases += 1
                self.assertLess(phases, 1000)
        self.assertGreater(exclusions, 0)
        self.assertEqual(triple.completed_workflows, 2)
        for env in (oe, ge, triple):
            self.assertAlmostEqual(env.assignment_energy_total, env.total_energy, places=6)
            self.assertAlmostEqual(env.lower_energy_emitted, env.total_energy, places=6)
        self.assertAlmostEqual(energy_rewards, -triple.energy_reward_scale * triple.total_energy, places=6)

    def test_filtered_summary_excludes_cheapest_slow_vm_without_mutating_raw_rows(self):
        env = self.env(TripleEnv)
        tid = env.ready_task_ids[0]
        rows, durations = {}, {}
        for hi, host_id in enumerate(env.host_ids):
            for slot, j in enumerate(env.host_to_vm_indices[host_id]):
                duration = (10. if slot == 4 else 30.) * (5 if hi == 2 else 1)
                durations[j] = duration
                rows[j] = (duration, 20. if slot == 4 else 1., 0.)
        untouched, before = deepcopy(rows), physical_state(env)
        with (patch.object(OEEnv, "_candidate_rows", return_value=rows),
              patch.object(env, "_task_duration", side_effect=lambda task, j: durations[j])):
            retained = env._candidate_rows(tid)
            self.assertEqual(set(retained), {indices[4] for indices in env.host_to_vm_indices.values()})
            obs, mask = env._build_host_obs_for_task(tid)
            np.testing.assert_array_equal(mask, [1, 1, 0])
            summary = obs[env.base_host_obs_dim + 1:].reshape(env.num_hosts, 5)
            scale = env.variant_metadata()["energy_ref_j"]
            self.assertAlmostEqual(summary[0, 3] * scale, 20., places=5)
            self.assertAlmostEqual(summary[0, 4] * env.variant_metadata()["time_ref_s"], 10., places=5)
            np.testing.assert_array_equal(summary[2], np.zeros(5))
            vm_obs, vm_mask = env._build_vm_obs_for_task_host(tid, env.host_ids[0])
            self.assertEqual(list(np.flatnonzero(vm_mask > .5)), [4])
            start = env.base_vm_obs_dim + 1 + env.num_hosts + 2
            extras = vm_obs[start:].reshape(env.max_vms_per_host, 2)
            self.assertTrue(np.all(extras[vm_mask <= .5] == 0.))
        self.assertEqual(rows, untouched)
        self.assertEqual(physical_state(env), before)

    def test_ge_explores_only_after_pf_and_leaves_deterministic_policy_to_q(self):
        state, durations = np.zeros(2), np.array([10., 30., 12.])
        mask = performance_mask(np.ones(3), durations)
        for cls in (OEGuidedAgent, OEGuidedFilterAgent):
            agent = cls(2, 3, hidden_dims=(8,), eps_start=1., eps_end=1., device=torch.device("cpu"))
            with patch("common.mahdrl_performance.random.random", side_effect=[.1, .1]):
                self.assertEqual(agent.select_action(state, mask, performance_times=durations), 0)
            with (patch("common.mahdrl_performance.random.random", side_effect=[.1, .9]),
                  patch("common.mahdrl_performance.np.random.choice", return_value=2) as choose):
                self.assertEqual(agent.select_action(state, mask, performance_times=durations), 2)
                np.testing.assert_array_equal(choose.call_args.args[0], [0, 2])
            with torch.no_grad():
                for p in agent.online.parameters():
                    p.zero_()
                agent.online.adv[-1].bias.copy_(torch.tensor([0., 100., 10.]))
            self.assertEqual(agent.select_action(state, mask, deterministic=True), 2)
            self.assertEqual(agent.select_action(state, np.ones(3), deterministic=True), 1)
            self.assertEqual(agent._eps_steps, 0)

    def test_filtered_successor_mask_constrains_bootstrap(self):
        agent = OEGuidedFilterAgent(2, 3, hidden_dims=(8,), batch_size=1, device=torch.device("cpu"))
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for p in network.parameters():
                    p.zero_()
            agent.online.adv[-1].bias.copy_(torch.tensor([0., 100., 10.]))
            agent.target.adv[-1].bias.copy_(torch.tensor([0., 1000., 20.]))
            expected = -1. + agent.gamma * agent.target(torch.zeros(1, 2))[0, 2].item()
        state = np.zeros(2)
        mask = performance_mask(np.ones(3), [10., 30., 12.])
        agent.remember(state, mask, 0, -1., state, mask, 0.)
        agent.update()
        self.assertAlmostEqual(agent.f11.snapshot()["target_mean"], expected, places=4)

    def test_forced_energy_still_uses_oe_settlement_once(self):
        for cls in (OEGEEnv, TripleEnv):
            env = self.env(cls)
            env._fuse_zero_assign_limit = 1
            traces = [PendingDecision(lambda *args: None) for _ in range(2)]
            for trace in traces:
                trace.record(np.ones(2), np.ones(2), 0, 0., host_index=0)
            with redirect_stdout(io.StringIO()):
                _, info = env.finish_phase_and_advance()
                delta = info["lower_assignment_energy_unemitted"]
                self.assertGreater(delta, 0.)
                finish_lower_phase(env, *traces, info)
            for trace, alpha in zip(traces, (env.alpha_delay_host, env.alpha_delay_vm)):
                self.assertAlmostEqual(trace.recorded_reward, -(1-alpha)*delta*env.energy_reward_scale)
            self.assertAlmostEqual(env.lower_energy_emitted, env.assignment_energy_total, places=7)

    def test_checkpoint_identity_and_role_reject_singles_and_other_combination(self):
        directory = self.root / "checkpoint_controls"
        directory.mkdir()
        envs = {OEAgent: self.env(OEEnv), OEGuidedAgent: self.env(OEGEEnv),
                OEGuidedFilterAgent: self.env(TripleEnv)}
        files = {}
        for cls in (OriginalAgent, OEAgent, GuidedExplorationAgent, PerformanceFilterAgent,
                    OEGuidedAgent, OEGuidedFilterAgent):
            agent = cls(2, 3, hidden_dims=(8,), device=torch.device("cpu"))
            if cls in envs:
                binder = bind_oe if cls is OEAgent else bind_variant
                binder(agent, envs[cls].variant_metadata(), "TA")
            filename = directory / (cls.__module__.replace(".", "_") + "_" + cls.__name__ + ".pth")
            agent.save(filename)
            files[cls] = filename
        for cls in (OEGuidedAgent, OEGuidedFilterAgent):
            agent = cls(2, 3, hidden_dims=(8,), device=torch.device("cpu"))
            metadata = envs[cls].variant_metadata()
            bind_variant(agent, metadata, "TA")
            agent.load(files[cls])
            for other, filename in files.items():
                if other is not cls:
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        agent.load(filename)
            bind_variant(agent, metadata, "HA")
            with self.assertRaisesRegex(ValueError, "variant/role mismatch"):
                agent.load(files[cls])
            foreign = envs[OEGuidedFilterAgent if cls is OEGuidedAgent else OEGuidedAgent].variant_metadata()
            with self.assertRaisesRegex(ValueError, "matching"):
                bind_variant(agent, foreign, "TA")
            broken = deepcopy(metadata)
            broken["performance_action_filter"] = not broken["performance_action_filter"]
            with self.assertRaisesRegex(ValueError, "matching"):
                bind_variant(agent, broken, "TA")

    def test_scenario_namespaces_and_cli(self):
        names = ("mahdrl", "mahdrl_oe", "mahdrl_ge", "mahdrl_pf", "mahdrl_ts", "mahdrl_ub",
                 "mahdrl_oe_ge", "mahdrl_oe_ge_pf")
        for resource in "SML":
            for deadline in ("Tight", "Medium", "Loose"):
                scene = Scenario(resource=resource, deadline=deadline)
                self.assertEqual(len({scene.checkpoint_dir(name) for name in names}), len(names))
                for algorithm in names[-2:]:
                    self.assertIn("f17_normalized_weights_v1", scene.model_name(algorithm))
        for algorithm in names[-2:]:
            for mode in ("train", "eval"):
                with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(SystemExit) as result:
                        parse_scenario(mode, algorithm)
                self.assertEqual(result.exception.code, 0)
                self.assertIn(algorithm, json.loads(out.getvalue())["checkpoint_dir"])

    def test_actual_training_save_load_evaluation_and_existing_outputs_preserved(self):
        root, cache, scales = self.root / "training", self.cache, self.scales

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
        configs = {}
        for algorithm in ("mahdrl_oe", "mahdrl_oe_ge", "mahdrl_oe_ge_pf"):
            trainer = importlib.import_module("algorithms." + algorithm + ".train")
            env_class, agent_class = trainer.CloudWorkflowEnv_VMAgents, trainer.D3QNAgent

            class TinyEpisode(env_class):
                def __init__(self, **kwargs):
                    kwargs["workflows_per_episode"] = 2
                    super().__init__(**kwargs)

            captured, agents = [], []

            def small_agent(*args, **kwargs):
                captured.append(dict(kwargs))
                kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
                agent = agent_class(*args, **kwargs)
                if algorithm != "mahdrl_oe":
                    original_select = agent.select_action
                    agent.guided_test_calls = 0

                    def checked_select(state, mask, deterministic=False, count_step=True, **extra):
                        role = agent.variant_metadata["role"]
                        if not deterministic and role in ("HA", "VA"):
                            self.assertIn("performance_times", extra)
                            agent.guided_test_calls += 1
                        else:
                            self.assertNotIn("performance_times", extra)
                        return original_select(state, mask, deterministic, count_step, **extra)

                    agent.select_action = checked_select
                agents.append(agent)
                return agent

            with (patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)),
                  patch.object(trainer, "CloudWorkflowEnv_VMAgents", TinyEpisode),
                  patch.object(trainer, "D3QNAgent", side_effect=small_agent), redirect_stdout(io.StringIO())):
                trainer.train()
            configs[algorithm] = list(captured)
            if algorithm == "mahdrl_oe":
                protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in (scene.checkpoint_dir(algorithm), scene.log_dir(algorithm))
                             for p in folder.rglob("*") if p.is_file()}
                continue
            self.assertEqual(configs[algorithm], configs["mahdrl_oe"])
            for path, digest in protected.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            policy = OE_GE_PF_POLICY if algorithm.endswith("_pf") else OE_GE_POLICY
            for agent in agents:
                self.assertGreater(agent._updates, 0)
                self.assertEqual(agent.guided_test_calls > 0, agent.variant_metadata["role"] != "TA")
                self.assertTrue(any(row[-1] for row in agent.buffer))
                for row in agent.buffer:
                    self.assertGreater(row[1][row[2]], .5)
                    if not row[-1]:
                        self.assertTrue(np.any(row[5] > .5))
            directory = scene.checkpoint_dir(algorithm)
            for role in ("vm", "host", "manager"):
                for name in ("best_" + role + ".pth", role + "_final.pth"):
                    checkpoint = torch.load(directory / name, weights_only=True)
                    self.assertGreater(checkpoint["updates"], 0)
                    metadata = checkpoint["normalization"]
                    self.assertEqual(metadata["performance_policy"], asdict(policy))
                    self.assertEqual(metadata["algorithm_variant"]["scheme"], policy.scheme)
            variant = json.loads((directory / "algorithm_variant.json").read_text(encoding="utf-8"))
            self.assertEqual(variant["algorithm"], algorithm)
            self.assertEqual(variant["components"], ["OE", "GE", "PF"] if algorithm.endswith("_pf") else ["OE", "GE"])
            measurements = json.loads((scene.log_dir(algorithm) / "measurement_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(measurements["algorithm"], algorithm)
            self.assertEqual(measurements["agents"]["VA"]["config"], asdict(policy))
            records = [json.loads(line) for line in (scene.log_dir(algorithm) / "f11_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
            for key in ("physical_energy_balance", "lower_energy_balance", "HA_reward_formula_balance", "VA_reward_formula_balance"):
                self.assertAlmostEqual(records[-1]["reward_audit"][key], 0., places=6)

            evaluator = importlib.import_module("algorithms." + algorithm + ".evaluate")

            def recorder(*args, **kwargs):
                return EvaluationRecorder(*args, **kwargs, directory=root / (algorithm + "_measurements"))

            evaluation = replace(scene, mode="eval", seed_spec="700")
            with (patch.object(evaluator, "SCENARIO", evaluation), patch.object(evaluator, "PROJECT_DIR", str(root)),
                  patch.object(evaluator, "EnvHRLMix", TinyEpisode),
                  patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
                  patch.object(evaluator, "EvaluationRecorder", side_effect=recorder), redirect_stdout(io.StringIO())):
                evaluator.main()
                episode = json.loads((root / (algorithm + "_measurements/episodes/seed700.json")).read_text(encoding="utf-8"))
                self.assertEqual(episode["algorithm"], algorithm)
                self.assertEqual(episode["metrics"]["wf_completed_count"], 2)
                self.assertGreater(episode["metrics"]["total_energy"], 0)
                self.assertEqual(len(list((root / "outputs/run").rglob("eval_" + algorithm + "_*.csv"))), 1)
                old = replace(evaluation, checkpoint_override=str(scene.checkpoint_dir("mahdrl_oe")))
                with patch.object(evaluator, "SCENARIO", old):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        evaluator.build_three_layer_agents_and_load(evaluator.build_env_kwargs_hrl_mix(700))
            with patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
