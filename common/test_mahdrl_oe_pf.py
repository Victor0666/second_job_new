"""OE/PF is isolated and uses original uniform exploration, never GE."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.mahdrl_oe import train as oe_trainer
from algorithms.mahdrl_oe.agent import D3QNAgent as OEAgent, bind_variant as bind_oe
from algorithms.mahdrl_oe.reward import finish_lower_phase
from algorithms.mahdrl_oe_pf import train as trainer, evaluate as evaluator
from algorithms.mahdrl_oe_pf.agent import D3QNAgent, OE_PF_POLICY, bind_variant
from common.d3qn_agent_EXDehid import D3QNAgent as OriginalAgent
from common.experiment_measurements import EvaluationRecorder
from common.f11 import PendingDecision
from common.f18_state_normalization import VERSION
from common.mahdrl_combinations import OEGuidedAgent, OEGuidedFilterAgent, bind_variant as bind_combination
from common.mahdrl_performance import GuidedExplorationAgent, PerformanceFilterAgent
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config
from common.test_mahdrl_combinations import physical_state
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as OEEnv
from environments.mahdrl_oe_ge import CloudWorkflowEnv_VMAgents as OEGEEnv
from environments.mahdrl_oe_ge_pf import CloudWorkflowEnv_VMAgents as TripleEnv
from environments.mahdrl_oe_pf import CloudWorkflowEnv_VMAgents as Env


class OEPFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flags = patch.dict("os.environ", F11_DIAGNOSTICS="1", F19_DIAGNOSTICS="1", EXPERIMENT_METRICS="1")
        flags.start()
        cls.addClassCleanup(flags.stop)
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="mahdrl_oe_pf_")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.scales, cls.cache = cls.root / "scales.json", cls.root / "cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,600,700", cache_output=str(cls.cache)))

    def env(self, cls=Env):
        env = cls(dax_paths=Scenario().dax_paths(), random_seed=0, workflows_per_episode=2,
                  horizon=1e9, arrival_lambda=.03, num_hosts=3, vms_per_host=(9, 8, 8),
                  max_ready_tasks=100, pc_tiers=(1., 2., 4., 6., 8.),
                  bw_tiers=(1000., 2000., 4000., 6000., 8000.), deadline_mode="cache_fcfs",
                  deadline_cache_path=str(self.cache), deadline_alpha_small_prob=.8,
                  normalization_mode=VERSION, normalization_config=str(self.scales),
                  source_train_scene="SS_T")
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    def test_agent_uses_original_action_selection_updates_and_random_draws(self):
        self.assertNotIn(GuidedExplorationAgent, D3QNAgent.__mro__)
        self.assertIs(D3QNAgent.select_action, OriginalAgent.select_action)
        self.assertIs(D3QNAgent.update, OriginalAgent.update)
        agent = D3QNAgent(2, 4, hidden_dims=(8,), eps_start=1., eps_end=1., device=torch.device("cpu"))
        original = OriginalAgent(2, 4, hidden_dims=(8,), eps_start=1., eps_end=1., device=torch.device("cpu"))
        original.online.load_state_dict(agent.online.state_dict())
        self.assertEqual(agent.cfg.guided_probability, 0.)
        state, mask = np.zeros(2), np.array([1., 0., 1., 1.])
        old_python, old_numpy = random.getstate(), np.random.get_state()
        try:
            for deterministic in (False, True):
                random.seed(42)
                np.random.seed(42)
                actions = [agent.select_action(state, mask, deterministic) for _ in range(40)]
                after_python, after_numpy = random.getstate(), np.random.get_state()
                random.seed(42)
                np.random.seed(42)
                expected = [original.select_action(state, mask, deterministic) for _ in range(40)]
                self.assertEqual(actions, expected)
                self.assertEqual(random.getstate(), after_python)
                np.testing.assert_array_equal(np.random.get_state()[1], after_numpy[1])
                self.assertEqual(np.random.get_state()[2:], after_numpy[2:])
                self.assertTrue(set(actions) <= {0, 2, 3})
        finally:
            random.setstate(old_python)
            np.random.set_state(old_numpy)

    def test_environment_masks_summaries_and_rewards_match_filtered_oe(self):
        env, triple, oe = self.env(), self.env(TripleEnv), self.env(OEEnv)
        all_envs = (env, triple, oe)
        energy_rewards = phases = 0
        with redirect_stdout(io.StringIO()):
            while not env.done_flag:
                while True:
                    states = [e.get_host_state_for_next_assignment() for e in all_envs]
                    self.assertEqual(len({ok for _, ok in states}), 1)
                    if not states[0][1]:
                        break
                    for field in ("obs", "mask"):
                        np.testing.assert_array_equal(states[0][0][field], states[1][0][field])
                    host = int(np.flatnonzero(states[0][0]["mask"] > .5)[0])
                    for e in all_envs:
                        e.host_select(host)
                    states = [e.get_vm_state_for_current_task()[0] for e in all_envs]
                    for field in ("obs", "mask"):
                        np.testing.assert_array_equal(states[0][field], states[1][field])
                    slot = int(np.flatnonzero(states[0]["mask"] > .5)[0])
                    results = [e.vm_assign(slot) for e in all_envs]
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(results[0][:2], results[2][:2])
                    energy_rewards += results[0][2]["r_energy_total"]
                    self.assertEqual(physical_state(env), physical_state(triple))
                    self.assertEqual(physical_state(env), physical_state(oe))
                results = [e.finish_phase_and_advance() for e in all_envs]
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[0][0], results[2][0])
                self.assertEqual(physical_state(env), physical_state(oe))
                phases += 1
                self.assertLess(phases, 1000)
        self.assertEqual(env.completed_workflows, 2)
        self.assertAlmostEqual(env.assignment_energy_total, env.total_energy, places=6)
        self.assertAlmostEqual(energy_rewards, -env.energy_reward_scale * env.total_energy, places=6)
        metadata = env.variant_metadata()
        self.assertEqual(metadata["components"], ["OE", "PF"])
        self.assertEqual(metadata["algorithm"], "mahdrl_oe_pf")
        self.assertEqual(metadata["host_summary_scope"], "performance_candidates")
        self.assertEqual(metadata["performance_policy"], asdict(OE_PF_POLICY))

    def test_forced_energy_settlement_is_still_oe(self):
        env = self.env()
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

    def test_checkpoint_rejects_singles_other_combinations_and_wrong_role(self):
        directory = self.root / "identities"
        directory.mkdir()
        controls = [
            (OriginalAgent, None, None), (OEAgent, self.env(OEEnv), bind_oe),
            (GuidedExplorationAgent, None, None), (PerformanceFilterAgent, None, None),
            (OEGuidedAgent, self.env(OEGEEnv), bind_combination),
            (OEGuidedFilterAgent, self.env(TripleEnv), bind_combination),
            (D3QNAgent, self.env(), bind_variant),
        ]
        saved = []
        for i, (cls, environment, binder) in enumerate(controls):
            agent = cls(2, 3, hidden_dims=(8,), device=torch.device("cpu"))
            if binder:
                binder(agent, environment.variant_metadata(), "TA")
            filename = directory / f"control_{i}.pth"
            agent.save(filename)
            saved.append((agent, filename))
        agent = saved[-1][0]
        agent.load(saved[-1][1])
        for other, filename in saved[:-1]:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                agent.load(filename)
            if isinstance(other, (OEAgent, GuidedExplorationAgent, PerformanceFilterAgent)):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    other.load(saved[-1][1])
        bind_variant(agent, controls[-1][1].variant_metadata(), "HA")
        with self.assertRaisesRegex(ValueError, "variant/role mismatch"):
            agent.load(saved[-1][1])
        with self.assertRaisesRegex(ValueError, "without GE"):
            bind_variant(agent, self.env(TripleEnv).variant_metadata(), "TA")

    def test_cli_and_output_namespaces(self):
        names = ("mahdrl", "mahdrl_oe", "mahdrl_ge", "mahdrl_pf", "mahdrl_oe_ge", "mahdrl_oe_ge_pf", "mahdrl_oe_pf")
        for resource in "SML":
            for deadline in ("Tight", "Medium", "Loose"):
                scene = Scenario(resource=resource, deadline=deadline)
                self.assertEqual(len({scene.checkpoint_dir(name) for name in names}), len(names))
        for mode in ("train", "eval"):
            with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result:
                    parse_scenario(mode, "mahdrl_oe_pf")
            self.assertEqual(result.exception.code, 0)
            self.assertIn("mahdrl_oe_pf", json.loads(out.getvalue())["checkpoint_dir"])

    def test_actual_training_roundtrip_evaluation_and_original_outputs(self):
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
        captures = {}
        for algorithm, module in (("mahdrl_oe", oe_trainer), ("mahdrl_oe_pf", trainer)):
            env_class, agent_class = module.CloudWorkflowEnv_VMAgents, module.D3QNAgent

            class TinyEpisode(env_class):
                def __init__(self, **kwargs):
                    kwargs["workflows_per_episode"] = 2
                    super().__init__(**kwargs)

            configs, agents = [], []

            def small_agent(*args, **kwargs):
                configs.append(dict(kwargs))
                kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
                agent = agent_class(*args, **kwargs)
                original_select = agent.select_action

                def select_without_ge(*args, **kwargs):
                    self.assertNotIn("performance_times", kwargs)
                    return original_select(*args, **kwargs)

                agent.select_action = select_without_ge
                agents.append(agent)
                return agent

            with (patch.object(module, "SCENARIO", scene), patch.object(module, "ROOT_DIR", str(root)),
                  patch.object(module, "CloudWorkflowEnv_VMAgents", TinyEpisode),
                  patch.object(module, "D3QNAgent", side_effect=small_agent), redirect_stdout(io.StringIO())):
                module.train()
            captures[algorithm] = list(configs)
            if algorithm == "mahdrl_oe":
                protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in (scene.checkpoint_dir(algorithm), scene.log_dir(algorithm))
                             for p in folder.rglob("*") if p.is_file()}
                continue
            self.assertEqual(captures[algorithm], captures["mahdrl_oe"])
            for filename, digest in protected.items():
                self.assertEqual(hashlib.sha256(filename.read_bytes()).hexdigest(), digest)
            for agent in agents:
                self.assertGreater(agent._updates, 0)
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
                    self.assertEqual(checkpoint["normalization"]["performance_policy"], asdict(OE_PF_POLICY))
                    self.assertEqual(checkpoint["normalization"]["algorithm_variant"]["components"], ["OE", "PF"])
            variant = json.loads((directory / "algorithm_variant.json").read_text(encoding="utf-8"))
            self.assertEqual(variant["algorithm"], algorithm)
            records = [json.loads(line) for line in (scene.log_dir(algorithm) / "f11_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
            for key in ("physical_energy_balance", "lower_energy_balance", "HA_reward_formula_balance", "VA_reward_formula_balance"):
                self.assertAlmostEqual(records[-1]["reward_audit"][key], 0., places=6)

            def recorder(*args, **kwargs):
                return EvaluationRecorder(*args, **kwargs, directory=root / "measurements")

            evaluation = replace(scene, mode="eval", seed_spec="700")
            with (patch.object(evaluator, "SCENARIO", evaluation), patch.object(evaluator, "PROJECT_DIR", str(root)),
                  patch.object(evaluator, "EnvHRLMix", TinyEpisode),
                  patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
                  patch.object(evaluator, "EvaluationRecorder", side_effect=recorder), redirect_stdout(io.StringIO())):
                evaluator.main()
                episode = json.loads((root / "measurements/episodes/seed700.json").read_text(encoding="utf-8"))
                self.assertEqual(episode["algorithm"], algorithm)
                self.assertEqual(episode["metrics"]["wf_completed_count"], 2)
                self.assertGreater(episode["metrics"]["total_energy"], 0)
                self.assertEqual(len(list((root / "outputs/run").rglob("eval_mahdrl_oe_pf_*.csv"))), 1)
                old = replace(evaluation, checkpoint_override=str(scene.checkpoint_dir("mahdrl_oe")))
                with patch.object(evaluator, "SCENARIO", old):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        evaluator.build_three_layer_agents_and_load(evaluator.build_env_kwargs_hrl_mix(700))
            with patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
