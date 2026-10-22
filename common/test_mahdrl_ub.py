"""TA upper-bound targets, unmodified greedy choices, provenance and end-to-end isolation."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from algorithms.mahdrl import train as baseline
from algorithms.mahdrl import evaluate as baseline_eval
from algorithms.mahdrl_ub import train as trainer
from algorithms.mahdrl_ub import evaluate as evaluator
from algorithms.mahdrl_ub.agent import TAUpperBoundAgent, TA_TARGET_RULE
from common.d3qn_agent_EXDehid import D3QNAgent
from common.experiment_measurements import EvaluationRecorder
from common.f18_state_normalization import VERSION
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, obs):
        return self.values.unsqueeze(0).expand(len(obs), -1)


class MAHDRLUBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        flags = patch.dict(os.environ, F11_DIAGNOSTICS="1", F19_DIAGNOSTICS="1", EXPERIMENT_METRICS="1")
        flags.start()
        self.addCleanup(flags.stop)

    def agent(self, cls=TAUpperBoundAgent):
        return cls(2, 3, gamma=0.9, batch_size=1, buffer_size=16,
                   hidden_dims=(8, 4), device=torch.device("cpu"))

    def fixed_update(self, target_values, done=False, next_mask=(1., 1., 0.)):
        agent = self.agent()
        agent.online, agent.target = FixedQ([1., 3., 999.]), FixedQ(target_values)
        agent.optim = torch.optim.Adam(agent.online.parameters(), lr=0.0003)
        agent.remember(np.array([1., 2.]), np.array([1., 1., 0.]), 0, -2.,
                       np.array([3., 4.]), np.array(next_mask), float(done))
        captured = []
        original_loss = F.smooth_l1_loss

        def capture(q, target):
            captured.append(target.detach().clone())
            return original_loss(q, target)

        with patch("torch.nn.functional.smooth_l1_loss", side_effect=capture):
            loss = agent.update()
        self.assertTrue(np.isfinite(loss))
        return agent, captured[0].item()

    def test_positive_future_is_projected_but_raw_values_remain_visible(self):
        agent, target = self.fixed_update([-4., 7., 888.])
        self.assertEqual(target, -2.)
        profiles = agent.f11.numerics.snapshot()["profiles"]
        self.assertEqual(profiles["Q_next_raw"]["P50"], [7.])
        self.assertAlmostEqual(profiles["Bootstrap_raw"]["P50"][0], 6.3, places=5)
        self.assertAlmostEqual(profiles["TD_target_raw"]["P50"][0], 4.3, places=5)
        self.assertEqual(profiles["Bootstrap_projected"]["P50"], [0.])
        self.assertEqual(profiles["TD_target"]["P50"], [-2.])
        self.assertEqual(profiles["Q_selected"]["P50"], [1.])
        telemetry = agent.f11.snapshot()["ta_future_upper_bound"]
        self.assertEqual(telemetry["eligible_samples"], 1)
        self.assertEqual(telemetry["projected_samples"], 1)
        self.assertEqual(telemetry["last_batch_projection_ratio"], 1.)

    def test_negative_future_double_dqn_and_illegal_mask_are_preserved(self):
        agent, target = self.fixed_update([-4., -10., 888.])
        # Online prefers legal action 1; target's preferred action 0 must not replace it.
        self.assertAlmostEqual(target, -11.)
        self.assertAlmostEqual(agent.f11.snapshot()["future_abs_mean"], 9.)
        self.assertEqual(agent.f11.snapshot()["ta_future_upper_bound"]["projected_samples"], 0)

    def test_terminal_future_is_zero_and_excluded_from_projection_denominator(self):
        for mask in ((0., 0., 0.), (1., 1., 0.)):
            with self.subTest(mask=mask):
                agent, target = self.fixed_update([-4., 7., 888.], done=True, next_mask=mask)
                self.assertEqual(target, -2.)
                self.assertEqual(agent.f11.snapshot()["future_abs_mean"], 0.)
                telemetry = agent.f11.snapshot()["ta_future_upper_bound"]
                self.assertEqual(telemetry["eligible_samples"], 0)
                self.assertEqual(telemetry["projected_samples"], 0)

    def test_projection_rate_counts_only_eligible_samples_across_updates(self):
        agent = self.agent()
        agent.online, agent.target = FixedQ([1., 3., 999.]), FixedQ([-4., 7., 888.])
        agent.optim = torch.optim.Adam(agent.online.parameters(), lr=0.0003)
        for future, done in ((7., False), (-10., False), (7., True)):
            agent.buffer.clear()
            with torch.no_grad():
                agent.target.values[1] = future
            agent.remember(np.ones(2), np.array([1., 1., 0.]), 0, -2.,
                           np.ones(2), np.array([1., 1., 0.]), float(done))
            agent.update()
        telemetry = agent.f11.snapshot()["ta_future_upper_bound"]
        self.assertEqual(telemetry["eligible_samples"], 2)
        self.assertEqual(telemetry["projected_samples"], 1)
        self.assertEqual(telemetry["cumulative_projection_ratio"], 0.5)
        self.assertEqual(telemetry["last_batch_eligible"], 0)

    def test_projection_still_applies_with_diagnostics_disabled(self):
        enabled, enabled_target = self.fixed_update([-4., 7., 888.])
        with patch.dict(os.environ, F11_DIAGNOSTICS="0", F19_DIAGNOSTICS="0"):
            disabled, disabled_target = self.fixed_update([-4., 7., 888.])
        self.assertEqual(enabled_target, disabled_target)
        self.assertEqual(disabled_target, -2.)
        self.assertTrue(torch.equal(enabled.online.values, disabled.online.values))
        self.assertFalse(disabled.f11.numerics.snapshot()["enabled"])

    def test_greedy_action_uses_unprojected_q(self):
        agent = self.agent()
        agent.online = FixedQ([1., 3., 999.])
        action = agent.select_action(np.ones(2), np.array([1., 1., 0.]), deterministic=True, count_step=False)
        self.assertEqual(action, 1)  # Clamping all positive action scores would incorrectly tie at action 0.
        self.assertIs(TAUpperBoundAgent.select_action, D3QNAgent.select_action)

    def test_nonpositive_reward_precondition_and_nonterminal_validation(self):
        agent = self.agent()
        with self.assertRaisesRegex(ValueError, "nonpositive rewards"):
            agent.remember(np.ones(2), np.ones(3), 0, 0.1, np.ones(2), np.ones(3), 0.)
        self.assertEqual(len(agent.buffer), 0)
        with self.assertRaisesRegex(ValueError, "No legal action"):
            agent.remember(np.ones(2), np.ones(3), 0, -1., np.ones(2), np.zeros(3), 0.)

    def test_negative_future_update_matches_baseline_and_input_hooks_survive(self):
        torch.manual_seed(42)
        original = self.agent(D3QNAgent)
        torch.manual_seed(42)
        variant = self.agent()
        for agent in (original, variant):
            with torch.no_grad():
                for parameter in agent.target.parameters():
                    parameter.zero_()
                agent.target.val[-1].bias.fill_(-0.5)
            agent.remember(np.ones(2), np.ones(3), 1, -2., np.zeros(2), np.ones(3), 0.)
        random.seed(7)
        original_loss = original.update()
        random.seed(7)
        variant_loss = variant.update()
        self.assertEqual(original_loss, variant_loss)
        for name in ("online", "target"):
            for key, value in getattr(original, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(variant, name).state_dict()[key]))
        self.assertEqual(original.epsilon(), variant.epsilon())
        profiles = variant.f11.numerics.snapshot()["profiles"]
        self.assertIn("online_input", profiles)
        self.assertIn("target_input", profiles)

    def test_checkpoint_identifies_ta_rule_and_rejects_unmarked_weights(self):
        with tempfile.TemporaryDirectory(prefix="ub_ckpt_") as temporary:
            root = Path(temporary)
            agent, original = self.agent(), self.agent(D3QNAgent)
            agent.save(str(root / "bound.pth"))
            agent.load(str(root / "bound.pth"), require_learning_model=True)
            checkpoint = torch.load(root / "bound.pth", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["normalization"]["ta_future_value_target"], TA_TARGET_RULE)
            original.save(str(root / "original.pth"))
            with self.assertRaisesRegex(ValueError, "target-rule mismatch"):
                agent.load(str(root / "original.pth"))
            checkpoint["normalization"]["ta_future_value_target"]["q_upper_bound"] = 1.
            torch.save(checkpoint, root / "wrong_bound.pth")
            with self.assertRaisesRegex(ValueError, "target-rule mismatch"):
                agent.load(str(root / "wrong_bound.pth"))
            checkpoint["normalization"] = None
            torch.save(checkpoint, root / "missing_metadata.pth")
            with self.assertRaisesRegex(ValueError, "target-rule mismatch"):
                agent.load(str(root / "missing_metadata.pth"))

    def test_namespaces_environment_and_cli(self):
        self.assertIs(trainer.D3QNAgent, baseline.D3QNAgent)
        self.assertIs(trainer.CloudWorkflowEnv_VMAgents, baseline.CloudWorkflowEnv_VMAgents)
        self.assertIs(evaluator.EnvHRLMix, baseline_eval.EnvHRLMix)
        for resource in "SML":
            for deadline in ("Tight", "Medium", "Loose"):
                selected = Scenario(resource=resource, deadline=deadline)
                names = ("mahdrl", "mahdrl_oe", "mahdrl_ts", "mahdrl_ub")
                self.assertEqual(len({selected.checkpoint_dir(name) for name in names}), 4)
                with patch.object(evaluator, "SCENARIO", selected), patch.object(baseline_eval, "SCENARIO", selected):
                    self.assertEqual(evaluator.build_env_kwargs_hrl_mix(700), baseline_eval.build_env_kwargs_hrl_mix(700))
        for mode in ("train", "eval"):
            with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result:
                    parse_scenario(mode, "mahdrl_ub")
            self.assertEqual(result.exception.code, 0)
            self.assertIn("mahdrl_ub", json.loads(out.getvalue())["checkpoint_dir"])

    def test_real_updates_save_load_evaluation_and_isolation(self):
        with tempfile.TemporaryDirectory(prefix="ub_smoke_") as temporary:
            root = Path(temporary)
            scales, cache = root / "scales.json", root / "cache.json"
            fixture_config(scales)
            with redirect_stdout(io.StringIO()):
                precompute(Scenario(mode="precompute", seed_spec="0,600,700", cache_output=str(cache)))

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

            class TinyEpisode(trainer.CloudWorkflowEnv_VMAgents):
                def __init__(self, **kwargs):
                    kwargs["workflows_per_episode"] = 2
                    super().__init__(**kwargs)

            configs = []

            def factory(cls):
                def small(*args, **kwargs):
                    configs.append(dict(kwargs))
                    kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
                    return cls(*args, **kwargs)
                return small

            selected = LocalScenario(mode="train", episodes=1, normalization_mode=VERSION,
                                     normalization_config=str(scales))
            captures = {}
            for algorithm, module in (("mahdrl", baseline), ("mahdrl_ub", trainer)):
                configs.clear()
                with (patch.object(module, "SCENARIO", selected), patch.object(module, "ROOT_DIR", str(root)),
                      patch.object(module, "CloudWorkflowEnv_VMAgents", TinyEpisode),
                      patch.object(module, "D3QNAgent", side_effect=factory(D3QNAgent)), redirect_stdout(io.StringIO())):
                    if algorithm == "mahdrl_ub":
                        with patch.object(module, "TAUpperBoundAgent", side_effect=factory(TAUpperBoundAgent)):
                            module.train()
                    else:
                        module.train()
                captures[algorithm] = list(configs)
                if algorithm == "mahdrl":
                    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for folder in (selected.checkpoint_dir(algorithm), selected.log_dir(algorithm))
                                 for p in folder.rglob("*") if p.is_file()}
            self.assertEqual(len(captures["mahdrl_ub"]), 3)
            self.assertEqual(captures["mahdrl"], captures["mahdrl_ub"])
            for path, digest in protected.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            for name in ("mahdrl_oe", "mahdrl_ts"):
                self.assertFalse(selected.checkpoint_dir(name).exists())
                self.assertFalse(selected.log_dir(name).exists())

            directory = selected.checkpoint_dir("mahdrl_ub")
            for role in ("vm", "host", "manager"):
                checkpoint = torch.load(directory / ("best_" + role + ".pth"), map_location="cpu", weights_only=True)
                self.assertGreater(checkpoint["updates"], 0)
                self.assertEqual(checkpoint["optim"]["param_groups"][0]["lr"], 0.0003)
                self.assertEqual(checkpoint["tau"], 0.01 if role == "manager" else 0.005)
                if role == "manager":
                    self.assertEqual(checkpoint["normalization"]["ta_future_value_target"], TA_TARGET_RULE)
                else:
                    self.assertNotIn("ta_future_value_target", checkpoint["normalization"])
            record = json.loads((selected.log_dir("mahdrl_ub") / "f11_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            telemetry = record["learners"]["TA"]["ta_future_upper_bound"]
            self.assertGreater(telemetry["eligible_samples"], 0)
            self.assertLessEqual(telemetry["projected_samples"], telemetry["eligible_samples"])
            for name in ("Q_selected", "TD_target", "TD_error", "Q_next_raw", "TD_target_raw", "Bootstrap_raw", "Bootstrap_projected"):
                self.assertIn(name, record["numerics"]["TA"]["profiles"])
            self.assertLessEqual(record["numerics"]["TA"]["profiles"]["TD_target"]["P99"][0], 0.)
            for role in ("HA", "VA"):
                self.assertNotIn("ta_future_upper_bound", record["learners"][role])
            self.assertGreater(record["learners"]["TA"]["terminal_transitions"], 0)

            def recorder(*args, **kwargs):
                return EvaluationRecorder(*args, **kwargs, directory=root / "measurements")

            evaluation_scene = replace(selected, mode="eval", seed_spec="700")
            with (patch.object(evaluator, "SCENARIO", evaluation_scene),
                  patch.object(evaluator, "PROJECT_DIR", str(root)), patch.object(evaluator, "EnvHRLMix", TinyEpisode),
                  patch.object(evaluator, "D3QNAgent", side_effect=factory(D3QNAgent)),
                  patch.object(evaluator, "TAUpperBoundAgent", side_effect=factory(TAUpperBoundAgent)),
                  patch.object(evaluator, "EvaluationRecorder", side_effect=recorder), redirect_stdout(io.StringIO())):
                evaluator.main()
                outputs = list((root / "outputs/run").rglob("*.csv"))
                self.assertEqual(len(outputs), 1)
                self.assertTrue(outputs[0].name.startswith("eval_mahdrl_ub_"))
                result = json.loads((root / "measurements/episodes/seed700.json").read_text(encoding="utf-8"))
                self.assertEqual(result["algorithm"], "mahdrl_ub")
                self.assertGreater(result["metrics"]["total_energy"], 0.)
                self.assertEqual(result["metrics"]["wf_completed_count"], 2)
                old_scene = replace(evaluation_scene, checkpoint_override=str(selected.checkpoint_dir("mahdrl")))
                with patch.object(evaluator, "SCENARIO", old_scene):
                    kwargs = evaluator.build_env_kwargs_hrl_mix(700)
                    with self.assertRaisesRegex(ValueError, "target-rule mismatch"):
                        evaluator.build_three_layer_agents_and_load(kwargs)

            with patch.object(trainer, "SCENARIO", selected), patch.object(trainer, "ROOT_DIR", str(root)):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
