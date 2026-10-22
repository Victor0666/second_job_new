"""MAHDRL-TS isolation, real updates, checkpoint round trip and evaluation outputs."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from algorithms.mahdrl import train as baseline
from algorithms.mahdrl import evaluate as baseline_eval
from algorithms.mahdrl_ts import train as trainer
from algorithms.mahdrl_ts import evaluate as evaluator
from common.d3qn_agent_EXDehid import D3QNAgent
from common.experiment_measurements import EvaluationRecorder
from common.f18_state_normalization import VERSION
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config


class MAHDRLTSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_namespaces_environment_and_cli(self):
        self.assertIs(trainer.D3QNAgent, baseline.D3QNAgent)
        self.assertIs(trainer.CloudWorkflowEnv_VMAgents, baseline.CloudWorkflowEnv_VMAgents)
        self.assertIs(evaluator.EnvHRLMix, baseline_eval.EnvHRLMix)
        for resource in "SML":
            for deadline in ("Tight", "Medium", "Loose"):
                scene = Scenario(resource=resource, deadline=deadline)
                paths = [scene.checkpoint_dir(name) for name in ("mahdrl", "mahdrl_oe", "mahdrl_ts")]
                self.assertEqual(len(set(paths)), 3)
                with patch.object(evaluator, "SCENARIO", scene), patch.object(baseline_eval, "SCENARIO", scene):
                    self.assertEqual(evaluator.build_env_kwargs_hrl_mix(700), baseline_eval.build_env_kwargs_hrl_mix(700))
        for mode in ("train", "eval"):
            with patch("sys.argv", ["entry", "--dry-run"]), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as result:
                    parse_scenario(mode, "mahdrl_ts")
            self.assertEqual(result.exception.code, 0)
            self.assertIn("mahdrl_ts", json.loads(out.getvalue())["checkpoint_dir"])

    def test_training_checkpoint_evaluation_and_original_preservation(self):
        with tempfile.TemporaryDirectory(prefix="mahdrl_ts_tests_") as temporary:
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

            def small_agent(*args, **kwargs):
                configs.append(dict(kwargs))
                kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=16, device=torch.device("cpu"))
                return D3QNAgent(*args, **kwargs)

            scene = LocalScenario(mode="train", episodes=1, normalization_mode=VERSION,
                                  normalization_config=str(scales))
            captures = {}
            for algorithm, module in (("mahdrl", baseline), ("mahdrl_ts", trainer)):
                configs.clear()
                with (patch.object(module, "SCENARIO", scene), patch.object(module, "ROOT_DIR", str(root)),
                      patch.object(module, "CloudWorkflowEnv_VMAgents", TinyEpisode),
                      patch.object(module, "D3QNAgent", side_effect=small_agent), redirect_stdout(io.StringIO())):
                    module.train()
                captures[algorithm] = list(configs)
                if algorithm == "mahdrl":
                    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for folder in (scene.checkpoint_dir(algorithm), scene.log_dir(algorithm))
                                 for path in folder.rglob("*") if path.is_file()}

            # Capture real constructor arguments before substituting tiny networks for the smoke test.
            self.assertEqual(len(captures["mahdrl_ts"]), 3)
            self.assertEqual(captures["mahdrl"][:2], captures["mahdrl_ts"][:2])
            old_ta, new_ta = captures["mahdrl"][2], captures["mahdrl_ts"][2]
            differences = {key: (old_ta[key], new_ta[key]) for key in old_ta if old_ta[key] != new_ta[key]}
            self.assertEqual(differences, {"lr": (0.0003, 0.0001), "target_update_tau": (0.01, 0.005)})
            for path, digest in protected.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertFalse(scene.checkpoint_dir("mahdrl_oe").exists())
            self.assertFalse(scene.log_dir("mahdrl_oe").exists())

            directory = scene.checkpoint_dir("mahdrl_ts")
            for role in ("vm", "host", "manager"):
                for filename in ("best_" + role + ".pth", role + "_final.pth"):
                    ckpt = torch.load(directory / filename, map_location="cpu", weights_only=True)
                    self.assertGreater(ckpt["updates"], 0)
                    self.assertEqual(ckpt["tau"], 0.005)
                    self.assertEqual(ckpt["optim"]["param_groups"][0]["lr"], 0.0001 if role == "manager" else 0.0003)
            metadata = json.loads((scene.log_dir("mahdrl_ts") / "measurement_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm"], "mahdrl_ts")
            self.assertEqual(metadata["agents"]["TA"]["settings"]["target_update_tau"], 0.005)
            self.assertEqual(metadata["agents"]["TA"]["optimizer_groups"]["optim"][0]["lr"], 0.0001)
            diagnostics = scene.log_dir("mahdrl_ts") / "f11_diagnostics.jsonl"
            record = json.loads(diagnostics.read_text(encoding="utf-8").splitlines()[-1])
            for name in ("Q_selected", "TD_target", "TD_error", "loss"):
                self.assertIn(name, record["numerics"]["TA"]["profiles"])
            self.assertGreater(record["learners"]["TA"]["terminal_transitions"], 0)

            # Evaluation also writes its measurement files into the temporary sandbox.
            def recorder(*args, **kwargs):
                return EvaluationRecorder(*args, **kwargs, directory=root / "measurements")

            evaluation_scene = replace(scene, mode="eval", seed_spec="700")
            with (patch.object(evaluator, "SCENARIO", evaluation_scene),
                  patch.object(evaluator, "PROJECT_DIR", str(root)),
                  patch.object(evaluator, "EnvHRLMix", TinyEpisode),
                  patch.object(evaluator, "D3QNAgent", side_effect=small_agent),
                  patch.object(evaluator, "EvaluationRecorder", side_effect=recorder), redirect_stdout(io.StringIO())):
                evaluator.main()
                run_files = list((root / "outputs/run").rglob("*.csv"))
                self.assertEqual(len(run_files), 1)
                self.assertTrue(run_files[0].name.startswith("eval_mahdrl_ts_"))
                episode = json.loads((root / "measurements/episodes/seed700.json").read_text(encoding="utf-8"))
                self.assertEqual(episode["algorithm"], "mahdrl_ts")
                self.assertGreater(episode["metrics"]["total_energy"], 0)
                self.assertEqual(episode["metrics"]["wf_completed_count"], 2)
                old_scene = replace(evaluation_scene, checkpoint_override=str(scene.checkpoint_dir("mahdrl")))
                with patch.object(evaluator, "SCENARIO", old_scene):
                    kwargs = evaluator.build_env_kwargs_hrl_mix(700)
                    with self.assertRaisesRegex(ValueError, "MAHDRL-TS requires"):
                        evaluator.build_three_layer_agents_and_load(kwargs)

            with patch.object(trainer, "SCENARIO", scene), patch.object(trainer, "ROOT_DIR", str(root)):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    trainer.train()


if __name__ == "__main__":
    unittest.main()
