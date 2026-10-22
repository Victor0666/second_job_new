"""Regression checks for actual entry-point configurations, without training or loading weights."""

import argparse
import contextlib
import importlib
import io
import itertools
import os
from common.read_xml_opt_Tsize import DATA_MODEL
from contextlib import nullcontext
from pathlib import Path
import unittest
from unittest.mock import patch

from common.scenarios import ROOT, MODEL_NAMES, Scenario, parse_seeds
from common.precompute_deadlines import build_env_kwargs as precompute_kwargs

EVAL_BUILDERS = {
    "mahdrl": "build_env_kwargs_hrl_mix",
    "mahdrl_oe": "build_env_kwargs_hrl_mix",
    "mahdrl_ts": "build_env_kwargs_hrl_mix",
    "mahdrl_ub": "build_env_kwargs_hrl_mix",
    "mahdrl_ge": "build_env_kwargs_hrl_mix",
    "mahdrl_pf": "build_env_kwargs_hrl_mix",
    "mahdrl_oe_ge": "build_env_kwargs_hrl_mix",
    "mahdrl_oe_ge_pf": "build_env_kwargs_hrl_mix",
    "mahdrl_oe_pf": "build_env_kwargs_hrl_mix",
    "ablation_hostrand": "build_env_kwargs_hostrand",
    "ablation_mgrrand": "build_env_kwargs_mgrrand",
    "ablation_vmrand": "build_env_kwargs_vm_random",
    "ablation_nohost": "build_env_kwargs_nohost",
    "ablation_tl2": "build_env_kwargs_nohost",
    "irws": "build_env_kwargs_irws",
}
ENVIRONMENTS = dict.fromkeys(MODEL_NAMES, "CloudWorkflowEnv_VMAgents")
ENVIRONMENTS.update(
    marl="CloudWorkflowEnv_MARL", irws="IRWSEnvRouteA", pd3qn="CloudWorkflowEnv_PD3QN_Cost"
)


class CapturedConfiguration(BaseException):
    def __init__(self, kwargs):
        self.kwargs = kwargs


def capture_environment(*args, **kwargs):
    raise CapturedConfiguration(kwargs)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        # Intercept only new cache existence checks: configuration tests never load caches.
        real_exists = os.path.exists
        mocked = patch("os.path.exists", side_effect=lambda path: DATA_MODEL in str(path) or real_exists(path))
        mocked.start()
        self.addCleanup(mocked.stop)

    def assert_configuration(self, kwargs, scenario):
        self.assertEqual(kwargs["num_hosts"], scenario.num_hosts)
        self.assertEqual(tuple(kwargs["vms_per_host"]), scenario.vms_per_host)
        self.assertEqual(len(kwargs["vms_per_host"]), kwargs["num_hosts"])
        self.assertEqual(kwargs["dax_paths"], scenario.dax_paths())
        self.assertEqual(Path(kwargs["deadline_cache_path"]).resolve(), scenario.cache_path)
        self.assertEqual(kwargs["workflows_per_episode"], 50)
        if "deadline_alpha_small_prob" in kwargs:
            self.assertEqual(kwargs["deadline_alpha_small_prob"], scenario.small_probability)
        else:
            probs = kwargs.get("deadline_alpha_probs", kwargs.get("alpha_deadline_probs"))
            self.assertIsNotNone(probs)
            for actual, expected in zip(probs, scenario.deadline_probs):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(scenario.cache_path.parent.name, DATA_MODEL)
        self.assertNotEqual(scenario.cache_path, scenario.legacy_cache_path)
        self.assertTrue(all(Path(path).is_file() for path in scenario.dax_paths()))

    def test_all_training_configurations(self):
        # Intercept the first environment constructor before agents, logs or models are created.
        for algorithm in MODEL_NAMES:
            entry = "ablation_nohost" if algorithm == "ablation_tl2" else algorithm
            module = importlib.import_module(f"algorithms.{entry}.train")
            for task, resource, deadline in itertools.product(
                "SML", "SML", ("Tight", "Medium", "Loose")
            ):
                scenario = Scenario(
                    task=task, resource=resource, deadline=deadline, mode="train", episodes=1
                )
                with self.subTest(
                    algorithm=algorithm, task=task, resource=resource, deadline=deadline
                ):
                    with (patch.object(module, "_load_cache_seeds", return_value=list(range(1001)))
                          if algorithm == "marl" else nullcontext()), patch.object(module, "SCENARIO", scenario), patch.object(
                        module, ENVIRONMENTS[algorithm], side_effect=capture_environment
                    ), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(CapturedConfiguration) as captured:
                            if algorithm in ("ablation_nohost", "ablation_tl2"):
                                module.train(scenario, algorithm=algorithm, env_cls=capture_environment)
                            else:
                                module.train()
                    self.assert_configuration(captured.exception.kwargs, scenario)
                    self.assertEqual(captured.exception.kwargs["random_seed"], 0)

    def test_all_evaluation_configurations(self):
        for algorithm in [*MODEL_NAMES, "ertf_fifs"]:
            entry = "ablation_nohost" if algorithm == "ablation_tl2" else algorithm
            module = importlib.import_module(f"algorithms.{entry}.evaluate")
            for task, resource, deadline in itertools.product(
                "SML", "SML", ("Tight", "Medium", "Loose")
            ):
                scenario = Scenario(task=task, resource=resource, deadline=deadline)
                with self.subTest(
                    algorithm=algorithm, task=task, resource=resource, deadline=deadline
                ), patch.object(module, "SCENARIO", scenario), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    if algorithm in EVAL_BUILDERS:
                        kwargs = getattr(module, EVAL_BUILDERS[algorithm])(random_seed=2)
                    elif algorithm == "marl":
                        kwargs = module.build_env_kwargs(str(scenario.cache_path))
                    elif algorithm == "pd3qn":
                        with patch.object(
                            module, ENVIRONMENTS[algorithm], side_effect=capture_environment
                        ):
                            with self.assertRaises(CapturedConfiguration) as captured:
                                module.build_agent_and_env_kwargs()
                        kwargs = captured.exception.kwargs
                    else:

                        def capture_run(env, kwargs, seed):
                            raise CapturedConfiguration(kwargs)

                        with patch.object(module, "run_one_episode", side_effect=capture_run):
                            with self.assertRaises(CapturedConfiguration) as captured:
                                module.main()
                        kwargs = captured.exception.kwargs
                    self.assert_configuration(kwargs, scenario)

    def test_precomputation_and_model_selection(self):
        for task, resource in itertools.product("SML", "SML"):
            scenario = Scenario(task=task, resource=resource, mode="precompute", seed_spec="2")
            kwargs = precompute_kwargs(scenario)
            self.assertEqual(kwargs["deadline_mode"], "none")
            self.assertEqual(kwargs["vms_per_host"], scenario.vms_per_host)
            self.assertEqual(kwargs["dax_paths"], scenario.dax_paths())
            self.assertNotEqual(scenario.precompute_path, scenario.cache_path)
        for algorithm in MODEL_NAMES:
            training = Scenario(task="M", resource="L", deadline="Medium", mode="train")
            evaluation = Scenario(task="L", resource="L", deadline="Medium", train_task="M")
            self.assertEqual(
                training.checkpoint_dir(algorithm), evaluation.checkpoint_dir(algorithm)
            )
        for task, expected in [("S", 5), ("M", 50), ("L", 50)]:
            self.assertEqual(len(Scenario(task=task).dax_paths()), expected)
        self.assertEqual(Scenario(resource="L", deadline="Medium", mode="train").num_hosts, 9)
        self.assertEqual(sum(Scenario(resource="L").vms_per_host), 75)
        self.assertEqual(
            Scenario(mode="precompute", seed_spec="0-1000").precompute_path, Scenario().cache_path
        )

    def test_seed_validation(self):
        self.assertEqual(parse_seeds("2-4"), (2, 3, 4))
        self.assertEqual(parse_seeds("2,5,8"), (2, 5, 8))
        for value in ["", "5-2", "-1", "2,2", "bad"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_seeds(value)


if __name__ == "__main__":
    unittest.main()
