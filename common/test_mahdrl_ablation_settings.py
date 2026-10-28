"""The eleven public ablations share raw legal actions and strict SR>92% selection."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import mahdrl_ablation as selection
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from algorithms.mahdrl_oe_ge_apf_score_fast import selection as full_selection


VARIANTS = ["nta", "nha", "nva", "tl", "sync_task", "sync_phase", "ertf", "edf", "cp", "r090", "r010_nopf"]


def modules(variant):
    package = ("algorithms.mahdrl_oe_ge_score_fast_r010_nopf" if variant == "r010_nopf"
               else "algorithms.mahdrl_oe_ge_apf_score_fast_" + variant)
    runtime = "sync" if variant.startswith("sync_") else "fixed_ta" if variant in ("ertf", "edf", "cp") else variant
    runtime_package = (package if variant == "r010_nopf"
                       else "algorithms.mahdrl_oe_ge_apf_score_fast_" + runtime)
    return importlib.import_module(package + ".scenario").ScoreScenario, importlib.import_module(runtime_package + ".runtime")


def row(successes, count=50, energy=100.):
    return dict(wf_count=count, wf_success_count=successes,
                wf_success_rate=successes/count, total_energy=energy)


def raw_physics(env):
    """Pair physical/reward comparisons under the same unfiltered action space."""
    from types import MethodType
    from environments.mahdrl_as_fast_noapf import CloudWorkflowEnv_VMAgents as RawEnv
    for name in ("_apf_snapshot", "host_select", "vm_assign"):
        setattr(env, name, MethodType(getattr(RawEnv, name), env))
    return env


def assert_flat_csv(test, scene):
    path = scene.evaluation_csv()
    test.assertTrue(path.is_file())
    test.assertEqual(path.parent, scene.evaluation_root)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    test.assertEqual(len(rows), len(scene.seeds))
    for value in rows:
        test.assertEqual(value["scene"], scene.target_tag)
        test.assertEqual(value["source_scene"], scene.source_tag)
        test.assertEqual(value["host_apf_enabled"], "False")
        test.assertEqual(value["vm_apf_enabled"], "False")


class AblationSettingsTests(unittest.TestCase):
    def test_strict_boundary_aggregate_gate_energy_fallback_and_ties(self):
        below = selection.validation_selection([row(45, energy=1.)])
        equal = selection.validation_selection([row(46, energy=1.)])
        above = selection.validation_selection([row(47, energy=1000.)])
        self.assertFalse(below["feasible"])
        self.assertFalse(equal["feasible"])
        self.assertTrue(above["feasible"])
        self.assertLess(selection.selection_key(above), selection.selection_key(equal))
        self.assertLess(selection.selection_key(equal), selection.selection_key(below))
        cheaper = selection.validation_selection([row(47, energy=900.)])
        self.assertLess(selection.selection_key(cheaper), selection.selection_key(above))
        tie = selection.validation_selection([row(48, energy=900.)])
        self.assertLess(selection.selection_key(tie), selection.selection_key(cheaper))
        # Aggregate 92/100 is excluded even though one individual instance is perfect.
        self.assertFalse(selection.validation_selection([row(50), row(42)])["feasible"])
        self.assertTrue(selection.validation_selection([row(50), row(43)])["feasible"])
        self.assertEqual(selection.selection_metadata()["workflow_success_comparison"], "strictly_greater")
        self.assertEqual(full_selection.WORKFLOW_SUCCESS_THRESHOLD, .99)
        self.assertFalse(full_selection.validation_selection([row(47)])["feasible"])
        with self.assertRaises(ValueError):
            selection.validation_selection([dict(row(47), wf_success_rate=.92)])

    def test_297_final_CSV_names_are_unique_flat_and_source_models_are_short(self):
        names = set()
        maximum = 0
        for variant in VARIANTS:
            cls, runtime = modules(variant)
            parents = set()
            for resource in "SML":
                for deadline in ("Tight", "Medium", "Loose"):
                    source = cls(mode="train", resource=resource, deadline=deadline)
                    self.assertLessEqual(len(source.model_name()), 32)
                    self.assertLess(len(str(source.checkpoint_dir()/"best_manager.pth")), 240)
                    for task in "SML":
                        target = replace(source, mode="eval", task=task, train_task="S",
                                         train_resource=resource, train_deadline=deadline)
                        self.assertEqual(source.checkpoint_dir(), target.checkpoint_dir())
                        path = target.evaluation_csv()
                        parents.add(path.parent)
                        self.assertNotIn(path.name, names)
                        names.add(path.name)
                        self.assertIn(target.target_tag + "_from_" + target.source_tag, path.name)
                        self.assertIn("noapf_srgt092", path.name)
                        maximum = max(maximum, len(str(path)))
                        self.assertLess(len(str(target.evaluation_details_dir()/"measurements/episodes/seed700.json")), 240)
                        description = target.path_settings()[1]
                        self.assertFalse(description["host_apf_enabled"])
                        self.assertFalse(description["vm_apf_enabled"])
                        self.assertEqual(target.selection_protocol()["checkpoint_selection"]["workflow_success_threshold"], .92)
            self.assertEqual(parents, {source.evaluation_root})
        self.assertEqual(len(names), 297)
        self.assertLess(maximum, 240)

    def test_paths_distinguish_config_source_model_and_noncontiguous_seed_sets(self):
        cls, _ = modules("nta")
        target = cls(mode="eval", task="M", resource="S", deadline="Loose",
                     train_resource="S", train_deadline="Tight")
        for changed in (replace(target, model_seed=2), replace(target, train_deadline="Medium"),
                        replace(target, seed_spec="700-728"), replace(target, seed_spec="700,702,729")):
            self.assertNotEqual(target.evaluation_csv(), changed.evaluation_csv())
            self.assertEqual(target.evaluation_csv().parent, changed.evaluation_csv().parent)
        self.assertNotEqual(replace(target, seed_spec="700,702,729").evaluation_csv(),
                            replace(target, seed_spec="700,703,729").evaluation_csv())
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"one.json", Path(directory)/"two.json"
            first.write_text("{}")
            second.write_text("{}")
            scene = replace(target, normalization_config=str(first))
            self.assertEqual(scene.model_name(), replace(scene, normalization_config=str(second)).model_name())
            second.write_text('{"changed":true}')
            self.assertNotEqual(scene.model_name(), replace(scene, normalization_config=str(second)).model_name())

    def test_all_11_environments_retain_slow_overdue_choices_but_reject_busy_slots(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="ablation_noapf_") as directory:
            root = Path(directory)
            scales, cache = root/"scales.json", root/"cache.json"
            fixture_config(scales)
            with redirect_stdout(io.StringIO()):
                precompute(Scenario(mode="precompute", seed_spec="0", cache_output=str(cache)))
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    cls, runtime = modules(variant)
                    scene = cls(mode="train", episodes=1, legacy_split=True,
                        normalization_config=str(scales), cache_file=str(cache),
                        output_root=str(root), device="cpu")
                    kwargs = runtime.build_env_kwargs
                    with patch.object(runtime, "build_env_kwargs", side_effect=lambda s, seed: dict(
                            kwargs(s, seed), workflows_per_episode=1, max_ready_tasks=100)):
                        env = runtime.build_env(scene, 0)
                    with redirect_stdout(io.StringIO()):
                        env.reset()
                    env.vm_available_at[0] = env.current_time+100.
                    env.task_baseline_finish = [env.current_time-1.] * len(env.task_baseline_finish)
                    owners = {j: hi for hi, hid in enumerate(env.host_ids) for j in env.host_to_vm_indices[hid]}
                    with patch("environments.mahdrl_oe_ge_apf_score_fast.apf_candidates", side_effect=AssertionError("APF executed")), \
                         patch("environments.mahdrl_oe_ge_apf_score.apf_candidates", side_effect=AssertionError("APF executed")), \
                         patch.object(env, "_task_duration", side_effect=lambda tid, j: [1., 100., 10000.][owners[j]]):
                        if variant == "tl":
                            state, ok = env.get_vm_state_for_next_assignment()
                        else:
                            host, ok = env.get_host_state_for_next_assignment()
                            self.assertTrue(ok)
                            np.testing.assert_array_equal(host["mask"], np.ones(env.num_hosts))
                            env.host_select(env.num_hosts-1)
                            state, ok = env.get_vm_state_for_current_task()
                        self.assertTrue(ok)
                        snapshot = env._apf_snapshot(env._cur_tid)
                        retained = set().union(*snapshot["vms"].values())
                        self.assertEqual(retained, set(range(env.num_vms)) - {0})
                        self.assertIsNone(snapshot["r_H"])
                        self.assertIsNone(snapshot["r_V"])
                        self.assertEqual(env.variant_metadata()["host_apf_enabled"], False)
                        if variant == "tl":
                            self.assertEqual(state["mask"][0], 0.)
                            with self.assertRaises(ValueError):
                                env.vm_assign_global(0)
                            env.vm_assign_global(env.num_vms-1)
                        else:
                            with self.assertRaises(ValueError):
                                env.vm_assign(len(state["mask"]))
                            env.vm_assign(int(np.flatnonzero(state["mask"])[-1]))
                        self.assertFalse(env.apf_decisions[-1]["host_apf_enabled"])
                        self.assertFalse(env.apf_decisions[-1]["vm_apf_enabled"])


if __name__ == "__main__":
    unittest.main()
