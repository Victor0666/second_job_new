"""Matched resource TL ablations: global actions, reference parity and lifecycle."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import csv
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common.scenarios import ROOT
from common.test_f18 import fixture_config
from common.test_mahdrl_as_fast_vaaware_tl import tiny_agents
from common.test_mahdrl_as_fast_vaaware import tiny_agents as full_agents
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_vaaware import runtime as full_runtime
from algorithms.mahdrl_as_fast_vaaware_tl import runtime
from algorithms.mahdrl_as_fast_vaaware_tl.config import variant_metadata, implementation_metadata
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import ScoreScenario as BaseTL
from algorithms.mahdrl_as_fast_vaaware_tl.updates import UpdateClock
from algorithms.mahdrl_as_fast_vaaware_r100.scenario import ScoreScenario as Full100
from algorithms.mahdrl_as_fast_vaaware_r150.scenario import ScoreScenario as Full150
from algorithms.mahdrl_as_fast_vaaware_r100_tl.scenario import ScoreScenario as R100, parse_scenario as parse100
from algorithms.mahdrl_as_fast_vaaware_r150_tl.scenario import ScoreScenario as R150, parse_scenario as parse150


class ResourceTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="vaaware_resource_tl_")
        cls.root = Path(cls.temp.name)
        cls.scales = cls.root/"scales.json"
        fixture_config(cls.scales)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, cls=R100, **kwargs):
        return cls(**dict(dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), output_root=str(self.root/self._testMethodName)), **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene, module=runtime, seed=1):
        with patch.object(module, "build_env_kwargs", self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_matched_references_short_paths_and_flat_unique_filenames(self):
        models, names = set(), set()
        for cls, full, hosts, vms in ((R100, Full100, 12, 100), (R150, Full150, 18, 150)):
            for task in "SML":
                for deadline in ("Tight", "Medium", "Loose"):
                    for seed in (1, 4294967295):
                        scene = self.scene(cls, task=task, deadline=deadline, model_seed=seed)
                        parent = self.scene(full, task=task, deadline=deadline, model_seed=seed)
                        self.assertEqual((scene.num_hosts, sum(scene.vms_per_host)), (hosts, vms))
                        for method in ("resource_metadata", "normalization_kwargs", "reference_identity"):
                            self.assertEqual(getattr(scene, method)(), getattr(parent, method)())
                        self.assertEqual(runtime.build_env_kwargs(scene, 1), full_runtime.build_env_kwargs(parent, 1))
                        self.assertEqual(runtime.build_env_kwargs(scene, 1)["workflows_per_episode"], 50)
                        self.assertEqual(scene.arrival_lambda, .03)
                        self.assertEqual(scene.variant_metadata()["learned_roles"], ["TA", "VA"])
                        self.assertNotEqual(scene.checkpoint_dir(), parent.checkpoint_dir())
                        self.assertEqual(len(scene.model_name()), 16 if seed == 1 else 25)
                        self.assertEqual(scene.log_dir().name, scene.model_name())
                        models.add(scene.checkpoint_dir())
                        for target_task in "SML":
                            target = replace(scene, mode="eval", task=target_task, train_task=task,
                                train_resource=scene.resource, train_deadline=deadline)
                            self.assertEqual(target.checkpoint_dir(), scene.checkpoint_dir())
                            self.assertEqual(target.normalization_kwargs(), scene.normalization_kwargs())
                            path = target.evaluation_csv()
                            self.assertEqual(path.parent, Path(scene.output_root)/"outputs/run"/scene.output_tag)
                            self.assertIn(f"_{target.target_tag}_from_{scene.source_tag}_m{seed}_s700-729_", path.name)
                            self.assertNotIn("p1h4", path.name)
                            self.assertNotIn(path.name, names)
                            names.add(path.name)
        self.assertEqual((len(models), len(names)), (36, 108))
        self.assertEqual(BaseTL().variant_metadata(), variant_metadata())
        self.assertEqual(BaseTL().implementation_metadata(), implementation_metadata())

    def test_parser_and_cache_enforce_fixed_scale_and_protocol(self):
        for cls, parse in ((R100, parse100), (R150, parse150)):
            scene = self.scene(cls, legacy_split=False)
            runtime.validate_inputs(scene)
            wrong = BaseTL(resource="S").cache_path
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                runtime.validate_inputs(replace(scene, cache_file=str(wrong)))
            for mode in ("train", "eval", "precompute"):
                with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as e:
                    parse(mode, ["--dry-run"])
                self.assertEqual(e.exception.code, 0)
                data = json.loads(output.getvalue())
                self.assertEqual(data["algorithm"], scene.algorithm)
                self.assertEqual(data["variant"]["learned_roles"], ["TA", "VA"])
                self.assertEqual(data["variant"]["resource_experiment"], scene.resource_metadata())
            for args in (("--resource", "M"), ("--train-resource", "R50"),
                         ("--resource-config", "topology.json"), ("--va-preview",),
                         ("--ha-update-period", "4"), ("--workload-profile", "stress_a1_l006_ref003")):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                    parse("train", list(args))
                self.assertEqual(e.exception.code, 2)

    def test_global_mapping_busy_masks_and_exact_full_physics_reward_parity(self):
        for cls, full, count in ((R100, Full100, 100), (R150, Full150, 150)):
            env = self.env(self.scene(cls))
            busy = env.host_to_vm_indices[env.host_ids[0]]
            env.vm_available_at[busy] = env.current_time+100
            with patch("environments.mahdrl_oe_ge_apf_score_fast.apf_candidates", side_effect=AssertionError("APF called")):
                vm, ok = env.get_vm_state_for_next_assignment()
            self.assertTrue(ok)
            self.assertEqual(vm["obs"]["candidates"].shape, (count, 14))
            self.assertEqual(vm["obs"]["context"].shape, (23,))
            expected = np.flatnonzero(env._vm_idle_mask() & (env.vm_available_at <= env.current_time+1e-12))
            np.testing.assert_array_equal(np.flatnonzero(vm["mask"]), expected)
            self.assertTrue(all(vm["mask"][i] == 0 for i in busy))
            with self.assertRaises(ValueError): env.vm_assign_global(busy[0])
            env, parent = self.env(self.scene(cls)), self.env(self.scene(full), full_runtime)
            self.assertEqual(env.variant_metadata()["model_version"], cls.model_version)
            with redirect_stdout(io.StringIO()):
                while not env.done_flag:
                    while True:
                        vm, oka = env.get_vm_state_for_next_assignment()
                        _, okb = parent.get_host_state_for_next_assignment()
                        self.assertEqual(oka, okb)
                        if not oka: break
                        self.assertEqual(env._cur_tid, parent._cur_tid)
                        j = int(np.flatnonzero(vm["mask"])[-1])
                        hid = env.vms[env.vm_ids[j]].host_id
                        slot = env.host_to_vm_indices[hid].index(j)
                        parent.host_select(parent.host_ids.index(hid))
                        _, rv, info = parent.vm_assign(slot)
                        tl_rv, tl_info = env.vm_assign_global(j)
                        self.assertEqual(rv, tl_rv)
                        self.assertEqual(tl_info["vm_id"], env.vm_ids[j])
                        for key in ("assignment_energy", "r_delay", "r_energy_total", "host_id"):
                            self.assertEqual(info[key], tl_info[key])
                        self.assertEqual(env._records, parent._records)
                    self.assertEqual(env.finish_phase_and_advance(), parent.finish_phase_and_advance())
                    if not env.done_flag:
                        env.apply_manager_delta(np.zeros(5)); parent.apply_manager_delta(np.zeros(5))
            self.assertEqual(env.total_energy, parent.total_energy)

    def test_real_global_learning_and_frozen_historical_replay_at_both_scales(self):
        for cls in (R100, R150):
            scene = self.scene(cls)
            learners, env, rows = tiny_agents(scene), self.env(scene), []
            before = parameter_hash(learners)
            with patch.object(env, "host_select", side_effect=AssertionError("HA called")), redirect_stdout(io.StringIO()):
                result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append)
            self.assertEqual(result["wf_completed_count"], 2)
            self.assertNotEqual(before, parameter_hash(learners))
            self.assertEqual(set(learners), {"TA", "VA"})
            self.assertTrue(all(a._updates > 0 for a in learners.values()))
            self.assertEqual(result["update_clock"]["transitions"]["VA"], len(rows))
            self.assertTrue(learners["VA"].buffer[-1][-1])
            for transition in learners["VA"].buffer:
                self.assertEqual(transition[0]["candidates"].shape, (sum(scene.vms_per_host), 14))
                self.assertFalse(transition[0]["candidates"].flags.writeable)
            for key in ("physical_energy_balance", "lower_energy_balance", "VA_reward_formula_balance"):
                self.assertAlmostEqual(result["reward_audit"][key], 0., places=6)
            self.assertEqual(result["candidate_diagnostics"]["global_vm"]["max"], max(r["legal_vm_count"] for r in rows))

    def test_production_width_CPU_and_available_CUDA_gradient_updates(self):
        options = runtime.learning_kwargs
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for cls in (R100, R150):
                with self.subTest(device=device, resource=cls.resource):
                    scene = self.scene(cls, device=device)
                    with patch.object(runtime, "learning_kwargs", side_effect=lambda role, dev:
                                      dict(options(role, dev), batch_size=2, buffer_size=8)):
                        learners = runtime.build_agents(scene)
                    env = self.env(scene)
                    vm, _ = env.get_vm_state_for_next_assignment()
                    observations = dict(TA=(env.get_manager_state(), env.get_manager_action_mask()), VA=(vm["obs"], vm["mask"]))
                    before = parameter_hash(learners)
                    clock = UpdateClock(learners, training=True, warmup=0)
                    for _ in range(5):
                        for role, (obs, mask) in observations.items():
                            clock.remember(role, obs, mask, int(np.flatnonzero(mask)[0]), -.1, obs, mask, 0.)
                        clock.barrier()
                    self.assertEqual({r: a._updates for r,a in learners.items()}, dict(TA=4, VA=4))
                    self.assertNotEqual(before, parameter_hash(learners))

    def test_real_train_reload_frozen_evaluation_and_checkpoint_isolation(self):
        original = runtime.build_env_kwargs
        for cls, full in ((R100, Full100), (R150, Full150)):
            scene = self.scene(cls)
            with patch.object(runtime, "build_env_kwargs", self.small_kwargs(original)), \
                 patch.object(runtime, "build_agents", side_effect=tiny_agents), redirect_stdout(io.StringIO()):
                trained = runtime.train(scene)
                self.assertTrue(all(a._updates > 0 for a in trained.values()))
                evaluation = replace(scene, mode="eval", seed_spec="700")
                loaded = runtime.load_agents(evaluation)
                before = parameter_hash(loaded)
                counters = {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()}
                with patch.object(runtime, "load_agents", return_value=loaded): runtime.evaluate(evaluation)
                self.assertEqual(before, parameter_hash(loaded))
                self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()})
                with self.assertRaises(FileExistsError): runtime.evaluate(evaluation)
                with self.assertRaises(FileExistsError): runtime.train(scene)
                for other in (BaseTL, R150 if cls is R100 else R100):
                    for role, stem in (("TA", "manager"), ("VA", "vm")):
                        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                            tiny_agents(self.scene(other))[role].load(str(scene.checkpoint_dir()/f"best_{stem}.pth"))
                with self.assertRaises(ValueError):
                    full_agents(self.scene(full))["TA"].load(str(scene.checkpoint_dir()/"best_manager.pth"))
                with self.assertRaises(ValueError):
                    tiny_agents(replace(scene, model_seed=2))["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"))
                final = self.root/f"final_{scene.resource}"
                final.mkdir()
                for stem in ("manager", "vm"):
                    shutil.copy2(scene.checkpoint_dir()/f"{stem}_final.pth", final/f"best_{stem}.pth")
                with self.assertRaisesRegex(ValueError, "best checkpoints"):
                    runtime.load_agents(replace(evaluation, checkpoint_override=str(final)))
                path = scene.checkpoint_dir()/"best_vm.pth"
                payload = torch.load(path, weights_only=True)
                payload["normalization"]["checkpoint_group"] = "different-group"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "different saved groups"): runtime.load_agents(evaluation)
            self.assertEqual(sorted(p.name for p in scene.checkpoint_dir().glob("*.pth")),
                ["best_manager.pth", "best_vm.pth", "manager_final.pth", "vm_final.pth"])
            with evaluation.evaluation_csv().open(encoding="utf8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["resource"], scene.resource)
            self.assertEqual(row["vm_selection_scope"], "global")
            self.assertEqual(row["model_version"], scene.model_version)
            self.assertEqual(int(row["physical_num_vms"]), sum(scene.vms_per_host))
            self.assertEqual(row["normalization_reference_scene"], "SM_T")

    def test_nohup_commands_and_failure_stop(self):
        git = shutil.which("git")
        candidates = ([Path(git).parent.parent/"bin/bash.exe"] if git else [])
        if shutil.which("bash"): candidates.append(Path(shutil.which("bash")))
        bash = next((str(p) for p in candidates if p.is_file()), None)
        if not bash: self.skipTest("Bash unavailable")
        script = ROOT/"docs/mahdrl_as_fast_vaaware_resources_tl_commands.sh"
        subprocess.run([bash, "--noprofile", "--norc", "-n", str(script)], check=True, capture_output=True)
        directory = self.root/self._testMethodName
        directory.mkdir()
        capture = directory/"calls.txt"
        source = f"source {shlex.quote(script.as_posix())}\n"
        body = source + r'''
[[ ! -e outputs ]] || exit 91
python() { printf '%s\n' "$*" >> "$CAPTURE"; }
nohup() { "$@"; }
export -f python
for res in R100 R150; do
    for ddl in Tight Medium Loose; do
        train_mahdrl_as_fast_vaaware_resource_tl "$res" "$ddl"; wait
        eval_mahdrl_as_fast_vaaware_resource_tl_all "$res" "$ddl"; wait
    done
done
train_mahdrl_as_fast_vaaware_r100_tl Tight; wait
train_mahdrl_as_fast_vaaware_r150_tl Tight; wait
eval_mahdrl_as_fast_vaaware_r100_tl_all Tight; wait
eval_mahdrl_as_fast_vaaware_r150_tl_all Tight; wait
train_mahdrl_as_fast_vaaware_r100_tl Tight 2; wait
eval_mahdrl_as_fast_vaaware_r150_tl_all Tight 'M L' 2; wait
if train_mahdrl_as_fast_vaaware_resource_tl R50 Tight; then exit 92; fi
if eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight ' '; then exit 93; fi
if eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight X; then exit 94; fi
if train_mahdrl_as_fast_vaaware_resource_tl R100 Tight -1; then exit 95; fi
if train_mahdrl_as_fast_vaaware_resource_tl R100 Tight 1 on 4; then exit 96; fi
if eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight 'S M L' 1 on; then exit 97; fi
exit 0
'''
        env = dict(os.environ, CAPTURE=capture.as_posix())
        subprocess.run([bash, "--noprofile", "--norc", "-c", body], cwd=directory, env=env, check=True, capture_output=True)
        calls = [shlex.split(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(calls), 35)
        for args in calls:
            self.assertNotIn("--va-preview", args)
            self.assertNotIn("--ha-update-period", args)
            self.assertIn("_tl/", args[2])
            self.assertEqual(args[args.index("--device")+1], "cuda")
            parser = parse100 if "r100_tl/" in args[2] else parse150
            parser("train" if args[2].endswith("train.py") else "eval", args[3:])
        self.assertEqual(calls[-1][calls[-1].index("--model-seed")+1], "2")
        failed = directory/"failed"
        failed.mkdir()
        failure_capture = failed/"calls.txt"
        failure = source + r'''
python() { printf '%s\n' "$*" >> "$CAPTURE"; [[ " $* " != *" --task M "* ]]; }
nohup() { "$@"; }
export -f python
eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight
wait
'''
        subprocess.run([bash, "--noprofile", "--norc", "-c", failure], cwd=failed,
            env=dict(env, CAPTURE=failure_capture.as_posix()), check=True, capture_output=True)
        failed_calls = failure_capture.read_text().splitlines()
        self.assertEqual(len(failed_calls), 2)
        self.assertNotIn("--task L", "\n".join(failed_calls))


if __name__ == "__main__":
    unittest.main()
