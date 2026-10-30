"""Fixed-load resource presets: references, real learning, parity and strict lifecycle."""
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

from common.test_f18 import fixture_config
from common.test_mahdrl_as_fast_vaaware import tiny_agents
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from common.read_xml_opt_Tsize import workflow_instance_hash
from common.scenarios import ROOT
from algorithms.mahdrl_as_fast_vaaware import runtime
from algorithms.mahdrl_as_fast_vaaware.config import variant_metadata, implementation_metadata
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware.preview import prepare_decision
from algorithms.mahdrl_as_fast_vaaware.resources import precompute_reference
from algorithms.mahdrl_as_fast_vaaware.updates import UpdateClock
from algorithms.mahdrl_as_fast_vaaware_r100.scenario import ScoreScenario as R100, parse_scenario as parse100
from algorithms.mahdrl_as_fast_vaaware_r150.scenario import ScoreScenario as R150, parse_scenario as parse150


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="vaaware_resources_")
        cls.root = Path(cls.temp.name)
        cls.scales = cls.root/"scales.json"
        fixture_config(cls.scales)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, cls=R100, **kwargs):
        values = dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), output_root=str(self.root/self._testMethodName))
        return cls(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene, seed=1):
        with patch.object(runtime, "build_env_kwargs", self.small_kwargs(runtime.build_env_kwargs)):
            env = runtime.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_presets_keep_workload_references_dimensions_and_unique_paths(self):
        models, csvs = set(), set()
        for cls, hosts, vms in ((R100, 12, 100), (R150, 18, 150)):
            for deadline in ("Tight", "Medium", "Loose"):
                for seed in (1, 2):
                    scene = self.scene(cls, deadline=deadline, model_seed=seed)
                    self.assertEqual((scene.num_hosts, sum(scene.vms_per_host)), (hosts, vms))
                    expected = json.loads((ROOT/f"configs/mahdrl_as_{hosts}hosts.json").read_text())
                    self.assertEqual(list(scene.vms_per_host), expected["vms_per_host"])
                    self.assertEqual(scene.arrival_lambda, .03)
                    kw = runtime.build_env_kwargs(scene, 1)
                    self.assertEqual(kw["workflows_per_episode"], 50)
                    self.assertEqual(scene.cache_path, BaseScenario(resource="M").cache_path)
                    self.assertEqual(scene.normalization_kwargs(),
                        replace(BaseScenario(resource="M", deadline=deadline),
                            normalization_config=str(self.scales), mode="train").normalization_kwargs())
                    models.add(scene.checkpoint_dir())
                    for task in "SML":
                        target = replace(scene, mode="eval", task=task, train_resource=scene.resource,
                                         train_deadline=deadline, seed_spec="700-729")
                        self.assertEqual(target.checkpoint_dir(), scene.checkpoint_dir())
                        self.assertEqual(target.normalization_kwargs(), scene.normalization_kwargs())
                        self.assertIn(f"_from_{scene.source_tag}_", target.evaluation_csv().name)
                        csvs.add(target.evaluation_csv())
        self.assertEqual(len(models), 12)
        self.assertEqual(len(csvs), 36)
        for cls in (R100, R150):
            scene = self.scene(cls)
            control = replace(scene, va_preview=False, ha_update_period=1)
            self.assertNotEqual(scene.checkpoint_dir(), control.checkpoint_dir())
        base = BaseScenario()
        self.assertEqual(base.variant_metadata(), variant_metadata(base.mechanism))
        self.assertEqual(base.implementation_metadata(), implementation_metadata(base.mechanism))

    def test_cli_locks_source_and_target_and_rejects_load_changes(self):
        for parse, code in ((parse100, "R100"), (parse150, "R150")):
            for mode in ("train", "eval", "precompute"):
                scene = parse(mode, [])
                self.assertEqual(scene.resource, code)
            for args in (("--resource", "S"), ("--train-resource", "M"),
                         ("--resource-config", "topology.json"),
                         ("--workload-profile", "stress_a1_l006_ref003")):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse("train", list(args))

    def test_short_training_paths_and_flat_unique_scene_CSVs(self):
        models, csvs, filenames = set(), set(), set()
        for cls in (R100, R150):
            for source_task in "SML":
                for deadline in ("Tight", "Medium", "Loose"):
                    for seed in (1, 4294967295):
                        for preview, period in ((True, 4), (False, 1)):
                            source = self.scene(cls, task=source_task, deadline=deadline, model_seed=seed,
                                                va_preview=preview, ha_update_period=period)
                            self.assertLessEqual(len(source.checkpoint_dir().name), 25)
                            if seed == 1:
                                self.assertEqual(len(source.checkpoint_dir().name), 16)
                            self.assertEqual(source.log_dir().name, source.checkpoint_dir().name)
                            self.assertEqual(source.checkpoint_dir().parent.name, source.output_tag)
                            self.assertNotIn(source.resource.removeprefix("R"), source.checkpoint_dir().name[:3])
                            models.add(source.checkpoint_dir())
                            for task in "SML":
                                target = replace(source, mode="eval", task=task, train_task=source_task,
                                    train_resource=source.resource, train_deadline=deadline)
                                self.assertEqual(target.checkpoint_dir(), source.checkpoint_dir())
                                path = target.evaluation_csv()
                                self.assertEqual(path.parent, Path(source.output_root)/"outputs/run"/source.output_tag)
                                self.assertIn(f"_{target.target_tag}_from_{source.source_tag}_", path.name)
                                self.assertIn(f"_m{seed}_s700-729_", path.name)
                                self.assertNotIn(path, csvs)
                                self.assertNotIn(path.name, filenames)
                                csvs.add(path)
                                filenames.add(path.name)
        self.assertEqual(len(models), 72)
        self.assertEqual(len(csvs), 216)
        source = self.scene()
        old_location = self.root/"previous_checkpoint_location"
        target = replace(source, mode="eval", checkpoint_override=str(old_location))
        self.assertEqual(target.checkpoint_dir(), old_location)

    def test_strict_reference_validation_accepts_R50_and_rejects_other_topology(self):
        for cls in (R100, R150):
            scene = self.scene(cls, legacy_split=False)
            runtime.validate_inputs(scene)
            self.assertEqual(scene.describe()["physical_source_scene"], scene.source_tag)
            self.assertEqual(scene.describe()["normalization_reference_scene"], "SM_T")
            wrong = BaseScenario(resource="S").cache_path
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                runtime.validate_inputs(replace(scene, cache_file=str(wrong)))

    def test_reference_precompute_reuses_without_changes_and_generates_R50_only(self):
        scene = R100(mode="precompute", seed_spec="0-1000")
        original = scene.cache_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(precompute_reference(scene), scene.cache_path)
        self.assertEqual(original, scene.cache_path.read_bytes())
        output = self.root/"new_reference.json"
        partial = self.scene(R150, mode="precompute", seed_spec="0", cache_output=str(output))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(precompute_reference(partial), output)
        meta = json.loads(output.read_text())["meta"]
        self.assertEqual((meta["num_hosts"], sum(meta["vms_per_host"])), (6, 50))
        with self.assertRaises(FileExistsError):
            precompute_reference(partial)

    def test_fixed_workflows_and_deadlines_across_three_physical_topologies(self):
        reference = None
        for cls in (BaseScenario, R100, R150):
            scene = self.scene(cls, **({"resource": "M"} if cls is BaseScenario else {}))
            env, learners = self.env(scene), tiny_agents(scene)
            rows = []
            with redirect_stdout(io.StringIO()):
                result = runtime.run_episode(env, learners, assignment_log=rows.append)
            identities = [(wf.arrival_time, wf.deadline, workflow_instance_hash(wf.graph, wf.tasks))
                          for wf in env.workflows]
            if reference is None:
                reference = identities
            self.assertEqual(identities, reference)
            self.assertEqual(result["wf_completed_count"], 2)
            self.assertEqual(env.algorithm, scene.algorithm)
            self.assertEqual(env.variant_metadata()["model_version"], scene.model_version)
            for row in rows:
                self.assertGreaterEqual(row["global_legal_vm_count"], row["legal_vm_count"])
                self.assertLessEqual(row["global_legal_vm_count"], env.num_vms)

    def test_host_previews_and_real_CPU_updates_at_both_scales(self):
        for cls, count in ((R100, 12), (R150, 18)):
            scene = self.scene(cls)
            env, learners = self.env(scene), tiny_agents(scene)
            host, ok = env.get_host_state_for_next_assignment()
            self.assertTrue(ok)
            enhanced, states, _ = prepare_decision(env, learners["VA"], host, training=False)
            self.assertEqual(enhanced["obs"]["candidates"].shape, (count, 14))
            self.assertEqual(len(states), count)
            self.assertEqual({len(s["mask"]) for s in states.values()}, {8, 9})
            before = parameter_hash(learners)
            with redirect_stdout(io.StringIO()):
                result = runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
            self.assertNotEqual(before, parameter_hash(learners))
            self.assertEqual(result["wf_completed_count"], 2)
            for role in ("TA", "HA", "VA"):
                self.assertGreater(learners[role]._updates, 0)
            self.assertEqual(learners["HA"]._updates, learners["HA"].eligible_steps//4)

    def test_production_width_network_updates_on_CPU_and_available_CUDA(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        options = runtime.learning_kwargs
        for device in devices:
            for cls in (R100, R150):
                with self.subTest(device=device, resource=cls.resource):
                    scene = self.scene(cls, device=device)
                    env = self.env(scene)
                    with patch.object(runtime, "learning_kwargs", side_effect=lambda role, dev:
                                      dict(options(role, dev), batch_size=2, buffer_size=8)):
                        learners = runtime.build_agents(scene)
                    host, _ = env.get_host_state_for_next_assignment()
                    enhanced, states, _ = prepare_decision(env, learners["VA"], host, training=False)
                    vm = next(iter(states.values()))
                    observations = dict(TA=(env.get_manager_state(), env.get_manager_action_mask()),
                        HA=(enhanced["obs"], enhanced["mask"]), VA=(vm["obs"], vm["mask"]))
                    clock = UpdateClock(learners, training=True, warmup=0)
                    for _ in range(5):
                        for role, (obs, mask) in observations.items():
                            clock.remember(role, obs, mask, int(np.flatnonzero(mask)[0]), -.1, obs, mask, 0.)
                        clock.barrier()
                    self.assertEqual([learners[r]._updates for r in ("TA", "HA", "VA")], [4, 1, 4])
                    del learners

    def test_real_train_reload_frozen_evaluate_and_cross_branch_rejection(self):
        original = runtime.build_env_kwargs
        for cls in (R100, R150):
            scene = self.scene(cls)
            with patch.object(runtime, "build_env_kwargs", self.small_kwargs(original)), \
                 patch.object(runtime, "build_agents", side_effect=tiny_agents), redirect_stdout(io.StringIO()):
                runtime.train(scene)
                evaluation = replace(scene, mode="eval", seed_spec="700")
                loaded = runtime.load_agents(evaluation)
                before = parameter_hash(loaded)
                counters = {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()}
                runtime.run_episode(self.env(evaluation, 700), loaded)
                self.assertEqual(before, parameter_hash(loaded))
                self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()})
                runtime.evaluate(evaluation)
                with self.assertRaises(FileExistsError):
                    runtime.evaluate(evaluation)
                with self.assertRaises(FileExistsError):
                    runtime.train(scene)
                for other in (BaseScenario, R150 if cls is R100 else R100):
                    incompatible = tiny_agents(self.scene(other))
                    for role, stem in (("TA", "manager"), ("HA", "host"), ("VA", "vm")):
                        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                            incompatible[role].load(str(scene.checkpoint_dir()/f"best_{stem}.pth"))
                for changed in (replace(scene, model_seed=2), replace(scene, ha_update_period=1)):
                    with self.assertRaises(ValueError):
                        tiny_agents(changed)["VA"].load(str(scene.checkpoint_dir()/"best_vm.pth"))
                final = self.root/f"final_{scene.resource}"
                final.mkdir()
                for stem in ("manager", "host", "vm"):
                    shutil.copy2(scene.checkpoint_dir()/f"{stem}_final.pth", final/f"best_{stem}.pth")
                with self.assertRaisesRegex(ValueError, "best checkpoints"):
                    runtime.load_agents(replace(evaluation, checkpoint_override=str(final)))
            with evaluation.evaluation_csv().open(encoding="utf8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["resource"], scene.resource)
            self.assertEqual(row["model_version"], scene.model_version)
            self.assertEqual(int(row["physical_num_vms"]), sum(scene.vms_per_host))
            self.assertEqual(row["deadline_reference_resource"], "M")
            self.assertEqual(row["normalization_reference_scene"], "SM_T")


    def test_nohup_commands_match_CLI_and_stop_on_failure(self):
        git = shutil.which("git")
        candidates = ([Path(git).parent.parent/"bin/bash.exe"] if git else [])
        bash_on_path = shutil.which("bash")
        if bash_on_path:
            candidates.append(Path(bash_on_path))
        bash = next((str(p) for p in candidates if p.is_file()), None)
        if not bash:
            self.skipTest("Bash not installed")
        script = ROOT/"docs/mahdrl_as_fast_vaaware_resources_commands.sh"
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
        train_mahdrl_as_fast_vaaware_resource "$res" "$ddl"
        wait
        eval_mahdrl_as_fast_vaaware_resource_all "$res" "$ddl"
        wait
    done
done
train_mahdrl_as_fast_vaaware_r100 Tight; wait
train_mahdrl_as_fast_vaaware_r150 Tight; wait
eval_mahdrl_as_fast_vaaware_r100_all Tight; wait
eval_mahdrl_as_fast_vaaware_r150_all Tight; wait
train_mahdrl_as_fast_vaaware_r100 Tight 2 off 1; wait
eval_mahdrl_as_fast_vaaware_r150_all Tight 'M L' 2 off 1; wait
if train_mahdrl_as_fast_vaaware_resource R50 Tight; then exit 92; fi
if eval_mahdrl_as_fast_vaaware_resource_all R100 Tight ' '; then exit 93; fi
if eval_mahdrl_as_fast_vaaware_resource_all R100 Tight X; then exit 94; fi
if train_mahdrl_as_fast_vaaware_resource R100 Tight -1; then exit 95; fi
if train_mahdrl_as_fast_vaaware_resource R100 Tight 1 on 0; then exit 96; fi
exit 0
'''
        env = dict(os.environ, CAPTURE=capture.as_posix())
        subprocess.run([bash, "--noprofile", "--norc", "-c", body], cwd=directory, env=env,
                       check=True, capture_output=True, text=True)
        calls = [shlex.split(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(calls), 35)
        for args in calls:
            self.assertNotIn("--resource", args)
            self.assertNotIn("--train-resource", args)
            self.assertEqual(args[args.index("--device")+1], "cuda")
            self.assertEqual(args[args.index("--diagnostics-every")+1], "100")
            self.assertEqual(args[args.index("--profile-mode")+1], "sampled")
            self.assertEqual(args[args.index("--profile-every")+1], "100")
        self.assertIn("--no-va-preview", calls[-1])
        self.assertEqual(calls[-1][calls[-1].index("--model-seed")+1], "2")
        failed = directory/"failed"
        failed.mkdir()
        failure_capture = failed/"calls.txt"
        failure = source + r'''
python() { printf '%s\n' "$*" >> "$CAPTURE"; [[ " $* " != *" --task M "* ]]; }
nohup() { "$@"; }
export -f python
eval_mahdrl_as_fast_vaaware_resource_all R100 Tight
wait
'''
        subprocess.run([bash, "--noprofile", "--norc", "-c", failure], cwd=failed,
                       env=dict(env, CAPTURE=failure_capture.as_posix()), check=True, capture_output=True)
        failed_calls = failure_capture.read_text().splitlines()
        self.assertEqual(len(failed_calls), 2)
        self.assertNotIn("--task L", "\n".join(failed_calls))


if __name__ == "__main__":
    unittest.main()
