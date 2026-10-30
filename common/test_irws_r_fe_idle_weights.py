"""Direct IDLE reward variants: physics, SR092 selection, lifecycle and outputs."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from algorithms.irws_r_fe_idle import runner as base
from algorithms.irws_r_fe_idle_010 import runner as r010, config as c010, scenario as s010
from algorithms.irws_r_fe_idle_090 import runner as r090, config as c090, scenario as s090
from algorithms.irws_r_fe_r090_idle import runner as previous_r090_idle
from common.experiment_protocol import protocol_binding
from common.read_xml_opt_Tsize import Task
from common.test_irws_paper import fixture, synthetic_env

VARIANTS = ((r010, c010, s010), (r090, c090, s090))


class IdleWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='irws_idle_weights_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scales, scene = fixture(self.root)
        protocol = Path(scene.experiment_protocol)
        data = json.loads(protocol.read_text())
        data.update(train=[32, 33], episodes=2)
        protocol.write_text(json.dumps(data))
        calibration = json.loads(self.scales.read_text())
        calibration['generation'] = dict(**protocol_binding(protocol), calibration_seeds={'SS_T': [32, 33]})
        self.scales.write_text(json.dumps(calibration))
        self.scene_values = dict(vars(scene), episodes=2)

    def scene(self, scenario_module):
        class SmallScenario(scenario_module.FEScenario):
            @property
            def vms_per_host(self):
                return (2, 1)
        return SmallScenario(**self.scene_values)

    @contextmanager
    def environment(self, module, one_vm=False):
        def make(**kwargs):
            extra = dict(num_hosts=1, vms_per_host=(1,), workflows_per_episode=1, pc_tiers=(1.,)) if one_vm else {}
            return module.PaperCloudEnv(**dict(kwargs, **module.COMMON_REWARD, **extra))
        with synthetic_env(self.scales, make) as env:
            if one_vm:
                graph = nx.empty_graph(2, create_using=nx.DiGraph)
                tasks = [Task(task_id=i, workload_mi=12.3, parents=[], children=[]) for i in range(2)]
                with patch('environments.irws.load_workflow_from_dax',
                           side_effect=lambda *a, **kw: (graph.copy(), deepcopy(tasks))), \
                     patch('environments.irws.poisson_arrival_times', return_value=np.array([0.])):
                    yield env
            else:
                yield env

    def test_direct_idle_source_and_sr092_learning_are_preserved(self):
        root = Path(__file__).resolve().parents[1] / 'algorithms'
        for runner, config, scenario in VARIANTS:
            with self.subTest(algorithm=config.ALGORITHM):
                for name in ('environment.py', 'learner.py', 'networks.py', 'numerics.py', 'selection.py'):
                    self.assertEqual((root/'irws_r_fe_idle'/name).read_bytes(),
                                     (root/config.ALGORITHM/name).read_bytes())
                self.assertEqual(config.PaperConfig().as_dict(), base.PaperConfig().as_dict())
                self.assertEqual(config.IMPLEMENTATION, base.IMPLEMENTATION)
                self.assertEqual(runner.selection_metadata(), base.selection_metadata())
                self.assertEqual(runner.selection_metadata()['workflow_success_threshold'], .92)
                self.assertEqual(config.SCHEDULING_PROTOCOL, 'idle_only_periodic_v1')
                self.assertEqual(runner.env_kwargs(self.scene(scenario), 32)['alpha_delay_vm'], config.COMMON_REWARD['alpha_delay_vm'])
                snapshot = json.loads((root/config.ALGORITHM/'source_snapshot_sha256.json').read_text())
                self.assertTrue(all(key.startswith('algorithms/irws_r_fe_idle/') for key in snapshot))
        self.assertEqual(c010.COMMON_REWARD['alpha_delay_vm'], .1)
        self.assertEqual(c090.COMMON_REWARD['alpha_delay_vm'], .9)
        self.assertEqual(base.COMMON_REWARD['alpha_delay_vm'], .75)

    def test_fixed_actions_preserve_idle_physics_and_change_only_external_reward(self):
        trajectories, rewards = [], []
        for module in (base, r010, r090):
            alpha = module.COMMON_REWARD['alpha_delay_vm']
            with self.environment(module) as env:
                env.reset(seed=32)
                env.task_baseline_finish = [float(value)-1e4 for value in env.task_baseline_finish]
                steps = []
                while not env.done_flag:
                    if not env.can_dispatch():
                        env.finish_phase_and_advance()
                        continue
                    env.select_task(env.get_ready_task_candidates()['gids'][0])
                    mask = env.queue_mask()
                    vm = int(np.flatnonzero(mask)[0])
                    self.assertIsNone(env.vm_running[vm])
                    reward, _, info = env.paper_step(vm)
                    self.assertAlmostEqual(reward, alpha*info['r_delay']-(1-alpha)*.001*info['delta_energy_total'], places=12)
                    self.assertAlmostEqual(info['outer_cost'], -reward, places=12)
                    self.assertFalse(info['queued'])
                    self.assertFalse(any(env.vm_queues.values()))
                    steps.append((reward, info['fusion_weight'], mask.tolist()))
                trajectories.append((env.task_placement.copy(), env.task_start_time.copy(),
                                     env.task_end_time.copy(), env.records.copy(), env.total_energy))
                rewards.append(steps)
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertEqual(trajectories[0], trajectories[2])
        for previous, current in zip(rewards, rewards[1:]):
            self.assertTrue(any(a[0] != b[0] for a, b in zip(previous, current)))
            self.assertEqual([x[1:] for x in previous], [x[1:] for x in current])

    def test_busy_vm_rejected_without_mutation_and_global_ready_waits_for_tick(self):
        for runner, config, _ in VARIANTS:
            with self.subTest(algorithm=config.ALGORITHM), self.environment(runner, one_vm=True) as env:
                env.reset(seed=32)
                env.select_task(0)
                _, _, first = env.paper_step(0)
                self.assertAlmostEqual(first['end'], 12.3)
                self.assertEqual(env.task_state[1], 'Ready')
                self.assertNotIn(1, env.task_placement)
                env.select_task(1)
                before = (env.state_revision, env.total_energy, env.task_state.copy(), env.task_placement.copy(), len(env.records))
                with self.assertRaisesRegex(ValueError, 'requires an idle VM'):
                    env.paper_step(0)
                self.assertEqual(before, (env.state_revision, env.total_energy, env.task_state.copy(), env.task_placement.copy(), len(env.records)))
                self.assertFalse(env.can_dispatch())
                env.finish_phase_and_advance()
                self.assertEqual(env.current_time, 15.)
                _, _, second = env.paper_step(0)
                self.assertEqual(second['start'], 15.)
                self.assertFalse(second['queued'])
                env.finish_phase_and_advance()
                self.assertTrue(env.done_flag)
                self.assertFalse(any(env.vm_queues.values()))

    def test_train_resume_frozen_evaluate_and_cross_variant_models_rejected(self):
        for runner, config, scenario in VARIANTS:
            with self.subTest(algorithm=config.ALGORITHM), self.environment(runner) as template, \
                 patch.object(runner, 'PaperCloudEnv', side_effect=lambda **kw: deepcopy(template)), \
                 redirect_stdout(io.StringIO()):
                scene = self.scene(scenario)
                directory = self.root/config.ALGORITHM
                kwargs = runner.env_kwargs(scene, 32)
                _, models = runner.train(replace(scene, episodes=1), kwargs=kwargs, out=directory)
                trained, resumed = runner.train(scene, kwargs=kwargs, resume=models/config.FINAL_CHECKPOINT)
                self.assertEqual(resumed, models)
                self.assertEqual(trained.steps, 24)
                learner, saved = runner.load_checkpoint(models/config.BEST_CHECKPOINT, scene)
                self.assertEqual(saved['algorithm'], config.ALGORITHM)
                self.assertEqual(saved['reward_settings'], config.COMMON_REWARD)
                self.assertEqual(saved['checkpoint_selection']['workflow_success_threshold'], .92)
                self.assertEqual(saved['scheduling_protocol'], config.SCHEDULING_PROTOCOL)
                for other in (base, previous_r090_idle, r090 if runner is r010 else r010):
                    with self.assertRaisesRegex(ValueError, 'algorithm/version mismatch'):
                        other.load_checkpoint(models/config.BEST_CHECKPOINT, scene)
                for field, value, message in (
                    ('reward_settings', base.COMMON_REWARD, 'reward objective mismatch'),
                    ('scheduling_protocol', 'queue_periodic', 'scheduling protocol mismatch'),
                    ('checkpoint_selection', dict(runner.selection_metadata(), workflow_success_threshold=.99), 'selection rule mismatch'),
                ):
                    wrong = self.root/'wrong.pth'
                    torch.save(dict(saved, **{field: value}), wrong)
                    with self.assertRaisesRegex(ValueError, message):
                        runner.load_checkpoint(wrong, scene)
                before = deepcopy(learner.actor.state_dict())
                counters = learner.steps, learner.updates
                runner.evaluate(learner, scene, [632], 'validation', kwargs=kwargs)
                self.assertEqual(counters, (learner.steps, learner.updates))
                for key, value in before.items():
                    torch.testing.assert_close(value, learner.actor.state_dict()[key], rtol=0, atol=0)
                evaluation = replace(scene, mode='eval', seed_spec='2', checkpoint_override=str(models))
                root = self.root/(config.ALGORITHM+'_eval')
                with patch.object(runner, 'parse_scenario', return_value=evaluation), \
                     patch.object(runner, 'output_dir', return_value=root):
                    runner.evaluate_main()
                    with self.assertRaises(FileExistsError):
                        runner.evaluate_main()
                csv_path = root/runner.artifact_name(evaluation, 'eval')
                report = json.loads((root/runner.artifact_name(evaluation, 'summary')).read_text())
                self.assertTrue(csv_path.is_file())
                self.assertEqual(Path(report['evaluation_csv']), csv_path)
                self.assertEqual(report['reward_settings'], config.COMMON_REWARD)
                with self.assertRaises(FileExistsError):
                    runner.train(scene, out=models)

    def test_nine_sources_27_flat_csvs_and_cli_identity_for_each_variant(self):
        for runner, config, scenario in VARIANTS:
            csvs, sources = set(), set()
            for resource in 'SML':
                for deadline in ('Tight', 'Medium', 'Loose'):
                    source = scenario.FEScenario(resource=resource, deadline=deadline)
                    checkpoint = runner.output_dir(source, 'ckpts')
                    sources.add(checkpoint)
                    self.assertLessEqual(len(checkpoint.name), 30)
                    for task in 'SML':
                        target = replace(source, mode='eval', task=task, train_task='S', train_resource=resource, train_deadline=deadline)
                        self.assertEqual(checkpoint, runner.output_dir(target, 'ckpts'))
                        csv = runner.output_dir(target, 'run')/runner.artifact_name(target, 'eval')
                        csvs.add(csv)
                        self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_', csv.name)
                        self.assertLess(len(str(runner.evaluation_directory(target)/'measurements/episodes/seed700.json')), 260)
                        output = io.StringIO()
                        args = ['--task', task, '--resource', resource, '--deadline', deadline,
                                '--train-resource', resource, '--train-deadline', deadline, '--dry-run']
                        with redirect_stdout(output), self.assertRaises(SystemExit) as stop:
                            scenario.parse_scenario('eval', argv=args)
                        self.assertEqual(stop.exception.code, 0)
                        info = json.loads(output.getvalue())
                        self.assertEqual(info['algorithm'], config.ALGORITHM)
                        self.assertEqual(info['reward_settings'], config.COMMON_REWARD)
                        self.assertFalse(info['queue']['waiting_enabled'])
                        self.assertEqual(info['checkpoint_selection']['workflow_success_threshold'], .92)
            self.assertEqual(len(sources), 9)
            self.assertEqual(len(csvs), 27)
            self.assertEqual(len({csv.parent for csv in csvs}), 1)
            scene = self.scene(scenario)
            for old in (base, previous_r090_idle, r090 if runner is r010 else r010):
                self.assertNotEqual(runner.output_dir(scene, 'ckpts'), old.output_dir(scene, 'ckpts'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_cuda_update_and_validation_for_both_weights(self):
        for runner, config, scenario in VARIANTS:
            with self.subTest(algorithm=config.ALGORITHM), self.environment(runner) as template, \
                 patch.object(runner, 'PaperCloudEnv', side_effect=lambda **kw: deepcopy(template)):
                scene = self.scene(scenario)
                learner = runner.PaperLearner(3, config.PaperConfig(device='cuda'))
                result = runner.run_episode(template, learner, 32, True)
                self.assertEqual(sum(row['sample_presentations'] for row in result['updates']), 12)
                self.assertEqual(result['objective']['settings'], config.COMMON_REWARD)
                rows, _ = runner.evaluate(learner, scene, [632], 'validation', kwargs=runner.env_kwargs(scene, 32))
                self.assertEqual(rows[0]['wf_count'], 2)
                self.assertFalse(any(template.vm_queues.values()))
                self.assertTrue(all(torch.isfinite(p).all() for p in learner.actor.parameters()))


if __name__ == '__main__':
    unittest.main()
