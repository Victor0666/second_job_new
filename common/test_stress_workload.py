"""Opt-in workload isolation, paired physical inputs and real CPU roundtrips."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.marl.scenario import MARLScenario, parse_scenario as parse_marl
from algorithms.marl.identity import evaluation_csv as marl_csv, evaluation_directory as marl_details
from algorithms.pd3qn_fe.scenario import FEScenario as PDScene, parse_scenario as parse_pd
from algorithms.pd3qn_fe import runtime as pd
from algorithms.irws_r_fe_idle.scenario import FEScenario as IRScene, parse_scenario as parse_ir
from algorithms.irws_r_fe_idle import runner as ir
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario, parse_scenario as parse_ma
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as ma
from common.f22_instances import instance_identity
from common.read_xml_opt_Tsize import poisson_arrival_times, workflow_instance_hash, dax_template_id
from common.scenarios import ROOT

PROFILE = 'stress_a1_l006_ref003'
SCENES = (MARLScenario, PDScene, IRScene, ScoreScenario)


def checkpoint(scene):
    return ir.output_dir(scene, 'ckpts') if isinstance(scene, IRScene) else scene.checkpoint_dir()


def csv_path(scene):
    if isinstance(scene, MARLScenario):
        return marl_csv(scene, ROOT)
    if isinstance(scene, PDScene):
        return scene.evaluation_csv
    if isinstance(scene, IRScene):
        return ir.output_dir(scene, 'run') / ir.artifact_name(scene, 'eval')
    return scene.evaluation_csv()


class StressWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_cli_opt_in_and_invalid_combinations(self):
        for parse in (parse_marl, parse_pd, parse_ir, parse_ma):
            for mode in ('train', 'eval'):
                out = io.StringIO()
                args = ['--dry-run', '--workload-profile', PROFILE]
                with redirect_stdout(out), self.assertRaises(SystemExit) as stopped:
                    if parse is parse_marl:
                        with patch('sys.argv', ['test', *args]):
                            parse(mode)
                    else:
                        parse(mode, argv=args)
                self.assertEqual(stopped.exception.code, 0)
                data = json.loads(out.getvalue())
                self.assertEqual(data['workload']['arrival_lambda'], .06)
                self.assertEqual(data['workload']['reference_arrival_lambda'], .03)
                self.assertEqual(data['deadline_choices'], [1., 1.])
                self.assertEqual(data['experiment_protocol']['workload'], data['workload'])
        for cls in SCENES:
            for invalid in (dict(deadline='Loose'), dict(train_deadline='Medium'),
                            dict(legacy_split=True), dict(mode='precompute')):
                with self.subTest(cls=cls, invalid=invalid), self.assertRaises(ValueError):
                    cls(workload_profile=PROFILE, **invalid)

    def test_paths_scales_and_protocol_isolation(self):
        for cls in SCENES:
            for resource in 'SML':
                old = cls(mode='train', resource=resource)
                self.assertEqual(old.describe(), replace(old, workload_profile='standard').describe())
                source = replace(old, workload_profile=PROFILE)
                self.assertNotEqual(checkpoint(source), checkpoint(old))
                self.assertEqual(source.normalization_kwargs(), old.normalization_kwargs())
                self.assertEqual(source.protocol().metadata(), old.protocol().metadata())
                self.assertNotIn('workload', old.reproducibility_metadata())
                self.assertEqual(source.reproducibility_metadata()['workload']['profile'], PROFILE)
                for task in 'SML':
                    target = replace(source, mode='eval', task=task, train_resource=resource,
                                     train_deadline='Tight')
                    self.assertEqual(checkpoint(source), checkpoint(target))
                    self.assertNotEqual(csv_path(target), csv_path(replace(target, workload_profile='standard')))
                    self.assertLess(len(str(checkpoint(source) / 'final_irws_r_fe_idle.pth')), 260)
                    if cls is MARLScenario:
                        details = marl_details(target, ROOT)
                    elif cls is IRScene:
                        details = ir.evaluation_directory(target)
                    elif cls is PDScene:
                        details = target.evaluation_dir
                    else:
                        details = target.evaluation_dir()
                    self.assertLess(len(str(details / 'measurements/episodes/seed700.json')), 260)

    def snapshot(self, cls, task, resource, seed):
        scene = cls(task=task, resource=resource, workload_profile=PROFILE)
        if cls is MARLScenario:
            from algorithms.marl import evaluate
            with patch.object(evaluate, 'SCENARIO', scene):
                kwargs = evaluate.build_env_kwargs(str(scene.cache_path))
            kwargs.update(random_seed=seed, print_deadlines=False, dump_deadlines=False)
            with patch('environments.marl.poisson_arrival_times', wraps=poisson_arrival_times) as generate:
                env = evaluate.CloudWorkflowEnv_MARL(**kwargs)
                env.reset(seed=seed)
                generate.assert_called_once_with(.06, n_max=50, seed=seed)
        elif cls is PDScene:
            env = pd.CloudWorkflowEnv_PD3QN_Cost(**dict(pd.env_kwargs(scene), random_seed=seed))
            env.reset(seed=seed)
        elif cls is IRScene:
            env = ir.PaperCloudEnv(**ir.env_kwargs(scene, seed))
            env.reset(seed=seed)
        else:
            env = ma.build_env(scene, seed)
            env.reset(seed=seed)
        try:
            env.current_time = float(env.arrival_times[-1])
            method = '_add_workflows_up_to_current_time' if cls is IRScene else '_add_workflow_if_arrived'
            getattr(env, method)()
            self.assertEqual(len(env.workflows), 50)
            record = next(row for row in json.loads(scene.cache_path.read_text(encoding='utf-8'))['data'] if row['seed'] == seed)
            np.testing.assert_array_equal(env.arrival_times, poisson_arrival_times(.06, n_max=50, seed=seed))
            np.testing.assert_allclose(env.arrival_times, np.array(record['arrival_times']) * .5, rtol=0, atol=1e-10)
            np.testing.assert_allclose([w.deadline - w.arrival_time for w in env.workflows],
                                       record['wf_makespans'], rtol=0, atol=1e-10)
            hashes = [workflow_instance_hash(w.graph, w.tasks) for w in env.workflows]
            self.assertEqual(hashes, record['wf_instance_hashes'])
            names = [env.wf_dax_name[i] if cls is MARLScenario else w.dax_path for i, w in enumerate(env.workflows)]
            self.assertEqual(list(map(dax_template_id, names)), list(map(dax_template_id, record['wf_dax_names'])))
            return instance_identity(env, scene, seed)['content_hash'], np.array(env.task_baseline_finish)
        finally:
            env.close()

    def test_fifty_workflows_match_all_four_with_unscaled_reference(self):
        for task, resource, seed in [(t, r, 700) for t in 'SML' for r in 'SML'] + [('S', 'S', 0), ('S', 'S', 600)]:
            expected = self.snapshot(MARLScenario, task, resource, seed)
            for cls in SCENES[1:]:
                with self.subTest(cls=cls, task=task, resource=resource, seed=seed):
                    actual = self.snapshot(cls, task, resource, seed)
                    self.assertEqual(actual[0], expected[0])
                    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-10)

    def test_marl_standard_keeps_cache_arrivals(self):
        from algorithms.marl import evaluate
        with patch.object(evaluate, 'SCENARIO', MARLScenario()):
            kwargs = evaluate.build_env_kwargs(str(MARLScenario().cache_path))
        kwargs.update(dump_deadlines=False, print_deadlines=False)
        with patch('environments.marl.poisson_arrival_times', side_effect=AssertionError('standard arrival changed')):
            env = evaluate.CloudWorkflowEnv_MARL(**kwargs)
            try:
                env.reset(seed=700)
                record = env._deadline_cache_by_seed[700]
                np.testing.assert_array_equal(env.arrival_times, record['arrival_times'])
            finally:
                env.close()

    def test_marl_stress_train_validate_save_load_evaluate(self):
        from common.smoke_marl_q import run_smoke
        result = run_smoke(workload_profile=PROFILE)
        self.assertTrue(result['checkpoint_reload'])
        self.assertTrue(result['evaluation_no_update'])

    def test_pd_stress_train_validate_save_load_evaluate(self):
        from algorithms.pd3qn_fe.train import train
        from algorithms.pd3qn_fe.evaluate import evaluate
        from algorithms.pd3qn.config import agent_kwargs
        with tempfile.TemporaryDirectory(prefix='stress_pd_') as directory:
            root = Path(directory)

            class LocalScene(PDScene):
                def checkpoint_dir(self, algorithm='pd3qn_fe'): return root / 'ckpts'
                def log_dir(self, algorithm='pd3qn_fe'): return root / 'logs'
                @property
                def evaluation_root(self): return root / 'results'
                @property
                def evaluation_dir(self): return root / 'eval'

            scene = LocalScene(mode='train', episodes=1, workload_profile=PROFILE, paper_options={'device': 'cpu'})
            settings = dict(agent_kwargs(scene), hidden_dims=(16, 16), batch_size=8, buffer_size=128)
            with patch.object(pd, 'agent_kwargs', return_value=settings), redirect_stdout(io.StringIO()):
                agent, _ = train(scene)
                self.assertGreater(agent._updates, 0)
                target = replace(scene, mode='eval', seed_spec='700')
                rows, _ = evaluate(target)
                self.assertEqual(rows[0]['wf_count'], 50)
                self.assertIn('wf_avg_tardiness', rows[0])
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    pd.load_best_checkpoint(agent, scene.checkpoint_dir() / 'best_pd3qn_fe.pth',
                                            replace(target, workload_profile='standard'))

    def test_ir_stress_train_resume_load_evaluate(self):
        from algorithms.irws_r_fe_idle.config import PaperConfig, BEST_CHECKPOINT, FINAL_CHECKPOINT
        with tempfile.TemporaryDirectory(prefix='stress_ir_') as directory, redirect_stdout(io.StringIO()):
            scene = IRScene(mode='train', episodes=1, workload_profile=PROFILE)
            kwargs = dict(ir.env_kwargs(scene, 0), workflows_per_episode=2)
            learner, models = ir.train(scene, config=PaperConfig(actor_hidden=(8,), critic_hidden=(8,)),
                                       kwargs=kwargs, out=Path(directory) / 'ckpts')
            self.assertGreater(learner.updates, 0)
            ir.load_checkpoint(models / BEST_CHECKPOINT, scene)
            ir.train(scene, kwargs=kwargs, resume=models / FINAL_CHECKPOINT)
            target = replace(scene, mode='eval', seed_spec='700')
            rows, _ = ir.evaluate(learner, target, [700], kwargs=kwargs)
            self.assertEqual(rows[0]['wf_count'], 2)
            self.assertIn('wf_avg_tardiness', rows[0])
            for training, filename in ((False, BEST_CHECKPOINT), (True, FINAL_CHECKPOINT)):
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    ir.load_checkpoint(models / filename, replace(scene, workload_profile='standard'),
                                       restore_training=training)

    def test_ma_stress_train_validate_save_load_evaluate(self):
        from common.test_mahdrl_as_fast import tiny_agents
        with tempfile.TemporaryDirectory(prefix='stress_ma_') as directory:
            scene = ScoreScenario(mode='train', episodes=1, workload_profile=PROFILE, output_root=directory,
                                  device='cpu', profile_mode='off')
            original = ma.build_env_kwargs

            def small(scene, seed):
                return dict(original(scene, seed), workflows_per_episode=2)

            with patch.object(ma, 'build_env_kwargs', side_effect=small), \
                    patch.object(ma, 'build_agents', side_effect=tiny_agents), redirect_stdout(io.StringIO()):
                agents = ma.train(scene)
                self.assertTrue(all(a._updates > 0 for a in agents.values()))
                target = replace(scene, mode='eval', seed_spec='700')
                ma.load_agents(target)
                ma.evaluate(target)
                wrong = replace(target, workload_profile='standard', checkpoint_override=str(scene.checkpoint_dir()))
                with self.assertRaises(ValueError):
                    ma.load_agents(wrong)
            report = json.loads((target.evaluation_dir() / 'measurements/episodes/seed700.json').read_text(encoding='utf-8'))
            self.assertEqual(report['provenance']['workload']['profile'], PROFILE)


if __name__ == '__main__':
    unittest.main()
