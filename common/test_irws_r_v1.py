"""IRWS-R identity, independent artifacts and unchanged scheduling behavior."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.irws_r_v1 import runner
from algorithms.irws_r_v1.config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, BEST_CHECKPOINT, FINAL_CHECKPOINT, PaperConfig
from algorithms.irws_restored.config import PaperConfig as SourceConfig, MODEL_VERSION as SOURCE_VERSION
from algorithms.irws_restored import runner as source_runner
from common.test_irws_paper import fixture, synthetic_env


class IRWSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='irws_r_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.scene = fixture(self.root)

    def test_source_algorithms_and_learning_settings_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'algorithms/irws_r/copied_from_sha256.json').read_text())
        for file, digest in manifest.items():
            if file == 'common/scenarios.py':
                # Algorithm registration may grow; shared physical settings must stay fixed.
                from common.scenarios import Scenario
                self.assertEqual(Scenario().vms_per_host, (9, 8, 8))
                self.assertEqual(Scenario(resource='M').vms_per_host, (9, 9, 8, 8, 8, 8))
                self.assertEqual(Scenario(resource='L').vms_per_host, (9, 9, 9, 8, 8, 8, 8, 8, 8))
                self.assertEqual(Scenario().deadline_probs, (.8, .2))
                self.assertEqual(Scenario().protocol().train, tuple(range(600)))
                self.assertEqual(Scenario().validation_seeds(), [600])
                self.assertEqual(Scenario().seeds, tuple(range(700, 730)))
                continue
            raw = (root/file).read_bytes()
            lf = raw.replace(b'\r\n', b'\n')
            self.assertIn(digest, [hashlib.sha256(data).hexdigest()
                                  for data in (raw, lf, lf.replace(b'\n', b'\r\n'))], file)
        for file in ('environment.py', 'learner.py', 'networks.py'):
            self.assertEqual((root/'algorithms/irws_r_v1'/file).read_bytes(),
                             (root/'algorithms/irws_restored'/file).read_bytes())
        self.assertEqual(PaperConfig().as_dict(), SourceConfig().as_dict())

    def test_output_names_and_model_seeds_are_separate(self):
        for kind in ('ckpts', 'run'):
            current = runner.output_dir(self.scene, kind)
            self.assertIn(ALGORITHM, current.parts)
            self.assertIn(MODEL_VERSION, current.parts)
            self.assertEqual(current.name, 'model1')
            self.assertNotEqual(current, source_runner.output_dir(self.scene, kind))
            self.assertNotEqual(current, runner.output_dir(replace(self.scene, model_seed=2), kind))
        for kind in ('train', 'eval', 'summary'):
            self.assertIn('_irws_r_', runner.artifact_name(self.scene, kind))
        self.assertIn(ALGORITHM, BEST_CHECKPOINT)
        self.assertIn(ALGORITHM, FINAL_CHECKPOINT)

    def test_same_seed_produces_same_schedule_as_source(self):
        with synthetic_env(self.path, runner.PaperCloudEnv) as current, synthetic_env(
                self.path, source_runner.PaperCloudEnv) as previous:
            runner.seed_everything(31)
            learner = runner.PaperLearner(current.num_vms, PaperConfig())
            runner.seed_everything(31)
            old_learner = source_runner.PaperLearner(previous.num_vms, SourceConfig())
            actual = runner.run_episode(current, learner, 32, False)
            expected = source_runner.run_episode(previous, old_learner, 32, False)
            self.assertEqual(actual['metrics'], expected['metrics'])
            self.assertEqual(current.task_placement, previous.task_placement)
            np.testing.assert_array_equal(current.task_end_time, previous.task_end_time)

    def test_existing_training_or_evaluation_output_is_preserved(self):
        existing = self.root/'existing'
        existing.mkdir()
        sentinel = existing/'old_result.txt'
        sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            runner.train(self.scene, out=existing)
        out_root = self.root/'run'
        target = out_root/(self.scene.task+self.scene.resource+'_'+self.scene.deadline)
        target.mkdir(parents=True)
        old_csv = target/runner.artifact_name(self.scene, 'eval')
        old_csv.write_text('old csv', encoding='utf-8')
        with patch.object(runner, 'parse_scenario', return_value=self.scene), patch.object(
                runner, 'output_dir', side_effect=lambda scene, kind:out_root if kind=='run' else self.root/'ckpts'), patch.object(
                runner, 'load_checkpoint') as load:
            with self.assertRaisesRegex(FileExistsError, 'IRWS-R evaluation artifacts'):
                runner.evaluate_main()
            load.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')
        self.assertEqual(old_csv.read_text(encoding='utf-8'), 'old csv')

    def test_train_load_evaluate_and_resume_use_irws_r_identity(self):
        with synthetic_env(self.path, runner.PaperCloudEnv) as template:
            def output(scene, kind):
                return self.root/'verification'/kind/f'model{scene.model_seed}'
            with patch.object(runner, 'PaperCloudEnv', side_effect=lambda **kw:deepcopy(template)), patch.object(
                    runner, 'output_dir', side_effect=output), redirect_stdout(io.StringIO()):
                runner.seed_everything(23)
                _, models = runner.train(self.scene, kwargs=runner.env_kwargs(self.scene,32), out=self.root/'models')
                _, obj = runner.load_checkpoint(models/BEST_CHECKPOINT, self.scene)
                self.assertEqual(obj['algorithm'], ALGORITHM)
                self.assertEqual(obj['display_name'], DISPLAY_NAME)
                self.assertEqual(obj['model_version'], MODEL_VERSION)
                self.assertTrue((models/FINAL_CHECKPOINT).is_file())
                training = json.loads((models/runner.artifact_name(self.scene,'train')).read_text().splitlines()[0])
                self.assertEqual(training['algorithm'], ALGORITHM)
                measurement = json.loads((models/'logs/training_measurements.jsonl').read_text().splitlines()[0])
                self.assertEqual(measurement['algorithm'], ALGORITHM)
                for version in (SOURCE_VERSION, 'irws_restored_queue_v1', 'irws_cloud_energy_tardiness_v2'):
                    torch.save(dict(obj,model_version=version), self.root/'old.pth')
                    with self.assertRaisesRegex(ValueError,'algorithm/version mismatch'):
                        runner.load_checkpoint(self.root/'old.pth',self.scene)
                with self.assertRaisesRegex(ValueError,'model seed mismatch'):
                    runner.load_checkpoint(models/BEST_CHECKPOINT,replace(self.scene,model_seed=2))
                evaluation = replace(self.scene,mode='eval',seed_spec='2',checkpoint_override=str(models))
                with patch.object(runner,'parse_scenario',return_value=evaluation):
                    runner.evaluate_main()
                    with self.assertRaises(FileExistsError):runner.evaluate_main()
                result_dir = output(evaluation,'run')/'SS_Tight'
                with (result_dir/runner.artifact_name(evaluation,'eval')).open(encoding='utf-8',newline='') as f:
                    row = next(csv.DictReader(f))
                self.assertEqual(row['algorithm'],ALGORITHM)
                self.assertEqual(row['display_name'],DISPLAY_NAME)
                self.assertEqual(row['model_version'],MODEL_VERSION)
                summary = json.loads((result_dir/runner.artifact_name(evaluation,'summary')).read_text())
                self.assertEqual(summary['algorithm'],ALGORITHM)
                self.assertEqual(Path(summary['source_checkpoint']).name,BEST_CHECKPOINT)
                details = json.loads((result_dir/'measurements/episodes/seed2.json').read_text())
                self.assertEqual(details['algorithm'],ALGORITHM)
                self.assertTrue(details['checkpoints'])
                self.assertEqual(Path(details['checkpoints'][0]['path']).name,BEST_CHECKPOINT)
                with self.assertRaises(FileExistsError):runner.train(self.scene,out=models)
                runner.train(self.scene,out=models,resume=models/FINAL_CHECKPOINT)


if __name__ == '__main__':
    unittest.main()
