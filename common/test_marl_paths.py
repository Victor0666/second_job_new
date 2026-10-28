"""Compact model paths, flat scene-specific CSVs and isolated evaluation details."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from algorithms.marl.evaluate import protect_evaluation_outputs
from algorithms.marl.identity import MODEL_VERSION, evaluation_csv, evaluation_directory
from algorithms.marl.scenario import MARLScenario
from common.scenarios import ROOT, Scenario


class MARLPathTests(unittest.TestCase):
    def test_nine_short_training_paths_and_27_distinct_csvs_in_one_folder(self):
        files = []
        checkpoints = []
        for resource in 'SML':
            for deadline in ('Tight', 'Medium', 'Loose'):
                source = MARLScenario(mode='train', resource=resource, deadline=deadline)
                checkpoint = source.checkpoint_dir()
                checkpoints.append(checkpoint)
                self.assertEqual(checkpoint.parent, ROOT / 'outputs/ckpts' / MODEL_VERSION)
                self.assertEqual(source.log_dir().parent, ROOT / 'outputs/logs' / MODEL_VERSION)
                self.assertEqual(checkpoint.name, source.log_dir().name)
                self.assertLessEqual(len(checkpoint.name), 32)
                self.assertLessEqual(len(checkpoint.relative_to(ROOT).as_posix()), 65)
                for task in 'SML':
                    target = replace(source, mode='eval', task=task, train_task='S',
                                     train_resource=resource, train_deadline=deadline)
                    self.assertEqual(checkpoint, target.checkpoint_dir())
                    csv = evaluation_csv(target, ROOT)
                    self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_', csv.name)
                    self.assertIn('_m1_s700-729_c', csv.name)
                    self.assertIn(MODEL_VERSION, csv.name)
                    self.assertEqual(csv.parent, ROOT / 'outputs/run' / MODEL_VERSION)
                    details = evaluation_directory(target, ROOT)
                    self.assertEqual(details.parent, csv.parent / '_details')
                    self.assertLess(len(str(details / 'measurements/episodes/seed700.json')), 260)
                    files.append(csv)
        self.assertEqual(len(set(checkpoints)), 9)
        self.assertEqual(len({path.name for path in files}), 27)
        self.assertEqual(len({path.parent for path in files}), 1)

    def test_exact_test_seeds_source_model_and_configuration_remain_distinct(self):
        scene = MARLScenario(mode='eval', task='L')
        variants = [scene, replace(scene, train_task='M'), replace(scene, model_seed=2),
                    replace(scene, seed_spec='701-729'), replace(scene, seed_spec='700,702,704'),
                    replace(scene, normalization_mode='f18_shared_v1'), replace(scene, legacy_split=True)]
        files = [evaluation_csv(s, ROOT) for s in variants]
        self.assertEqual(len(set(files)), len(variants))
        self.assertTrue(all(path.parent == files[0].parent for path in files))
        self.assertIn('_s700+702+704_', files[4].name)
        self.assertEqual(len({evaluation_directory(s, ROOT) for s in variants}), len(variants))

    def test_fingerprint_is_portable_and_detects_calibration_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'a.json', Path(directory) / 'b.json'
            first.write_text('{}'); second.write_text('{}')
            scene = MARLScenario(mode='train', normalization_config=str(first))
            initial = scene.model_name()
            self.assertEqual(initial, replace(scene, normalization_config=str(second)).model_name())
            second.write_text('{"changed":true}')
            self.assertNotEqual(initial, replace(scene, normalization_config=str(second)).model_name())

    def test_shared_csv_parent_does_not_block_other_scenes_or_overwrite_partial_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MARLScenario(mode='eval')
            second = replace(first, task='M')
            csv = evaluation_csv(first, root)
            details = evaluation_directory(first, root)
            protect_evaluation_outputs(csv, details)
            details.mkdir(parents=True)
            csv.write_bytes(b'first result')
            protect_evaluation_outputs(evaluation_csv(second, root), evaluation_directory(second, root))
            with self.assertRaises(FileExistsError): protect_evaluation_outputs(csv, details)
            csv.unlink()
            (details / 'partial.json').write_bytes(b'partial result')
            with self.assertRaises(FileExistsError): protect_evaluation_outputs(csv, details)

    def test_explicit_paths_are_honored_and_other_algorithms_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = MARLScenario(mode='eval', checkpoint_override=str(root / 'existing_models'),
                                 marl_options={'out': str(root / 'chosen.csv')})
            self.assertEqual(scene.checkpoint_dir(), root / 'existing_models')
            self.assertEqual(evaluation_csv(scene, ROOT), root / 'chosen.csv')
            self.assertEqual(evaluation_directory(scene, ROOT).parent, root / '_details')
        for algorithm in ('mahdrl', 'pd3qn', 'irws'):
            self.assertEqual(MARLScenario().model_name(algorithm), Scenario().model_name(algorithm))
            self.assertEqual(MARLScenario().checkpoint_dir(algorithm), Scenario().checkpoint_dir(algorithm))
            self.assertEqual(MARLScenario().log_dir(algorithm), Scenario().log_dir(algorithm))


if __name__ == '__main__':
    unittest.main()
