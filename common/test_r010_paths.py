"""Keep downloaded CSVs distinguishable and training directories compact."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from algorithms.pd3qn_fe_r010.scenario import R010Scenario
from algorithms.irws_r_fe_r010.scenario import FEScenario
from algorithms.irws_r_fe_r010 import runner as irws
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.scenario import ScoreScenario


class OutputPathTests(unittest.TestCase):
    @staticmethod
    def paths(scene):
        if isinstance(scene, FEScenario):
            return (irws.output_dir(scene, 'ckpts'), irws.evaluation_directory(scene),
                    irws.output_dir(scene, 'run') / irws.artifact_name(scene, 'eval'))
        directory = scene.evaluation_dir() if isinstance(scene, ScoreScenario) else scene.evaluation_dir
        return scene.checkpoint_dir(), directory, scene.evaluation_csv() if isinstance(scene, ScoreScenario) else scene.evaluation_csv

    def test_27_csvs_per_algorithm_share_a_parent_and_have_unique_scene_names(self):
        for klass in (R010Scenario, FEScenario, ScoreScenario):
            files = []
            for resource in 'SML':
                for deadline in ('Tight', 'Medium', 'Loose'):
                    source = klass(resource=resource, deadline=deadline)
                    source_checkpoint = self.paths(source)[0]
                    self.assertLessEqual(len(source_checkpoint.name), 40)
                    self.assertLessEqual(len(str(source_checkpoint.relative_to(source_checkpoint.parents[3]))), 80)
                    for task in 'SML':
                        target = replace(source, mode='eval', task=task, train_task='S',
                                         train_resource=resource, train_deadline=deadline)
                        checkpoint, details, csv = self.paths(target)
                        self.assertEqual(checkpoint, source_checkpoint)
                        expected = (f'_{target.target_tag}_from_{target.source_tag}_' if isinstance(target, ScoreScenario)
                                    else f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_')
                        self.assertIn(expected, csv.name)
                        self.assertIn('_m1_s700-729_c', csv.name)
                        self.assertEqual(csv.suffix, '.csv')
                        self.assertEqual(details.parent.name, '_details')
                        self.assertLess(len(str(details / 'measurements/episodes/seed700.json')), 260)
                        files.append(csv)
            self.assertEqual(len({p.parent for p in files}), 1)
            self.assertEqual(len({p.name for p in files}), 27)

    def test_model_source_and_exact_test_seeds_are_distinguishable_after_download(self):
        for klass in (R010Scenario, FEScenario, ScoreScenario):
            target = klass(mode='eval', task='L', train_task='S', train_resource='S', train_deadline='Tight')
            variants = [target, replace(target, train_task='M'), replace(target, model_seed=2),
                        replace(target, seed_spec='701-729'), replace(target, seed_spec='700,702,704')]
            files = [self.paths(scene)[2] for scene in variants]
            self.assertEqual(len({p.name for p in files}), len(variants))
            self.assertTrue(all(p.parent == files[0].parent for p in files))
            self.assertIn('_s700-704_' if klass is ScoreScenario else '_s700+702+704_', files[-1].name)

    def test_configuration_fingerprints_are_portable_and_keep_runs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'a.json', Path(directory)/'b.json'
            first.write_text('{}'); second.write_text('{}')
            for klass in (R010Scenario, FEScenario, ScoreScenario):
                scene = klass(normalization_config=str(first))
                initial = self.paths(scene)[0].name
                moved = self.paths(replace(scene, normalization_config=str(second)))[0].name
                self.assertEqual(initial, moved)
                self.assertNotEqual(initial, self.paths(replace(scene, normalization_mode='f18_shared_v1'))[0].name)
                second.write_text('{"changed":true}')
                self.assertNotEqual(initial, self.paths(replace(scene, normalization_config=str(second)))[0].name)
                second.write_text('{}')
            scene = FEScenario(normalization_config=str(first))
            self.assertNotEqual(self.paths(scene)[0], self.paths(replace(scene, paper_options={'queue_capacity': 11}))[0])


if __name__ == '__main__':
    unittest.main()
