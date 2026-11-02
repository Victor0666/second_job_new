"""Compact training names and self-identifying CSVs survive flat downloads."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from common import test_mahdrl_as_fast_hscore as fixtures
from common import test_mahdrl_as_fast_hscore_speed as speed_tests
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore_speed import runtime as speed
from algorithms.mahdrl_as_fast_hscore_speed.scenario import ScoreScenario as SpeedScenario
from algorithms.mahdrl_as_fast_hscore_speed2 import runtime as speed2
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import ScoreScenario as Speed2Scenario
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task import runtime as sync_task
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task.scenario import ScoreScenario as SyncTaskScenario


VARIANTS = ((SpeedScenario, speed), (Speed2Scenario, speed2), (SyncTaskScenario, sync_task))


class PathMatrixTests(unittest.TestCase):
    def test_short_training_paths_and_flat_unique_evaluation_names(self):
        all_models, all_csvs = set(), set()
        for cls, _ in VARIANTS:
            root = Path(cls().output_root)
            evaluation_roots = set()
            for preview in (False, True):
                for period in (1, 4):
                    for resource in 'SML':
                        for deadline in ('Tight', 'Medium', 'Loose'):
                            source = cls(va_preview=preview, ha_update_period=period,
                                         resource=resource, deadline=deadline)
                            self.assertEqual(len(source.model_name()), 17)
                            self.assertLessEqual(len(source.checkpoint_dir().relative_to(root).as_posix()), 64)
                            self.assertNotIn(source.checkpoint_dir(), all_models)
                            all_models.add(source.checkpoint_dir())
                            for task in 'SML':
                                target = replace(source, mode='eval', task=task, train_task='S',
                                                 train_resource=resource, train_deadline=deadline)
                                self.assertEqual(source.checkpoint_dir(), target.checkpoint_dir())
                                csv_path = target.evaluation_csv()
                                self.assertEqual(csv_path.parent, root/'outputs/run'/source.output_tag)
                                self.assertIn(f'_{target.target_tag}_from_{source.source_tag}_', csv_path.name)
                                self.assertIn('_m1_s700-729_', csv_path.name)
                                self.assertIn(f'_p{int(preview)}h{period}_', csv_path.name)
                                self.assertNotIn(csv_path.name, all_csvs)
                                all_csvs.add(csv_path.name)
                                evaluation_roots.add(csv_path.parent)
                                self.assertEqual(target.evaluation_dir().parent, csv_path.parent/'_details')
            self.assertEqual(len(evaluation_roots), 1)
            target = cls(mode='eval', train_resource='S', train_deadline='Tight')
            self.assertNotEqual(target.evaluation_csv().name, replace(target, model_seed=2).evaluation_csv().name)
            self.assertNotEqual(target.evaluation_csv().name, replace(target, deadline='Loose').evaluation_csv().name)
            self.assertNotEqual(replace(target, seed_spec='700,702,703').evaluation_csv().name,
                                replace(target, seed_spec='700,701,703').evaluation_csv().name)
        self.assertEqual((len(all_models), len(all_csvs)), (36*len(VARIANTS), 108*len(VARIANTS)))


class ActualCSVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HScoreTests.setUpClass()
        cls.root, cls.scales, cls.cache = (fixtures.HScoreTests.root,
            fixtures.HScoreTests.scales, fixtures.HScoreTests.cache)
        cls.caches = {'S': cls.cache}
        for task in 'ML':
            cls.caches[task] = cls.root/f'cache_{task}.json'
            with redirect_stdout(io.StringIO()):
                precompute(Scenario(mode='precompute', task=task, seed_spec='700',
                                    cache_output=str(cls.caches[task])))

    @classmethod
    def tearDownClass(cls):
        fixtures.HScoreTests.tearDownClass()

    def test_source_SS_produces_SS_MS_LS_csvs_in_one_directory_for_all_variants(self):
        for cls, runner in VARIANTS:
            source = cls(mode='train', episodes=1, legacy_split=True, device='cpu',
                normalization_config=str(self.scales), cache_file=str(self.cache),
                output_root=str(self.root/cls.output_tag))
            with patch.object(base, 'build_env_kwargs', fixtures.HScoreTests.small_kwargs(base.build_env_kwargs)), \
                    redirect_stdout(io.StringIO()):
                runner.train(source, agent_builder=speed_tests.tiny_agents)
                paths = []
                for task in 'SML':
                    target = replace(source, mode='eval', task=task, train_resource='S',
                        train_deadline='Tight', seed_spec='700', cache_file=str(self.caches[task]))
                    runner.evaluate(target, agent_builder=speed_tests.tiny_agents)
                    path = target.evaluation_csv()
                    paths.append(path)
                    with path.open(encoding='utf8', newline='') as stream:
                        row = next(csv.DictReader(stream))
                    self.assertEqual((row['target_scene'], row['source_scene']), (task+'S_T', 'SS_T'))
                    self.assertEqual(row['model_version'], source.model_version)
                self.assertEqual(len({p.parent for p in paths}), 1)
                self.assertEqual(len({p.name for p in paths}), 3)
                self.assertEqual(len(list(paths[0].parent.glob('*.csv'))), 3)
                with self.assertRaises(FileExistsError):
                    runner.evaluate(target, agent_builder=speed_tests.tiny_agents)


if __name__ == '__main__':
    unittest.main()
