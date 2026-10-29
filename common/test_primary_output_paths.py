"""Short training names, portable CSV identities and historical model lookup."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.pd3qn_fe.scenario import FEScenario as PDScene
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario
from common.test_stress_workload import SCENES, PROFILE, checkpoint, csv_path


class PrimaryOutputPathTests(unittest.TestCase):
    def test_all_four_flat_unique_csvs_and_compact_training_names(self):
        for cls in SCENES:
            files = []
            for profile, deadlines in [('standard', ('Tight', 'Medium', 'Loose')), (PROFILE, ('Tight',))]:
                for resource in 'SML':
                    for deadline in deadlines:
                        source = cls(mode='train', resource=resource, deadline=deadline, workload_profile=profile)
                        self.assertLessEqual(len(checkpoint(source).name), 32)
                        for task in 'SML':
                            target = replace(source, mode='eval', task=task, train_task='S',
                                             train_resource=resource, train_deadline=deadline)
                            self.assertEqual(checkpoint(target), checkpoint(source))
                            path = csv_path(target)
                            self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_', path.name)
                            self.assertIn('_m1_s700-729_c', path.name)
                            self.assertEqual('_a1l006r003' in path.name, profile == PROFILE)
                            files.append(path)
            self.assertEqual(len({p.parent for p in files}), 1)
            self.assertEqual(len({p.name for p in files}), 36)

    def test_downloaded_names_distinguish_source_seeds_and_settings(self):
        for cls in SCENES:
            target = cls(mode='eval', task='L', train_resource='S', train_deadline='Tight')
            variants = [target, replace(target, train_task='M'), replace(target, model_seed=2),
                        replace(target, seed_spec='701-729'), replace(target, seed_spec='700,702,704'),
                        replace(target, normalization_mode='f18_shared_v1'),
                        replace(target, workload_profile=PROFILE)]
            self.assertEqual(len({csv_path(s).name for s in variants}), len(variants))

    def test_legacy_model_fallback_does_not_redirect_new_training(self):
        with tempfile.TemporaryDirectory(prefix='primary_paths_') as directory:
            root = Path(directory)
            with patch('algorithms.pd3qn_fe.scenario.ROOT', root):
                for cls in (PDScene, ScoreScenario):
                    for profile in ('standard', PROFILE):
                        scene = cls(mode='train', workload_profile=profile,
                                    **({'output_root': str(root)} if cls is ScoreScenario else {}))
                        legacy = root / 'outputs/ckpts' / scene.legacy_model_name()
                        legacy.mkdir(parents=True)
                        current = scene.checkpoint_dir()
                        self.assertNotEqual(current, legacy)
                        target = replace(scene, mode='eval')
                        self.assertEqual(target.checkpoint_dir(), legacy)
                        current.mkdir(parents=True)
                        self.assertEqual(target.checkpoint_dir(), current)
                        self.assertEqual(replace(target, checkpoint_override=str(legacy)).checkpoint_dir(), legacy)

    def test_existing_flat_csv_blocks_loading_even_without_details(self):
        from algorithms.pd3qn_fe.evaluate import evaluate as evaluate_pd
        from algorithms.mahdrl_oe_ge_apf_score_fast.runtime import evaluate as evaluate_ma
        with tempfile.TemporaryDirectory(prefix='primary_csv_') as directory:
            root = Path(directory)
            with patch('algorithms.pd3qn_fe.scenario.ROOT', root):
                for scene, evaluate, loader in (
                    (PDScene(), evaluate_pd, 'algorithms.pd3qn_fe.evaluate.make_agent'),
                    (ScoreScenario(output_root=str(root)), evaluate_ma,
                     'algorithms.mahdrl_oe_ge_apf_score_fast.runtime.load_agents')):
                    path = csv_path(scene)
                    path.parent.mkdir(parents=True)
                    path.write_text('preserve result', encoding='utf-8')
                    with patch(loader) as load, self.assertRaises(FileExistsError):
                        evaluate(scene)
                    load.assert_not_called()
                    self.assertEqual(path.read_text(), 'preserve result')


if __name__ == '__main__':
    unittest.main()
