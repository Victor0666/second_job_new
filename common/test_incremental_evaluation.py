"""Seed results remain readable if a later evaluation seed fails."""
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.scenarios import Scenario
from algorithms.irws_paper import runner
from algorithms.ertf_fifs import evaluate as heuristic


class IncrementalEvaluationTests(unittest.TestCase):
    def test_irws_main_flushes_first_seed_before_second_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=Scenario(seed_spec='700-701')
            def run(env,learner,seed,training=False,evaluation=None):
                if seed==701:
                    with (root/'SS_Tight/episodes.csv').open(newline='',encoding='utf-8') as f:
                        self.assertEqual([r['seed'] for r in csv.DictReader(f)],['700'])
                    raise RuntimeError('interrupted second seed')
                return dict(metrics=dict(total_energy=12.,makespan=3.))
            with patch.object(runner,'parse_scenario',return_value=scene),patch.object(runner,'output_dir',return_value=root),\
                 patch.object(runner,'load_checkpoint',return_value=(object(),{})),patch.object(runner,'PaperCloudEnv',return_value=object()),\
                 patch.object(runner,'env_kwargs',return_value={}),patch.object(runner,'run_episode',side_effect=run),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):runner.evaluate_main()
            self.assertFalse((root/'SS_Tight/summary.json').exists())

    def test_heuristic_closes_each_seed_and_writes_single_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'results.csv')
            heuristic.save_csv([dict(seed=700,total_energy=1.)],path,append=True)
            with open(path,newline='',encoding='utf-8-sig') as f:self.assertEqual([r['seed'] for r in csv.DictReader(f)],['700'])
            heuristic.save_csv([dict(seed=701,total_energy=2.)],path,append=True)
            with open(path,newline='',encoding='utf-8-sig') as f:self.assertEqual([r['seed'] for r in csv.DictReader(f)],['700','701'])

    def test_irws_callback_keeps_return_and_summary_unchanged(self):
        scene=Scenario(seed_spec='700-701');saved=[]
        def run(env,learner,seed,training=False,evaluation=None):return dict(metrics={'total_energy':float(seed)})
        with patch.object(runner,'PaperCloudEnv',return_value=object()),patch.object(runner,'output_dir',return_value=Path('unused')),\
             patch.object(runner,'env_kwargs',return_value={}),patch.object(runner,'run_episode',side_effect=run):
            rows,summary=runner.evaluate(object(),scene,scene.seeds,on_result=saved.append)
        self.assertEqual(rows,saved);self.assertEqual(summary['total_energy']['mean'],700.5)


if __name__=='__main__':unittest.main()
