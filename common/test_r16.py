"""R16: selected protocol governs calibration, resumption and formal test seeds."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.experiment_protocol import DEFAULT_PROTOCOL, protocol_binding
from common.f18_calibrate_normalization import main, validate_seeds, summarize
from common.f18_state_normalization import validate_calibration
from common.scenarios import Scenario, parse_scenario
from common.test_f18 import fixture_config


class R16Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r16_tests_')
        self.root = Path(self.temp.name)
        self.protocol = self.root/'protocol.json'
        self.protocol.write_text(json.dumps(dict(version='r16_custom', train=[32,33], validation=[632],
            test=[2,3], episodes=2, selection='minimum_validation_total_energy')), encoding='utf-8')
        self.config = self.root/'scales.json'
        fixture_config(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_test_list_and_versioned_extension(self):
        for seeds in ('900', '600', '599', '700,900'):
            with self.assertRaises(ValueError): Scenario(seed_spec=seeds)
        self.assertEqual(Scenario(seed_spec='700,729').seeds, (700,729))
        self.assertEqual(Scenario(experiment_protocol=str(self.protocol), seed_spec='2-3').seeds, (2,3))
        with self.assertRaises(ValueError): Scenario(experiment_protocol=str(self.protocol), seed_spec='700')
        for algorithm in ('mahdrl','marl','pd3qn','irws','irws_paper','ertf_fifs', 'ablation_tl2'):
            with patch('sys.argv', ['entry','--dry-run','--seeds','900']), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit) as result: parse_scenario('eval',algorithm)
            self.assertEqual(result.exception.code, 2)

    def test_calibration_checks_selected_training_list(self):
        validate_seeds([2])  # default training; custom test
        with self.assertRaises(ValueError): validate_seeds([2], self.protocol)
        validate_seeds([32,33], self.protocol)
        for seeds in ([], [32,32], [632], [900]):
            with self.assertRaises(ValueError): validate_seeds(seeds,self.protocol)
        with self.assertRaises(ValueError):
            main(['--experiment-protocol',str(self.protocol),'--seeds','2','--dry-run'])
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--experiment-protocol',str(self.protocol),'--seeds','32','--all-nine','--dry-run']),0)
        self.assertEqual(json.loads(output.getvalue())['experiment_protocol']['test'],[2,3])

    def test_loader_rejects_missing_wrong_and_heldout_provenance(self):
        scene=Scenario(mode='train', normalization_config=str(self.config))
        scene.normalization_metadata()
        custom=replace(scene,episodes=2,experiment_protocol=str(self.protocol))
        with self.assertRaisesRegex(ValueError,'protocol'): custom.normalization_metadata()
        obj=json.loads(self.config.read_text())
        obj['generation'].update(protocol_binding(self.protocol))
        obj['generation']['calibration_seeds']['SS_T']=[32]
        self.config.write_text(json.dumps(obj))
        custom.normalization_metadata()
        # A model generalizing to L still validates the S calibration source.
        replace(custom,mode='eval',task='L',seed_spec='2').normalization_metadata()
        obj['generation']['calibration_seeds']['SS_T']=[2]
        self.config.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'training partition'): custom.normalization_metadata()
        obj.pop('generation');self.config.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'missing'): custom.normalization_metadata()

    def test_resume_summary_and_loader_bind_same_protocol(self):
        out=self.root/'calibration'
        episode=dict(status='ok',seed=32,ready=[1,2],deadlines=[10,20],
            physical={k:[1.,2.] for k in ('workload_mi_ref','input_bits_ref','output_bits_ref','up_rank_ref','down_rank_ref')},
            empty_observations=0,decisions=2,assignments=2,vm_pc_ref=8.,vm_bw_ref=8000.,host_pc_ref=40.)
        argv=['--experiment-protocol',str(self.protocol),'--seeds','32','--out',str(out)]
        with patch('common.f18_calibrate_normalization.cache_info',return_value={'sha256':'fixture'}), patch(
                'common.f18_calibrate_normalization.sample_episode',return_value=episode) as sample, redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv),0)
            self.assertEqual(main(argv+['--resume']),0)
            self.assertEqual(sample.call_count,1)
        path=out/'f18_shared_normalization_v1.json'
        validate_calibration(path,'SS_T',self.protocol)
        with self.assertRaisesRegex(ValueError,'protocol'): validate_calibration(path,'SS_T',DEFAULT_PROTOCOL)
        manifest=out/'f18_calibration_manifest.json';before=manifest.read_bytes()
        for args in (['--seeds','101','--resume'], ['--summary-only']):
            with self.assertRaisesRegex(ValueError,'protocol'): main(['--out',str(out),*args])
            self.assertEqual(manifest.read_bytes(),before)
        self.assertEqual(main(['--out',str(out),'--summary-only','--experiment-protocol',str(self.protocol)]),0)
        obj=json.loads(manifest.read_text());obj['scenes']['SS_T']['seeds']=[2]
        with self.assertRaises(ValueError): summarize(out,obj,self.protocol)
        obj.pop('experiment_protocol_hash')
        with self.assertRaises(ValueError): summarize(out,obj,self.protocol)


if __name__=='__main__': unittest.main()
