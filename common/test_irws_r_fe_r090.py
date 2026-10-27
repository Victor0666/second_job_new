"""R090 reward/physics separation and isolated checkpoint/CLI integration."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.irws_r_fe import runner as baseline
from algorithms.irws_r_fe_r090 import runner, selection
from algorithms.irws_r_fe_r090.config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION,
    BEST_CHECKPOINT, FINAL_CHECKPOINT, COMMON_REWARD, PaperConfig)
from algorithms.irws_r_fe_r090.scenario import FEScenario, parse_scenario
from common.experiment_protocol import protocol_binding
from common.test_irws_paper import fixture, synthetic_env


class SmallScenario(FEScenario):
    @property
    def vms_per_host(self):return (2,1)


class IRWSRFER090Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='irws_r_fe_r090_test_')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.path,scene=fixture(self.root)
        protocol=Path(scene.experiment_protocol)
        data=json.loads(protocol.read_text());data.update(train=[32,33],episodes=2)
        protocol.write_text(json.dumps(data))
        calibration=json.loads(self.path.read_text())
        calibration['generation']=dict(**protocol_binding(protocol),calibration_seeds={'SS_T':[32,33]})
        self.path.write_text(json.dumps(calibration))
        self.scene=SmallScenario(**dict(vars(scene),episodes=2))

    @staticmethod
    def environment(**kwargs):
        return runner.PaperCloudEnv(**dict(kwargs,**COMMON_REWARD))

    def test_only_reward_coefficients_change_for_identical_actions(self):
        records=[];physics=[]
        for module in (baseline,runner):
            make=lambda **kw:module.PaperCloudEnv(**dict(kw,**module.COMMON_REWARD))
            with synthetic_env(self.path,make) as env:
                env.reset(seed=32);steps=[]
                while not env.done_flag:
                    env.select_task(env.get_ready_task_candidates()['gids'][0])
                    reward,_,info=env.paper_step(int(np.flatnonzero(env.queue_mask())[0]))
                    expected=(1-module.COMMON_REWARD['alpha_delay_vm'])*(-.001*info['delta_energy_total'])
                    expected+=module.COMMON_REWARD['alpha_delay_vm']*info['r_delay']
                    self.assertAlmostEqual(reward,expected,places=12)
                    self.assertAlmostEqual(info['outer_cost'],-reward,places=12)
                    steps.append((reward,info['r_delay'],info['delta_energy_total'],info['fusion_weight']))
                    if not env.can_dispatch():env.finish_phase_and_advance()
                records.append(steps)
                physics.append((env.task_placement.copy(),env.task_start_time.copy(),env.task_end_time.copy(),env.total_energy))
        self.assertEqual(physics[0][0],physics[1][0]);self.assertEqual(physics[0][3],physics[1][3])
        np.testing.assert_array_equal(physics[0][1],physics[1][1])
        np.testing.assert_array_equal(physics[0][2],physics[1][2])
        self.assertEqual(len(records[0]),len(records[1]))
        self.assertTrue(any(a[0]!=b[0] for a,b in zip(*records)))
        for old,new in zip(*records):self.assertEqual(old[1:],new[1:])

    def test_acceleration_and_selector_preserved_with_independent_outputs(self):
        root=Path(__file__).resolve().parents[1]/'algorithms'
        for name in ('learner.py','networks.py','environment.py','numerics.py','selection.py'):
            self.assertEqual((root/'irws_r_fe'/name).read_bytes(),(root/'irws_r_fe_r090'/name).read_bytes())
        self.assertEqual(PaperConfig().as_dict(),baseline.PaperConfig().as_dict())
        self.assertEqual(selection.selection_metadata(),baseline.selection_metadata())
        self.assertEqual(runner.IMPLEMENTATION,baseline.IMPLEMENTATION)
        for kind in ('ckpts','run'):
            self.assertNotEqual(runner.output_dir(self.scene,kind),baseline.output_dir(self.scene,kind))
            self.assertIn(ALGORITHM,runner.output_dir(self.scene,kind).parts)
            self.assertIn(MODEL_VERSION,runner.output_dir(self.scene,kind).parts)
        self.assertEqual(baseline.COMMON_REWARD['alpha_delay_vm'],.75)
        self.assertEqual(COMMON_REWARD['alpha_delay_vm'],.9)
        self.assertEqual(runner.env_kwargs(self.scene,32)['alpha_delay_vm'],.9)

    def test_cli_nine_sources_and_27_targets_report_r090(self):
        destinations=set();sources=set()
        norm=dict(normalization_version='f19_shared_v1',normalization_config_hash='test_calibration_hash')
        with patch.object(FEScenario,'normalization_metadata',return_value=norm):
            for resource in 'SML':
                for deadline in ('Tight','Medium','Loose'):
                    args=['--resource',resource,'--deadline',deadline,'--dry-run']
                    output=io.StringIO()
                    with redirect_stdout(output),self.assertRaises(SystemExit):parse_scenario('train',argv=args)
                    source=json.loads(output.getvalue());sources.add(source['checkpoint_dir'])
                    self.assertEqual(source['display_name'],'IRWS-R-FE_R090')
                    self.assertEqual(source['reward_settings']['alpha_delay_vm'],.9)
                    for task in 'SML':
                        output=io.StringIO()
                        with redirect_stdout(output),self.assertRaises(SystemExit):
                            parse_scenario('eval',argv=args+['--task',task,'--train-task','S',
                                '--train-resource',resource,'--train-deadline',deadline])
                        target=json.loads(output.getvalue())
                        self.assertEqual(source['checkpoint_dir'],target['checkpoint_dir'])
                        destinations.add(target['evaluation_directory'])
        self.assertEqual(len(sources),9);self.assertEqual(len(destinations),27)

    def test_train_resume_selected_checkpoint_evaluate_and_weight_mismatch(self):
        with synthetic_env(self.path,self.environment) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)),redirect_stdout(io.StringIO()):
            kwargs=runner.env_kwargs(self.scene,32)
            _,models=runner.train(replace(self.scene,episodes=1),kwargs=kwargs,out=self.root/'models')
            continued,_=runner.train(self.scene,kwargs=kwargs,out=models,resume=models/FINAL_CHECKPOINT)
            self.assertEqual(continued.steps,24)
            _,obj=runner.load_checkpoint(models/BEST_CHECKPOINT,self.scene)
            self.assertEqual(obj['algorithm'],ALGORITHM);self.assertEqual(obj['display_name'],DISPLAY_NAME)
            self.assertEqual(obj['reward_settings']['alpha_delay_vm'],.9)
            logs=[json.loads(s) for s in (models/runner.artifact_name(self.scene,'train')).read_text().splitlines()]
            self.assertEqual(len(logs),2)
            self.assertTrue(all(r['objective']['settings']['alpha_delay_vm']==.9 for r in logs))
            self.assertTrue(all(r['updates'] for r in logs))
            with self.assertRaisesRegex(ValueError,'algorithm/version mismatch'):
                baseline.load_checkpoint(models/BEST_CHECKPOINT,self.scene)
            bad=self.root/'wrong_weight.pth'
            torch.save(dict(obj,reward_settings=baseline.COMMON_REWARD),bad)
            with self.assertRaisesRegex(ValueError,'reward objective mismatch'):runner.load_checkpoint(bad,self.scene)
            with self.assertRaises(FileExistsError):runner.train(self.scene,out=models)
            evaluation=replace(self.scene,mode='eval',seed_spec='2',checkpoint_override=str(models))
            with patch.object(runner,'parse_scenario',return_value=evaluation),patch.object(
                    runner,'output_dir',return_value=self.root/'evaluation'):
                runner.evaluate_main()
                with self.assertRaises(FileExistsError):runner.evaluate_main()
            report=json.loads((self.root/'evaluation'/'SS_Tight'/runner.artifact_name(evaluation,'summary')).read_text())
            self.assertEqual(report['reward_settings']['alpha_delay_vm'],.9)
            self.assertEqual(report['checkpoint_selection'],selection.selection_metadata())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_r090_update_and_validation(self):
        with synthetic_env(self.path,self.environment) as template,patch.object(
                runner,'PaperCloudEnv',side_effect=lambda **kw:deepcopy(template)):
            learner=runner.PaperLearner(3,PaperConfig(device='cuda'))
            result=runner.run_episode(template,learner,32,True)
            rows,_=runner.evaluate(learner,self.scene,[632],'validation',kwargs=runner.env_kwargs(self.scene,32))
            self.assertEqual(sum(u['sample_presentations'] for u in result['updates']),12)
            self.assertEqual(result['objective']['settings']['alpha_delay_vm'],.9)
            self.assertEqual(rows[0]['wf_count'],2)
            self.assertTrue(all(torch.isfinite(p).all() for p in learner.actor.parameters()))


if __name__=='__main__':unittest.main()
