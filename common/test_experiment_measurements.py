"""Measurements preserve RNG, physical trajectories and network updates."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from common.experiment_measurements import TrainingRecorder, EvaluationRecorder, episode_metrics, workflow_rows
from common.summarize_experiments import summarize
from common.test_irws_paper import fixture, synthetic_env
from algorithms.irws_paper.config import PaperConfig
from algorithms.irws_paper.learner import PaperLearner
from algorithms.irws_paper.runner import run_episode


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='measurements_tests_');self.root=Path(self.temp.name)
        self.path,self.scenario=fixture(self.root)

    def tearDown(self):self.temp.cleanup()

    def trial(self,observed,device="cpu"):
        random.seed(99);np.random.seed(99);torch.manual_seed(99)
        with patch.dict('os.environ',{'EXPERIMENT_METRICS':str(int(observed))}),synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig(device=device))
            recorder=TrainingRecorder(self.root/str(observed),self.scenario,'irws_paper',env,{'IRWS':learner})
            result=run_episode(env,learner,32,True)
            before_calls=recorder.policy_calls.copy()
            # Validation must not be counted as training policy calls or updates.
            with synthetic_env(self.path) as validation:
                def validate():
                    from common.experiment_measurements import collect_validation
                    out=run_episode(validation,learner,2,False);collect_validation(validation);return out
                recorder.validation(validate)
            if observed:self.assertEqual(before_calls,recorder.policy_calls)
            recorder.close()
            states={n:{k:v.detach().clone() for k,v in getattr(learner,n).state_dict().items()} for n in ('task','actor','critic','intrinsic')}
            return result,states,torch.get_rng_state().clone(),np.random.get_state(),random.getstate(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def test_training_records_do_not_change_updates_or_rng(self):
        off=self.trial(False);on=self.trial(True)
        self.assertEqual(off[0]['metrics'],on[0]['metrics'])
        self.assertEqual(off[0]['updates'],on[0]['updates'])
        for name,state in off[1].items():
            for key,value in state.items():self.assertTrue(torch.equal(value,on[1][name][key]),(name,key))
        self.assertTrue(torch.equal(off[2],on[2]));np.testing.assert_array_equal(off[3][1],on[3][1]);self.assertEqual(off[4],on[4])
        rows=[json.loads(s) for s in (self.root/'True/training_measurements.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1);row=rows[0]
        self.assertEqual(row['assignment_calls'],12)
        self.assertEqual(set(row['optimizer_steps'].values()),{1})
        self.assertEqual(set(row['optimizer_sample_presentations'].values()),{12})
        self.assertEqual(row['rollout_transition_presentations']['IRWS'],12)
        self.assertEqual(len(row['validation']),1)
        self.assertGreaterEqual(row['train_seconds'],0)
        self.assertGreater(row['validation_seconds'],0)
        self.assertFalse((self.root/'False/training_measurements.jsonl').exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_measurements_preserve_weights_and_rng(self):
        off=self.trial(False,device='cuda');on=self.trial(True,device='cuda')
        self.assertEqual(off[0]['metrics'],on[0]['metrics'])
        for name,state in off[1].items():
            for key,value in state.items():self.assertTrue(torch.equal(value,on[1][name][key]),(name,key))
        self.assertTrue(torch.equal(off[2],on[2]));self.assertEqual(off[4],on[4])
        for a,b in zip(off[5],on[5]):self.assertTrue(torch.equal(a,b))
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig(device='cuda'))
            result=run_episode(env,learner,2,False,evaluation=(self.scenario,self.root/'cuda_eval'))
            measured=json.loads((self.root/'cuda_eval/episodes/seed2.json').read_text())
            self.assertEqual(measured['metrics']['total_energy'],result['metrics']['total_energy'])
            self.assertGreater(measured['timing']['decision_mean_ms'],0.)

    def test_evaluation_exports_match_physics_and_restore_methods(self):
        random.seed(4);np.random.seed(4);torch.manual_seed(4)
        with synthetic_env(self.path) as env:
            learner=PaperLearner(3,PaperConfig())
            initial=learner.prepare.__func__
            # run_episode installs the observer after reset.
            result=run_episode(env,learner,2,False,evaluation=(self.scenario,self.root/'evaluation'))
            self.assertIs(learner.prepare.__func__,initial)
            measured=json.loads((self.root/'evaluation/episodes/seed2.json').read_text())
            self.assertEqual(measured['metrics']['total_energy'],result['metrics']['total_energy'])
            self.assertEqual(measured['metrics']['makespan'],result['metrics']['makespan'])
            self.assertEqual(measured['timing']['allocation_count'],12)
            self.assertEqual(len(workflow_rows(env)),2)
            self.assertEqual(measured['metrics']['wf_count'],2)
            self.assertTrue((self.root/'evaluation/workflows/seed2.csv').exists())
            self.assertGreaterEqual(measured['timing']['decision_p99_ms'],measured['timing']['decision_mean_ms'])

    def test_completed_workflows_and_duration_origin(self):
        from types import SimpleNamespace
        env=SimpleNamespace(random_seed=1,current_time=17.,total_energy=10.,arrival_times=[5.],
            task_meta=[(0,0),(0,1)],task_state=['Finished','Finished'],task_end_time=[12.,17.],task_baseline_finish=[11.,18.],
            workflows=[SimpleNamespace(arrival_time=5.,deadline=16.)])
        m=episode_metrics(env)
        self.assertEqual(m['makespan'],12.);self.assertEqual(m['wf_avg_tardiness'],1.)
        self.assertEqual(m['wf_success_count'],0)
        env.task_state[1]='Running'
        self.assertEqual(episode_metrics(env)['wf_completed_count'],0)
        self.assertEqual(workflow_rows(env)[0]['success'],0)

    def test_global_vm_interface_is_counted_once(self):
        from types import SimpleNamespace
        env=SimpleNamespace(random_seed=1,current_time=17.,total_energy=10.,arrival_times=[5.],
            task_meta=[(0,0)],task_state=['Finished'],task_end_time=[17.],task_baseline_finish=[18.],
            workflows=[SimpleNamespace(arrival_time=5.,deadline=20.)],vm_assign=lambda a:None,
            vm_assign_global=lambda a:(1.,{}),reset=lambda:None)
        recorder=TrainingRecorder(self.root/'tl',self.scenario,'ablation_nohost',env,{})
        env.vm_assign_global(0);recorder.close()
        row=json.loads((self.root/'tl/training_measurements.jsonl').read_text())
        self.assertEqual(row['assignment_calls'],1)
        with patch.dict('os.environ',{'EXPERIMENT_METRICS':'1'}):
            observer=EvaluationRecorder(self.scenario,'ablation_nohost',env,{},self.root/'tl_eval')
            env.vm_assign_global(0);result=observer.finish()
        self.assertEqual(result['timing']['allocation_count'],1)

    def test_ppo_sample_budget_includes_short_minibatch_and_reuse(self):
        from types import SimpleNamespace
        env=SimpleNamespace(random_seed=1,current_time=17.,total_energy=10.,arrival_times=[5.],
            task_meta=[(0,0)],task_state=['Finished'],task_end_time=[17.],task_baseline_finish=[18.],
            workflows=[SimpleNamespace(arrival_time=5.,deadline=20.)],step=lambda a:None,reset=lambda:None)
        agent=SimpleNamespace(device='cpu',cfg=SimpleNamespace(minibatch_size=3),buf=SimpleNamespace(size=lambda:5))
        agent.opt=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.1)
        def update():
            for _ in range(4):agent.opt.step()  # Two epochs of batches [3,2].
        agent.update=update
        recorder=TrainingRecorder(self.root/'ppo_budget',self.scenario,'irws',env,{'PPO':agent})
        agent.update();recorder.close()
        row=json.loads((self.root/'ppo_budget/training_measurements.jsonl').read_text())
        self.assertEqual(row['optimizer_steps']['PPO.opt'],4)
        self.assertEqual(row['optimizer_sample_presentations']['PPO.opt'],10)

    def test_summary_uses_seed_trials_and_checks_pairing(self):
        for algorithm,folder,energy in [('a','a',10.),('b','b',12.)]:
            d=self.root/folder/'episodes';d.mkdir(parents=True)
            for seed in (700,701):
                row=dict(algorithm=algorithm,model_seed=1,task='S',resource='S',deadline='Tight',seed=seed,
                    checkpoints=[],provenance=dict(data_split='same',deadline_cache={'sha256':'same'},instance_generator='same'),
                    metrics=dict(total_energy=energy,makespan=3.,wf_success_rate=1.,wf_avg_tardiness=0.,wf_avg_response_time=2.,task_success_rate=1.,task_avg_tardiness=0.),
                    timing=dict(scope='test',decision_mean_ms=1.))
                (d/f'seed{seed}.json').write_text(json.dumps(row))
        result=summarize(self.root/'a',(700,701),reference=self.root/'b')
        self.assertEqual(result['qos_event']['trials'],2)
        self.assertLess(result['qos_event']['exact_binomial_ci'][0],.99)
        self.assertAlmostEqual(result['paired_reference']['comparisons']['total_energy']['energy_reduction_percent_from_means'],100/6)
        p=self.root/'b/episodes/seed701.json';row=json.loads(p.read_text());row['task']='M';p.write_text(json.dumps(row))
        with self.assertRaises(ValueError):summarize(self.root/'a',(700,701),reference=self.root/'b')


if __name__=='__main__':unittest.main()
