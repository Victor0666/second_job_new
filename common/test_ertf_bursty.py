"""ERTF preserves physical instances, deadline budgets, and the ESP boundary."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import math
import unittest
from unittest.mock import patch

import numpy as np

from algorithms.ertf_fifs import evaluate
from algorithms.ertf_fifs.evaluate_suite import ESP_THRESHOLD, METRICS, summarize, validate_episode
from common.bursty_workload import PROFILE, apply_arrival_profile
from common.f22_instances import instance_identity
from common.read_xml_opt_Tsize import workflow_instance_hash
from common.scenarios import Scenario


class ERTFBurstyTests(unittest.TestCase):
    def ertf_environment(self, scene, env_cls=evaluate.MaskOnlyERTFEnv):
        kwargs = evaluate.build_env_kwargs(scene)
        for key in ('energy_reward_scale', 'task_baseline_norm', 'energy_norm_per_mi_ref',
                    'alpha_delay_host', 'alpha_delay_vm'):
            kwargs.pop(key)
        return env_cls(**kwargs)

    def snapshot(self, env, scene):
        try:
            env.reset(seed=700)
            env.current_time = float(env.arrival_times[-1])
            env._add_workflow_if_arrived()
            return (list(env.arrival_times), instance_identity(env, scene, 700)['content_hash'],
                    [w.deadline - w.arrival_time for w in env.workflows],
                    [workflow_instance_hash(w.graph, w.tasks) for w in env.workflows])
        finally:
            env.close()

    def test_fifty_workflow_payloads_deadlines_and_arrivals_match_original_environment(self):
        combinations = [(t, r, 'Tight') for t in 'SML' for r in 'SML']
        combinations += [('S', 'S', d) for d in ('Medium', 'Loose')]
        for task, resource, deadline in combinations:
            with self.subTest(task=task, resource=resource, deadline=deadline), redirect_stdout(io.StringIO()):
                standard = Scenario(task=task, resource=resource, deadline=deadline, seed_spec='700')
                scene = replace(standard, workload_profile=PROFILE)
                baseline = self.snapshot(self.ertf_environment(scene), scene)
                reference = self.snapshot(self.ertf_environment(standard), standard)
                original = self.snapshot(self.ertf_environment(scene, evaluate.CloudWorkflowEnv_VMAgents), scene)
                self.assertEqual(baseline[1], original[1])
                self.assertEqual(baseline[3], reference[3])
                self.assertEqual(baseline[0], apply_arrival_profile(reference[0], PROFILE, 700, 50, .03))
                self.assertEqual(len(baseline[0]), 50)
                np.testing.assert_allclose(baseline[2], reference[2], rtol=0, atol=1e-9)
                self.assertEqual(scene.cache_path, standard.cache_path)

    def test_esp_includes_exactly_46_of_50(self):
        rows = [dict(seed=700+i, wf_count=50, wf_success_count=success,
                     wf_avg_response_time=10., decision_mean_ms=1., **dict.fromkeys(METRICS, 1.))
                for i, success in enumerate((45, 46, 50))]
        actual = summarize(rows, (700, 701, 702), esp_threshold=ESP_THRESHOLD)
        self.assertEqual(actual['esp'], 2 / 3)

    def test_single_scene_cli_accepts_bursty_without_changing_standard_defaults(self):
        args = ['evaluate.py', '--workload-profile', PROFILE, '--task', 'S', '--resource', 'S',
                '--deadline', 'Medium', '--dry-run']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            evaluate.parse_scenario('eval', 'ertf_fifs', allow_bursty=True)
        self.assertEqual(stopped.exception.code, 0)
        self.assertNotIn('arrival_profile', evaluate.build_env_kwargs(Scenario()))

    def test_validation_accepts_rounded_alpha_and_rejects_wrong_arrival_stream(self):
        scene = Scenario(workload_profile=PROFILE, seed_spec='700')
        arrivals = apply_arrival_profile([0.] * 50, PROFILE, 700, 50, .03)
        alpha = math.nextafter(2., 3.)
        workflows = [dict(arrival=arrival, reference_makespan=100., deadline_alpha=alpha,
                          deadline=arrival + alpha * 100.) for arrival in arrivals]
        record = dict(seed=700, timing=dict(allocation_count=1),
                      metrics=dict(wf_count=50, wf_completed_count=50, task_count=1,
                                   **dict.fromkeys(METRICS, 1.)))
        native = dict(task_finished=1, energy_check=1., total_energy=1., makespan=1.,
                      wf_avg_success=1., wf_avg_tardiness=1.)
        validate_episode(scene, native, record, workflows)
        workflows[0]['arrival'] += .001
        with self.assertRaisesRegex(ValueError, 'arrival times'):
            validate_episode(scene, native, record, workflows)


if __name__ == '__main__':
    unittest.main()
