"""Checks for workload comparability and complete per-seed summaries."""
from contextlib import redirect_stdout
import io
import unittest
import numpy as np

from algorithms.ertf_fifs.evaluate import (build_env_kwargs, CloudWorkflowEnv_VMAgents, MaskOnlyERTFEnv,
    _force_fcfs_manager, _pick_first_idle_host, _pick_fifs_vm)
from algorithms.ertf_fifs.evaluate_suite import METRICS, PROFILES, summarize
from algorithms.mahdrl_oe_ge_apf_score_fast.runtime import build_env_kwargs as main_kwargs
from common.precompute_deadlines import build_env_kwargs as reference_kwargs
from common.scenarios import Scenario


class ERTFSuiteTests(unittest.TestCase):
    def test_mask_only_preserves_masks_actions_and_every_task_completion_time(self):
        for task in 'SML':
            for resource in 'SML':
                with self.subTest(task=task, resource=resource), redirect_stdout(io.StringIO()):
                    kwargs = reference_kwargs(Scenario(task=task, resource=resource))
                    kwargs.pop('deadline_alpha')
                    kwargs.update(workflows_per_episode=3, random_seed=700)
                    original, fast = (cls(**kwargs) for cls in (CloudWorkflowEnv_VMAgents, MaskOnlyERTFEnv))
                    original.reset(); fast.reset()
                    _force_fcfs_manager(original); _force_fcfs_manager(fast)
                    for phase in range(20000):
                        while True:
                            h1, ok1 = original.get_host_state_for_next_assignment()
                            h2, ok2 = fast.get_host_state_for_next_assignment()
                            self.assertEqual(ok1, ok2)
                            np.testing.assert_array_equal(h1['mask'], h2['mask'])
                            if not ok1:
                                break
                            self.assertEqual(original._cur_tid, fast._cur_tid)
                            a1, a2 = (_pick_first_idle_host(e, h['mask']) for e, h in ((original, h1), (fast, h2)))
                            self.assertEqual(a1, a2)
                            original.host_select(a1); fast.host_select(a2)
                            v1, ok1 = original.get_vm_state_for_current_task()
                            v2, ok2 = fast.get_vm_state_for_current_task()
                            self.assertEqual(ok1, ok2)
                            np.testing.assert_array_equal(v1['mask'], v2['mask'])
                            a1, a2 = (_pick_fifs_vm(e, v['mask'], a) for e, v, a in ((original, v1, a1), (fast, v2, a2)))
                            self.assertEqual(a1, a2)
                            np.testing.assert_array_equal(original.vm_assign(a1)[:2], fast.vm_assign(a2)[:2])
                        np.testing.assert_equal(original.finish_phase_and_advance()[0], fast.finish_phase_and_advance()[0])
                        if original.done_flag:
                            self.assertTrue(fast.done_flag)
                            break
                        _force_fcfs_manager(original); _force_fcfs_manager(fast)
                    else:
                        self.fail('Episode did not finish')
                    np.testing.assert_array_equal(original.task_end_time, fast.task_end_time)
                    self.assertEqual(original.total_energy, fast.total_energy)

    def test_physical_inputs_match_main_experiment_for_all_profiles_and_sizes(self):
        for profile in PROFILES:
            for task in 'SML':
                for resource in 'SML':
                    scene = Scenario(task=task, resource=resource, workload_profile=profile, seed_spec='700')
                    baseline = build_env_kwargs(scene)
                    main = main_kwargs(scene, 700)
                    for key in ('dax_paths', 'horizon', 'arrival_lambda', 'random_seed', 'workflows_per_episode',
                                'num_hosts', 'vms_per_host', 'pc_tiers', 'bw_tiers', 'deadline_mode',
                                'deadline_cache_path', 'deadline_cache_strict', 'deadline_alpha_small',
                                'deadline_alpha_large', 'deadline_alpha_small_prob'):
                        self.assertEqual(baseline[key], main[key], (profile, task, resource, key))
                    self.assertEqual(baseline['arrival_lambda'], .03 if profile == 'standard' else .06)

    def test_summary_uses_sample_standard_deviation_and_requires_all_seeds_once(self):
        rows = [dict(seed=700, **dict.fromkeys(METRICS, 1.)), dict(seed=701, **dict.fromkeys(METRICS, 3.))]
        result = summarize(rows, (700, 701))
        self.assertEqual(result['total_energy_mean'], 2.)
        self.assertAlmostEqual(result['total_energy_std'], 2.**.5)
        self.assertLess(result['total_energy_ci95_low'], 2.)
        for invalid in (rows[:1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid, (700, 701))
        rows[0]['total_energy'] = float('nan')
        with self.assertRaises(ValueError):
            summarize(rows, (700, 701))


if __name__ == '__main__':
    unittest.main()
