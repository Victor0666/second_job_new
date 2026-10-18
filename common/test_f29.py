"""F29: padded local VM actions must preserve the real host slot mapping."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
import numpy as np

from algorithms.ertf_fifs.evaluate import (_pick_fifs_vm, _pick_first_idle_host,
                                           _force_fcfs_manager)
from common.precompute_deadlines import build_env_kwargs, run_baseline_episode_fcfs_vmfirst
from common.scenarios import Scenario
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
from environments.base_fcfs import CloudWorkflowEnv_VMAgents as ReferenceEnv


def fixture(mapping=None):
    mapping = list(range(9,17)) if mapping is None else mapping
    available = np.zeros(25)
    available[mapping] = np.arange(1, len(mapping)+1)
    available[mapping[0]], available[mapping[1]] = 4., 1.
    return SimpleNamespace(host_ids=[10,20,30], max_vms_per_host=9,
                           host_to_vm_indices={20:mapping}, vm_available_at=available)


class F29Tests(unittest.TestCase):
    def test_eight_vm_padded_mask_uses_earliest_idle_not_lowest_slot(self):
        env = fixture()
        self.assertEqual(_pick_fifs_vm(env, [1]*8+[0], 1), 1)

    def test_nine_vm_host(self):
        env = fixture(list(range(9)))
        env.vm_available_at[8] = -1.
        self.assertEqual(_pick_fifs_vm(env, np.ones(9), 1), 8)

    def test_returns_local_slot_using_authoritative_order(self):
        env = fixture([16,9,15,10,14,11,13,12])
        # Global index 9 is local slot 1, not slot 0 in ascending global order.
        self.assertEqual(_pick_fifs_vm(env, [1]*8+[0], 1), 1)

    def test_busy_vm_excluded_and_ties_use_lower_slot(self):
        env = fixture()
        env.vm_available_at[10] = -5.
        mask = np.array([1,0,1,1,1,1,1,1,0])
        env.vm_available_at[11] = env.vm_available_at[12] = 2.
        self.assertEqual(_pick_fifs_vm(env, mask, 1), 2)

    def test_invalid_masks_mapping_and_host_fail_explicitly(self):
        cases = [([1]*9, 1), ([0]*9, 1), ([1]*8, 1),
                 (np.ones((1,9)), 1), ([1]*8+[float("nan")], 1), ([1]*8+[0], 3)]
        for mask, host in cases:
            with self.subTest(mask=mask, host=host), self.assertRaises(ValueError):
                _pick_fifs_vm(fixture(), mask, host)
        for bad_index in (-1,25):
            env = fixture(); env.host_to_vm_indices[20][1] = bad_index
            with self.assertRaises(ValueError): _pick_fifs_vm(env, [1]*8+[0], 1)
        env = fixture(); env.vm_available_at[10] = np.inf
        with self.assertRaises(ValueError): _pick_fifs_vm(env, [1]*8+[0], 1)

    def test_real_two_stage_episodes_match_global_first_idle_reference(self):
        for resource in "SML":
            with self.subTest(resource=resource), redirect_stdout(io.StringIO()):
                kwargs = build_env_kwargs(Scenario(resource=resource))
                kwargs.update(workflows_per_episode=2, random_seed=0)
                reference = ReferenceEnv(**kwargs); reference.reset()
                expected_makespans, _, _ = run_baseline_episode_fcfs_vmfirst(reference)
                kwargs.pop("deadline_alpha")
                env = CloudWorkflowEnv_VMAgents(**kwargs); env.reset()
                _force_fcfs_manager(env)
                assignments = 0
                for phase in range(10000):
                    while True:
                        host, ready = env.get_host_state_for_next_assignment()
                        if not ready: break
                        ah = _pick_first_idle_host(env, host["mask"]); env.host_select(ah)
                        vm, ready = env.get_vm_state_for_current_task()
                        self.assertTrue(ready)
                        av = _pick_fifs_vm(env, vm["mask"], ah)
                        mapping = env.host_to_vm_indices[env.host_ids[ah]]
                        self.assertLess(av, len(mapping))
                        # Independent global oracle: equivalent here under ordered VM IDs/ties.
                        idle = np.flatnonzero(env._vm_idle_mask())
                        expected = min(idle, key=lambda j:(env.vm_available_at[j], int(j)))
                        self.assertEqual(mapping[av], expected)
                        _, _, info = env.vm_assign(av)
                        self.assertEqual(info["invalid"], 0)
                        assignments += 1
                    env.finish_phase_and_advance()
                    if env.done_flag: break
                    _force_fcfs_manager(env)
                else: self.fail("Episode did not finish")
                self.assertGreater(assignments, 0)
                self.assertTrue(env._episode_should_end())
                actual_makespans = [env.wf_finish_time[i]-wf.arrival_time for i,wf in enumerate(env.workflows)]
                np.testing.assert_allclose(actual_makespans, expected_makespans, rtol=0, atol=1e-8)
                np.testing.assert_allclose(env.task_end_time, reference.task_end_time, rtol=0, atol=1e-8)
                self.assertAlmostEqual(env.total_energy, reference.total_energy, places=7)


if __name__ == "__main__":
    unittest.main()
