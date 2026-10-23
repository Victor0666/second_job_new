"""Physical equivalence and online graph checks for the DRL-GNN adapter."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from algorithms.drl_gnn.environment import DRLGNNEnv
from common.workflow_opt import exec_time_components
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents


class DRLGNNEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="drl_gnn_environment_")
        self.addCleanup(self.directory.cleanup)
        self.dax = Path(self.directory.name) / "Fork_3.xml"
        self.dax.write_text(
            '<adag xmlns="http://pegasus.isi.edu/schema/DAX">'
            '<job id="a"/><job id="b"/><job id="c"/>'
            '<child ref="c"><parent ref="a"/><parent ref="b"/></child></adag>',
            encoding="utf-8",
        )

    def kwargs(self, **changes):
        result = dict(
            dax_paths=[str(self.dax)], horizon=1e9, workflows_per_episode=1, arrival_lambda=.03,
            num_hosts=1, vms_per_host=(2,), pc_tiers=(1., 8.), bw_tiers=(1000., 1000.),
            deadline_mode="none", normalization_mode="legacy", max_ready_tasks="auto",
            random_seed=2,
        )
        result.update(changes)
        return result

    def test_raw_features_full_compatibility_and_all_vm_estimates(self):
        env = DRLGNNEnv(**self.kwargs())
        env.reset(seed=2)
        env.vm_available_at[:] = env.current_time + np.array([1000., 100.])
        state = env.graph_state()
        self.assertEqual(state.task_gids, (-1, 0, 1, 2, -2))
        self.assertEqual(state.task_features.dtype, np.float32)
        self.assertEqual(state.vm_features.dtype, np.float32)
        self.assertTrue(state.compatibility_mask.all())
        self.assertFalse(state.action_mask.any())
        np.testing.assert_array_equal(state.task_features[:, :2],
                                      [[1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
        self.assertEqual(float(state.task_features[1, 2]), float(np.float32(env.task_mi[0])))
        candidates = [max(env.current_time, env.vm_available_at[j]) + exec_time_components(
            env.workflows[0].tasks[0], env.vms[j].pc, env.vms[j].bw,
            parents_loc={}, child_vm_id=j, child_host_id=env.vms[j].host_id,
        )[-1] for j in range(env.num_vms)]
        self.assertAlmostEqual(float(state.task_features[1, 3]), min(candidates), places=3)
        np.testing.assert_allclose(state.vm_features[:, 1], [1., 1. / 8.])
        host = env.hosts[0]
        np.testing.assert_allclose(state.vm_features[:, 2],
                                   (host.power(1.) - host.power(0.)) / host.total_pc)
        self.assertEqual(set(map(tuple, state.dependencies)), {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)})

    def test_pair_validation_is_atomic_and_reward_is_public(self):
        env = DRLGNNEnv(**self.kwargs())
        env.reset(seed=2)
        state = env.graph_state()
        with self.assertRaises(ValueError):
            env.pair_step(0)  # Shared source is never an executable task.
        self.assertEqual(env.assignment_count, 0)
        self.assertIsNone(env.current_task_gid)
        before = env.total_energy
        reward, _, info = env.pair_step(2)  # Task 0, VM 0.
        self.assertAlmostEqual(info["delta_energy_total"], env.total_energy - before)
        expected = .25 * (-.001 * info["delta_energy_total"]) + .75 * info["r_delay"]
        self.assertAlmostEqual(reward, expected)
        self.assertEqual((info["task_gid"], info["vm_id"], info["action_count"]), (0, 0, 1))
        self.assertEqual(len(env.records), 1)
        self.assertEqual(float(state.task_features[1, 0]), 0.)  # Historical snapshot is unchanged.
        physical = (list(env.task_state), env.total_energy, len(env.records), env.current_time)
        with self.assertRaises(ValueError):
            env.pair_step(4)  # Task 1, busy VM 0.
        self.assertEqual((list(env.task_state), env.total_energy, len(env.records), env.current_time), physical)
        self.assertIsNone(env.current_task_gid)
        env.pair_step(5)  # Task 1, VM 1.
        next_state = env.graph_state()
        self.assertEqual(float(next_state.task_features[1, 3]), float(np.float32(env.task_end_time[0])))
        legal = np.flatnonzero(next_state.action_mask.reshape(-1))
        self.assertGreater(len(legal), 0)
        _, done, _ = env.pair_step(int(legal[0]))
        self.assertTrue(done)
        final = env.graph_state()
        self.assertEqual(float(final.task_features[-1, 0]), 1.)
        self.assertFalse(final.action_mask.any())
        previous_vm = int(legal[0]) % env.num_vms
        self.assertIn((previous_vm + 1, 3), set(map(tuple, final.dependencies)))

    def test_later_workflow_arrival_updates_shared_dummy_without_early_end(self):
        env = DRLGNNEnv(**self.kwargs(workflows_per_episode=2, arrival_lambda=1e-6))
        env.reset(seed=2)
        first = env.graph_state()
        self.assertEqual(len(env.workflows), 1)
        self.assertEqual(first.task_features.shape, (5, 4))
        saw_second = False
        while not env.done_flag:
            state = env.graph_state()
            self.assertEqual(float(state.task_features[-1, 0]), 0.)
            actions = np.flatnonzero(state.action_mask.reshape(-1))
            self.assertGreater(len(actions), 0)
            _, _, _ = env.pair_step(int(actions[0]))
            saw_second |= len(env.workflows) == 2
        self.assertTrue(saw_second)
        self.assertEqual(env.completed_workflows, 2)
        self.assertEqual(env.assignment_count, 6)
        final = env.graph_state()
        self.assertEqual(final.task_features.shape, (8, 4))
        self.assertEqual(first.task_features.shape, (5, 4))
        self.assertEqual(final.task_gids.count(-1), 1)
        self.assertEqual(final.task_gids.count(-2), 1)

    def test_real_dax_fixed_assignments_match_mahdrl_physics(self):
        dax = Path(__file__).resolve().parents[1] / "data/dax/Montage_25.xml"
        kwargs = self.kwargs(dax_paths=[str(dax)], workflows_per_episode=2,
                             num_hosts=2, vms_per_host=(2, 1),
                             pc_tiers=(2., 4.), bw_tiers=(1000., 2000.))
        env = DRLGNNEnv(**kwargs)
        reference = CloudWorkflowEnv_VMAgents(**kwargs)
        env.reset(seed=2)
        reference.reset(seed=2)
        total_reward = 0.
        total_delay_reward = 0.
        while not env.done_flag:
            state = env.graph_state()
            action = int(np.flatnonzero(state.action_mask.reshape(-1))[0])
            task_index, vm_id = divmod(action, env.num_vms)
            gid = state.task_gids[task_index]
            self.assertIn(gid, reference.ready_task_ids)
            self.assertLessEqual(reference.vm_available_at[vm_id], reference.current_time + 1e-12)
            reference._assign_task_to_specific_vm(gid, vm_id)
            reference._advance_until_decision_energy_only()
            reward, _, info = env.pair_step(action)
            total_reward += reward
            total_delay_reward += info["r_delay"]
            self.assertAlmostEqual(env.current_time, reference.current_time, places=8)
            np.testing.assert_allclose(env.vm_available_at, reference.vm_available_at, rtol=0, atol=1e-8)
            np.testing.assert_allclose(env.task_end_time, reference.task_end_time, rtol=0, atol=1e-8)
            np.testing.assert_allclose(env.task_baseline_finish, reference.task_baseline_finish, rtol=0, atol=1e-6)
        self.assertTrue(reference.done_flag)
        self.assertEqual(env.completed_workflows, reference.completed_workflows)
        self.assertAlmostEqual(env.total_energy, reference.total_energy, places=7)
        self.assertAlmostEqual(total_reward, -.25 * .001 * env.total_energy + .75 * total_delay_reward, places=9)


if __name__ == "__main__":
    unittest.main()
