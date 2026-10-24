"""Physical equivalence and online graph checks for the DRL-GNN adapter."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.drl_gnn.config import (COMMON_REWARD, DRLGNNConfig, INPUT_ENCODING,
                                       TASK_FEATURES, VM_FEATURES)
from algorithms.drl_gnn.environment import DRLGNNEnv
from algorithms.drl_gnn.ppo import PPOAgent, Transition
from algorithms.drl_gnn.runner import evaluate, load_checkpoint, save_checkpoint, train
from common.scenarios import Scenario
from common.workflow_opt import exec_time_components
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents


class DRLGNNEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

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

    def test_public_task_deadlines_and_payload_are_signed_current_snapshots(self):
        env = DRLGNNEnv(**self.kwargs())
        env.reset(seed=2)
        now = float(env.current_time)
        env.task_baseline_finish[0] = now - 7.
        env.workflows[0].deadline = now + 13.
        state = env.graph_state()
        self.assertEqual(state.task_features.shape, (5, len(TASK_FEATURES)))
        np.testing.assert_allclose(state.task_features[1:-1, 4],
                                   np.asarray(env.task_baseline_finish) - now)
        np.testing.assert_array_equal(state.task_features[1:-1, 5], [13.] * 3)
        np.testing.assert_allclose(state.task_features[1:-1, 6], env.task_in_bits)
        np.testing.assert_allclose(state.task_features[1:-1, 7], env.task_out_bits)
        np.testing.assert_array_equal(state.task_features[[0, -1], 4:], 0.)
        env.task_baseline_finish[0] = now + 5.
        changed = env.graph_state()
        self.assertEqual(float(state.task_features[1, 4]), -7.)
        self.assertEqual(float(changed.task_features[1, 4]), 5.)
        np.testing.assert_array_equal(state.task_features[:, :4], changed.task_features[:, :4])
        np.testing.assert_array_equal(state.vm_features, changed.vm_features)
        env.workflows[0].deadline = now - 3.
        np.testing.assert_array_equal(env.graph_state().task_features[1:-1, 5], [-3.] * 3)

    def test_public_vm_bandwidth_busy_time_and_shared_host_state(self):
        env = DRLGNNEnv(**self.kwargs(num_hosts=2, vms_per_host=(2, 1),
                                     bw_tiers=(1000., 2000.)))
        env.reset(seed=2)
        now = float(env.current_time)
        env.vm_available_at[:] = [now + 11., now - 1., now]
        state = env.graph_state()
        self.assertEqual(state.vm_features.shape, (3, len(VM_FEATURES)))
        np.testing.assert_array_equal(state.vm_features[:, 3], [1000., 2000., 1000.])
        np.testing.assert_array_equal(state.vm_features[:, 4], [11., 0., 0.])
        load = env.vms[0].pc / env.hosts[0].total_pc
        np.testing.assert_allclose(state.vm_features[:, 5], [load, load, 0.])
        np.testing.assert_allclose(state.vm_features[:, 6],
            [env.hosts[0].power(load)] * 2 + [env.hosts[1].power(0.)])
        np.testing.assert_allclose(state.vm_features[:, 7],
            [env.hosts[0].power(0.)] * 2 + [env.hosts[1].power(0.)])
        np.testing.assert_array_equal(state.vm_features[:, 8], [0., 0., 1.])
        self.assertFalse(state.action_mask[:, 0].any())
        self.assertTrue(state.action_mask[1, 1:].all())
        # Moving a VM to busy updates every VM's view of the same host only.
        env.vm_available_at[1] = now + 9.
        changed = env.graph_state()
        np.testing.assert_array_equal(changed.vm_features[:2, 5], [1., 1.])
        np.testing.assert_array_equal(state.vm_features[2], changed.vm_features[2])
        self.assertEqual(float(state.vm_features[1, 4]), 0.)

    def test_later_workflow_arrival_updates_shared_dummy_without_early_end(self):
        env = DRLGNNEnv(**self.kwargs(workflows_per_episode=2, arrival_lambda=1e-6))
        env.reset(seed=2)
        first = env.graph_state()
        self.assertEqual(len(env.workflows), 1)
        self.assertEqual(first.task_features.shape, (5, len(TASK_FEATURES)))
        saw_second = False
        while not env.done_flag:
            state = env.graph_state()
            self.assertEqual(float(state.task_features[-1, 0]), 0.)
            for gid, meta in enumerate(env.task_meta):
                expected = env.workflows[meta.wf_id].deadline - env.current_time
                self.assertAlmostEqual(float(state.task_features[gid + 1, 5]),
                                       float(np.float32(expected)))
            actions = np.flatnonzero(state.action_mask.reshape(-1))
            self.assertGreater(len(actions), 0)
            _, _, _ = env.pair_step(int(actions[0]))
            saw_second |= len(env.workflows) == 2
        self.assertTrue(saw_second)
        self.assertEqual(env.completed_workflows, 2)
        self.assertEqual(env.assignment_count, 6)
        final = env.graph_state()
        self.assertEqual(final.task_features.shape, (8, len(TASK_FEATURES)))
        self.assertEqual(first.task_features.shape, (5, len(TASK_FEATURES)))
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

    def test_expanded_inputs_train_and_checkpoint_roundtrip_without_old_version_mix(self):
        torch.manual_seed(73)
        kwargs = self.kwargs(num_hosts=2, vms_per_host=(2, 1),
                             bw_tiers=(1000., 2000.))
        env = DRLGNNEnv(**kwargs)
        agent = PPOAgent(DRLGNNConfig())
        records = []
        first_state = None
        for seed in (2, 3):
            env.reset(seed=seed)
            while not env.done_flag:
                state = env.graph_state()
                if first_state is None:
                    first_state = state
                action, logp, value = agent.act(state)
                self.assertTrue(state.action_mask.reshape(-1)[action])
                reward, done, _ = env.pair_step(action)
                next_value = 0. if done else agent.value(env.graph_state())
                records.append(Transition(state, action, logp, value, reward, done, next_value))
        before = {name: p.detach().clone() for name, p in agent.network.named_parameters()}
        result = agent.update(records)
        self.assertEqual(result['transitions'], 6)
        self.assertEqual(result['epochs'], agent.cfg.update_epochs)
        self.assertTrue(np.isfinite([result[k] for k in ('loss', 'entropy', 'grad_norm')]).all())
        for role in ('actor', 'critic'):
            for kind, native in (('task', 4), ('vm', 3)):
                name = f'{role}.network.{kind}_encoder.0.weight'
                after = dict(agent.network.named_parameters())[name].detach()
                self.assertGreater(float((after[:, native:] - before[name][:, native:]).abs().max()), 0., name)
        # Old shapes fail before they can reach an encoder with different semantics.
        old = replace(first_state, task_features=first_state.task_features[:, :4],
                      vm_features=first_state.vm_features[:, :3])
        with self.assertRaisesRegex(ValueError, 'public-state policy'):
            agent.network(old)
        scene = Scenario(mode='train', episodes=1)
        self.assertIn(INPUT_ENCODING, scene.model_name('drl_gnn'))
        self.assertNotIn('source_raw_task4_vm3_bn_v1', scene.model_name('drl_gnn'))
        self.assertEqual(scene.describe('drl_gnn')['normalization']['policy_input'], INPUT_ENCODING)
        path = Path(self.directory.name) / 'public_state.pth'
        save_checkpoint(path, agent, scene, 1, 1., kwargs, {'episodes': 1})
        loaded, obj = load_checkpoint(path, scene)
        self.assertEqual(obj['observation_fields'], dict(task=list(TASK_FEATURES), vm=list(VM_FEATURES)))
        for key, value in agent.network.state_dict().items():
            torch.testing.assert_close(loaded.network.state_dict()[key], value, rtol=0, atol=0)
        a, _ = evaluate(agent, scene, [700], kwargs=kwargs)
        b, _ = evaluate(loaded, scene, [700], kwargs=kwargs)
        self.assertEqual(a, b)
        legacy = deepcopy(obj)
        legacy['model_version'] = 'drl_gnn_source_be67762_v1'
        legacy['policy_input'] = 'source_raw_task4_vm3_bn_v1'
        torch.save(legacy, path)
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            load_checkpoint(path, scene)

    def test_runner_small_graph_train_validation_and_tail_rollout(self):
        scene = Scenario(mode='train', episodes=1)
        kwargs = self.kwargs(**COMMON_REWARD)
        # The tiny synthetic DAX has no FCFS cache; only mock its known task budget.
        with patch('algorithms.drl_gnn.runner.planned_assignments', return_value=3):
            agent, directory = train(scene, DRLGNNConfig(), kwargs,
                                     Path(self.directory.name) / 'models')
        self.assertEqual((agent.steps, agent.updates), (3, 1))
        self.assertTrue((directory / 'best.pth').is_file())
        loaded, obj = load_checkpoint(directory / 'final.pth', scene)
        self.assertEqual((loaded.steps, loaded.updates), (3, 1))
        self.assertEqual(obj['protocol']['selection'], 'minimum_validation_total_energy')
        summary = json.loads((directory / 'logs/training_summary.json').read_text())
        self.assertEqual(summary['policy_input'], INPUT_ENCODING)
        self.assertEqual(summary['completed_episodes'], 1)
        self.assertTrue(summary['final_rollout_retained'])


if __name__ == "__main__":
    unittest.main()
