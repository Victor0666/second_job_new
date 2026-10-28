"""Idle-only admission, global Ready waiting, physical features and V4.2 identity."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from algorithms.marl.identity import MODEL_VERSION, OBSERVATION_LAYOUT, SCHEDULING_POLICY
from algorithms.marl.q_agents import QAgent, QConfig, IndependentLocalAgents
from algorithms.marl.runtime import selection_record, validate_pair
from algorithms.marl.scenario import MARLScenario
from algorithms.marl.selection import validation_selection
from common.read_xml_opt_Tsize import Task
from common.f18_state_normalization import bind_normalization
from common.test_f18 import fixture_config
from common.test_marl_v4 import queue_fixture
from common.workflow_opt import Workflow, energy_from_records


def independent_fixture(count=6, **kwargs):
    env = queue_fixture(**kwargs)
    tasks = [Task(task_id=i, workload_mi=1.) for i in range(count)]
    graph = nx.DiGraph(); graph.add_nodes_from(range(count))
    env.workflows = [Workflow(0, graph, tasks, 0., 0, deadline=100.)]
    env.task_meta = [(0, i) for i in range(count)]
    env.gid_of = {(0, i): i for i in range(count)}
    env.task_state = ['Ready'] * count
    env.task_ready_time = [0.] * count
    env.task_end_time = [0.] * count
    env.task_baseline_finish = [100.] * count
    env.ready_task_ids = list(range(count))
    env.wf_remaining_tasks = {0: count}
    env._cur_task_gid = None
    env._update_current_task_fcfs()
    return env


class IdleAdmissionTests(unittest.TestCase):
    def test_busy_and_padding_masks_and_rejected_actions_preserve_state(self):
        env = independent_fixture()
        env._assign_task_to_vm(0, 0)
        env._assign_task_to_vm(1, 1)
        np.testing.assert_array_equal(env.get_local_state(0)[1], [0., 0.])
        np.testing.assert_array_equal(env.get_local_state(1)[1], [0., 1.])
        np.testing.assert_array_equal(env.get_global_state()[1], [0., 1.])
        before = (list(env.task_state), list(env.ready_task_ids), env.vm_available_at.copy(), len(env.records))
        for action in (lambda: env._assign_task_to_vm(2, 0), lambda: env.step(0, 0),
                       lambda: env.step(1, 0), lambda: env.step(0, 1)):
            with self.assertRaises(ValueError): action()
            self.assertEqual((env.task_state, env.ready_task_ids), before[:2])
            np.testing.assert_array_equal(env.vm_available_at, before[2])
            self.assertEqual(len(env.records), before[3])
        # An idle VM cannot bypass task dependencies or execute a task twice.
        blocked = queue_fixture()
        with self.assertRaisesRegex(ValueError, 'Ready'): blocked._assign_task_to_vm(2, 0)
        with self.assertRaisesRegex(ValueError, 'Ready'): env._assign_task_to_vm(0, 2)

    def test_all_busy_waits_globally_until_completion_and_keeps_fcfs(self):
        env = independent_fixture()
        for gid, duration in enumerate((3., 5., 7.)):
            env.workflows[0].tasks[gid].workload_mi = duration
            env._assign_task_to_vm(gid, gid)
        self.assertEqual(env.ready_task_ids, [3, 4, 5])
        self.assertEqual(env.task_end_time[3:], [0., 0., 0.])
        self.assertEqual(set(env.parents_loc_by_wf[0]), {0, 1, 2})
        self.assertFalse(env._has_decision_point())
        np.testing.assert_array_equal(env.get_global_state()[1], [0., 0.])
        rg, _ = env._advance_until_decision()
        self.assertEqual(env.current_time, 3.)
        self.assertFalse(env.done_flag)
        self.assertEqual(env.ready_task_ids, [3, 4, 5])
        np.testing.assert_array_equal(env.get_global_state()[1], [1., 0.])
        self.assertEqual(env._cur_task_gid, 3)
        self.assertLess(rg, 0.)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts, end_time=3.))
        self.assertEqual(env._assign_task_to_vm(3, 0), 4.)
        self.assertEqual(env.records[-1].start_time, 3.)
        env._advance_until_decision()
        while not env.done_flag:
            host = int(np.flatnonzero(env.get_global_state()[1])[0])
            vm = int(np.flatnonzero(env.get_local_state(env.host_ids[host])[1])[0])
            env.step(host, vm)
        self.assertTrue(all(state == 'Finished' for state in env.task_state))
        self.assertFalse(hasattr(env, 'vm_waiting_tasks'))
        self.assertFalse(hasattr(env, 'queue_capacity'))
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts))

    def test_arrivals_while_all_busy_do_not_create_illegal_decisions(self):
        env = independent_fixture(count=4)
        env.task_state[3] = 'unReady'; env.ready_task_ids.remove(3)
        env.arrival_times = [0., 2.]
        for gid in range(3):
            env.workflows[0].tasks[gid].workload_mi = 3.
            env._assign_task_to_vm(gid, gid)
        seen = []
        def arrival():
            seen.append(env.current_time)
            if env.next_arrival_idx == 1 and env.current_time >= 2.:
                env.next_arrival_idx = 2
                env.task_state[3] = 'Ready'
                env.task_ready_time[3] = 2.
                env.ready_task_ids.append(3)
        with patch.object(env, '_add_workflow_if_arrived', side_effect=arrival):
            env._advance_until_decision()
        self.assertEqual(seen, [0., 2., 3.])
        self.assertEqual(env.current_time, 3.)
        self.assertEqual(env.task_state, ['Finished'] * 3 + ['Ready'])
        self.assertEqual(env.task_end_time[3], 0.)
        self.assertNotIn(3, env.parents_loc_by_wf[0])

    def test_zero_duration_task_finishes_before_the_next_decision(self):
        env = independent_fixture(count=2)
        env.workflows[0].tasks[0].workload_mi = 0.
        env.step(0, 0)
        self.assertEqual(env.current_time, 0.)
        self.assertEqual(env.task_state, ['Finished', 'Ready'])
        self.assertEqual(env.get_local_state(0)[1][0], 1.)
        env.step(0, 0)
        self.assertTrue(env.done_flag)
        self.assertEqual(env.current_time, 1.)

    def test_horizon_keeps_unassigned_ready_tasks_unbound(self):
        env = independent_fixture(count=4, horizon=.5)
        for gid in range(3): env._assign_task_to_vm(gid, gid)
        env._advance_until_decision()
        self.assertTrue(env.done_flag)
        self.assertEqual(env.current_time, .5)
        self.assertEqual(env.task_state[3], 'Ready')
        self.assertEqual(env.task_end_time[3], 0.)
        self.assertNotIn(3, env.parents_loc_by_wf[0])
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts, end_time=.5))


class ObservationAndIdentityTests(unittest.TestCase):
    def test_busy_wait_visible_but_masked_and_idle_task_starts_now(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'scale.json'; fixture_config(config)
            env = queue_fixture(normalization_mode='f19_shared_v1', normalization_config=str(config), source_train_scene='SS_T')
            env._assign_task_to_vm(0, 0)
            local, mask, _ = env.get_local_state(0)
            vm = local[4:].reshape(-1, 5)[0]
            self.assertEqual(mask[0], 0.)
            self.assertEqual(vm[2], 0.)
            self.assertAlmostEqual(vm[3], 2. / 123.)
            self.assertAlmostEqual(vm[4], env._task_duration(1, 0) / 123.)
            self.assertEqual(local.shape, (14,))
            self.assertTrue(np.all(local[9:] == 0.))
            idle, mask, mapping = env.get_local_state(1)
            self.assertEqual(mask[0], 1.)
            self.assertEqual(idle[4+3], 0.)
            expected = env._task_duration(1, mapping[0])
            self.assertEqual(env._assign_task_to_vm(1, mapping[0]), env.current_time + expected)
            self.assertEqual(env.records[-1].start_time, env.current_time)

    def test_swapping_busy_remaining_times_changes_local_observation(self):
        env = independent_fixture()
        for gid, vm, duration in ((0, 1, 1.), (1, 2, 100.)):
            env.workflows[0].tasks[gid].workload_mi = duration
            env._assign_task_to_vm(gid, vm)
        first = env.get_local_state(1)[0]
        env.vm_available_at[1:] = [100., 1.]
        second = env.get_local_state(1)[0]
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first[:4], second[:4])
        np.testing.assert_array_equal(first[4:].reshape(-1, 5)[:, 4], second[4:].reshape(-1, 5)[:, 4])

    def test_metadata_dimensions_and_checkpoint_scheduling_binding(self):
        scene = MARLScenario()
        self.assertEqual(MODEL_VERSION, 'marl_q_v4_2')
        self.assertIn(MODEL_VERSION, str(scene.checkpoint_dir()))
        for resource, global_size in [('S', 11), ('M', 17), ('L', 23)]:
            description = MARLScenario(resource=resource).describe()
            self.assertEqual(description['global_obs_dim'], global_size)
            self.assertEqual(description['local_obs_dim'], 49)
            self.assertNotIn('queue_capacity', description)
            self.assertEqual(description['scheduling_policy'], SCHEDULING_POLICY)
            self.assertEqual(description['observation_layout'], OBSERVATION_LAYOUT)
        cfg = QConfig(hidden_dims=())
        global_agent = QAgent(2, 2, cfg, peer_act_dim=2)
        locals = IndependentLocalAgents(2, 2, 2, cfg)
        rows = [dict(seed=600, wf_count=50, wf_success_count=50, wf_success_rate=1., total_energy=1.)]
        candidate = validation_selection(rows)
        metadata = selection_record(scene, 0, candidate, candidate, rows, 'best')
        global_agent.training_selection = locals.training_selection = metadata
        validate_pair(global_agent, locals, scene)
        for key, wrong in [('observation_layout', 'old'), ('scheduling_policy', 'vm_fifo_queue_cap10_v1')]:
            global_agent.training_selection = locals.training_selection = dict(metadata, **{key: wrong})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_pair(global_agent, locals, scene)

    def test_checkpoint_load_rejects_changed_scheduling_even_with_same_dimensions(self):
        scene = MARLScenario()
        cfg = QConfig(hidden_dims=())
        agent = bind_normalization(QAgent(2, 2, cfg), scene)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'global.pth'
            agent.save(path)
            agent.load(path)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['normalization']['scheduling_policy'] = 'vm_fifo_queue_cap10_v1'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'scheduling_policy'):
                agent.load(path)


if __name__ == '__main__':
    unittest.main()
