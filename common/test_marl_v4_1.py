"""Bounded FIFO admission, physical time features and V4.1 checkpoint identity."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from algorithms.marl.identity import MODEL_VERSION, QUEUE_CAPACITY, OBSERVATION_LAYOUT
from algorithms.marl.q_agents import QAgent, QConfig, IndependentLocalAgents
from algorithms.marl.runtime import selection_record, validate_pair
from algorithms.marl.scenario import MARLScenario
from algorithms.pd3qn_dag_fe.selection import validation_selection
from common.read_xml_opt_Tsize import Task
from common.f18_state_normalization import bind_normalization
from common.test_f18 import fixture_config
from common.test_marl_v4 import queue_fixture
from common.workflow_opt import Workflow, energy_from_records


def independent_fixture(count=35, **kwargs):
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


class CapacityTests(unittest.TestCase):
    def test_capacity_counts_ten_waiters_and_excludes_running_task(self):
        env = independent_fixture()
        self.assertEqual(env.queue_capacity, 10)
        env._assign_task_to_vm(0, 0)
        self.assertEqual(len(env.vm_waiting_tasks[0]), 0)
        for gid in range(1, 11):
            self.assertEqual(env.get_local_state(0)[1][0], 1.)
            env._assign_task_to_vm(gid, 0)
        self.assertEqual(env.vm_running_task[0], 0)
        self.assertEqual(list(env.vm_waiting_tasks[0]), list(range(1, 11)))
        np.testing.assert_array_equal(env.get_local_state(0)[1], [0., 0.])
        np.testing.assert_array_equal(env.get_global_state()[1], [0., 1.])
        before = (list(env.task_state), list(env.ready_task_ids), env.vm_available_at.copy(), len(env.records))
        with self.assertRaisesRegex(ValueError, 'full'):
            env._assign_task_to_vm(11, 0)
        with self.assertRaisesRegex(ValueError, 'Host'):
            env.step(0, 0)
        self.assertEqual((env.task_state, env.ready_task_ids), before[:2])
        np.testing.assert_array_equal(env.vm_available_at, before[2])
        self.assertEqual(len(env.records), before[3])
        # One full VM does not hide another admissible VM on the same Host.
        for gid in range(11, 22): env._assign_task_to_vm(gid, 1)
        np.testing.assert_array_equal(env.get_local_state(1)[1], [0., 1.])
        np.testing.assert_array_equal(env.get_global_state()[1], [0., 1.])
        with self.assertRaisesRegex(ValueError, 'VM'):
            env.step(1, 0)

    def test_all_full_queues_advance_until_waiting_slot_is_released(self):
        env = independent_fixture()
        for vm in range(3):
            for gid in range(vm * 11, (vm + 1) * 11): env._assign_task_to_vm(gid, vm)
        self.assertEqual(len(env.ready_task_ids), 2)
        self.assertFalse(env._has_decision_point())
        np.testing.assert_array_equal(env.get_global_state()[1], [0., 0.])
        rg, _ = env._advance_until_decision()
        self.assertEqual(env.current_time, 1.)
        self.assertFalse(env.done_flag)
        self.assertEqual([len(q) for q in env.vm_waiting_tasks], [9, 9, 9])
        self.assertTrue(np.all(env.vm_running_task >= 0))
        np.testing.assert_array_equal(env.get_global_state()[1], [1., 1.])
        self.assertLess(rg, 0.)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts, end_time=1.))
        while not env.done_flag:
            host = int(np.flatnonzero(env.get_global_state()[1])[0])
            vm = int(np.flatnonzero(env.get_local_state(env.host_ids[host])[1])[0])
            env.step(host, vm)
        self.assertTrue(all(state == 'Finished' for state in env.task_state))
        self.assertTrue(all(not q for q in env.vm_waiting_tasks))
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts))

    def test_zero_duration_waiter_preserves_fifo_and_frees_capacity(self):
        env = independent_fixture(count=3, queue_capacity=2)
        env.workflows[0].tasks[2].workload_mi = 0.
        env._assign_task_to_vm(0, 0)
        env._assign_task_to_vm(2, 0)
        env._assign_task_to_vm(1, 0)
        with patch.object(env, '_start_task', wraps=env._start_task) as start:
            env._advance_until_decision()
        self.assertEqual([call.args[0] for call in start.call_args_list], [2, 1])
        self.assertEqual(env.current_time, 2.)
        self.assertTrue(env.done_flag)
        self.assertFalse(env.vm_waiting_tasks[0])

    def test_capacity_is_a_positive_finite_integer(self):
        for value in (0, -1, True, 1.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                queue_fixture(queue_capacity=value)


class ObservationAndIdentityTests(unittest.TestCase):
    def test_wait_and_execution_are_separate_and_use_frozen_time_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'scale.json'; fixture_config(config)
            env = queue_fixture(normalization_mode='f19_shared_v1', normalization_config=str(config), source_train_scene='SS_T')
            env._assign_task_to_vm(0, 0)
            tid = 1
            duration = env._task_duration(tid, 0)
            local = env.get_local_state(0)[0]
            vm = local[4:].reshape(-1, 5)[0]
            self.assertAlmostEqual(vm[3], 2. / 123.)
            self.assertAlmostEqual(vm[4], duration / 123.)
            end = env._assign_task_to_vm(tid, 0)
            self.assertAlmostEqual(float(vm[3] + vm[4]) * 123., end, places=6)
            self.assertEqual(local.shape, (14,))
            self.assertTrue(np.all(local[9:] == 0.))  # Missing VM slots only.

    def test_swapping_queue_waits_changes_local_observation(self):
        env = queue_fixture()
        env.vm_available_at[1:] = [1., 100.]
        first = env.get_local_state(1)[0]
        env.vm_available_at[1:] = [100., 1.]
        second = env.get_local_state(1)[0]
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first[:4], second[:4])
        np.testing.assert_array_equal(first[4:].reshape(-1, 5)[:, 4], second[4:].reshape(-1, 5)[:, 4])

    def test_metadata_dimensions_and_checkpoint_capacity_binding(self):
        scene = MARLScenario()
        self.assertEqual(MODEL_VERSION, 'marl_q_v4_1')
        self.assertIn(MODEL_VERSION, scene.model_name())
        self.assertNotIn('marl_q_v4_task', scene.model_name())
        for resource, global_size in [('S', 11), ('M', 17), ('L', 23)]:
            description = MARLScenario(resource=resource).describe()
            self.assertEqual(description['global_obs_dim'], global_size)
            self.assertEqual(description['local_obs_dim'], 49)
            self.assertEqual(description['queue_capacity'], QUEUE_CAPACITY)
            self.assertEqual(description['observation_layout'], OBSERVATION_LAYOUT)
        cfg = QConfig(hidden_dims=())
        global_agent = QAgent(2, 2, cfg, peer_act_dim=2)
        locals = IndependentLocalAgents(2, 2, 2, cfg)
        rows = [dict(seed=600, wf_count=50, wf_success_count=50, wf_success_rate=1., total_energy=1.)]
        candidate = validation_selection(rows)
        metadata = selection_record(scene, 0, candidate, candidate, rows, 'best')
        global_agent.training_selection = locals.training_selection = metadata
        validate_pair(global_agent, locals, scene)
        for key, wrong in [('queue_capacity', 11), ('observation_layout', 'old'), ('scheduling_policy', 'old')]:
            global_agent.training_selection = locals.training_selection = dict(metadata, **{key: wrong})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_pair(global_agent, locals, scene)

    def test_checkpoint_load_rejects_changed_queue_capacity(self):
        scene = MARLScenario()
        cfg = QConfig(hidden_dims=())
        agent = bind_normalization(QAgent(2, 2, cfg), scene)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'global.pth'
            agent.save(path)
            agent.load(path)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['normalization']['queue_capacity'] = 11
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'queue_capacity'):
                agent.load(path)


if __name__ == '__main__':
    unittest.main()
