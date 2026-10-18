"""PD3QN regressions for DAG completion events and episode energy windows."""

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from common.read_xml_opt_Tsize import Task
from common.scenarios import Scenario
from common.workflow_opt import energy_from_records
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


class PD3QNCompletionTests(unittest.TestCase):
    def make_env(self, durations, edges=(), arrivals=(0.0,), requested=None):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(durations)))
        graph.add_edges_from(edges)
        tasks = [
            Task(task_id=i, workload_mi=duration,
                 parents=list(graph.predecessors(i)), children=list(graph.successors(i)))
            for i, duration in enumerate(durations)
        ]
        self.enterContext(patch(
            "environments.pd3qn.poisson_arrival_times", return_value=list(arrivals)
        ))
        self.enterContext(patch(
            "environments.pd3qn.load_workflow_from_dax", return_value=(graph, tasks)
        ))
        return CloudWorkflowEnv_PD3QN_Cost(
            dax_paths=["synthetic.xml"], deadline_mode="none",
            workflows_per_episode=len(arrivals) if requested is None else requested,
            num_hosts=2, vms_per_host=(3, 1), pc_tiers=(1.0,), bw_tiers=(1000.0,),
        )

    def test_join_waits_for_all_parents_and_episode_drains_running_tail(self):
        env = self.make_env([10.0, 20.0, 5.0, 0.0, 30.0], [(0, 2), (1, 2), (2, 3)])
        env.step(0)
        self.assertEqual(env.task_state[0], "Running")
        self.assertEqual(env.task_state[2], "unReady")
        self.assertEqual(env.wf_remaining_tasks[0], 5)
        self.assertEqual(env.completed_workflows, 0)
        env.step(1)
        self.assertEqual(env.task_state[2], "unReady")
        env.step(2)
        self.assertEqual(env.current_time, 20.0)
        self.assertEqual(env.task_state[:3], ["Finished", "Finished", "Ready"])
        self.assertEqual(env.task_ready_time[2], 20.0)
        self.assertEqual(env.task_state[4], "Running")
        self.assertEqual(env.wf_remaining_tasks[0], 3)
        self.assertEqual(env.ready_task_ids, [2])
        env.step(0)
        self.assertEqual(env.current_time, 25.0)
        self.assertEqual(env.task_state[3], "Ready")
        self.assertFalse(env.done_flag)
        _, _, done, truncated, _ = env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(env.current_time, 30.0)
        self.assertTrue(all(state == "Finished" for state in env.task_state))
        self.assertEqual(env.wf_remaining_tasks[0], 0)
        self.assertEqual(env.completed_workflows, 1)
        self.assertEqual(env.get_episode_metrics()["makespan"], 30.0)
        for hid in env.hosts:
            self.assertEqual(max(r.end_time for r in env.records if r.server_id == hid), 30.0)
        expected = (env.hosts[0].power(1.0) * 10.0
                    + env.hosts[0].power(2 / 3) * 15.0
                    + env.hosts[0].power(1 / 3) * 5.0
                    + env.hosts[1].power(0.0) * 30.0)
        self.assertAlmostEqual(env.total_energy, expected)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts))

    def test_simultaneous_parent_completions_release_child_once(self):
        env = self.make_env([20.0, 20.0, 1.0], [(0, 2), (1, 2)])
        env.step(0)
        env.step(1)
        self.assertEqual(env.current_time, 20.0)
        self.assertEqual(env.ready_task_ids, [2])
        env.step(0)
        self.assertEqual(env.completed_workflows, 1)
        self.assertEqual(env.current_time, 21.0)

    def test_future_arrivals_and_fewer_arrivals_than_requested(self):
        env = self.make_env([10.0], arrivals=(0.0, 40.0), requested=50)
        _, _, done, _, _ = env.step(0)
        self.assertFalse(done)
        self.assertEqual(env.current_time, 40.0)
        self.assertEqual(env.completed_workflows, 1)
        _, _, done, _, _ = env.step(0)
        self.assertTrue(done)
        self.assertEqual(env.completed_workflows, 2)
        self.assertEqual(env.current_time, 50.0)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records, env.hosts))

    def test_no_arrivals(self):
        env = self.make_env([], arrivals=(), requested=50)
        self.assertTrue(env.done_flag)
        self.assertEqual(env.total_energy, 0.0)

    def test_ss_seed2_all_dependency_edges_and_actual_finish(self):
        scenario = Scenario(task="S", resource="S")
        env = CloudWorkflowEnv_PD3QN_Cost(
            dax_paths=scenario.dax_paths(), random_seed=2,
            deadline_mode="none",
        )
        starts = {}
        for _ in range(10000):
            if env.done_flag:
                break
            gid = env._cur_task_gid
            base = env.wf_task_base[env.task_meta[gid][0]]
            for parent in env.task_parents[gid]:
                self.assertEqual(env.task_state[base + parent], "Finished")
                self.assertLessEqual(env.task_end_time[base + parent], env.current_time)
            before = len(env.records)
            action = int(np.flatnonzero(env.get_action_mask() > 0.5)[0])
            env.step(action)
            starts[gid] = next(r.start_time for r in env.records[before:] if r.vm_pc > 0)
        self.assertTrue(env.done_flag)
        self.assertEqual(len(starts), len(env.task_state))
        for gid, parents in enumerate(env.task_parents):
            base = env.wf_task_base[env.task_meta[gid][0]]
            for parent in parents:
                self.assertGreaterEqual(starts[gid], env.task_end_time[base + parent])
        self.assertGreater(env.task_end_time[2], starts[2])
        self.assertGreaterEqual(starts[3], env.task_end_time[2])
        self.assertAlmostEqual(env.current_time, max(env.task_end_time))
        self.assertEqual(env.completed_workflows, 50)
        print(f"SS seed=2: parent 2 finish={env.task_end_time[2]:.6f}, "
              f"child 3 start={starts[3]:.6f}, finish={env.task_end_time[3]:.6f}; "
              f"tasks={len(starts)}, episode finish={env.current_time:.6f}")


if __name__ == "__main__":
    unittest.main()
