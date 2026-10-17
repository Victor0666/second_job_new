"""IRWS uses workflow-local dependencies and communication keys."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from common.read_xml_opt_Tsize import Task
from common.scenarios import Scenario
from environments.irws import IRWSEnvRouteA


class IRWSNumberingTests(unittest.TestCase):
    def make_env(self, two_parents=False):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from([(0, 2), (1, 2)] if two_parents else [(0, 1)])
        tasks = [Task(task_id=i, workload_mi=1.0,
                      parents=list(graph.predecessors(i)), children=list(graph.successors(i)))
                 for i in range(10)]
        child = tasks[2 if two_parents else 1]
        child.workload_mi = 10.0
        child.parent_in_bits = {0: 2e6, 1: 6e6} if two_parents else {0: 8e6}
        child.from_parents_bits = 8e6
        if two_parents:
            tasks[1].workload_mi = 4.0
        self.enterContext(patch("environments.irws.poisson_arrival_times", return_value=[0.0, 0.0]))
        self.enterContext(patch(
            "environments.irws.load_workflow_from_dax",
            side_effect=lambda *args, **kwargs: (graph.copy(), deepcopy(tasks)),
        ))
        env = IRWSEnvRouteA(
            dax_paths=["synthetic.xml"], workflows_per_episode=2,
            deadline_mode="none", max_ready_tasks=20, normalize=False,
            num_hosts=2, vms_per_host=(2, 1), pc_tiers=(1.0,), bw_tiers=(1.0,),
        )
        env.reset()
        return env

    def test_dependencies_stay_local_in_later_workflows(self):
        env = self.make_env()
        self.assertEqual(env.task_parents[11], [0])
        self.assertEqual(env.task_children[10], [1])
        self.assertEqual(env._task_obj_from_gid(11).parent_in_bits, {0: 8e6})

    def test_execution_and_prediction_for_first_and_later_workflows(self):
        for wf_id in (0, 1):
            for vm_id, expected in ((0, 10.0), (1, 18.0), (2, 18.0)):
                with self.subTest(workflow=wf_id, vm=vm_id):
                    env = self.make_env()
                    base = wf_id * 10
                    env.select_task(base)
                    env.step(0)
                    env.current_time = 1.0
                    env._process_completions_at_time(1.0)
                    self.assertEqual(env.task_state[base + 1], "Ready")
                    env.select_task(base + 1)
                    obs = env.get_vm_obs_and_mask()["obs"]
                    vm_feats = obs[7:].reshape(env.num_vms, 6)
                    np.testing.assert_allclose(vm_feats[:, 4], [10.0, 18.0, 18.0])
                    _, _, _, _, info = env.step(vm_id)
                    self.assertEqual(info["dur"], expected)
                    self.assertEqual(info["start"], 1.0)
                    self.assertEqual(info["dur"], vm_feats[vm_id, 4])

    def test_later_workflow_join_uses_only_its_own_parents(self):
        env = self.make_env(two_parents=True)
        env.select_task(10)
        env.step(0)
        env.select_task(11)
        env.step(2)
        env.current_time = 1.0
        env._process_completions_at_time(1.0)
        self.assertEqual(env.task_state[12], "unReady")
        env.current_time = 4.0
        env._process_completions_at_time(4.0)
        self.assertEqual(env.task_state[12], "Ready")
        self.assertEqual(env.ready_task_ids.count(12), 1)
        self.assertEqual(env.task_state[2], "unReady")
        env.select_task(12)
        vm_feats = env.get_vm_obs_and_mask()["obs"][7:].reshape(env.num_vms, 6)
        np.testing.assert_allclose(vm_feats[:, 4], [16.0, 18.0, 12.0])
        _, _, _, _, info = env.step(2)
        self.assertEqual(info["dur"], 12.0)
        self.assertEqual(info["start"], 4.0)

    def test_real_ss_episode_prediction_matches_execution_and_dependencies(self):
        scenario = Scenario()
        env = IRWSEnvRouteA(
            dax_paths=scenario.dax_paths(), random_seed=2,
            deadline_mode="none", normalize=False,
        )
        env.reset()
        for _ in range(10000):
            if env.done_flag:
                break
            gid = env.get_ready_task_candidates()["gids"][0]
            meta = env.task_meta[gid]
            base = gid - meta.local_id
            task = env._task_obj_from_gid(gid)
            self.assertEqual(env.task_parents[gid], task.parents)
            for p in task.parents:
                self.assertEqual(env.task_state[base + p], "Finished")
                self.assertLessEqual(env.task_end_time[base + p], env.current_time + 1e-12)
            env.select_task(gid)
            vm_data = env.get_vm_obs_and_mask()
            vm_id = int(np.flatnonzero(vm_data["mask"] > 0.5)[0])
            predicted = vm_data["obs"][7:].reshape(env.num_vms, 6)[vm_id, 4]
            _, _, _, _, info = env.step(vm_id)
            np.testing.assert_allclose(info["dur"], predicted, rtol=1e-6)
        self.assertTrue(env.done_flag)
        self.assertTrue(all(state == "Finished" for state in env.task_state))
        self.assertEqual(env.completed_workflows, 50)


if __name__ == "__main__":
    unittest.main()
