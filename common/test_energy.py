"""All environments integrate all hosts over one episode window."""

from contextlib import ExitStack
from copy import deepcopy
import importlib
import unittest
from unittest.mock import patch

import networkx as nx

from common.read_xml_opt_Tsize import Task
from common.resource_opt import create_cluster
from common.workflow_opt import LoadRecord, energy_from_records


class EnergyWindowTests(unittest.TestCase):
    def setUp(self):
        self.hosts, _ = create_cluster(num_hosts=3, vms_per_host=(1, 1, 1), pc_tiers=(1.0,), bw_tiers=(1.0,))

    def test_unused_hosts_share_common_record_window(self):
        self.assertAlmostEqual(energy_from_records([LoadRecord(5, 15, 0, 1)], self.hosts), 750.0)

    def test_each_host_includes_idle_before_and_after_its_own_tasks(self):
        records = [LoadRecord(5, 10, 0, 1), LoadRecord(10, 15, 1, 1)]
        expected = (self.hosts[0].power(1) * 5 + self.hosts[0].power(0) * 5
                    + self.hosts[1].power(0) * 5 + self.hosts[1].power(1) * 5
                    + self.hosts[2].power(0) * 10)
        self.assertAlmostEqual(energy_from_records(records, self.hosts), expected)

    def test_explicit_window_includes_head_tail_and_empty_hosts(self):
        records = [LoadRecord(10, 20, 0, 1)]
        self.assertAlmostEqual(energy_from_records(records, self.hosts, start_time=5, end_time=25), 1208.0)
        self.assertAlmostEqual(energy_from_records([], self.hosts, start_time=5, end_time=25), 916.0)
        self.assertEqual(energy_from_records([], self.hosts), 0.0)
        self.assertEqual(energy_from_records(records, self.hosts, start_time=5, end_time=5), 0.0)

    def test_clipping_overlaps_and_zero_load_records(self):
        hosts, _ = create_cluster(num_hosts=1, vms_per_host=(2,), pc_tiers=(1.0,), bw_tiers=(1.0,))
        records = [LoadRecord(0, 10, 0, 1), LoadRecord(5, 15, 0, 1), LoadRecord(0, 20, 0, 0)]
        expected = hosts[0].power(0.5) * 10 + hosts[0].power(1) * 5 + hosts[0].power(0) * 5
        self.assertAlmostEqual(energy_from_records(records, hosts), expected)
        self.assertAlmostEqual(energy_from_records(records, hosts, start_time=6, end_time=8), hosts[0].power(1) * 2)

    def run_synthetic(self, module_name, arrivals):
        module = importlib.import_module('environments.' + module_name)
        graph = nx.DiGraph()
        graph.add_node(0)
        task = Task(task_id=0, workload_mi=10.0)
        with ExitStack() as stack:
            # Inherited reset/arrival methods resolve globals in their defining modules.
            for name in ('base_heft', 'base_fcfs', 'mahdrl_heft', 'mahdrl_fcfs', 'marl', 'irws', 'pd3qn'):
                if hasattr(importlib.import_module('environments.' + name), 'poisson_arrival_times'):
                    stack.enter_context(patch('environments.' + name + '.poisson_arrival_times', return_value=list(arrivals)))
                stack.enter_context(patch('environments.' + name + '.load_workflow_from_dax',
                                         side_effect=lambda *a, **k: (graph.copy(), [deepcopy(task)])))
            cls_name = {'pd3qn': 'CloudWorkflowEnv_PD3QN_Cost', 'irws': 'IRWSEnvRouteA',
                        'marl': 'CloudWorkflowEnv_MARL'}.get(module_name, 'CloudWorkflowEnv_VMAgents')
            kwargs = dict(dax_paths=['synthetic.xml'], workflows_per_episode=len(arrivals),
                          deadline_mode='none', max_ready_tasks=20, num_hosts=3,
                          vms_per_host=(1, 1, 1), pc_tiers=(1.0,), bw_tiers=(1.0,))
            if module_name in ('mahdrl_heft', 'base_heft'):
                kwargs.pop('deadline_mode')
            if module_name == 'marl':
                kwargs.update(deadline_mode='poisson_heftlike', dump_deadlines=False, print_deadlines=False)
            env = getattr(module, cls_name)(**kwargs)
            env.reset()
            self.assertEqual(env.total_energy, 0.0)
            reward_energy = 0.0
            for _ in range(10):
                if env.done_flag:
                    break
                if module_name == 'pd3qn':
                    _, reward, _, _, info = env.step(0)
                    self.assertAlmostEqual(reward, (1 - env.alpha_delay_vm) * info['r_energy_total']
                                           + env.alpha_delay_vm * info['r_delay'])
                    reward_energy += info['r_energy_total']
                elif module_name == 'irws':
                    env.select_task(env.get_ready_task_candidates()['gids'][0])
                    _, reward, _, _, info = env.step(0)
                    self.assertAlmostEqual(reward, (1 - env.alpha_delay_vm) * info['r_energy_total']
                                           + env.alpha_delay_vm * info['r_delay'])
                    reward_energy += info['r_energy_total']
                elif module_name == 'marl':
                    gid = env.ready_task_ids[0]
                    env._assign_task_to_vm(gid, 0)
                    r, by_host = env._advance_until_decision()
                    self.assertAlmostEqual(r, sum(by_host.values()))
                    reward_energy += r
                else:
                    env._assign_task_to_specific_vm(env.ready_task_ids[0], 0)
                    reward_energy += env._advance_until_decision_energy_only()
            self.assertTrue(env.done_flag)
            end = arrivals[-1] + 10.0
            expected = 750.0 * len(arrivals) + 45.8 * (end - arrivals[0] - 10 * len(arrivals))
            self.assertAlmostEqual(env.current_time, end)
            self.assertAlmostEqual(env.total_energy, expected)
            records = env.records if hasattr(env, 'records') else env._records
            self.assertAlmostEqual(energy_from_records(records, env.hosts, start_time=arrivals[0], end_time=end), expected)
            self.assertAlmostEqual(reward_energy, -expected * env.energy_reward_scale)
            if hasattr(env, '_energy_cache_by_host'):
                self.assertAlmostEqual(sum(env._energy_cache_by_host.values()), expected)
                self.assertAlmostEqual(env._energy_cache_by_host[1], 13.1 * (end - arrivals[0]))
                self.assertAlmostEqual(env._energy_cache_by_host[2], 16.8 * (end - arrivals[0]))

    def test_all_environment_families_single_task_and_idle_gap(self):
        for module_name in ('mahdrl_heft', 'mahdrl_fcfs', 'ablation_nohost', 'base_heft', 'base_fcfs', 'marl', 'irws', 'pd3qn'):
            for arrivals in ((5.0,), (5.0, 25.0)):
                with self.subTest(environment=module_name, arrivals=arrivals):
                    self.run_synthetic(module_name, arrivals)


if __name__ == '__main__':
    unittest.main()
