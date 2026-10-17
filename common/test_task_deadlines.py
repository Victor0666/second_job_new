"""MARL and the other algorithms share identical task sub-deadlines."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from common.read_xml_opt_Tsize import Task
from common.scenarios import Scenario
from common.workflow_opt import COMMUNICATION_MODEL


class TaskDeadlineTests(unittest.TestCase):
    def collect(self, paths, cache, count, resource_kwargs=None, synthetic=None):
        deadlines = {}
        for name, cls_name in (
            ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
            ('base_fcfs', 'CloudWorkflowEnv_VMAgents'),
            ('pd3qn', 'CloudWorkflowEnv_PD3QN_Cost'),
            ('irws', 'IRWSEnvRouteA'),
            ('marl', 'CloudWorkflowEnv_MARL'),
        ):
            with ExitStack() as stack:
                if synthetic is not None:
                    graph, tasks = synthetic
                    for mod in ('mahdrl_heft', 'mahdrl_fcfs', 'base_heft', 'base_fcfs', 'pd3qn', 'irws'):
                        module = importlib.import_module('environments.' + mod)
                        if hasattr(module, 'poisson_arrival_times'):
                            stack.enter_context(patch.object(module, 'poisson_arrival_times', return_value=[5.0, 5.0]))
                        stack.enter_context(patch.object(module, 'load_workflow_from_dax',
                                                        side_effect=lambda *a, **k: (graph.copy(), deepcopy(tasks))))
                    stack.enter_context(patch('environments.marl.load_workflow_from_dax',
                                              side_effect=lambda *a, **k: (graph.copy(), deepcopy(tasks))))
                kwargs = dict(dax_paths=paths, random_seed=2, workflows_per_episode=count,
                              horizon=1e9, arrival_lambda=0.03, max_ready_tasks=100,
                              deadline_mode='cache_fcfs', deadline_cache_path=str(cache))
                kwargs.update(resource_kwargs or {})
                if name == 'mahdrl_fcfs':
                    kwargs.update(deadline_alpha_small=2.0, deadline_alpha_large=2.0)
                elif name == 'marl':
                    kwargs.update(alpha_deadline_choices=(2.0,), alpha_deadline_probs=(1.0,),
                                  dump_deadlines=False, print_deadlines=False)
                else:
                    kwargs['deadline_alpha'] = 2.0
                env = getattr(importlib.import_module('environments.' + name), cls_name)(**kwargs)
                env.reset()
                deadlines[name] = np.array(env.task_baseline_finish)
                self.assertAlmostEqual(env.workflows[0].deadline,
                                       env.workflows[0].arrival_time + 2 * json.loads(cache.read_text())['data'][0]['wf_makespans'][0])
        return deadlines

    def test_heterogeneous_vms_fork_join_and_no_earliest_finish_clamp(self):
        graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        tasks = [Task(task_id=i, workload_mi=work, parents=list(graph.predecessors(i)),
                      children=list(graph.successors(i))) for i, work in enumerate((4, 8, 12, 16))]
        for reference, expected in ((15.0, [17.5, 25.0, 25.0, 35.0]),
                                    (0.5, [-11.5, -4.0, -4.0, 6.0])):
            with self.subTest(reference=reference), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory) / 'ddl_fcfs_taskS_resS_seed2.json'
                cache.write_text(json.dumps(dict(
                    meta=dict(communication_model=COMMUNICATION_MODEL),
                    data=[dict(seed=2, arrival_times=[5.0, 5.0],
                               wf_dax_names=['synthetic.xml'] * 2, wf_makespans=[reference] * 2)],
                )), encoding='utf-8')
                result = self.collect(['synthetic.xml'], cache, 2,
                                      dict(num_hosts=1, vms_per_host=(2,), pc_tiers=(1.0, 4.0), bw_tiers=(1.0, 4.0)),
                                      (graph, tasks))
                for name, actual in result.items():
                    with self.subTest(environment=name):
                        np.testing.assert_allclose(actual, expected * 2, rtol=0, atol=1e-12)

    def test_input_output_and_compute_are_averaged_per_vm(self):
        graph = nx.DiGraph([(0, 1)])
        tasks = [Task(task_id=0, workload_mi=4, children=[1]),
                 Task(task_id=1, workload_mi=8, parents=[0], ext_in_bits=2e6,
                      parent_in_bits={0: 6e6}, from_parents_bits=6e6,
                      out_file_size_sum_bits=2e6)]
        # Child durations are 18 and 4.5 seconds, averaging to 11.25.
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'ddl_fcfs_taskS_resS_seed2.json'
            cache.write_text(json.dumps(dict(
                meta=dict(communication_model=COMMUNICATION_MODEL),
                data=[dict(seed=2, arrival_times=[5.0, 5.0], wf_dax_names=['synthetic.xml'] * 2,
                           wf_makespans=[22.5, 22.5])],
            )), encoding='utf-8')
            result = self.collect(['synthetic.xml'], cache, 2,
                                  dict(num_hosts=1, vms_per_host=(2,), pc_tiers=(1.0, 4.0), bw_tiers=(1.0, 4.0)),
                                  (graph, tasks))
            for name, actual in result.items():
                with self.subTest(environment=name):
                    np.testing.assert_allclose(actual, [38.75, 50.0] * 2, rtol=0, atol=1e-12)

    def test_real_ss_seed2_alpha2_matches_task_by_task(self):
        from common.precompute_deadlines import main
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'ddl_fcfs_taskS_resS_seed2.json'
            scenario = Scenario(mode='precompute', seed_spec='2', cache_output=str(cache))
            with redirect_stdout(io.StringIO()):
                main(scenario)
            result = self.collect(scenario.dax_paths(), cache, 50)
            reference = result['mahdrl_fcfs']
            for name, actual in result.items():
                with self.subTest(environment=name):
                    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
            max_gap = max(float(np.max(np.abs(actual - reference))) for actual in result.values())
            print(f'SS seed=2 alpha=2: {len(reference)} task deadlines, maximum gap={max_gap:.12f}s')


if __name__ == '__main__':
    unittest.main()
