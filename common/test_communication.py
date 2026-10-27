"""Only same-VM parent data transfers are waived, across all environments."""

from contextlib import contextmanager, ExitStack, redirect_stdout
import io
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from common.read_xml_opt_Tsize import data_model_metadata, Task, DAX_TEMPLATE_ID_SCHEME
from common.workflow_opt import exec_time_components


class CommunicationTests(unittest.TestCase):
    @contextmanager
    def environment(self, name, single_vm=False):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(10))
        graph.add_edge(0, 1)
        tasks = [Task(task_id=i, workload_mi=1.0, parents=list(graph.predecessors(i)),
                      children=list(graph.successors(i))) for i in range(10)]
        tasks[1].workload_mi = 10.0
        tasks[1].parent_in_bits = {0: 8e6}
        tasks[1].from_parents_bits = 8e6
        with ExitStack() as stack:
            for mod in ('base_heft', 'base_fcfs', 'mahdrl_heft', 'mahdrl_fcfs', 'marl', 'irws', 'pd3qn'):
                module = importlib.import_module('environments.' + mod)
                if hasattr(module, 'poisson_arrival_times'):
                    stack.enter_context(patch.object(module, 'poisson_arrival_times', return_value=[5.0, 5.0]))
                stack.enter_context(patch.object(module, 'load_workflow_from_dax',
                                                side_effect=lambda *a, **k: (graph.copy(), deepcopy(tasks))))
            module = importlib.import_module('environments.' + name)
            cls = {'marl': 'CloudWorkflowEnv_MARL', 'irws': 'IRWSEnvRouteA',
                   'pd3qn': 'CloudWorkflowEnv_PD3QN_Cost'}.get(name, 'CloudWorkflowEnv_VMAgents')
            kwargs = dict(dax_paths=['synthetic_10.xml'], workflows_per_episode=2,
                          deadline_mode='none', max_ready_tasks=20, normalize=False,
                          num_hosts=2, vms_per_host=(2, 1), pc_tiers=(1.0,), bw_tiers=(1.0,))
            if single_vm:
                kwargs.update(num_hosts=1, vms_per_host=(1,))
            if name in ('base_heft', 'mahdrl_heft'):
                kwargs.pop('deadline_mode')
            if name == 'marl':
                kwargs.update(deadline_mode='poisson_heftlike', dump_deadlines=False, print_deadlines=False)
            env = getattr(module, cls)(**kwargs)
            env.reset()
            yield env

    def assign(self, env, name, gid, vm):
        if name == 'pd3qn':
            env._cur_task_gid = gid
            return env.step(vm)[-1]['task_end']
        if name == 'irws':
            env.select_task(gid)
            return env.step(vm)[-1]['end']
        if name == 'marl':
            return env._assign_task_to_vm(gid, vm)
        env._cur_tid = gid
        env._cur_host_id = env.vms[vm].host_id
        if name == 'ablation_nohost':
            return env.vm_assign_global(vm)[-1]['finish_time']
        slot = env.host_to_vm_indices[env._cur_host_id].index(vm)
        return env.vm_assign(slot)[-1]['finish_time']

    def prediction(self, env, name, gid, vm):
        if name == 'pd3qn':
            return None
        if name == 'marl':
            host = env.vms[env.vm_ids[vm]].host_id
            mapping = list(env.host_vm_indices[host])
            slot = mapping.index(vm)
            mapping += [-1] * (env.max_vms_per_host - len(mapping))
            features = env._build_local_obs(gid, host, mapping)[4:].reshape(-1, 5)
            return features[slot, 4] * env._obs_ref('T_ref', max(env.horizon, 1.))
        if name == 'irws':
            return env._build_vm_obs_for_task(gid)[7:].reshape(env.num_vms, 6)[vm, 4]
        host = env.vms[vm].host_id
        if name == 'ablation_nohost':
            obs, _ = env._build_global_vm_obs_for_task(gid)
            offset = env.block1_dim + env.block2_dim
            finish = obs[offset:].reshape(env.num_vms, 8)[vm, 4] * env.horizon
        else:
            obs, _ = env._build_vm_obs_for_task_host(gid, host)
            offset = env.block1_dim + env.block2_dim + env.host_feat_dim
            slot = env.host_to_vm_indices[host].index(vm)
            finish = obs[offset:].reshape(-1, 5)[slot, 4] * env.horizon
        return finish - env.current_time

    def test_same_vm_same_host_and_cross_host_in_all_environments(self):
        for name in ('mahdrl_heft', 'mahdrl_fcfs', 'base_heft', 'base_fcfs',
                     'ablation_nohost', 'marl', 'irws', 'pd3qn'):
            for base in (0, 10):
                for vm, duration in ((0, 10.0), (1, 18.0), (2, 18.0)):
                    with self.subTest(environment=name, base=base, vm=vm), self.environment(name) as env:
                        self.assertEqual(self.assign(env, name, base, 0), 6.0)
                        env.current_time = 6.0
                        if name == 'pd3qn':
                            env._process_completions()
                        elif name == 'irws':
                            env._process_completions_at_time(6.0)
                        elif name == 'marl':
                            env._process_finish_events()
                        else:
                            env._process_finish_events_at_current_time()
                        gid = base + 1
                        self.assertEqual(env.task_state[gid], 'Ready')
                        predicted = self.prediction(env, name, gid, vm)
                        if predicted is not None:
                            self.assertAlmostEqual(predicted, duration, delta=1e-3)
                        self.assertAlmostEqual(self.assign(env, name, gid, vm), 6.0 + duration)
                        if hasattr(env, 'parents_loc_by_wf'):
                            self.assertEqual(env.parents_loc_by_wf[base // 10][1], (vm, env.vms[vm].host_id))

    def test_all_cache_readers_reject_old_communication_model(self):
        from common.workflow_opt import COMMUNICATION_MODEL
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ddl_fcfs_taskS_resS_seed0-1000.json"
            record = dict(seed=1, wf_makespans=[11], wf_dax_names=["synthetic.xml"], arrival_times=[0])
            for model in (None, COMMUNICATION_MODEL):
                obj = dict(meta=dict(communication_model=model, **data_model_metadata()), data=[record])
                path.write_text(json.dumps(obj), encoding="utf-8")
                for name in ("base_fcfs", "mahdrl_fcfs", "pd3qn", "irws", "marl"):
                    with self.subTest(reader=name, model=model):
                        module = importlib.import_module("environments." + name)
                        if name == "marl":
                            env = module.CloudWorkflowEnv_MARL(
                                dax_paths=["synthetic.xml"], deadline_mode="poisson_heftlike",
                                dump_deadlines=False, print_deadlines=False,
                            )
                            load = env._load_deadline_cache_once
                        else:
                            cls_name = {"irws": "IRWSEnvRouteA", "pd3qn": "CloudWorkflowEnv_PD3QN_Cost"}.get(
                                name, "CloudWorkflowEnv_VMAgents")
                            load = getattr(module, cls_name)._load_deadline_cache
                        if model is None:
                            with self.assertRaisesRegex(ValueError, "precompute_deadlines"):
                                load(str(path))
                        else:
                            load(str(path))

    def test_fcfs_precompute_inherits_same_vm_rule(self):
        from common.precompute_deadlines import run_baseline_episode_fcfs_vmfirst
        with self.environment("base_fcfs", single_vm=True) as env:
            makespans, _, _ = run_baseline_episode_fcfs_vmfirst(env)
            self.assertEqual(len(makespans), 2)
            for gid, parents in enumerate(env.task_parents):
                if parents:
                    record = next(r for r in env._records if r.end_time == env.task_end_time[gid])
                    self.assertEqual(record.end_time - record.start_time, 10.0)

    def test_external_input_output_and_multi_parent_data_are_preserved(self):
        task = Task(task_id=2, workload_mi=10, ext_in_bits=2e6,
                    out_file_size_sum_bits=3e6, from_parents_bits=8e6,
                    parent_in_bits={0: 2e6, 1: 6e6})
        locations = {0: (0, 0), 1: (2, 1)}
        for vm, host, expected in ((0, 0, 21.0), (1, 0, 23.0), (2, 1, 17.0)):
            with self.subTest(vm=vm):
                parts = exec_time_components(task, 1, 1, parents_loc=locations,
                                             child_vm_id=vm, child_host_id=host)
                self.assertEqual(parts[1:3], (10, 3))
                self.assertEqual(parts[-1], expected)
        self.assertEqual(exec_time_components(task, 1, 1)[-1], 23.0)

    def test_precompute_writes_current_model_metadata(self):
        from common.precompute_deadlines import main
        from common.scenarios import Scenario
        from common.workflow_opt import COMMUNICATION_MODEL, validate_deadline_cache
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ddl_fcfs_taskS_resS_seed2.json"
            scenario = Scenario(mode="precompute", seed_spec="2", cache_output=str(output))
            with redirect_stdout(io.StringIO()):
                main(scenario)
            obj = json.loads(output.read_text(encoding="utf-8"))
            validate_deadline_cache(obj, output)
            self.assertEqual(obj["meta"]["communication_model"], COMMUNICATION_MODEL)
            self.assertEqual(obj["meta"]["dax_template_id_scheme"], DAX_TEMPLATE_ID_SCHEME)
            self.assertEqual(len(obj["data"]), 1)
            self.assertEqual(len(obj["data"][0]["wf_makespans"]), 50)

    def test_heft_reference_uses_placed_parent_vm(self):
        tasks = [Task(task_id=0, workload_mi=1, children=[1]),
                 Task(task_id=1, workload_mi=10, parents=[0],
                      parent_in_bits={0: 8e6}, from_parents_bits=8e6)]
        for name in ('base_heft', 'mahdrl_heft'):
            module = importlib.import_module('environments.' + name)
            makespan, finish, _ = module._heft_makespan_and_finish_times(
                tasks, np.array([1.0, 1.0]), np.array([1e6, 1e6]))
            self.assertEqual(makespan, 11.0)
            np.testing.assert_allclose(finish, [1.0, 11.0])


if __name__ == '__main__':
    unittest.main()
