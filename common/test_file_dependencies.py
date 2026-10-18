"""F07 file identities, complete parent outputs, cache separation and DAX invariants."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import networkx as nx
from common.read_xml_opt_Tsize import (
    DATA_MODEL, _randomize_payloads, _read_dax_to_graph,
    load_workflow_from_dax, workflow_instance_hash, validate_cached_instance, data_model_metadata)
from common.scenarios import DAX_GROUPS, ROOT, Scenario
from common.workflow_opt import COMMUNICATION_MODEL, exec_time_components, validate_deadline_cache

BITS_PER_MB = 1024 * 1024 * 8.0


def assert_payload(test, graph, tasks):
    definitions = [f for t in tasks for f in t.out_files] + [f for t in tasks if not t.parents for f in t.in_files]
    test.assertEqual(len(definitions), len({f.name for f in definitions}))
    for t in tasks:
        test.assertEqual(set(t.parents), set(graph.predecessors(t.task_id)))
        test.assertEqual(set(t.children), set(graph.successors(t.task_id)))
        test.assertTrue(60 <= t.workload_mi <= 90)
        test.assertTrue(1 <= len(t.out_files) <= 3)
        test.assertAlmostEqual(t.out_file_size_sum_bits, sum(f.size_bits for f in t.out_files), delta=1e-4)
        test.assertAlmostEqual(t.in_file_size_sum_bits, sum(f.size_bits for f in t.in_files), delta=1e-4)
        test.assertEqual(t.from_parents_bits, sum(t.parent_in_bits.values()))
        if t.parents:
            expected = {f for p in t.parents for f in tasks[p].out_files}
            test.assertEqual(set(t.in_files), expected)
            test.assertEqual(len(t.in_files), len(expected))
            test.assertEqual(t.ext_in_bits, 0)
            test.assertEqual(t.in_file_size_sum_bits, t.from_parents_bits)
            test.assertEqual(set(t.parent_in_bits), set(t.parents))
        else:
            test.assertTrue(1 <= len(t.in_files) <= 3)
            test.assertGreater(t.ext_in_bits, 0)
            test.assertEqual(t.ext_in_bits, t.in_file_size_sum_bits)
            test.assertEqual(t.parent_in_bits, {})
            test.assertEqual(t.from_parents_bits, 0)
            test.assertTrue(all(f.source_task_id is None for f in t.in_files))
        for f in t.out_files:
            test.assertEqual(f.source_task_id, t.task_id)
            test.assertTrue(512 * BITS_PER_MB <= f.size_bits <= 1024 * BITS_PER_MB)
        for p in t.parents:
            test.assertGreater(t.parent_in_bits[p], 0)
            test.assertEqual(t.parent_in_bits[p], tasks[p].out_file_size_sum_bits)
            test.assertEqual(set(tasks[p].out_files) & set(t.in_files), set(tasks[p].out_files))
            for f in tasks[p].out_files:
                test.assertIs(next(item for item in t.in_files if item.name == f.name), f)


class FileDependencyTests(unittest.TestCase):
    def test_01_one_parent_three_children_1400_mb(self):
        class FixedRng:
            def __init__(self, seed):
                self.n = 0
            def randint(self, a, b):
                return 2
            def uniform(self, a, b):
                if (a, b) == (60, 90):
                    return 75
                self.n += 1
                return 600 if self.n % 2 else 800
        graph = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
        with patch('common.read_xml_opt_Tsize.random.Random', FixedRng):
            tasks = _randomize_payloads(graph, seed=2)
        self.assertEqual(tasks[0].out_file_size_sum_bits, 1400 * BITS_PER_MB)
        for child in (1, 2, 3):
            self.assertEqual(tasks[child].parent_in_bits[0], 1400 * BITS_PER_MB)
        assert_payload(self, graph, tasks)

    def test_02_join_receives_all_parent_outputs(self):
        graph = nx.DiGraph([(0, 2), (1, 2)])
        assert_payload(self, graph, _randomize_payloads(graph, seed=2))

    def test_03_thirteen_parents_twenty_six_input_files(self):
        graph = nx.DiGraph((i, 13) for i in range(13))
        tasks = _randomize_payloads(graph, seed=2, io_file_count_range=(2, 2))
        self.assertEqual(len(tasks[13].in_files), 26)
        assert_payload(self, graph, tasks)

    def test_04_external_input_only_at_entries(self):
        graph = nx.DiGraph([(0, 1), (1, 2)])
        assert_payload(self, graph, _randomize_payloads(graph, seed=7))

    def test_05_shared_immutable_file_identity_and_workflow_scope(self):
        graph = nx.DiGraph([(0, 1), (0, 2)])
        first = _randomize_payloads(graph, seed=2)
        second = _randomize_payloads(graph, seed=2)
        assert_payload(self, graph, first)
        self.assertIsNot(first[0].out_files[0], second[0].out_files[0])
        self.assertEqual(first[0].out_files[0], second[0].out_files[0])
        with self.assertRaises(FrozenInstanceError):
            first[0].out_files[0].size_bits = 0

    def test_06_every_edge_file_intersection(self):
        graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)])
        assert_payload(self, graph, _randomize_payloads(graph, seed=31))

    def test_07_reproducible_hash_excludes_scheduling_state(self):
        graph = nx.DiGraph([(0, 1)])
        tasks = _randomize_payloads(graph, seed=2)
        expected = workflow_instance_hash(graph, tasks)
        self.assertEqual(expected, workflow_instance_hash(graph, _randomize_payloads(graph, seed=2)))
        self.assertNotEqual(expected, workflow_instance_hash(graph, _randomize_payloads(graph, seed=3)))
        tasks[0].state, tasks[0].finish_time, tasks[0].assigned_server_id = 'Finished', 120.0, 4
        self.assertEqual(expected, workflow_instance_hash(graph, tasks))
        validate_cached_instance({2: dict(wf_instance_hashes=[expected])}, 2, 0, graph, tasks)
        tasks[0].workload_mi += 1
        with self.assertRaisesRegex(ValueError, 'instance hash mismatch'):
            validate_cached_instance({2: dict(wf_instance_hashes=[expected])}, 2, 0, graph, tasks)

    def test_08_same_vm_waives_network_only(self):
        graph = nx.DiGraph([(0, 1)])
        tasks = _randomize_payloads(graph, seed=2)
        bits = tasks[1].parent_in_bits[0]
        self.assertGreater(bits, 0)
        self.assertEqual(exec_time_components(tasks[1], 1, 1000, parents_loc={0: (5, 2)},
                                             child_vm_id=5, child_host_id=2)[0], 0)
        self.assertGreater(exec_time_components(tasks[1], 1, 1000, parents_loc={0: (5, 2)},
                                               child_vm_id=6, child_host_id=2)[0], 0)
        self.assertEqual(tasks[1].parent_in_bits[0], bits)

    def test_09_cache_rejects_every_mismatched_parameter(self):
        meta = dict(communication_model=COMMUNICATION_MODEL, **data_model_metadata())
        validate_deadline_cache(dict(meta=meta), 'fixture.json')
        validate_deadline_cache(dict(meta=meta), Scenario().cache_path)
        for key in meta:
            for value in (None, 'wrong'):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_deadline_cache(dict(meta={**meta, key: value}), 'fixture.json')
        with self.assertRaisesRegex(ValueError, 'data_model'):
            validate_deadline_cache(dict(meta=dict(communication_model=COMMUNICATION_MODEL)), 'old.json')
        self.assertEqual(Scenario().cache_path.parent.name, DATA_MODEL)
        self.assertNotEqual(Scenario().cache_path, Scenario().legacy_cache_path)

    def test_10_all_fifteen_real_dax_templates_and_topology_only_mode(self):
        for group in DAX_GROUPS:
            for name in group:
                with self.subTest(template=name):
                    path = ROOT / 'data' / 'dax' / name
                    original, _ = _read_dax_to_graph(path)
                    graph, tasks = load_workflow_from_dax(path, seed=2)
                    self.assertEqual(set(original.nodes), set(graph.nodes))
                    self.assertEqual(set(original.edges), set(graph.edges))
                    assert_payload(self, graph, tasks)
                    empty_graph, empty = load_workflow_from_dax(path, seed=2, randomize_payloads=False)
                    self.assertEqual(set(graph.edges), set(empty_graph.edges))
                    for task in empty:
                        self.assertEqual((task.workload_mi, task.ext_in_bits, task.from_parents_bits,
                                          task.in_file_size_sum_bits, task.out_file_size_sum_bits), (0, 0, 0, 0, 0))
                        self.assertEqual((task.in_files, task.out_files, task.parent_in_bits), ([], [], {}))

    def test_11_ignore_xml_payload_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.xml'
            xml = '<adag xmlns="http://pegasus.isi.edu/schema/DAX"><job id="a"><uses file="x" size="{}"/></job><job id="b"/><child ref="b"><parent ref="a"/></child></adag>'
            path.write_text(xml.format('1'), encoding='utf-8')
            graph, tasks = load_workflow_from_dax(path, seed=2)
            path.write_text(xml.format('999999999999999'), encoding='utf-8')
            other_graph, other_tasks = load_workflow_from_dax(path, seed=2)
            self.assertEqual(workflow_instance_hash(graph, tasks), workflow_instance_hash(other_graph, other_tasks))

    def test_12_precompute_never_overwrites_existing_cache(self):
        from common.precompute_deadlines import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'legacy.json'
            output.write_bytes(b'historical bytes')
            with self.assertRaisesRegex(FileExistsError, 'overwrite'):
                main(Scenario(mode='precompute', seed_spec='2', cache_output=str(output)))
            self.assertEqual(output.read_bytes(), b'historical bytes')

    def test_13_all_cache_readers_check_file_model(self):
        import importlib
        meta = dict(communication_model=COMMUNICATION_MODEL, **data_model_metadata())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for key in ('data_model', 'file_size_mb_range', 'payload_generator_version'):
                path.write_text(json.dumps(dict(meta={**meta, key: 'wrong'}, data=[])), encoding='utf-8')
                for name, cls_name in [('base_fcfs', 'CloudWorkflowEnv_VMAgents'), ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
                                      ('pd3qn', 'CloudWorkflowEnv_PD3QN_Cost'), ('irws', 'IRWSEnvRouteA'), ('marl', 'CloudWorkflowEnv_MARL')]:
                    cls = getattr(importlib.import_module('environments.' + name), cls_name)
                    if name == 'marl':
                        env = object.__new__(cls)
                        env._deadline_cache_by_seed = None
                        load = env._load_deadline_cache_once
                    else:
                        load = cls._load_deadline_cache
                    with self.subTest(reader=name, key=key), self.assertRaisesRegex(ValueError, key):
                        load(str(path))


    def test_14_all_nine_real_environment_chains_complete(self):
        import importlib
        from common.test_paired_instances import ALGORITHMS
        reference_hash = None
        for algorithm, (module_name, cls_name) in ALGORITHMS.items():
            with self.subTest(algorithm=algorithm):
                cls = getattr(importlib.import_module('environments.' + module_name), cls_name)
                kwargs = dict(dax_paths=[str(ROOT / 'data' / 'dax' / DAX_GROUPS[0][0])],
                              random_seed=2, workflows_per_episode=1, max_ready_tasks=100,
                              deadline_mode='none', num_hosts=1, vms_per_host=(1,),
                              pc_tiers=(1.0,), bw_tiers=(1000.0,))
                if algorithm == 'marl':
                    kwargs.update(deadline_mode='poisson_heftlike', dump_deadlines=False, print_deadlines=False)
                env = cls(**kwargs)
                env.reset()
                for _ in range(500):
                    if env.done_flag:
                        break
                    if algorithm == 'pd3qn':
                        env.step(0)
                    elif algorithm == 'irws':
                        env.select_task(env.get_ready_task_candidates()['gids'][0])
                        env.step(0)
                    elif algorithm == 'marl':
                        env._assign_task_to_vm(env.ready_task_ids[0], 0)
                        env._advance_until_decision()
                    else:
                        env._assign_task_to_specific_vm(env.ready_task_ids[0], 0)
                        env._advance_until_decision_energy_only()
                self.assertTrue(env.done_flag)
                self.assertTrue(all(state == 'Finished' for state in env.task_state))
                wf = env.workflows[0]
                assert_payload(self, wf.graph, wf.tasks)
                actual = workflow_instance_hash(wf.graph, wf.tasks)
                if reference_hash is None:
                    reference_hash = actual
                self.assertEqual(actual, reference_hash)

    def test_15_all_nine_runtime_chains_reject_corrupted_instance_hash(self):
        from contextlib import redirect_stdout
        import io
        from common.precompute_deadlines import main
        from common.test_paired_instances import ALGORITHMS, PairedInstanceTests
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            cache = Path(directory) / 'reference.json'
            main(Scenario(mode='precompute', seed_spec='2', cache_output=str(cache)))
            obj = json.loads(cache.read_text(encoding='utf-8'))
            obj['data'][0]['wf_instance_hashes'][0] = '0' * 64
            cache.write_text(json.dumps(obj), encoding='utf-8')
            checker = PairedInstanceTests()
            for algorithm in ALGORITHMS:
                with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, 'instance hash mismatch'):
                    checker.snapshot(algorithm, Scenario(), cache)


if __name__ == '__main__':
    unittest.main()
