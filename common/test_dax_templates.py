"""DAX cache names are case-insensitive IDs, resolved only within configured paths."""

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.read_xml_opt_Tsize import data_model_metadata, dax_template_id
from common.scenarios import Scenario
from common.workflow_opt import COMMUNICATION_MODEL
from environments.marl import CloudWorkflowEnv_MARL


class DAXTemplateTests(unittest.TestCase):
    def cache_fixture(self, directory, names, arrivals=None):
        path = Path(directory) / 'name_resolution_fixture.json'
        # Isolate template identity from reference makespan/communication semantics.
        path.write_text(json.dumps(dict(
            meta=dict(communication_model=COMMUNICATION_MODEL, **data_model_metadata()),
            data=[dict(seed=2, arrival_times=arrivals or [float(i) for i in range(len(names))],
                       wf_dax_names=names, wf_makespans=[100.0] * len(names))],
        )), encoding='utf-8')
        return path

    def test_portable_case_insensitive_identifier(self):
        for name in ('Ligo_50.xml', 'LIGO_50.XML', r'X:\dax\Ligo_50.xml', '/dax/LIGO_50.xml'):
            self.assertEqual(dax_template_id(name), 'ligo_50.xml')
        self.assertNotEqual(dax_template_id('LIGO_30.xml'), dax_template_id('LIGO_50.xml'))

    def test_m_task_all_resources_reach_fifth_and_all_fifty_workflows(self):
        for resource in 'SML':
            with self.subTest(resource=resource), tempfile.TemporaryDirectory() as directory:
                scenario = Scenario(task='M', resource=resource)
                historical = json.loads(scenario.legacy_cache_path.read_text(encoding='utf-8'))
                record = next(r for r in historical['data'] if r['seed'] == 2)
                names = record['wf_dax_names']
                self.assertEqual(dax_template_id(names[4]), 'ligo_50.xml')
                cache = self.cache_fixture(directory, names, record['arrival_times'])
                env = CloudWorkflowEnv_MARL(
                    dax_paths=scenario.dax_paths(), random_seed=2, workflows_per_episode=50,
                    deadline_cache_path=str(cache), print_deadlines=False, dump_deadlines=False,
                    num_hosts=scenario.num_hosts, vms_per_host=scenario.vms_per_host,
                )
                env.reset()
                env.current_time = record['arrival_times'][4]
                env._add_workflow_if_arrived()
                self.assertEqual(len(env.workflows), 5)
                self.assertEqual(env.wf_dax_name[4], 'LIGO_50.xml')
                self.assertEqual(len(env.workflows[4].tasks), 50)
                env.current_time = record['arrival_times'][-1]
                env._add_workflow_if_arrived()
                self.assertEqual(len(env.workflows), 50)
                self.assertEqual([dax_template_id(env.wf_dax_name[i]) for i in range(50)],
                                 [dax_template_id(name) for name in names])

    def test_existing_unconfigured_file_does_not_bypass_template_map(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(Scenario().dax_paths()[2])
            unconfigured = Path(directory) / 'unconfigured.xml'
            unconfigured.write_bytes(source.read_bytes())
            cache = self.cache_fixture(directory, [str(unconfigured)])
            with self.assertRaisesRegex(FileNotFoundError, 'configured dax_paths'):
                CloudWorkflowEnv_MARL(dax_paths=[str(source)], random_seed=2, workflows_per_episode=1,
                                      deadline_cache_path=str(cache), dump_deadlines=False, print_deadlines=False)

    def test_ambiguous_paths_rejected_but_weighted_repeats_preserved(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous DAX template ID'):
            CloudWorkflowEnv_MARL(dax_paths=['a/LIGO_50.xml', 'b/Ligo_50.xml'],
                                  deadline_mode='poisson_heftlike', dump_deadlines=False)
        paths = Scenario(task='M').dax_paths()
        env = CloudWorkflowEnv_MARL(dax_paths=paths, deadline_mode='poisson_heftlike', dump_deadlines=False)
        self.assertEqual(env.dax_paths, paths)
        self.assertEqual(len(env.dax_map), 10)

    def test_other_cache_readers_share_case_rule_and_reject_real_mismatch(self):
        for name, cls_name in (('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
                               ('base_fcfs', 'CloudWorkflowEnv_VMAgents'),
                               ('pd3qn', 'CloudWorkflowEnv_PD3QN_Cost'), ('irws', 'IRWSEnvRouteA')):
            with self.subTest(environment=name):
                cls = getattr(importlib.import_module('environments.' + name), cls_name)
                env = object.__new__(cls)
                env._deadline_cache = {2: dict(wf_makespans=[42.0], wf_dax_names=['Ligo_50.xml'])}
                env.deadline_cache_strict = True
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(env._cache_lookup_makespan(2, 0, 'LIGO_50.xml'), 42.0)
                self.assertEqual(output.getvalue(), '')
                with self.assertRaisesRegex(ValueError, 'DAX template mismatch'):
                    env._cache_lookup_makespan(2, 0, 'LIGO_30.xml')

    def test_all_nine_historical_cache_template_sequences_match_generator(self):
        compared = 0
        for task in 'SML':
            for resource in 'SML':
                scenario = Scenario(task=task, resource=resource)
                obj = json.loads(scenario.legacy_cache_path.read_text(encoding='utf-8'))
                paths = scenario.dax_paths()
                for record in obj['data']:
                    names = record['wf_dax_names']
                    indices = np.random.RandomState(record['seed']).randint(len(paths), size=len(names))
                    expected = [dax_template_id(paths[i]) for i in indices]
                    self.assertEqual([dax_template_id(name) for name in names], expected,
                                     f'task={task} resource={resource} seed={record["seed"]}')
                    compared += len(names)
        print(f'9 historical caches: {compared} workflow template IDs match the current generator')


if __name__ == '__main__':
    unittest.main()
