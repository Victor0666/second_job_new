"""Independent Q4.2 reward identities with unchanged learners and idle physics."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.marl import q_agents as base_agents
from algorithms.marl import identity as base_identity
from algorithms.marl.scenario import MARLScenario as BaseScenario
from common.f18_state_normalization import bind_normalization
from common.experiment_measurements import EvaluationRecorder
from common.test_f18 import fixture_config
from common.test_marl_v4 import queue_fixture
from common.smoke_marl_q42_weights import run_smoke


def modules(code):
    prefix = 'algorithms.marl_q_v4_2_' + code
    return tuple(importlib.import_module(prefix + '.' + part)
                 for part in ('identity', 'scenario', 'q_agents', 'runtime', 'environment', 'selection'))


class MARLRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_qconfig_updates_actual_action_trace_and_selection_preserved(self):
        root = Path(__file__).resolve().parents[1]
        original = ast.parse((root/'algorithms/marl/q_agents.py').read_text(encoding='utf-8'))
        original_classes = {node.name: node for node in original.body if isinstance(node, ast.ClassDef)}
        for code, alpha in (('010', .1), ('090', .9)):
            identity, scenario, agents, runtime, environment, selection = modules(code)
            self.assertEqual(asdict(agents.QConfig()), asdict(base_agents.QConfig()))
            self.assertEqual(identity.DELAY_WEIGHT, alpha)
            self.assertEqual(selection.selection_metadata()['workflow_success_threshold'], .92)
            self.assertEqual(identity.SCHEDULING_POLICY, base_identity.SCHEDULING_POLICY)
            self.assertEqual(identity.OBSERVATION_LAYOUT, base_identity.OBSERVATION_LAYOUT)
            current = ast.parse((root/'algorithms'/identity.MODEL_VERSION/'q_agents.py').read_text(encoding='utf-8'))
            classes = {node.name: node for node in current.body if isinstance(node, ast.ClassDef)}
            for name in ('QConfig', 'QDecisionTrace', 'Decision'):
                self.assertEqual(ast.dump(classes[name]), ast.dump(original_classes[name]))
            old = {n.name: n for n in original_classes['QAgent'].body if isinstance(n, ast.FunctionDef)}
            for node in classes['QAgent'].body:
                if isinstance(node, ast.FunctionDef) and node.name not in ('save', 'load'):
                    self.assertEqual(ast.dump(node), ast.dump(old[node.name]))
            self.assertEqual((root/'algorithms'/identity.MODEL_VERSION/'selection.py').read_text(encoding='utf-8'),
                             (root/'algorithms/marl/selection.py').read_text(encoding='utf-8'))
        self.assertEqual(base_identity.MODEL_VERSION, 'marl_q_v4_2')

    def fixture(self, environment=None, **kwargs):
        if environment is None:
            return queue_fixture(**kwargs)
        with patch('common.test_marl_v4.CloudWorkflowEnv_MARL', environment):
            return queue_fixture(**kwargs)

    def test_fixed_actions_keep_physics_observations_masks_and_reward_scopes(self):
        environments = [self.fixture()]
        environments += [self.fixture(modules(code)[4].CloudWorkflowEnv_MARL) for code in ('010', '090')]
        for env in environments:
            env.task_baseline_finish = [-1e4] * len(env.task_baseline_finish)
        changed = 0
        while not environments[0].done_flag:
            gobs, gmask = environments[0].get_global_state()
            host = int(np.flatnonzero(gmask)[0])
            lobs, lmask, _ = environments[0].get_local_state(environments[0].host_ids[host])
            vm = int(np.flatnonzero(lmask)[0])
            infos, rewards = [], []
            for env in environments:
                obs, mask = env.get_global_state()
                local, local_mask, mapping = env.get_local_state(env.host_ids[host])
                np.testing.assert_array_equal(obs, gobs)
                np.testing.assert_array_equal(mask, gmask)
                np.testing.assert_array_equal(local, lobs)
                np.testing.assert_array_equal(local_mask, lmask)
                self.assertTrue(env._vm_idle_mask()[mapping[vm]])
                _, global_reward, local_reward, _, info = env.step(host, vm)
                alpha = env.alpha_delay
                self.assertAlmostEqual(global_reward, (1-alpha)*info['r_energy_total']-alpha*info['delay_norm'])
                self.assertAlmostEqual(local_reward, (1-alpha)*info['r_energy_host']-alpha*info['delay_norm'])
                infos.append(info); rewards.append((global_reward, local_reward))
                self.assertNotIn('Queued', env.task_state)
            self.assertEqual(infos[0], infos[1]); self.assertEqual(infos[0], infos[2])
            changed += int(rewards[0] != rewards[1] and rewards[0] != rewards[2])
        self.assertGreater(changed, 0)
        for env in environments[1:]:
            self.assertEqual(env.records, environments[0].records)
            self.assertEqual(env.total_energy, environments[0].total_energy)
            self.assertEqual(env.task_end_time, environments[0].task_end_time)

    def test_reward_guards_and_busy_actions_fail_before_physical_mutation(self):
        for code in ('010', '090'):
            identity, _, _, _, environment, _ = modules(code)
            with self.assertRaisesRegex(ValueError, 'alpha_delay'):
                environment.CloudWorkflowEnv_MARL(alpha_delay=.75)
            env = self.fixture(environment.CloudWorkflowEnv_MARL)
            before = (env.task_state.copy(), env.ready_task_ids.copy(), env.records.copy(), env.total_energy)
            env.alpha_delay = .75
            with self.assertRaisesRegex(ValueError, 'alpha_delay'):
                env.step(0, 0)
            self.assertEqual(before, (env.task_state, env.ready_task_ids, env.records, env.total_energy))
            env.alpha_delay = identity.DELAY_WEIGHT
            env.step(0, 0)
            before = (env.task_state.copy(), env.ready_task_ids.copy(), env.records.copy(), env.total_energy)
            with self.assertRaises(ValueError):
                env.step(0, 0)
            self.assertEqual(before, (env.task_state, env.ready_task_ids, env.records, env.total_energy))

    def test_strict_reward_metadata_cross_version_and_paired_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales = root/'scales.json'; fixture_config(scales)
            for code in ('010', '090'):
                identity, scenario, agents, runtime, _, selection = modules(code)
                scene = scenario.MARLScenario(normalization_config=str(scales))
                cfg = agents.QConfig(hidden_dims=(8, 4))
                global_agent = bind_normalization(agents.QAgent(2, 2, cfg, peer_act_dim=2), scene)
                locals = bind_normalization(agents.IndependentLocalAgents(2, 2, 2, cfg), scene)
                rows = [dict(seed=600, wf_count=50, wf_success_count=46, wf_success_rate=.92, total_energy=1.)]
                candidate = selection.validation_selection(rows)
                self.assertTrue(candidate['feasible'])
                record = runtime.selection_record(scene, 0, candidate, candidate, rows, 'best')
                global_agent.training_selection = locals.training_selection = record
                gp, lp = root/(code+'_global.pth'), root/(code+'_local.pth')
                global_agent.save(gp); locals.save(lp)
                global_agent.load(gp); locals.load(lp)
                runtime.validate_pair(global_agent, locals, scene)
                old_weights = deepcopy(global_agent.actor.state_dict())
                for section in (None, 'normalization'):
                    wrong = torch.load(gp, weights_only=True)
                    metadata = wrong if section is None else wrong[section]
                    metadata['reward_settings']['alpha_delay'] = .75
                    path = root/'wrong.pth'; torch.save(wrong, path)
                    with self.assertRaisesRegex(ValueError, 'reward'):
                        global_agent.load(path)
                for key, value in old_weights.items():
                    torch.testing.assert_close(value, global_agent.actor.state_dict()[key], rtol=0, atol=0)
                other = modules('090' if code == '010' else '010')
                for other_agents, other_scene in (
                    (base_agents, BaseScenario(normalization_config=str(scales))),
                    (other[2], other[1].MARLScenario(normalization_config=str(scales))),
                ):
                    foreign = bind_normalization(other_agents.QAgent(2, 2, other_agents.QConfig(hidden_dims=(8, 4)), peer_act_dim=2), other_scene)
                    foreign_locals = bind_normalization(other_agents.IndependentLocalAgents(2, 2, 2, other_agents.QConfig(hidden_dims=(8, 4))), other_scene)
                    with self.assertRaises(ValueError): foreign.load(gp)
                    with self.assertRaises(ValueError): foreign_locals.load(lp)
                locals.training_selection = dict(record, pair_id='another_pair')
                with self.assertRaisesRegex(ValueError, 'selected pair'):
                    runtime.validate_pair(global_agent, locals, scene)
                wrong = dict(record, reward_settings=dict(identity.reward_settings(), alpha_delay=.75))
                global_agent.training_selection = locals.training_selection = wrong
                with self.assertRaisesRegex(ValueError, 'reward_settings'):
                    runtime.validate_pair(global_agent, locals, scene)

    def test_nine_short_sources_and_27_flat_csvs_cli_paths_are_variant_local(self):
        for code in ('010', '090'):
            identity, scenario, _, _, _, _ = modules(code)
            files, sources = [], set()
            for resource in 'SML':
                for deadline in ('Tight', 'Medium', 'Loose'):
                    source = scenario.MARLScenario(resource=resource, deadline=deadline)
                    checkpoint = source.checkpoint_dir()
                    sources.add(checkpoint)
                    self.assertLessEqual(len(checkpoint.name), 30)
                    for task in 'SML':
                        target = replace(source, mode='eval', task=task, train_task='S', train_resource=resource, train_deadline=deadline)
                        self.assertEqual(checkpoint, target.checkpoint_dir())
                        csv = identity.evaluation_csv(target, checkpoint.parents[3])
                        self.assertIn(f'_{task}{resource}_{deadline}_fromS{resource}_{deadline}_', csv.name)
                        self.assertIn(identity.MODEL_VERSION, csv.name)
                        files.append(csv)
                        self.assertLess(len(str(identity.evaluation_directory(target, checkpoint.parents[3])/'measurements/episodes/seed700.json')), 260)
                        args = ['evaluate.py', '--task', task, '--resource', resource, '--deadline', deadline,
                                '--train-resource', resource, '--train-deadline', deadline, '--dry-run']
                        output = io.StringIO()
                        with patch.object(sys, 'argv', args), redirect_stdout(output), self.assertRaises(SystemExit):
                            scenario.parse_scenario('eval')
                        info = json.loads(output.getvalue())
                        self.assertEqual(info['algorithm'], identity.MODEL_VERSION)
                        self.assertEqual(info['reward_settings'], identity.reward_settings())
                        self.assertEqual(info['checkpoint_selection']['workflow_success_threshold'], .92)
                        self.assertEqual(Path(info['evaluation_csv']), csv)
            self.assertEqual(len(sources), 9)
            self.assertEqual(len({p.name for p in files}), 27)
            self.assertEqual(len({p.parent for p in files}), 1)

    def test_real_cpu_training_validation_save_load_and_frozen_csv_evaluation(self):
        for code in ('010', '090'):
            with self.subTest(variant=code):
                report = run_smoke(variant=code)
                self.assertEqual(report['reward_settings'], modules(code)[0].reward_settings())
                self.assertTrue(report['checkpoint_reload'])
                self.assertTrue(report['evaluation_no_update'])
                self.assertTrue(report['evaluation_csv_flat'])

    def test_measurements_track_explicit_checkpoint_paths_and_refuse_repeated_seed(self):
        for code in ('010', '090'):
            identity, scenario, _, _, environment, _ = modules(code)
            with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'EXPERIMENT_METRICS': '1'}):
                root = Path(directory)
                paths = [root/'custom_global.pth', root/'elsewhere'/'custom_local.pth']
                for index, path in enumerate(paths):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(bytes([index]))
                env = self.fixture(environment.CloudWorkflowEnv_MARL)
                scene = scenario.MARLScenario(mode='eval')
                observer = EvaluationRecorder(scene, identity.MODEL_VERSION, env, {},
                    directory=root/'measurements', checkpoint_paths=paths)
                while not env.done_flag:
                    host = int(np.flatnonzero(env.get_global_state()[1])[0])
                    vm = int(np.flatnonzero(env.get_local_state(env.host_ids[host])[1])[0])
                    env.step(host, vm)
                record = observer.finish()
                self.assertEqual(record['checkpoints'], [
                    dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])
                saved = {path: path.read_bytes() for path in (root/'measurements').rglob('*') if path.is_file()}
                with self.assertRaises(FileExistsError):
                    EvaluationRecorder(scene, identity.MODEL_VERSION, env, {},
                        directory=root/'measurements', checkpoint_paths=paths)
                self.assertTrue(all(path.read_bytes() == data for path, data in saved.items()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_real_cuda_training_and_frozen_evaluation(self):
        for code in ('010', '090'):
            with self.subTest(variant=code):
                report = run_smoke(variant=code, device='cuda')
                self.assertEqual(report['device'], 'cuda')
                self.assertTrue(report['checkpoint_reload'])
                self.assertTrue(report['evaluation_no_update'])


if __name__ == '__main__':
    unittest.main()
