"""R090 reward/physics parity, exact learner reuse, provenance and real lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.experiment_protocol import protocol_binding
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario, MODEL_NAMES
from common.test_f18 import fixture_config
from common.test_mahdrl_oe_ge_apf_score import observation, parameter_hash
from common.test_mahdrl_as_fast import tiny_agents as base_tiny_agents, validation_row
from common.f11 import PendingDecision
from algorithms.mahdrl_oe.reward import finish_lower_phase
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as base_runtime, agent as base_agent, config as base_config
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_oe_ge_apf_score_fast_r090 import runtime, agent, config, selection
from algorithms.mahdrl_oe_ge_apf_score_fast_r090.scenario import ScoreScenario, parse_scenario
from environments.mahdrl_oe_ge_apf_score_fast_r090 import CloudWorkflowEnv_VMAgents


def tiny_agents(scene):
    return base_tiny_agents(scene, agent)


class R090Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='as_r090_')
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/'scales.json', cls.root/'cache.json'
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute', seed_spec='0,1,600,700', cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=ScoreScenario, **kwargs):
        values = dict(mode='train', episodes=1, legacy_split=True, normalization_config=str(self.scales),
                      cache_file=str(self.cache), output_root=str(self.root/'run'), device='cpu')
        return klass(**dict(values, **kwargs))

    def env(self, module=runtime):
        scene = self.scene(BaseScenario if module is base_runtime else ScoreScenario)
        original = module.build_env_kwargs
        with patch.object(module, 'build_env_kwargs', side_effect=lambda s, seed:
                          dict(original(s, seed), workflows_per_episode=2, max_ready_tasks=100)):
            env = module.build_env(scene, 0)
        with redirect_stdout(io.StringIO()):
            env.reset()
            env.apply_manager_delta(np.zeros(5))
        return env

    def test_fixed_weights_reset_and_mutation_guard(self):
        old, new = self.env(base_runtime), self.env()
        for name in ('manager_alpha_delay', 'alpha_delay_host', 'alpha_delay_vm'):
            self.assertEqual(getattr(old, name), .75)
            self.assertEqual(getattr(new, name), .9)
        new.reset()
        config.check_reward_environment(new)
        new.alpha_delay_vm = .75
        before = repr(new._records)
        with self.assertRaisesRegex(ValueError, 'alpha_delay_vm'):
            new.vm_assign(0)
        self.assertEqual(repr(new._records), before)
        new.alpha_delay_vm = .9
        new.manager_alpha_delay = .75
        with self.assertRaisesRegex(ValueError, 'manager_alpha_delay'):
            new.finish_phase_and_advance()
        args = runtime.build_env_kwargs(self.scene(), 0)
        args['manager_alpha_delay'] = .75
        with self.assertRaisesRegex(ValueError, 'manager_alpha_delay'):
            CloudWorkflowEnv_VMAgents(**args)

    def test_same_actions_keep_observations_masks_times_energy_and_change_three_rewards(self):
        old, new = self.env(base_runtime), self.env()
        # Exercise nonzero clipped child penalties and the unclipped manager penalty.
        for env in (old, new):
            env.task_baseline_finish = [float(value)-1e6 for value in env.task_baseline_finish]
        assignments = phases = changed = 0
        with redirect_stdout(io.StringIO()):
            while not old.done_flag:
                while True:
                    ha, oka = old.get_host_state_for_next_assignment()
                    hb, okb = new.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    for key in ha['obs']:
                        np.testing.assert_array_equal(ha['obs'][key], hb['obs'][key])
                    np.testing.assert_array_equal(ha['mask'], hb['mask'])
                    np.testing.assert_array_equal(ha['performance_times'], hb['performance_times'])
                    ah = int(np.flatnonzero(ha['mask'])[0])
                    old.host_select(ah); new.host_select(ah)
                    va, _ = old.get_vm_state_for_current_task()
                    vb, _ = new.get_vm_state_for_current_task()
                    for key in va['obs']:
                        np.testing.assert_array_equal(va['obs'][key], vb['obs'][key])
                    np.testing.assert_array_equal(va['mask'], vb['mask'])
                    av = int(np.flatnonzero(va['mask'])[0])
                    oa, ob, ia = old.vm_assign(av)
                    na, nb, ib = new.vm_assign(av)
                    self.assertEqual(ia, ib)
                    energy, delay = ia['r_energy_total'], ia['r_delay']
                    np.testing.assert_allclose([oa, ob], [.25*energy+.75*delay]*2, rtol=0, atol=1e-10)
                    np.testing.assert_allclose([na, nb], [(1-.9)*energy+.9*delay]*2, rtol=0, atol=1e-10)
                    changed += int(oa != na)
                    self.assertEqual(old.apf_decisions[-1], new.apf_decisions[-1])
                    self.assertEqual(old.assignment_ledger.segments, new.assignment_ledger.segments)
                    assignments += 1
                ra, ia = old.finish_phase_and_advance()
                rb, ib = new.finish_phase_and_advance()
                self.assertEqual(ia['phase_energy_cost'], ib['phase_energy_cost'])
                self.assertEqual(ia['phase_delay_cost'], ib['phase_delay_cost'])
                self.assertAlmostEqual(ra, -(.75*ia['phase_delay_cost']+.25*ia['phase_energy_cost']), places=9)
                self.assertAlmostEqual(rb, -(.9*ib['phase_delay_cost']+(1-.9)*ib['phase_energy_cost']), places=9)
                np.testing.assert_array_equal(old.get_manager_state(), new.get_manager_state())
                phases += 1
                if not old.done_flag:
                    old.apply_manager_delta(np.zeros(5)); new.apply_manager_delta(np.zeros(5))
        self.assertGreater(assignments, 20)
        self.assertGreater(phases, 0)
        self.assertGreater(changed, 0)
        self.assertEqual(old.total_energy, new.total_energy)
        self.assertEqual(old._records, new._records)
        self.assertEqual(repr(old.rng.get_state()), repr(new.rng.get_state()))

    def test_force_assignment_energy_settlement_uses_point_one(self):
        traces = [PendingDecision(lambda *args: None) for _ in range(2)]
        for trace in traces:
            trace.record(np.ones(1), np.ones(1), 0, 0.)
        env = SimpleNamespace(alpha_delay_host=.9, alpha_delay_vm=.9, energy_reward_scale=.001, done_flag=False)
        finish_lower_phase(env, *traces, dict(lower_assignment_energy_unemitted=100.))
        for trace in traces:
            self.assertAlmostEqual(trace.pending['reward'], -.01, places=14)
            self.assertAlmostEqual(trace.recorded_reward, -.01, places=14)

    def compare_learning(self, device):
        for role in ('TA', 'HA', 'VA'):
            options = dict(config.learning_kwargs(role, torch.device(device)), batch_size=2, buffer_size=8)
            clsname = 'TAAgent' if role == 'TA' else 'SetD3QNAgent'
            args = (15, 243) if role == 'TA' else ((23, 10) if role == 'HA' else (30, 7))
            if role == 'TA': options['hidden_dims'] = config.TA_HIDDEN
            runtime.set_seed(43)
            original = getattr(base_agent, clsname)(*args, **options)
            runtime.set_seed(43)
            new = getattr(agent, clsname)(*args, **options)
            self.assertIs(new.__class__.update, original.__class__.update)
            self.assertIs(new.__class__._update, original.__class__._update)
            self.assertEqual(config.learning_kwargs(role, device), base_config.learning_kwargs(role, device))
            for i in range(2):
                if role == 'TA':
                    s, m, nxt, nm = np.arange(15)/15, np.ones(243), np.arange(15)/13, np.ones(243)
                    if i: nm[:] = 0
                else:
                    c, f = args
                    s, m = observation(3, c, f), np.ones(3)
                    nxt, nm = observation(0 if i else 6, c, f), np.ones(0 if i else 6)
                for learner in (original, new): learner.remember(s, m, 0, -.2, nxt, nm, bool(i))
            random.seed(44)
            expected_loss = original.update()
            expected_rng = random.getstate()
            random.seed(44)
            self.assertEqual(expected_loss, new.update())
            self.assertEqual(expected_rng, random.getstate())
            for model in ('online', 'target'):
                for a, b in zip(getattr(original, model).parameters(), getattr(new, model).parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(original._updates, new._updates)

    def test_formal_networks_and_fixed_replay_updates_match_cpu(self):
        self.compare_learning('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_formal_networks_and_fixed_replay_updates_match_cuda(self):
        self.compare_learning('cuda')

    def test_selector_is_v2_rule_and_reward_settings_are_independent(self):
        self.assertIs(selection.selection_key, base_runtime.selection_key)
        self.assertIs(selection.validation_selection, base_runtime.validation_selection)
        self.assertFalse(selection.validation_selection([validation_row(49)])['feasible'])
        self.assertTrue(selection.validation_selection([validation_row(50)])['feasible'])
        settings = config.reward_settings()
        settings['alpha_delay_vm'] = .75
        self.assertEqual(config.reward_settings()['alpha_delay_vm'], .9)
        for key in ('architecture', 'observation_layout', 'apf', 'components', 'pooling'):
            self.assertEqual(config.variant_metadata()[key], base_config.variant_metadata()[key])
        self.assertNotIn('reward_settings', base_config.variant_metadata())

    def test_checkpoint_identity_reward_tampering_roles_groups_and_cross_version_rejection(self):
        scene, old_scene = self.scene(), self.scene(BaseScenario)
        new, old = tiny_agents(scene), base_tiny_agents(old_scene)
        rows = [validation_row(seed=scene.validation_seeds()[0])]
        candidate = selection.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            new_dir, old_dir = root/'new', root/'old'
            new_dir.mkdir(); old_dir.mkdir()
            kwargs = dict(episode=0, candidate=candidate, best=best, rows=rows, best_rows=rows)
            runtime.save_agents(new, new_dir, 'best', **kwargs)
            base_runtime.save_agents(old, old_dir, 'best', **kwargs)
            with patch.object(runtime, 'build_agents', side_effect=tiny_agents):
                loaded = runtime.load_agents(replace(scene, mode='eval', train_resource='S', checkpoint_override=str(new_dir)))
            self.assertEqual(parameter_hash(new), parameter_hash(loaded))
            for role, name in runtime.FILES.items():
                obj = torch.load(new_dir/f'best_{name}.pth', weights_only=True)
                normal = obj['normalization']
                for actual in (normal, normal['algorithm_variant'], normal['implementation'], normal[config.CHECKPOINT_METADATA_KEY]):
                    self.assertEqual(actual['reward_settings'], config.reward_settings())
                    self.assertEqual(actual['reward_protocol'], config.REWARD_PROTOCOL)
                with self.assertRaises(ValueError): new[role].load(str(old_dir/f'best_{name}.pth'))
                with self.assertRaises(ValueError): old[role].load(str(new_dir/f'best_{name}.pth'))
            checkpoint = torch.load(new_dir/'best_host.pth', weights_only=True)
            for section in (None, 'algorithm_variant', 'implementation', config.CHECKPOINT_METADATA_KEY):
                tampered = deepcopy(checkpoint)
                metadata = tampered['normalization'] if section is None else tampered['normalization'][section]
                metadata['reward_settings']['alpha_delay_vm'] = .75
                path = root/'tampered.pth'
                torch.save(tampered, path)
                with self.subTest(section=section), self.assertRaises(ValueError): new['HA'].load(str(path))
            with self.assertRaises(ValueError): new['HA'].load(str(new_dir/'best_vm.pth'))
            with self.assertRaisesRegex(ValueError, 'strict'): new['HA'].load(str(new_dir/'best_host.pth'), strict=False)
            new['TA'].checkpoint_group = 'foreign_group'
            new['TA'].save(str(new_dir/'best_manager.pth'))
            with patch.object(runtime, 'build_agents', side_effect=tiny_agents):
                with self.assertRaisesRegex(ValueError, 'different saved groups'):
                    runtime.load_agents(replace(scene, mode='eval', train_resource='S', checkpoint_override=str(new_dir)))

    def test_real_train_save_load_frozen_eval_selection_and_reused_cache(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            protocol = root/'protocol.json'
            protocol.write_text(json.dumps(dict(version='as_r090_smoke_v1', train=[0, 1], validation=[600],
                test=[700], episodes=2, selection='minimum_validation_total_energy')), encoding='utf-8')
            scales = root/'scales.json'
            cfg = fixture_config(scales)
            cfg['generation'] = dict(**protocol_binding(protocol), calibration_seeds={key: [0] for key in cfg['scenes']})
            scales.write_text(json.dumps(cfg), encoding='utf-8')
            scene = self.scene(episodes=2, legacy_split=False, experiment_protocol=str(protocol),
                               normalization_config=str(scales), output_root=str(root/'run'))
            base_scene = self.scene(BaseScenario, output_root=scene.output_root)
            base_scene.checkpoint_dir().mkdir(parents=True)
            sentinel = base_scene.checkpoint_dir()/'keep.txt'
            sentinel.write_text('original R075 output', encoding='utf-8')
            original_kwargs, original_episode = runtime.build_env_kwargs, runtime.run_episode
            snapshots, seen = [], []
            def small(s, seed):
                return dict(original_kwargs(s, seed), workflows_per_episode=2, max_ready_tasks=100)
            def checked(env, learners, **kwargs):
                config.check_reward_environment(env)
                seen.append(kwargs.get('training', False))
                result = original_episode(env, learners, **kwargs)
                if kwargs.get('training', False):
                    snapshots.append(parameter_hash(learners))
                    for key in ('HA_reward_formula_balance', 'VA_reward_formula_balance', 'lower_energy_balance'):
                        self.assertAlmostEqual(result['reward_audit'][key], 0., places=7)
                return result
            with patch.object(runtime, 'build_agents', side_effect=tiny_agents), \
                 patch.object(runtime, 'build_env_kwargs', side_effect=small), patch.object(runtime, 'run_episode', side_effect=checked):
                with redirect_stdout(io.StringIO()):
                    trained = runtime.train(scene)
                    target = replace(scene, mode='eval', seed_spec='700', train_resource='S', train_deadline='Tight')
                    loaded = runtime.load_agents(target)
                    before = parameter_hash(loaded)
                    updates = {r: a._updates for r, a in loaded.items()}
                    with patch.object(runtime, 'load_agents', return_value=loaded): runtime.evaluate(target)
                    self.assertEqual(parameter_hash(loaded), before)
                    self.assertEqual({r: a._updates for r, a in loaded.items()}, updates)
                    with self.assertRaises(FileExistsError): runtime.train(scene)
                    with self.assertRaises(FileExistsError): runtime.evaluate(target)
            self.assertEqual(seen, [True, False, True, False, False])
            rows = [json.loads(line) for line in (scene.log_dir()/'validation_selection.jsonl').read_text().splitlines()]
            expected = min(rows, key=lambda row: selection.selection_key(row['validation_selection']))['episode']
            self.assertEqual(before, snapshots[expected])
            self.assertEqual(parameter_hash(trained), snapshots[-1])
            self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
            for a in loaded.values(): self.assertEqual(a.loaded_selection['episode'], expected)
            summary = json.loads((scene.log_dir()/'training_summary.json').read_text())
            self.assertEqual(summary['reward_settings'], config.reward_settings())
            self.assertEqual(summary['selected_episode'], expected)
            self.assertEqual(summary['checkpoint_selection']['workflow_success_threshold'], .99)
            measurements = [json.loads(line) for line in (scene.log_dir()/'training_measurements.jsonl').read_text().splitlines()]
            self.assertEqual(len(measurements), 2)
            self.assertTrue(all(v > 0 for row in measurements for v in row['optimizer_steps'].values()))
            self.assertTrue((scene.log_dir()/'stage_timings.jsonl').exists())
            self.assertTrue((target.evaluation_dir()/'evaluation_mahdrl_as_fast_r090.csv').exists())
            self.assertTrue((target.evaluation_dir()/'measurements/episodes/seed700.json').exists())
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'original R075 output')

    def test_dry_runs_and_all_27_target_paths_keep_source_model_and_isolate_outputs(self):
        registry = dict(MODEL_NAMES)
        for mode in ('train', 'eval', 'precompute'):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
                parse_scenario(mode, ['--dry-run'])
            self.assertEqual(stop.exception.code, 0)
            data = json.loads(out.getvalue())
            self.assertEqual(data['algorithm'], config.ALGORITHM)
            self.assertEqual(data['reward_settings'], config.reward_settings())
        for res in 'SML':
            for ddl in ('Tight', 'Medium', 'Loose'):
                source = ScoreScenario(mode='train', resource=res, deadline=ddl)
                targets = [ScoreScenario(mode='eval', task=t, resource=res, deadline=ddl,
                    train_task='S', train_resource=res, train_deadline=ddl) for t in 'SML']
                self.assertEqual({str(t.checkpoint_dir()) for t in targets}, {str(source.checkpoint_dir())})
                self.assertEqual(len({str(t.evaluation_dir()) for t in targets}), 3)
                for t in targets:
                    self.assertLess(len(str(t.evaluation_dir()/'measurements/episodes/seed700.json')), 260)
                self.assertNotEqual(source.checkpoint_dir(), BaseScenario(mode='train', resource=res, deadline=ddl).checkpoint_dir())
        self.assertEqual(registry, MODEL_NAMES)


if __name__ == '__main__':
    unittest.main()
