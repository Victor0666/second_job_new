"""VAware mechanism, decision atomicity, physical parity and model lifecycle."""
from contextlib import redirect_stdout
from dataclasses import replace
from copy import deepcopy
import io
import json
import csv
import random
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_vaaware import runtime, agent
from algorithms.mahdrl_as_fast_vaaware.config import Mechanism, MODEL_VERSION
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_as_fast_vaaware.preview import behavior_probabilities, prepare_decision
from algorithms.mahdrl_as_fast_vaaware.updates import UpdateClock
from algorithms.mahdrl_oe_ge_apf_score_fast import runtime as baseline


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=scene.device or "cpu")
    learners = dict(TA=agent.TAAgent(15, 243, **options),
        HA=agent.SetD3QNAgent(23, scene.mechanism.host_features, entity_dims=(8, 8), **options),
        VA=agent.SetD3QNAgent(30, 7, entity_dims=(8, 8), **options))
    return {role: a.bind(scene, role) for role, a in learners.items()}


class VAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="mahdrl_vaaware_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/"scales.json", cls.root/"cache.json"
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode="precompute", seed_spec="0,1,2,600,700", cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, **kwargs):
        values = dict(mode="train", episodes=2, legacy_split=True, device="cpu",
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName))
        return ScoreScenario(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene=None, seed=0):
        scene = scene or self.scene()
        with patch.object(runtime, "build_env_kwargs", self.small_kwargs(runtime.build_env_kwargs)):
            env = runtime.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_probability_mixture_ties_masks_and_deterministic_limit(self):
        np.testing.assert_allclose(behavior_probabilities([10., 5., 100.], [1, 1, 0], [3., 1., 0.], .4), [.7, .3, 0])
        np.testing.assert_allclose(behavior_probabilities([10., 5., 5.], [1, 1, 1], [3., 1., 1.], .6), [.5, .25, .25])
        np.testing.assert_array_equal(behavior_probabilities([10., 10.], [1, 1], [3., 1.], 0), [1, 0])
        for eps in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                behavior_probabilities([1.], [1], [1.], eps)
        with self.assertRaises(ValueError):
            behavior_probabilities([1.], [0], [1.], .5)

    def test_preview_is_pure_matches_local_states_and_expectations(self):
        env, learners = self.env(), tiny_agents(self.scene())
        host, ok = env.get_host_state_for_next_assignment()
        self.assertTrue(ok)
        before = (env.current_time, deepcopy(env._records), deepcopy(env.assignment_ledger.segments),
                  env._cur_tid, env._cur_host_id, repr(env.rng.get_state()))
        python_rng, numpy_rng, torch_rng = random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone()
        counters = (learners['VA']._updates, learners['VA']._eps_steps, learners['VA']._action_calls)
        enhanced, states, extra = prepare_decision(env, learners['VA'], host, training=True)
        self.assertEqual(before, (env.current_time, env._records, env.assignment_ledger.segments,
                         env._cur_tid, env._cur_host_id, repr(env.rng.get_state())))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(repr(np.random.get_state()), numpy_rng)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(counters, (learners['VA']._updates, learners['VA']._eps_steps, learners['VA']._action_calls))
        self.assertEqual(enhanced['obs']['candidates'].shape, (env.num_hosts, 14))
        np.testing.assert_array_equal(enhanced['obs']['candidates'][:, :10], host['obs']['candidates'])
        self.assertFalse(enhanced['obs']['candidates'].flags.writeable)
        for hid, s in states.items():
            self.assertAlmostEqual(s['probabilities'].sum(), 1)
            self.assertTrue((s['probabilities'][s['mask'] == 0] == 0).all())
            i = env.host_ids.index(hid)
            ref = env._obs_ref('T_ref', 300.)
            self.assertAlmostEqual(extra[i, 1], s['probabilities'] @ s['rows'][:, 1]/(100*ref), places=6)
        hid = next(iter(states))
        env.host_select(env.host_ids.index(hid))
        real, ok = env.get_vm_state_for_current_task()
        self.assertTrue(ok)
        for key in real['obs']:
            np.testing.assert_array_equal(real['obs'][key], states[hid]['obs'][key])
        np.testing.assert_array_equal(real['mask'], states[hid]['mask'])

    def test_busy_slow_overdue_candidates_and_policy_version_guard(self):
        env, learners = self.env(), tiny_agents(self.scene())
        busy = env.host_to_vm_indices[env.host_ids[0]]
        env.vm_available_at[busy] = env.current_time+100
        with patch('environments.mahdrl_oe_ge_apf_score_fast.apf_candidates', side_effect=AssertionError('APF called')):
            host, ok = env.get_host_state_for_next_assignment()
            self.assertTrue(ok)
            env.task_baseline_finish[env._cur_tid] = env.current_time-10
            enhanced, states, _ = prepare_decision(env, learners['VA'], host, training=False)
        self.assertNotIn(env.host_ids[0], states)
        self.assertEqual(enhanced['mask'].sum(), 2)
        for s in states.values():
            self.assertTrue(np.all(s['rows'][s['mask'] > .5, 2] > 0))
            self.assertEqual(s['epsilon'], 0)
        sample = next(iter(states.values()))
        learners['VA']._updates += 1
        with self.assertRaisesRegex(RuntimeError, 'VA changed'):
            learners['VA'].select_preview_action(sample, training=False)

    def test_update_period_does_not_slow_HA_epsilon(self):
        results = []
        for period in (1, 4):
            scene = self.scene(ha_update_period=period)
            learners, env = tiny_agents(scene), self.env(scene)
            host, _ = env.get_host_state_for_next_assignment()
            state, _, _ = prepare_decision(env, learners['VA'], host, training=True)
            action = int(np.flatnonzero(state['mask'])[0])
            clock = UpdateClock(learners, training=True, warmup=0)
            for i in range(9):
                clock.remember('HA', state['obs'], state['mask'], action, -.1, state['obs'], state['mask'], 0.)
                self.assertEqual(learners['HA']._updates, max(0, i-1)//period)
                clock.barrier()
            self.assertEqual(learners['HA'].eligible_steps, 8)
            self.assertEqual(learners['HA']._updates, 8//period)
            results.append(learners['HA'].epsilon())
            clock.summary()
        self.assertEqual(*results)

    def test_fixed_actions_preserve_V3_physics_and_all_rewards(self):
        scene = self.scene()
        env = self.env(scene)
        kwargs = self.small_kwargs(baseline.build_env_kwargs)(scene, 0)
        old = baseline.CloudWorkflowEnv_VMAgents(**kwargs)
        old.energy_reward_scale = .001
        old.task_baseline_norm = 300.
        old.alpha_delay_host = old.alpha_delay_vm = .75
        learners = tiny_agents(scene)
        with redirect_stdout(io.StringIO()):
            old.reset()
            while not old.done_flag:
                while True:
                    host, oka = old.get_host_state_for_next_assignment()
                    raw, okb = env.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka:
                        break
                    prepare_decision(env, learners['VA'], raw, training=False)
                    ah = int(np.flatnonzero(host['mask'])[0])
                    old.host_select(ah); env.host_select(ah)
                    vm, _ = old.get_vm_state_for_current_task()
                    av = int(np.flatnonzero(vm['mask'])[0])
                    self.assertEqual(old.vm_assign(av), env.vm_assign(av))
                    self.assertEqual(old._records, env._records)
                self.assertEqual(old.finish_phase_and_advance(), env.finish_phase_and_advance())
                if not old.done_flag:
                    old.apply_manager_delta(np.zeros(5)); env.apply_manager_delta(np.zeros(5))
        self.assertEqual(old.total_energy, env.total_energy)

    def test_real_training_rollout_atomicity_replay_terminal_and_reward_balance(self):
        scene = self.scene()
        learners, env, rows = tiny_agents(scene), self.env(scene), []
        runtime.set_seed(7)
        original = learners['VA'].select_preview_action
        versions = []
        def select(s, **kwargs):
            versions.append((s['va_version'], learners['VA']._updates))
            return original(s, **kwargs)
        with patch.object(learners['VA'], 'select_preview_action', side_effect=select), redirect_stdout(io.StringIO()):
            result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append)
        self.assertGreater(learners['VA']._updates, 10)
        self.assertEqual(learners['HA']._updates, learners['HA'].eligible_steps//4)
        self.assertTrue(all(a == b for a, b in versions))
        self.assertEqual(result['task_count'], len(rows))
        self.assertEqual(result['wf_completed_count'], 2)
        for value in result['reward_audit'].values():
            self.assertTrue(np.isfinite(value))
        for key in ('physical_energy_balance', 'lower_energy_balance', 'HA_reward_formula_balance', 'VA_reward_formula_balance'):
            self.assertAlmostEqual(result['reward_audit'][key], 0., places=6)
        for transition in learners['HA'].buffer:
            self.assertEqual(transition[0]['candidates'].shape[1], 14)
            self.assertEqual(transition[4]['candidates'].shape[1], 14)
            self.assertFalse(transition[0]['candidates'].flags.writeable)
        self.assertTrue(learners['HA'].buffer[-1][-1])
        self.assertTrue(learners['VA'].buffer[-1][-1])
        self.assertTrue(all(not d['host_apf_enabled'] and not d['vm_apf_enabled'] for d in env.apf_decisions))

    def test_all_four_controls_paths_settings_and_CLI(self):
        paths = []
        for enabled in (False, True):
            for period in (1, 4):
                scene = self.scene(va_preview=enabled, ha_update_period=period)
                paths.append(scene.checkpoint_dir())
                learners, env = tiny_agents(scene), self.env(scene)
                host, _ = env.get_host_state_for_next_assignment()
                enhanced, _, _ = prepare_decision(env, learners['VA'], host, training=False)
                self.assertEqual(enhanced['obs']['candidates'].shape[1], 14 if enabled else 10)
                with redirect_stdout(io.StringIO()):
                    rollout = runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
                self.assertEqual(rollout['wf_completed_count'], 2)
                self.assertEqual(learners['HA']._updates, learners['HA'].eligible_steps//period)
        self.assertEqual(len(set(paths)), 4)
        parsed = parse_scenario('train', ['--no-va-preview', '--ha-update-period', '1', '--device', 'cpu'])
        self.assertEqual(parsed.mechanism, Mechanism(False, 1))
        stress = parse_scenario('eval', ['--workload-profile', 'stress_a1_l006_ref003'])
        self.assertEqual(runtime.build_env_kwargs(stress, 700)['arrival_lambda'], .06)
        self.assertEqual(runtime.build_env_kwargs(stress, 700)['deadline_alpha_small'], 1.)
        with self.assertRaises(ValueError):
            Mechanism(True, 0)

    def test_compact_training_and_flat_scene_named_CSV_paths(self):
        models, csv_paths = set(), set()
        parent = Path(self.scene().output_root)/'outputs/run/mahdrl_vaaware'
        for enabled in (False, True):
            for period in (1, 4):
                for resource in 'SML':
                    for deadline in ('Tight', 'Medium', 'Loose'):
                        source = self.scene(resource=resource, deadline=deadline,
                            va_preview=enabled, ha_update_period=period)
                        self.assertRegex(source.model_name(), r'^S[SML]_[TML]_m1_c[0-9a-f]{8}$')
                        self.assertEqual(len(source.checkpoint_dir().name), 17)
                        self.assertEqual(source.checkpoint_dir().parent.name, 'mahdrl_vaaware')
                        self.assertEqual(source.log_dir().name, source.checkpoint_dir().name)
                        models.add(source.checkpoint_dir())
                        for task in 'SML':
                            target = replace(source, mode='eval', task=task, train_resource=resource,
                                train_deadline=deadline, seed_spec='700-729')
                            path = target.evaluation_csv()
                            self.assertEqual(path.parent, parent)
                            self.assertIn(f'_{target.target_tag}_from_{source.source_tag}_', path.name)
                            self.assertIn(f'_p{int(enabled)}h{period}_', path.name)
                            self.assertIn('_s700-729_', path.name)
                            self.assertEqual(target.checkpoint_dir(), source.checkpoint_dir())
                            csv_paths.add(path)
        self.assertEqual(len(models), 36)
        self.assertEqual(len(csv_paths), 108)
        first = self.scene(mode='eval', seed_spec='700,702,703')
        second = replace(first, seed_spec='700,701,703')
        self.assertNotEqual(first.evaluation_csv(), second.evaluation_csv())
        standard = replace(first, legacy_split=False)
        self.assertNotEqual(standard.evaluation_csv(), replace(standard, workload_profile='stress_a1_l006_ref003').evaluation_csv())

    def test_source_normalization_transfers_to_unequal_VM_topology(self):
        topology, cache = self.root/'topology.json', self.root/'custom_cache.json'
        topology.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])), encoding='utf8')
        preparation = self.scene(mode='precompute', resource_config=str(topology), cache_output=str(cache), seed_spec='700')
        with redirect_stdout(io.StringIO()):
            precompute(preparation)
        scene = self.scene(mode='eval', resource_config=str(topology), cache_file=str(cache),
            train_resource='S', train_deadline='Tight', seed_spec='700')
        env, learners = self.env(scene, 700), tiny_agents(scene)
        host, _ = env.get_host_state_for_next_assignment()
        enhanced, states, _ = prepare_decision(env, learners['VA'], host, training=False)
        self.assertEqual(enhanced['obs']['candidates'].shape, (4, 14))
        self.assertEqual([len(s['mask']) for s in states.values()], [1, 2, 1, 3])
        with redirect_stdout(io.StringIO()):
            result = runtime.run_episode(self.env(scene, 700), learners)
        self.assertEqual(result['wf_completed_count'], 2)

    def test_real_train_save_load_evaluate_identity_and_frozen_policy(self):
        scene = self.scene()
        original = runtime.build_env_kwargs
        with patch.object(runtime, 'build_env_kwargs', self.small_kwargs(original)), \
             patch.object(runtime, 'build_agents', side_effect=tiny_agents), \
             patch('environments.mahdrl_oe_ge_apf_score_fast.apf_candidates', side_effect=AssertionError('APF called')), \
             redirect_stdout(io.StringIO()):
            trained = runtime.train(scene)
            evaluation = replace(scene, mode='eval', seed_spec='700')
            loaded = runtime.load_agents(evaluation)
            before = parameter_hash(loaded)
            counters = {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps, len(a.buffer)) for r, a in loaded.items()}
            env = self.env(evaluation, 700)
            result_a = runtime.run_episode(env, loaded)
            result_b = runtime.run_episode(self.env(evaluation, 700), loaded)
            self.assertEqual(result_a['total_energy'], result_b['total_energy'])
            self.assertEqual(before, parameter_hash(loaded))
            self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, a.eligible_steps, len(a.buffer)) for r, a in loaded.items()})
            summary = runtime.evaluate(evaluation)
        self.assertTrue(evaluation.evaluation_csv().is_file())
        with evaluation.evaluation_csv().open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['target_scene'], 'SS_T')
        self.assertEqual(rows[0]['source_scene'], 'SS_T')
        self.assertEqual(summary['wf_success_rate']['mean'], result_a['wf_success_rate'])
        saved_summary = json.loads((evaluation.evaluation_dir()/'summary.json').read_text(encoding='utf8'))
        self.assertEqual(saved_summary['model_version'], MODEL_VERSION)
        best = json.loads((scene.log_dir()/'training_summary.json').read_text(encoding='utf8'))
        self.assertEqual(best['checkpoint_selection']['workflow_success_threshold'], .92)
        self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
        incompatible = tiny_agents(replace(scene, ha_update_period=1))
        with self.assertRaises(ValueError):
            incompatible['VA'].load(str(scene.checkpoint_dir()/'best_vm.pth'))
        final_dir = self.root/'renamed_final'
        final_dir.mkdir()
        for stem in ('manager', 'host', 'vm'):
            shutil.copy2(scene.checkpoint_dir()/f'{stem}_final.pth', final_dir/f'best_{stem}.pth')
        with patch.object(runtime, 'build_agents', side_effect=tiny_agents), self.assertRaisesRegex(ValueError, 'best checkpoints'):
            runtime.load_agents(replace(evaluation, checkpoint_override=str(final_dir)))
        payload_path = scene.checkpoint_dir()/'best_vm.pth'
        payload = torch.load(payload_path, weights_only=True)
        payload['normalization']['checkpoint_group'] = 'wrong-group'
        torch.save(payload, payload_path)
        with patch.object(runtime, 'build_agents', side_effect=tiny_agents), self.assertRaisesRegex(ValueError, 'different saved groups'):
            runtime.load_agents(evaluation)
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            runtime.evaluate(evaluation)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_CUDA_preview_and_real_updates(self):
        scene = self.scene(device='cuda')
        env, learners = self.env(scene), tiny_agents(scene)
        with redirect_stdout(io.StringIO()):
            result = runtime.run_episode(env, learners, training=True, warmup=0)
        self.assertGreater(learners['VA']._updates, 0)
        self.assertGreater(learners['HA']._updates, 0)
        self.assertEqual(result['wf_completed_count'], 2)

    def test_production_width_networks_preview_and_optimizer_steps(self):
        scene, env = self.scene(), self.env()
        options = runtime.learning_kwargs
        with patch.object(runtime, 'learning_kwargs', side_effect=lambda role, device: dict(options(role, device), batch_size=2, buffer_size=8)):
            learners = runtime.build_agents(scene)
        host, _ = env.get_host_state_for_next_assignment()
        enhanced, states, _ = prepare_decision(env, learners['VA'], host, training=False)
        ah = int(np.flatnonzero(enhanced['mask'])[0])
        vm = states[env.host_ids[ah]]
        state, mask = env.get_manager_state(), env.get_manager_action_mask()
        observations = dict(TA=(state, mask), HA=(enhanced['obs'], enhanced['mask']), VA=(vm['obs'], vm['mask']))
        clock = UpdateClock(learners, training=True, warmup=0)
        for _ in range(5):
            for role, (obs, legal) in observations.items():
                clock.remember(role, obs, legal, int(np.flatnonzero(legal)[0]), -.1, obs, legal, 0.)
            clock.barrier()
        self.assertEqual(learners['HA']._updates, 1)
        self.assertEqual(learners['VA']._updates, 4)
        self.assertEqual(learners['TA']._updates, 4)


if __name__ == '__main__':
    unittest.main()
