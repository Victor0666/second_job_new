"""VAware TL: global legality, physical parity, atomic updates and isolated models."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from common.test_mahdrl_oe_ge_apf_score import parameter_hash
from algorithms.mahdrl_as_fast_vaaware_tl import runtime, agent, config, selection
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import ScoreScenario, parse_scenario
from algorithms.mahdrl_as_fast_vaaware_tl.updates import UpdateClock, DecisionTrace
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.reward import finish_lower_phase
from algorithms.mahdrl_as_fast_vaaware.preview import behavior_probabilities
from algorithms.mahdrl_as_fast_vaaware import runtime as parent_runtime
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario as ParentScene
from algorithms.mahdrl_oe_ge_apf_score_fast_tl import runtime as old_runtime
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.scenario import ScoreScenario as OldScene


def tiny_agents(scene):
    options = dict(hidden_dims=(16, 8), batch_size=2, buffer_size=8, device=scene.device or "cpu")
    learners = dict(TA=agent.TAAgent(15, 243, **options),
        VA=agent.SetD3QNAgent(23, 14, entity_dims=(8, 8), **options))
    return {role: a.bind(scene, role) for role, a in learners.items()}


class VAwareTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="vaaware_tl_")
        cls.root = Path(cls.temp.name)
        cls.scales, cls.cache = cls.root/'scales.json', cls.root/'cache.json'
        fixture_config(cls.scales)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute', seed_spec='0,1,2,600,700', cache_output=str(cls.cache)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, klass=ScoreScenario, **kwargs):
        values = dict(mode='train', episodes=2, legacy_split=True, device='cpu',
            normalization_config=str(self.scales), cache_file=str(self.cache),
            output_root=str(self.root/self._testMethodName))
        return klass(**dict(values, **kwargs))

    @staticmethod
    def small_kwargs(original):
        return lambda scene, seed: dict(original(scene, seed), workflows_per_episode=2, max_ready_tasks=100)

    def env(self, scene=None, module=runtime, seed=0):
        scene = scene or self.scene()
        with patch.object(module, 'build_env_kwargs', self.small_kwargs(module.build_env_kwargs)):
            env = module.build_env(scene, seed)
        with redirect_stdout(io.StringIO()):
            env.reset()
        return env

    def test_global_candidates_match_local_information_without_APF_or_deadline_filter(self):
        env = self.env()
        busy = env.host_to_vm_indices[env.host_ids[0]][0]
        env.vm_available_at[busy] = env.current_time+100
        with patch('environments.mahdrl_oe_ge_apf_score_fast.apf_candidates', side_effect=AssertionError('APF called')):
            vm, ok = env.get_vm_state_for_next_assignment()
            self.assertTrue(ok)
            self.assertEqual(vm['obs']['context'].shape, (23,))
            self.assertEqual(vm['obs']['candidates'].shape, (env.num_vms, 14))
            expected = np.flatnonzero(env._vm_idle_mask() & (env.vm_available_at <= env.current_time+1e-12))
            np.testing.assert_array_equal(np.flatnonzero(vm['mask']), expected)
            self.assertEqual(vm['mask'][busy], 0)
            for hid in env.host_ids:
                local, mask = env._build_vm_obs_for_task_host(env._cur_tid, hid)
                indices = env.host_to_vm_indices[hid]
                b, k = env.base_vm_obs_dim, len(indices)
                np.testing.assert_array_equal(vm['obs']['context'], np.r_[local[:22], local[b:b+1]])
                np.testing.assert_array_equal(vm['obs']['candidates'][indices, :5], local[27:b].reshape(env.max_vms_per_host, 5)[:k])
                np.testing.assert_array_equal(vm['obs']['candidates'][indices, 7:12], np.broadcast_to(local[22:27], (k, 5)))
                np.testing.assert_array_equal(vm['mask'][indices], mask[:k])
            self.assertFalse(vm['obs']['candidates'].flags.writeable)
            with self.assertRaises(ValueError): env.vm_assign_global(int(busy))
            with self.assertRaises(RuntimeError): env.host_select(0)
        # A fresh, overdue task retains every physically legal VM, including slow ones.
        env = self.env()
        host, _ = env.get_host_state_for_next_assignment()
        tid = env._cur_tid
        env.task_baseline_finish[tid] = env.current_time-10
        env._invalidate_decision()
        vm = env._build_global_vm_state_for_task(tid)
        self.assertEqual(int(vm['mask'].sum()), env.num_vms)

    def test_same_assignments_preserve_VAware_physics_and_rewards(self):
        env = self.env()
        parent = self.env(self.scene(ParentScene), parent_runtime)
        with redirect_stdout(io.StringIO()):
            while not env.done_flag:
                while True:
                    vm, oka = env.get_vm_state_for_next_assignment()
                    host, okb = parent.get_host_state_for_next_assignment()
                    self.assertEqual(oka, okb)
                    if not oka: break
                    self.assertEqual(env._cur_tid, parent._cur_tid)
                    j = int(np.flatnonzero(vm['mask'])[-1])
                    hid = env.vms[env.vm_ids[j]].host_id
                    slot = env.host_to_vm_indices[hid].index(j)
                    parent.host_select(parent.host_ids.index(hid))
                    rh, rv, info = parent.vm_assign(slot)
                    tl_rv, tl_info = env.vm_assign_global(j)
                    self.assertEqual(rv, tl_rv)
                    for key in ('assignment_energy', 'r_delay', 'r_energy_total'):
                        self.assertEqual(info[key], tl_info[key])
                    self.assertEqual(env._records, parent._records)
                self.assertEqual(env.finish_phase_and_advance(), parent.finish_phase_and_advance())
                if not env.done_flag:
                    env.apply_manager_delta(np.zeros(5)); parent.apply_manager_delta(np.zeros(5))
        self.assertEqual(env.total_energy, parent.total_energy)

    def test_global_GE_probabilities_and_deterministic_ties(self):
        scene, env = self.scene(), self.env()
        va = tiny_agents(scene)['VA']
        vm, _ = env.get_vm_state_for_next_assignment()
        n = len(vm['mask'])
        q = np.arange(n, dtype=np.float32)
        q[-2:] = 100
        times = np.arange(n, 0, -1, dtype=float)
        with patch.object(va.online, 'forward', return_value=torch.tensor(q[None])), patch.object(va, 'epsilon', return_value=.4), \
             patch('numpy.random.choice', return_value=1) as draw:
            va.select_action(vm['obs'], vm['mask'], performance_times=times)
            np.testing.assert_allclose(draw.call_args.kwargs['p'], behavior_probabilities(q, vm['mask'], times, .4))
            self.assertEqual(va.select_action(vm['obs'], vm['mask'], deterministic=True, count_step=False,
                performance_times=times), n-2)
            self.assertEqual(draw.call_count, 1)
        self.assertEqual(va._action_calls, 1)

    def test_real_updates_follow_assignment_and_preserve_replay_and_energy(self):
        scene, env = self.scene(), self.env()
        learners, rows, phases = tiny_agents(scene), [], []
        va = learners['VA']
        select, assign = va.select_action, env.vm_assign_global
        selected_version = None
        def choose(*args, **kwargs):
            nonlocal selected_version
            selected_version = va._updates
            return select(*args, **kwargs)
        def execute(action):
            self.assertEqual(va._updates, selected_version)
            return assign(action)
        with patch.object(va, 'select_action', side_effect=choose), patch.object(env, 'vm_assign_global', side_effect=execute), \
             patch.object(env, 'host_select', side_effect=AssertionError('HA selection called')), redirect_stdout(io.StringIO()):
            result = runtime.run_episode(env, learners, training=True, warmup=0, assignment_log=rows.append, phase_log=phases.append)
        self.assertEqual(result['wf_completed_count'], 2)
        self.assertEqual(len(rows), result['task_count'])
        self.assertGreater(va._updates, 10)
        self.assertEqual(set(result['update_clock']['optimizer_steps']), {'TA', 'VA'})
        self.assertEqual(result['update_clock']['transitions']['VA'], len(rows))
        self.assertEqual(result['update_clock']['transitions']['TA'], sum(p['assign_cnt'] > 0 for p in phases))
        for key in ('physical_energy_balance', 'lower_energy_balance', 'VA_reward_formula_balance'):
            self.assertAlmostEqual(result['reward_audit'][key], 0, places=6)
        self.assertNotIn('HA_reward_formula_balance', result['reward_audit'])
        self.assertTrue(va.buffer[-1][-1])
        for transition in va.buffer:
            self.assertEqual(transition[0]['candidates'].shape[1], 14)
            self.assertFalse(transition[0]['candidates'].flags.writeable)
            if not transition[-1]: self.assertGreater(transition[5].sum(), 0)
        self.assertTrue(all(not d['host_apf_enabled'] and not d['vm_apf_enabled'] for d in env.apf_decisions))

    def test_clock_defers_updates_and_terminal_settlement_is_once(self):
        learners, env = tiny_agents(self.scene()), self.env()
        vm, _ = env.get_vm_state_for_next_assignment()
        clock = UpdateClock(learners, training=True, warmup=0)
        trace = DecisionTrace(clock, 'VA')
        action = int(np.flatnonzero(vm['mask'])[0])
        for _ in range(4):
            trace.advance(vm['obs'], vm['mask'])
            before = learners['VA']._updates
            trace.record(vm['obs'], vm['mask'], action, -.1)
            self.assertEqual(learners['VA']._updates, before)
            clock.barrier()
        trace.add_reward(-.25)
        trace.finish(truncated=True)
        trace.finish(truncated=True)
        clock.barrier()
        self.assertEqual(clock.summary()['transitions']['VA'], 4)
        self.assertEqual(learners['VA']._updates, 3)
        self.assertEqual(learners['VA'].buffer[-1][3], -.35)
        self.assertTrue(learners['VA'].buffer[-1][-1])

    def test_deferred_energy_is_attached_once_to_the_retained_VA(self):
        learners, env = tiny_agents(self.scene()), self.env()
        vm, _ = env.get_vm_state_for_next_assignment()
        clock = UpdateClock(learners, training=True, warmup=0)
        trace = DecisionTrace(clock, 'VA')
        trace.record(vm['obs'], vm['mask'], int(np.flatnonzero(vm['mask'])[0]), -.1)
        env.done_flag = True
        info = dict(lower_assignment_energy_unemitted=8.)
        finish_lower_phase(env, trace, info)
        self.assertEqual(clock.summary()['transitions']['VA'], 1)
        self.assertAlmostEqual(learners['VA'].buffer[-1][3], -.1-.25*.001*8.)
        self.assertTrue(learners['VA'].buffer[-1][-1])
        with self.assertRaisesRegex(RuntimeError, 'no preceding VA'):
            finish_lower_phase(env, trace, info)

    def test_checkpoints_reject_wrong_identity_roles_and_selection_evidence(self):
        scene = self.scene()
        learners = tiny_agents(scene)
        rows = [dict(seed=scene.validation_seeds()[0], wf_count=50, wf_success_count=46,
                     wf_success_rate=.92, total_energy=10.)]
        candidate = selection.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        scene.checkpoint_dir().mkdir(parents=True)
        runtime.save_agents(learners, scene.checkpoint_dir(), 'best', episode=0,
            candidate=candidate, best=best, rows=rows, best_rows=rows)
        path = scene.checkpoint_dir()/'best_vm.pth'
        valid = torch.load(path, weights_only=True)
        for kind in ('old_model', 'role', 'scope', 'evidence'):
            payload = deepcopy(valid)
            meta = payload['normalization']
            if kind == 'old_model': meta['model_version'] = 'mahdrl_as_fast_v2_tl_noapf_srgt092_v2'
            if kind == 'role': meta['algorithm_variant']['role'] = 'HA'
            if kind == 'scope': meta['tl_policy']['vm_selection_scope'] = 'local'
            if kind == 'evidence': meta[config.CHECKPOINT_METADATA_KEY]['validation_rows'][0]['wf_success_count'] = 0
            torch.save(payload, path)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                tiny_agents(scene)['VA'].load(str(path))
        with self.assertRaises(ValueError): tiny_agents(scene)['VA'].bind(scene, 'HA')
        with self.assertRaises(ValueError):
            agent.SetD3QNAgent(30, 7, hidden_dims=(16, 8), entity_dims=(8, 8)).bind(scene, 'VA')

    def test_train_save_reload_frozen_eval_and_strict_identity(self):
        scene = self.scene()
        with patch.object(runtime, 'build_env_kwargs', self.small_kwargs(runtime.build_env_kwargs)), \
             patch.object(runtime, 'build_agents', side_effect=tiny_agents), redirect_stdout(io.StringIO()):
            trained = runtime.train(scene)
            target = replace(scene, mode='eval', seed_spec='700')
            loaded = runtime.load_agents(target)
            before = parameter_hash(loaded)
            counters = {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()}
            with patch.object(runtime, 'load_agents', return_value=loaded): runtime.evaluate(target)
            self.assertEqual(parameter_hash(loaded), before)
            self.assertEqual(counters, {r: (a._updates, a._eps_steps, a._action_calls, len(a.buffer)) for r,a in loaded.items()})
            with self.assertRaises(FileExistsError): runtime.train(scene)
            with self.assertRaises(FileExistsError): runtime.evaluate(target)
            final_dir = self.root/'renamed_final'; final_dir.mkdir()
            for stem in ('manager', 'vm'):
                shutil.copy2(scene.checkpoint_dir()/f'{stem}_final.pth', final_dir/f'best_{stem}.pth')
            with self.assertRaisesRegex(ValueError, 'best checkpoints'):
                runtime.load_agents(replace(target, checkpoint_override=str(final_dir)))
            payload_path = scene.checkpoint_dir()/'best_vm.pth'
            payload = torch.load(payload_path, weights_only=True)
            payload['normalization']['checkpoint_group'] = 'different-group'
            torch.save(payload, payload_path)
            with self.assertRaisesRegex(ValueError, 'different saved groups'): runtime.load_agents(target)
        self.assertTrue(all(a._updates > 0 for a in trained.values()))
        self.assertEqual(sorted(p.name for p in scene.checkpoint_dir().glob('*.pth')),
            ['best_manager.pth', 'best_vm.pth', 'manager_final.pth', 'vm_final.pth'])
        with target.evaluation_csv().open(encoding='utf8', newline='') as f: rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]['source_scene'], 'SS_T')
        self.assertEqual(rows[0]['vm_selection_scope'], 'global')
        if os.environ.get('EXPERIMENT_METRICS', '1') != '0':
            measure = json.loads((scene.log_dir()/'training_measurements.jsonl').read_text().splitlines()[0])
            self.assertEqual(set(measure['policy_calls']), {'TA', 'VA'})
            self.assertGreater(measure['policy_calls']['VA'], 0)
        for klass, module in ((ParentScene, parent_runtime), (OldScene, old_runtime)):
            other = self.scene(klass)
            # Same-width TA still must not accept a different scientific identity.
            with self.assertRaises(ValueError):
                module.build_agents(other)['TA'].load(str(scene.checkpoint_dir()/'best_manager.pth'))

    def test_paths_protocol_CLI_and_workload_are_aligned(self):
        parent = self.scene(ParentScene)
        source = self.scene()
        self.assertEqual(source.normalization_kwargs(), parent.normalization_kwargs())
        self.assertEqual(source.cache_path, parent.cache_path)
        self.assertIs(selection.selection_key, parent_runtime.selection_key)
        row = dict(seed=600, wf_count=50, wf_success_count=46, wf_success_rate=.92, total_energy=1.)
        self.assertTrue(selection.validation_selection([row])['feasible'])
        models, csvs = set(), set()
        for res in 'SML':
            for ddl in ('Tight', 'Medium', 'Loose'):
                src = replace(source, resource=res, deadline=ddl)
                self.assertEqual(len(src.model_name()), 17)
                models.add(src.checkpoint_dir())
                for task in 'SML':
                    target = replace(src, mode='eval', task=task, train_resource=res, train_deadline=ddl)
                    self.assertEqual(target.checkpoint_dir(), src.checkpoint_dir())
                    self.assertEqual(target.evaluation_csv().parent.name, config.OUTPUT_TAG)
                    self.assertIn(f'_{target.target_tag}_from_{src.source_tag}_', target.evaluation_csv().name)
                    csvs.add(target.evaluation_csv())
        self.assertEqual((len(models), len(csvs)), (9, 27))
        a = replace(source, mode='eval', seed_spec='700,702,703')
        self.assertNotEqual(a.evaluation_csv(), replace(a, seed_spec='700,701,703').evaluation_csv())
        for mode in ('train', 'eval', 'precompute'):
            with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as e:
                parse_scenario(mode, ['--dry-run'])
            self.assertEqual(e.exception.code, 0)
            self.assertEqual(json.loads(output.getvalue())['variant']['learned_roles'], ['TA', 'VA'])
        for args in (['--va-preview'], ['--ha-update-period', '4']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e: parse_scenario('train', args)
            self.assertEqual(e.exception.code, 2)
        stress = parse_scenario('eval', ['--workload-profile', 'stress_a1_l006_ref003'])
        self.assertEqual(runtime.build_env_kwargs(stress, 700)['arrival_lambda'], .06)
        self.assertEqual(runtime.build_env_kwargs(stress, 700)['deadline_alpha_small'], 1.)

    def test_transfer_to_unequal_VM_topology(self):
        spec, cache = self.root/'topology.json', self.root/'custom_cache.json'
        spec.write_text(json.dumps(dict(vms_per_host=[1, 2, 1, 3])), encoding='utf8')
        prep = self.scene(mode='precompute', resource_config=str(spec), cache_output=str(cache), seed_spec='700')
        with redirect_stdout(io.StringIO()): precompute(prep)
        source = self.scene()
        learners = tiny_agents(source)
        rows = [dict(seed=source.validation_seeds()[0], wf_count=50, wf_success_count=46, wf_success_rate=.92, total_energy=100.)]
        candidate = selection.validation_selection(rows)
        best = dict(candidate, episode=0, completed_episodes=1)
        source.checkpoint_dir().mkdir(parents=True)
        runtime.save_agents(learners, source.checkpoint_dir(), 'best', episode=0,
            candidate=candidate, best=best, rows=rows, best_rows=rows)
        target = self.scene(mode='eval', resource_config=str(spec), cache_file=str(cache), seed_spec='700',
            train_resource='S', train_deadline='Tight', checkpoint_override=str(source.checkpoint_dir()))
        runtime.validate_inputs(target)
        with patch.object(runtime, 'build_agents', side_effect=tiny_agents): loaded = runtime.load_agents(target)
        env = self.env(target, seed=700)
        vm, _ = env.get_vm_state_for_next_assignment()
        self.assertEqual(vm['obs']['candidates'].shape, (7, 14))
        before = parameter_hash(loaded)
        with redirect_stdout(io.StringIO()): result = runtime.run_episode(self.env(target, seed=700), loaded)
        self.assertEqual(result['wf_completed_count'], 2)
        self.assertEqual(parameter_hash(loaded), before)

    def test_production_width_real_optimizer_steps(self):
        scene, env = self.scene(), self.env()
        options = runtime.learning_kwargs
        with patch.object(runtime, 'learning_kwargs', side_effect=lambda role, device: dict(options(role, device), batch_size=2, buffer_size=8)):
            learners = runtime.build_agents(scene)
        vm, _ = env.get_vm_state_for_next_assignment()
        state, mask = env.get_manager_state(), env.get_manager_action_mask()
        clock = UpdateClock(learners, training=True, warmup=0)
        for _ in range(5):
            for role, (obs, legal) in dict(TA=(state, mask), VA=(vm['obs'], vm['mask'])).items():
                clock.remember(role, obs, legal, int(np.flatnonzero(legal)[0]), -.1, obs, legal, 0.)
            clock.barrier()
        self.assertEqual({r:a._updates for r,a in learners.items()}, dict(TA=4, VA=4))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_CUDA_global_policy_and_real_updates(self):
        scene = self.scene(device='cuda')
        learners = tiny_agents(scene)
        with redirect_stdout(io.StringIO()): result = runtime.run_episode(self.env(scene), learners, training=True, warmup=0)
        self.assertEqual(result['wf_completed_count'], 2)
        self.assertTrue(all(a._updates > 0 for a in learners.values()))


if __name__ == '__main__':
    unittest.main()
