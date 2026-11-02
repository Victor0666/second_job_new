"""Repeated seeds preserve the split, learning state and checkpoint isolation."""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import csv
import io
import json
import unittest
from unittest.mock import patch

from common.experiment_protocol import DEFAULT_PROTOCOL, ExperimentProtocol, protocol_binding
from common.mahdrl_ablation import validate_checkpoint_selection, validation_selection
from common.scenarios import Scenario
from common import test_mahdrl_as_fast_hscore as fixtures
from common.test_mahdrl_as_fast_hscore_speed import tiny_agents
from algorithms.mahdrl_as_fast_hscore import runtime as base
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as Original, parse_scenario as original_parse
from algorithms.mahdrl_as_fast_hscore_speed.scenario import ScoreScenario as Speed, parse_scenario as speed_parse
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import ScoreScenario as Speed2, parse_scenario as speed2_parse
from algorithms.mahdrl_as_fast_hscore_speed2 import runtime

VARIANTS = ((Original, original_parse), (Speed, speed_parse), (Speed2, speed2_parse))


class ScheduleTests(unittest.TestCase):
    def test_checkpoint_budget_covers_all_cycles_and_rejects_invalid_evidence(self):
        rows = [dict(seed=600, wf_count=50, wf_success_count=47, wf_success_rate=.94, total_energy=10.)]
        metrics = validation_selection(rows)
        schedule = dict(order='fixed_seed_cycle', episodes=6000, train_seed_cycles=10, unique_training_seeds=600)

        def evidence(episode, training_schedule):
            protocol = dict(train=list(range(600)))
            if training_schedule is not None:
                protocol['training_schedule'] = training_schedule
            expected = dict(protocol=protocol, validation_seeds=[600])
            record = dict(expected, episode=episode, selected_episode=episode, completed_episodes=episode+1,
                validation_rows=rows, best_validation_rows=rows, validation_selection=metrics,
                best_validation_selection=dict(metrics, episode=episode, completed_episodes=episode+1), checkpoint_kind='best')
            return record, expected

        validate_checkpoint_selection(*evidence(5999, schedule))
        validate_checkpoint_selection(*evidence(599, None))
        for episode, selected_schedule in ((6000, schedule), (600, None)):
            with self.assertRaisesRegex(ValueError, 'outside training protocol'):
                validate_checkpoint_selection(*evidence(episode, selected_schedule))
        for change in (dict(episodes=5999), dict(train_seed_cycles=0), dict(train_seed_cycles=True),
                       dict(unique_training_seeds=599), dict(order='unknown')):
            with self.assertRaisesRegex(ValueError, 'Invalid checkpoint training seed schedule'):
                validate_checkpoint_selection(*evidence(1, dict(schedule, **change)))

    def test_6000_episodes_are_ten_disjoint_training_cycles(self):
        for cls, _ in VARIANTS:
            scene = cls(mode='train', episodes=6000, train_seed_cycles=10)
            seeds = [scene.training_seed(i) for i in range(6000)]
            self.assertEqual(seeds, list(range(600))*10)
            self.assertEqual(Counter(seeds), Counter({seed: 10 for seed in range(600)}))
            self.assertEqual([seeds[i] for i in (0, 599, 600, 1199, 5999)], [0, 599, 0, 599, 599])
            self.assertFalse(set(seeds) & set(scene.validation_seeds()))
            self.assertFalse(set(seeds) & set(scene.seeds))
            self.assertEqual(scene.validation_seeds(), [600])
            self.assertEqual(scene.protocol().episodes, 600)
            for episode in (-1, 6000):
                with self.assertRaises(ValueError):
                    scene.training_seed(episode)

    def test_old_budget_and_invalid_schedules_still_rejected(self):
        with self.assertRaises(ValueError):
            Scenario(mode='train', episodes=6000)
        with self.assertRaises(ValueError):
            replace(ExperimentProtocol.load(DEFAULT_PROTOCOL), episodes=6000)
        for cls, _ in VARIANTS:
            default = cls(mode='train')
            self.assertEqual([default.training_seed(i) for i in range(600)], list(range(600)))
            self.assertNotIn('training_schedule', default.selection_protocol())
            for options in (dict(episodes=6000), dict(train_seed_cycles=0), dict(train_seed_cycles=True),
                            dict(mode='eval', episodes=6000),
                            dict(train_seed_cycles=10), dict(episodes=5999, train_seed_cycles=10),
                            dict(episodes=6000, train_seed_cycles=10, legacy_split=True),
                            dict(mode='precompute', episodes=6000, train_seed_cycles=10)):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(ValueError):
                    cls(**dict(dict(mode='train'), **options))

    def test_train_eval_identity_and_calibration_reuse(self):
        for cls, parser in VARIANTS:
            for resource in 'SML':
                scene = cls(mode='train', resource=resource, episodes=6000, train_seed_cycles=10)
                default = cls(mode='train', resource=resource)
                scene.validate_calibration()
                self.assertEqual(scene.normalization_kwargs(), default.normalization_kwargs())
                self.assertEqual(scene.cache_path, default.cache_path)
                self.assertIn('_e6000r10_', scene.checkpoint_dir().name)
                self.assertLess(len(scene.checkpoint_dir().name), 32)
                self.assertNotEqual(scene.checkpoint_dir(), default.checkpoint_dir())
                for task in 'SML':
                    target = replace(scene, mode='eval', task=task, train_resource=resource, train_deadline='Tight')
                    self.assertEqual(target.checkpoint_dir(), scene.checkpoint_dir())
                    self.assertEqual(target.selection_protocol(), scene.selection_protocol())
                    self.assertIn('_e6000r10.csv', target.evaluation_csv().name)
                    self.assertIn(f'_{task}{resource}_T_from_S{resource}_T_', target.evaluation_csv().name)
                    self.assertEqual(target.evaluation_csv().parent, default.evaluation_csv().parent)
            for mode in ('train', 'eval'):
                with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as caught:
                    parser(mode, ['--episodes', '6000', '--train-seed-cycles', '10', '--dry-run'])
                self.assertEqual(caught.exception.code, 0)
                metadata = json.loads(out.getvalue())['experiment_protocol']
                self.assertEqual(metadata['training_schedule']['episodes'], 6000)
                self.assertEqual(metadata['train'], list(range(600)))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                parser('eval', ['--train-seed-cycles', '10'])
            self.assertEqual(caught.exception.code, 2)


class ContinuousTrainingTests(unittest.TestCase):
    def test_real_wrap_keeps_learners_replay_optimizer_and_clocks(self):
        fixtures.HScoreTests.setUpClass()
        try:
            root = fixtures.HScoreTests.root
            protocol_path = root/'two_seeds.json'
            protocol_path.write_text(json.dumps(dict(version='test_two_seed_cycles', train=[0, 1],
                validation=[600], test=[700], episodes=2, selection='minimum_validation_total_energy')), encoding='utf8')
            scales = json.loads(fixtures.HScoreTests.scales.read_text(encoding='utf8'))
            scales['generation'] = dict(**protocol_binding(protocol_path), calibration_seeds={'SS_T': [0, 1]})
            scales_path = root/'cycle_scales.json'
            scales_path.write_text(json.dumps(scales), encoding='utf8')
            scene = Speed2(mode='train', episodes=4, train_seed_cycles=2, device='cpu',
                experiment_protocol=str(protocol_path), normalization_config=str(scales_path),
                cache_file=str(fixtures.HScoreTests.cache), output_root=str(root/'cycles'))
            created, starts, ends, seeds = [], [], [], []

            def build_agents(request):
                agents = tiny_agents(request)
                created.append(agents)
                return agents

            def state(agents):
                return {role: (id(a), id(a.buffer), id(a.optim), a._updates, a._eps_steps,
                               a.eligible_steps, len(a.buffer), a.epsilon()) for role, a in agents.items()}

            def rollout(env, agents, **kwargs):
                training = kwargs.get('training', False)
                if training:
                    seeds.append(env.random_seed)
                    starts.append(state(agents))
                result = base.run_episode(env, agents, warmup=0, **kwargs)
                if training:
                    ends.append(state(agents))
                return result

            with patch.object(base, 'build_env_kwargs', fixtures.HScoreTests.small_kwargs(base.build_env_kwargs)), \
                    redirect_stdout(io.StringIO()):
                runtime.train(scene, agent_builder=build_agents, rollout=rollout)
                evaluation = replace(scene, mode='eval', seed_spec='700', train_resource='S', train_deadline='Tight')
                loaded = base.load_agents(evaluation, agent_builder=tiny_agents)
                final = tiny_agents(evaluation)['TA']
                final.load(str(scene.checkpoint_dir()/'manager_final.pth'))
                runtime.evaluate(evaluation, agent_builder=tiny_agents, rollout=rollout)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    base.load_agents(replace(evaluation, episodes=2, train_seed_cycles=1,
                        checkpoint_override=str(scene.checkpoint_dir())), agent_builder=tiny_agents)
            self.assertEqual(len(created), 1)
            self.assertEqual(seeds, [0, 1, 0, 1])
            self.assertEqual(starts[1:], ends[:-1])
            self.assertTrue(all(ends[-1][role][3] > ends[1][role][3] for role in ('TA', 'HA', 'VA')))
            self.assertEqual(len({a.loaded_group for a in loaded.values()}), 1)
            self.assertEqual(final.loaded_selection['completed_episodes'], 4)
            with evaluation.evaluation_csv().open(encoding='utf8', newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((row['training_episodes'], row['train_seed_cycles']), ('4', '2'))
            rows = [json.loads(line) for line in (scene.log_dir()/'episodes.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([row['instance_seed'] for row in rows], seeds)
            summary = json.loads((scene.log_dir()/'training_summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['episodes'], 4)
            self.assertEqual(summary['protocol']['training_schedule']['train_seed_cycles'], 2)
        finally:
            fixtures.HScoreTests.tearDownClass()


if __name__ == '__main__':
    unittest.main()
