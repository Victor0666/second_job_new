"""MARL Q targets, policy isolation, action inputs and actual decision traces."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from algorithms.marl.q_agents import (
    Decision, IndependentLocalAgents, MODEL_VERSION, QAgent, QConfig, QDecisionTrace)
from common.scenarios import Scenario
from algorithms.marl.identity import DISPLAY_NAME, evaluation_directory, evaluation_csv
from environments.marl import CloudWorkflowEnv_MARL


class MARLQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agent(self, peer=0):
        return QAgent(2, 2, QConfig(gamma=.9, hidden_dims=()), peer_act_dim=peer)

    def decision(self, action=0, peer=None):
        return Decision(np.ones(2), np.ones(2), action, peer)

    def test_global_q_reads_both_actions(self):
        agent = self.agent(peer=2)
        with torch.no_grad():
            layer = agent.critic[0]
            layer.weight.zero_(); layer.bias.zero_()
            layer.weight[0, 3] = 4  # Global action 1.
            layer.weight[0, 5] = 9  # Local action 1.
        self.assertEqual(agent._q(self.decision(0, 0)).item(), 0)
        self.assertEqual(agent._q(self.decision(1, 0)).item(), 4)
        self.assertEqual(agent._q(self.decision(0, 1)).item(), 9)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            agent._q(self.decision())

    def test_critic_uses_actual_successor_q_and_terminal_is_zero(self):
        agent = self.agent()
        with torch.no_grad():
            agent.critic[0].weight.zero_(); agent.critic[0].bias.fill_(10)
            agent.critic[0].weight[0, 3] = 1000
            agent.actor[0].weight.zero_()
            agent.actor[0].bias.copy_(torch.log(torch.tensor([.75,.25])))
        with patch.object(agent, 'select_action', side_effect=AssertionError('Extra action draw')), \
                patch.object(agent, '_q', wraps=agent._q) as q:
            tail = agent.successor_q(self.decision(0))
            self.assertEqual(tail, 10)  # Neither the expectation 260 nor the maximum 1010.
            q.assert_called_once()
        self.assertEqual(agent.successor_q(self.decision(1)), 1010)
        self.assertTrue(all(p.grad is None for p in (*agent.actor.parameters(), *agent.critic.parameters())))
        agent.update_critic(self.decision(), -2, tail)
        self.assertAlmostEqual(agent.f11.snapshot()['target_mean'], 7, places=4)
        agent = self.agent()
        agent.update_critic(self.decision(), -2)
        self.assertEqual(agent.f11.snapshot()['target_mean'], -2)

    def test_behavior_samples_policy_probabilities_and_respects_mask(self):
        agent = self.agent()
        with torch.no_grad():
            agent.actor[0].weight.zero_()
            agent.actor[0].bias.copy_(torch.log(torch.tensor([.75,.25])))
        with torch.random.fork_rng():
            torch.manual_seed(31)
            draws = [agent.select_action(np.ones(2), np.ones(2)) for _ in range(512)]
            self.assertTrue(.15 < np.mean(draws) < .35)  # Policy sampling, not argmax or uniform.
            for _ in range(8):
                self.assertEqual(agent.select_action(np.ones(2), np.array([0,1])), 1)

    def test_global_successor_reads_actual_pair_without_local_policy_query(self):
        agent = self.agent(peer=2)
        with torch.no_grad():
            agent.actor[0].weight.zero_()
            agent.actor[0].bias.copy_(torch.log(torch.tensor([.25,.75])))
            agent.critic[0].weight.zero_(); agent.critic[0].bias.fill_(2)
            agent.critic[0].weight[0,3] = 4
            agent.critic[0].weight[0,5] = 10
        locals = IndependentLocalAgents(2,2,2,QConfig(hidden_dims=()))
        with patch.object(locals.agents[0], 'select_action', side_effect=AssertionError('Local queried')), \
                patch.object(locals.agents[1], 'select_action', side_effect=AssertionError('Local queried')), \
                patch.object(agent, '_q', wraps=agent._q) as q, \
                patch.object(agent, 'select_action', side_effect=AssertionError('Extra Global draw')):
            rng = torch.get_rng_state().clone()
            tail = agent.successor_q(self.decision(1, 1))
            self.assertEqual(tail, 16)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            q.assert_called_once()
        with self.assertRaises(ValueError): agent.successor_q(self.decision(1))
        self.assertFalse(hasattr(agent, 'sample_target_action'))
        self.assertFalse(hasattr(agent, 'sampled_q'))

    def test_terminal_trace_has_no_successor_sampling(self):
        agent = self.agent(peer=2)
        trace = QDecisionTrace(agent)
        trace.record(self.decision(0,1), -2)
        with patch.object(agent, 'successor_q', side_effect=AssertionError('Terminal bootstrap')):
            trace.finish()
        self.assertEqual(agent.f11.snapshot()['target_mean'], -2)

    def test_trace_updates_critic_then_actor_using_updated_q_once_per_transition(self):
        agent = self.agent()
        with torch.no_grad():
            for p in (*agent.actor.parameters(), *agent.critic.parameters()): p.zero_()
        trace, events = QDecisionTrace(agent), []
        critic_step, actor_step = agent.update_critic, agent.update_actor
        def critic(*args):
            events.append('critic')
            return critic_step(*args)
        def actor(decision):
            events.append('actor')
            updated_q = agent._q(decision).item()
            self.assertLess(updated_q, 0)  # Was zero before this reward's Critic update.
            result = actor_step(decision)
            self.assertNotEqual(result, 0)
            return result
        with patch.object(agent, 'update_critic', side_effect=critic), \
                patch.object(agent, 'update_actor', side_effect=actor):
            trace.record(self.decision(), -2)
            self.assertEqual(events, [])
            trace.advance(self.decision())
            self.assertEqual(events, ['critic','actor'])
            trace.record(self.decision(1), -3)
            trace.finish(); trace.finish()
        self.assertEqual(events, ['critic','actor','critic','actor'])
        self.assertEqual(agent.f11.transitions, 2)
        self.assertEqual(agent.f11.nonterminal, 1)

    def test_terminal_reward_changes_actor_on_the_same_transition(self):
        agents = [self.agent(), self.agent()]
        for agent, reward in zip(agents, (0., -100.)):
            with torch.no_grad():
                for p in (*agent.actor.parameters(), *agent.critic.parameters()): p.zero_()
            trace = QDecisionTrace(agent)
            trace.record(self.decision(), reward)
            trace.finish()
        self.assertFalse(all(torch.equal(a,b) for a,b in zip(agents[0].actor.parameters(), agents[1].actor.parameters())))

    def test_policy_uses_detached_q_without_entropy_or_clipping(self):
        agent = self.agent()
        with torch.no_grad():
            for p in agent.actor.parameters(): p.zero_()
            agent.critic[0].weight.zero_(); agent.critic[0].bias.fill_(3)
        critic = {k:v.clone() for k,v in agent.critic.state_dict().items()}
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=AssertionError('clipping enabled')):
            loss = agent.update_actor(self.decision())
            self.assertAlmostEqual(loss, 3*np.log(2), places=5)
            agent.update_critic(self.decision(), -2)
        # Actor update must not backpropagate into its Critic.
        other = self.agent(); other.critic.load_state_dict(critic)
        before = {k:v.clone() for k,v in other.critic.state_dict().items()}
        other.update_actor(self.decision())
        self.assertTrue(all(torch.equal(v, other.critic.state_dict()[k]) for k,v in before.items()))
        for kwargs in (dict(ent_coef=.01), dict(grad_clip=10)):
            with self.assertRaises(ValueError): QConfig(**kwargs)

    def test_hosts_and_actor_critic_have_no_shared_parameters_or_optimizer_states(self):
        locals = IndependentLocalAgents(2, 2, 2, QConfig(hidden_dims=(8,)))
        a, b = locals.agents
        sets = [{id(p) for p in m.parameters()} for m in (a.actor, a.critic, b.actor, b.critic)]
        self.assertEqual(len(set.union(*sets)), sum(map(len, sets)))
        before = {k:v.clone() for k,v in b.actor.state_dict().items()}
        a.update(self.decision(), -2)
        self.assertTrue(all(torch.equal(v, b.actor.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(b.actor_optim.state); self.assertFalse(b.critic_optim.state)

    def test_local_A_B_A_only_bootstraps_same_host_and_flushes_every_tail(self):
        a, b = self.agent(), self.agent()
        ta, tb = QDecisionTrace(a), QDecisionTrace(b)
        d = self.decision()
        ta.record(d, -2)
        tb.advance(d); tb.record(d, -3)
        self.assertEqual(a.f11.transitions, 0)
        self.assertIsNotNone(ta.pending)
        ta.advance(d); ta.record(self.decision(1), -4)
        self.assertEqual(a.f11.nonterminal, 1)
        self.assertEqual(b.f11.transitions, 0)
        ta.finish(); tb.finish()
        self.assertEqual(a.f11.transitions, 2); self.assertEqual(b.f11.transitions, 1)
        self.assertEqual(ta.recorded_reward, -6); self.assertEqual(tb.recorded_reward, -3)
        self.assertEqual(ta.emitted_reward, -6); self.assertEqual(tb.emitted_reward, -3)
        self.assertIsNone(ta.pending); self.assertIsNone(tb.pending)

    def test_checkpoint_roundtrip_and_reject_historical_models(self):
        g = self.agent(peer=2)
        locals = IndependentLocalAgents(2, 2, 2, QConfig(hidden_dims=()))
        with tempfile.TemporaryDirectory() as root:
            gp, lp = Path(root)/'global.pth', Path(root)/'locals.pth'
            g.update(self.decision(0, 1), -2)
            g.save(gp); locals.save(lp)
            restored = self.agent(peer=2); restored.load(gp)
            restored_l = IndependentLocalAgents(2, 2, 2, QConfig(hidden_dims=())); restored_l.load(lp)
            for net in ('actor','critic'):
                self.assertTrue(all(torch.equal(v,getattr(restored, net).state_dict()[k])
                                    for k,v in getattr(g, net).state_dict().items()))
            self.assertTrue(restored.actor_optim.state); self.assertTrue(restored.critic_optim.state)
            for path, agent in ((gp, restored), (lp, restored_l)):
                old = torch.load(path, weights_only=True)
                for version in ('marl_q_v4_1','marl_q_v4','marl_q_v3','marl_q_v2','marl_independent_q_v2','marl_independent_q_v1'):
                    old['model_version'] = version
                    torch.save(old, path)
                    before = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, 'retrain'): agent.load(path)
                    self.assertEqual(path.read_bytes(), before)
            old = torch.load(gp, weights_only=True)
            old.pop('model_version'); torch.save(old, gp)
            with self.assertRaisesRegex(ValueError, 'retrain'): restored.load(gp)
            locals.save(lp)
            wrong = IndependentLocalAgents(2, 2, 3, QConfig(hidden_dims=()))
            with self.assertRaisesRegex(ValueError, 'topology'): wrong.load(lp)
        self.assertIn(MODEL_VERSION, Scenario().model_name('marl'))
        self.assertNotEqual(Scenario(model_seed=1).model_name('marl'), Scenario(model_seed=2).model_name('marl'))

    def test_output_identity_separates_legacy_models_sources_and_evaluation_scenes(self):
        self.assertEqual(DISPLAY_NAME, 'MARL-Q4.2')
        self.assertEqual(MODEL_VERSION, 'marl_q_v4_2')
        scene = Scenario()
        root = Path('verification-root').resolve()
        directory = evaluation_directory(scene, root)
        self.assertIn(MODEL_VERSION, str(directory))
        self.assertIn(MODEL_VERSION, evaluation_csv(scene, root).name)
        self.assertIn(MODEL_VERSION, str(scene.checkpoint_dir('marl')))
        for other in (replace(scene, model_seed=2), replace(scene, train_task='M'),
                      replace(scene, train_resource='M'), replace(scene, train_deadline='Loose'),
                      replace(scene, task='M'), replace(scene, resource='M'), replace(scene, deadline='Loose')):
            self.assertNotEqual(directory, evaluation_directory(other, root))
        described = scene.describe('marl')
        self.assertEqual(described['algorithm'], MODEL_VERSION)
        self.assertEqual(described['display_name'], DISPLAY_NAME)

    def test_old_outputs_are_ignored_and_existing_evaluation_outputs_are_preserved(self):
        from algorithms.marl.evaluate import protect_evaluation_outputs
        from common.f11 import protect_training_outputs
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root/'outputs'/'run'/'marl'/'old.csv'
            old.parent.mkdir(parents=True); old.write_bytes(b'historical result')
            scene = Scenario()
            directory, csv = evaluation_directory(scene, root), evaluation_csv(scene, root)
            protect_evaluation_outputs(csv, directory)
            protect_training_outputs(root/'ckpts'/MODEL_VERSION, root/'logs'/MODEL_VERSION)
            directory.mkdir(parents=True); csv.write_bytes(b'new result')
            with self.assertRaises(FileExistsError): protect_evaluation_outputs(csv, directory)
            self.assertEqual(csv.read_bytes(), b'new result')
            self.assertEqual(old.read_bytes(), b'historical result')
            custom = replace(scene, marl_options={'out': str(root/'another.csv')})
            self.assertNotEqual(directory, evaluation_directory(custom, root))
            protect_evaluation_outputs(evaluation_csv(custom, root), evaluation_directory(custom, root))
            # Even an explicitly selected historical CSV cannot be appended to.
            with self.assertRaises(FileExistsError): protect_evaluation_outputs(old, root/'unused')
            partial = root/'partial'; partial.mkdir(); (partial/'seed700.json').write_bytes(b'partial')
            with self.assertRaises(FileExistsError): protect_evaluation_outputs(root/'missing.csv', partial)

    def test_evaluation_main_rejects_existing_csv_before_loading_or_writing(self):
        from algorithms.marl import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv = root/'existing.csv'; csv.write_bytes(b'history')
            cache = root/'cache.json'; cache.write_text('{}')
            for name in ('best_global.pth','best_local_actor_critics.pth'):
                (root/name).write_bytes(b'placeholder')
            scene = Scenario(checkpoint_override=str(root),
                marl_options={'out':str(csv),'deadline_cache':str(cache),'device':'cpu'})
            with patch.object(evaluate, 'SCENARIO', scene), \
                    patch.object(evaluate, 'build_agents_and_probe_dims') as load, \
                    patch.object(evaluate, 'CSVLogger') as logger:
                with self.assertRaises(FileExistsError): evaluate.main()
                load.assert_not_called(); logger.assert_not_called()
            self.assertEqual(csv.read_bytes(), b'history')

    def test_direct_evaluation_recorder_rejects_existing_seed_artifacts(self):
        from common.experiment_measurements import EvaluationRecorder
        with tempfile.TemporaryDirectory() as temporary, patch.dict('os.environ', {'EXPERIMENT_METRICS':'1'}):
            root = Path(temporary); path = root/'episodes'/'seed700.json'
            path.parent.mkdir(); path.write_bytes(b'historical seed')
            with self.assertRaises(FileExistsError):
                EvaluationRecorder(Scenario(), MODEL_VERSION,
                    SimpleNamespace(random_seed=700, episode_seed=700), {}, directory=root)
            self.assertEqual(path.read_bytes(), b'historical seed')

    def test_global_observation_distinguishes_hosts_and_local_has_no_identity_block(self):
        env = CloudWorkflowEnv_MARL(dax_paths=Scenario().dax_paths(), num_hosts=2,
            vms_per_host=(2,2), pc_tiers=(1.,4.), bw_tiers=(1000.,4000.),
            workflows_per_episode=1, deadline_mode='poisson_heftlike',
            dump_deadlines=False, print_deadlines=False)
        env.reset(seed=2)
        env.vm_running_task[:] = [0, -1, -1, 1]
        env._host_running_pc = {0:1., 1:4.}
        a, ma = env.get_global_state()
        env.vm_running_task[:] = [-1, 0, 1, -1]
        env._host_running_pc = {0:4., 1:1.}
        b, mb = env.get_global_state()
        self.assertEqual(a.shape, (5+2*2,))
        self.assertFalse(np.array_equal(a[-4:], b[-4:]))
        np.testing.assert_array_equal(ma, mb)
        local, mask, _ = env.get_local_state(env.host_ids[0])
        self.assertEqual(local.shape, (4+5*2,))
        self.assertEqual(len(local), env.local_obs_dim)
        np.testing.assert_array_equal(local[:4], b[:4])

    def test_retained_task_payload_matches_shared_instance(self):
        from common.precompute_deadlines import main as precompute
        from common.test_f18 import fixture_config
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root/'scales.json'; fixture_config(config)
            cache = root/'cache.json'
            with redirect_stdout(io.StringIO()):
                precompute(Scenario(mode='precompute',seed_spec='0',cache_output=str(cache)))
            kwargs = dict(dax_paths=Scenario().dax_paths(), random_seed=0, workflows_per_episode=2,
                          num_hosts=3, vms_per_host=(9,8,8), horizon=1e9, arrival_lambda=.03,
                          pc_tiers=(1.,2.,4.,6.,8.), bw_tiers=(1000.,2000.,4000.,6000.,8000.),
                          deadline_mode='cache_fcfs', deadline_cache_path=str(cache),
                          normalization_mode='f19_shared_v1', normalization_config=str(config),
                          source_train_scene='SS_T')
            marl = CloudWorkflowEnv_MARL(**kwargs,alpha_deadline_choices=(2.,3.),
                alpha_deadline_probs=(.8,.2),dump_deadlines=False,print_deadlines=False)
            ma = CloudWorkflowEnv_VMAgents(**kwargs,deadline_alpha_small_prob=.8)
            with redirect_stdout(io.StringIO()): ma.reset(); marl.reset(seed=0)
            tid = marl._cur_task_gid
            np.testing.assert_allclose(marl.get_global_state()[0][:3],ma._build_host_obs_for_task(tid)[0][:3],rtol=1e-6,atol=1e-6)
            # The retained task payload is shared, but V4 no longer copies HA's layout.
            local = marl.get_local_state(marl.host_ids[0])[0]
            self.assertEqual(local.shape, (49,))
            marl.get_global_state()
            np.testing.assert_array_equal(local,marl.get_local_state(marl.host_ids[0])[0])


if __name__ == '__main__':
    unittest.main()
