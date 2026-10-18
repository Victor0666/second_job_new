"""F11 acceptance T01-T20: actual targets, decision traces and rollout boundaries."""
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common.f11 import (PendingDecision, DQNDecisionTrace, LocalDecisionTrace,
                        pd3qn_transition, finish_lower_phase, protect_training_outputs,
                        F11Logger, LEARNING_MODEL)
from common.d3qn_agent_EXDehid import D3QNAgent
from common.d3qn_agent_PER import D3QNAgent as PERAgent
from algorithms.marl.agents import A2CAgent, A2CConfig, LocalSharedActorIndependentCriticA2C
from algorithms.irws.vm_agent import PPOVMAgent, PPOConfig


def dqn(cls=D3QNAgent):
    return cls(2, 3, gamma=0.9, batch_size=1, buffer_size=16, hidden_dims=(8, 4), device=torch.device('cpu'))


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(values, dtype=torch.float32))
    def forward(self, obs):
        return self.q.unsqueeze(0).expand(len(obs), -1)


class F11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def target_check(self, cls, done):
        agent = dqn(cls)
        agent.online, agent.target = FixedQ([0., 1., 999.]), FixedQ([-4., -10., 888.])
        agent.optim = torch.optim.Adam(agent.online.parameters(), lr=3e-4)
        agent.remember(np.array([1., 2.]), np.array([1., 1., 0.]), 0, -2.,
                       np.array([3., 4.]), np.array([1., 1., 0.]), done)
        captured = []
        original = F.smooth_l1_loss
        def capture(input, target, **kwargs):
            captured.append(target.detach().clone())
            return original(input, target, **kwargs)
        with patch('torch.nn.functional.smooth_l1_loss', side_effect=capture):
            loss = agent.update()
        self.assertTrue(math.isfinite(loss))
        expected = -2. if done else -11.
        self.assertAlmostEqual(captured[0].item(), expected)
        if cls is PERAgent:
            self.assertAlmostEqual(agent.priorities[0], abs(expected) + 1e-6, places=5)
        return agent

    def test_T01_terminal_d3qn_target_minus_two(self):
        for cls in (D3QNAgent, PERAgent):
            self.target_check(cls, True)

    def test_T02_nonterminal_d3qn_target_minus_eleven(self):
        for cls in (D3QNAgent, PERAgent):
            agent = self.target_check(cls, False)
            self.assertAlmostEqual(agent.f11.snapshot()['future_abs_mean'], 9.)

    def test_T03_online_selects_target_evaluates_and_per_priority_changes(self):
        # Target itself prefers action 0; Online selects action 1; illegal action 2 is largest.
        for cls in (D3QNAgent, PERAgent):
            self.target_check(cls, False)

    def test_T04_zero_next_mask_rejected_only_for_nonterminal(self):
        for cls in (D3QNAgent, PERAgent):
            agent = dqn(cls)
            with self.assertRaisesRegex(ValueError, 'No legal action'):
                agent.remember(np.ones(2), np.ones(3), 0, -2, np.ones(2), np.zeros(3), 0)
            agent.remember(np.ones(2), np.ones(3), 0, -2, np.zeros(2), np.zeros(3), 1)
            self.assertEqual(len(agent.buffer), 1)

    def test_T05_host_trace_actual_successor_and_snapshot(self):
        agent = dqn()
        trace = DQNDecisionTrace(agent, 1.0)
        obs, mask = np.array([1., 2.]), np.ones(3)
        trace.record(obs, mask, 1, -2)
        obs[:] = 999; mask[:] = 0
        next_obs = np.array([3., 4.])
        trace.advance(next_obs, np.ones(3))
        next_obs[:] = 777
        transition = agent.buffer[0]
        np.testing.assert_array_equal(transition[0], [1, 2])
        np.testing.assert_array_equal(transition[4], [3, 4])
        self.assertEqual(transition[-1], 0)

    def test_T06_vm_trace_uses_next_selected_host_decision(self):
        agent = dqn()
        trace = DQNDecisionTrace(agent, 1.0)
        trace.record(np.array([1., 1.]), np.array([1., 0., 0.]), 0, -2, host_index=1)
        trace.advance(np.array([3., 9.]), np.array([0., 1., 0.]), host_index=3)
        np.testing.assert_array_equal(agent.buffer[0][4], [3, 9])
        np.testing.assert_array_equal(agent.buffer[0][5], [0, 1, 0])

    def test_T07_phase_end_is_not_terminal_and_three_assignments(self):
        agent = dqn()
        trace = DQNDecisionTrace(agent, 1.0)
        for i in range(3):
            trace.advance(np.array([i, 1.]), np.ones(3))
            trace.record(np.array([i, 1.]), np.ones(3), 0, 0. if i < 2 else -2)
            # A phase boundary itself does not close the trace.
            self.assertIsNotNone(trace.pending)
        trace.finish()
        self.assertEqual([t[-1] for t in agent.buffer], [0., 0., 1.])
        class IndexedQ(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
            def forward(self, obs):
                return self.values[obs[:, 0].long()].unsqueeze(1).expand(-1, 3)
        # Values at the last two real decisions encode the delayed terminal cost.
        agent.online, agent.target = IndexedQ([0., 0., 0.]), IndexedQ([-1.62, -1.8, -2.])
        agent.optim = torch.optim.Adam(agent.online.parameters(), lr=3e-4)
        targets=[]; original=F.smooth_l1_loss
        def capture(input, target, **kwargs):
            targets.extend(target.detach().cpu().tolist())
            return original(input,target,**kwargs)
        with patch('common.d3qn_agent_EXDehid.random.sample', return_value=list(agent.buffer)), \
             patch('torch.nn.functional.smooth_l1_loss',side_effect=capture):
            agent.batch_size = 3
            agent.update()
        np.testing.assert_allclose(targets, [-1.62, -1.8, -2.], atol=1e-6)


    def test_T08_last_lower_transition_is_terminal(self):
        for _ in range(2):
            agent = dqn(); trace = DQNDecisionTrace(agent, 1.0)
            trace.record(np.ones(2), np.ones(3), 0, -2)
            trace.finish(); trace.finish()
            self.assertIsNone(trace.pending)
            self.assertEqual(len(agent.buffer), 1)
            self.assertEqual(agent.buffer[0][-1], 1)

    def test_T09_rewards_settled_once_and_tail_not_lost(self):
        class Env:
            alpha_delay_host = alpha_delay_vm = 0.75
            done_flag = False
            def _episode_should_end(self): return True
        env = Env(); host = DQNDecisionTrace(dqn(), 1.0); vm = DQNDecisionTrace(dqn(), 1.0)
        for trace in (host, vm): trace.record(np.ones(2), np.ones(3), 0, 0, host_index=3)
        finish_lower_phase(env, host, vm, dict(lower_energy_already_emitted=True, r_energy_phase=-10))
        self.assertEqual(host.pending['reward'], 0)
        env.done_flag = True
        finish_lower_phase(env, host, vm, dict(lower_energy_already_emitted=False,
                                             r_energy_phase=-10, r_energy_by_host={3: -4}))
        self.assertEqual(host.emitted_reward, -2.5)
        self.assertEqual(vm.emitted_reward, -1.)
        self.assertEqual(host.recorded_reward, host.emitted_reward)

    def test_T10_pd3qn_real_successor_terminal_and_truncation(self):
        agent = dqn(PERAgent)
        pd3qn_transition(agent, np.ones(2), np.ones(3), 1, -2, np.array([8., 9.]), np.ones(3), False, False)
        np.testing.assert_array_equal(agent.buffer[-1][4], [8., 9.])
        self.assertEqual(agent.buffer[-1][-1], 0)
        pd3qn_transition(agent, np.ones(2), np.ones(3), 1, -2, np.ones(2), np.ones(3), False, True)
        self.assertEqual(agent.buffer[-1][-1], 0)
        pd3qn_transition(agent, np.ones(2), np.ones(3), 1, -2, np.zeros(2), np.zeros(3), True, False)
        self.assertEqual(agent.buffer[-1][-1], 1)
        pd3qn_transition(agent, np.ones(2), np.ones(3), 1, -2, np.zeros(2), np.zeros(3), False, True)
        self.assertEqual(agent.buffer[-1][-1], 1)
        self.assertEqual(agent.f11.truncation_cutoffs, 1)

    def a2c(self, local=False):
        cfg=A2CConfig(gamma=.9, hidden_dims=(8, 4))
        agent = LocalSharedActorIndependentCriticA2C(2, 3, 3, cfg) if local else A2CAgent(2, 3, cfg)
        with torch.no_grad():
            for p in agent.net.parameters(): p.zero_()
            if local:
                for head, val in zip(agent.net.value_heads, [3., 6., -10.]): head.bias.fill_(val)
            else: agent.net.value_head.bias.fill_(-10.)
        return agent

    def test_pd3qn_temporarily_missing_current_task_is_not_terminal(self):
        from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost
        from common.scenarios import Scenario
        env = CloudWorkflowEnv_PD3QN_Cost(dax_paths=Scenario().dax_paths(), random_seed=2,
                                         workflows_per_episode=1, deadline_mode='none')
        env.reset()
        env._cur_task_gid = None
        obs, reward, term, trunc, info = env.step(0)
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertTrue(np.any(env.get_action_mask() > .5))
        self.assertTrue(np.isfinite(obs).all())

    def test_stalled_environments_raise_instead_of_fabricating_terminal(self):
        from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Main
        from environments.marl import CloudWorkflowEnv_MARL as Marl
        from environments.irws import IRWSEnvRouteA as Irws
        for cls, advance in ((Main, '_advance_until_decision_energy_only'), (Marl, '_advance_until_decision')):
            env = object.__new__(cls)
            env.host_ids = []
            env._add_workflow_if_arrived = lambda: None
            env._has_decision_point = lambda: False
            env.event_heap = []
            env._no_more_arrivals = lambda: True
            env._episode_should_end = lambda: False
            with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
                getattr(env, advance)()
        env = object.__new__(Irws)
        env.current_time = 0
        env._process_completions_at_time = lambda t: None
        env._check_done = lambda: False
        env.ready_task_ids = []
        env.vm_available_at = np.zeros(1)
        env.next_arrival_idx = 0
        env.arrival_times = []
        env._next_completion_time = lambda: None
        with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
            env._advance_until_decision()

    def test_T11_global_a2c_real_one_step_target(self):
        agent=self.a2c()
        self.assertTrue(math.isfinite(agent.update_one_step(np.ones(2),np.ones(3),0,-2,np.ones(2),np.ones(3),0)))
        self.assertAlmostEqual(agent.f11.snapshot()['target_mean'],-11.)

    def test_T12_local_trace_actual_successor_and_same_host(self):
        agent=self.a2c(local=True); trace=LocalDecisionTrace(agent)
        trace.record(np.ones(2),np.ones(3),0,-2,host_index=2)
        trace.advance(np.array([8.,9.]),np.ones(3),host_index=2)
        self.assertAlmostEqual(agent.f11.snapshot()['target_mean'],-11.)
        self.assertIsNone(trace.pending)
        with self.assertRaisesRegex(ValueError,'No legal action'):
            trace.advance(np.ones(2),np.zeros(3),host_index=2)

    def test_local_state_before_step_matches_actual_conditional_vm_decision(self):
        from environments.marl import CloudWorkflowEnv_MARL
        from common.scenarios import Scenario
        env = CloudWorkflowEnv_MARL(dax_paths=Scenario().dax_paths(), random_seed=2,
                                   workflows_per_episode=1, deadline_mode='poisson_heftlike',
                                   dump_deadlines=False, print_deadlines=False)
        obs, _ = env.reset()
        host_index = int(np.flatnonzero(obs['global_mask'] > .5)[0])
        preview, mask, vm_map = env.get_local_state(env.host_ids[host_index])
        self.assertTrue(vm_map)
        self.assertTrue(np.any(mask > .5))
        actual = []
        original = env.get_local_state
        def capture(host):
            result = original(host)
            actual.append(result)
            return result
        with patch.object(env, 'get_local_state', side_effect=capture):
            env.step(host_index, int(np.flatnonzero(mask > .5)[0]))
        np.testing.assert_array_equal(preview, actual[0][0])
        np.testing.assert_array_equal(mask, actual[0][1])

    def test_T13_local_host_one_to_three_uses_next_critic(self):
        agent=self.a2c(local=True); trace=LocalDecisionTrace(agent)
        trace.record(np.ones(2),np.ones(3),0,-2,host_index=0)
        trace.advance(np.array([3.,3.]),np.ones(3),host_index=2)
        self.assertAlmostEqual(agent.f11.snapshot()['target_mean'],-11.)
        trace.record(np.ones(2),np.ones(3),0,-2,host_index=2)
        trace.finish()
        self.assertEqual(agent.f11.transitions,2)
        self.assertEqual(agent.f11.nonterminal,1)

    def ppo(self):
        return PPOVMAgent(PPOConfig(2,3,gamma=.9,update_epochs=1,minibatch_size=4),hidden=(8,4))

    def ppo_returns(self, done, tail):
        agent=self.ppo()
        action,info=agent.select_action(np.ones(2),np.ones(3))
        agent.remember(np.ones(2),np.ones(3),action,info['logp'],0,-2,done)
        targets=[];original=F.mse_loss
        def capture(input,target,**kwargs):
            targets.extend(target.detach().cpu().tolist())
            return original(input,target,**kwargs)
        with patch('torch.nn.functional.mse_loss',side_effect=capture):
            loss=agent.update(last_value=tail)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(agent.buf.size(),0)
        return targets

    def test_T14_nonterminal_ppo_tail_and_singleton_not_dropped(self):
        self.assertEqual(self.ppo_returns(0,-10),[-11.])
        agent=self.ppo()
        agent.remember(np.ones(2),np.ones(3),0,-1,0,-2,0)
        with self.assertRaisesRegex(ValueError,'next-decision value'):agent.update()
        self.assertEqual(agent.buf.size(),1)

    def test_T15_terminal_gae_does_not_cross_episode_boundary(self):
        self.assertEqual(self.ppo_returns(1,-10),[-2.])
        agent=self.ppo()
        for reward,done in [(-2,1),(100,0)]: agent.remember(np.ones(2),np.ones(3),0,-1,0,reward,done)
        targets=[];original=F.mse_loss
        def capture(input,target,**kwargs):
            targets.extend(target.detach().cpu().tolist());return original(input,target,**kwargs)
        with patch('torch.nn.functional.mse_loss',side_effect=capture):agent.update(last_value=7)
        self.assertIn(-2.,targets)

    def test_T16_ablation_learning_layers_preserved_and_real_traces_wired(self):
        import ast
        expected={'mahdrl':{'host','vm'},'ablation_mgrrand':{'host','vm'},
                  'ablation_hostrand':{'vm'},'ablation_vmrand':{'host'},'ablation_nohost':{'vm'}}
        for name,layers in expected.items():
            path=Path(__file__).resolve().parents[1]/'algorithms'/name/'train.py'
            tree=ast.parse(path.read_text(encoding='utf-8'))
            constructors=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                          and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                          and node.value.func.id=='DQNDecisionTrace']
            actual={node.targets[0].id.split('_')[0] for node in constructors}
            self.assertEqual(actual,layers)
            self.assertNotIn('z_vm_s',path.read_text(encoding='utf-8'))

    def test_T17_heuristic_fcfs_episode_matches_ertf_single_vm(self):
        from common.precompute_deadlines import build_env_kwargs, run_baseline_episode_fcfs_vmfirst
        from common.scenarios import Scenario
        from environments.base_fcfs import CloudWorkflowEnv_VMAgents as Reference
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as Heuristic
        kwargs=build_env_kwargs(Scenario());kwargs.update(workflows_per_episode=1,num_hosts=1,vms_per_host=(1,))
        outcomes=[]
        for cls in (Reference,Heuristic):
            local_kwargs=dict(kwargs)
            if cls is Heuristic:local_kwargs.pop("deadline_alpha")
            env=cls(**local_kwargs);env.reset();makespans,_,_=run_baseline_episode_fcfs_vmfirst(env)
            outcomes.append((makespans,env.total_energy))
        np.testing.assert_allclose(outcomes[0][0],outcomes[1][0])
        self.assertAlmostEqual(outcomes[0][1],outcomes[1][1])

    def test_T18_numerical_diagnostics_and_disable_switch(self):
        agent=self.target_check(D3QNAgent,False)
        for value in agent.f11.snapshot().values():
            if value is not None:self.assertTrue(math.isfinite(value))
        with tempfile.TemporaryDirectory() as directory:
            log=F11Logger(directory);log.log(0,{'DQN':agent},pending=0)
            data=json.loads(log.path.read_text())
            self.assertGreater(data['learners']['DQN']['bootstrap_nonzero_ratio'],0)
            with patch.dict('os.environ',{'F11_DIAGNOSTICS':'0'}):
                log.log(1,{'DQN':agent})
            self.assertEqual(len(log.path.read_text().splitlines()),1)

    def test_T19_save_load_all_learning_interfaces(self):
        agents=[dqn(),dqn(PERAgent),self.a2c(),self.a2c(local=True),self.ppo()]
        with tempfile.TemporaryDirectory() as directory:
            for i,agent in enumerate(agents):
                path=str(Path(directory)/f'{i}.pth');agent.save(path)
                clone=[dqn,dqn, self.a2c,self.a2c,self.ppo][i]() if i not in (1,3) else (dqn(PERAgent) if i==1 else self.a2c(local=True))
                clone.load(path, require_learning_model=True)
                net=getattr(agent,'online',getattr(agent,'net',None));other=getattr(clone,'online',getattr(clone,'net',None))
                for a,b in zip(net.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))
                ckpt=torch.load(path,map_location='cpu',weights_only=True)
                ckpt.pop('learning_model')
                torch.save(ckpt,path)
                clone.load(path)  # The historical explicit loading API remains compatible.
                with self.assertRaisesRegex(ValueError,'not an F11-trained model'):
                    clone.load(path,require_learning_model=True)

    def test_T20_no_cross_episode_pending_and_protect_history(self):
        agent=dqn();trace=DQNDecisionTrace(agent,1.)
        for ep in range(2):
            trace.record(np.array([ep,1]),np.ones(3),0,-2)
            trace.finish()
        self.assertEqual(len(agent.buffer),2)
        self.assertTrue(all(item[-1]==1 for item in agent.buffer))
        self.assertEqual(trace.recorded_reward,trace.emitted_reward)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'old.pth').write_bytes(b'history')
            with self.assertRaises(FileExistsError):protect_training_outputs(path,path/'logs')
            self.assertEqual((path/'old.pth').read_bytes(),b'history')


if __name__=='__main__':unittest.main()
