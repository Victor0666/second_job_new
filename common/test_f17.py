"""F17 normalized TA transitions, candidate masks, state lifecycle and real DQN interaction."""
from contextlib import redirect_stdout
import importlib
import io
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from common.ta_weights import WEIGHT_TOLERANCE, TA_WEIGHT_MODEL, TA_ALGORITHMS
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Base, MANAGER_ACTION_TABLE


class F17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def weights(self, values=(.2, .2, .2, .2, .2)):
        env = object.__new__(Base)
        env.task_heur_dim = 5
        env.set_manager_combo(values)
        return env

    def environment(self, **kwargs):
        from common.scenarios import Scenario
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
        return CloudWorkflowEnv_VMAgents(dax_paths=Scenario().dax_paths(), random_seed=2,
                                        workflows_per_episode=2, max_ready_tasks=100,
                                        deadline_mode='none', **kwargs)

    def invariant(self, env):
        self.assertTrue(np.isfinite(env.combo_weights).all())
        self.assertTrue((env.combo_weights >= 0).all())
        self.assertAlmostEqual(float(env.combo_weights.astype(np.float64).sum()), 1., places=7)
        self.assertNotIn('manager_raw_w', vars(env))
        mask = env.get_manager_action_mask()
        self.assertEqual(mask.shape, (243,))
        noop = np.flatnonzero((MANAGER_ACTION_TABLE == 0).all(axis=1))[0]
        self.assertEqual(mask[noop], 1.)

    def test_T01_constructor_and_episode_reset(self):
        env = self.environment()
        np.testing.assert_array_equal(env.combo_weights, np.full(5,.2,dtype=np.float32))
        env.reset(); self.invariant(env)
        env.apply_manager_delta([.1,0,0,0,0]); env.reset()
        np.testing.assert_array_equal(env.combo_weights,np.full(5,.2,dtype=np.float32))
        np.testing.assert_array_equal(env.get_manager_state()[-5:],env.combo_weights)
        custom = [1,2,3,4,5]
        env = self.environment(combo_weights=custom)
        custom[:] = [100]*5
        env.reset(); np.testing.assert_allclose(env.combo_weights,np.arange(1,6)/15)
        env.apply_manager_delta([.1,0,0,0,0]); env.reset()
        np.testing.assert_allclose(env.combo_weights,np.arange(1,6)/15)

    def test_T02_paper_formula_not_hidden_raw(self):
        env = self.weights()
        env.apply_manager_delta([.1,0,0,0,0])
        np.testing.assert_allclose(env.combo_weights,[3/11,2/11,2/11,2/11,2/11],atol=2e-8)
        self.assertGreater(abs(env.combo_weights[0]-.215686),.05)

    def test_T03_negative_candidate_mask_and_rejection_preserve_state(self):
        env = self.weights([.05,.25,.25,.25,.2])
        delta = np.array([-.1,0,0,0,0],dtype=np.float32)
        action = np.flatnonzero((MANAGER_ACTION_TABLE==delta).all(axis=1))[0]
        self.assertEqual(env.get_manager_action_mask()[action],0)
        before = env.combo_weights.copy()
        env._phase_started,env._phase_tasks=True,[3,7]
        with self.assertRaisesRegex(ValueError,'Illegal TA'):
            env.apply_manager_delta(delta)
        np.testing.assert_array_equal(env.combo_weights,before)
        self.assertTrue(env._phase_started);self.assertEqual(env._phase_tasks,[3,7])

    def test_T04_legal_decrement(self):
        env=self.weights();env.apply_manager_delta([-.1,0,0,0,0])
        np.testing.assert_allclose(env.combo_weights,[1/9,2/9,2/9,2/9,2/9],atol=2e-8)
        self.invariant(env)

    def test_T05_noop_exactly_preserves_weights(self):
        for values in ([.2]*5,[1,0,0,0,0],[.01,.09,.3,.5,.1]):
            env=self.weights(values);before=env.combo_weights.copy()
            env.apply_manager_delta(np.zeros(5))
            np.testing.assert_array_equal(env.combo_weights,before)
            self.invariant(env)

    def test_T06_sequential_updates_use_last_normalized_state(self):
        env=self.environment();env.reset()
        snapshot=env.get_manager_state().copy()
        rng=np.random.default_rng(17)
        for _ in range(300):
            mask=env.get_manager_action_mask()
            action=int(rng.choice(np.flatnonzero(mask)))
            expected=env.combo_weights.astype(np.float64)+MANAGER_ACTION_TABLE[action].astype(np.float64)
            expected=np.maximum(expected,0);expected/=expected.sum()
            env.apply_manager_delta(MANAGER_ACTION_TABLE[action]);self.invariant(env)
            np.testing.assert_allclose(env.combo_weights,expected,atol=3e-8)
            np.testing.assert_array_equal(env.get_manager_state()[-5:],env.combo_weights)
            ids,features=env._compute_task_heuristics_for_ready()
            expected_order=sorted(range(len(ids)), key=lambda i:(-float(features[i]@env.combo_weights),env.task_ready_time[ids[i]],ids[i]))
            env._phase_prepare_tasks()
            self.assertEqual(env._phase_tasks,[ids[i] for i in expected_order])
        np.testing.assert_array_equal(snapshot[-5:],np.full(5,.2,dtype=np.float32))

    def test_T07_same_normalized_state_has_same_successor(self):
        a=self.weights([1,2,3,4,5]);b=self.weights([10,20,30,40,50])
        np.testing.assert_array_equal(a.combo_weights,b.combo_weights)
        delta=[.1,0,0,-.1,0]
        a.apply_manager_delta(delta);b.apply_manager_delta(delta)
        np.testing.assert_array_equal(a.combo_weights,b.combo_weights)
        np.testing.assert_array_equal(a.get_manager_action_mask(),b.get_manager_action_mask())

    def test_T08_all_243_mask_entries_match_execution(self):
        rng=np.random.default_rng(17)
        states=[[.2]*5,[.05,.25,.25,.25,.2],[1,0,0,0,0],[.1,.1,.1,.1,.6]]
        states += rng.dirichlet(np.ones(5),size=15).tolist()
        np.testing.assert_array_equal(MANAGER_ACTION_TABLE,np.array(list(itertools.product((-.1,0.,.1),repeat=5)),dtype=np.float32))
        for state in states:
            env=self.weights(state);mask=env.get_manager_action_mask();before=env.combo_weights.copy()
            for action,delta in enumerate(MANAGER_ACTION_TABLE):
                candidate=before.astype(np.float64)+delta.astype(np.float64)
                expected=bool((candidate>=-WEIGHT_TOLERANCE).all() and np.maximum(candidate,0).sum()>0)
                self.assertEqual(bool(mask[action]),expected)
                env.combo_weights=before.copy()
                if expected:
                    env.apply_manager_delta(delta);self.invariant(env)
                else:
                    with self.assertRaises(ValueError):env.apply_manager_delta(delta)
                    np.testing.assert_array_equal(env.combo_weights,before)

    def test_T09_boundaries_nonfinite_and_invalid_totals(self):
        for values in ([0,1,0,0,0],[1e-12,1-1e-12,0,0,0],[.1,.2,.3,.4,0]):
            env=self.weights(values);self.invariant(env)
            for delta in ([.1]*5,[-.1]*5):
                before=env.combo_weights.copy()
                candidate=before.astype(np.float64)+delta
                if np.all(candidate>=-WEIGHT_TOLERANCE):env.apply_manager_delta(delta);self.invariant(env)
                else:
                    with self.assertRaises(ValueError):env.apply_manager_delta(delta)
        env=self.weights([.1,.2,.3,.4,0])
        env.apply_manager_delta(MANAGER_ACTION_TABLE[np.flatnonzero((MANAGER_ACTION_TABLE==np.array([-.1,0,0,0,0],dtype=np.float32)).all(axis=1))[0]])
        self.assertGreaterEqual(env.combo_weights[0],0)
        for delta in ([np.nan,0,0,0,0],[np.inf,0,0,0,0],[.2,0,0,0,0],[-.2]*5,[0]*4):
            env=self.weights();before=env.combo_weights.copy()
            with self.assertRaises(ValueError):env.apply_manager_delta(delta)
            np.testing.assert_array_equal(env.combo_weights,before)
        env=self.weights();env.combo_weights=np.zeros(5)
        with self.assertRaisesRegex(ValueError,'simplex'):env.get_manager_action_mask()

    def test_T10_external_weights_strict_validation_and_copy(self):
        env=self.weights();values=np.array([1.,2.,3.,4.,5.]);env.set_manager_combo(values)
        values[:]=0
        np.testing.assert_allclose(env.combo_weights,np.arange(1,6)/15)
        for values in ([0]*5,[-1,1,1,1,1],[np.nan]*5,[np.inf]*5,[1]*4):
            before=env.combo_weights.copy()
            with self.assertRaises(ValueError):env.set_manager_combo(values)
            np.testing.assert_array_equal(env.combo_weights,before)
        env.set_manager_combo([1e308]*5);self.invariant(env)
        env.set_manager_combo([1e-300]*5);self.invariant(env)

    def test_T11_real_dynamic_training_and_evaluation(self):
        from common.d3qn_agent_EXDehid import D3QNAgent
        from common.f11 import DQNDecisionTrace, finish_lower_phase
        from algorithms.mahdrl.train import evaluate_hrl_three_layer_multi_seed
        from common.scenarios import Scenario
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
        kwargs=dict(dax_paths=Scenario().dax_paths(),random_seed=2,workflows_per_episode=2,
                    deadline_mode='none',max_ready_tasks=100,num_hosts=2,vms_per_host=(2,2))
        env=CloudWorkflowEnv_VMAgents(**kwargs);env.reset()
        h,ok=env.get_host_state_for_next_assignment();env.host_select(int(np.flatnonzero(h['mask'])[0]))
        v,ok=env.get_vm_state_for_current_task();env.reset()
        def agent(dim,actions):
            return D3QNAgent(dim,actions,hidden_dims=(16,8),batch_size=4,buffer_size=32,device=torch.device('cpu'))
        ha,va=agent(len(h['obs']),len(h['mask'])),agent(len(v['obs']),len(v['mask']))
        ta=agent(len(env.get_manager_state()),243)
        ht,vt=DQNDecisionTrace(ha,0),DQNDecisionTrace(va,0)
        phases=0
        while not env.done_flag:
            before=env.get_manager_state().copy();mask=env.get_manager_action_mask()
            action=ta.select_action(before,mask)
            self.assertGreater(mask[action],.5)
            env.apply_manager_delta(MANAGER_ACTION_TABLE[action]);self.invariant(env)
            while True:
                host,ok=env.get_host_state_for_next_assignment()
                if not ok:break
                ht.advance(host['obs'],host['mask']);ah=ha.select_action(host['obs'],host['mask'])
                env.host_select(ah);vm,ok=env.get_vm_state_for_current_task();self.assertTrue(ok)
                vt.advance(vm['obs'],vm['mask']);av=va.select_action(vm['obs'],vm['mask'])
                rh,rv,info=env.vm_assign(av);self.assertEqual(info['invalid'],0)
                ht.record(host['obs'],host['mask'],ah,rh,host_index=env.host_ids[ah])
                vt.record(vm['obs'],vm['mask'],av,rv,host_index=info['host_id'])
            reward,info=env.finish_phase_and_advance();finish_lower_phase(env,ht,vt,info)
            ta.remember(before,mask,action,reward,env.get_manager_state(),env.get_manager_action_mask(),float(env.done_flag))
            loss=ta.update()
            if loss is not None:self.assertTrue(np.isfinite(loss))
            phases+=1;self.assertLess(phases,1000)
        self.assertIsNone(ht.pending);self.assertIsNone(vt.pending)
        self.assertEqual(env.completed_workflows,2)
        self.assertTrue(all(learner._updates>0 for learner in (ta,ha,va)))
        metrics=evaluate_hrl_three_layer_multi_seed(CloudWorkflowEnv_VMAgents,kwargs,va,ha,ta,[2])
        self.assertTrue(np.isfinite(metrics).all())
        self.assertGreater(metrics[-1],0)

    def test_versioned_outputs_and_shared_ablation_paths(self):
        from common.scenarios import Scenario
        for algorithm in TA_ALGORITHMS:
            self.assertIn(TA_WEIGHT_MODEL,Scenario().model_name(algorithm))
            for mode in ('train','evaluate'):
                module=importlib.import_module(f'algorithms.{algorithm}.{mode}')
                classes=[v for v in vars(module).values() if isinstance(v,type) and issubclass(v,Base)]
                self.assertEqual(len(classes),1)
                self.assertIs(classes[0].apply_manager_delta,Base.apply_manager_delta)
                self.assertIs(classes[0].get_manager_action_mask,Base.get_manager_action_mask)
        for algorithm in ('pd3qn','marl','irws'):
            self.assertNotIn(TA_WEIGHT_MODEL,Scenario().model_name(algorithm))
        # A stale raw field in an externally restored object never participates in decisions.
        env=self.weights();env.manager_raw_w=np.ones(5)*100
        ref=self.weights();delta=[.1,0,0,0,0]
        env.apply_manager_delta(delta);ref.apply_manager_delta(delta)
        np.testing.assert_array_equal(env.combo_weights,ref.combo_weights)


if __name__=='__main__':unittest.main()
