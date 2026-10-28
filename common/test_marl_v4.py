"""MARL physical execution, compact observations, paired SR-gated selection and isolation."""
from dataclasses import replace
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from algorithms.marl.q_agents import QAgent, QConfig, IndependentLocalAgents
from algorithms.marl.runtime import isolated_evaluation, selection_record, validate_pair
from algorithms.marl.scenario import MARLScenario
from algorithms.pd3qn_dag_fe.selection import validation_selection, selection_key, SELECTION_TAG
from common.experiment_measurements import episode_metrics
from common.read_xml_opt_Tsize import Task
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.workflow_opt import Workflow, energy_from_records
from environments.marl import CloudWorkflowEnv_MARL


def queue_fixture(**kwargs):
    env = CloudWorkflowEnv_MARL(dax_paths=Scenario().dax_paths(), num_hosts=2,
        vms_per_host=(1,2), pc_tiers=(1.,), bw_tiers=(1000.,), workflows_per_episode=1,
        deadline_mode='poisson_heftlike', dump_deadlines=False, print_deadlines=False, **kwargs)
    tasks = [Task(task_id=0, children=[2], workload_mi=2.),
             Task(task_id=1, workload_mi=3.), Task(task_id=2, parents=[0], workload_mi=1.)]
    graph = nx.DiGraph(); graph.add_nodes_from(range(3)); graph.add_edge(0,2)
    env.workflows = [Workflow(0, graph, tasks, 0., 0, deadline=20.)]
    env.episode_seed = 0
    env.arrival_times = [0.]; env.next_arrival_idx = 1
    env.task_meta = [(0,i) for i in range(3)]
    env.gid_of = {(0,i):i for i in range(3)}
    env.task_state = ['Ready','Ready','unReady']
    env.task_ready_time = [0.,0.,0.]; env.task_end_time = [0.,0.,0.]
    env.task_baseline_finish = [20.,20.,20.]
    env.ready_task_ids = [0,1]; env.parents_loc_by_wf = {0:{}}
    env.wf_remaining_tasks = {0:3}
    env._update_current_task_fcfs()
    return env


class ExecutionTests(unittest.TestCase):
    def test_idle_assignment_and_dependency_release_preserve_real_load(self):
        env = queue_fixture()
        _, _, _, done, first = env.step(0,0)
        self.assertFalse(done)
        self.assertEqual(env.current_time, 0.)
        self.assertEqual(env.get_local_state(0)[1].tolist(), [0.,0.])
        self.assertEqual(env.get_global_state()[1].tolist(), [0.,1.])
        self.assertEqual(first['task_end_time'], 2.)
        with self.assertRaisesRegex(ValueError, 'idle'):
            env._assign_task_to_vm(1,0)
        self.assertEqual(env._assign_task_to_vm(1,1), 3.)
        self.assertEqual(env.task_state, ['Running','Running','unReady'])
        self.assertEqual(env.vm_running_task[0], 0)
        self.assertEqual(env._host_running_pc[0], 1.)
        self.assertEqual(env.vm_available_at[0], 2.)
        env._advance_until_decision()
        self.assertEqual(env.current_time, 2.)
        self.assertEqual(env.task_state, ['Finished','Running','Ready'])
        self.assertEqual(env.vm_running_task[0], -1)
        self.assertEqual(env._host_running_pc[0], 0.)
        self.assertEqual(env._assign_task_to_vm(2,0), 3.)
        env._advance_until_decision()
        self.assertTrue(env.done_flag)
        self.assertEqual(env.current_time, 3.)
        self.assertEqual(env.task_state, ['Finished']*3)
        self.assertEqual(env._host_running_pc[0], 0.)
        self.assertEqual([(r.start_time,r.end_time) for r in env.records], [(0.,2.),(0.,3.),(2.,3.)])
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records,env.hosts))

    def test_step_accounts_lateness_and_does_not_replace_actions(self):
        env = queue_fixture(); env.task_baseline_finish[1] = 1.
        env.step(0,0)
        with self.assertRaises(ValueError): env.step(0,0)
        _, rg, rl, _, info = env.step(1,0)
        self.assertEqual(info['assigned_task_id'], 1)
        self.assertEqual(info['vm_id'], 1)
        self.assertEqual(info['task_end_time'], 3.)
        self.assertEqual(info['lateness'], 2.)
        self.assertAlmostEqual(rg, .75*(-2/300)+.25*info['r_energy_total'])
        self.assertAlmostEqual(rl, .75*(-2/300)+.25*info['r_energy_host'])
        fresh = queue_fixture()
        with self.assertRaises(ValueError): fresh.step(0,1)  # Padded VM remains illegal.
        self.assertEqual(fresh.ready_task_ids, [0,1])

    def test_truncation_does_not_finish_running_work_or_claim_success(self):
        env = queue_fixture(horizon=1.)
        env.step(0,0)
        _, _, _, done, info = env.step(1,0)
        self.assertTrue(done); self.assertTrue(info['truncated'])
        self.assertEqual(env.current_time, 1.)
        self.assertEqual(env.task_state[:2], ['Running','Running'])
        self.assertEqual(episode_metrics(env)['wf_success_count'], 0)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records,env.hosts,end_time=1.))

    def test_compact_observations_have_only_wait_exec_and_no_other_host_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'scale.json'; fixture_config(config)
            env = queue_fixture(normalization_mode='f19_shared_v1', normalization_config=str(config), source_train_scene='SS_T')
            with patch.object(env, '_task_duration', side_effect=AssertionError('Prediction in observation')):
                global_obs, _ = env.get_global_state()
            local, _, _ = env.get_local_state(0)
            self.assertEqual(global_obs.shape, (9,)); self.assertEqual(local.shape, (14,))
            np.testing.assert_allclose(local[:4], [2/400,0,0,20/123])
            np.testing.assert_allclose(local[4:9], [1/8,1000/8000,1,0,2/123])
            np.testing.assert_array_equal(local[9:], [0,0,0,0,0])
            self.assertAlmostEqual(global_obs[4], 2/3)
            env.ready_task_ids.append(2)
            env._host_running_pc[1] = 1.; env.vm_running_task[1] = 99
            np.testing.assert_array_equal(local, env.get_local_state(0)[0])
            env.workflows[0].deadline = -1.
            self.assertLess(env.get_local_state(0)[0][3], 0.)
        for resource, size in [('S',11),('M',17),('L',23)]:
            scene = MARLScenario(resource=resource)
            self.assertEqual(scene.describe()['global_obs_dim'],size)
            self.assertEqual(scene.describe()['local_obs_dim'],49)


class SelectionTests(unittest.TestCase):
    def test_training_selects_entire_feasible_pair_even_with_measurements_disabled(self):
        from algorithms.marl import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root/'scales.json'; fixture_config(config_path)
            class Scene(MARLScenario):
                def model_name(self, algorithm='marl'): return str(root/'logs')
                def checkpoint_dir(self, algorithm='marl'): return root/'ckpts'
                def log_dir(self, algorithm='marl'): return root/'logs'
            scene = Scene(mode='train', episodes=3, legacy_split=True, normalization_config=str(config_path))
            original_env = train.CloudWorkflowEnv_MARL
            def environment(**kwargs):
                kwargs.update(deadline_mode='poisson_heftlike',workflows_per_episode=1,
                    dump_deadlines=False,print_deadlines=False)
                return original_env(**kwargs)
            def config(**kwargs):
                return QConfig(**dict(kwargs,hidden_dims=(4,),device='cpu'))
            def validation(success, energy):
                return dict(rows=[dict(seed=1,wf_count=1,wf_success_count=success,
                    wf_success_rate=float(success),total_energy=energy)],
                    ret_g=0.,ret_l=0.,energy=energy,total_late=0.,avg_late=0.,wf_avg_late=0.)
            with patch.object(train,'SCENARIO',scene), patch.object(train,'QConfig',side_effect=config), \
                    patch.object(train,'CloudWorkflowEnv_MARL',side_effect=environment), \
                    patch.object(train,'_load_cache_seeds',return_value=[1,2,3]), \
                    patch.object(train,'evaluate_marl',side_effect=[validation(0,1.),validation(1,100.),validation(0,.01)]), \
                    patch.dict('os.environ',{'EXPERIMENT_METRICS':'0'}), redirect_stdout(io.StringIO()):
                train.train()
            best_g=torch.load(root/'ckpts'/'best_global.pth',weights_only=True)['training_selection']
            best_l=torch.load(root/'ckpts'/'best_local_actor_critics.pth',weights_only=True)['training_selection']
            self.assertEqual(best_g,best_l)
            self.assertEqual(best_g['episode'],1)
            self.assertTrue(best_g['validation_selection']['feasible'])
            self.assertEqual(best_g['validation_selection']['mean_total_energy'],100.)
            records=[json.loads(line) for line in (root/'logs'/'validation_selection.jsonl').read_text().splitlines()]
            self.assertEqual([row['selected_as_best'] for row in records],[True,True,False])
            summary=json.loads((root/'logs'/'training_summary.json').read_text())
            self.assertTrue(summary['feasible_model_found'])

    def test_shared_selector_boundaries_ranking_fallback_and_ties(self):
        def candidate(success, energy, count=100):
            return validation_selection([dict(wf_count=count,wf_success_count=success,
                wf_success_rate=success/count,total_energy=energy)])
        self.assertTrue(candidate(99,1)['feasible'])
        self.assertFalse(candidate(49,1,50)['feasible'])
        self.assertTrue(candidate(50,1,50)['feasible'])
        ordered = [candidate(100,10),candidate(99,10),candidate(100,11),candidate(98,1),candidate(97,.1)]
        self.assertEqual(sorted(ordered,key=selection_key),ordered)
        self.assertEqual(selection_key(ordered[0]),selection_key(candidate(100,10)))

    def test_effective_metadata_changes_only_marl_not_frozen_protocol(self):
        scene = MARLScenario()
        self.assertEqual(scene.selection_protocol()['selection'], SELECTION_TAG)
        self.assertEqual(scene.protocol().selection,'minimum_validation_total_energy')
        self.assertEqual(Scenario().protocol().selection,'minimum_validation_total_energy')
        self.assertIn(SELECTION_TAG,scene.output_version)
        self.assertEqual(scene.normalization_metadata()['experiment_protocol']['selection'],SELECTION_TAG)

    def test_paired_checkpoint_selection_rejects_mismatched_episode_and_metrics(self):
        scene = MARLScenario()
        cfg = QConfig(hidden_dims=())
        g = QAgent(2,2,cfg,peer_act_dim=2)
        locals = IndependentLocalAgents(2,2,2,cfg)
        rows = [dict(seed=600,wf_count=50,wf_success_count=50,wf_success_rate=1.,total_energy=10.)]
        candidate = validation_selection(rows)
        info = selection_record(scene,3,candidate,dict(candidate,episode=3),rows,'best')
        g.training_selection = locals.training_selection = info
        self.assertEqual(validate_pair(g,locals,scene),info)
        locals.training_selection = dict(info,episode=4)
        with self.assertRaisesRegex(ValueError,'pair'): validate_pair(g,locals,scene)
        locals.training_selection = g.training_selection = dict(info,validation_selection=dict(candidate,mean_total_energy=1.))
        with self.assertRaisesRegex(ValueError,'results'): validate_pair(g,locals,scene)

    def test_validation_restores_rng_modes_and_weights(self):
        cfg=QConfig(hidden_dims=(4,))
        g=QAgent(2,2,cfg); locals=IndependentLocalAgents(2,2,2,cfg)
        g.critic.eval()
        before=[p.detach().clone() for a in (g,*locals.agents) for m in (a.actor,a.critic) for p in m.parameters()]
        state=random.getstate(),np.random.get_state(),torch.get_rng_state()
        with isolated_evaluation(g,locals):
            random.random();np.random.rand();torch.rand(1)
            self.assertFalse(g.actor.training);self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(random.getstate(),state[0]);np.testing.assert_array_equal(np.random.get_state()[1],state[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(),state[2]))
        self.assertTrue(g.actor.training);self.assertFalse(g.critic.training)
        after=[p for a in (g,*locals.agents) for m in (a.actor,a.critic) for p in m.parameters()]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))


if __name__ == '__main__':
    unittest.main()
