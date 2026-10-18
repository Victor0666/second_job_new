"""F15 five-attribute semantics, Ready-local normalization and shared environment paths."""
import importlib
import itertools
from types import SimpleNamespace
import unittest
import numpy as np
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Env, MANAGER_ACTION_TABLE


class F15Tests(unittest.TestCase):
    def env(self, ready=(0, 1, 2), normalize=True):
        env = object.__new__(Env)
        env.task_heur_dim = 5
        env.current_time = 100.
        env.normalize = normalize
        env.ready_task_ids = list(ready)
        env.task_state = ['Ready'] * 4
        env.task_state[3] = 'unReady'
        env.vms = {0: SimpleNamespace(pc=2., bw=1.), 1: SimpleNamespace(pc=6., bw=3.)}
        env.task_ready_time = [90., 70., 80., -999.]
        env.task_mi = [40., 80., 120., 1e6]
        env.task_in_bits = [2e6, 4e6, 6e6, 1e15]
        env.task_out_bits = [2e6, 4e6, 6e6, 1e15]
        env.task_up_rank = [30., 90., 60., 1e10]
        env.task_baseline_finish = [132., 114., 136., -1e8]
        env.task_children = [[1, 2, 3], [], [3], []]
        env.combo_weights = np.ones(5) / 5
        env._add_workflow_if_arrived = lambda: None
        return env

    def order(self, env, column):
        env.combo_weights = np.eye(5)[column]
        env._phase_prepare_tasks()
        return env._phase_tasks

    def test_T01_fcfs_ready_time(self):
        for normalize in (True, False):
            self.assertEqual(self.order(self.env(normalize=normalize), 0), [1, 2, 0])

    def test_T02_sjf_processing_seconds(self):
        for normalize in (True, False):
            self.assertEqual(self.order(self.env(normalize=normalize), 1), [0, 1, 2])
        _, values = self.env(normalize=False)._compute_task_heuristics_for_ready()
        np.testing.assert_allclose(values[:, 1], [-12., -24., -36.])

    def test_T03_mcf_rank_not_child_count(self):
        for normalize in (True, False):
            self.assertEqual(self.order(self.env(normalize=normalize), 2), [1, 2, 0])

    def test_T04_hur_signed_slack(self):
        env = self.env(normalize=False)
        _, values = env._compute_task_heuristics_for_ready()
        np.testing.assert_allclose(-values[:, 3], [20., -10., 0.])
        for normalize in (True, False):
            self.assertEqual(self.order(self.env(normalize=normalize), 3), [1, 2, 0])

    def test_T05_edf_is_deadline_not_slack(self):
        env = self.env()
        self.assertEqual(self.order(env, 4), [1, 0, 2])
        self.assertEqual(self.order(env, 3), [1, 2, 0])

    def test_T06_independent_ready_minmax(self):
        env = self.env(ready=(2, 0, 3, 1))
        ids, values = env._compute_task_heuristics_for_ready()
        self.assertEqual(ids, [2, 0, 1])
        expected = np.array([[.5, 0., .5, 2/3, 0.], [0., 1., 0., 0., 2/11], [1., .5, 1., 1., 1.]])
        np.testing.assert_allclose(values, expected, atol=1e-14)
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_T07_equal_values_and_invalid_inputs(self):
        env = self.env()
        for name in ('task_ready_time', 'task_mi', 'task_in_bits', 'task_out_bits', 'task_up_rank', 'task_baseline_finish'):
            setattr(env, name, [10.] * 4)
        _, values = env._compute_task_heuristics_for_ready()
        np.testing.assert_array_equal(values, np.zeros((3, 5)))
        env.task_up_rank[1] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            env._compute_task_heuristics_for_ready()
        env = self.env(); env.task_baseline_finish = []
        with self.assertRaisesRegex(ValueError, 'sub-deadline'):
            env._compute_task_heuristics_for_ready()
        for normalize in (True, False):
            extreme = self.env(normalize=normalize)
            extreme.task_up_rank = [1e300, -1e300, 0., 0.]
            self.assertTrue(np.isfinite(extreme._compute_task_heuristics_for_ready()[1]).all())
        env = self.env(); env.vms[0].pc = float('inf')
        with self.assertRaisesRegex(ValueError, 'capacity'):
            env._compute_task_heuristics_for_ready()

    def test_T08_empty_and_singleton(self):
        env = self.env(ready=())
        self.assertEqual(env._compute_task_heuristics_for_ready()[1].shape, (0, 5))
        env._phase_prepare_tasks(); self.assertEqual(env._phase_tasks, [])
        env = self.env(ready=(2,))
        np.testing.assert_array_equal(env._compute_task_heuristics_for_ready()[1], np.zeros((1, 5)))
        env._phase_prepare_tasks(); self.assertEqual(env._phase_tasks, [2])

    def test_T09_five_weight_positions(self):
        expected = [[1, 2, 0], [0, 1, 2], [1, 2, 0], [1, 2, 0], [1, 0, 2]]
        for col, order in enumerate(expected):
            self.assertEqual(self.order(self.env(), col), order)

    def test_T10_action_table_mask_and_weight_updates_unchanged(self):
        expected = np.array(list(itertools.product((-.1, 0., .1), repeat=5)), dtype=np.float32)
        np.testing.assert_array_equal(MANAGER_ACTION_TABLE, expected)
        self.assertEqual(MANAGER_ACTION_TABLE.shape, (243, 5))
        env = self.env(); env.manager_raw_w = np.array([0., .2, 0., .3, .5], dtype=np.float32)
        mask = env.get_manager_action_mask()
        np.testing.assert_array_equal(mask, (~((expected < 0) & (env.manager_raw_w[None, :] == 0)).any(axis=1)).astype(np.float32))
        env.apply_manager_delta(np.array([.1, -.1, .1, 0., -.1]))
        self.assertTrue((env.combo_weights >= 0).all())
        self.assertAlmostEqual(float(env.combo_weights.sum()), 1., places=6)
        self.assertEqual(env.combo_weights.shape, (5,))

    def test_T11_hand_computed_weighted_order(self):
        env = self.env()
        env.combo_weights = np.array([.1, .2, .3, .15, .25])
        _, values = env._compute_task_heuristics_for_ready()
        scores = values @ env.combo_weights
        np.testing.assert_allclose(scores, [.2 + .25*2/11, .1 + .1 + .3 + .15 + .25, .05 + .15 + .1])
        env._phase_prepare_tasks(); self.assertEqual(env._phase_tasks, [1, 2, 0])

    def test_T12_ties_ready_time_then_task_id_and_repeatability(self):
        env = self.env(ready=(2, 1, 0))
        env.task_up_rank = [10.] * 4
        env.task_ready_time = [80., 70., 70., 0.]
        for _ in range(5):self.assertEqual(self.order(env, 2), [1, 2, 0])

    def test_T13_real_dag_episode_ready_deadlines_and_rank(self):
        from common.scenarios import Scenario
        from common.workflow_opt import task_subdeadlines, energy_from_records
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents
        env = CloudWorkflowEnv_VMAgents(dax_paths=Scenario().dax_paths(), random_seed=2,
                                       workflows_per_episode=1, deadline_mode='none', num_hosts=1,
                                       vms_per_host=(2,), max_ready_tasks=100)
        env.reset()
        initial_ready = {gid: env.task_ready_time[gid] for gid in env.ready_task_ids}
        initial_now = env.current_time
        for gid, ready in initial_ready.items():self.assertEqual(ready, initial_now)
        previous_state = list(env.task_state)
        assigned = 0
        while not env.done_flag:
            env.apply_manager_delta(np.zeros(5))
            while True:
                host, ok = env.get_host_state_for_next_assignment()
                if not ok:break
                gid = env._cur_tid
                self.assertEqual(env.task_state[gid], 'Ready')
                for prior in initial_ready:
                    self.assertEqual(env.task_ready_time[prior], initial_ready[prior])
                env.host_select(int(np.flatnonzero(host['mask'] > .5)[0]))
                vm, ok = env.get_vm_state_for_current_task(); self.assertTrue(ok)
                _, _, info = env.vm_assign(int(np.flatnonzero(vm['mask'] > .5)[0])); assigned += 1
                self.assertEqual(info['invalid'], 0)
                self.assertEqual(info['deadline'], env.task_baseline_finish[gid])
                wf_id, local = env.task_meta[gid]
                task = env.workflows[wf_id].tasks[local]
                self.assertEqual(env.task_up_rank[gid], task.upward_rank)
                for child, state in enumerate(env.task_state):
                    if previous_state[child] == 'unReady' and state == 'Ready':
                        self.assertEqual(env.task_ready_time[child], env.current_time)
                previous_state = list(env.task_state)
            env.finish_phase_and_advance()
        self.assertEqual(assigned, len(env.task_state))
        self.assertTrue(all(state == 'Finished' for state in env.task_state))
        wf = env.workflows[0]
        expected = task_subdeadlines(wf.tasks, env._heft_vm_pc, env._heft_vm_bw_bps, wf.arrival_time, wf.deadline)
        np.testing.assert_allclose(env.task_baseline_finish, expected)
        self.assertAlmostEqual(env.total_energy, energy_from_records(env.records if hasattr(env,'records') else env._records,
                               env.hosts, start_time=env.first_arrival_time, end_time=env.current_time))

    def test_T14_all_train_eval_and_ablation_paths_share_definition(self):
        for algorithm in ('mahdrl', 'ablation_hostrand', 'ablation_mgrrand', 'ablation_vmrand', 'ablation_nohost'):
            for mode in ('train', 'evaluate'):
                module = importlib.import_module(f'algorithms.{algorithm}.{mode}')
                classes = [value for value in vars(module).values() if isinstance(value, type) and issubclass(value, Env)]
                self.assertEqual(len(classes), 1)
                cls = classes[0]
                self.assertIs(cls._compute_task_heuristics_for_ready, Env._compute_task_heuristics_for_ready)
                self.assertIs(cls._phase_prepare_tasks, Env._phase_prepare_tasks)
        from environments.base_heft import CloudWorkflowEnv_VMAgents as Base
        self.assertIsNot(Base._compute_task_heuristics_for_ready, Env._compute_task_heuristics_for_ready)


if __name__ == '__main__':unittest.main()
