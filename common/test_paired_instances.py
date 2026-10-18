"""Paired alpha and workflow constraints across every experiment algorithm."""

from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.scenarios import Scenario
from common.workflow_opt import sample_deadline_alpha
from common.read_xml_opt_Tsize import load_workflow_from_dax, workflow_instance_hash


ALGORITHMS = {
    'mahdrl': ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
    'ablation_hostrand': ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
    'ablation_mgrrand': ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
    'ablation_vmrand': ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
    'ablation_nohost': ('ablation_nohost', 'CloudWorkflowEnv_VMAgents'),
    'ertf_fifs': ('mahdrl_fcfs', 'CloudWorkflowEnv_VMAgents'),
    'pd3qn': ('pd3qn', 'CloudWorkflowEnv_PD3QN_Cost'),
    'irws': ('irws', 'IRWSEnvRouteA'),
    'marl': ('marl', 'CloudWorkflowEnv_MARL'),
}


class PairedInstanceTests(unittest.TestCase):
    def sampler(self, algorithm, seed, probability):
        module_name, cls_name = ALGORITHMS[algorithm]
        cls = getattr(importlib.import_module('environments.' + module_name), cls_name)
        env = object.__new__(cls)
        env.random_seed = env.episode_seed = seed
        env.rng = np.random.RandomState(seed)
        env.deadline_alpha = env.alpha_deadline = 1.5
        env.deadline_alpha_choices = env.alpha_deadline_choices = (2.0, 3.0)
        env.deadline_alpha_probs = env.alpha_deadline_probs = (probability, 1 - probability)
        env.deadline_alpha_small = 2.0
        env.deadline_alpha_large = 3.0
        env.deadline_alpha_small_prob = probability
        method = env._sample_deadline_alpha_for_workflow if hasattr(env, '_sample_deadline_alpha_for_workflow') else env._sample_deadline_alpha
        return env, method

    def test_all_samplers_preserve_mahdrl_instances_and_do_not_consume_rng(self):
        for seed in (1, 2, 7, 31):
            for probability in (0.8, 0.5, 0.2):
                expected = [2.0 if np.random.RandomState(
                    (seed * 1000003 + wf * 9176 + 20260316) & 0xFFFFFFFF
                ).rand() < probability else 3.0 for wf in range(50)]
                for algorithm in ALGORITHMS:
                    with self.subTest(algorithm=algorithm, seed=seed, probability=probability):
                        env, draw = self.sampler(algorithm, seed, probability)
                        before = env.rng.get_state()
                        self.assertEqual([draw(wf) for wf in range(50)], expected)
                        self.assertEqual([draw(wf) for wf in reversed(range(50))], expected[::-1])
                        self.assertEqual(draw(17), draw(17))
                        after = env.rng.get_state()
                        self.assertEqual(before[0], after[0])
                        np.testing.assert_array_equal(before[1], after[1])
                        self.assertEqual(before[2:], after[2:])

    def test_marl_uses_active_episode_seed_and_distribution_order_is_canonical(self):
        env, draw = self.sampler('marl', 7, 0.8)
        env.random_seed = 999
        for wf in range(50):
            expected = sample_deadline_alpha(7, wf, [2, 3], [0.8, 0.2])
            self.assertEqual(draw(wf), expected)
            self.assertEqual(sample_deadline_alpha(7, wf, [3, 2], [0.2, 0.8]), expected)
        for probabilities in ([0, 0], [-1, 2], [float('nan'), 1], [float('inf'), 1]):
            with self.assertRaises(ValueError):
                sample_deadline_alpha(2, 0, [2, 3], probabilities)

    def test_pd3qn_rejects_a_different_alpha_seed_offset(self):
        from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost
        with self.assertRaisesRegex(ValueError, 'shared value'):
            CloudWorkflowEnv_PD3QN_Cost(dax_paths=['synthetic.xml'], deadline_mode='none',
                                       deadline_alpha_seed_offset=20260317)

    def snapshot(self, algorithm, scenario, cache):
        module_name, cls_name = ALGORITHMS[algorithm]
        cls = getattr(importlib.import_module('environments.' + module_name), cls_name)
        kwargs = dict(dax_paths=scenario.dax_paths(), random_seed=2, workflows_per_episode=50,
                      horizon=1e9, arrival_lambda=0.03, max_ready_tasks=100,
                      deadline_mode='cache_fcfs', deadline_cache_path=str(cache))
        if algorithm == 'marl':
            kwargs.update(alpha_deadline_choices=(2.0, 3.0), alpha_deadline_probs=scenario.deadline_probs,
                          print_deadlines=False, dump_deadlines=False)
        elif algorithm in ('pd3qn', 'irws'):
            kwargs.update(deadline_alpha_choices=(2.0, 3.0), deadline_alpha_probs=scenario.deadline_probs)
        else:
            kwargs['deadline_alpha_small_prob'] = scenario.small_probability
        method_name = '_add_workflows_up_to_current_time' if algorithm == 'irws' else '_add_workflow_if_arrived'
        loader_module = getattr(cls, method_name).__module__
        payload_seeds = []

        def record_load(*args, **kwargs):
            payload_seeds.append(kwargs['seed'])
            return load_workflow_from_dax(*args, **kwargs)

        with patch(loader_module + '.load_workflow_from_dax', side_effect=record_load):
            env = cls(**kwargs)
            payload_seeds.clear()
            env.reset()
            env.current_time = float(env.arrival_times[-1])
            getattr(env, method_name)()
        names, alphas, payloads = [], [], []
        for wf in env.workflows:
            wf_id = wf.workflow_id
            names.append(env.wf_dax_name[wf_id] if algorithm == 'marl' else Path(wf.dax_path).name)
            if algorithm == 'marl':
                alphas.append(env.wf_deadline_alpha[wf_id])
            elif algorithm in ('pd3qn', 'irws'):
                alphas.append(wf.deadline_alpha_used)
            else:
                alphas.append(wf.deadline_alpha)
            payloads.append(tuple((t.task_id, tuple(t.parents), tuple(t.children), t.workload_mi,
                                   t.ext_in_bits, t.from_parents_bits, t.out_file_size_sum_bits,
                                   tuple(sorted(t.parent_in_bits.items()))) for t in wf.tasks))
        self.assertEqual(len(env.workflows), 50)
        self.assertEqual(payload_seeds, list(range(2, 52)))
        return dict(names=names, alphas=alphas, payloads=payloads, payload_seeds=payload_seeds,
                    hashes=[workflow_instance_hash(w.graph, w.tasks) for w in env.workflows],
                    arrivals=list(env.arrival_times), deadlines=[wf.deadline for wf in env.workflows],
                    task_deadlines=env.task_baseline_finish)

    def test_actual_fifty_workflows_are_identical_for_all_algorithms_and_distributions(self):
        from common.precompute_deadlines import main
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'ddl_fcfs_taskS_resS_seed2.json'
            with redirect_stdout(io.StringIO()):
                main(Scenario(mode='precompute', seed_spec='2', cache_output=str(cache)))
            for deadline in ('Tight', 'Medium', 'Loose'):
                scenario = Scenario(deadline=deadline)
                reference = self.snapshot('mahdrl', scenario, cache)
                for algorithm in ALGORITHMS:
                    with self.subTest(algorithm=algorithm, deadline=deadline):
                        actual = self.snapshot(algorithm, scenario, cache)
                        for key in ('names', 'alphas', 'payload_seeds', 'payloads', 'hashes'):
                            self.assertEqual(actual[key], reference[key], key)
                        for key in ('arrivals', 'deadlines', 'task_deadlines'):
                            np.testing.assert_allclose(actual[key], reference[key], rtol=0, atol=1e-10, err_msg=key)
                print(f'SS seed=2 {deadline}: all 9 algorithms match on 50 complete workflow instances')


if __name__ == '__main__':
    unittest.main()
