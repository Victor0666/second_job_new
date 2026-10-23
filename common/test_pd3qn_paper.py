"""Paper PD3QN: numerical targets, training clocks and legacy compatibility."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import torch
from torch import nn

from common.d3qn_agent_PER import D3QNAgent
from common.f18_state_normalization import PD3QN_UNITS, load_normalizer


class TableQ(nn.Module):
    """A two-state Q table makes the expected Bellman target unambiguous."""

    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values[states[:, 0].long()]


class PD3QNPaperTests(unittest.TestCase):
    def make_agent(self, **overrides):
        options = dict(input_dim=1, output_dim=2, hidden_dims=(4,), paper_mode=True,
                       batch_size=2, buffer_size=10, gamma=0.5, lr=0.01,
                       eps_start=1.0, eps_end=0.0, eps_decay_steps=100,
                       target_update_tau=0.0, target_update_freq=3)
        options.update(overrides)
        return D3QNAgent(**options)

    def table_agent(self, **overrides):
        agent = self.make_agent(**overrides)
        agent.online = TableQ([[0.0, 0.0], [5.0, 2.0]])
        agent.target = TableQ([[0.0, 0.0], [1.0, 10.0]])
        agent.optim = torch.optim.SGD(agent.online.parameters(), lr=0.01)
        return agent

    @staticmethod
    def remember(agent, reward, done=True):
        agent.remember(np.array([0.0]), np.ones(2), 0, reward,
                       np.array([1.0]), np.zeros(2) if done else np.ones(2),
                       float(done))

    def test_new_priorities_and_training_share_eq29_target(self):
        agent = self.table_agent()
        self.remember(agent, 3.0, done=False)
        # Eq.(29) selects target's action 1 (10), not online's action 0 (1).
        self.assertAlmostEqual(agent.priorities[0], 8.0 + 1e-6, places=5)
        self.remember(agent, 1.0)
        self.assertAlmostEqual(agent.priorities[1], 1.0 + 1e-6, places=5)
        with patch("common.d3qn_agent_PER.np.random.choice", return_value=np.array([0, 1])):
            loss = agent.update()
        self.assertAlmostEqual(loss, (8.0 ** 2 + 1.0) / 2, places=5)
        np.testing.assert_allclose(agent.priorities, [8.0 + 1e-6, 1.0 + 1e-6])

    def test_eq29_differs_from_legacy_double_but_execution_uses_online(self):
        for paper, expected in ((True, 7.5), (False, -1.05)):
            agent = self.table_agent(paper_mode=paper, gamma=0.95)
            target, _ = agent._td_target(torch.tensor([-2.0]), torch.tensor([[1.0]]),
                                        torch.ones((1, 2)), torch.zeros(1))
            self.assertAlmostEqual(target.item(), expected, places=5)
            self.assertEqual(agent.select_action(np.array([1.0]), np.ones(2),
                                                 deterministic=True, count_step=False), 0)

    def test_eq29_masks_illegal_actions_and_zeroes_terminal_future(self):
        agent = self.table_agent()
        target, future = agent._td_target(torch.tensor([3.0, 3.0, 3.0]),
            torch.ones((3, 1)), torch.tensor([[1., 0.], [0., 0.], [1., 1.]]),
            torch.tensor([0., 1., 1.]))
        torch.testing.assert_close(target, torch.tensor([3.5, 3.0, 3.0]))
        torch.testing.assert_close(future, torch.tensor([1., 0., 0.]))
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(target.requires_grad)
        # A finite masking sentinel must not outrank a very negative legal value.
        agent.target = TableQ([[0., 0.], [-2e10, 10.]])
        _, future = agent._td_target(torch.zeros(1), torch.ones((1, 1)),
                                     torch.tensor([[1., 0.]]), torch.zeros(1))
        self.assertEqual(future.item(), -2e10)

    def test_proportional_replay_unweighted_mse_and_no_clipping(self):
        agent = self.table_agent(per_alpha=1.0, per_beta_start=1.0, per_beta_end=1.0,
                                 grad_clip=10.0)
        self.remember(agent, 2.0)
        self.remember(agent, 4.0)
        agent.priorities[:] = [1.0, 9.0]
        with patch("common.d3qn_agent_PER.np.random.choice", return_value=np.array([0, 1])) as sample, \
             patch("torch.nn.utils.clip_grad_norm_", side_effect=AssertionError("extra clipping")):
            loss = agent.update()
        np.testing.assert_allclose(sample.call_args.kwargs["p"], [0.1, 0.9], atol=1e-6)
        self.assertAlmostEqual(loss, 10.0, places=5)
        self.assertEqual(agent.per_beta_count, 0)

    def test_epsilon_advances_on_training_decisions_only(self):
        agent = self.table_agent()
        state, mask = np.array([0.0]), np.ones(2)
        agent.select_action(state, mask)
        self.assertAlmostEqual(agent.epsilon(), 0.99)
        agent.select_action(state, mask, deterministic=True)
        agent.select_action(state, mask, deterministic=False, count_step=False)
        self.assertAlmostEqual(agent.epsilon(), 0.99)
        self.remember(agent, 1.0)
        self.remember(agent, 2.0)
        agent.update()
        self.assertAlmostEqual(agent.epsilon(), 0.99)
        agent.begin_episode()
        self.assertAlmostEqual(agent.epsilon(), 0.99)

    def test_hard_copy_uses_episode_decision_boundary(self):
        agent = self.table_agent()
        self.remember(agent, 1.0)
        self.remember(agent, 2.0)
        state, mask = np.array([0.0]), np.ones(2)
        agent.begin_episode()
        agent.select_action(state, mask)
        agent.update()
        torch.testing.assert_close(agent.target.values, agent.online.values)
        before = agent.target.values.detach().clone()
        for _ in range(2):
            agent.select_action(state, mask)
            agent.update()
        torch.testing.assert_close(agent.target.values, before)
        agent.select_action(state, mask)
        agent.update()
        torch.testing.assert_close(agent.target.values, agent.online.values)
        agent.begin_episode()
        agent.select_action(state, mask)
        agent.update()
        torch.testing.assert_close(agent.target.values, agent.online.values)

    def test_fixed_units_preserve_sign_and_values_above_reference(self):
        from common.f18_state_normalization import check_checkpoint

        normalizer = load_normalizer(PD3QN_UNITS)
        self.assertEqual(normalizer.scale(-2500, "T_ref"), -2.5)
        self.assertEqual(normalizer.scale(300, "workload_mi_ref"), 3.0)
        self.assertEqual(normalizer.scale(16 * 1024 ** 3, "input_bits_ref"), 2.0)
        self.assertEqual(normalizer.scale(8, "vm_pc_ref"), 8.0)
        self.assertEqual(normalizer.scale(8000, "vm_bw_ref"), 8.0)
        self.assertEqual(normalizer.rr(12), 12)
        old_metadata = dict(normalizer.metadata)
        old_metadata.pop("observation_layout")
        with self.assertRaisesRegex(ValueError, "observation_layout"):
            check_checkpoint(old_metadata, normalizer.metadata)
        old_metadata['observation_layout'] = 'task6_vm4_parentdata_v2'
        with self.assertRaisesRegex(ValueError, "observation_layout"):
            check_checkpoint(old_metadata, normalizer.metadata)

    def test_paper_and_legacy_checkpoint_roundtrips(self):
        state, mask = np.array([0.0]), np.ones(2)
        with tempfile.TemporaryDirectory() as directory:
            for paper in (True, False):
                original = self.make_agent(paper_mode=paper)
                original.select_action(state, mask)
                self.remember(original, 1.0)
                self.remember(original, 2.0)
                original.update()
                path = str(Path(directory) / ("paper.pt" if paper else "legacy.pt"))
                original.save(path)
                restored = self.make_agent(paper_mode=paper)
                restored.load(path)
                self.assertEqual(restored.paper_mode, paper)
                self.assertEqual(restored._updates, original._updates)
                self.assertEqual(restored._eps_steps, original._eps_steps)
                for key, value in original.online.state_dict().items():
                    torch.testing.assert_close(restored.online.state_dict()[key], value)
                incompatible = self.make_agent(paper_mode=not paper)
                before = {key: value.clone() for key, value in incompatible.online.state_dict().items()}
                with self.assertRaises(ValueError):
                    incompatible.load(path)
                for key, value in before.items():
                    torch.testing.assert_close(incompatible.online.state_dict()[key], value)

    def test_unversioned_legacy_checkpoint_still_loads(self):
        original = self.make_agent(paper_mode=False)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "old.pt")
            original.save(path)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint.pop("pd3qn_implementation")
            checkpoint.pop("network")
            torch.save(checkpoint, path)
            restored = self.make_agent(paper_mode=False)
            restored.load(path)
            for key, value in original.online.state_dict().items():
                torch.testing.assert_close(restored.online.state_dict()[key], value)
            with self.assertRaises(ValueError):
                self.make_agent().load(path)

    def test_previous_pd3qn_checkpoints_are_rejected_before_mutating_network(self):
        agent = self.make_agent()
        before = copy.deepcopy(agent.online.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'v4.pt')
            agent.save(path)
            checkpoint = torch.load(path, weights_only=True)
            for previous in ('paper_minimal_v4', 'paper_minimal_v5'):
                checkpoint['pd3qn_implementation'] = previous
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'implementation mismatch'):
                    agent.load(path)
        for key, value in before.items():
            torch.testing.assert_close(value, agent.online.state_dict()[key])

    def test_train_eval_defaults_share_paper_settings_and_batch_only_warmup(self):
        from algorithms.pd3qn import train, evaluate
        from algorithms.pd3qn.config import agent_kwargs

        for entrypoint in (train, evaluate):
            self.assertEqual(entrypoint.SCENARIO.normalization_mode, PD3QN_UNITS)
        settings = agent_kwargs(train.SCENARIO)
        self.assertEqual(settings, agent_kwargs(evaluate.SCENARIO))
        self.assertTrue(settings["paper_mode"])
        self.assertEqual(settings["hidden_dims"], (1024, 1024, 512, 512, 256))
        self.assertEqual((settings["lr"], settings["gamma"], settings["batch_size"],
                          settings["buffer_size"], settings["target_update_freq"]),
                         (0.01, 0.95, 128, 10000, 60))
        agent = self.make_agent(batch_size=2)
        self.remember(agent, 1.0)
        self.assertFalse(train._warmup_ready(agent))
        self.remember(agent, 2.0)
        self.assertTrue(train._warmup_ready(agent))
        legacy = replace(train.SCENARIO, normalization_mode="legacy")
        self.assertFalse(agent_kwargs(legacy)["paper_mode"])
        self.assertEqual(agent_kwargs(legacy)["lr"], 0.0003)

    def test_cli_defaults_and_model_paths_are_isolated_without_calibration(self):
        from common.experiment_protocol import protocol_hash
        from common.f11 import LEARNING_MODEL
        from common.scenarios import Scenario, parse_scenario
        from common.workflow_opt import ENERGY_ACCOUNTING

        for mode, algorithm in (("train", "pd3qn"), ("eval", "pd3qn"), ("train", "mahdrl")):
            output = io.StringIO()
            with patch("sys.argv", [algorithm, "--dry-run"]), redirect_stdout(output), \
                 patch("common.f22_cache.check_cache", side_effect=AssertionError("cache accessed")), \
                 self.assertRaises(SystemExit) as stop:
                parse_scenario(mode, algorithm)
            self.assertEqual(stop.exception.code, 0)
            result = json.loads(output.getvalue())
            normalization = result["normalization"]
            version = normalization.get("normalization_version", normalization.get("normalization_mode"))
            self.assertEqual(version, PD3QN_UNITS if algorithm == "pd3qn" else "f19_shared_v1")
            if algorithm == "pd3qn":
                self.assertEqual(result["checkpoint_selection"], "minimum_validation_total_energy")

        paper = Scenario(mode="train", normalization_mode=PD3QN_UNITS,
                         normalization_config="missing-calibration-file.json")
        with patch("common.f18_state_normalization.validate_calibration",
                   side_effect=AssertionError("calibration accessed")), \
             patch.object(Scenario, "reproducibility_metadata", return_value={}):
            self.assertEqual(paper.normalization_metadata()["normalization_version"], PD3QN_UNITS)
            self.assertEqual(paper.normalization_metadata()["observation_layout"],
                             "task6_vm4_parentdata_host2_v3")
        self.assertIn("pd3qn_dag_", paper.model_name("pd3qn"))
        self.assertIn("task6_vm4_parentdata_host2_v3", paper.output_version)
        self.assertIn("pd3qn_dag_v1", paper.output_version)
        self.assertNotIn("paper_minimal_v4", paper.output_version)
        v2_output = (PD3QN_UNITS + "_task6_vm4_parentdata_v2_protocol_"
                     + protocol_hash(paper.experiment_protocol)[:12] + "_" + ENERGY_ACCOUNTING)
        self.assertNotEqual(paper.output_version, v2_output)
        old_output = (PD3QN_UNITS + "_protocol_" + protocol_hash(paper.experiment_protocol)[:12]
                      + "_" + ENERGY_ACCOUNTING)
        self.assertNotEqual(paper.output_version, old_output)
        self.assertEqual(paper.checkpoint_dir("pd3qn"), replace(paper, mode="eval").checkpoint_dir("pd3qn"))
        self.assertNotEqual(paper.checkpoint_dir("pd3qn"),
                            replace(paper, model_seed=2).checkpoint_dir("pd3qn"))
        legacy = replace(paper, normalization_mode="legacy", legacy_split=True)
        self.assertNotEqual(paper.checkpoint_dir("pd3qn"), legacy.checkpoint_dir("pd3qn"))
        self.assertEqual(legacy.model_name("pd3qn"),
                         "pd3qn_taskS_resS_warmup_mix_nr_Tight_" + LEARNING_MODEL + "_" + ENERGY_ACCOUNTING)
        self.assertEqual(legacy.output_version, "legacy_Legacy_" + ENERGY_ACCOUNTING)

    def test_fixed_units_change_observations_only_not_physics(self):
        from common.read_xml_opt_Tsize import Task
        from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost

        graph = nx.DiGraph([(0, 2), (1, 2)])
        graph.add_node(3)
        tasks = [Task(task_id=i, workload_mi=work,
                      parents=list(graph.predecessors(i)), children=list(graph.successors(i)))
                 for i, work in enumerate((10.0, 20.0, 5.0, 30.0))]
        with patch("environments.pd3qn.poisson_arrival_times", return_value=[0.0, 40.0]), \
             patch("environments.pd3qn.load_workflow_from_dax",
                   side_effect=lambda *args, **kwargs: (graph.copy(), copy.deepcopy(tasks))):
            options = dict(dax_paths=["synthetic.xml"], workflows_per_episode=2,
                           deadline_mode="none", random_seed=7, num_hosts=2,
                           vms_per_host=(2, 1), pc_tiers=(1.0,), bw_tiers=(1000.0,))
            legacy = CloudWorkflowEnv_PD3QN_Cost(**options, normalization_mode="legacy")
            fixed = CloudWorkflowEnv_PD3QN_Cost(**options, normalization_mode=PD3QN_UNITS)
            legacy_obs, _ = legacy.reset()
            fixed_obs, _ = fixed.reset()
            self.assertEqual(fixed_obs.size, legacy_obs.size + fixed.num_vms + 2 * fixed.num_hosts)
            self.assertFalse(np.allclose(legacy_obs, fixed_obs[:legacy_obs.size]))
            steps = 0
            while not legacy.done_flag:
                np.testing.assert_array_equal(legacy.get_action_mask(), fixed.get_action_mask())
                action = int(np.flatnonzero(legacy.get_action_mask() > 0.5)[0])
                legacy_result, fixed_result = legacy.step(action), fixed.step(action)
                self.assertEqual(legacy_result[1:4], fixed_result[1:4])
                self.assertEqual(legacy.current_time, fixed.current_time)
                self.assertEqual(legacy.total_energy, fixed.total_energy)
                np.testing.assert_array_equal(legacy.task_end_time, fixed.task_end_time)
                steps += 1
                self.assertLessEqual(steps, 8)
            self.assertTrue(fixed.done_flag)
            self.assertEqual(steps, 8)
            self.assertEqual(legacy.completed_workflows, 2)
            self.assertEqual(fixed.completed_workflows, 2)

    def test_evaluation_loads_best_checkpoint_in_paper_and_legacy_modes(self):
        from algorithms.pd3qn import evaluate

        for mode, filename in ((PD3QN_UNITS, "best_pd3qn_dag.pth"), ("legacy", "best_pd3qn.pth")):
            scenario = replace(evaluate.SCENARIO, normalization_mode=mode)
            agent, environment = MagicMock(), MagicMock()
            agent.paper_mode = mode == PD3QN_UNITS
            environment.reset.return_value = (np.zeros(1), {})
            environment.action_space.n = 2
            with patch.object(evaluate, "SCENARIO", scenario), \
                 patch.object(evaluate, "CloudWorkflowEnv_PD3QN_Cost", return_value=environment), \
                 patch.object(evaluate, "D3QNAgent", return_value=agent), \
                 patch.object(evaluate, "bind_normalization"), \
                 patch.object(evaluate.os.path, "exists", return_value=True), \
                 redirect_stdout(io.StringIO()):
                evaluate.build_agent_and_env_kwargs()
            self.assertEqual(Path(agent.load.call_args.args[0]).name, filename)
            self.assertTrue(agent.load.call_args.kwargs["require_learning_model"])

    def test_dag_cli_identity_and_historical_mode_rejection(self):
        from common.scenarios import parse_scenario

        for mode, entry in (('train', 'pd3qn_dag'), ('eval', 'pd3qn_dag'), ('train', 'pd3qn_v5')):
            output = io.StringIO()
            with patch('sys.argv', [entry, '--dry-run']), redirect_stdout(output), \
                 self.assertRaises(SystemExit) as stopped:
                parse_scenario(mode, entry)
            self.assertEqual(stopped.exception.code, 0)
            configuration = json.loads(output.getvalue())
            self.assertEqual(configuration['algorithm'], 'pd3qn_dag')
            self.assertEqual(configuration['display_name'], 'PD3QN-DAG')
            self.assertIn('pd3qn_dag_', configuration['checkpoint_dir'])
        with patch('sys.argv', ['pd3qn_dag', '--normalization-mode', 'legacy']), \
             redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), \
             self.assertRaises(SystemExit) as stopped:
            parse_scenario('train', 'pd3qn_dag')
        self.assertEqual(stopped.exception.code, 2)

    def test_dag_loader_does_not_fall_back_to_old_pd3qn_names(self):
        from algorithms.pd3qn import evaluate

        scenario = replace(evaluate.SCENARIO, checkpoint_override='old_v5_directory')
        agent, environment = MagicMock(), MagicMock()
        environment.reset.return_value = (np.zeros(1), {})
        environment.action_space.n = 2
        with patch.object(evaluate, 'SCENARIO', scenario), \
             patch.object(evaluate, 'CloudWorkflowEnv_PD3QN_Cost', return_value=environment), \
             patch.object(evaluate, 'D3QNAgent', return_value=agent), \
             patch.object(evaluate, 'bind_normalization'), \
             patch.object(evaluate.os.path, 'exists',
                          side_effect=lambda path: Path(path).suffix != '.pth' or Path(path).name == 'best_pd3qn.pth'), \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                evaluate.build_agent_and_env_kwargs()
        agent.load.assert_not_called()

    def test_old_traces_are_isolated_and_duplicate_training_is_protected(self):
        from common.f11 import protect_training_outputs
        from common.scenarios import Scenario
        from algorithms.pd3qn_dag.identity import protect_evaluation_outputs

        with tempfile.TemporaryDirectory() as directory, patch('common.scenarios.ROOT', Path(directory)):
            root = Path(directory)
            scenario = Scenario(mode='train', normalization_mode=PD3QN_UNITS)
            old_paths = [root / 'outputs' / 'ckpts' / 'pd3qn_old' / 'best_pd3qn.pth',
                         root / 'outputs' / 'ckpts' / 'pd3qn_v5_old' / 'best_pd3qn_v5.pth',
                         root / 'outputs' / 'run' / 'pd3qn_v5' / 'episodes' / 'seed700.json']
            for path in old_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('old artifact', encoding='utf-8')
            protect_training_outputs(scenario.checkpoint_dir('pd3qn'), scenario.log_dir('pd3qn'))
            protect_evaluation_outputs(replace(scenario, mode='eval', seed_spec='700'), root)
            new_checkpoint = scenario.checkpoint_dir('pd3qn') / 'best_pd3qn_dag.pth'
            new_checkpoint.parent.mkdir(parents=True)
            new_checkpoint.write_text('new artifact', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                protect_training_outputs(scenario.checkpoint_dir('pd3qn'), scenario.log_dir('pd3qn'))
            self.assertEqual(new_checkpoint.read_text(encoding='utf-8'), 'new artifact')
            for path in old_paths:
                self.assertEqual(path.read_text(encoding='utf-8'), 'old artifact')

    def test_dag_evaluation_collision_fails_before_loading_or_opening_csv(self):
        from algorithms.pd3qn import evaluate
        from algorithms.pd3qn_dag.identity import evaluation_csv, evaluation_directory

        scenario = replace(evaluate.SCENARIO, seed_spec='700')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = evaluation_csv(scenario, root)
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text('existing results', encoding='utf-8')
            with patch.object(evaluate, 'SCENARIO', scenario), patch.object(evaluate, 'ROOT_DIR', str(root)), \
                 patch.object(evaluate, 'build_agent_and_env_kwargs', side_effect=AssertionError('loaded')), \
                 patch.object(evaluate, 'CSVLogger', side_effect=AssertionError('opened')), \
                 redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                evaluate.main()
            self.assertEqual(csv_path.read_text(encoding='utf-8'), 'existing results')
            self.assertNotEqual(evaluation_directory(scenario, root),
                                evaluation_directory(replace(scenario, model_seed=2), root))
            self.assertNotEqual(evaluation_directory(scenario, root),
                                evaluation_directory(replace(scenario, train_task='M'), root))

    def test_dag_measurement_collision_keeps_existing_detail(self):
        from types import SimpleNamespace
        from common.experiment_measurements import EvaluationRecorder
        from algorithms.pd3qn_dag.identity import protect_evaluation_outputs, evaluation_directory
        from algorithms.pd3qn import evaluate

        scenario = replace(evaluate.SCENARIO, seed_spec='700')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measurements = evaluation_directory(scenario, root) / 'measurements'
            path = measurements / 'workflows' / 'seed700.csv'
            path.parent.mkdir(parents=True)
            path.write_text('existing details', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                protect_evaluation_outputs(scenario, root)
            with patch.dict('os.environ', EXPERIMENT_METRICS='1'), self.assertRaises(FileExistsError):
                EvaluationRecorder(scenario, 'pd3qn_dag', SimpleNamespace(random_seed=700), {},
                                   directory=measurements)
            self.assertEqual(path.read_text(encoding='utf-8'), 'existing details')

    def make_parent_env(self, parents=(0,), bits=(8e9,), arrivals=(0.0,)):
        from common.read_xml_opt_Tsize import Task
        from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost

        child = len(parents)
        graph = nx.DiGraph((parent, child) for parent in parents)
        tasks = [Task(task_id=i, workload_mi=0.0, children=[child], out_file_size_sum_bits=value)
                 for i, value in zip(parents, bits)]
        tasks.append(Task(task_id=child, workload_mi=1.0, parents=list(parents),
                          parent_in_bits=dict(zip(parents, bits)), from_parents_bits=sum(bits)))
        self.enterContext(patch("environments.pd3qn.poisson_arrival_times", return_value=list(arrivals)))
        self.enterContext(patch("environments.pd3qn.load_workflow_from_dax",
                                side_effect=lambda *args, **kwargs: (graph.copy(), copy.deepcopy(tasks))))
        return CloudWorkflowEnv_PD3QN_Cost(dax_paths=["parent_payload.xml"],
            workflows_per_episode=len(arrivals), deadline_mode="none", random_seed=7,
            num_hosts=1, vms_per_host=(2,), pc_tiers=(1.0,), bw_tiers=(1000.0,),
            normalization_mode=PD3QN_UNITS)

    def test_parent_vm_location_disambiguates_identical_original_observations(self):
        on_vm0, on_vm1 = self.make_parent_env(), self.make_parent_env()
        initial0, _ = on_vm0.reset()
        initial1, _ = on_vm1.reset()
        parent_slice = slice(6 + 3 * on_vm0.num_vms, 6 + 4 * on_vm0.num_vms)
        np.testing.assert_array_equal(initial0[parent_slice], [0.0, 0.0])
        np.testing.assert_array_equal(initial1[parent_slice], [0.0, 0.0])
        obs0, *_ = on_vm0.step(0)
        obs1, *_ = on_vm1.step(1)
        self.assertEqual(obs0.size, 6 + 4 * on_vm0.num_vms + 2 * on_vm0.num_hosts)
        np.testing.assert_array_equal(obs0[:parent_slice.start], obs1[:parent_slice.start])
        np.testing.assert_array_equal(obs0[parent_slice.stop:], obs1[parent_slice.stop:])
        scale = 8 * 1024 ** 3
        np.testing.assert_allclose(obs0[parent_slice], [8e9 / scale, 0.0])
        np.testing.assert_allclose(obs1[parent_slice], [0.0, 8e9 / scale])
        # Both VMs share a host: locality is per VM, not per host.
        self.assertEqual(on_vm1.task_assigned_host_id[0], on_vm1.vms[0].host_id)
        durations = []
        for env in (on_vm0, on_vm1):
            start = env.current_time
            terminal, _, done, _, _ = env.step(0)
            durations.append(env.task_end_time[1] - start)
            self.assertTrue(done)
            np.testing.assert_array_equal(terminal[parent_slice], [0.0, 0.0])
        self.assertEqual(durations, [1.0, 9.0])

    def test_parent_payload_sums_per_vm_with_workflow_global_ids(self):
        env = self.make_parent_env(parents=(0, 1), bits=(2e9, 3e9), arrivals=(0.0, 0.0))
        env.reset()
        self.assertEqual(len(env.workflows), 2)
        first_base, second_base = env.wf_task_base[0], env.wf_task_base[1]
        for base, vm in ((first_base, 1), (second_base, 0)):
            for local_parent in (0, 1):
                gid = base + local_parent
                env.task_state[gid] = "Finished"
                env.task_assigned_vm_id[gid] = vm
                env.task_assigned_host_id[gid] = 0
            env.task_state[base + 2] = "Ready"
        env.ready_task_ids = [first_base + 2, second_base + 2]
        env._cur_task_gid = second_base + 2
        parent_slice = slice(6 + 3 * env.num_vms, 6 + 4 * env.num_vms)
        np.testing.assert_allclose(env._build_obs()[parent_slice], [5e9 / (8 * 1024 ** 3), 0.0])
        env._cur_task_gid = first_base + 2
        np.testing.assert_allclose(env._build_obs()[parent_slice], [0.0, 5e9 / (8 * 1024 ** 3)])

    def test_host_history_disambiguates_equal_prefix_with_different_energy(self):
        from common.read_xml_opt_Tsize import Task
        from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost

        def run(actions):
            first, second = nx.DiGraph(), nx.DiGraph()
            first.add_nodes_from([0, 1])
            second.add_node(0)
            loads = iter([(first, [Task(task_id=0, workload_mi=1.),
                                   Task(task_id=1, workload_mi=90.)]),
                          (second, [Task(task_id=0, workload_mi=1.)])])
            with patch('environments.pd3qn.poisson_arrival_times', return_value=[0., 100.]), \
                 patch('environments.pd3qn.load_workflow_from_dax',
                       side_effect=lambda *a, **k: copy.deepcopy(next(loads))):
                env = CloudWorkflowEnv_PD3QN_Cost(dax_paths=['history.xml'], deadline_mode='none',
                    workflows_per_episode=2, num_hosts=2, vms_per_host=(1, 1),
                    pc_tiers=(1.,), bw_tiers=(1000.,), normalization_mode=PD3QN_UNITS)
                np.testing.assert_array_equal(env._build_obs()[-4:], np.zeros(4))
                busy, *_ = env.step(actions[0])
                expected = np.zeros((2, 2))
                expected[actions[0], 0] = 1.
                np.testing.assert_array_equal(busy[-4:], expected.reshape(-1))
                obs, *_ = env.step(actions[1])
                self.assertEqual(env.current_time, 100.)
                self.assertEqual(env.observation_space.shape, obs.shape)
                terminal, reward, done, _, info = env.step(0)
                self.assertTrue(done)
                self.assertTrue(np.isfinite(terminal).all())
                return obs, reward, info['delta_energy']

        obs_a, reward_a, energy_a = run([0, 1])
        obs_b, reward_b, energy_b = run([1, 0])
        np.testing.assert_array_equal(obs_a[:-4], obs_b[:-4])
        np.testing.assert_allclose(obs_a[-4:], [1., .099, 1., .010])
        np.testing.assert_allclose(obs_b[-4:], [1., .010, 1., .099])
        self.assertAlmostEqual(energy_a, 1619.2)
        self.assertAlmostEqual(energy_b, 204.1)
        self.assertNotEqual(reward_a, reward_b)

    def test_host_history_uses_shared_window_and_clears_on_reset(self):
        env = self.make_parent_env()
        env._add_interval_load(0, 1., 2., 1.)
        env._add_interval_load(0, 5., 10., 1.)
        env.current_time = 12.
        obs = env._build_obs()
        np.testing.assert_allclose(obs[-2:], [1., .002])
        # Repeated observation is read-only; both VMs contribute to one Host window.
        segments, energy = copy.deepcopy(env._host_segments), env.total_energy
        np.testing.assert_array_equal(obs, env._build_obs())
        self.assertEqual(segments, env._host_segments)
        self.assertEqual(energy, env.total_energy)
        reset_obs, _ = env.reset()
        np.testing.assert_array_equal(reset_obs[-2:], [0., 0.])

    def test_old_agent_defaults_keep_update_clock_and_max_initial_priority(self):
        agent = self.make_agent(paper_mode=False)
        agent.select_action(np.array([0.0]), np.ones(2))
        self.assertEqual(agent._eps_steps, 0)
        self.remember(agent, 4.0)
        self.remember(agent, 1.0)
        self.assertEqual(agent.priorities, [1.0, 1.0])
        agent.update()
        self.assertEqual(agent._eps_steps, 1)


if __name__ == "__main__":
    unittest.main()
