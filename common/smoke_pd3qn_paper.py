"""Real PD3QN train/save/load/evaluate check in temporary output directories."""
import argparse
import csv
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from common.f18_state_normalization import PD3QN_UNITS
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from algorithms.pd3qn import train, evaluate


def run_smoke():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='pd3qn_paper_') as directory:
        root = Path(directory)
        cache = root / 'cache.json'
        console = io.StringIO()
        with redirect_stdout(console):
            precompute(Scenario(mode='precompute', seed_spec='0,1,600,700', cache_output=str(cache)))

        class LocalScenario(Scenario):
            @property
            def cache_path(self):
                return cache

            def model_name(self, algorithm):
                return str(root / 'logs' / super().model_name(algorithm))

            def log_dir(self, algorithm):
                return Path(self.model_name(algorithm))

            def checkpoint_dir(self, algorithm):
                return root / 'ckpts'

        scenario = LocalScenario(mode='train', episodes=2, normalization_mode=PD3QN_UNITS,
                                 normalization_config=str(root / 'no_calibration_needed.json'))
        with patch.object(train, 'SCENARIO', scenario), redirect_stdout(console):
            train.train()
        checkpoint = scenario.checkpoint_dir('pd3qn') / 'pd3qn_dag_final.pth'
        assert checkpoint.is_file()
        best_checkpoint = checkpoint.parent / 'best_pd3qn_dag.pth'
        assert best_checkpoint.is_file()
        assert not (checkpoint.parent / 'best_pd3qn.pth').exists()
        metadata = json.loads((scenario.log_dir('pd3qn') / 'measurement_metadata.json').read_text(encoding='utf-8'))
        assert metadata['algorithm'] == 'pd3qn_dag'
        assert 'PD3QN-DAG' in metadata['agents']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        best_saved = torch.load(best_checkpoint, map_location='cpu', weights_only=True)
        with (scenario.log_dir('pd3qn') / 'train_metrics_pd3qn_dag_cost.csv').open(
                encoding='utf-8', newline='') as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == scenario.episodes
        best_row = min(rows, key=lambda row: float(row['eval_energy']))
        assert best_saved['eps_steps'] == int(best_row['step'])
        assert best_saved['updates'] == int(best_row['step']) - 127
        assert saved['network']['input_dim'] == 6 + 4 * sum(scenario.vms_per_host) + 2 * scenario.num_hosts
        assert saved['network']['hidden_dims'] == [1024, 1024, 512, 512, 256]
        assert saved['normalization']['observation_layout'] == 'task6_vm4_parentdata_host2_v3'
        assert all(group['lr'] == 0.01 for group in saved['optim']['param_groups'])
        diagnostics = [json.loads(line) for line in
                       (scenario.log_dir('pd3qn') / 'f11_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
        transitions = diagnostics[-1]['learners']['PD3QN-DAG']['transitions']
        updates = diagnostics[-1]['learners']['PD3QN-DAG']['gradient_updates']
        assert saved['eps_steps'] == transitions
        assert updates == transitions - 127 and updates > 0
        assert all(torch.isfinite(value).all() for value in saved['online'].values())

        eval_scenario = LocalScenario(mode='eval', seed_spec='700', normalization_mode=PD3QN_UNITS,
                                      normalization_config=scenario.normalization_config)
        recorder = evaluate.EvaluationRecorder
        def local_recorder(*args, **kwargs):
            kwargs['directory'] = root / 'evaluation_measurements'
            return recorder(*args, **kwargs)
        with patch.object(evaluate, 'SCENARIO', eval_scenario), \
             patch.object(evaluate, 'EvaluationRecorder', side_effect=local_recorder), \
             redirect_stdout(console):
            kwargs, agent = evaluate.build_agent_and_env_kwargs()
            assert agent._eps_steps == best_saved['eps_steps']
            assert agent._updates == best_saved['updates']
            assert all(torch.equal(value, agent.online.state_dict()[key].cpu())
                       for key, value in best_saved['online'].items())
            weights = {key: value.clone() for key, value in agent.online.state_dict().items()}
            counters = (agent._eps_steps, agent._updates, agent._episode_steps)
            result = evaluate.evaluate_one_episode(evaluate.CloudWorkflowEnv_PD3QN_Cost, kwargs, agent)
            with patch.object(evaluate, 'ROOT_DIR', str(root)), \
                 patch.object(evaluate, 'build_agent_and_env_kwargs', return_value=(kwargs, agent)), \
                 patch.object(evaluate, 'EvaluationRecorder', recorder):
                evaluate.main()
                try:
                    evaluate.main()
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Duplicate evaluation must refuse existing output')
        evaluation_csvs = list(root.rglob('eval_pd3qn_dag_*.csv'))
        assert len(evaluation_csvs) == 1
        assert '[PD3QN-DAG train completed]' in console.getvalue()
        assert '[PD3QN-DAG eval completed]' in console.getvalue()
        assert counters == (agent._eps_steps, agent._updates, agent._episode_steps)
        assert all(torch.equal(value, agent.online.state_dict()[key]) for key, value in weights.items())
        reward, assignments, metrics, _ = result
        assert math.isfinite(reward) and assignments > 0
        assert metrics['num_workflows'] == 50 and metrics['num_tasks'] == assignments
        assert abs(metrics['energy_recalc_diff']) < 1e-6
        last_loss = diagnostics[-1]['learners']['PD3QN-DAG']['last_loss']
        assert np.isfinite(last_loss)
        return dict(description='Two real 50-workflow training episodes, configured PD3QN settings; minimum-validation-energy best-model evaluation. Not a convergence test.',
                    algorithm='pd3qn_dag', display_name='PD3QN-DAG',
                    training_log='train_metrics_pd3qn_dag_cost.csv', evaluation_csv=evaluation_csvs[0].name,
                    duplicate_evaluation_rejected=True,
                    model=saved['pd3qn_implementation'], normalization=PD3QN_UNITS,
                    observation_layout=saved['normalization']['observation_layout'],
                    learning_rate=saved['optim']['param_groups'][0]['lr'], training_seeds=[0, 1],
                    validation_seeds=[600], test_seed=700,
                    transitions=transitions, updates=updates, epsilon_steps=saved['eps_steps'],
                    last_loss=last_loss, parameters=sum(value.numel() for value in saved['online'].values()),
                    checkpoint='best_pd3qn_dag.pth', best_episode=int(best_row['episode']),
                    best_validation_energy=float(best_row['eval_energy']),
                    selected_checkpoint_epsilon_steps=best_saved['eps_steps'], final_checkpoint_saved=True,
                    observation_dim=saved['network']['input_dim'], calibration_file_needed=False,
                    evaluation_assignments=assignments, evaluation_return=reward,
                    evaluation_energy=metrics['total_energy'], energy_balance=metrics['energy_recalc_diff'],
                    evaluation_changes_training_state=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run_smoke()
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
