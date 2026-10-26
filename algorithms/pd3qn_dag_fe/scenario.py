"""Variant-local CLI and provenance; no registration in shared algorithm tables."""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path

from algorithms.pd3qn_dag_fe.config import agent_kwargs
from algorithms.pd3qn_dag_fe import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OBSERVATION_LAYOUT, HEAD_LAYOUT
from algorithms.pd3qn_dag_fe.selection import SELECTION_TAG, selection_metadata
from common.experiment_protocol import DEFAULT_PROTOCOL, protocol_hash
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG
from common.scenarios import ROOT, SMALL_PROBABILITY, VM_COUNTS, Scenario
from common.workflow_opt import ENERGY_ACCOUNTING


@dataclass(frozen=True)
class DAGFEScenario(Scenario):
    normalization_mode: str = CURRENT_VERSION

    def __post_init__(self):
        super().__post_init__()
        if self.normalization_mode != CURRENT_VERSION:
            raise ValueError('PD3QN-DAG-FE requires f19_shared_v1 calibration')
        if self.mode not in ('train', 'eval') or self.protocol() is None:
            raise ValueError('PD3QN-DAG-FE requires a fixed, disjoint experiment protocol')

    def model_name(self, algorithm=ALGORITHM):
        if algorithm != ALGORITHM:
            raise ValueError('PD3QN-DAG-FE paths only accept pd3qn_dag_fe')
        task = self.task if self.mode == 'train' else self.train_task
        resource = self.resource if self.mode == 'train' else (self.train_resource or self.resource)
        deadline = self.deadline if self.mode == 'train' else (self.train_deadline or self.deadline)
        return (f'{MODEL_VERSION}_task{task}_res{resource}_model{self.model_seed}_{deadline}_'
                f'{self.normalization_mode}_p{protocol_hash(self.experiment_protocol)[:12]}_'
                f'{ENERGY_ACCOUNTING}_{SELECTION_TAG}')

    def checkpoint_dir(self, algorithm=ALGORITHM):
        if self.mode == 'eval' and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return ROOT / 'outputs' / 'ckpts' / self.model_name(algorithm)

    def log_dir(self, algorithm=ALGORITHM):
        return ROOT / 'outputs' / 'logs' / self.model_name(algorithm)

    @property
    def evaluation_dir(self):
        target = f'task{self.task}_res{self.resource}_{self.deadline}_seeds{self.seed_spec.replace(",", "_")}'
        return ROOT / 'outputs' / 'run' / ALGORITHM / self.model_name() / target

    @property
    def evaluation_csv(self):
        seeds = self.seed_spec.replace(',', '_')
        return self.evaluation_dir / (f'eval_{ALGORITHM}_task{self.task}_res{self.resource}_'
            f'{self.deadline}_model{self.model_seed}_seeds{seeds}.csv')

    @property
    def output_version(self):
        return MODEL_VERSION + '_' + OBSERVATION_LAYOUT + '_' + super().output_version + '_' + SELECTION_TAG

    def selection_protocol(self):
        metadata = self.protocol().metadata()
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata())
        return metadata

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(algorithm=ALGORITHM, model_version=MODEL_VERSION,
                        observation_layout=OBSERVATION_LAYOUT, head_layout=HEAD_LAYOUT,
                        data_split=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(model_version=MODEL_VERSION, observation_layout=OBSERVATION_LAYOUT,
                        head_layout=HEAD_LAYOUT, experiment_protocol=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=ALGORITHM):
        result = super().describe()
        result.update(algorithm=ALGORITHM, display_name=DISPLAY_NAME, model_version=MODEL_VERSION,
            observation_layout=OBSERVATION_LAYOUT, input_dim=3 + 3 * sum(self.vms_per_host),
            head_layout=HEAD_LAYOUT, agent=agent_kwargs(self),
            checkpoint_selection=selection_metadata(), experiment_protocol=self.selection_protocol(),
            checkpoint_dir=str(self.checkpoint_dir()), training_log_dir=str(self.log_dir()))
        if self.mode == 'eval':
            result['evaluation_directory'] = str(self.evaluation_dir)
            result['evaluation_csv'] = str(self.evaluation_csv)
        return result


def parse_scenario(mode, argv=None):
    parser = argparse.ArgumentParser(description=f'{DISPLAY_NAME}: {mode}')
    parser.add_argument('--task', choices=VM_COUNTS, default='S')
    parser.add_argument('--resource', choices=VM_COUNTS, default='S')
    parser.add_argument('--deadline', choices=SMALL_PROBABILITY, default='Tight')
    parser.add_argument('--model-seed', type=int, default=1)
    parser.add_argument('--experiment-protocol', default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--normalization-mode', choices=(CURRENT_VERSION,), default=CURRENT_VERSION)
    parser.add_argument('--normalization-config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    parser.add_argument('--dry-run', action='store_true')
    if mode == 'train':
        parser.add_argument('--episodes', type=int, default=600)
    else:
        parser.add_argument('--seeds', default='700-729')
        parser.add_argument('--train-task', choices=VM_COUNTS, default='S')
        parser.add_argument('--train-resource', choices=VM_COUNTS)
        parser.add_argument('--train-deadline', choices=SMALL_PROBABILITY)
        parser.add_argument('--checkpoint-dir')
    values = vars(parser.parse_args(argv))
    dry_run = values.pop('dry_run')
    values['paper_options'] = dict(device=values.pop('device'))
    if mode == 'eval':
        values['seed_spec'] = values.pop('seeds')
        values['checkpoint_override'] = values.pop('checkpoint_dir')
    try:
        scenario = DAGFEScenario(mode=mode, **values)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if dry_run:
        print(json.dumps(scenario.describe(), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    from common.f22_cache import check_cache
    scenario.validate_calibration()
    check_cache(scenario)
    return scenario
