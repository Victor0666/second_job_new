"""Variant-local CLI and provenance; no registration in shared algorithm tables."""
import argparse
import json
from pathlib import Path

from algorithms.pd3qn.config import agent_kwargs
from algorithms.pd3qn_fe_r010 import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, reward_metadata
from algorithms.pd3qn_fe_r010.selection import SELECTION_TAG, selection_metadata
from common.experiment_protocol import DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG, SHARED_VERSIONS
from common.scenarios import ROOT, SMALL_PROBABILITY, VM_COUNTS, Scenario
from common.r010_paths import training_name, evaluation_stem


class R010Scenario(Scenario):
    def __post_init__(self):
        super().__post_init__()
        if self.normalization_mode not in (*SHARED_VERSIONS, 'legacy'):
            raise ValueError('PD3QN-FE-R010 requires historical PD3QN, not PD3QN-DAG')
        if self.mode not in ('train', 'eval') or self.protocol() is None:
            raise ValueError('PD3QN-FE-R010 requires a fixed, disjoint experiment protocol')

    def model_name(self, algorithm=ALGORITHM):
        if algorithm != ALGORITHM:
            raise ValueError('PD3QN-FE-R010 paths only accept pd3qn_fe_r010')
        return training_name(self, MODEL_VERSION,
            dict(protocol=self.selection_protocol(), reward=reward_metadata(), agent=agent_kwargs(self)))

    def checkpoint_dir(self, algorithm=ALGORITHM):
        if self.mode == 'eval' and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return ROOT / 'outputs' / 'ckpts' / ALGORITHM / self.model_name(algorithm)

    def log_dir(self, algorithm=ALGORITHM):
        return ROOT / 'outputs' / 'logs' / ALGORITHM / self.model_name(algorithm)

    @property
    def evaluation_root(self):
        return ROOT / 'outputs' / 'run' / ALGORITHM

    @property
    def evaluation_stem(self):
        return evaluation_stem(self, ALGORITHM, self.model_name())

    @property
    def evaluation_dir(self):
        return self.evaluation_root / '_details' / self.evaluation_stem

    @property
    def evaluation_csv(self):
        return self.evaluation_root / (self.evaluation_stem + '.csv')

    @property
    def output_version(self):
        return MODEL_VERSION + '_' + super().output_version + '_' + SELECTION_TAG

    def selection_protocol(self):
        metadata = self.protocol().metadata()
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata())
        return metadata

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(algorithm=ALGORITHM, model_version=MODEL_VERSION,
                        reward=reward_metadata(), data_split=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(model_version=MODEL_VERSION, reward=reward_metadata(), experiment_protocol=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=ALGORITHM):
        result = super().describe()
        result.update(algorithm=ALGORITHM, display_name=DISPLAY_NAME, model_version=MODEL_VERSION,
            reward=reward_metadata(), agent=agent_kwargs(self),
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
    parser.add_argument('--normalization-mode', choices=(*SHARED_VERSIONS, 'legacy'), default=CURRENT_VERSION)
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
        scenario = R010Scenario(mode=mode, **values)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if dry_run:
        print(json.dumps(scenario.describe(), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    from common.f22_cache import check_cache
    scenario.validate_calibration()
    check_cache(scenario)
    return scenario
