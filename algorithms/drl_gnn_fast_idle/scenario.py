"""Variant-local CLI and provenance; no registration in shared algorithm tables."""
import argparse
import hashlib
import json
from pathlib import Path

from .config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG, INPUT_ENCODING,
                     SCHEDULING_PROTOCOL, GRAPHS_PER_BACKWARD, PPO_UPDATE_EXECUTION)
from .selection import SELECTION_TAG, selection_metadata
from common.experiment_protocol import DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG
from common.r010_paths import training_name
from common.scenarios import ROOT, SMALL_PROBABILITY, VM_COUNTS, Scenario


class IdleScenario(Scenario):
    def __post_init__(self):
        super().__post_init__()
        if self.normalization_mode != CURRENT_VERSION:
            raise ValueError('DRL-GNN-Fast-Idle requires fixed F19 source-training references')
        if self.mode not in ('train', 'eval') or self.protocol() is None:
            raise ValueError('DRL-GNN-Fast-Idle requires a fixed, disjoint experiment protocol')

    def model_name(self, algorithm=ALGORITHM):
        if algorithm != ALGORITHM:
            raise ValueError('DRL-GNN-Fast-Idle paths only accept drl_gnn_fast_idle')
        name = training_name(self, MODEL_VERSION, dict(protocol=self.selection_protocol(),
            input_encoding=INPUT_ENCODING, scheduling_protocol=SCHEDULING_PROTOCOL))
        return f'{self.source_tag}_v6_sr092' + name[name.index('_m'):]

    @staticmethod
    def scene_tag(task, resource, deadline):
        return task + resource + '_' + {'Tight': 'T', 'Medium': 'M', 'Loose': 'L'}[deadline]

    @property
    def source_tag(self):
        if self.mode != 'eval':
            return self.scene_tag(self.task, self.resource, self.deadline)
        return self.scene_tag(self.train_task, self.train_resource or self.resource,
                              self.train_deadline or self.deadline)

    @property
    def target_tag(self):
        return self.scene_tag(self.task, self.resource, self.deadline)

    def checkpoint_dir(self, algorithm=ALGORITHM):
        if self.mode == 'eval' and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return ROOT / 'outputs' / 'ckpts' / OUTPUT_TAG / self.model_name(algorithm)

    def log_dir(self, algorithm=ALGORITHM):
        return ROOT / 'outputs' / 'logs' / OUTPUT_TAG / self.model_name(algorithm)

    @property
    def evaluation_root(self):
        return ROOT / 'outputs' / 'run' / OUTPUT_TAG

    @property
    def evaluation_dir(self):
        key = hashlib.sha256(self.evaluation_csv.name.encode()).hexdigest()[:16]
        return self.evaluation_root / '_details' / key

    @property
    def evaluation_csv(self):
        seeds = sorted(self.seeds)
        seed_tag = str(seeds[0]) if len(seeds) == 1 else f'{seeds[0]}-{seeds[-1]}'
        if seeds != list(range(seeds[0], seeds[-1]+1)):
            seed_tag += '_' + hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
        config_tag = self.model_name().rsplit('_', 1)[-1]
        return self.evaluation_root / (
            f'eval_{OUTPUT_TAG}_v6_sr092_{self.target_tag}_from_{self.source_tag}_'
            f'm{self.model_seed}_s{seed_tag}_{config_tag}.csv')

    @property
    def output_version(self):
        return super().output_version + '_' + SELECTION_TAG

    def selection_protocol(self):
        metadata = self.protocol().metadata()
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata())
        return metadata

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(algorithm=ALGORITHM, data_split=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(experiment_protocol=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=ALGORITHM):
        result = super().describe(None)
        result.update(algorithm=ALGORITHM, display_name=DISPLAY_NAME, model_version=MODEL_VERSION,
            policy_input=INPUT_ENCODING, scheduling_protocol=SCHEDULING_PROTOCOL,
            execution=dict(ppo_update=PPO_UPDATE_EXECUTION,
                graphs_per_backward=(self.paper_options or {}).get('graphs_per_backward', GRAPHS_PER_BACKWARD)),
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
    if mode == 'train':
        parser.add_argument('--graphs-per-backward', type=int, choices=(1, 2, 4), default=4,
                            help='Execution grouping only; PPO minibatch remains 128')
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
    values['paper_options'] = dict(device=values.pop('device'),
                                  graphs_per_backward=values.pop('graphs_per_backward', GRAPHS_PER_BACKWARD))
    if mode == 'eval':
        values['seed_spec'] = values.pop('seeds')
        values['checkpoint_override'] = values.pop('checkpoint_dir')
    try:
        scenario = IdleScenario(mode=mode, **values)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if dry_run:
        print(json.dumps(scenario.describe(), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    from common.f22_cache import check_cache
    scenario.validate_calibration()
    check_cache(scenario)
    return scenario
