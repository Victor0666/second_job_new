"""Short training directories and self-describing, flat evaluation files."""
import hashlib

from common.r010_paths import (evaluation_stem as compact_evaluation_stem,
    source_scene, training_name as compact_training_name)
from common.scenarios import ROOT
from .config import (ARCHITECTURE, COMMON_REWARD, INPUT_ENCODING, MODEL_VERSION,
    OUTPUT_TAG, REWARD_PROTOCOL, SCHEDULING_PROTOCOL, SOURCE_COMMIT, TwoStageGATConfig)
from .selection import selection_protocol


def training_name(scenario):
    ppo = TwoStageGATConfig(seed=scenario.model_seed).as_dict()
    ppo.pop('device')  # CPU/CUDA evaluation must resolve the same source model.
    settings = dict(protocol=selection_protocol(scenario), architecture=ARCHITECTURE,
        policy_input=INPUT_ENCODING, reward_protocol=REWARD_PROTOCOL, reward=COMMON_REWARD,
        scheduling=SCHEDULING_PROTOCOL, source_commit=SOURCE_COMMIT, ppo=ppo)
    name = compact_training_name(scenario, MODEL_VERSION, settings)
    source = source_scene(scenario)
    pair, deadline = source.rsplit('_', 1)
    short = pair + '_' + {'Tight':'T', 'Medium':'M', 'Loose':'L'}[deadline]
    return short + name[len(source):]


def checkpoint_directory(scenario):
    if scenario.mode == 'eval' and scenario.checkpoint_override:
        from pathlib import Path
        return Path(scenario.checkpoint_override).resolve()
    return ROOT / 'outputs/ckpts' / OUTPUT_TAG / training_name(scenario)


def log_directory(scenario):
    return ROOT / 'outputs/logs' / OUTPUT_TAG / training_name(scenario)


def evaluation_root():
    return ROOT / 'outputs/run' / OUTPUT_TAG


def evaluation_stem(scenario):
    return compact_evaluation_stem(scenario, OUTPUT_TAG, training_name(scenario))


def evaluation_csv(scenario):
    return evaluation_root() / (evaluation_stem(scenario) + '.csv')


def evaluation_directory(scenario):
    digest = hashlib.sha256(evaluation_stem(scenario).encode()).hexdigest()[:16]
    return evaluation_root() / '_details' / digest


def evaluation_identity(scenario):
    return dict(target_scene=f'{scenario.task}{scenario.resource}_{scenario.deadline}',
        source_scene=source_scene(scenario), task=scenario.task, resource=scenario.resource,
        deadline=scenario.deadline, train_task=scenario.train_task,
        train_resource=scenario.train_resource or scenario.resource,
        train_deadline=scenario.train_deadline or scenario.deadline,
        model_seed=scenario.model_seed, model_version=MODEL_VERSION)
