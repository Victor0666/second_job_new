"""Compact, portable output names shared only by the three R010 variants."""
import hashlib
import json
from pathlib import Path

from common.workflow_opt import ENERGY_ACCOUNTING


def source_scene(scenario):
    if scenario.mode != 'eval':
        return f'{scenario.task}{scenario.resource}_{scenario.deadline}'
    return (f'{scenario.train_task}{scenario.train_resource or scenario.resource}_'
            f'{scenario.train_deadline or scenario.deadline}')


def training_name(scenario, model_version, settings):
    calibration = Path(scenario.normalization_config)
    scales = (hashlib.sha256(calibration.read_bytes()).hexdigest()
              if scenario.normalization_mode != 'legacy' and calibration.is_file() else None)
    identity = dict(model_version=model_version, settings=settings,
                    normalization=scenario.normalization_mode, calibration_sha256=scales,
                    energy_accounting=ENERGY_ACCOUNTING)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8]
    return f'{source_scene(scenario)}_m{scenario.model_seed}_c{digest}'


def evaluation_stem(scenario, algorithm_tag, train_name):
    seeds = sorted(scenario.seeds)
    if len(seeds) == 1:
        seed_tag = str(seeds[0])
    elif seeds == list(range(seeds[0], seeds[-1] + 1)):
        seed_tag = f'{seeds[0]}-{seeds[-1]}'
    else:
        seed_tag = '+'.join(map(str, seeds))
        if len(seed_tag) > 28:
            digest = hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
            seed_tag = f'n{len(seeds)}-{digest}'
    config_tag = train_name.rsplit('_', 1)[-1]
    return (f'eval_{algorithm_tag}_{scenario.task}{scenario.resource}_{scenario.deadline}_'
            f'from{source_scene(scenario)}_m{scenario.model_seed}_s{seed_tag}_{config_tag}')
