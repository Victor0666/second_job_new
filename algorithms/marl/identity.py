"""Stable experiment identity and isolated outputs for the adapted MARL baseline."""
from pathlib import Path

MODEL_VERSION = 'marl_q_v2'
DISPLAY_NAME = 'MARL-Q'


def evaluation_directory(scenario, project_dir):
    custom_csv = (scenario.marl_options or {}).get('out')
    if custom_csv:
        csv = Path(custom_csv).resolve()
        return csv.parent / (csv.stem + '_' + MODEL_VERSION + '_artifacts')
    source = scenario.normalization_kwargs()['source_train_scene']
    return (Path(project_dir) / 'outputs' / 'run' / MODEL_VERSION / scenario.output_version
            / f'from_{source}_model{scenario.model_seed}'
            / f'eval_{scenario.task}{scenario.resource}_{scenario.deadline}')


def evaluation_csv(scenario, project_dir):
    custom_csv = (scenario.marl_options or {}).get('out')
    return Path(custom_csv).resolve() if custom_csv else (
        evaluation_directory(scenario, project_dir) / f'evaluation_{MODEL_VERSION}.csv')
