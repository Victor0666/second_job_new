"""Independent names and result paths for the DAG workflow baseline."""
from pathlib import Path

RUN_NAME = 'pd3qn_dag'
DISPLAY_NAME = 'PD3QN-DAG'
MODEL_VERSION = 'pd3qn_dag_v1'


def evaluation_directory(scenario, root):
    source = (f'from_task{scenario.train_task}_res{scenario.train_resource or scenario.resource}_'
              f'{scenario.train_deadline or scenario.deadline}')
    target = f'task{scenario.task}_res{scenario.resource}_{scenario.deadline}_{source}_model{scenario.model_seed}'
    return Path(root) / 'outputs' / 'run' / RUN_NAME / scenario.output_version / target


def evaluation_csv(scenario, root):
    seeds = scenario.seed_spec.replace(',', '_')
    return evaluation_directory(scenario, root) / (
        f'eval_{RUN_NAME}_task{scenario.task}_res{scenario.resource}_'
        f'{scenario.deadline}_model{scenario.model_seed}_seeds{seeds}.csv')


def protect_evaluation_outputs(scenario, root):
    paths = [evaluation_csv(scenario, root)]
    directory = evaluation_directory(scenario, root) / 'measurements'
    for seed in scenario.seeds:
        paths.extend((directory / 'episodes' / f'seed{seed}.json',
                      directory / 'workflows' / f'seed{seed}.csv',
                      directory / 'decisions' / f'seed{seed}.json'))
    for path in paths:
        if path.exists():
            raise FileExistsError(f'{DISPLAY_NAME}: refusing to overwrite evaluation artifact: {path}')
