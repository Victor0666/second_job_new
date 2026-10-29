"""Stable experiment identity and isolated outputs for the adapted MARL baseline."""
from pathlib import Path
import hashlib

from common.r010_paths import training_name, evaluation_stem
from algorithms.marl.selection import selection_metadata

MODEL_VERSION = 'marl_q_v4_2'
DISPLAY_NAME = 'MARL-Q4.2'
OBSERVATION_LAYOUT = 'task4_ready1_host2_local_vm5_wait_exec_v1'
SCHEDULING_POLICY = 'idle_vm_only_v1'


def training_run_name(scenario):
    protocol = scenario.protocol()
    return training_name(scenario, MODEL_VERSION, scenario.with_workload_metadata(dict(
        protocol=protocol.metadata() if protocol else {'data_split': 'Legacy'},
        checkpoint_selection=selection_metadata(), observation_layout=OBSERVATION_LAYOUT,
        scheduling_policy=SCHEDULING_POLICY)))


def evaluation_csv(scenario, project_dir):
    custom_csv = (scenario.marl_options or {}).get('out')
    return Path(custom_csv).resolve() if custom_csv else (
        Path(project_dir) / 'outputs' / 'run' / MODEL_VERSION /
        (evaluation_stem(scenario, MODEL_VERSION, training_run_name(scenario)) + '.csv'))


def evaluation_directory(scenario, project_dir):
    csv = evaluation_csv(scenario, project_dir)
    key = hashlib.sha256(csv.name.encode()).hexdigest()[:16]
    return csv.parent / '_details' / key
