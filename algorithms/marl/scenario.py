"""MARL-only effective selection policy; shared split/calibration bindings stay frozen."""
from common.scenarios import Scenario, ROOT, parse_scenario as shared_parse
from algorithms.marl.identity import MODEL_VERSION, OBSERVATION_LAYOUT, SCHEDULING_POLICY, training_run_name
from algorithms.marl.selection import SELECTION_TAG, selection_metadata


class MARLScenario(Scenario):
    def selection_protocol(self):
        metadata = self.protocol().metadata() if self.protocol() else {'data_split': 'Legacy'}
        return self.with_workload_metadata(dict(metadata, selection=SELECTION_TAG, checkpoint_selection=selection_metadata()))

    def model_name(self, algorithm='marl'):
        return training_run_name(self) if algorithm == 'marl' else super().model_name(algorithm)

    def checkpoint_dir(self, algorithm='marl'):
        if algorithm != 'marl' or (self.mode == 'eval' and self.checkpoint_override):
            return super().checkpoint_dir(algorithm)
        return ROOT / 'outputs' / 'ckpts' / MODEL_VERSION / self.model_name(algorithm)

    def log_dir(self, algorithm='marl'):
        if algorithm != 'marl':
            return super().log_dir(algorithm)
        return ROOT / 'outputs' / 'logs' / MODEL_VERSION / self.model_name(algorithm)

    @property
    def output_version(self):
        return super().output_version + '_' + SELECTION_TAG

    def reproducibility_metadata(self):
        return dict(super().reproducibility_metadata(), algorithm=MODEL_VERSION,
            observation_layout=OBSERVATION_LAYOUT, scheduling_policy=SCHEDULING_POLICY,
            data_split=self.selection_protocol(), checkpoint_selection=selection_metadata())

    def normalization_metadata(self):
        return dict(super().normalization_metadata(), observation_layout=OBSERVATION_LAYOUT,
            scheduling_policy=SCHEDULING_POLICY, experiment_protocol=self.selection_protocol(),
            checkpoint_selection=selection_metadata())

    def describe(self, algorithm='marl'):
        return dict(super().describe(algorithm), observation_layout=OBSERVATION_LAYOUT,
            scheduling_policy=SCHEDULING_POLICY, checkpoint_selection=selection_metadata(),
            experiment_protocol=self.selection_protocol(), global_obs_dim=5+2*self.num_hosts,
            local_obs_dim=4+5*max(self.vms_per_host))


def parse_scenario(mode, algorithm='marl'):
    return shared_parse(mode, algorithm, scenario_class=MARLScenario)
