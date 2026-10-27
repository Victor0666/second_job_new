"""MARL-only effective selection policy; shared split/calibration bindings stay frozen."""
from common.scenarios import Scenario, parse_scenario as shared_parse
from algorithms.marl.identity import MODEL_VERSION, OBSERVATION_LAYOUT, SCHEDULING_POLICY, QUEUE_CAPACITY
from algorithms.pd3qn_dag_fe.selection import SELECTION_TAG, selection_metadata


class MARLScenario(Scenario):
    def selection_protocol(self):
        metadata = self.protocol().metadata() if self.protocol() else {'data_split': 'Legacy'}
        return dict(metadata, selection=SELECTION_TAG, checkpoint_selection=selection_metadata())

    def model_name(self, algorithm='marl'):
        return super().model_name(algorithm) + ('_' + SELECTION_TAG if algorithm == 'marl' else '')

    @property
    def output_version(self):
        return super().output_version + '_' + SELECTION_TAG

    def reproducibility_metadata(self):
        return dict(super().reproducibility_metadata(), algorithm=MODEL_VERSION,
            observation_layout=OBSERVATION_LAYOUT, scheduling_policy=SCHEDULING_POLICY,
            queue_capacity=QUEUE_CAPACITY, queue_capacity_counts='waiting_tasks_only',
            data_split=self.selection_protocol(), checkpoint_selection=selection_metadata())

    def normalization_metadata(self):
        return dict(super().normalization_metadata(), observation_layout=OBSERVATION_LAYOUT,
            scheduling_policy=SCHEDULING_POLICY, experiment_protocol=self.selection_protocol(),
            queue_capacity=QUEUE_CAPACITY, queue_capacity_counts='waiting_tasks_only',
            checkpoint_selection=selection_metadata())

    def describe(self, algorithm='marl'):
        return dict(super().describe(algorithm), observation_layout=OBSERVATION_LAYOUT,
            scheduling_policy=SCHEDULING_POLICY, checkpoint_selection=selection_metadata(),
            queue_capacity=QUEUE_CAPACITY, queue_capacity_counts='waiting_tasks_only',
            experiment_protocol=self.selection_protocol(), global_obs_dim=5+2*self.num_hosts,
            local_obs_dim=4+5*max(self.vms_per_host))


def parse_scenario(mode, algorithm='marl'):
    return shared_parse(mode, algorithm, scenario_class=MARLScenario)
