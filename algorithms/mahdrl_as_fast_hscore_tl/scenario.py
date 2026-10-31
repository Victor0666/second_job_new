"""Reuse the two-learner parser and compact flat outputs without editing it."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import parse_scenario as base_parse
from .config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG, VM_CAPACITY,
                     VM_INPUT_DIM, variant_metadata, implementation_metadata)


@dataclass(frozen=True)
class ScoreScenario(BaseScenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG

    def __post_init__(self):
        super().__post_init__()
        if max(self.vms_per_host) > VM_CAPACITY:
            raise ValueError("HScore TL fixed VA capacity is 9 VMs per host")

    def variant_metadata(self):
        return variant_metadata()

    def implementation_metadata(self):
        return implementation_metadata()

    def evaluation_identity(self):
        return dict(super().evaluation_identity(), vm_capacity=VM_CAPACITY,
            vm_network_input_dim=VM_INPUT_DIM, vm_network_output_dim=VM_CAPACITY)


def parse_scenario(mode, argv=None):
    return base_parse(mode, argv, scenario_class=ScoreScenario)
