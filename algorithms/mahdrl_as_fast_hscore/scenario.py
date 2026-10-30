"""Reuse parsing and short paths without changing the VAware implementation."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware.scenario import parse_scenario as base_parse
from .config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG,
                     VM_CAPACITY, variant_metadata, implementation_metadata)


@dataclass(frozen=True)
class ScoreScenario(BaseScenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG

    def __post_init__(self):
        super().__post_init__()
        if max(self.vms_per_host) > VM_CAPACITY:
            raise ValueError(f"Fixed VA capacity is {VM_CAPACITY} VMs per host; topology exceeds capacity")

    def variant_metadata(self):
        return variant_metadata(self.mechanism)

    def implementation_metadata(self):
        return implementation_metadata(self.mechanism)

    def evaluation_identity(self):
        return dict(super().evaluation_identity(), vm_capacity=VM_CAPACITY,
            vm_input_dim=93, vm_selection_scope="host_local_fixed_slots")


def parse_scenario(mode, argv=None):
    return base_parse(mode, argv, scenario_class=ScoreScenario)
