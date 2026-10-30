"""Fixed R100 TL with the same R50 references as full VAware-R100."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_vaaware.resources import ResourceReferences
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import ScoreScenario as TLScenario, parse_scenario as parse_base
from .config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG, VMS_PER_HOST


@dataclass(frozen=True)
class ScoreScenario(ResourceReferences, TLScenario):
    resource: str = "R100"
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG
    physical_vms = VMS_PER_HOST


def parse_scenario(mode, argv=None):
    return parse_base(mode, argv, scenario_class=ScoreScenario,
                      resource_choices=("R100",), resource_default="R100")
