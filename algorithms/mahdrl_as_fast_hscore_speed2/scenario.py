"""Original protocol and parser, with separate Speed2 paths and checkpoints."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_hscore_speed.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_hscore_speed.scenario import parse_scenario as base_parse
from .config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG
from .config import variant_metadata, implementation_metadata


@dataclass(frozen=True)
class ScoreScenario(BaseScenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG

    def variant_metadata(self):
        return variant_metadata(self.mechanism)

    def implementation_metadata(self):
        return implementation_metadata(self.mechanism)


def parse_scenario(mode, argv=None):
    return base_parse(mode, argv, scenario_class=ScoreScenario)
