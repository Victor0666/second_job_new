"""Keep the original parser/protocol and isolate all models and outputs."""
from dataclasses import dataclass
from inspect import signature
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware.scenario import parse_scenario as base_parse
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


def parse_scenario(mode, argv=None, *, scenario_class=ScoreScenario):
    options = dict(scenario_class=scenario_class)
    # Bursty support may be present locally before its shared parser is published.
    if 'allow_bursty' in signature(base_parse).parameters:
        options['allow_bursty'] = True
    return base_parse(mode, argv, **options)
