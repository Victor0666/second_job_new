"""Independent short paths; inherit Speed2's split, scales and seed cycles."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_hscore_speed.scenario import parse_scenario as base_parse
from .config import CONFIG, ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG, CHECKPOINT_METADATA_KEY
from .config import variant_metadata, implementation_metadata


@dataclass(frozen=True)
class ScoreScenario(BaseScenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG
    checkpoint_key = CHECKPOINT_METADATA_KEY
    variant = CONFIG
    learned_roles = CONFIG.roles

    def variant_metadata(self):
        return variant_metadata(self.mechanism)

    def implementation_metadata(self):
        return implementation_metadata(self.mechanism)

    def selection_protocol(self):
        return dict(super().selection_protocol(), hscore_ablation=self.variant_metadata())


def parse_scenario(mode, argv=None):
    return base_parse(mode, argv, scenario_class=ScoreScenario)
