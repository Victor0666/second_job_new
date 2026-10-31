"""Reuse source normalization, workload support, short paths and flat CSVs."""
from algorithms.mahdrl_as_fast_hscore.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware.scenario import parse_scenario as base_parse
from .config import Variant


class AblationScenario(BaseScenario):
    suffix = None

    @property
    def variant(self):
        return Variant(self.suffix)

    @property
    def algorithm(self):
        return self.variant.algorithm

    @property
    def display_name(self):
        return "MAHDRL-AS-Fast-HScore-" + self.suffix.upper()

    @property
    def model_version(self):
        return self.variant.model_version

    @property
    def output_tag(self):
        return self.variant.output_tag

    @property
    def checkpoint_key(self):
        return self.output_tag

    @property
    def learned_roles(self):
        return self.variant.roles

    def variant_metadata(self):
        return self.variant.metadata(self.mechanism)

    def implementation_metadata(self):
        return self.variant.implementation(self.mechanism)

    def selection_protocol(self):
        return dict(super().selection_protocol(), hscore_ablation=self.variant_metadata())


def parse_scenario(scene_class, mode, argv=None):
    return base_parse(mode, argv, scenario_class=scene_class)
