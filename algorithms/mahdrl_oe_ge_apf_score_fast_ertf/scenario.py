"""Dedicated ERTF entry point and checkpoint/CSV identity."""
from algorithms.mahdrl_oe_ge_apf_score_fast_fixed_ta.scenario import FixedTAScenario, parse_scenario as _parse


class ScoreScenario(FixedTAScenario):
    ordering_rule = "ertf"


def parse_scenario(mode, argv=None):
    return _parse(ScoreScenario, mode, argv)
