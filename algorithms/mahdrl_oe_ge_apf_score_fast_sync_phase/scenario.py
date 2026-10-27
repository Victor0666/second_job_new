"""Mode-specific paths/parser without modifying the shared algorithm registry."""
from algorithms.mahdrl_oe_ge_apf_score_fast_sync.scenario import SyncScenario, parse_scenario as _parse


class ScoreScenario(SyncScenario):
    sync_mode = 'phase'


def parse_scenario(mode, argv=None):
    return _parse(ScoreScenario, mode, argv)
