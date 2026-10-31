"""Short source paths and scenario-named flat evaluation CSVs."""
from algorithms.mahdrl_as_fast_hscore_ablation.scenario import AblationScenario, parse_scenario as base_parse

class ScoreScenario(AblationScenario):
    suffix = "sync_task"
    display_name = "MAHDRL-AS-Fast-HScore-SYNC_TASK"

def parse_scenario(mode, argv=None):
    return base_parse(ScoreScenario, mode, argv)
