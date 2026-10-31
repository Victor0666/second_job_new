"""Precompute matching physical deadlines for HScore cp."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_hscore_cp.scenario import parse_scenario
from common.precompute_deadlines import main as precompute

if __name__ == "__main__":
    precompute(parse_scenario("precompute"))
