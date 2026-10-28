"""Precompute the fixed ERTF task-ordering ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_oe_ge_apf_score_fast_ertf.scenario import parse_scenario
from common.precompute_deadlines import main

if __name__ == "__main__":
    main(parse_scenario("precompute"))
