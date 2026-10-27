"""Evaluate frozen TA/VA with global learned VM selection and per-host VM APF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.scenario import parse_scenario
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.runtime import evaluate

if __name__ == "__main__":
    evaluate(parse_scenario("eval"))
