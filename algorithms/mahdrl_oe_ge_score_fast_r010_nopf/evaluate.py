"""Evaluate one frozen MAHDRL-AS-Fast-R010-NoPF model on preset or custom resource counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.scenario import parse_scenario
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.runtime import evaluate

if __name__ == "__main__":
    evaluate(parse_scenario("eval"))
