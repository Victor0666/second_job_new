"""Evaluate one frozen MAHDRL-AS-Fast model on preset or custom resource counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_vaaware_tl.scenario import parse_scenario
from algorithms.mahdrl_as_fast_vaaware_tl.runtime import evaluate

if __name__ == "__main__":
    evaluate(parse_scenario("eval"))
