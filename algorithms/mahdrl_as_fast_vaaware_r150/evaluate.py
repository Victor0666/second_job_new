"""Evaluate matched VAware-R150 checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_vaaware.runtime import evaluate
from algorithms.mahdrl_as_fast_vaaware_r150.scenario import parse_scenario

if __name__ == "__main__":
    evaluate(parse_scenario("eval"))
