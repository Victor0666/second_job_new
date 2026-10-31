"""train HScore r090."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_hscore_r090.scenario import parse_scenario
from algorithms.mahdrl_as_fast_hscore_ablation.runtime import train

if __name__ == "__main__":
    train(parse_scenario("train"))
