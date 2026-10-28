"""Train the independent MAHDRL-AS-Fast-R010-NoPF variant."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.scenario import parse_scenario
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.runtime import train

if __name__ == "__main__":
    train(parse_scenario("train"))
