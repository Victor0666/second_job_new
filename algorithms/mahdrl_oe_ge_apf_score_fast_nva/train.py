"""Train TA/HA with random local idle VM selection and no VM APF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_oe_ge_apf_score_fast_nva.scenario import parse_scenario
from algorithms.mahdrl_oe_ge_apf_score_fast_nva.runtime import train

if __name__ == "__main__":
    train(parse_scenario("train"))
