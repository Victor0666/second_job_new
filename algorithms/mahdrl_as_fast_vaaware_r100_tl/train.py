"""train for matched VAware-R100-TL."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_vaaware_tl.runtime import train
from algorithms.mahdrl_as_fast_vaaware_r100_tl.scenario import parse_scenario

if __name__ == "__main__":
    train(parse_scenario("train"))
