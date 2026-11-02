"""Train the independently isolated Speed2 implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_hscore_speed2.scenario import parse_scenario
from algorithms.mahdrl_as_fast_hscore_speed2.runtime import train

if __name__ == '__main__':
    train(parse_scenario('train'))
