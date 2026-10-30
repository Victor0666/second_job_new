"""Generate or validate the explicitly identified R50 deadline reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_vaaware.resources import precompute_reference
from algorithms.mahdrl_as_fast_vaaware_r150.scenario import parse_scenario

if __name__ == "__main__":
    precompute_reference(parse_scenario("precompute"))
