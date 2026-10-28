"""Use the unchanged FCFS reference generator for custom evaluation topologies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.precompute_deadlines import main
from algorithms.mahdrl_oe_ge_score_fast_r010_nopf.scenario import parse_scenario

if __name__ == "__main__":
    main(parse_scenario("precompute"))
