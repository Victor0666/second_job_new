"""Evaluate Speed2 sync-task checkpoints under the same test protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task.scenario import parse_scenario
from algorithms.mahdrl_as_fast_hscore_speed2_sync_task.runtime import evaluate

if __name__ == '__main__':
    evaluate(parse_scenario('eval'))
