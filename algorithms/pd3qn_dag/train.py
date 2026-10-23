"""Train the independent PD3QN-DAG baseline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.pd3qn import train as implementation
from common.scenarios import parse_scenario

if __name__ == '__main__':
    implementation.SCENARIO = parse_scenario('train', 'pd3qn_dag')
    implementation.train()
