"""Train the independent two-stage GAT baseline on the frozen common instances."""
from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from algorithms.two_stage_gat.runner import train_main

if __name__ == '__main__':
    train_main()
