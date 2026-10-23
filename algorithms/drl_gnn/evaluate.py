"""Evaluate a frozen DRL-GNN checkpoint on paired test instances."""
from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from algorithms.drl_gnn.runner import evaluate_main

if __name__ == '__main__':
    evaluate_main()
