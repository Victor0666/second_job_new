"""Train the independent DRL-GNN-Fast variant."""
from pathlib import Path
import sys
root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)
from algorithms.drl_gnn_fast.runner import train_main
if __name__ == '__main__':
    train_main()
