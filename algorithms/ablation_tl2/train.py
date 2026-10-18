"""TL_2: TA + global VA, with global energy reward."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scenarios import parse_scenario
from algorithms.ablation_nohost.train import train
from environments.ablation_tl2 import CloudWorkflowEnv_VMAgents

if __name__ == "__main__":
    train(parse_scenario("train", "ablation_tl2"), algorithm="ablation_tl2",
          env_cls=CloudWorkflowEnv_VMAgents)
