"""TL_2 evaluation using its own checkpoints and the shared TL runner."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scenarios import parse_scenario
from algorithms.ablation_nohost.evaluate import main
from environments.ablation_tl2 import CloudWorkflowEnv_VMAgents

if __name__ == "__main__":
    main(parse_scenario("eval", "ablation_tl2"), algorithm="ablation_tl2",
         env_cls=CloudWorkflowEnv_VMAgents)
