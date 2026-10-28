from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithms.irws_r_fe_r010.runner import evaluate_main
if __name__=='__main__':evaluate_main()