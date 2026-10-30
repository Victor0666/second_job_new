from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithms.irws_r_fe_idle_090.runner import train_main
if __name__=='__main__':train_main()