from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.strengthening import run_suite
run_suite()
print('All strengthening experiments completed and validated.')
