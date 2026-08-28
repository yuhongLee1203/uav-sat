"""Run the unmodified parent v36 trainer with the generated route config."""

import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PARENT))
runpy.run_path(str(PARENT / "train_multirate_a.py"), run_name="__main__")
