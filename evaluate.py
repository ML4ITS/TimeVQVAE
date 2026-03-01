from pathlib import Path
import runpy
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

runpy.run_module("timevqvae.evaluate", run_name="__main__")
