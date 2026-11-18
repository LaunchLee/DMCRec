"""Ablation entry point; shares every data/model option with main.py."""
from pathlib import Path
import sys

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dmcrec.main import parse_args, run


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--variants" not in arguments:
        arguments += ["--variants", "full", "woGCN", "woKNN", "woDynKNN", "woCL", "woMask",
                      "woText", "woDebias", "woMACR-A", "woMACR-B", "concat"]
    run(parse_args(arguments))
