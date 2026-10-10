"""Model A — original representation, no representation learning: linear probe directly
on raw CrossTask (or synthetic) features. Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.train_progress_probe import train_progress_probe  # noqa: E402
from utils.config import load_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_progress_probe(cfg, args.config, args.runs_dir)
