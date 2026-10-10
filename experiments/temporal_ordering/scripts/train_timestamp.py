"""Model B — timestamp-trained representation (the control): encoder trained to regress
normalized temporal position, then frozen and probed with the SAME probe as every other
condition. Usage:
    python scripts/train_timestamp.py --config configs/timestamp.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.train_progress_probe import train_progress_probe  # noqa: E402
from train.train_representation import train_representation  # noqa: E402
from utils.config import load_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/timestamp.yaml")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_representation(cfg, args.config, args.runs_dir)
    train_progress_probe(cfg, args.config, args.runs_dir)
