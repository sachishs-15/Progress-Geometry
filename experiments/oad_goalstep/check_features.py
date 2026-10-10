"""Compare every feature file's length against its target and index.csv row.

  python check_features.py --root <oad dir> --features <dense feature dir>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main(args):
    bad = missing = 0
    rows = list(csv.DictReader(open(Path(args.root) / "index.csv")))
    for r in rows:
        f = Path(args.features) / f"{r['video_uid']}.npy"
        if not f.exists():
            missing += 1
            print("MISSING", r["video_uid"])
            continue
        L = np.load(f, mmap_mode="r").shape[0]
        T = np.load(Path(args.root) / "target_perframe" / f"{r['video_uid']}.npy", mmap_mode="r").shape[0]
        if not (L == T == int(r["L"])):
            bad += 1
            print("MISMATCH", r["video_uid"], "features", L, "targets", T, "expected", r["L"])
    print(f"{len(rows)} videos: {missing} missing, {bad} length mismatches")
    sys.exit(1 if missing or bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--features", required=True)
    main(p.parse_args())
