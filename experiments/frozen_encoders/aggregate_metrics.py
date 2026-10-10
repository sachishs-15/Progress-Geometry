"""Aggregate every run's `<tag>_metrics.json` (written by run.py) into one
combined table — useful after a batch of parallel/background runs (e.g. the
PBS job in submit_all.pbs) where results are scattered across
models/*/outputs/ instead of living in one place.

Usage:
    python aggregate_metrics.py                          # print + save results_summary.csv
    python aggregate_metrics.py --out my_summary.csv
    python aggregate_metrics.py --sort curvature_deg --order asc
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_metrics(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("models/*/outputs/*_metrics.json"))
    rows = []
    for f in files:
        try:
            row = json.load(open(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"skipping {f}: {e}")
            continue
        row["metrics_file"] = str(f)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).parent), help="frozen_encoders/ dir to scan under")
    parser.add_argument("--out", default="results_summary.csv")
    parser.add_argument("--sort", default="ordinality_score",
                         choices=["ordinality_score", "curvature_deg", "model", "video", "timestamp"])
    parser.add_argument("--order", choices=["asc", "desc"], default=None,
                         help="default: desc for ordinality_score, asc for curvature_deg, asc otherwise")
    args = parser.parse_args()

    root = Path(args.root)
    df = load_all_metrics(root)
    if df.empty:
        print(f"No *_metrics.json files found under {root}/models/*/outputs/")
        raise SystemExit(0)

    if args.order:
        ascending = args.order == "asc"
    else:
        ascending = args.sort != "ordinality_score"
    df = df.sort_values(args.sort, ascending=ascending)

    out_path = root / args.out
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}\n")

    cols = ["model", "video", "stride", "max_frames", "pooling", "method",
            "num_frames", "ordinality_score", "curvature_deg"]
    print("=== All runs ===")
    print(df[cols].to_string(index=False))

    print("\n=== Per-model averages (across all videos/settings) ===")
    print(df.groupby("model")[["ordinality_score", "curvature_deg"]].agg(["mean", "std", "count"]).to_string())

    print("\n=== Per-video averages (across all models) ===")
    print(df.groupby("video")[["ordinality_score", "curvature_deg"]].agg(["mean", "count"]).to_string())
