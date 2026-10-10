"""Run eval_planning.py over a batch of encoders declared in one JSON spec, in parallel,
then collect every one's summary into a single comparison CSV -- same pattern as
../progress_metrics/run_progress_metrics_batch.py.

Each experiment is run as its own `eval_planning.py` subprocess (real OS-level
parallelism, no shared in-process CUDA/torch state between experiments).

Spec format (see planning_experiments.example.json):
{
  "defaults": {                                    // merged into every experiment below
    "manifest": ".../ssv2_validation.csv",
    "device": "cuda"
  },
  "experiments": [
    {
      "name": "dinov2_vits14",                      // required, unique -- CSV row label
      "features": ".../dinov2/ssv2_validation.pt",
      "predictor_checkpoint": "checkpoints/dinov2_vits14/best.ckpt"  // omit for interp-only
    }
  ]
}

Recognized per-experiment keys (all but "name" may instead live in "defaults"): features,
manifest, path_col, predictor_checkpoint, batch_size, device.

Usage:
    python run_planning_batch.py --spec planning_experiments.example.json \\
        --csv-out results/planning_summary.csv --workers 4
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parent / "eval_planning.py"
CLI_KEYS = ["features", "manifest", "path_col", "source_checkpoint", "source_config",
            "predictor_checkpoint", "batch_size", "device"]


def build_args(exp: dict) -> list:
    if not exp.get("features") or not exp.get("manifest"):
        raise ValueError(f"experiment {exp['name']!r} is missing \"features\" or \"manifest\" "
                          f"(set it per-experiment or in \"defaults\")")
    argv = []
    for key in CLI_KEYS:
        if key in exp and exp[key] is not None:
            argv += [f"--{key.replace('_', '-')}", str(exp[key])]
    argv += ["--out", exp["out"]]
    return argv


def run_one(exp: dict, results_dir: Path) -> dict:
    name = exp["name"]
    exp.setdefault("out", str(results_dir / f"{name}.json"))
    log_path = results_dir / f"{name}.log"

    row = {"name": name, "mode": "trained" if exp.get("predictor_checkpoint") else "interp_only",
           "features": exp.get("features"), "predictor_checkpoint": exp.get("predictor_checkpoint")}
    try:
        argv = build_args(exp)
    except ValueError as e:
        row["status"] = "bad_spec"
        row["error"] = str(e)
        return row

    proc = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True)
    log_path.write_text(f"$ {' '.join(argv)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")

    if proc.returncode != 0:
        row["status"] = "failed"
        row["error"] = f"exit {proc.returncode}, see {log_path}"
        return row

    with open(exp["out"]) as f:
        result = json.load(f)
    row["status"] = "ok"
    row["n_clips"] = result["n_clips"]
    row["n_examples"] = result["n_examples"]
    for k, v in result["interpolation"].items():
        row[f"interp_{k}"] = v
    if "trained" in result:
        for k, v in result["trained"].items():
            row[f"trained_{k}"] = v
        row["gap_closed_by_training"] = result["gap_closed_by_training"]
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="JSON file listing experiments (see module docstring)")
    parser.add_argument("--results-dir", default="results/planning_batch",
                         help="where each experiment's own --out json/.log lands, unless it sets its own \"out\"")
    parser.add_argument("--csv-out", required=True, help="combined comparison CSV, one row per experiment")
    parser.add_argument("--workers", type=int, default=4, help="experiments to run concurrently")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    defaults = spec.get("defaults", {})
    experiments = spec["experiments"]

    names = [e["name"] for e in experiments]
    if len(names) != len(set(names)):
        raise SystemExit(f"experiment names must be unique, got: {names}")

    merged = [{**defaults, **e} for e in experiments]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows_by_name = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, exp, results_dir): exp["name"] for exp in merged}
        for fut in as_completed(futures):
            name = futures[fut]
            row = fut.result()
            rows_by_name[name] = row
            status = row["status"]
            print(f"[{status}] {name}" + (f" -- {row['error']}" if status != "ok" else ""))

    # preserve the spec's own experiment order in the CSV, regardless of completion order
    rows = [rows_by_name[name] for name in names]
    df = pd.DataFrame(rows)
    lead_cols = [c for c in ["name", "status", "mode", "n_clips", "n_examples", "features",
                              "predictor_checkpoint", "error"] if c in df.columns]
    df = df[lead_cols + [c for c in df.columns if c not in lead_cols]]

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"\n{n_ok}/{len(df)} experiments succeeded. Saved -> {args.csv_out}")
