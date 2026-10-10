"""Run compute_progress_metrics.py over a batch of experiments (raw baselines and/or
trained checkpoints, any mix) declared in one JSON spec, in parallel, then collect every
experiment's summary_median into a single comparison CSV.

Each experiment is run as its own `compute_progress_metrics.py` subprocess -- not
imported and looped in-process -- so a bad checkpoint/config in one experiment can't
corrupt torch/CUDA state for the others, and "parallel" is real OS-level process
parallelism rather than something threaded around one shared model.

Spec format (see progress_metrics_experiments.example.json):
{
  "defaults": {                                 // merged into every experiment below;
    "manifest": ".../ssv2_validation.csv",       // any key here can be overridden per-experiment
    "device": "cuda"
  },
  "experiments": [
    {
      "name": "raw_cls_dinov2",                 // required, unique -- used for the CSV row
                                                  // and, if "out" is omitted, the result
                                                  // filename (<results-dir>/<name>.json)
      "raw": "cls",                              // "cls" | "mean" | "flatten"  (raw mode)
      "features": ".../dinov2/ssv2_validation.pt"
    },
    {
      "name": "pwm_cls_mlp_k400",                // (checkpoint mode instead -- path reaches
      "checkpoint": "../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/best.ckpt",
      "config": "../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/config.yaml",
      "features": ".../dinov2/ssv2_validation.pt"
    }
  ]
}

Recognized per-experiment keys (all but "name" and exactly one of "raw" /
"checkpoint"+"config" may instead live in "defaults"): raw, checkpoint, config, features,
manifest, path_col, batch_size, device, out.

Usage:
    python run_progress_metrics_batch.py --spec progress_metrics_experiments.example.json \\
        --csv-out results/progress_metrics_summary.csv --workers 4
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parent / "compute_progress_metrics.py"

# name/out are handled separately (labeling + output path); everything else maps
# straight onto compute_progress_metrics.py's own CLI flags.
CLI_KEYS = ["raw", "checkpoint", "config", "features", "manifest", "path_col", "batch_size", "device"]


def build_args(exp: dict) -> list:
    # "checkpoint" alone (no "config") is valid too -- a LiFT checkpoint, per
    # compute_progress_metrics.py's own --checkpoint/--config validation, which this
    # mirrors.
    missing_mode = exp.get("raw") is None and not exp.get("checkpoint")
    both_modes = exp.get("raw") is not None and (exp.get("checkpoint") or exp.get("config"))
    if missing_mode or both_modes:
        raise ValueError(f"experiment {exp['name']!r} needs exactly one of \"raw\", or "
                          f"\"checkpoint\" (optionally with \"config\" for a ProceduralWorldModel "
                          f"checkpoint; omit \"config\" for a LiFT checkpoint)")
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

    row = {"name": name, "mode": f"raw:{exp['raw']}" if exp.get("raw") else "checkpoint",
           "features": exp.get("features"), "checkpoint": exp.get("checkpoint")}
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
    row.update(result["summary_median"])
    for k in ("effective_rank", "effective_rank_fraction", "dead_dim_fraction", "mean_pairwise_l2_across_clips"):
        if k in result:
            row[k] = result[k]
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="JSON file listing experiments (see module docstring)")
    parser.add_argument("--results-dir", default="results/progress_metrics_batch",
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
    lead_cols = [c for c in ["name", "status", "mode", "n_clips", "features", "checkpoint", "error"] if c in df.columns]
    df = df[lead_cols + [c for c in df.columns if c not in lead_cols]]

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"\n{n_ok}/{len(df)} experiments succeeded. Saved -> {args.csv_out}")
