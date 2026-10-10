"""Merge several injection-eval runs (same dataset samples) into one comparison table.

Each run directory holds <dataset>_auroc.csv and config.json from run_injection_eval.py.
Runs that used the same --n-clips and --seed saw identical clips and, for the index-based
injections, identical perturbations, so their columns are directly comparable.

    python combine_results.py results/vicreg3_vs_raw results/rawpix_vs_pixel --out results/combined.md
    python combine_results.py results/a results/b --readouts raw vicreg3 rawpix --out results/combined.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HEADLINE = {  # (predictor score, geometry score); kept in step with run_injection_eval.HEADLINE
    "reverse": ("pe_mean", "cv_mean"), "swap": ("pe_max", "jump_spike"),
    "shuffle": ("pe_max", "jump_spike"), "drop": ("pe_max", "jump_spike"),
    "splice_other": ("pe_max", "chord_dev"), "splice_same": ("pe_max", "chord_dev"),
    "freeze": ("pe_max", "stall"), "speed_up": ("pe_mean", "step_cv"),
    "slow_down": ("pe_mean", "step_cv"), "appearance": ("pe_mean", "jump_max"),
    "blur": ("pe_mean", "jump_max"), "null_split": ("pe_mean", "jump_max"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--readouts", nargs="+", help="keep only these readouts, in this order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames, skills = [], {}
    sampling = None
    seen = {}
    for run in map(Path, args.runs):
        config = json.loads((run / "config.json").read_text())
        run_sampling = tuple(config["args"].get(k) for k in ("seed", "n_clips"))
        if sampling is None:
            sampling = run_sampling
        elif run_sampling != sampling:
            raise SystemExit(f"sampling mismatch in {run}: expected seed/n_clips={sampling}, got {run_sampling}")
        for csv in sorted(run.glob("*_auroc.csv")):
            frame = pd.read_csv(csv)
            for dataset, readout in frame[["dataset", "readout"]].drop_duplicates().itertuples(index=False, name=None):
                key = (dataset, readout)
                if key in seen:
                    raise SystemExit(f"duplicate dataset/readout {key} in {seen[key]} and {run}; give the runs distinct readout names")
                seen[key] = run
            frames.append(frame.assign(run=run.name))
        meta = config["meta"]
        for ds, m in meta.items():
            for rn, v in m.get("predictor_skill", {}).items():
                skills[(ds, rn)] = v["skill"]
    df = pd.concat(frames, ignore_index=True)
    lines = []
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        names = args.readouts or list(dict.fromkeys(d.readout))
        names = [n for n in names if n in set(d.readout)]
        cols = []
        for rn in names:
            cols.append((rn, "geom"))
            if (ds, rn) in skills:
                cols.append((rn, "pred"))
        lines += [f"### {ds}: AUROC of perturbed vs clean (0.5 = chance)", "",
                  "| injection | " + " | ".join(f"{rn} {'predictor' if k == 'pred' else 'geometry'}" for rn, k in cols) + " |",
                  "|" + "---|" * (1 + len(cols))]
        for inj, (ps, gs) in HEADLINE.items():
            cells = []
            for rn, k in cols:
                sub = d[(d.readout == rn) & (d.injection == inj) & (d.score == (ps if k == "pred" else gs))]
                cells.append(f"{sub.auroc.iloc[0]:.2f}" if len(sub) else "-")
            if any(c != "-" for c in cells):
                lines.append(f"| **{inj}** | " + " | ".join(cells) + " |")
        lines.append("| *predictor skill* | " + " | ".join(
            f"{skills[(ds, rn)]:.2f}" if k == "pred" else "" for rn, k in cols) + " |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
