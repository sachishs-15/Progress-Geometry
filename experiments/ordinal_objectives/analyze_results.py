"""Turn run_progress_metrics_batch.py's per-clip output into a comparison that can
actually support a ranking.

Why this exists rather than just reading the summary CSV. That CSV reports the MEDIAN of
each metric across clips, matching P-JEPA's Table 6 convention -- but `time_progress_
spearman` is a Spearman correlation over T=16 points, which can only take the discrete
values 1 - k/680. Its median across clips therefore lands on one of those values too, so
the whole table is quantized to steps of 2/680 = 0.0029 and objectives that differ by less
than that are reported as identical. Measured on the real run: 32 rows collapsed onto 14
distinct medians, with consecutive gaps of exactly 0.00294.

So this script works from `per_clip` instead:

  * means with standard errors over all N clips, which are not quantized;
  * a PAIRED comparison against a reference row -- every run is scored on the same clips
    in the same order, so the per-clip difference removes clip difficulty entirely and is
    far more powerful than comparing two marginal medians;
  * Wilcoxon signed-rank on those differences (the per-clip metric is bounded and
    discrete, so a t-test's normality assumption is not free), Holm-corrected across the
    rows being compared;
  * a collapse flag, because `time_progress_spearman` is scale-invariant by construction
    -- it L2-normalizes each frame and projects onto the clip's own chord -- so a
    representation that has contracted to nearly a point still scores well on it. That is
    not a hypothetical: several objectives here do exactly that. start_end_displacement
    and adjacent_cosine_distance are what reveal it.

Usage:
    python analyze_results.py --results ../encoder_planning_eval/progress_metrics/results/progress_metrics_batch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def load(results_dir: Path, metric: str):
    runs = {}
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        pc = d.get("per_clip")
        if not pc:
            continue
        runs[f.stem] = {
            "metric": np.array([c[metric] for c in pc], dtype=float),
            "disp": np.array([c["start_end_displacement"] for c in pc], dtype=float),
            "adj": np.array([c["adjacent_cosine_distance"] for c in pc], dtype=float),
            "n": len(pc),
        }
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="directory of per-row .json files")
    p.add_argument("--metric", default="time_progress_spearman")
    p.add_argument("--reference", default="baseline_raw_dinov2",
                   help="row every other row is paired against")
    p.add_argument("--collapse-frac", type=float, default=0.5,
                   help="flag a run whose median start_end_displacement falls below this "
                        "fraction of the reference's")
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    runs = load(Path(args.results), args.metric)
    if args.reference not in runs:
        raise SystemExit(f"reference {args.reference!r} not among {sorted(runs)}")
    ref = runs[args.reference]
    ref_disp = float(np.median(ref["disp"]))

    rows, pending = [], []
    for name, r in sorted(runs.items()):
        n = r["metric"].size
        mean = r["metric"].mean()
        sem = r["metric"].std(ddof=1) / np.sqrt(n)
        row = {"name": name, "n": n, "mean": mean, "sem": sem,
               "median_disp": float(np.median(r["disp"])),
               "median_adj_cos": float(np.median(r["adj"]))}
        row["collapsed"] = row["median_disp"] < args.collapse_frac * ref_disp
        if name != args.reference and n == ref["n"]:
            diff = r["metric"] - ref["metric"]
            row["delta_vs_ref"] = diff.mean()
            row["delta_sem"] = diff.std(ddof=1) / np.sqrt(n)
            # All-zero differences mean the two rows are the same encoder; wilcoxon
            # raises on that rather than returning p=1.
            row["_p"] = 1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue
            pending.append(len(rows))
        rows.append(row)

    df = pd.DataFrame(rows)
    if pending:
        idx = df.index[pending]
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "_p"].values)
    df = df.drop(columns=["_p"], errors="ignore").sort_values("mean", ascending=False)

    pd.set_option("display.width", 200)
    show = df.copy()
    show["mean±sem"] = [f"{m:.4f}±{s:.4f}" for m, s in zip(show["mean"], show["sem"])]
    show["Δ vs untrained"] = [
        "  (reference)" if pd.isna(d) else f"{d:+.4f}±{e:.4f}"
        for d, e in zip(show.get("delta_vs_ref", pd.Series(dtype=float)),
                        show.get("delta_sem", pd.Series(dtype=float)))]
    show["sig"] = ["" if pd.isna(q) else ("***" if q < 1e-3 else "**" if q < 1e-2
                   else "*" if q < 0.05 else "ns") for q in show.get("p_holm", pd.Series(dtype=float))]
    show["collapse"] = np.where(show["collapsed"], "COLLAPSED", "")
    print(show[["name", "mean±sem", "Δ vs untrained", "sig", "median_disp",
                "median_adj_cos", "collapse"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nreference = {args.reference} (median start_end_displacement {ref_disp:.4f}); "
          f"n = {ref['n']} clips; Holm-corrected over {len(pending)} comparisons")
    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"saved -> {args.csv_out}")


if __name__ == "__main__":
    main()
