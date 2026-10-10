"""Compare all saved geometry scores in raw DINOv2 and model latent space.

Reads an existing run_egoper_style_eval.py output; no model inference is needed.
Pass the names of the baseline and model readouts for the run being compared.
"""
import argparse
import ast
import csv
import math
import struct
import zipfile
from pathlib import Path


SCORES = ("jump_max", "jump_spike", "cv_max", "cv_mean", "chord_dev",
          "backslide", "step_cv", "stall")
REFERENCE = {"drop": "clean12", "speed_up": "clean12"}


def load_array(archive, key):
    with archive.open(key + ".npy") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"invalid NPY array: {key}")
        version = f.read(2)
        header_size = struct.unpack("<H" if version[0] == 1 else "<I",
                                    f.read(2 if version[0] == 1 else 4))[0]
        header = ast.literal_eval(f.read(header_size).decode())
        if header["fortran_order"] or len(header["shape"]) != 1:
            raise ValueError(f"expected a one-dimensional C-order array: {key}")
        dtype = header["descr"]
        n = header["shape"][0]
        if dtype == "|b1":
            return [bool(x) for x in f.read(n)]
        if dtype not in ("<f8", "<f4"):
            raise ValueError(f"unsupported dtype {dtype}: {key}")
        fmt = "d" if dtype == "<f8" else "f"
        return struct.unpack("<" + fmt * n, f.read(n * struct.calcsize(fmt)))


def quantile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    low = int(x)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (x - low)


def auc(positive, negative):
    negative = sorted(negative)
    from bisect import bisect_left, bisect_right
    wins = sum((bisect_left(negative, p) + bisect_right(negative, p)) / 2
               for p in positive)
    return wins / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", default="raw")
    parser.add_argument("--model", default="vicreg3")
    args = parser.parse_args()
    run = args.run_dir
    with (run / "segment_metrics.csv").open(newline="") as f:
        existing = list(csv.DictReader(f))
    injections = list(dict.fromkeys(r["injection"] for r in existing))
    baseline, model = args.baseline, args.model
    if not {baseline, model} <= {r["readout"] for r in existing}:
        parser.error(f"run must have {baseline} and {model} readouts")
    rows = []
    with zipfile.ZipFile(run / "per_clip_scores.npz") as archive:
        for injection in injections:
            ref = REFERENCE.get(injection, "clean16")
            for score in SCORES:
                result = {"injection": injection, "score": score,
                          "role": "control" if injection == "appearance" else "alteration"}
                for readout, label in ((baseline, "baseline"), (model, "model")):
                    prefix = lambda split, condition, field: f"{split}|{readout}|{condition}|{field}"
                    cal = load_array(archive, prefix("calibration", ref, score))
                    pert = load_array(archive, prefix("test", injection, score))
                    clean = load_array(archive, prefix("test", ref, score))
                    valid = load_array(archive, prefix("test", injection, "valid"))
                    if not (len(pert) == len(clean) == len(valid)):
                        raise ValueError(f"unaligned arrays for {readout}/{injection}")
                    threshold = quantile([v for v in cal if math.isfinite(v)], .95)
                    pairs = [(p, c) for p, c, ok in zip(pert, clean, valid)
                             if ok and math.isfinite(p) and math.isfinite(c)]
                    pvals, cvals = zip(*pairs)
                    result[f"{label}_n"] = len(pvals)
                    result[f"{label}_auroc"] = auc(pvals, cvals)
                    result[f"{label}_flag_rate"] = sum(p > threshold for p in pvals) / len(pvals)
                    result[f"{label}_fpr"] = sum(c > threshold for c in cvals) / len(cvals)
                result["delta_auroc"] = result["model_auroc"] - result["baseline_auroc"]
                rows.append(result)
    columns = ("injection", "role", "score", "baseline_n", "baseline_auroc", "baseline_flag_rate",
               "baseline_fpr", "model_n", "model_auroc", "model_flag_rate",
               "model_fpr", "delta_auroc")
    out = run / "geometry_only_comparison.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = ["# Geometry-only comparison", "",
               f"Same held-out clips and injections as the parent run. Baseline: `{baseline}`; model: `{model}`.",
               "Both columns use the same geometry score. Predictor scores are excluded.",
               "Flag rates use each readout's 95th percentile of clean calibration clips.",
               "Appearance is a benign control, so its flag rate measures false alarms.",
               "", f"| Injection | Score | {baseline} AUROC | {model} geometry AUROC | Difference |",
               "|---|---|---:|---:|---:|"]
    for r in rows:
        summary.append(f"| {r['injection']} | {r['score']} | {r['baseline_auroc']:.3f} | "
                       f"{r['model_auroc']:.3f} | {r['delta_auroc']:+.3f} |")
    (run / "geometry_only_comparison.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {len(rows)} score comparisons to {out}")


if __name__ == "__main__":
    main()
