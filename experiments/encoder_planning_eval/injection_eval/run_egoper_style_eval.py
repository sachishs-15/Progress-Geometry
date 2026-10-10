"""Held-out synthetic-error evaluation using EgoPER-style segment and temporal metrics.

This is not an EgoPER benchmark run: K400/SSv2 clips contain no real mistake labels.
Calibration uses clean source videos only. All edits of a source video remain in its split.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from injections import INJECTIONS, PIXEL_ONLY, REFERENCE, SEAM_INJECTIONS
from run_injection_eval import (DATASETS, DEFAULT_MODELS, HEADLINE, ModelReadout,
                                PixelBaselineReadout, RawReadout, load_sample,
                                load_sample_pixels, load_world_model, run_dataset)
from scores import compare, localization

HERE = Path(__file__).resolve().parent
LOCALIZED = ("splice_other", "splice_same", "freeze")
NEGATIVE_CONTROLS = {"appearance", "blur"}


def split_by_source(labels, seed, calibration_fraction=0.2):
    """Stratify source clips; retain two test clips per class for same-class donors."""
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(np.asarray(labels) == label)
        idx = rng.permutation(idx)
        n_cal = min(max(1, int(round(len(idx) * calibration_fraction))), max(0, len(idx) - 2))
        cal.extend(idx[:n_cal].tolist())
        test.extend(idx[n_cal:].tolist())
    return np.sort(cal), np.sort(test)


def load_exact_sample(cfg, n, seed):
    """Sample exactly n feature clips, retaining three per class when possible."""
    data = torch.load(cfg["features"], mmap=True, weights_only=False)
    manifest = pd.read_csv(cfg["manifest"], dtype=str)
    label_of = dict(zip(manifest["video_path"], manifest[cfg["label_col"]]))
    paths = data["video_paths"]
    groups = {}
    for i, path in enumerate(paths):
        if path in label_of:
            groups.setdefault(label_of[path], []).append(i)
    if n > sum(map(len, groups.values())):
        raise ValueError(f"requested {n} clips but only {sum(map(len, groups.values()))} are labeled")
    if any(len(indices) < 3 for indices in groups.values()) or n < 3 * len(groups):
        raise ValueError("exact stratified sample needs at least three clips per class")
    rng = np.random.default_rng(seed)
    selected, remaining = [], []
    for label in sorted(groups):
        shuffled = rng.permutation(groups[label])
        selected.extend(shuffled[:3].tolist())
        remaining.extend(shuffled[3:].tolist())
    selected.extend(rng.choice(remaining, n - len(selected), replace=False).tolist())
    indices = np.sort(np.asarray(selected, dtype=int))
    return data["features"][torch.as_tensor(indices)].float(), [label_of[paths[i]] for i in indices], [paths[i] for i in indices]


def split_by_source_exact(labels, seed, n_cal):
    """Exactly n_cal clean sources; keep at least two test sources per class."""
    rng = np.random.default_rng(seed)
    groups = {label: rng.permutation(np.flatnonzero(np.asarray(labels) == label))
              for label in sorted(set(labels))}
    if n_cal < len(groups) or n_cal > len(labels) - 2 * len(groups):
        raise ValueError("calibration count cannot retain one calibration and two test clips per class")
    cal = [int(indices[0]) for indices in groups.values()]
    candidates = [int(i) for indices in groups.values() for i in indices[1:-2]]
    cal.extend(rng.choice(candidates, n_cal - len(cal), replace=False).tolist())
    cal = np.sort(np.asarray(cal, dtype=int))
    test = np.setdiff1d(np.arange(len(labels)), cal, assume_unique=True)
    return cal, test


def subset(sample, idx):
    x, labels, paths = sample
    return x[torch.as_tensor(idx, dtype=torch.long)], [labels[i] for i in idx], [paths[i] for i in idx]


def altered_frames(injection, seams, n_frames):
    """Frame labels for edits whose altered span is unambiguous."""
    mask = np.zeros(n_frames, dtype=bool)
    if injection.startswith("splice_"):
        start = seams[0] + 1
        mask[start:start + 4] = True
    elif injection == "freeze":
        end = seams[0]
        mask[end - 2:end + 1] = True
    else:
        raise ValueError(f"no altered-frame labels defined for {injection}")
    return mask


def frame_metrics(pert_signal, clean_signal, seams, valid):
    """Map transition t to target frame t+1, then score labeled frames plus clean clips."""
    scores, targets = [], []
    for p, c, s, ok in zip(pert_signal, clean_signal, seams, valid):
        if not ok or not (np.isfinite(p).all() and np.isfinite(c).all()):
            continue
        frame_mask = altered_frames("splice_other" if len(s) == 2 else "freeze", s, len(p) + 1)
        scores.extend((p, c))
        targets.extend((frame_mask[1:], np.zeros(len(c), dtype=bool)))
    if not scores:
        return {"frame_auc": np.nan, "frame_ap": np.nan, "n_videos": 0, "positive_frames": 0}
    y = np.concatenate(targets)
    s = np.concatenate(scores)
    return {"frame_auc": float(roc_auc_score(y, s)),
            "frame_ap": float(average_precision_score(y, s)),
            "n_videos": len(scores) // 2, "positive_frames": int(y.sum())}


def evaluate(name, calibration, test, readouts, injections, n_boot, seed):
    segment_rows, frame_rows, loc_rows = [], [], []
    for r in readouts:
        for inj in injections:
            ref = REFERENCE[inj]
            p = test[(r.name, inj)]
            c = test[(r.name, ref)]
            clean_cal = calibration[(r.name, ref)]
            variants = [("geometry", HEADLINE[inj][1])]
            if r.has_predictor:
                variants.append(("predictor", HEADLINE[inj][0]))
            for channel, score in variants:
                threshold = float(np.quantile(clean_cal[score][np.isfinite(clean_cal[score])], 0.95))
                valid = p["valid"] & np.isfinite(p[score]) & np.isfinite(c[score])
                pv, cv = p[score][valid], c[score][valid]
                stats = compare(pv, cv, n_boot=n_boot, seed=seed)
                tp, fn = int((pv > threshold).sum()), int((pv <= threshold).sum())
                fp, tn = int((cv > threshold).sum()), int((cv <= threshold).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else np.nan
                segment_rows.append({"dataset": name, "readout": r.name, "channel": channel,
                                     "injection": inj, "score": score, "threshold": threshold, **stats,
                                     "recall": recall, "fpr": fp / (fp + tn) if fp + tn else np.nan,
                                     "precision": precision, "f1": 2 * precision * recall / (precision + recall)
                                     if precision + recall else 0.0,
                                     "balanced_accuracy": 0.5 * (recall + tn / (tn + fp)) if tn + fp else np.nan,
                                     "tp": tp, "fn": fn, "fp": fp, "tn": tn})
            valid = p["valid"] & np.isfinite(p["step"]).all(axis=1) & np.isfinite(c["step"]).all(axis=1)
            if inj in LOCALIZED:
                frame_signals = (["step", "pe"] if r.has_predictor else ["step"])
                if inj == "freeze":
                    frame_signals = ["negative_step"] + (["pe"] if r.has_predictor else [])
                for frame_signal in frame_signals:
                    p_values = -p["step"] if frame_signal == "negative_step" else p[frame_signal]
                    c_values = -c["step"] if frame_signal == "negative_step" else c[frame_signal]
                    m = frame_metrics(p_values, c_values, p["seams"], valid)
                    frame_rows.append({"dataset": name, "readout": r.name, "injection": inj,
                                       "signal": frame_signal, **m})
            if inj in SEAM_INJECTIONS:
                seams = [s if ok else None for s, ok in zip(p["seams"], valid)]
                for sig in (["step", "tokstep", "pe"] if r.has_predictor else ["step", "tokstep"]):
                    loc_rows.append({"dataset": name, "readout": r.name, "injection": inj,
                                     "signal": sig, **localization(p[sig], seams)})
    return pd.DataFrame(segment_rows), pd.DataFrame(frame_rows), pd.DataFrame(loc_rows)


def markdown_table(df, columns):
    if df.empty:
        return "No rows."
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in df[columns].itertuples(index=False, name=None):
        cells = [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mark_negative_controls(segment):
    """Keep control AUROC/FPR but do not call a benign edit a true mistake."""
    segment = segment.copy()
    segment["role"] = np.where(segment.injection.isin(NEGATIVE_CONTROLS), "negative_control", "mistake")
    control = segment.role == "negative_control"
    segment["control_flag_rate"] = np.where(control, segment.recall, np.nan)
    segment.loc[control, ["recall", "precision", "f1", "balanced_accuracy", "tp", "fn"]] = np.nan
    return segment


def write_summary(out, dataset, n_cal, n_test, segment, frame, loc):
    lines = [f"# EgoPER-style synthetic evaluation: {dataset}", "",
             f"Calibration source clips: {n_cal}; held-out source clips: {n_test}.",
             "Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.",
             "Appearance and blur are benign controls; control flag rate measures false alarms on these edits.",
             "Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.",
             "", "## Segment detection", "", markdown_table(segment, ["readout", "channel", "injection", "role", "score", "n",
                 "auroc", "ci_lo", "ci_hi", "recall", "fpr", "f1", "control_flag_rate"]), "",
             "## Frame scores", "", markdown_table(frame, ["readout", "injection", "signal",
                 "frame_auc", "frame_ap", "n_videos", "positive_frames"]), "",
             "## Seam localization", "", markdown_table(loc, ["readout", "injection", "signal",
                 "hit", "chance", "n"]), ""]
    (out / "summary.md").write_text("\n".join(lines))


def rescore_saved(out):
    """Recreate metrics from saved per-clip signals without another model forward pass."""
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    seams = json.loads((out / "test_seams.json").read_text())
    records = {"calibration": {}, "test": {}}
    with np.load(out / "per_clip_scores.npz", allow_pickle=False) as saved:
        for key in saved.files:
            split, readout, condition, field = key.split("|", 3)
            records[split].setdefault((readout, condition), {})[field] = saved[key]
    for (readout, condition), record in records["test"].items():
        record["seams"] = seams.get(condition, [[] for _ in range(len(record["valid"]))])
    args = protocol["args"]
    baseline = next(name for name, _ in records["test"] if name not in protocol["models"])
    readouts = [SimpleNamespace(name=baseline, has_predictor=False)]
    readouts += [SimpleNamespace(name=name, has_predictor=protocol.get("predictor_enabled", True))
                 for name in protocol["models"]]
    segment, frame, loc = evaluate(args["dataset"], records["calibration"], records["test"],
                                    readouts, list(seams), args["n_boot"], args["seed"])
    segment = mark_negative_controls(segment)
    segment.to_csv(out / "segment_metrics.csv", index=False)
    frame.to_csv(out / "frame_metrics.csv", index=False)
    loc.to_csv(out / "localization.csv", index=False)
    write_summary(out, args["dataset"], protocol["calibration_clips"], protocol["test_clips"],
                  segment, frame, loc)
    print(f"rescored -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, default="ssv2")
    ap.add_argument("--model", action="append", metavar="NAME=RUN_DIR[@CKPT]",
                    help="default: vicreg3 best checkpoint")
    ap.add_argument("--n-clips", type=int, default=500)
    ap.add_argument("--calibration-clips", type=int,
                    help="use an exact-size feature sample and this many clean calibration sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--geometry-pooling", choices=("mean", "concat"), default="mean",
                    help="pool encoder patch tokens for geometry; predictor scores are unchanged")
    ap.add_argument("--skip-predictor", action="store_true",
                    help="score only encoder geometry; useful for pooling comparisons")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--injections", nargs="+", choices=list(INJECTIONS))
    ap.add_argument("--frame-cache", default=str(HERE / "results" / "frame_cache"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "egoper_style_ssv2"))
    ap.add_argument("--rescore", type=Path, help="rebuild metric tables from an existing output directory")
    args = ap.parse_args()
    if args.rescore:
        rescore_saved(args.rescore)
        return
    if args.n_clips < 20 or args.batch_size < 1 or args.n_boot < 1:
        ap.error("n-clips must be >=20, batch-size >=1, and n-boot >=1")
    device = torch.device(args.device)
    specs = []
    for item in args.model or []:
        nm, rest = item.split("=", 1)
        directory, _, ckpt = rest.partition("@")
        specs.append((nm, Path(directory), ckpt or "best.ckpt"))
    specs = specs or DEFAULT_MODELS[:1]
    readouts, model_meta = [], {}
    for nm, directory, ckpt in specs:
        model, cfg, epoch = load_world_model(directory / ckpt, directory / "config.yaml", device,
                                             with_predictor=True)
        readouts.append(ModelReadout(nm, model, cfg, geometry_pooling=args.geometry_pooling,
                                     compute_predictor=not args.skip_predictor))
        model_meta[nm] = {"checkpoint": str(directory / ckpt), "epoch": epoch}
    pixel_modes = {getattr(r.model, "pixel_encoder", None) is not None for r in readouts}
    if len(pixel_modes) != 1:
        raise SystemExit("feature and pixel models must run separately")
    pixels = True in pixel_modes
    injections = args.injections or [i for i in INJECTIONS if pixels or i not in PIXEL_ONLY]
    if not pixels and set(injections) & set(PIXEL_ONLY):
        raise SystemExit("blur needs a pixel-input model")
    readouts.insert(0, PixelBaselineReadout() if pixels else RawReadout())
    cfg = DATASETS[args.dataset]
    if args.calibration_clips is not None and pixels:
        ap.error("--calibration-clips exact sampling currently supports feature models only")
    sample = (load_sample_pixels(args.dataset, cfg, args.n_clips, args.seed, args.frame_cache)
              if pixels else load_exact_sample(cfg, args.n_clips, args.seed)
              if args.calibration_clips is not None else load_sample(cfg, args.n_clips, args.seed))
    cal_idx, test_idx = (split_by_source_exact(sample[1], args.seed, args.calibration_clips)
                         if args.calibration_clips is not None else split_by_source(sample[1], args.seed))
    cal_sample, test_sample = subset(sample, cal_idx), subset(sample, test_idx)
    del sample
    cal_args = argparse.Namespace(n_clips=len(cal_idx), seed=args.seed, batch_size=args.batch_size,
                                   injections=[])
    test_args = argparse.Namespace(n_clips=len(test_idx), seed=args.seed, batch_size=args.batch_size,
                                    injections=injections)
    calibration, _, _ = run_dataset(args.dataset + "/cal", cfg, readouts, cal_args, device,
                                     pixels, sample=cal_sample)
    test, _, _ = run_dataset(args.dataset + "/test", cfg, readouts, test_args, device,
                             pixels, sample=test_sample)
    segment, frame, loc = evaluate(args.dataset, calibration, test, readouts, injections,
                                    args.n_boot, args.seed)
    segment = mark_negative_controls(segment)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment.to_csv(out / "segment_metrics.csv", index=False)
    frame.to_csv(out / "frame_metrics.csv", index=False)
    loc.to_csv(out / "localization.csv", index=False)
    arrays = {}
    for split_name, records in (("calibration", calibration), ("test", test)):
        for (readout, condition), record in records.items():
            for key, value in record.items():
                if key != "seams":
                    arrays[f"{split_name}|{readout}|{condition}|{key}"] = value
    np.savez_compressed(out / "per_clip_scores.npz", **arrays)
    (out / "test_seams.json").write_text(json.dumps(
        {inj: test[(readouts[0].name, inj)]["seams"] for inj in injections}))
    (out / "protocol.json").write_text(json.dumps({"args": vars(args), "models": model_meta,
        "predictor_enabled": not args.skip_predictor,
        "calibration_paths": cal_sample[2], "test_paths": test_sample[2],
        "calibration_clips": len(cal_idx), "test_clips": len(test_idx),
        "threshold_rule": "95th percentile of clean calibration scores per readout/score/reference",
        "segment_prevalence": "one edited and one clean clip per valid source video",
        "frame_label_rule": "transition t scores frame t+1; positive when that target frame was replaced or held",
        "scope": "synthetic edits on K400/SSv2, not the EgoPER dataset"}, indent=2, default=str))
    write_summary(out, args.dataset, len(cal_idx), len(test_idx), segment, frame, loc)
    print(f"saved -> {out} ({len(cal_idx)} calibration, {len(test_idx)} test clips)")


if __name__ == "__main__":
    main()
