"""Replay saved clips causally with prefix jump-spike thresholds.

At frame t, only transitions among frames 1..t are used. The clean threshold for
that prefix length is the 95th percentile of calibration clips at that length.
Report both direct prefix crossings and a stream-calibrated first-alert rule:
the latter uses the 95th percentile of each calibration clip's maximum normalized
prefix score, accounting for repeated looks across one clip.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from injections import INJECTIONS, REFERENCE, rng_for
from scores import auroc


DEFAULT_RUNS = (
    ("DINOv2 CLS", "egoper_style_ssv2_dinov2_cls", "dinov2_cls"),
    ("DINOv2 patch mean", "egoper_style_ssv2_vicreg_clean_concat_epoch49", "raw"),
    ("DINOv2 patch concat", "egoper_style_ssv2_dinov2_patch_concat", "dinov2_patch_concat"),
    ("VICREG patch concat", "egoper_style_ssv2_vicreg_clean_concat_epoch49", "vicreg_clean"),
)


def prefix_jump_spike(steps, min_frames=5):
    """Return (N, T) scores, with NaN until enough frames have been observed."""
    steps = np.asarray(steps)
    out = np.full((len(steps), steps.shape[1] + 1), np.nan)
    for frames in range(min_frames, steps.shape[1] + 2):
        observed = steps[:, :frames - 1]
        out[:, frames - 1] = observed.max(axis=1) / (np.median(observed, axis=1) + 1e-8)
    return out


def first_crossing(ratio):
    mask = np.isfinite(ratio) & (ratio > 1)
    return np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, np.nan)


def first_altered_frames(edit, n, seed):
    """Replay source-index edits to locate the first changed output frame."""
    if edit in ("reverse", "appearance"):
        return np.ones(n, dtype=int)
    source = torch.arange(16, dtype=torch.float32)[:, None]
    donor = source + 100
    starts = np.empty(n, dtype=int)
    for i in range(n):
        edited, _ = INJECTIONS[edit](source, rng_for(seed, edit, i), donor)
        mismatch = (edited[:, 0] != source[:len(edited), 0]).numpy()
        starts[i] = int(np.flatnonzero(mismatch)[0]) + 1
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--min-frames", type=int, default=5)
    args = parser.parse_args()
    rows = []
    for label, folder, readout in DEFAULT_RUNS:
        protocol = json.loads((args.results / folder / "protocol.json").read_text())
        seed = protocol.get("seed", protocol.get("args", {}).get("seed"))
        if seed is None:
            raise ValueError(f"missing injection seed in {folder}/protocol.json")
        with np.load(args.results / folder / "per_clip_scores.npz") as saved:
            def get(split, cond, field):
                return saved[f"{split}|{readout}|{cond}|{field}"]

            for edit, reference in REFERENCE.items():
                if edit == "blur":
                    continue  # pixel-only injection, absent from these feature runs
                cal = prefix_jump_spike(get("calibration", reference, "step"), args.min_frames)
                clean = prefix_jump_spike(get("test", reference, "step"), args.min_frames)
                altered = prefix_jump_spike(get("test", edit, "step"), args.min_frames)
                valid = get("test", edit, "valid").astype(bool)
                thresholds = np.full(cal.shape[1], np.nan)
                for index in range(args.min_frames - 1, cal.shape[1]):
                    finite = cal[:, index][np.isfinite(cal[:, index])]
                    thresholds[index] = np.quantile(finite, 0.95)
                clean_ratio = clean[valid] / thresholds
                altered_ratio = altered[valid] / thresholds
                first_clean = first_crossing(clean_ratio)
                first_altered = first_crossing(altered_ratio)
                # Calibrate the *ever alerted* rule, accounting for repeated looks.
                cal_ratio = cal / thresholds
                stream_threshold = float(np.quantile(np.nanmax(cal_ratio, axis=1), 0.95))
                first_clean_calibrated = first_crossing(clean_ratio / stream_threshold)
                first_altered_calibrated = first_crossing(altered_ratio / stream_threshold)
                onset = first_altered_frames(edit, len(valid), seed)[valid]
                alert_mask = np.isfinite(altered_ratio) & (altered_ratio > stream_threshold)
                prefix_frames = np.arange(1, altered.shape[1] + 1)
                before = (alert_mask & (prefix_frames[None, :] < onset[:, None])).any(axis=1)
                after = (alert_mask & (prefix_frames[None, :] >= onset[:, None])).any(axis=1)
                rows.append({
                    "edit": edit, "representation": label, "n": int(valid.sum()),
                    "min_frames": args.min_frames, "clip_frames": altered.shape[1],
                    "ever_flag_recall": float(np.isfinite(first_altered).mean()),
                    "ever_flag_fpr": float(np.isfinite(first_clean).mean()),
                    "first_flag_frame_median": float(np.nanmedian(first_altered)) if np.isfinite(first_altered).any() else np.nan,
                    "first_flag_frame_mean": float(np.nanmean(first_altered)) if np.isfinite(first_altered).any() else np.nan,
                    "stream_threshold_ratio": stream_threshold,
                    "stream_flag_recall": float(np.isfinite(first_altered_calibrated).mean()),
                    "stream_flag_fpr": float(np.isfinite(first_clean_calibrated).mean()),
                    "stream_first_flag_frame_median": float(np.nanmedian(first_altered_calibrated)) if np.isfinite(first_altered_calibrated).any() else np.nan,
                    "pre_onset_flag_rate": float(before.mean()),
                    "post_onset_first_alert_rate": float((~before & after).mean()),
                    "max_prefix_ratio_auroc": auroc(np.nanmax(altered_ratio, axis=1),
                                                     np.nanmax(clean_ratio, axis=1)),
                    "final_prefix_auroc": auroc(altered[valid, -1], clean[valid, -1]),
                })
    out = args.results / "ssv2_live_jump_spike_all_edits.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
