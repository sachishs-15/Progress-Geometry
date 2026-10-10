"""Online Progress Error (OPE) from GTCC (Donahue & Elhamifar, CVPR 2024, Sec. 4.4 / Eq. 12).

Predicted progress at frame i is the cumulative inter-frame distance of the latents up to i,
divided by G, the mean total path length over reference (train) videos. Ground-truth progress
gives every phase an equal share, rising linearly within it. OPE is the mean absolute
difference, averaged over videos.
"""
import numpy as np


def segments(labels):
    """[(label, start, end_inclusive)] for consecutive runs of equal labels."""
    labels = np.asarray(labels)
    cuts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
    return [(int(labels[s]), s, e - 1) for s, e in zip(cuts[:-1], cuts[1:])]


def gt_progress(labels):
    """(T,) ground-truth progress; mirrors GTCC utils/tensorops.py:get_trueprogress."""
    segs = segments(labels)
    share = 1.0 / len(segs)
    progress = np.zeros(len(labels))
    prev = 0.0
    for _, s, e in segs:
        progress[s:e + 1] = np.linspace(prev, prev + share, e - s + 1)
        prev += share
    return progress


def cum_path(z):
    """(T,) cumulative arc length of latents z (T, D); mirrors GTCC get_cum_matrix."""
    z = np.asarray(z, dtype=np.float64)
    steps = np.linalg.norm(np.diff(z, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def video_scores(z, gt, ref_len):
    """Per-video OPE against reference length, plus two diagnostics:
    oracle: normalized by the video's own total length (isolates path shape from the
            reference-length estimate);
    linear_time: error of the constant-speed prediction t/(T-1) (what a perfectly straight,
            constant-pace trajectory would score)."""
    path = cum_path(z)
    total = path[-1]
    if total == 0:
        return None
    t = np.linspace(0.0, 1.0, len(gt))
    return {
        "ope": float(np.mean(np.abs(gt - path / ref_len))),
        "ope_oracle": float(np.mean(np.abs(gt - path / total))),
        "ope_linear_time": float(np.mean(np.abs(gt - t))),
        "total_len": float(total),
    }


def reference_length(latents):
    """G: mean total path length over reference videos. Returns (mean, variance)."""
    totals = [cum_path(z)[-1] for z in latents]
    totals = [t for t in totals if np.isfinite(t) and t > 0]
    return float(np.mean(totals)), float(np.var(totals))
