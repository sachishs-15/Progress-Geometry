"""Vectorised metrics must match the reference clip-by-clip implementation."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "progress_metrics"))
from metrics_batch import per_clip_metrics, dataset_metrics, summarize  # noqa: E402
from compute_progress_metrics import compute_metrics_dict  # noqa: E402

FEATS = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_validation.pt"


def check(name, n=None):
    z = torch.load(FEATS.format(name), map_location="cpu")["features"].float().numpy()
    if n:
        z = z[:n]
    t0 = time.time()
    ref = compute_metrics_dict(z)
    t_ref = time.time() - t0
    t0 = time.time()
    pc, keep = per_clip_metrics(z)
    ds = dataset_metrics(z)
    t_new = time.time() - t0
    worst = 0.0
    for how in ("mean", "median"):
        mine = summarize(pc, keep, how)
        theirs = ref[f"summary_{how}"]
        for k, v in theirs.items():
            err = abs(mine[k] - v)
            worst = max(worst, err)
            # the reference runs the NN search in float32; exact-duplicate frames tie-break differently (<=3e-4)
            tol = 5e-4 if k == "adjacent_temporal_nn_fraction" else 1e-5 * max(1.0, abs(v))
            assert err < tol, (name, how, k, mine[k], v)
    for k in ("effective_rank_fraction", "dead_dim_fraction", "mean_norm", "mean_pairwise_l2_across_clips"):
        assert abs(ds[k] - ref[k]) < 1e-6 * max(1.0, abs(ref[k])), (name, k, ds[k], ref[k])
    assert keep.sum() == ref["n_clips"], (keep.sum(), ref["n_clips"])
    print(f"{name}: OK  max|diff|={worst:.2e}  ref {t_ref:.1f}s  batch {t_new:.1f}s")


if __name__ == "__main__":
    check("dinov2")
    check("vjepa2.1", n=500)
