"""Trajectory straightness / ordinality metrics on Pouring latents.

Per video, takes the every-frame latents of each model, subsamples them to N evenly spaced
frames (N in --scales; "all" = every frame), and scores each trajectory with the project's own
per-clip metrics (encoder_planning_eval/progress_metrics: LiFT's clip_progress_metrics plus
its raw-space additions). Label-free. Models: raw DINOv2 patch pooling, and checkpoints
(encoder output pooled over patches by mean / flatten, before the aggregation head, or agg).

    python straightness.py --checkpoints name=path.ckpt ... --out-dir results_straightness
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from latents import encode_checkpoint, load_checkpoint, reduce_tokens  # noqa: E402
from pouring_data import cache_dir, load_split  # noqa: E402

sys.path.insert(0, str(HERE.parent / "encoder_planning_eval" / "progress_metrics"))
import compute_progress_metrics as cpm  # noqa: E402


def trajectory_metrics(z):
    m = cpm.clip_progress_metrics(z)
    if m is None:
        return None

    def put(name, value):
        m[name] = 0.0 if value is None else float(value)

    put("pc1_time_spearman", cpm.pc1_time_spearman(z))
    put("distance_to_goal_spearman", cpm.distance_to_goal_spearman(z))
    put("mean_velocity_cosine", cpm.mean_velocity_cosine(z))
    put("mean_velocity_cosine_raw", cpm.mean_velocity_cosine(z, normalize=False))
    put("mean_chord_alignment", cpm.mean_chord_alignment(z))
    put("mean_chord_alignment_raw", cpm.mean_chord_alignment(z, normalize=False))
    put("trajectory_straightness_raw", cpm.trajectory_straightness_raw(z))
    put("adjacent_l2_distance", cpm.adjacent_l2_distance(z))
    return {k: float(v) for k, v in m.items()}


def subsample(z, scale):
    if scale == "all" or int(scale) >= len(z):
        return z
    return z[np.linspace(0, len(z) - 1, int(scale)).astype(int)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=[], help="name=path/to/last.ckpt")
    ap.add_argument("--ckpt-pools", nargs="+", default=["mean", "flatten", "agg"])
    ap.add_argument("--raw-pools", nargs="*", default=["mean", "flatten"])
    ap.add_argument("--scales", nargs="+", default=["16", "32", "64", "128", "all"])
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos per split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(HERE / "results_straightness"))
    args = ap.parse_args()

    tokens_dir = cache_dir("hubdinov2_vits14", None, 196, 1)
    ckpts = {}
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        ckpts[name] = load_checkpoint(path, args.device)[0]
    rows = [(f"raw_dinov2_patch_{p}", None, p) for p in args.raw_pools]
    rows += [(n, m, p) for n, m in ckpts.items() for p in args.ckpt_pools]

    per_video = {}  # (model, pool, split, scale) -> [metrics dict per video]
    for split in args.splits:
        for v in load_split(split)[:args.limit]:
            patch = torch.load(tokens_dir / f"{v.name}.patch.pt")["tokens"].float()
            grid = int(round(patch.shape[1] ** 0.5))
            for name, model, pool in rows:
                if model is None:
                    z = reduce_tokens(None, patch, (grid, grid), pool).numpy()
                else:
                    z = encode_checkpoint(model, None, patch, args.device, pool).numpy()
                z = z.astype(np.float64)
                for scale in args.scales:
                    m = trajectory_metrics(subsample(z, scale))
                    if m is not None:
                        per_video.setdefault((name, pool, split, scale), []).append(m)
            print(f"{split}: {v.name}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pool", "split", "scale", "metric", "median", "mean", "n_videos"])
        for (name, pool, split, scale), ms in per_video.items():
            for k in ms[0]:
                vals = [m[k] for m in ms]
                w.writerow([name, pool, split, scale, k, np.median(vals), np.mean(vals), len(vals)])
    (out / "per_video.json").write_text(json.dumps({"|".join(k): v for k, v in per_video.items()}))
    print("wrote", out / "summary.csv")
