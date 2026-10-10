"""Autoregressive latent rollout: encode only the real first frame, then feed the
predictor's own output back in as the next "observed" latent, for T-1 steps -- never
touching ground-truth frames 1..T-1. Tests whether the PREDICTOR's own imagined
trajectory stays geometrically healthy (straight, high-rank, temporally ordered) or
drifts/collapses, as opposed to every other script in this project, which only scores the
ENCODER's ground-truth-conditioned output. Reports ground-truth vs. rollout metrics for
the same clips side by side, plus a per-step drift metric (cosine similarity / L2
distance between z_rollout[k] and the real z[k]) showing how fast prediction error
compounds with horizon.

model.predictor consumes RAW encoder output, not the agg_head's aggregated target (see
world_model.py's aggregate() docstring) -- so rollout happens in raw encoder space
regardless of whether the checkpoint has an agg head. --pooling only matters for
token_mode='patch' (mean/concat over the patch dim, same convention as
run_encoder_pooled_eval.py), reducing both the ground-truth and rollout tensors to
(N,T,D) before scoring.

Usage (cls/raw):
    python rollout_eval.py --checkpoint .../best.ckpt --config .../config.yaml \\
        --features .../dinov2/ssv2_validation.pt --manifest ssv2_validation.csv \\
        --out results/rollout_<name>.json

Usage (patch, with pooling):
    python rollout_eval.py --checkpoint .../best.ckpt --config .../config.yaml \\
        --features .../dinov2_noreg/ssv2_validation.pt --manifest ssv2_validation.csv \\
        --pooling mean --out results/rollout_<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common.loading import load_world_model, load_matched_features  # noqa: E402
from compute_progress_metrics import compute_metrics_dict  # noqa: E402


def _pool(z, pooling):
    if pooling == "mean":
        return z.mean(dim=2)
    b, t, p, d = z.shape
    return z.reshape(b, t, p * d)


@torch.no_grad()
def rollout(model, features, device, batch_size, pooling=None):
    """Returns (z_gt, z_rollout), both (N,T,D) numpy -- ground-truth encoder output (full
    real clip) and the autoregressive self-rollout (real frame 0, predicted 1..T-1),
    pooled identically if `pooling` is given (patch mode)."""
    gt_chunks, roll_chunks = [], []
    for s in range(0, len(features), batch_size):
        x = features[s:s + batch_size].to(device)
        if model.pixel_encoder is not None:
            x = model.pixel_encoder(x)
        z_gt = model.encoder(x)  # (B,T,D) cls/raw, (B,T,P,D) patch -- ground truth, full clip
        t = z_gt.shape[1]

        z_roll = z_gt[:, :1]  # seed: only the REAL first frame, nothing else touches ground truth
        for _ in range(1, t):
            z_pred = model.predictor(z_roll)  # same causal predictor every step, growing prefix
            z_roll = torch.cat([z_roll, z_pred[:, -1:]], dim=1)  # append predicted next step

        if pooling is not None:
            z_gt = _pool(z_gt, pooling)
            z_roll = _pool(z_roll, pooling)
        gt_chunks.append(z_gt.cpu().numpy())
        roll_chunks.append(z_roll.cpu().numpy())
    return np.concatenate(gt_chunks), np.concatenate(roll_chunks)


def drift_by_step(z_gt, z_roll):
    """Per-step cosine similarity and L2 distance between rollout and ground truth,
    averaged over clips -- how fast prediction error compounds with horizon. Step 0 is
    always identical (both are the real first frame): a trivial sanity check, not signal."""
    cos, l2 = [], []
    for k in range(z_gt.shape[1]):
        a, b = z_gt[:, k], z_roll[:, k]
        cos_k = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
        cos.append(float(cos_k.mean()))
        l2.append(float(np.linalg.norm(a - b, axis=-1).mean()))
    return {"step_cosine_sim": cos, "step_l2_distance": l2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--pooling", choices=["mean", "concat"], default=None,
                         help="required for token_mode='patch' checkpoints (reduces the "
                              "(N,T,P,D) ground-truth/rollout tensors to (N,T,D) before "
                              "scoring); omit for cls/raw.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _, epoch = load_world_model(args.checkpoint, args.config, device, with_predictor=True)
    model.eval()

    df = pd.read_csv(args.manifest, dtype=str)
    features, out_df = load_matched_features(args.features, df, args.path_col)

    if features.dim() == 4 and args.pooling is None:
        raise SystemExit(f"--features is (N,T,P,D) patch-shaped {tuple(features.shape)} -- "
                          f"pass --pooling mean/concat")
    if features.dim() == 3 and args.pooling is not None:
        raise SystemExit(f"--features is already (N,T,D) {tuple(features.shape)} -- "
                          f"--pooling only applies to patch-shaped input")

    print(f"Loaded {args.checkpoint} (epoch {epoch}); {len(features)} clips, "
          f"rolling out {features.shape[1]} steps from the real first frame...")
    z_gt, z_roll = rollout(model, features, device, args.batch_size, args.pooling)

    gt_metrics = compute_metrics_dict(z_gt, out_df)
    roll_metrics = compute_metrics_dict(z_roll, out_df)
    drift = drift_by_step(z_gt, z_roll)

    result = {
        "checkpoint": args.checkpoint, "checkpoint_epoch": epoch,
        "features": args.features, "manifest": args.manifest, "pooling": args.pooling,
        "ground_truth": gt_metrics, "rollout": roll_metrics, "drift": drift,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}")

    print(f"\neffective_rank_fraction: gt={gt_metrics['effective_rank_fraction']:.4f}  "
          f"rollout={roll_metrics['effective_rank_fraction']:.4f}")
    print("\nground_truth vs rollout (summary_mean):")
    for k in gt_metrics["summary_mean"]:
        print(f"  {k}: gt={gt_metrics['summary_mean'][k]:.4f}  rollout={roll_metrics['summary_mean'][k]:.4f}")
    print("\nstep drift (cosine_sim / L2, step 0 is trivially identical -- both real frame 0):")
    for k, (c, l) in enumerate(zip(drift["step_cosine_sim"], drift["step_l2_distance"])):
        print(f"  t={k}: cos={c:.4f}  l2={l:.4f}")
