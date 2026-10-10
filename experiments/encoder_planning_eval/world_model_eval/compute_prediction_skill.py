"""Prediction skill of a trained ProceduralWorldModel's predictor, relative to a
"nothing changes" baseline that predicts the next latent equals the current one.

    skill = 1 − model_error / copy_error

1 = perfect, 0 = no better than assuming nothing changes, < 0 = worse than that. Raw
prediction error alone can't tell a good world model from a collapsed one (if the latent
barely moves, every error is small); the copy baseline shrinks along with the latent, so
the ratio can't be gamed that way.

Errors are measured in the predictor's own space (model.encoder output: per-patch
tokens for patch mode, per-frame vectors for cls mode) with the checkpoint's own
training loss (config pred_loss: l1 = mean |Δ|, l2 = mean Δ²), averaged over tokens and
dimensions.

Two evaluations per clip:
- one_step (teacher-forced, same as pred_loss): predict z[t+1] from true z[≤t], all t.
- rollout (open-loop): give the true first --context frames, then feed the model its own
  predictions back for the remaining T − context frames. Horizon h compares the h-th
  predicted frame with the truth; the copy baseline holds the last context frame fixed.

Only for ProceduralWorldModel checkpoints -- raw encoders have no predictor.

Usage:
    python compute_prediction_skill.py \\
        --checkpoint ../../predictive_world_model/checkpoints/<run>/best.ckpt \\
        --config ../../predictive_world_model/checkpoints/<run>/config.yaml \\
        --features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt \\
        --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \\
        --out results/<run>_skill.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_common.loading import load_matched_features, load_world_model  # noqa: E402


def position_error(pred, target, kind):
    """(B, n, ...) -> (B, n): per-position error, averaged over tokens and dims."""
    if kind == "l1":
        e = (pred - target).abs()
    elif kind == "l2":
        e = (pred - target).pow(2)
    elif kind == "angular":
        # Match WorldModelLoss exactly: cosine is over the final latent dimension.
        # Patch mode leaves a token dimension, which is averaged below.
        e = 1 - F.cosine_similarity(pred, target, dim=-1)
    else:
        raise ValueError(f"unknown prediction error {kind!r}; expected l1, l2, or angular")
    return e.flatten(2).mean(-1) if e.ndim > 2 else e


@torch.no_grad()
def rollout(predictor, context, steps):
    """Autoregressive open-loop rollout: each step's prediction is appended as the next
    input frame. Returns (B, steps, ...)."""
    seq, preds = context, []
    for _ in range(steps):
        nxt = predictor(seq)[:, -1:]
        preds.append(nxt)
        seq = torch.cat([seq, nxt], dim=1)
    return torch.cat(preds, dim=1)


def summarize(model_err, copy_err):
    """model_err, copy_err: (n_clips,) per-clip mean errors."""
    eps = 1e-12
    valid = copy_err > eps
    per_clip = 1 - model_err[valid] / copy_err[valid]
    copy_total = float(copy_err.sum())
    return {
        "model_error": float(model_err.mean()),
        "copy_error": float(copy_err.mean()),
        # With no movement in the representation, the copy baseline has zero error and
        # relative prediction skill is mathematically undefined (0/0 under exact
        # collapse). JSON null plus explicit counts is preferable to a non-standard NaN.
        "skill": None if copy_total <= eps else float(1 - model_err.sum() / copy_total),
        "skill_per_clip_median": float(np.median(per_clip)) if len(per_clip) else None,
        "n_skill_clips": int(valid.sum()),
        "n_degenerate_clips": int((~valid).sum()),
        "degenerate_copy_baseline": copy_total <= eps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--context", type=int, default=4, help="true frames given before the open-loop rollout")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, epoch = load_world_model(args.checkpoint, args.config, device, with_predictor=True)
    kind = cfg.get("pred_loss", "l2")
    features, _ = load_matched_features(args.features, pd.read_csv(args.manifest, dtype=str), args.path_col)

    one_m, one_c, roll_m, roll_c = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(features), args.batch_size):
            z = model.encoder(features[s:s + args.batch_size].to(device))
            t = z.shape[1]
            if not 1 <= args.context < t:
                raise SystemExit(f"--context must be in [1, {t - 1}] for T={t}")
            if cfg.get("token_mode", "cls") == "patch":
                one_pred = model.predictor(z[:, :-1])
            else:
                one_pred = model.predictor(z)[:, :-1]
            one_m.append(position_error(one_pred, z[:, 1:], kind).mean(1).cpu())
            one_c.append(position_error(z[:, :-1], z[:, 1:], kind).mean(1).cpu())
            preds = rollout(model.predictor, z[:, :args.context], t - args.context)
            truth = z[:, args.context:]
            roll_m.append(position_error(preds, truth, kind).cpu())
            roll_c.append(position_error(z[:, args.context - 1:args.context].expand_as(truth), truth, kind).cpu())

    one_m, one_c = torch.cat(one_m).numpy(), torch.cat(one_c).numpy()
    roll_m, roll_c = torch.cat(roll_m).numpy(), torch.cat(roll_c).numpy()  # (n_clips, horizons)

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "features": args.features,
        "manifest": args.manifest,
        "n_clips": len(one_m),
        "error_kind": kind,
        "context": args.context,
        "one_step": summarize(one_m, one_c),
        "rollout": {str(h + 1): summarize(roll_m[:, h], roll_c[:, h]) for h in range(roll_m.shape[1])},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}\n")
    print(f"error = {kind}, {len(one_m)} clips")
    o = result["one_step"]
    print(f"one-step (teacher-forced): model {o['model_error']:.4f}  copy {o['copy_error']:.4f}  "
          f"skill {o['skill']:.3f}  (per-clip median {o['skill_per_clip_median']:.3f})")
    print(f"open-loop rollout from {args.context} true frames:")
    for h, r in result["rollout"].items():
        print(f"  h={h:>2}: model {r['model_error']:.4f}  copy {r['copy_error']:.4f}  skill {r['skill']:.3f}")
