"""Extracts a checkpoint's own ENCODER output (z = model.encoder(x), the per-patch
channel-projected tokens BEFORE the trained AggregationHead) and reduces it to one
vector per frame with a fixed, untrained pooling op (mean/max/concat) instead of
model.aggregate()'s learned MLP. Saves in the same {"features": (N,T,D), "video_paths",
"n_frames"} schema every other precomputed feature file uses, so the result can be fed
back into compute_progress_metrics.py/compute_probes.py/eval_planning.py via --raw cls
unmodified -- no checkpoint needed downstream, since the "encoding" already happened here.

Only meaningful for token_mode='patch', projector_type='channel' checkpoints (P>1); for
projector_type='global' there's already only one token/frame and model.aggregate() is a
no-op, so this would just reproduce the existing --checkpoint/--config path.

Usage:
    python extract_encoder_pooled.py --checkpoint .../best.ckpt --config .../config.yaml \
        --features .../dinov2_noreg/ssv2_validation.pt --manifest ssv2_validation.csv \
        --pooling mean --out results/ssv2_validation_mean.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_common.loading import load_world_model, load_matched_features  # noqa: E402

import pandas as pd  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--pooling", choices=["mean", "max", "concat"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, epoch = load_world_model(args.checkpoint, args.config, device)
    model.eval()
    if cfg.get("token_mode") != "patch" or cfg.get("projector_type") != "channel":
        raise SystemExit(f"expected token_mode=patch, projector_type=channel; "
                          f"got token_mode={cfg.get('token_mode')!r}, projector_type={cfg.get('projector_type')!r}")

    df = pd.read_csv(args.manifest)
    feats, matched = load_matched_features(args.features, df, args.path_col)

    pooled = []
    with torch.no_grad():
        for s in tqdm(range(0, len(feats), args.batch_size)):
            x = feats[s:s + args.batch_size].to(device)
            z = model.encoder(x)  # (B, T, P, out_dim) -- BEFORE model.aggregate()
            if args.pooling == "mean":
                p = z.mean(dim=2)
            elif args.pooling == "max":
                p = z.max(dim=2).values
            else:  # concat
                b, t, pn, d = z.shape
                p = z.reshape(b, t, pn * d)
            pooled.append(p.cpu())

    features = torch.cat(pooled, dim=0)
    print(f"{features.shape[0]} clips, pooling={args.pooling} -> {tuple(features.shape)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "video_paths": matched[args.path_col].tolist(),
                "n_frames": cfg.get("n_frames", 16)}, args.out)
    print(f"Saved -> {args.out}")
