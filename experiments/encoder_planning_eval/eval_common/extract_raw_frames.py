"""Decodes real video frames (FeatureProvider's token_mode='raw' tier) and saves them in
the same {"features": (N,T,3,H,W) uint8, "video_paths": [...], "n_frames": T} schema
every other precomputed feature file in this project uses -- so a token_mode='raw'
checkpoint (models/world_model.py) can be evaluated with the SAME --checkpoint/--config
tooling (compute_progress_metrics.py, compute_probes.py, eval_planning.py) as any other
checkpoint. No model forward pass happens here; eval_common/loading.py's
encode_world_model() (and interpolation_planning/dataset.py's _run_checkpoint_encoder())
run model.pixel_encoder(x) on this file's raw frames before model.encoder(x), same order
as ProceduralWorldModel.forward().

Usage:
    python extract_raw_frames.py --manifest ssv2_validation.csv --path-col video_path \
        --n-frames 16 --raw-img-size 224 --out .../ssv2_validation.pt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PWM_ROOT = Path(__file__).resolve().parents[2] / "predictive_world_model"
sys.path.insert(0, str(PWM_ROOT))
from data.feature_provider import FeatureProvider  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--raw-img-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (smoke test)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    provider = FeatureProvider(n_frames=args.n_frames, token_mode="raw", raw_img_size=args.raw_img_size)

    df = pd.read_csv(args.manifest)
    paths = df[args.path_col].tolist()
    if args.limit:
        paths = paths[:args.limit]

    frames, kept_paths = [], []
    for p in tqdm(paths):
        try:
            frames.append(provider.get(p))
            kept_paths.append(p)
        except Exception as e:
            print(f"skip {p}: {e}")

    features = torch.stack(frames)  # (N, T, 3, H, W) uint8
    print(f"{features.shape[0]}/{len(paths)} videos decoded -> {tuple(features.shape)}, dtype={features.dtype}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "video_paths": kept_paths, "n_frames": args.n_frames}, args.out)
    print(f"Saved -> {args.out}")
