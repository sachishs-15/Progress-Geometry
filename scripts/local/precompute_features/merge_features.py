"""Combine the per-shard .pt files written by parallel precompute_features.py runs
(--shard-id/--num-shards) into a single features file for LiFT's train.py.

Usage:
    python merge_features.py --shards "features/k400_train.pt.shard*.pt" --out features/k400_train.pt
"""
import argparse
import glob
import hashlib
import json

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True, help="glob pattern matching the shard .pt files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        raise SystemExit(f"No files matched {args.shards!r}")
    print(f"Merging {len(shard_paths)} shards: {shard_paths}")

    shards = [torch.load(p) for p in shard_paths]
    n_frames = shards[0]["n_frames"]
    metadata = shards[0].get("extraction_metadata")
    fingerprint = shards[0].get("extraction_fingerprint")
    for p, s in zip(shard_paths, shards):
        if s["n_frames"] != n_frames:
            raise SystemExit(f"{p} has n_frames={s['n_frames']}, expected {n_frames} (from {shard_paths[0]})")
        if metadata is None or fingerprint is None or s.get("extraction_metadata") is None:
            raise SystemExit(f"{p} has no extraction metadata; regenerate legacy shards before merging")
        actual = hashlib.sha256(json.dumps(
            s["extraction_metadata"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        if s.get("extraction_fingerprint") != actual:
            raise SystemExit(f"{p} has corrupt or inconsistent extraction metadata")
        if s["extraction_fingerprint"] != fingerprint:
            raise SystemExit(f"{p} was extracted with settings different from {shard_paths[0]}")

    features = torch.cat([s["features"] for s in shards], dim=0)
    video_paths = [vp for s in shards for vp in s["video_paths"]]
    failures = [failure for s in shards for failure in s.get("failures", [])]
    print(f"Merged: {len(video_paths)} videos, features shape {tuple(features.shape)}")

    torch.save({"features": features, "video_paths": video_paths, "n_frames": n_frames,
                "failures": failures,
                "extraction_metadata": metadata,
                "extraction_fingerprint": fingerprint}, args.out)
    print(f"Saved -> {args.out}")
