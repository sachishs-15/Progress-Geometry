"""Fill train.py's live cache for a manifest in parallel, so training never decodes inline.

  python fill_live_cache.py --manifest <csv> --live-cache-dir <dir> --shard-id I --num-shards N

Builds the FeatureProvider with the same settings train.py uses for token_mode=patch, so
the cache files it writes are the ones train.py looks up. Each file is written to a temp
name and renamed, so a concurrent reader never sees a partial file.
"""
import argparse
import csv
import os

import torch

from data.feature_provider import FeatureProvider


def main(args):
    paths = [r[args.path_col] for r in csv.DictReader(open(args.manifest))][args.shard_id::args.num_shards]
    p = FeatureProvider(None, live_cache_dir=args.live_cache_dir, n_frames=16, token_mode="patch",
                        patch_hub_id="dinov2_vits14", patch_img_size=196, raw_img_size=224, device="cuda")
    for path in paths:
        target = p._live_cache_path(path)
        if target.exists():
            continue
        try:
            feats = p._compute_live(path)
            tmp = target.with_name(target.name + f".tmp{os.getpid()}")
            torch.save({"features": feats, "metadata": p._live_cache_metadata(path)}, tmp)
            os.replace(tmp, target)
            print(path, tuple(feats.shape), flush=True)
        except Exception as e:
            print(f"FAILED {path}: {e}", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--manifest", required=True)
    a.add_argument("--live-cache-dir", required=True)
    a.add_argument("--path-col", default="video_path")
    a.add_argument("--shard-id", type=int, default=0)
    a.add_argument("--num-shards", type=int, default=1)
    main(a.parse_args())
