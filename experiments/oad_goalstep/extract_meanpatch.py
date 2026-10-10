"""Frozen DINOv2 (no registers) frame features at a given fps over full Goal-Step videos.

  python extract_meanpatch.py --root <data root> --fps 4 --data-info <root>/data_info.json
         [--shard-id I --num-shards N] [--batch 16]

Frame k is the video frame at time k / fps (floor), so L matches build_targets.py --fps.
Writes <root>/dinov2_noreg_meanpatch/<uid>.npy (L, 384): the mean of the 196 patch tokens, and
<root>/dinov2_noreg_cls/<uid>.npy (L, 384): the CLS token, both float16. Same backbone and
preprocessing as predictive_world_model's token_mode='patch' (torch.hub dinov2_vits14,
squash-resize to 196px). Streams into preallocated files (memory does not grow with video
length), skips finished videos, and logs failures instead of dying.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from dense_provider import DensePatchProvider
from goalstep_common import VIDEOS


def main(args):
    info = json.load(open(args.data_info))["GoalStep"]
    uids = (info["train_session_set"] + info["test_session_set"])[args.shard_id::args.num_shards]
    root = Path(args.root)
    dirs = {"mean": root / "dinov2_noreg_meanpatch", "cls": root / "dinov2_noreg_cls"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    prov = DensePatchProvider(root, device="cuda")
    prov._load_backbone()
    for uid in uids:
        done = {k: d / f"{uid}.npy" for k, d in dirs.items()}
        if all(p.exists() for p in done.values()):
            continue
        try:
            vr = VideoReader(str(VIDEOS / f"{uid}.mp4"), ctx=cpu(0), num_threads=1)
            vfps, n_total = vr.get_avg_fps(), len(vr)
            n_out = int((n_total - 1) / vfps * args.fps) + 1
            idx = np.minimum((np.arange(n_out) / args.fps * vfps).astype(int), n_total - 1)
            out = {k: np.lib.format.open_memmap(p.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                                shape=(n_out, 384)) for k, p in done.items()}
            for i in range(0, n_out, args.batch):
                frames = vr.get_batch(idx[i:i + args.batch]).asnumpy()
                x = torch.stack([prov._preprocess(Image.fromarray(f)) for f in frames]).to("cuda")
                with torch.no_grad():
                    f = prov._backbone.forward_features(x)
                out["mean"][i:i + len(frames)] = f["x_norm_patchtokens"].mean(1).half().cpu().numpy()
                out["cls"][i:i + len(frames)] = f["x_norm_clstoken"].half().cpu().numpy()
            for k, p in done.items():
                out[k].flush()
                p.with_suffix(".tmp.npy").rename(p)
            print(uid, n_out, flush=True)
        except Exception as e:
            print(f"FAILED {uid}: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--data-info", required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    main(p.parse_args())
