"""Dense frozen-DINOv2 CLS features over full Goal-Step videos, one vector per 16 frames.

  python extract_features.py --out <dir> [--data-info data/goalstep/data_info.json]
         [--shard-id I --num-shards N] [--batch 256]

Writes <out>/<video_uid>.npy, shape (L, 384) float16, L = ceil(frames / 16), the same
length build_targets.py uses. Uses the same encoder plugin as scripts/local/
precompute_features (HF dinov2-with-registers-small, CLS pooling), so the features match
what the straightening encoder was trained on. Existing outputs are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frozen_encoders"))
import models  # noqa: E402,F401  (registers the encoder plugins)
from common.base import build  # noqa: E402

from goalstep_common import FPS, STRIDE, VIDEOS  # noqa: E402


def main(args):
    info = json.load(open(args.data_info))["GoalStep"]
    uids = info["train_session_set"] + info["test_session_set"]
    uids = uids[args.shard_id::args.num_shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model": "dinov2 (facebook/dinov2-with-registers-small)", "pooling": "cls",
               "stride_frames": STRIDE, "source_fps": FPS, "feature_fps": FPS / STRIDE,
               "dtype": "float16", "dim": 384, "preprocessing": "HF processor default",
               "layout": "<video_uid>.npy, (L, 384), feature k at frame k * stride_frames"},
              open(out / "meta.json", "w"), indent=1)
    with build("dinov2", device="cuda:0", pooling="cls") as enc:
        for uid in uids:
            path = out / f"{uid}.npy"
            if path.exists():
                continue
            try:
                vr = VideoReader(str(VIDEOS / f"{uid}.mp4"), ctx=cpu(0), num_threads=1)
                idx = np.arange(0, len(vr), STRIDE)
                feats = []
                for i in range(0, len(idx), args.batch):
                    frames = vr.get_batch(idx[i:i + args.batch]).asnumpy()
                    feats.append(enc.encode([Image.fromarray(f) for f in frames]).embeddings)
                np.save(path.with_suffix(".tmp.npy"), np.concatenate(feats).astype(np.float16))
                path.with_suffix(".tmp.npy").rename(path)
                print(uid, len(idx), flush=True)
            except Exception as e:
                print(f"FAILED {uid}: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data-info", default="data/goalstep/data_info.json")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    main(p.parse_args())
