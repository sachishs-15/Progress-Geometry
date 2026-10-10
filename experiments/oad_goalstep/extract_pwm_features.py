"""Per-frame latents from a trained ProceduralWorldModel checkpoint over full Goal-Step videos.

  python extract_pwm_features.py --root <data root> --fps 4 --data-info <root>/data_info.json \
      --run-dir <pwm checkpoint dir> --ckpt epoch19.ckpt --name pwm_ep19 [--shard-id I --num-shards N]

Frames are taken exactly as extract_meanpatch.py does (frame k = the video frame at time k / fps).
Each frame goes through frozen DINOv2 (no registers, 196px) -> the checkpoint's encoder, per frame
(batch-norm in eval mode, so nothing mixes across frames and every latent is causal). Writes
<root>/<name>_agg/<uid>.npy (L, 128): model.aggregate(encoder(x)), the latent the progress
metrics and the straightening/VICReg losses see, and <root>/<name>_z/<uid>.npy (L, 196*8): the
flattened per-patch latent. float16, streamed, resumable, failures logged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

EVAL = Path("/home/sachish/proc-planning/experiments/encoder_planning_eval")
sys.path.insert(0, str(EVAL))
from eval_common.loading import load_world_model  # noqa: E402

from dense_provider import DensePatchProvider  # noqa: E402
from goalstep_common import VIDEOS  # noqa: E402


def main(args):
    info = json.load(open(args.data_info))["GoalStep"]
    uids = (info["train_session_set"] + info["test_session_set"])[args.shard_id::args.num_shards]
    root, run = Path(args.root), Path(args.run_dir)
    dirs = {"agg": root / f"{args.name}_agg", "z": root / f"{args.name}_z"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    model, _, epoch = load_world_model(run / args.ckpt, run / "config.yaml", "cuda", with_predictor=False)
    model.eval()
    print(f"using {run.name}/{args.ckpt}, checkpoint epoch field: {epoch}", flush=True)
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
            shapes = {"agg": (n_out, 128), "z": (n_out, 196 * 8)}
            out = {k: np.lib.format.open_memmap(p.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                                shape=shapes[k]) for k, p in done.items()}
            for i in range(0, n_out, args.batch):
                frames = vr.get_batch(idx[i:i + args.batch]).asnumpy()
                x = torch.stack([prov._preprocess(Image.fromarray(f)) for f in frames]).to("cuda")
                with torch.no_grad():
                    patches = prov._backbone.forward_features(x)["x_norm_patchtokens"]   # (B, 196, 384)
                    z = model.encoder(patches[None])                                     # (1, B, 196, 8)
                    agg = model.aggregate(z)                                             # (1, B, 128)
                out["agg"][i:i + len(frames)] = agg[0].half().cpu().numpy()
                out["z"][i:i + len(frames)] = z[0].flatten(1).half().cpu().numpy()
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
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.ckpt")
    p.add_argument("--name", required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    main(p.parse_args())
