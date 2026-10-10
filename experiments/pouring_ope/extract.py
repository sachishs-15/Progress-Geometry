"""Cache per-frame CLS / patch tokens of a frozen encoder for every pouring video.

One file per video and token kind in <ROOT>/token_cache/<encoder tag>/: `<name>.cls.pt`
(T, D) and `<name>.patch.pt` (T, P, D), fp16, plus the original frame index of each row.
Existing files are reused. Pooling is applied later (run_ope.py), so changing it never
re-runs the backbone.

    python extract.py --encoder dinov2 --tokens cls patch --stride 1
"""
import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen_encoders"
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(HERE))

import models  # noqa: E402,F401  (registers every encoder plugin)
from common.base import REGISTRY, build  # noqa: E402
from common.video_utils import load_frames_stride  # noqa: E402
from pouring_data import cache_dir, load_split  # noqa: E402


def cls_pooling(cls):
    for name in ("cls", "pooler"):
        if name in cls.pooling_choices:
            return name
    return None


def encode_video(encoder, frames, want, batch):
    chunks = [frames] if batch <= 0 else [frames[i:i + batch] for i in range(0, len(frames), batch)]
    cls_rows, patch_rows, idx, offset, grid = [], [], [], 0, None
    for chunk in chunks:
        if "cls" in want:
            res = encoder.encode(chunk)
            cls_rows.append(torch.from_numpy(res.embeddings))
            idx.extend(offset + i for i in res.frame_indices)
        if "patch" in want:
            patches, grid = encoder.get_patch_tokens(chunk)
            patch_rows.append(patches)
            if "cls" not in want:
                idx.extend(range(offset, offset + len(patches)))
        offset += len(chunk)
    out = {"frame_idx": idx, "grid_hw": grid}
    if cls_rows:
        out["cls"] = torch.cat(cls_rows).half()
    if patch_rows:
        out["patch"] = torch.cat(patch_rows).half()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="dinov2", choices=sorted(REGISTRY))
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth video frame")
    ap.add_argument("--tokens", nargs="+", default=["cls", "patch"], choices=["cls", "patch"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--batch", type=int, default=64, help="frames per forward call (0 = whole video)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos per split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = cache_dir(args.encoder, args.hf_id, args.image_size, args.stride)
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"device": args.device}
    if args.hf_id:
        kwargs["hf_id"] = args.hf_id
    if args.image_size:
        kwargs["image_size"] = args.image_size
    cls_cls = REGISTRY[args.encoder]
    pooling = cls_pooling(cls_cls)
    if pooling:
        kwargs["pooling"] = pooling
    want_all = set(args.tokens)
    if "cls" in want_all and pooling is None:
        print(f"{args.encoder} has no CLS/pooler token; skipping cls")
        want_all.discard("cls")
    if "patch" in want_all and not cls_cls.supports_patch_tokens:
        print(f"{args.encoder} exposes no patch tokens; skipping patch")
        want_all.discard("patch")

    todo = []
    for split in args.splits:
        vids = load_split(split)[:args.limit]
        for v in vids:
            need = {k for k in want_all if not (out_dir / f"{v.name}.{k}.pt").exists()}
            if need:
                todo.append((v, need))
    print(f"{out_dir.name}: {len(todo)} videos to extract")

    with build(args.encoder, **kwargs) as encoder:
        for n, (v, need) in enumerate(todo, 1):
            t0 = time.time()
            frames, orig_idx = load_frames_stride(v.path, stride=args.stride, max_frames=10 ** 9)
            if len(v.labels) != (orig_idx[-1] + 1 if orig_idx else 0) and \
                    abs(len(v.labels) - (orig_idx[-1] + 1)) > args.stride:
                print(f"WARNING {v.name}: {len(v.labels)} labels vs ~{orig_idx[-1] + 1} decoded frames")
            res = encode_video(encoder, frames, need, args.batch)
            frame_idx = torch.tensor([orig_idx[min(i, len(orig_idx) - 1)] for i in res["frame_idx"]])
            for kind in need:
                torch.save({"tokens": res[kind], "frame_idx": frame_idx, "grid_hw": res["grid_hw"]},
                           out_dir / f"{v.name}.{kind}.pt")
            print(f"[{n}/{len(todo)}] {v.name}: {len(frames)} frames, {time.time() - t0:.1f}s", flush=True)
