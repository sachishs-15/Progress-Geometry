"""OPE on Pouring for a frozen encoder (any token/pooling choice) or a trained checkpoint.

Frozen encoder, CLS / mean-pooled patches / both concatenated:
    python run_ope.py --encoder dinov2 --pool cls
    python run_ope.py --encoder dinov2 --pool mean
    python run_ope.py --encoder dinov2 --pool cls+mean --part-norm l2
    python run_ope.py --encoder dinov2_noreg --pool grid2

Trained model (token mode comes from its config.yaml; --encoder must match its features):
    python run_ope.py --encoder dinov2 --checkpoint ../predictive_world_model/checkpoints/<run>/best.ckpt

G is the mean total path length over --ref-split videos; OPE is scored on --eval-split.
Missing token caches are extracted first (extract.py, in a subprocess).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from latents import CKPT_POOLS, encode_checkpoint, load_checkpoint, needed_tokens, reduce_tokens  # noqa: E402
from ope import gt_progress, reference_length, video_scores  # noqa: E402
from pouring_data import cache_dir, load_split  # noqa: E402


def ensure_cache(args, kinds, splits):
    out = cache_dir(args.encoder, args.hf_id, args.image_size, args.stride)
    missing = [k for k in kinds for s in splits for v in load_split(s) if not (out / f"{v.name}.{k}.pt").exists()]
    if not missing:
        return
    if args.encoder == "hubdinov2_vits14":
        if kinds != {"patch"} or args.image_size != 196 or args.hf_id:
            raise SystemExit("hubdinov2_vits14 requires patch tokens, --image-size 196, and no --hf-id")
        cmd = [sys.executable, str(HERE / "extract_dense.py"), "--stride", str(args.stride),
               "--splits", *splits, "--device", args.device]
    else:
        cmd = [sys.executable, str(HERE / "extract.py"), "--encoder", args.encoder, "--stride", str(args.stride),
               "--tokens", *sorted(kinds), "--splits", *splits, "--device", args.device]
    if args.encoder != "hubdinov2_vits14":
        if args.hf_id:
            cmd += ["--hf-id", args.hf_id]
        if args.image_size:
            cmd += ["--image-size", str(args.image_size)]
    print("extracting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_tokens(out, name, kinds):
    cls = patch = grid = idx = None
    if "cls" in kinds:
        d = torch.load(out / f"{name}.cls.pt")
        cls, idx = d["tokens"].float(), d["frame_idx"]
    if "patch" in kinds:
        d = torch.load(out / f"{name}.patch.pt")
        patch, grid, idx = d["tokens"].float(), d["grid_hw"], d["frame_idx"]
    return cls, patch, grid, idx.numpy()


def index_live_cache(path):
    """{video_path: cache file} for a predictive_world_model <checkpoint>/live_cache dir."""
    files = {}
    for f in Path(path).glob("*.pt"):
        files[torch.load(f, map_location="cpu")["metadata"]["video_path"]] = f
    return files


def load_live_tokens(files, v):
    """The (T, 196, D) patch tokens a training run computed for this video, plus the frame
    indices FeatureProvider sampled (linspace_index_v1: np.linspace(0, len - 1, T).astype(int))."""
    patch = torch.load(files[v.path], map_location="cpu")["features"].float()
    grid = int(round(patch.shape[1] ** 0.5))
    idx = np.linspace(0, len(v.labels) - 1, patch.shape[0]).astype(int)
    return None, patch, (grid, grid), idx


def video_latents(videos, args, model, mode, kinds, live=None):
    out = cache_dir(args.encoder, args.hf_id, args.image_size, args.stride)
    result = {}
    for v in videos:
        if live is not None:
            cls, patch, grid, idx = load_live_tokens(live, v)
        else:
            cls, patch, grid, idx = load_tokens(out, v.name, kinds)
        if args.n_frames is not None:
            if args.n_frames < 2 or len(idx) < args.n_frames:
                raise ValueError(f"{v.name}: cannot sample {args.n_frames} frames from {len(idx)} cached frames")
            pick = np.linspace(0, len(idx) - 1, args.n_frames).astype(int)
            idx = idx[pick]
            if cls is not None:
                cls = cls[pick]
            if patch is not None:
                patch = patch[pick]
        if np.any(idx < 0) or np.any(idx >= len(v.labels)):
            raise ValueError(f"{v.name}: cache frame indices exceed {len(v.labels)} labels")
        if model is not None:
            z = encode_checkpoint(model, cls, patch, args.device, args.ckpt_pool)
        else:
            z = reduce_tokens(cls, patch, grid, args.pool, args.part_norm)
        gt = gt_progress(v.labels)
        result[v.name] = (z.numpy(), gt[idx])
    return result


def summarize(scores):
    scores = [s for s in scores if s is not None]
    return {k: float(np.mean([s[k] for s in scores])) for k in ("ope", "ope_oracle", "ope_linear_time")} | {
        "ope_median": float(np.median([s["ope"] for s in scores])),
        "ope_std_over_videos": float(np.std([s["ope"] for s in scores])), "n_videos": len(scores)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="dinov2")
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=None,
                    help="sample this many frames uniformly from each cached video before scoring")
    ap.add_argument("--pool", default="cls", help="cls | mean | max | gridG | flatten, joined with '+' to concatenate")
    ap.add_argument("--part-norm", default="none", choices=["none", "l2"], help="L2-normalize each part before concatenating")
    ap.add_argument("--checkpoint", default=None, help="trained world-model checkpoint (needs config.yaml beside it)")
    ap.add_argument("--ckpt-pool", default="mean", choices=CKPT_POOLS,
                    help="checkpoint latent: encoder output pooled over patches by mean or concatenation "
                         "(flatten), both before the aggregation head; agg = after the head")
    ap.add_argument("--live-cache", default=None,
                    help="a training run's live_cache dir: score on exactly those T-frame patch tokens (ignores --stride/--encoder)")
    ap.add_argument("--ref-split", default="train")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--name", default=None, help="label for the results row/file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    model = mode = None
    if args.checkpoint:
        model, mode = load_checkpoint(args.checkpoint, args.device)
        kinds = {"cls"} if mode == "cls" else {"patch"}
        label = f"{args.encoder}__{Path(args.checkpoint).parent.name}"
    else:
        kinds = needed_tokens(args.pool)
        label = f"{args.encoder}__{args.pool}" + ("__l2" if args.part_norm == "l2" else "")
    label = args.name or label

    splits = sorted({args.ref_split, args.eval_split})
    live = None
    if args.live_cache:
        if "cls" in kinds:
            raise SystemExit("--live-cache holds patch tokens only; use a patch pooling (mean/max/gridG/flatten)")
        live = index_live_cache(args.live_cache)
    else:
        ensure_cache(args, kinds, splits)
    ref = load_split(args.ref_split)
    ev = load_split(args.eval_split)
    if live is not None:
        missing = [v.name for v in ref + ev if v.path not in live]
        if missing:
            raise SystemExit(f"{len(missing)} videos missing from {args.live_cache}: {missing[:3]}...")
    ref_lat = video_latents(ref, args, model, mode, kinds, live)
    ev_lat = ref_lat if args.eval_split == args.ref_split else video_latents(ev, args, model, mode, kinds, live)

    G, G_var = reference_length([z for z, _ in ref_lat.values()])
    per_video = {n: video_scores(z, gt, G) for n, (z, gt) in ev_lat.items()}
    ref_per_video = {n: video_scores(z, gt, G) for n, (z, gt) in ref_lat.items()}
    dim = next(iter(ev_lat.values()))[0].shape[1]
    summary = summarize(list(per_video.values()))
    result = {
        "label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "encoder": args.encoder, "hf_id": args.hf_id, "image_size": args.image_size, "stride": args.stride,
        "n_frames": args.n_frames,
        "pool": args.ckpt_pool if model else args.pool, "part_norm": None if model else args.part_norm,
        "checkpoint": args.checkpoint, "token_mode": mode, "latent_dim": int(dim),
        "ref_split": args.ref_split, "eval_split": args.eval_split,
        "G_mean": G, "G_cov": G_var / G ** 2, **summary,
        "ope_ref_in_sample": summarize(list(ref_per_video.values()))["ope"],
        "per_video": per_video,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in result.items() if k != "per_video" and (k != "n_frames" or v is not None)}
    csv_path = out_dir / "summary.csv"
    new = not csv_path.exists()
    if not new:
        with open(csv_path, newline="") as f:
            existing_fields = csv.reader(f).__next__()
        if existing_fields != list(row):
            raise SystemExit(f"{csv_path} has a different column layout; use a new --out-dir")
    (out_dir / f"{label}.json").write_text(json.dumps(result, indent=1))
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{label}: dim={dim}  OPE({args.eval_split})={summary['ope']:.4f} (median {summary['ope_median']:.4f})  oracle={summary['ope_oracle']:.4f}  "
          f"linear-time={summary['ope_linear_time']:.4f}  G cov={result['G_cov']:.3f}  "
          f"in-sample={result['ope_ref_in_sample']:.4f}")
