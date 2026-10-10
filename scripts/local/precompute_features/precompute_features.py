"""Precompute per-frame features for the videos listed in a CSV manifest (as built by
scripts/local/dataset/build_dataset_file.py — needs a `video_path` column, other columns
ignored), using a swappable frozen encoder.

Unlike external/chirality_in_action/LiFT/precompute_features.py (which this mirrors but
generalizes), the encoder isn't hardcoded to DINOv2 — it reuses
experiments/frozen_encoders' model registry, so --model can be any of that registry's
plugins (dinov2, dinov3, clip, siglip, vjepa2, vjepa2.1, lift, ...). Output format:
{"features": (N, n_frames, D), "video_paths": [...], "n_frames": T, "failures": [...],
"extraction_metadata": {...}, "extraction_fingerprint": ...} by default (one
pooled vector/frame, via --pooling), or with --patch-tokens: {"features":
(N, n_frames, P, D), ...} -- the RAW (unpooled) per-patch grid, for models with
supports_patch_tokens=True (currently: dinov2*, dinov3, vjepa2, vjepa2.1). Both shapes
are what experiments/predictive_world_model/data/feature_provider.py's FeatureProvider
accepts directly via --features (its tier-1 bulk-precomputed path already branches on
(T,D) vs (T,N,D) per video) — --patch-tokens output is for token_mode=patch there,
default output is for token_mode=cls.

Each video's n_frames frames are linearly sampled across its whole duration (same
semantics as LiFT's own load_frames_linspace) — needed because the encoders here are
per-frame image encoders (or lift, per-frame-latent), and downstream sequence models
(like LiFT) expect a fixed T across every video.

Decode or encode failures are skipped and written to `<out>.failures.csv`; replacement
black frames are never inserted. Existing outputs and resumable chunks are reused only
when their extraction fingerprint matches every requested setting and the source CSV.

Checkpointing: rather than holding every video's features in RAM for the whole run and
writing once at the end (the original design — an unbounded-growth OOM risk that lost an
entire ~24h/60K-video shard to a single kill with zero output), features are flushed to a
small "<out>.chunkN.pt" file every --checkpoint-every videos, bounding peak memory to one
chunk DURING extraction. On (re)start, existing chunks for this shard are detected, their
videos are skipped, and new work resumes after them — a kill only loses the in-progress
chunk, not the whole run. Once every video is processed, all chunks are concatenated into
the final "<out>.pt" (or "<out>.pt.shardN.pt") and deleted, so the on-disk output format
downstream tools (merge_features.py, train.py) expect is unchanged -- this final
concat-and-save step DOES briefly hold the whole shard in memory again (unavoidable, since
the output format is one contiguous tensor), so it's serialized across concurrent shards
of the same (csv, model) via a file lock, rather than every parallel shard hitting that
peak at once (see the lock around it below).

Usage:
    python precompute_features.py --csv manifests/k400_train.csv --out-dir features/
        -> writes features/k400_train.pt  (name mirrors the csv's own basename)

    python precompute_features.py --csv manifests/k400_train.csv --out-dir features/ \\
        --model dinov3 --n-frames 16 --pooling mean_patch

    python precompute_features.py --csv manifests/k400_train.csv --out-dir features/ \\
        --model dinov2_noreg --n-frames 16 --patch-tokens
        -> writes features/dinov2_noreg/k400_train.pt, features shape (N, 16, 196, 384)
"""
import argparse
import csv as csv_module
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

FROZEN_ENCODERS_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "frozen_encoders"
sys.path.insert(0, str(FROZEN_ENCODERS_ROOT))
import models  # noqa: E402  (import registers all plugins: dinov2, dinov3, clip, siglip, vjepa2, vjepa2.1, lift)
from common.base import REGISTRY, build  # noqa: E402


class VideoDecodeError(RuntimeError):
    pass


def load_frames_linspace(video_path: str, n: int) -> List[Image.Image]:
    """Return n index-uniform frames, raising when the video cannot be decoded."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        if len(vr) == 0:
            raise ValueError("video contains no frames")
        indices = np.linspace(0, len(vr) - 1, n).astype(int)
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in batch]
    except Exception as e:
        raise VideoDecodeError(f"{video_path}: {e}") from e


def read_video_paths(csv_path: str) -> List[str]:
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        if "video_path" not in reader.fieldnames:
            raise SystemExit(f"{csv_path} has no 'video_path' column (columns: {reader.fieldnames})")
        return [row["video_path"] for row in reader]


def out_base_name(args) -> str:
    name = f"{Path(args.csv).stem}.pt"
    if args.num_shards > 1:
        # Matches external/chirality_in_action/LiFT/merge_features.py's expected shard
        # glob (<name>.pt.shard*.pt), so that script can merge these directly.
        name += f".shard{args.shard_id}.pt"
    return name


def chunk_path(out_dir: Path, base_name: str, chunk_idx: int) -> Path:
    return out_dir / f"{base_name}.chunk{chunk_idx}.pt"


def existing_chunks(out_dir: Path, base_name: str):
    """Sorted (chunk_idx, path) for already-written chunks of this shard, oldest first."""
    found = []
    for p in out_dir.glob(f"{base_name}.chunk*.pt"):
        idx = int(p.name[len(f"{base_name}.chunk"):-len(".pt")])
        found.append((idx, p))
    return sorted(found)


def save_chunk(path: Path, features: torch.Tensor, video_paths: List[str], metadata: dict):
    # Write-then-rename so a kill mid-write can never leave a corrupt chunk that a resume
    # would try (and fail) to load.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"features": features, "video_paths": video_paths,
                "extraction_metadata": metadata,
                "extraction_fingerprint": metadata_fingerprint(metadata)}, tmp_path)
    tmp_path.rename(path)


def _coerce_kwarg_value(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_encoder_kwargs(args) -> dict:
    kwargs = {"pooling": args.pooling}
    if args.hf_id:
        kwargs["hf_id"] = args.hf_id
    if args.encoder_kwargs:
        for pair in args.encoder_kwargs.split(";"):
            key, sep, value = pair.partition("=")
            if not sep or not key.strip() or "," in pair or "=" in value:
                raise ValueError(f"invalid --encoder-kwargs item {pair!r}; expected key=value")
            kwargs[key.strip()] = _coerce_kwarg_value(value.strip())
    return kwargs


def build_extraction_metadata(args, source_csv: Path, encoder_kwargs: dict) -> dict:
    encoder_cls = REGISTRY[args.model]
    checkpoint_id = args.hf_id or getattr(encoder_cls, "default_hf_id", None)
    if checkpoint_id is None:
        hf_id_param = inspect.signature(encoder_cls.__init__).parameters.get("hf_id")
        if hf_id_param is not None and hf_id_param.default is not inspect.Parameter.empty:
            checkpoint_id = hf_id_param.default
    software_versions = {}
    for package in ("torch", "torchvision", "transformers", "timm", "decord", "Pillow"):
        try:
            software_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software_versions[package] = None
    return {
        "schema_version": 1,
        "model": args.model,
        "encoder_class": f"{encoder_cls.__module__}.{encoder_cls.__qualname__}",
        "checkpoint_id": checkpoint_id,
        "n_frames": args.n_frames,
        "pooling": args.pooling,
        "patch_tokens": bool(args.patch_tokens),
        "encoder_kwargs": encoder_kwargs,
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "frame_sampling": "linspace_index_v1",
        "software_versions": software_versions,
    }


def metadata_fingerprint(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_metadata(data: dict, expected: dict, path: Path) -> None:
    actual = data.get("extraction_metadata")
    actual_fingerprint = data.get("extraction_fingerprint")
    if actual is None or actual_fingerprint is None:
        raise SystemExit(
            f"{path} predates extraction metadata and cannot be safely reused. "
            "Move/delete it and re-run extraction."
        )
    if actual_fingerprint != metadata_fingerprint(actual):
        raise SystemExit(f"{path} has corrupt or inconsistent extraction metadata")
    if actual_fingerprint != metadata_fingerprint(expected):
        raise SystemExit(
            f"{path} was created with different extraction settings.\n"
            f"stored:    {json.dumps(actual, sort_keys=True)}\n"
            f"requested: {json.dumps(expected, sort_keys=True)}"
        )


def write_failures(path: Path, failures: list) -> None:
    if not failures:
        path.unlink(missing_ok=True)
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["video_path", "stage", "error"])
        writer.writeheader()
        writer.writerows(failures)
    tmp_path.replace(path)


def main(args):
    if args.model not in REGISTRY:
        raise KeyError(f"Unknown --model {args.model!r}. Available: {sorted(REGISTRY)}")
    if args.patch_tokens and not REGISTRY[args.model].supports_patch_tokens:
        raise SystemExit(f"--patch-tokens requires a model with supports_patch_tokens=True; "
                          f"{args.model!r} doesn't have it.")

    source_csv = Path(args.csv)
    encoder_kwargs = parse_encoder_kwargs(args)
    metadata = build_extraction_metadata(args, source_csv, encoder_kwargs)

    # Nested under the model name so re-running the same csv through a different --model
    # (e.g. comparing dinov2 vs dinov3 features) never collides or overwrites — each
    # model gets its own subtree under --out-dir.
    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = out_base_name(args)
    out_path = out_dir / base_name
    failures_path = out_dir / f"{base_name}.failures.csv"

    if out_path.exists():
        validate_metadata(torch.load(out_path, map_location="cpu", mmap=True), metadata, out_path)
        print(f"{out_path} already exists with matching extraction metadata — nothing to do")
        return

    video_paths = read_video_paths(args.csv)
    print(f"Found {len(video_paths)} videos in {args.csv}")
    if args.num_shards > 1:
        video_paths = video_paths[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(video_paths)} videos")

    chunks = existing_chunks(out_dir, base_name)
    done_paths = set()
    if chunks:
        for _, p in chunks:
            chunk = torch.load(p, map_location="cpu", mmap=True)
            validate_metadata(chunk, metadata, p)
            done_paths.update(chunk["video_paths"])
        print(f"Resuming: {len(done_paths)} videos already done in {len(chunks)} existing chunks")
    next_chunk_idx = (chunks[-1][0] + 1) if chunks else 0
    remaining_paths = [p for p in video_paths if p not in done_paths]
    print(f"{len(remaining_paths)} videos left to process")

    failures = []
    if remaining_paths:
        encoder = build(args.model, device=args.device, **encoder_kwargs)

        buf_features = []
        buf_paths = []
        with encoder:
            for path in tqdm(remaining_paths, desc=f"Extracting {args.model} features"):
                try:
                    frames = load_frames_linspace(path, n=args.n_frames)
                except VideoDecodeError as e:
                    print(f"skipping {path}: decode failed: {e}")
                    failures.append({"video_path": path, "stage": "decode", "error": str(e)})
                    continue
                try:
                    if args.patch_tokens:
                        feat, _grid_hw = encoder.get_patch_tokens(frames)  # (T, P, D), already CPU
                    else:
                        feat = torch.from_numpy(encoder.encode(frames).embeddings)  # (T, D)
                except Exception as e:
                    print(f"skipping {path}: {e}")
                    failures.append({"video_path": path, "stage": "encode", "error": str(e)})
                    continue
                if feat.shape[0] != args.n_frames:
                    print(f"skipping {path}: got {feat.shape[0]} embeddings, "
                          f"expected --n-frames={args.n_frames} (model {args.model!r} doesn't "
                          f"produce one embedding per input frame — try a different --n-frames "
                          f"or a per-frame model)")
                    failures.append({"video_path": path, "stage": "shape",
                                     "error": f"got {feat.shape[0]} embeddings; expected {args.n_frames}"})
                    continue
                buf_features.append(feat)
                buf_paths.append(path)

                if len(buf_paths) >= args.checkpoint_every:
                    save_chunk(chunk_path(out_dir, base_name, next_chunk_idx),
                               torch.stack(buf_features), buf_paths, metadata)
                    next_chunk_idx += 1
                    buf_features, buf_paths = [], []

        if buf_paths:
            save_chunk(chunk_path(out_dir, base_name, next_chunk_idx),
                       torch.stack(buf_features), buf_paths, metadata)
            next_chunk_idx += 1

    write_failures(failures_path, failures)
    if failures:
        print(f"Recorded {len(failures)} failed videos -> {failures_path}")

    chunks = existing_chunks(out_dir, base_name)
    if not chunks:
        raise SystemExit("No videos were successfully processed.")

    # Serialize this finalization step across concurrently-running shards of the SAME
    # (csv, model): concatenating one shard's chunks into a single in-memory tensor can be
    # tens of GB for patch-token features, and precompute_features_parallel.sh runs every
    # shard in parallel -- since they tend to finish extraction around the same time,
    # without this lock they'd all materialize their full tensor at once. That's exactly
    # what silently OOM-killed 3 of 4 shards mid-finalization on a real run
    # (k400_train_10pct, dinov2_noreg patch tokens: ~27GB/shard finalizing concurrently,
    # with zero indication in any single shard's log that the OTHER shards were the reason
    # it died). The lock bounds peak memory to one shard's tensor at a time, at the cost of
    # shards queuing up to finalize -- cheap compared to losing the work silently.
    with open(out_dir / ".finalize.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        all_features = []
        all_paths = []
        for _, p in chunks:
            data = torch.load(p)
            validate_metadata(data, metadata, p)
            all_features.append(data["features"])
            all_paths.extend(data["video_paths"])
        features = torch.cat(all_features, dim=0)  # (N, n_frames, D)
        print(f"Extracted features for {len(all_paths)}/{len(video_paths)} videos, shape {tuple(features.shape)}")

        torch.save({"features": features, "video_paths": all_paths, "n_frames": args.n_frames,
                    "failures": failures,
                    "extraction_metadata": metadata,
                    "extraction_fingerprint": metadata_fingerprint(metadata)}, out_path)
        for _, p in chunks:
            p.unlink()
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="manifest CSV with a video_path column")
    parser.add_argument("--out-dir", required=True,
                         help="output directory; filename is <csv stem>.pt")
    parser.add_argument("--n-frames", type=int, default=16, help="frames linearly sampled per video")
    parser.add_argument("--model", default="dinov2", help=f"one of {sorted(REGISTRY)}")
    parser.add_argument("--hf-id", default=None, help="override the model's default HF checkpoint")
    parser.add_argument("--pooling", default="cls", choices=["cls", "pooler", "mean_patch", "max_patch"],
                         help="ignored when --patch-tokens is set")
    parser.add_argument("--patch-tokens", action="store_true",
                         help="save the RAW (unpooled) per-patch grid via encoder.get_patch_tokens() "
                              "instead of one pooled vector/frame via encoder.encode() -- output "
                              "shape becomes (N, n_frames, P, D) instead of (N, n_frames, D). "
                              "Requires a model with supports_patch_tokens=True.")
    parser.add_argument("--encoder-kwargs", default=None,
                         help="extra plugin-specific constructor kwargs, SEMICOLON-separated "
                              "key=value pairs (bool/int/float auto-detected, else kept as string) "
                              "-- e.g. 'capture_attention=False;per_frame_batch_size=16' for vjepa2. "
                              "Semicolons, not commas, since this gets passed through qsub -v, whose "
                              "own syntax splits on top-level commas between DIFFERENT -v variables.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-id", type=int, default=0, help="0-indexed shard this process handles")
    parser.add_argument("--num-shards", type=int, default=1, help="total shards; --out-dir output is per-shard")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                         help="flush a chunk to disk every N processed videos, bounding peak RAM "
                              "and limiting how much work a kill can lose")
    main(parser.parse_args())
