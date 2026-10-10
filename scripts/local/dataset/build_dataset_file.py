"""Build a CSV manifest (video path + annotation/metadata columns) for a directory of
videos, with optional uniform random subsampling.

Recursively finds video files under --data-path, and if --annotations is given, joins
each video against a row in that CSV by video id (derived from the filename) to pull in
whatever metadata columns that CSV has (e.g. Kinetics' train.csv: label, time_start,
time_end, split, is_cc) — works with any annotation CSV, not just Kinetics', as long as it
has an id column and --id-column/--id-length are set to match how ids appear in it vs. in
the video filenames.

Examples:
    # Kinetics-400 train split: video_id is the first 11 chars of the filename stem
    # (youtube_id), annotations/train.csv has a "youtube_id" id column.
    python build_dataset_file.py \\
        --data-path /mnt/rdata7/proprogress/data/kinetics-dataset/k400/train \\
        --annotations /mnt/rdata7/proprogress/data/kinetics-dataset/k400/annotations/train.csv \\
        --id-column youtube_id --id-length 11 \\
        --out manifests/k400_train.csv

    # Same, but only a random 10% of videos:
    python build_dataset_file.py \\
        --data-path /mnt/rdata7/proprogress/data/kinetics-dataset/k400/train \\
        --annotations /mnt/rdata7/proprogress/data/kinetics-dataset/k400/annotations/train.csv \\
        --id-column youtube_id --id-length 11 \\
        --sample-ratio 0.1 --out manifests/k400_train_10pct.csv

    # No annotations, just an inventory of what's there:
    python build_dataset_file.py --data-path /some/video/dir --out manifests/videos.csv
"""
import argparse
import csv as csv_module
import random
from pathlib import Path

import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".webm", ".mkv", ".mov"}


def find_videos(data_path: str) -> list:
    root = Path(data_path)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def load_annotations(annotations_path: str, id_column: str) -> dict:
    """Returns {video_id: {other_column: value, ...}} — every column except id_column is
    carried through as extra metadata."""
    df = pd.read_csv(annotations_path, dtype=str)
    if id_column not in df.columns:
        raise SystemExit(f"--id-column {id_column!r} not found in {annotations_path} "
                          f"(columns: {list(df.columns)})")
    meta_columns = [c for c in df.columns if c != id_column]
    return {row[id_column]: {c: row[c] for c in meta_columns} for _, row in df.iterrows()}, meta_columns


def video_id_for(path: Path, id_length: int) -> str:
    stem = path.stem
    return stem[:id_length] if id_length else stem


def main(args):
    videos = find_videos(args.data_path)
    print(f"Found {len(videos)} videos under {args.data_path}")
    if not videos:
        raise SystemExit("No videos found — nothing to write")

    annotations, meta_columns = ({}, [])
    if args.annotations:
        annotations, meta_columns = load_annotations(args.annotations, args.id_column)
        print(f"Loaded {len(annotations)} annotation rows from {args.annotations} "
              f"(extra columns: {meta_columns})")

    rows = []
    n_matched = 0
    for path in videos:
        vid = video_id_for(path, args.id_length)
        meta = annotations.get(vid)
        if meta is not None:
            n_matched += 1
        row = {"video_path": str(path), "video_id": vid}
        row.update(meta or {c: "" for c in meta_columns})
        rows.append(row)
    if args.annotations:
        print(f"Matched annotations for {n_matched}/{len(videos)} videos")

    if args.sample_ratio < 1.0:
        rng = random.Random(args.seed)
        n_keep = max(1, round(len(rows) * args.sample_ratio))
        rows = rng.sample(rows, n_keep)
        print(f"Uniformly sampled {n_keep}/{len(videos)} videos (ratio={args.sample_ratio}, seed={args.seed})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_path", "video_id"] + meta_columns
    with open(out_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="root dir of videos, searched recursively")
    parser.add_argument("--annotations", default=None, help="optional CSV to join video metadata from")
    parser.add_argument("--id-column", default="youtube_id", help="id column name in --annotations")
    parser.add_argument("--id-length", type=int, default=None,
                         help="truncate each video filename's stem to this many chars before matching "
                              "against --id-column (Kinetics: 11, since filenames are "
                              "<youtube_id>_<start>_<end>.mp4); omit to match the full stem")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                         help="uniformly randomly keep this fraction of videos, e.g. 0.1 for 10%%")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="output manifest CSV path")
    main(parser.parse_args())
