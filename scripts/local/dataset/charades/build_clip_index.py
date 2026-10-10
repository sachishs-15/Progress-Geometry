"""Build a (video_path, start_time, end_time) index into Charades' RAW, uncut per-video
files, from the CiA chiral annotations
(external/chirality_in_action/cia_full_dataset/cia-charades-{train,validation}.csv).

Standalone / self-contained on purpose (no cross-import from
cia_full_dataset/prepare_chiral_clips.py, which does the analogous resolution but also
cuts clips with ffmpeg) — this only resolves paths + time ranges, no cutting. See
scripts/local/dataset/extract_clips.py for the shared (dataset-agnostic) cutting step
that consumes the CSV this writes.

Usage:
    python build_clip_index.py \\
        --csv ../../../external/chirality_in_action/cia_full_dataset/cia-charades-train.csv \\
        --raw-root /mnt/rdata7/proprogress/data/Charades/videos/Charades_v1 \\
        --out ../../../external/chirality_in_action/cia_full_dataset/clip_index/charades_train_index.csv
"""
import argparse
import csv as csv_module
from pathlib import Path

import pandas as pd

RAW_VIDEO_EXTS = (".mp4", ".MP4")


def find_raw_video(raw_root: str, video_id: str):
    for ext in RAW_VIDEO_EXTS:
        p = Path(raw_root) / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def build_index(csv_path: str, raw_root: str, out_path: str, limit: int = None):
    df = pd.read_csv(csv_path, dtype=str)
    if limit:
        df = df.head(limit)

    rows = []
    missing = set()
    for _, row in df.iterrows():
        raw_path = find_raw_video(raw_root, row["id"])
        if raw_path is None:
            missing.add(row["id"])
            continue
        rows.append({"video_path": str(raw_path), "start_time": row["start_time"], "end_time": row["end_time"]})

    if missing:
        print(f"Skipped {len(missing)} rows — raw video not found under {raw_root} "
              f"(first 20): {sorted(missing)[:20]}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["video_path", "start_time", "end_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)}/{len(df)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="cia-charades-{train,validation}.csv")
    parser.add_argument("--raw-root", required=True, help="root of Charades' raw per-video files")
    parser.add_argument("--out", required=True, help="output csv path (video_path, start_time, end_time)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for testing)")
    args = parser.parse_args()
    build_index(args.csv, args.raw_root, args.out, args.limit)
