"""Build a (video_path, start_time, end_time) index into EPIC-Kitchens' RAW, uncut
per-participant videos, from the CiA chiral annotations
(external/chirality_in_action/cia_full_dataset/cia-epic-{train,validation}.csv).

Standalone / self-contained on purpose (no cross-import from
cia_full_dataset/prepare_chiral_clips.py, which does the analogous resolution but also
cuts clips with ffmpeg) — this only resolves paths + time ranges, no cutting. See
scripts/local/dataset/extract_clips.py for the shared (dataset-agnostic) cutting step
that consumes the CSV this writes.

Usage:
    python build_clip_index.py \\
        --csv ../../../external/chirality_in_action/cia_full_dataset/cia-epic-train.csv \\
        --raw-root /rhome/sachish/datasets/epic-kitchens/EPIC-KITCHENS \\
        --out ../../../external/chirality_in_action/cia_full_dataset/clip_index/epic_train_index.csv
"""
import argparse
import csv as csv_module
from pathlib import Path

import pandas as pd

RAW_VIDEO_EXTS = (".MP4", ".mp4")


def find_raw_video(raw_root: str, participant_id: str, video_id: str):
    for ext in RAW_VIDEO_EXTS:
        p = Path(raw_root) / participant_id / "videos" / f"{video_id}{ext}"
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
        raw_path = find_raw_video(raw_root, row["participant_id"], row["video_id"])
        if raw_path is None:
            missing.add(f'{row["participant_id"]}/{row["video_id"]}')
            continue
        rows.append({"video_path": str(raw_path), "start_time": row["start_sec"], "end_time": row["stop_sec"]})

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
    parser.add_argument("--csv", required=True, help="cia-epic-{train,validation}.csv")
    parser.add_argument("--raw-root", required=True, help="root of EPIC-Kitchens' raw per-participant videos")
    parser.add_argument("--out", required=True, help="output csv path (video_path, start_time, end_time)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for testing)")
    args = parser.parse_args()
    build_index(args.csv, args.raw_root, args.out, args.limit)
