"""Cut clips from a (video_path, start_time, end_time) index — the format written by
epic/build_clip_index.py and charades/build_clip_index.py — into individual clip files.

Dataset-agnostic on purpose: it only ever reads those three columns, so the same script
cuts EPIC's, Charades', or any future dataset's index the same way, as long as that
dataset's own build_clip_index.py produces this csv shape.

Output filenames are derived, not read from the csv (which carries no clip-id column):
"<raw video's own filename stem>_<start>_<end>.mp4" — e.g. "P01_01_1.0_3.0.mp4" or
"KRF68_1.0_3.0.mp4", matching the naming convention
external/chirality_in_action/cia_full_dataset/prepare_chiral_clips.py already uses for
its own path_id/item_id-named clips.

Usage:
    python extract_clips.py \\
        --index /rhome/sachish/datasets/epic-kitchens/cia-data/epic_train_index.csv \\
        --out-dir /rhome/sachish/datasets/epic-kitchens/cia-data/train
"""
import argparse
import concurrent.futures as cf
import subprocess
from pathlib import Path

import pandas as pd


def clip_output_path(out_dir: Path, video_path: str, start: str, end: str) -> Path:
    stem = Path(video_path).stem
    return out_dir / f"{stem}_{start}_{end}.mp4"


def cut_clip(video_path: str, start: str, end: str, out_path: Path, pre_seek_buffer: float = 15.0):
    if out_path.exists():
        return True, None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start, end = float(start), float(end)
    duration = end - start
    coarse_seek = max(0.0, start - pre_seek_buffer)
    fine_seek = start - coarse_seek

    # Two-stage seek: fast (keyframe) seek before -i, then accurate seek after -i, so
    # short clips land on the right frame without decoding from t=0 — same approach as
    # cia_full_dataset/prepare_chiral_clips.py's _cut_clip.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{coarse_seek:.3f}", "-i", str(video_path),
        "-ss", f"{fine_seek:.3f}", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr[-2000:]
    return True, None


def extract_clips(index_path: str, out_dir: str, limit: int = None, num_workers: int = 4):
    df = pd.read_csv(index_path, dtype=str)
    if limit:
        df = df.head(limit)

    out_dir = Path(out_dir)
    jobs = [(row.video_path, row.start_time, row.end_time,
              clip_output_path(out_dir, row.video_path, row.start_time, row.end_time))
            for row in df.itertuples()]

    print(f"Cutting {len(jobs)} clips with {num_workers} workers...")
    failures = []
    with cf.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(cut_clip, v, s, e, o): o for v, s, e, o in jobs}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            ok, err = fut.result()
            if not ok:
                failures.append((futures[fut], err))
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({len(failures)} failed)")

    if failures:
        print(f"\n{len(failures)} clips failed:")
        for path, err in failures[:5]:
            print(f"  {path}: {err.splitlines()[-1] if err else 'unknown error'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="csv written by a dataset's build_clip_index.py")
    parser.add_argument("--out-dir", required=True, help="where to write cut clips")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (for testing)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    extract_clips(args.index, args.out_dir, args.limit, args.num_workers)
