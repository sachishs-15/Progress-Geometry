"""Extract Ego4D Goal-Step leaf sub-segments as individual cropped clips, plus one copy
of each source video, into a per-video folder structure — a raw-data prep step before
running Ego4D Goal-Step through the same frozen-encoder pipeline as CrossTask/SSv2/Kinetics
(a later prepare_ego4dgoalstep.py, analogous to prepare_ssv2.py, would encode these clips
into this project's unified feature cache).

Ego4D Goal-Step annotations nest "segments" (checked against goalstep_val.json: max
nesting depth found is 2 — some top-level segments have no "segments" of their own and are
leaves themselves; others have a nested "segments" list of true sub-steps). This script
extracts the LEAF segments only (the atomic steps — "segments within segments"), named
after their step_description.

Output layout:
  <out_dir>/videos/<video_uid>/full.mp4                  copy of the source video
  <out_dir>/videos/<video_uid>/<step_description>.mp4    one cropped clip per leaf segment
                                                            (numbered suffix on duplicates
                                                            — most videos repeat step
                                                            descriptions, e.g. "wash_hands"
                                                            several times)

Cropping defaults to ffmpeg fast stream-copy (-ss before -i, -c copy) — near-instant per
clip, but snapped to the nearest keyframe, so very short segments (861 of 6732 leaves in
the val split are under 3s) can come out shifted or occasionally empty. Chosen for speed;
pass --accurate to re-encode instead if that precision loss turns out to matter.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def sanitize(name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return name.strip("_") or "segment"


def leaves(segment: dict) -> list:
    sub = segment.get("segments") or []
    if not sub:
        return [segment]
    out = []
    for s in sub:
        out.extend(leaves(s))
    return out


def crop_clip(video_path: Path, start: float, end: float, out_path: Path, accurate: bool) -> None:
    duration = max(end - start, 0.0)
    if accurate:
        cmd = ["ffmpeg", "-y", "-ss", f"{start}", "-i", str(video_path), "-t", f"{duration}",
               "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)]
    else:
        cmd = ["ffmpeg", "-y", "-ss", f"{start}", "-i", str(video_path), "-t", f"{duration}",
               "-c", "copy", "-avoid_negative_ts", "make_zero", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def process_video(video_entry: dict, videos_dir: Path, out_dir: Path, accurate: bool) -> int:
    video_uid = video_entry["video_uid"]
    src = videos_dir / f"{video_uid}.mp4"
    if not src.exists():
        print(f"[{video_uid}] source video not found at {src}, skipping")
        return 0

    dest_dir = out_dir / "videos" / video_uid
    dest_dir.mkdir(parents=True, exist_ok=True)

    full_dest = dest_dir / "full.mp4"
    if not full_dest.exists():
        shutil.copy2(src, full_dest)

    all_leaves = []
    for top in video_entry.get("segments") or []:
        all_leaves.extend(leaves(top))

    name_counts = {}
    n_written = 0
    for leaf in all_leaves:
        base = sanitize(leaf.get("step_description") or leaf.get("step_category") or "segment")
        name_counts[base] = name_counts.get(base, 0) + 1
        name = base if name_counts[base] == 1 else f"{base}_{name_counts[base]}"
        out_path = dest_dir / f"{name}.mp4"
        try:
            crop_clip(src, leaf["start_time"], leaf["end_time"], out_path, accurate)
            n_written += 1
        except subprocess.CalledProcessError as e:
            stderr_tail = e.stderr.decode(errors="replace")[-300:] if e.stderr else ""
            print(f"[{video_uid}] failed to crop {name!r} "
                  f"({leaf['start_time']}-{leaf['end_time']}): {stderr_tail}")
    print(f"[{video_uid}] wrote {n_written}/{len(all_leaves)} clips -> {dest_dir}")
    return n_written


def main(args):
    with open(args.annotations) as f:
        data = json.load(f)
    videos = data["videos"]
    if args.video_uid:
        videos = [v for v in videos if v["video_uid"] == args.video_uid]
        if not videos:
            raise SystemExit(f"video_uid {args.video_uid!r} not found in {args.annotations}")
    elif args.max_videos:
        videos = videos[:args.max_videos]

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out)
    total = 0
    for v in videos:
        total += process_video(v, videos_dir, out_dir, args.accurate)
    print(f"done: {total} clips across {len(videos)} videos -> {out_dir}/videos/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations",
                         default="/mnt/rdata7/proprogress/data/ego4dgoal/annotations/goalstep_val.json")
    parser.add_argument("--videos-dir", default="/mnt/rdata7/proprogress/data/ego4dgoal/videos")
    parser.add_argument("--out", default="/mnt/rdata7/proprogress/data/ego4dgoalstep")
    parser.add_argument("--video-uid", default=None, help="process just this one video (pilot run)")
    parser.add_argument("--max-videos", type=int, default=None, help="cap number of videos processed")
    parser.add_argument("--accurate", action="store_true", help="re-encode for frame-accurate cuts (slower)")
    main(parser.parse_args())
