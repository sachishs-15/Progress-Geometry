"""Per-frame step/substep targets and LSTR's data_info.json for Goal-Step.

  python build_targets.py --out data/goalstep [--limit N]

Writes <out>/target_perframe/<video_uid>.npy, shape (L, 1 + C) with column 0 = background,
and <out>/data_info.json. A feature is "in" a segment when start <= t < end, so steps and
substeps that overlap give a multi-hot row. The official test split is unannotated, so
val is what LSTR's "test" phase scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from goalstep_common import FEATURE_FPS, STRIDE, FPS, flatten_steps, load_split, num_features, num_features_at, num_frames


def main(args):
    videos = {s: load_split(s) for s in ("train", "val")}
    if args.uids:
        videos = {s: [v for v in vs if v["video_uid"] in args.uids] for s, vs in videos.items()}
    if args.limit:
        videos = {s: v[:args.limit] for s, v in videos.items()}
    segments = {v["video_uid"]: flatten_steps(v["segments"]) for vs in videos.values() for v in vs}
    classes = sorted({c for segs in segments.values() for _, _, c in segs})
    train_uids = {v["video_uid"] for v in videos["train"]}

    out = Path(args.out)
    (out / "target_perframe").mkdir(parents=True, exist_ok=True)
    index = {c: i + 1 for i, c in enumerate(classes)}
    rows = []
    for uid, segs in segments.items():
        n = num_frames(uid)
        L = num_features_at(n, args.fps) if args.fps else num_features(n)
        rows.append((uid, "train" if uid in train_uids else "val", n, L))
        t = np.arange(L) / args.fps if args.fps else np.arange(L) * STRIDE / FPS
        target = np.zeros((L, len(classes) + 1), dtype=np.uint8)
        for start, end, cat in segs:
            target[(t >= start) & (t < end), index[cat]] = 1
        target[:, 0] = target[:, 1:].sum(1) == 0
        np.save(out / "target_perframe" / f"{uid}.npy", target)

    with open(out / "index.csv", "w") as f:
        f.write("video_uid,split,n_frames,L\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    json.dump({"GoalStep": {
        "data_root": str(out), "class_names": ["Background"] + classes,
        "num_classes": len(classes) + 1, "ignore_index": -100, "metrics": "AP",
        "fps": args.fps or FEATURE_FPS,
        "train_session_set": [v["video_uid"] for v in videos["train"]],
        "test_session_set": [v["video_uid"] for v in videos["val"]],
    }}, open(out / "data_info.json", "w"))
    print(f"{len(segments)} videos, {len(classes)} step/substep categories -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N videos per split (smoke tests)")
    p.add_argument("--uids", nargs="*", default=None, help="only these videos")
    p.add_argument("--fps", type=float, default=None,
                   help="feature rate; default is the Goal-Step paper's stride-16 grid (1.875 fps)")
    main(p.parse_args())
