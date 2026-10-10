"""Download (if needed) and convert the official CrossTask release into this project's
unified feature cache (data/cache.py), restricted to the 18 PRIMARY tasks (the only ones
with manual temporal-step annotations — see README section "CrossTask dataset").

NOT run as part of the synthetic smoke test — the features zip alone is ~30GB. Point
--raw-dir at an existing extraction to skip downloading.

File format (reverse-engineered from github.com/DmZhukov/CrossTask's data.py/args.py,
verified by fetching those files directly, not by downloading the dataset):
  <raw_dir>/crosstask_release/tasks_primary.txt   repeating 6-line blocks per task:
      task_id / title / url / n_steps / "step,step,..." / blank line
  <raw_dir>/crosstask_release/videos.csv           lines: task_id,video_id,youtube_url
  <raw_dir>/crosstask_release/annotations/<task_id>_<video_id>.csv
                                                    rows: step,start,end (step 1-indexed,
                                                    start/end in seconds, float)
  <raw_dir>/crosstask_features/<video_id>.npy      [T, 3200] float, one row per second
"""
import argparse
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import write_manifest, write_video_cache  # noqa: E402

RELEASE_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"
FEATURES_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _unzip(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / f".{zip_path.stem}.extracted"
    if marker.exists():
        return
    print(f"extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.touch()


def read_primary_tasks(path: Path) -> dict:
    """Returns {task_id: {"title":..., "n_steps": int, "steps": [str, ...]}}."""
    tasks = {}
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i < len(lines) and lines[i].strip():
        task_id = lines[i].strip()
        title = lines[i + 1].strip()
        n_steps = int(lines[i + 3].strip())
        steps = lines[i + 4].strip().split(",")
        tasks[task_id] = {"title": title, "n_steps": n_steps, "steps": steps}
        i += 6  # 5 fields + blank separator line
    return tasks


def read_videos_csv(path: Path) -> dict:
    """Returns {task_id: [video_id, ...]}."""
    task_vids = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task_id, video_id, _url = line.split(",")
            task_vids.setdefault(task_id, []).append(video_id)
    return task_vids


def read_step_labels(annotation_path: Path, t: int, n_steps: int) -> np.ndarray:
    """One 0-indexed step id per second, filled from `step,start,end` rows; seconds not
    covered by any annotated interval are forward/backward-filled from the nearest
    annotated neighbor so every timestep gets a label."""
    labels = np.full(t, -1, dtype=np.int64)
    with open(annotation_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step, start, end = line.split(",")
            start = int(math.floor(float(start)))
            end = min(int(math.ceil(float(end))), t)
            labels[start:end] = int(step) - 1

    # Forward-fill, then back-fill, any still-unlabeled seconds.
    last = -1
    for i in range(t):
        if labels[i] == -1:
            labels[i] = last
        else:
            last = labels[i]
    nxt = -1
    for i in range(t - 1, -1, -1):
        if labels[i] == -1:
            labels[i] = nxt
        else:
            nxt = labels[i]
    return labels


def main(args):
    raw_dir = Path(args.raw_dir)
    if not args.skip_download:
        _download(RELEASE_URL, raw_dir / "crosstask_release.zip")
        _download(FEATURES_URL, raw_dir / "crosstask_features.zip")
        _unzip(raw_dir / "crosstask_release.zip", raw_dir)
        _unzip(raw_dir / "crosstask_features.zip", raw_dir)

    release_dir = raw_dir / "crosstask_release"
    features_dir = raw_dir / "crosstask_features"
    tasks = read_primary_tasks(release_dir / "tasks_primary.txt")
    task_vids = read_videos_csv(release_dir / "videos.csv")

    video_to_task = {}
    n_written, n_missing_features, n_missing_annotation = 0, 0, 0
    for task_id, info in tasks.items():
        for video_id in task_vids.get(task_id, []):
            feature_path = features_dir / f"{video_id}.npy"
            if not feature_path.exists():
                n_missing_features += 1
                continue
            features = np.load(feature_path).astype(np.float32)

            annotation_path = release_dir / "annotations" / f"{task_id}_{video_id}.csv"
            step_labels = None
            if annotation_path.exists():
                step_labels = read_step_labels(annotation_path, features.shape[0], info["n_steps"])
            else:
                n_missing_annotation += 1

            write_video_cache(args.out, video_id, task_id, features, step_labels)
            video_to_task[video_id] = task_id
            n_written += 1

    write_manifest(args.out, name="crosstask_primary", feature_dim=3200, video_to_task=video_to_task)
    print(f"wrote {n_written} videos ({n_missing_features} missing .npy, "
          f"{n_missing_annotation} missing annotation csv) -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data_cache/_crosstask_raw",
                         help="where the release/features zips are downloaded + extracted")
    parser.add_argument("--out", default="data_cache/crosstask", help="unified cache output dir")
    parser.add_argument("--skip-download", action="store_true",
                         help="use an already-downloaded+extracted --raw-dir instead of fetching the zips")
    main(parser.parse_args())
