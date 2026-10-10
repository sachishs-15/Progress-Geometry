"""Video-level train/val/test splitting (never split individual frames — a whole video
stays in exactly one split), stratified by task where possible, persisted to disk so every
experiment reuses the exact same split.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from data.cache import CachedFeatureDataset


def make_splits(dataset: CachedFeatureDataset, train: float = 0.7, val: float = 0.15,
                 test: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    assert abs(train + val + test - 1.0) < 1e-6, "split ratios must sum to 1"
    rng = np.random.RandomState(seed)

    task_to_videos: Dict[str, List[str]] = {}
    for vid in dataset.video_ids:
        task_to_videos.setdefault(dataset.task_of(vid), []).append(vid)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for task, vids in task_to_videos.items():
        vids = sorted(vids)
        rng.shuffle(vids)
        n = len(vids)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # Clamp so we never overshoot n, and prefer keeping at least the split
        # boundaries in order train/val/test even for very small per-task video counts.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        splits["train"].extend(vids[:n_train])
        splits["val"].extend(vids[n_train:n_train + n_val])
        splits["test"].extend(vids[n_train + n_val:n_train + n_val + n_test])

    for key in splits:
        splits[key] = sorted(splits[key])
    return splits


def save_splits(splits_dir: str, dataset_name: str, splits: Dict[str, List[str]]) -> None:
    out_dir = Path(splits_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, vids in splits.items():
        with open(out_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(vids) + ("\n" if vids else ""))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"counts": {k: len(v) for k, v in splits.items()}}, f, indent=2)


def load_splits(splits_dir: str, dataset_name: str) -> Dict[str, List[str]]:
    in_dir = Path(splits_dir) / dataset_name
    splits = {}
    for name in ("train", "val", "test"):
        path = in_dir / f"{name}.txt"
        with open(path) as f:
            splits[name] = [line.strip() for line in f if line.strip()]
    return splits


def splits_exist(splits_dir: str, dataset_name: str) -> bool:
    in_dir = Path(splits_dir) / dataset_name
    return all((in_dir / f"{name}.txt").exists() for name in ("train", "val", "test"))


def get_or_create_splits(splits_dir: str, dataset: CachedFeatureDataset, train: float = 0.7,
                          val: float = 0.15, test: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    if splits_exist(splits_dir, dataset.name):
        return load_splits(splits_dir, dataset.name)
    splits = make_splits(dataset, train=train, val=val, test=test, seed=seed)
    save_splits(splits_dir, dataset.name, splits)
    return splits
