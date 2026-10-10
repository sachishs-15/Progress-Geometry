"""Unified on-disk cache format every dataset (CrossTask, synthetic, future V-JEPA, ...)
is converted into by a scripts/prepare_*.py script. Everything downstream (splits,
training, evaluation) only ever talks to `CachedFeatureDataset` — swapping the dataset
a training run uses is a one-line `dataset.cache_dir` change in a config, never a code
change.

On-disk layout for a cache dir:
    <cache_dir>/manifest.json          {"name": ..., "feature_dim": D, "videos": {video_id: task_id, ...}}
    <cache_dir>/videos/<video_id>.npz  features [T, D] float32, step_labels [T] int64 (or absent)
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VideoRecord:
    video_id: str
    task_id: str
    features: np.ndarray  # [T, D] float32
    step_labels: Optional[np.ndarray] = None  # [T] int64, 0-indexed step id per timestep, or None


def write_video_cache(cache_dir: str, video_id: str, task_id: str, features: np.ndarray,
                       step_labels: Optional[np.ndarray] = None) -> None:
    videos_dir = Path(cache_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    save_kwargs = {"features": features, "task_id": np.array(task_id)}
    if step_labels is not None:
        save_kwargs["step_labels"] = np.asarray(step_labels, dtype=np.int64)
    np.savez_compressed(videos_dir / f"{video_id}.npz", **save_kwargs)


def write_manifest(cache_dir: str, name: str, feature_dim: int, video_to_task: Dict[str, str]) -> None:
    manifest = {"name": name, "feature_dim": feature_dim, "videos": video_to_task}
    with open(Path(cache_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


class CachedFeatureDataset:
    """Read-only view over a cache dir written by a scripts/prepare_*.py script."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {cache_dir} — run the matching scripts/prepare_*.py first"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.name: str = manifest["name"]
        self.feature_dim: int = manifest["feature_dim"]
        self._video_to_task: Dict[str, str] = manifest["videos"]

    @property
    def video_ids(self) -> List[str]:
        return sorted(self._video_to_task)

    def task_of(self, video_id: str) -> str:
        return self._video_to_task[video_id]

    def load(self, video_id: str) -> VideoRecord:
        npz = np.load(self.cache_dir / "videos" / f"{video_id}.npz", allow_pickle=False)
        return VideoRecord(
            video_id=video_id,
            task_id=str(npz["task_id"]),
            features=npz["features"],
            step_labels=npz["step_labels"] if "step_labels" in npz else None,
        )

    def __len__(self) -> int:
        return len(self._video_to_task)
