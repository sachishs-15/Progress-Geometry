"""Pouring dataset: videos, per-frame phase labels, train/val split, token-cache paths."""
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("POURING_ROOT", "/rhome/sachish/datasets/pouring"))
CACHE_ROOT = ROOT / "token_cache"


@dataclass
class Video:
    name: str
    path: str
    labels: np.ndarray
    split: str


def load_split(split, root=ROOT):
    with open(Path(root) / f"{split}.pkl", "rb") as f:
        rows = pickle.load(f)
    return [Video(r["name"], str(Path(root) / r["video_file"]), r["frame_label"].numpy().astype(np.int64), split)
            for r in rows]


def cache_tag(encoder, hf_id=None, image_size=None, stride=1):
    tag = encoder
    if hf_id:
        tag += "_" + hf_id.replace("/", "-")
    if image_size:
        tag += f"_i{image_size}"
    return f"{tag}_s{stride}"


def cache_dir(encoder, hf_id=None, image_size=None, stride=1):
    return CACHE_ROOT / cache_tag(encoder, hf_id, image_size, stride)
