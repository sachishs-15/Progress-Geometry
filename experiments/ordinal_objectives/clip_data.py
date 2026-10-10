"""Dataset for objective comparison: per-frame features + the progress label every
objective here is ultimately supervised by, plus the two sampling knobs some of them
need (temporal crops, same-class clip pairs).

Deliberately thinner than the objectives that consume it. Anything an objective can do
by INDEXING the already-encoded (B, T, d) latents -- picking K frames, shuffling them,
reversing them, striding them for pace prediction -- is done inside the objective, on
GPU, not here: the encoder is strictly per-frame (frozen DINOv2 features -> an MLP), so
one forward pass over all T frames serves every possible subsample of them. The dataset
therefore only owns what genuinely has to happen at load time: which clip, which frames
of it, what its progress labels are, and which other clip it is paired with.

Progress labels
---------------
The precomputed feature files this reads (LiFT's precompute_features.py format) sample
n_frames=16 frames spread EVENLY across each whole video, so frame i's progress through
that video is exactly i/(T-1) -- no annotation needed, and exact rather than approximate.

That exactness is also a trap, and crop_size exists because of it: with every clip served
as the same length-16 evenly-spaced sequence, an objective supervised on i/(T-1) can
partly succeed by learning "which of 16 slots is this" from whatever positional
regularity DINOv2 features happen to carry, rather than learning anything about progress.
crop_size > 0 samples a random contiguous span of that length instead, and crop_label
decides what the labels mean:

  crop_label="global"  labels stay on the VIDEO's clock: a span starting at frame s gets
                       (s + arange(L)) / (T-1), so the same visual content always has the
                       same label and the model must infer absolute progress from
                       appearance alone -- it can no longer read it off the slot index,
                       because slot 0 of a crop is a different progress value every time.
  crop_label="local"   labels are renormalized to [0,1] within the crop
                       (arange(L)/(L-1)). This is what most of the ranking/ordering
                       literature implicitly assumes (only RELATIVE order within the
                       observed window is defined) and is the right setting for
                       objectives that are themselves relative. It is also strictly
                       harder to interpret for absolute-progress objectives, since two
                       identical frames get different labels in different crops.

Both are worth running; they answer different questions, which is why this is a config
knob and not a fixed choice. See README.md's "The crop confound".

Randomness
----------
Crop offsets and pair choices are drawn from the MODULE-LEVEL `random`, never from a
per-dataset `random.Random(seed)`. A dataset-owned generator looks more controlled but is
wrong here: DataLoader pickles the dataset into every worker, so each worker gets a clone
of that generator at the same state and never reseeds it. Measured, with num_workers=2
and crop_size=8, that produced byte-identical crop offsets in every epoch, with worker 1's
n-th draw equal to worker 0's -- i.e. the random cropping that `crop_size` exists to
provide was not random across epochs at all. PyTorch reseeds the global `random` (and
torch's own) per worker per epoch from a base seed derived from the main process's torch
generator, so using it gives correct independence AND stays reproducible from train.py's
single torch.manual_seed/random.seed.

Clip pairing
------------
TCC and (optionally) SCL need a SECOND sequence showing the same activity. pair_col names
a manifest column to group by -- `verb` for SSv2's ssv2_train.csv, which gives groups like
"Folding [something]". Videos whose group has only one member are dropped from the
dataset when pairing is on, rather than paired with themselves (a self-pair makes TCC's
cycle trivially consistent and would quietly inflate it).
"""
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ObjectiveClipDataset(Dataset):
    """Yields, per item:
        features     (L, D) float32   per-frame encoder INPUT (frozen features)
        progress     (L,)   float32   progress label in [0, 1], per crop_label above
        frame_index  (L,)   int64     index within the source clip (diagnostics; also what
                                      objectives use as the "timestamp" when they need one
                                      that is independent of the crop's label convention)
        video_path   str
      and, when pairing is on, `features_pair` / `progress_pair` / `frame_index_pair` of
      the same shapes, drawn from a different video in the same pair_col group.

    L is crop_size when crop_size > 0, else the clip's own T.
    """

    def __init__(self, video_paths: List[str], feature_provider, crop_size: int = 0,
                 crop_label: str = "global", groups: Optional[Dict[str, str]] = None):
        if crop_label not in ("global", "local"):
            raise ValueError(f"crop_label must be 'global' or 'local', got {crop_label!r}")
        self.provider = feature_provider
        self.crop_size = crop_size
        self.crop_label = crop_label
        self.groups = groups

        video_paths = list(video_paths)
        self.group_members: Dict[str, List[str]] = {}
        if groups is not None:
            for p in video_paths:
                g = groups.get(p)
                if g is None:
                    continue
                self.group_members.setdefault(g, []).append(p)
            # Singleton groups can't supply a DIFFERENT clip of the same activity, and
            # pairing a clip with itself would make TCC's cycle trivially consistent --
            # drop them instead of silently self-pairing.
            video_paths = [p for p in video_paths
                           if groups.get(p) is not None and len(self.group_members[groups[p]]) > 1]
        self.video_paths = video_paths

    @classmethod
    def from_manifest(cls, manifest_path: str, feature_provider, path_col: str = "video_path",
                      pair_col: Optional[str] = None, crop_size: int = 0,
                      crop_label: str = "global"):
        df = pd.read_csv(manifest_path, dtype=str)
        if path_col not in df.columns:
            raise ValueError(f"{manifest_path} has no {path_col!r} column "
                             f"(columns: {list(df.columns)})")
        groups = None
        if pair_col is not None:
            if pair_col not in df.columns:
                raise ValueError(f"{manifest_path} has no {pair_col!r} column to pair clips by "
                                 f"(columns: {list(df.columns)})")
            groups = dict(zip(df[path_col], df[pair_col]))
        return cls(df[path_col].tolist(), feature_provider, crop_size=crop_size,
                   crop_label=crop_label, groups=groups)

    def __len__(self):
        return len(self.video_paths)

    def _crop(self, features: torch.Tensor):
        """features: (T, D) -> (features, progress, frame_index) for one (possibly
        cropped) example. Progress is always computed from the ORIGINAL clip's length,
        then renormalized only if crop_label='local' -- see the module docstring."""
        t = features.shape[0]
        if self.crop_size > 0:
            if t < self.crop_size:
                raise ValueError(f"crop_size={self.crop_size} needs {self.crop_size} frames/clip, "
                                 f"but this clip only has {t}")
            start = random.randint(0, t - self.crop_size)  # module-level: see "Randomness"
            length = self.crop_size
        else:
            start, length = 0, t
        frame_index = torch.arange(start, start + length)
        features = features[start:start + length]
        if self.crop_label == "local" and length > 1:
            progress = torch.arange(length, dtype=torch.float32) / (length - 1)
        else:
            progress = frame_index.to(torch.float32) / max(t - 1, 1)
        return features, progress, frame_index

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        features, progress, frame_index = self._crop(self.provider.get(video_path))
        item = {"features": features, "progress": progress, "frame_index": frame_index,
                "video_path": video_path}
        if self.groups is not None:
            members = self.group_members[self.groups[video_path]]
            other = video_path
            while other == video_path:  # guaranteed to terminate: len(members) > 1 (see __init__)
                other = random.choice(members)
            f2, p2, i2 = self._crop(self.provider.get(other))
            item.update({"features_pair": f2, "progress_pair": p2, "frame_index_pair": i2,
                         "video_path_pair": other})
        return item


def progress_from_batch(batch, device):
    """(features, progress, optional pair tensors) moved to `device` as a plain dict --
    everything a objectives.base.Objective.forward signature needs, with the string
    columns dropped."""
    out = {}
    for k in ("features", "progress", "frame_index", "features_pair", "progress_pair",
              "frame_index_pair"):
        if k in batch:
            out[k] = batch[k].to(device, non_blocking=True)
    return out


def deterministic_subset(n: int, limit: int, seed: int = 0):
    """Sorted indices of a deterministic subsample of `n` items down to `limit` -- used by
    train.py's in-training ordinality monitor, which wants a fixed, cheap, and above all
    IDENTICAL set of clips every epoch (a resampled subset would make the monitor curve
    partly noise about which clips were drawn)."""
    if limit <= 0 or limit >= n:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, size=limit, replace=False))
