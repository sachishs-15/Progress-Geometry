"""Video-clip dataset backed by a FeatureProvider (precomputed-or-live, see
feature_provider.py).

Two ways to say which clips to train on, both ending up as the same list of video paths:
  - `from_manifest(...)` — a CSV with a video_path column (any dataset, whether or not
    its features have been precomputed; anything missing is computed live).
  - `ClipFeatureDataset(provider.video_paths, provider)` — every video in a bulk
    precomputed .pt, no manifest needed (the fully-precomputed case, e.g. K400).

window_size (default 0, disabled): external/temporal-straightening trains on fixed
short windows (num_hist=3 history + num_pred=1 target -> 4-frame examples), not whole
clips -- each __getitem__ there draws consecutive frames (with a frameskip) from a
longer trajectory. Reproducing that exactly at the video-decode level would mean
reworking FeatureProvider/precompute_features.py's "n_frames spread evenly across the
whole video" convention. Instead, window_size samples a random contiguous span of
(window_size + 1) frames from the ALREADY n_frames-sampled clip each __getitem__ call --
the model only ever sees window_size+1 frames per example, same as the reference,
without touching feature extraction. window_size=0 (default) keeps this project's
original behavior: the whole n_frames-length clip, trained causally in one pass (every
next-step prediction supervised per clip, a superset of what windowed training
supervises).
"""
import random
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_provider import FeatureProvider


class ClipFeatureDataset(Dataset):
    """variable_length_min (default None=disabled): synthetically simulates clips
    shorter than the usual n_frames, WITHOUT a new data-loading path -- every
    __getitem__ still reads the normal fixed-T precomputed features, then (if enabled)
    draws a random valid length L in [variable_length_min, T], zeros out frames[L:]
    (trailing padding only -- a clip never has real content after a padded frame, which
    is what lets every downstream consumer rely on causal attention already being unable
    to let a real frame see a later, padded one; see models/world_model.py and
    losses/world_model_loss.py's own docstrings for why only the LOSS needs a mask, not
    the predictor's attention mask), and returns the valid length alongside the
    zero-padded features.

    "length" is ALWAYS present in the returned dict (T when this feature is off, or
    whenever random_window picked a window shorter than T at window_size>0, so existing
    callers that don't look at it see no change, and collation never has to special-case
    a sometimes-missing key.
    """

    def __init__(self, video_paths: List[str], feature_provider: FeatureProvider,
                 window_size: int = 0, random_window: bool = True,
                 variable_length_min: int = None):
        self.video_paths = list(video_paths)
        self.provider = feature_provider
        self.window_size = window_size
        self.random_window = random_window
        self.variable_length_min = variable_length_min

    @classmethod
    def from_manifest(cls, manifest_path: str, feature_provider: FeatureProvider,
                       path_col: str = "video_path", window_size: int = 0,
                       random_window: bool = True, variable_length_min: int = None):
        df = pd.read_csv(manifest_path, dtype=str)
        if path_col not in df.columns:
            raise ValueError(f"{manifest_path} has no {path_col!r} column "
                              f"(columns: {list(df.columns)})")
        return cls(df[path_col].tolist(), feature_provider, window_size=window_size,
                   random_window=random_window, variable_length_min=variable_length_min)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        features = self.provider.get(video_path)
        if self.window_size > 0:
            span = self.window_size + 1
            t = features.shape[0]
            if t < span:
                raise ValueError(
                    f"window_size={self.window_size} needs {span} frames/clip, but "
                    f"{video_path} only has {t} (n_frames too small for this window_size)"
                )
            # Training draws a fresh crop. Validation uses the fixed centre crop so its
            # population does not change between epochs or depend on worker scheduling.
            start = random.randint(0, t - span) if self.random_window else (t - span) // 2
            features = features[start:start + span]

        length = features.shape[0]
        if self.variable_length_min is not None:
            if self.variable_length_min < 1 or self.variable_length_min > length:
                raise ValueError(f"variable_length_min={self.variable_length_min} must be "
                                  f"in [1, {length}] (this clip/window has {length} frames)")
            length = random.randint(self.variable_length_min, length)
            if length < features.shape[0]:
                features = features.clone()
                features[length:] = 0
        return {"features": features, "video_path": video_path, "length": length}


class RandomWindowDataset(Dataset):
    """n_frames consecutive frames at a fixed fps, from a random position in each clip: e.g.
    fps=4, n_frames=32 is a random 8 s of the clip. Frames come from the provider's dense
    per-clip live cache (FeatureProvider.get_dense), so every epoch draws a fresh window
    from the same cache. Validation (random_window=False) uses the centred window.

    short_clip_policy: clips with fewer than n_frames frames at this fps (duration <
    (n_frames - 1) / fps) are dropped ("drop") or have their last frame repeated to length
    ("pad", no mask -- the repeated frames look like a stalled clip). windows_per_clip > 1
    draws that many independent windows from each clip per epoch."""

    def __init__(self, video_paths: List[str], feature_provider: FeatureProvider, fps: float,
                 n_frames: int, random_window: bool = True, windows_per_clip: int = 1):
        self.video_paths = list(video_paths)
        self.provider = feature_provider
        self.fps, self.n_frames = fps, n_frames
        self.random_window, self.windows_per_clip = random_window, windows_per_clip

    @classmethod
    def from_manifest(cls, manifest_path: str, feature_provider: FeatureProvider, fps: float,
                       n_frames: int, path_col: str = "video_path", duration_col: str = "duration",
                       short_clip_policy: str = "drop", random_window: bool = True,
                       windows_per_clip: int = 1):
        if short_clip_policy not in ("drop", "pad"):
            raise ValueError(f"short_clip_policy must be 'drop' or 'pad', got {short_clip_policy!r}")
        df = pd.read_csv(manifest_path)
        for col in (path_col, duration_col):
            if col not in df.columns:
                raise ValueError(f"{manifest_path} has no {col!r} column (columns: {list(df.columns)})")
        if short_clip_policy == "drop":
            keep = df[duration_col].astype(float) >= (n_frames - 1) / fps
            print(f"[RandomWindowDataset] {n_frames} frames at {fps} fps = {n_frames / fps:.2f}s: "
                  f"dropping {int((~keep).sum())} of {len(df)} clips shorter than that")
            df = df[keep]
        return cls(df[path_col].tolist(), feature_provider, fps, n_frames,
                   random_window=random_window, windows_per_clip=windows_per_clip)

    def __len__(self):
        return len(self.video_paths) * self.windows_per_clip

    def __getitem__(self, idx):
        import numpy as np
        video_path = self.video_paths[idx % len(self.video_paths)]
        frames = self.provider.get_dense(video_path, self.fps)
        t = frames.shape[0]
        if t >= self.n_frames:
            start = random.randint(0, t - self.n_frames) if self.random_window else (t - self.n_frames) // 2
            window = np.array(frames[start:start + self.n_frames])
        else:
            window = np.concatenate([np.asarray(frames), np.repeat(np.asarray(frames[-1:]), self.n_frames - t, axis=0)])
        return {"features": torch.from_numpy(window).float(), "video_path": video_path}
