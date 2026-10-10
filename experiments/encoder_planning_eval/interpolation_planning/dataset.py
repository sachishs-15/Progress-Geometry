"""Loads precomputed features + a manifest into per-clip examples for the planning task
(see ../README.md): the whole (T,D) sequence, with z[0]/z[-1] as start/goal and z[1:-1]
as the intermediate frames to be filled in.

One example per CLIP (not per intermediate frame) -- PlanningTransformer fills in every
intermediate position in a single forward pass, so training/eval batch over whole clips.

Two ways to get that (T,D) sequence, both ending up in the same shape (same --raw vs.
--checkpoint/--config split progress_metrics/compute_progress_metrics.py uses, for the
same reason: no separate extraction script/intermediate file needed for testing a trained
checkpoint's own z):
  - Already cls-shaped (N,T,D) raw features (dinov2*, dinov3, vjepa2*, ...) -- used as-is.
  - A trained ProceduralWorldModel checkpoint (--checkpoint/--config): the SOURCE features
    file can be cls- OR patch-shaped (whatever that checkpoint's own encoder expects,
    e.g. (N,T,P,D) for token_mode=patch), and gets run through
    model.aggregate(model.encoder(x)) in memory, once, at load time -- the exact same
    call compute_progress_metrics.py uses -- to produce the (N,T,D) sequence this dataset
    actually indexes into.
"""
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

_EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
from eval_common.loading import load_world_model  # noqa: E402


def _load_checkpoint_encoder(checkpoint_path: str, config_path: str, device: str):
    """Load the encoder through the shared, constructor-signature-aware loader."""
    model, _, _ = load_world_model(
        checkpoint_path, config_path, device, with_predictor=False
    )
    return model


def _run_checkpoint_encoder(model, features: torch.Tensor, device: str, batch_size: int = 64) -> torch.Tensor:
    """features: (N,T,D) or (N,T,P,D) raw input this checkpoint's encoder expects, or
    (N,T,3,H,W) raw pixel frames for token_mode='raw' checkpoints (model.pixel_encoder
    runs first in that case, same order as ProceduralWorldModel.forward()).
    Returns (N,T,out_dim) -- model.aggregate(model.encoder(x)), batched to bound memory
    (patch-shaped inputs can be large: e.g. 1430x16x196x384 floats ~= 6.7GB)."""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device)
            if model.pixel_encoder is not None:
                batch = model.pixel_encoder(batch)
            z = model.aggregate(model.encoder(batch))  # (B, T, out_dim)
            chunks.append(z.cpu())
    return torch.cat(chunks, dim=0)


class PlanningDataset(Dataset):
    """One example per clip: the full (T, D) per-frame latent sequence. z[0] is the
    start, z[-1] is the goal, z[1:-1] are the intermediate frames to predict."""

    def __init__(self, features_path: str, manifest_path: str, path_col: str = "video_path",
                 checkpoint: Optional[str] = None, config: Optional[str] = None,
                 device: str = "cpu"):
        if (checkpoint is None) != (config is None):
            raise ValueError("checkpoint and config must be given together (or neither)")

        data = torch.load(features_path, map_location="cpu")
        features = data["features"]  # (N,T,D) or (N,T,P,D), depending on the source encoder
        video_paths = data["video_paths"]

        if checkpoint is not None:
            model = _load_checkpoint_encoder(checkpoint, config, device)
            features = _run_checkpoint_encoder(model, features, device)  # -> (N,T,out_dim)

        if features.dim() != 3:
            raise ValueError(f"PlanningDataset needs a cls-shaped (N,T,D) sequence (after "
                              f"the checkpoint's own encoder, if one was given), got shape "
                              f"{tuple(features.shape)} from {features_path}")

        path_to_idx = {p: i for i, p in enumerate(video_paths)}
        df = pd.read_csv(manifest_path, dtype=str)
        idxs: List[int] = [path_to_idx[p] for p in df[path_col] if p in path_to_idx]
        if not idxs:
            raise ValueError(f"no manifest rows in {manifest_path} matched any video_path "
                              f"in {features_path}")

        self.features = features[idxs].float()  # (M, T, D)
        self.n_clips, self.T, self.D = self.features.shape
        if self.T < 3:
            raise ValueError(f"need at least 3 frames/clip to have an intermediate frame, "
                              f"got T={self.T} from {features_path}")

    def __len__(self):
        return self.n_clips

    def __getitem__(self, i):
        return {"z": self.features[i]}  # (T, D); z[0]=start, z[-1]=goal, z[1:-1]=targets
