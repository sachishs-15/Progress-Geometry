"""Companion to materialize_ego4d_sample.py: extracts the DINOv2 CLS token (instead of
patch tokens) for the EXACT same 1000 (video, window) pairs that script chose, so the two
samples are a true apples-to-apples comparison -- same seed, same manifest filtering, same
sample() call, same per-row random.randint() call order reproduces the identical window
choices deterministically (the random module's seeded sequence doesn't depend on which
tensor gets read). Patch-mode's forward_features() already computes the CLS token
internally (it's just discarded by _compute_dense_patch, which only keeps
"x_norm_patchtokens") -- same backbone, same preprocessing, just reads "x_norm_clstoken"
instead. No live-cache reuse here (CLS was never cached), so every clip is a fresh decode +
forward pass.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.feature_provider import FeatureProvider

TRAIN_MANIFEST = "/home/sachish/proc-planning/data/ego4dgoal/ego4dgoal_train.csv"
N_FRAMES = 32
FPS = 4
N_SAMPLE = 1000
SEED = 0
OUT_PT = "/home/sachish/proc-planning/experiments/encoder_planning_eval/eval_common/results/ego4d_train_sample_1000_cls.pt"
OUT_CSV = "/home/sachish/proc-planning/experiments/encoder_planning_eval/eval_common/results/ego4d_train_sample_1000_cls.csv"


def dense_frame_idx(video_path: str, fps: float):
    """Same math as feature_provider.py's _compute_dense_patch: frame k (at `fps`) maps to
    video frame idx[k] = floor(k / fps * video_fps), clamped to the last frame."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    video_fps, n_total = vr.get_avg_fps(), len(vr)
    n_out = int((n_total - 1) / video_fps * fps) + 1
    idx = np.minimum((np.arange(n_out) / fps * video_fps).astype(int), n_total - 1)
    return vr, idx


if __name__ == "__main__":
    random.seed(SEED)
    df = pd.read_csv(TRAIN_MANIFEST)
    min_duration = (N_FRAMES - 1) / FPS
    df = df[df["duration"].astype(float) >= min_duration].reset_index(drop=True)
    sample = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    provider = FeatureProvider(
        features_path=None, live_cache_dir="/mnt/rdata7/proprogress/data/ego4dgoal/live_cache/pwm_dense_4fps",
        n_frames=N_FRAMES, model_id="vit_small_patch14_reg4_dinov2.lvd142m", device="cuda",
        token_mode="patch", patch_hub_id="dinov2_vits14", patch_img_size=196, raw_img_size=224,
    )
    provider._load_backbone_patch()

    cls_windows, video_paths = [], []
    n_padded = 0
    for i, row in enumerate(sample.itertuples()):
        # Same get_dense() call as materialize_ego4d_sample.py, purely to reproduce the
        # identical `t` and therefore the identical random.randint() draw -- cheap, reads
        # the already-cached dense patch file, no recompute.
        frames = provider.get_dense(row.video_path, FPS)
        t = frames.shape[0]
        vr, idx = dense_frame_idx(row.video_path, FPS)
        assert len(idx) == t, f"{row.video_path}: dense cache t={t} but recomputed idx has {len(idx)}"

        if t >= N_FRAMES:
            start = random.randint(0, t - N_FRAMES)
            frame_idx = idx[start:start + N_FRAMES]
        else:
            frame_idx = np.concatenate([idx, np.repeat(idx[-1:], N_FRAMES - t)])
            n_padded += 1

        raw_frames = vr.get_batch(frame_idx).asnumpy()
        x = torch.stack([provider._preprocess(Image.fromarray(f)) for f in raw_frames]).to(provider.device)
        with torch.no_grad():
            cls = provider._backbone.forward_features(x)["x_norm_clstoken"]  # (32, 384)
        cls_windows.append(cls.cpu().clone())
        video_paths.append(row.video_path)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{N_SAMPLE} ({n_padded} padded so far)")

    features = torch.stack(cls_windows).float()  # (1000, 32, 384)
    print("features shape:", tuple(features.shape))
    torch.save({"features": features, "video_paths": video_paths}, OUT_PT)
    pd.DataFrame({"video_path": video_paths}).to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_PT}, {OUT_CSV}")
