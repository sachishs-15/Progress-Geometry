"""One-off: materialize N randomly-sampled, randomly-windowed 32-frame/4fps DINOv2 patch
clips from Ego4D GoalStep train, using the same FeatureProvider + window-sampling logic
train.py uses (RandomWindowDataset with random_window=True), reusing whatever's already
in live_cache_dir (17126/27600 train videos already cached) and computing the rest fresh.
Saves a standard {"features","video_paths"} .pt file plus a matching minimal manifest CSV,
so the existing eval scripts (compute_progress_metrics.py --raw, run_encoder_pooled_eval.py)
work unmodified.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.feature_provider import FeatureProvider

TRAIN_MANIFEST = "/home/sachish/proc-planning/data/ego4dgoal/ego4dgoal_train.csv"
LIVE_CACHE_DIR = "/mnt/rdata7/proprogress/data/ego4dgoal/live_cache/pwm_dense_4fps"
N_FRAMES = 32
FPS = 4
N_SAMPLE = 1000
SEED = 0
OUT_PT = "/home/sachish/proc-planning/experiments/encoder_planning_eval/eval_common/results/ego4d_train_sample_1000.pt"
OUT_CSV = "/home/sachish/proc-planning/experiments/encoder_planning_eval/eval_common/results/ego4d_train_sample_1000.csv"

if __name__ == "__main__":
    random.seed(SEED)
    df = pd.read_csv(TRAIN_MANIFEST)
    min_duration = (N_FRAMES - 1) / FPS
    df = df[df["duration"].astype(float) >= min_duration].reset_index(drop=True)
    print(f"{len(df)} of train clips long enough for {N_FRAMES} frames @ {FPS}fps ({min_duration:.2f}s)")
    sample = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    provider = FeatureProvider(
        features_path=None, live_cache_dir=LIVE_CACHE_DIR,
        n_frames=N_FRAMES, model_id="vit_small_patch14_reg4_dinov2.lvd142m", device="cuda",
        token_mode="patch", patch_hub_id="dinov2_vits14", patch_img_size=196, raw_img_size=224,
    )

    # Short clips (t < N_FRAMES, the manifest's `duration` column is only an estimate of
    # the actual decoded/cached frame count) are padded by repeating the last frame --
    # same convention as RandomWindowDataset's short_clip_policy='pad' -- rather than
    # dropped, so every sampled row is kept.
    windows, video_paths = [], []
    n_padded = 0
    for i, row in enumerate(sample.itertuples()):
        frames = provider.get_dense(row.video_path, FPS)  # (t, 196, 384)
        t = frames.shape[0]
        if t >= N_FRAMES:
            start = random.randint(0, t - N_FRAMES)
            window = np.array(frames[start:start + N_FRAMES])
        else:
            window = np.concatenate([np.asarray(frames),
                                      np.repeat(np.asarray(frames[-1:]), N_FRAMES - t, axis=0)])
            n_padded += 1
        windows.append(window)
        video_paths.append(row.video_path)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{N_SAMPLE} ({n_padded} padded so far)")

    features = torch.from_numpy(np.stack(windows)).float()  # (1000, 32, 196, 384)
    print("features shape:", tuple(features.shape))
    torch.save({"features": features, "video_paths": video_paths}, OUT_PT)
    pd.DataFrame({"video_path": video_paths}).to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_PT}, {OUT_CSV}")
