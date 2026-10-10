"""Qualitative latent-space plots (section 15 / 14). `pca_2d` is adapted from
`experiments/frozen_encoders/common/viz.py::reduce_dim` (PCA branch only — we don't need
the t-SNE path or the PIL-frame/attention-overlay machinery that module also has, since we
only ever visualize CrossTask-style feature vectors, not raw video frames).
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_2d(x: np.ndarray) -> np.ndarray:
    return PCA(n_components=2).fit_transform(x)


def plot_pca_trajectories(video_reprs: Dict[str, np.ndarray], video_progress: Dict[str, np.ndarray],
                           out_path: str, title: str, n_videos: int = 6, seed: int = 0) -> None:
    """One PCA(2) scatter/trajectory per selected test video, points colored by normalized
    progress. A good ordinal representation should trace an early->middle->late path."""
    rng = np.random.RandomState(seed)
    vids = sorted(video_reprs)
    if len(vids) > n_videos:
        vids = list(rng.choice(vids, size=n_videos, replace=False))

    n = len(vids)
    ncols = min(3, n) or 1
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for i, vid in enumerate(vids):
        ax = axes[i // ncols][i % ncols]
        proj = pca_2d(video_reprs[vid])
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=video_progress[vid], cmap="viridis", s=18, zorder=2)
        ax.plot(proj[:, 0], proj[:, 1], c="gray", alpha=0.4, linewidth=1, zorder=1)
        ax.set_title(vid, fontsize=9)
        fig.colorbar(sc, ax=ax, label="progress")

    for i in range(len(vids), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_distance_vs_progress(distances: np.ndarray, progress_diffs: np.ndarray, spearman_rho: float,
                               out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(progress_diffs, distances, s=8, alpha=0.35)
    ax.set_xlabel("|progress_i - progress_j|")
    ax.set_ylabel("||z_i - z_j||")
    ax.set_title(f"{title}\nSpearman(distance, progress diff) = {spearman_rho:.3f}", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
