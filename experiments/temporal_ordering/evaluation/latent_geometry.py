"""Section 14: does distance in the learned latent space correspond to difference in
procedural progress? Pairs are sampled from the pooled set of all test-video timesteps
(any two timesteps, possibly from different videos) per the spec's "for test videos ...
randomly selected pairs" wording (unlike the same-video-only pairwise-ordering metric).
"""
from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr


def distance_progress_analysis(video_reprs: Dict[str, np.ndarray], video_progress: Dict[str, np.ndarray],
                                n_pairs: int = 2000, rng: Optional[np.random.RandomState] = None) -> dict:
    rng = rng or np.random.RandomState(0)
    pool = [(vid, t) for vid, z in video_reprs.items() for t in range(z.shape[0])]
    if len(pool) < 2:
        return {"spearman": float("nan"), "distances": np.array([]), "progress_diffs": np.array([])}

    idx_a = rng.randint(len(pool), size=n_pairs)
    idx_b = rng.randint(len(pool), size=n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]

    distances = np.empty(len(idx_a), dtype=np.float64)
    progress_diffs = np.empty(len(idx_a), dtype=np.float64)
    for n, (a, b) in enumerate(zip(idx_a, idx_b)):
        vid_a, ta = pool[a]
        vid_b, tb = pool[b]
        za, zb = video_reprs[vid_a][ta], video_reprs[vid_b][tb]
        distances[n] = np.linalg.norm(za - zb)
        progress_diffs[n] = abs(video_progress[vid_a][ta] - video_progress[vid_b][tb])

    if len(distances) < 2 or np.all(distances == distances[0]):
        rho = float("nan")
    else:
        rho, _ = spearmanr(distances, progress_diffs)
        rho = float(rho)
    return {"spearman": rho, "distances": distances, "progress_diffs": progress_diffs}
