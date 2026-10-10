"""Section 13: pairwise ordering accuracy. Reuses the already-trained linear progress
probe's predictions as the scoring function — predicted_progress = w^T z + b is an affine
(hence order-preserving) transform of w^T z, so the probe's own output is exactly the
"linear scoring function" the spec asks for, with no need to fit a second redundant model.
"""
from typing import Dict, Optional

import numpy as np


def pairwise_ordering_accuracy(video_preds: Dict[str, np.ndarray], n_pairs: int = 2000,
                                rng: Optional[np.random.RandomState] = None) -> float:
    rng = rng or np.random.RandomState(0)
    vids = [v for v, p in video_preds.items() if len(p) >= 2]
    if not vids:
        return float("nan")

    correct, total = 0, 0
    for _ in range(n_pairs):
        vid = vids[rng.randint(len(vids))]
        preds = video_preds[vid]
        i, j = rng.choice(len(preds), size=2, replace=False)
        label = i < j
        pred = preds[i] < preds[j]
        correct += int(pred == label)
        total += 1
    return correct / total
