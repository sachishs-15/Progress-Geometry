"""Temporal-ordering objective: sample K chronological timesteps, shuffle them, and train
a model to recover each shuffled element's original rank (section 6-8 of the spec).
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def shuffle_permutation(k: int, strength: str = "strong", sigma: float = None,
                         rng: np.random.RandomState = None) -> np.ndarray:
    """Return `perm` (length k) such that shuffled[i] = chronological[perm[i]] — i.e.
    shuffled slot i holds the element whose original rank was perm[i], so `perm` IS the
    ordering target (section 6: "target = 0..K-1 after undoing the shuffle").

    Implemented as one continuous knob (section 18's "mild vs strong shuffle" as two
    presets of it): give each chronological position `pos` a noisy key `pos + N(0, sigma)`
    and argsort by the noisy key. sigma -> 0 keeps the permutation close to identity (only
    nearby positions swap = mild/local shuffle); large sigma washes out the original order
    entirely (~uniform random permutation = strong shuffle).
    """
    rng = rng or np.random.RandomState()
    if sigma is None:
        sigma = {"mild": 1.0, "strong": 1000.0}[strength]
    noisy_keys = np.arange(k, dtype=np.float64) + rng.normal(0.0, sigma, size=k)
    return np.argsort(noisy_keys)


def ordering_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: [B, K, K] (per-slot distribution over original ranks), targets: [B, K]
    (original rank of each shuffled slot). Cross entropy over the rank classes."""
    b, k, _ = logits.shape
    return nn.functional.cross_entropy(logits.reshape(b * k, k), targets.reshape(b * k))
