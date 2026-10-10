"""Feature-level injections that mimic procedural mistakes, applied to one clip's
per-frame features. A clip is a tensor with the frame axis first: (T, ...features), for
example (16, 196, 384) DINOv2 patch tokens or (16, 384) CLS vectors. Because the world
model's encoder is frame-wise, frame selection and reordering commute with encoding.
Feature-space interpolation and appearance shifts are approximations to pixel changes.

Every injection returns (y, seams) or None when it cannot be applied:
    y      the perturbed clip, (T', ...)
    seams  transition indices t, meaning the discontinuity sits between frames t and t+1
           of y. Empty for global changes (reverse, pace, appearance).

Length-preserving injections keep T = 16 frames. drop and speed_up cannot (they remove
or skip frames), so they produce 12 frames and are compared with a 12-frame clean
window of the same clip.
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F

BLOCK = 4
LEN_FULL = 16
LEN_SHORT = 12

# injection -> name of the clean reference it is compared with
REFERENCE = {
    "reverse": "clean16", "swap": "clean16", "shuffle": "clean16", "splice_other": "clean16",
    "splice_same": "clean16", "freeze": "clean16", "slow_down": "clean16",
    "appearance": "clean16", "blur": "clean16", "drop": "clean12", "speed_up": "clean12",
}
PIXEL_ONLY = ("blur",)      # need real RGB frames, so they are skipped for feature inputs
NEEDS_DONOR = {"splice_other": "other", "splice_same": "same", "appearance": "other"}
SEAM_INJECTIONS = ("swap", "shuffle", "drop", "splice_other", "splice_same", "freeze")


def rng_for(seed, name, index):
    """Deterministic per (seed, injection, clip) so every model sees identical inputs."""
    return np.random.default_rng([seed, zlib.crc32(name.encode()), int(index)])


def seams_from_source(src):
    """Transitions where the source frame index does not advance by exactly 1."""
    return [t for t in range(len(src) - 1) if src[t + 1] != src[t] + 1]


def clean_window(x, rng, length=LEN_SHORT):
    """A random contiguous `length`-frame window of the clip."""
    start = int(rng.integers(0, x.shape[0] - length + 1))
    return x[start:start + length], []


def reverse(x, rng, donor=None):
    return x.flip(0), []


def swap(x, rng, donor=None):
    t, b = x.shape[0], BLOCK
    i = int(rng.integers(0, t - 2 * b + 1))
    j = int(rng.integers(i + b, t - b + 1))
    src = list(range(t))
    src[i:i + b], src[j:j + b] = list(range(j, j + b)), list(range(i, i + b))
    return x[torch.tensor(src)], seams_from_source(src)


def shuffle(x, rng, donor=None):
    t, b = x.shape[0], BLOCK
    n = t // b
    while True:
        perm = [int(k) for k in rng.permutation(n)]
        if perm != list(range(n)):
            break
    src = [k * b + o for k in perm for o in range(b)]
    return x[torch.tensor(src)], seams_from_source(src)


def drop(x, rng, donor=None):
    t, b = x.shape[0], BLOCK
    i = int(rng.integers(2, t - b - 1))          # block starts at 2..t-b-2, neighbours exist
    src = list(range(i)) + list(range(i + b, t))
    return x[torch.tensor(src)], seams_from_source(src)


def splice(x, rng, donor):
    if donor is None:
        return None
    t, b = x.shape[0], BLOCK
    i = int(rng.integers(2, t - b - 1))
    o = int(rng.integers(0, donor.shape[0] - b + 1))
    y = x.clone()
    y[i:i + b] = donor[o:o + b]
    return y, [i - 1, i + b - 1]


def freeze(x, rng, donor=None):
    t, b = x.shape[0], BLOCK
    i = int(rng.integers(2, t - b - 1))
    y = x.clone()
    y[i + 1:i + b] = x[i]                        # hold frame i for b frames, then jump
    return y, [i + b - 1]


def speed_up(x, rng, donor=None):
    """First 8 frames at normal pace, then every second frame: 12 frames spanning 16."""
    src = list(range(8)) + [9, 11, 13, 15]
    return x[torch.tensor(src)], []


def slow_down(x, rng, donor=None):
    """First 8 frames at normal pace, then half pace using linearly interpolated frames
    (7.5, 8, 8.5, ..., 11). Length stays 16."""
    pos = [float(k) for k in range(8)] + [7.5 + 0.5 * k for k in range(8)]
    lo = torch.tensor([int(np.floor(p)) for p in pos])
    hi = torch.tensor([min(int(np.ceil(p)), x.shape[0] - 1) for p in pos])
    w = torch.tensor([p - np.floor(p) for p in pos], dtype=x.dtype).reshape(-1, *([1] * (x.dim() - 1)))
    return x[lo] * (1 - w) + x[hi] * w, []


def is_pixels(x):
    """(T, 3, H, W) RGB frames, as opposed to (T, P, D) patch tokens or (T, D) vectors."""
    return x.dim() == 4 and x.shape[1] == 3


def appearance(x, rng, donor=None, strength=0.5):
    """A constant shift applied to every frame, leaving the dynamics alone. For feature
    inputs it moves the features toward another clip's mean appearance (a stand-in for a
    colour or camera change); for RGB frames it is a genuine global colour cast toward
    the donor's mean colour, clamped to [0, 1]."""
    if donor is None:
        return None
    if is_pixels(x):
        delta = strength * (donor.mean(dim=(0, 2, 3), keepdim=True) - x.mean(dim=(0, 2, 3), keepdim=True))
        return (x + delta).clamp(0, 1), []
    axes = tuple(range(x.dim() - 1))
    delta = strength * (donor.mean(dim=axes) - x.mean(dim=axes))
    return x + delta, []


def blur(x, rng, donor=None, sigma=2.0):
    """Gaussian blur of every RGB frame (negative control: dynamics unchanged). Only
    defined for pixel inputs."""
    if not is_pixels(x):
        return None
    k = 2 * int(round(3 * sigma)) + 1
    g = torch.exp(-(torch.arange(k, dtype=x.dtype) - k // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kern = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kern, groups=3), []


INJECTIONS = {
    "reverse": reverse, "swap": swap, "shuffle": shuffle, "drop": drop,
    "splice_other": splice, "splice_same": splice, "freeze": freeze,
    "speed_up": speed_up, "slow_down": slow_down, "appearance": appearance, "blur": blur,
}
