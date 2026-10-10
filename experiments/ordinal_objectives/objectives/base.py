"""Common interface every objective implements, plus the sampling helpers more than one
of them needs.

The interface is deliberately uniform so that train.py never branches on which objective
is selected, and so that adding a new one from the literature is a single new file plus
one registry line (same convention as predictive_world_model/models/norms.py's
NORM_REGISTRY and encoder_planning_eval/interpolation_planning/predictors/'s
PREDICTOR_REGISTRY):

    forward(z, progress, frame_index=None, z_pair=None, progress_pair=None,
            frame_index_pair=None) -> (loss, metrics)

  z             (B, T, d)  encoder latents for the whole (possibly cropped) clip. The
                           quantity under test -- whatever geometry an objective induces,
                           it induces it here.
  progress      (B, T)     progress label in [0, 1] (see clip_data.py).
  frame_index   (B, T)     index within the source clip. Objectives that need a
                           "timestamp" independent of the crop_label convention use this;
                           objectives supervised on progress use `progress`.
  *_pair                   the same tensors for a paired clip of the same activity, or
                           None. An objective that needs one sets requires_pair = True and
                           train.py refuses to start without --pair-col.
  metrics       dict of detached scalars, logged per epoch alongside the loss. Values stay
                on-device and undetached-to-Python (same reason as
                predictive_world_model/losses/world_model_loss.py's loss_dict: one
                host<->device sync per epoch, not one per term per step).

Objectives own their own frame sampling. They receive all T encoded frames and index into
them, because the encoder is per-frame and one forward pass therefore serves any
subsample (see clip_data.py's docstring).
"""
import torch
import torch.nn as nn


class Objective(nn.Module):
    #: True if this objective cannot be computed without a second clip of the same
    #: activity (TCC in its cross-video form). Checked by train.py before the first batch,
    #: so a missing --pair-col fails immediately rather than at step 0.
    requires_pair = False

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        raise NotImplementedError


def sorted_subset_indices(b: int, t: int, k: int, device, generator=None):
    """(b, k) long tensor: per row, k distinct indices from [0, t), sorted ascending.

    torch.argsort(torch.rand(...)) rather than a Python loop over the batch -- this runs
    every step, per objective, and a per-row randperm on CPU would show up in the profile
    well before the model does.
    """
    if k > t:
        raise ValueError(f"cannot sample {k} distinct frames from a clip of length {t}")
    noise = torch.rand(b, t, device=device, generator=generator)
    idx = noise.argsort(dim=1)[:, :k]
    return idx.sort(dim=1).values


def gather_frames(z, idx):
    """z: (B, T, d), idx: (B, K) -> (B, K, d)."""
    return z.gather(1, idx.unsqueeze(-1).expand(-1, -1, z.size(-1)))


def gather_scalar(x, idx):
    """x: (B, T), idx: (B, K) -> (B, K)."""
    return x.gather(1, idx)


def random_permutations(b: int, k: int, device, generator=None):
    """(b, k) long tensor of independent random permutations of [0, k)."""
    return torch.rand(b, k, device=device, generator=generator).argsort(dim=1)


def pairwise_sqdist(a, b):
    """a: (..., N, d), b: (..., M, d) -> (..., N, M) squared euclidean distances.

    torch.cdist(...)**2 would be the obvious call, but cdist's fast path is a matmul
    expansion whose gradients are numerically poor near zero distance -- and near-zero
    distances are exactly what TCC's soft-nearest-neighbour and TCN's positive term spend
    their time at. The explicit expansion below is the same arithmetic with a
    well-behaved gradient everywhere.
    """
    return (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1)
