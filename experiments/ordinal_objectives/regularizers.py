"""This project's own regularizers -- the ones that need something
predictive_world_model/losses/world_model_loss.py's terms never see: the progress LABEL.

Named regularizers.py, not losses.py, because pwm_bridge puts predictive_world_model/ at
the front of sys.path and `losses` is its package (see that module's docstring).
"""
import torch


def ordinal_entropy(z, progress, eps=1e-8):
    """Ordinal Entropy (Zhang et al., ICLR 2023, arXiv:2301.08915), Eqs. 8-9.

    Their argument is that cross-entropy on a discretised target beats MSE on the same
    continuous target because it learns a HIGHER-ENTROPY feature space, and that you can
    have that benefit without discretising by regularizing the features directly. Two
    terms, returned separately so train.py can weight them independently as the paper's
    lambda_d / lambda_t:

      diversity (Eq. 8)  L_d = -1/(M(M-1)) sum_{i != j} w_ij ||z_ci - z_cj||_2
                         with w_ij = |y_i - y_j|, z_c the mean feature per label.
                         NEGATIVE, so minimising it spreads the label centres apart --
                         and the w_ij weighting is what makes it ordinal rather than just
                         high-entropy: centres whose labels are far apart are pushed
                         apart harder than adjacent ones.

      tightness (Eq. 9)  L_t = 1/N sum_i ||z_i - z_ci||_2, pulling each frame to its own
                         label's centre.

    This is the one regularizer here that directly tests the ce-vs-regression claim the
    ordinal_classification configs are built around: if the gap between ordinal_ce and
    progress_regression really is a feature-entropy effect, adding this to
    progress_regression should close it.

    z: (B, T, D) latents. progress: (B, T) in [0,1]. Returns (L_d, L_t).

    Labels are grouped by exact value, which is what we want here: progress takes only T
    distinct values (i/(T-1)), so a batch of 128 clips gives M=16 centres of ~128 frames
    each -- well-populated, unlike the paper's own setting where they have to bin.
    """
    z = z.reshape(-1, z.shape[-1])
    y = progress.reshape(-1)
    labels, inverse = torch.unique(y, return_inverse=True)
    m = labels.numel()
    if m < 2:
        return z.sum() * 0.0, z.sum() * 0.0

    # Per-label feature centres, via scatter-add rather than a Python loop over labels.
    centres = torch.zeros(m, z.shape[-1], device=z.device, dtype=z.dtype)
    centres.index_add_(0, inverse, z)
    counts = torch.zeros(m, device=z.device, dtype=z.dtype).index_add_(
        0, inverse, torch.ones_like(y))
    centres = centres / counts.unsqueeze(-1).clamp(min=1.0)

    centre_dist = torch.cdist(centres, centres, p=2.0)          # ||z_ci - z_cj||
    label_dist = (labels.unsqueeze(1) - labels.unsqueeze(0)).abs()   # w_ij
    off = ~torch.eye(m, dtype=torch.bool, device=z.device)
    diversity = -(label_dist[off] * centre_dist[off]).sum() / (m * (m - 1))

    tightness = (z - centres[inverse]).norm(dim=-1).mean()
    return diversity, tightness
