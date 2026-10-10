"""Alternatives to plain per-frame nn.LayerNorm for ClsMlpEncoder's output, all sharing
its (B, T, D) -> (B, T, D) calling convention. Exists because LayerNorm computes an
INDEPENDENT statistic for every frame, which can (and, per this project's own K400 runs,
does) rescale frames within one clip by different amounts -- distorting the trajectory's
straightness, since a straight line only stays straight under a transform that treats
every point on it identically. See encoder.py's docstring and
checkpoints/pwm_cls_mlp_k400* / ablation_clip_rmsnorm for the investigation this came from.
"""
import torch
import torch.nn as nn


class ClipRMSNorm(nn.Module):
    """One RMS scalar per CLIP (pooled over time AND feature dims jointly), applied
    identically to every frame in that clip. A uniform positive rescale can't distort
    collinearity/angles/straightness within a clip, unlike LayerNorm's per-frame
    statistic. Also does not mean-center (unlike LayerNorm), so it can't erase
    straight-line motion aligned with the all-ones direction the way LayerNorm's
    per-frame mean-subtraction can.

    Verified (ablation_clip_rmsnorm, 200 epochs / 10k K400 clips): matched or beat both
    plain LayerNorm and raw DINOv2 on SSv2/K400 ordinality metrics, while keeping z_norm
    nearly flat (19.6->21.4) versus LayerNorm's 19.6->24 at the same scale (->146 at the
    full 500-epoch/241k-clip scale).

    Limitation: only guarantees consistency WITHIN one clip -- different clips can still
    land at different absolute scales from each other (unlike BatchNormOverFrames below).
    """
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        """x: (B, T, D)"""
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()  # (B, 1, 1)
        x = x / rms
        return x * self.weight if self.weight is not None else x


class BatchNormOverFrames(nn.Module):
    """nn.BatchNorm1d, wrapped to accept (B, T, D): batch and time are flattened together
    (every frame from every clip in the batch is one "example" for normalization
    purposes), giving per-dimension statistics across all of them, with a running EMA
    used at eval time. Unlike ClipRMSNorm, this makes EVERY frame from EVERY clip share
    the exact same normalizing statistic -- at eval time, literally one fixed affine
    transform for the whole dataset, the strongest "same latent space" guarantee of the
    three options here. Tradeoff: a train/eval behavioral difference and running-buffer
    bookkeeping neither LayerNorm nor ClipRMSNorm have, and can be noisy early in
    training before the running average stabilizes.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        b, t, d = x.shape
        x = self.bn(x.reshape(b * t, d))
        return x.reshape(b, t, d)


class HypersphereNorm(nn.Module):
    """Hard projection onto the unit hypersphere: z / ||z||, so ||z|| == 1 exactly for
    every frame.

    The other three options all leave the vector's LENGTH free. LayerNorm fixes
    per-FEATURE variance, ClipRMSNorm applies one RMS scalar per clip, BatchNormOverFrames
    fixes per-dimension statistics across the batch -- none of them constrain ||z||. That
    gap is not academic: checkpoints/pwm_cls_mlp_k400 (LayerNorm + VICReg) inflated ||z||
    from 20 to 146 over 500 epochs while VICReg's std penalty fell to ~0.003, i.e. the
    anti-collapse term reported "healthier than raw DINOv2" the whole time because
    relu(1 - std) is satisfiable by scale alone. Post-hoc the representation had ~4x LESS
    directional diversity than raw DINOv2 (per-dim std after L2-normalizing: 0.013 vs
    0.048). HypersphereNorm removes that escape hatch by construction.

    It also makes training match evaluation: encoder_planning_eval's clip_progress_metrics
    L2-normalizes every frame before computing anything, so with any other norm_type the
    losses shape a different object than the metrics score.

    Caveat worth knowing: under HypersphereNorm the radial direction of the PRE-norm activations is
    in the output's null space, so nothing in the loss constrains ||z_pre||. Measured over
    100 epochs on SSv2 it fell (4.97 -> 2.71) rather than drifting up, but with
    weight_decay=0 there is no mechanism that would stop it if it did.

    No learnable scale on purpose -- a scale parameter would immediately reintroduce the
    freedom this class exists to remove.
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D), each frame unit-norm."""
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)


NORM_REGISTRY = {
    "layernorm": lambda dim: nn.LayerNorm(dim),
    "clip_rmsnorm": lambda dim: ClipRMSNorm(dim),
    "batchnorm": lambda dim: BatchNormOverFrames(dim),
    "hypersphere": lambda dim: HypersphereNorm(dim),
    "none": lambda dim: nn.Identity(),
}
