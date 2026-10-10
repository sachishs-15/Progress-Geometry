"""Comparison metrics between predicted and true intermediate latents. Cosine similarity
and relative L2 are scale-invariant, so they stay comparable across encoders with very
different ||z|| (raw L2 would not -- see predictive_world_model/models/norms.py's own
discussion of how much ||z|| varies across encoders/training runs).
"""
import torch
import torch.nn.functional as F


def cosine_sim(z_hat: torch.Tensor, z_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.cosine_similarity(z_hat, z_true, dim=-1, eps=eps)


def relative_l2(z_hat: torch.Tensor, z_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """||z_hat - z_true|| / ||z_true|| -- roughly scale-invariant, unlike raw L2 distance."""
    return (z_hat - z_true).norm(dim=-1) / (z_true.norm(dim=-1) + eps)


def r2_score(z_hat: torch.Tensor, z_true: torch.Tensor) -> float:
    """Global vector R² based on squared Euclidean distances over the (N, D) batch.

    This computes ``1 - sum_i ||z_true_i - z_hat_i||² /
    sum_i ||z_true_i - mean(z_true)||²``. Equivalently, it aggregates residual and
    total sums of squares across dimensions before taking their ratio, so dimensions
    are weighted by their target variance rather than averaged uniformly. 1.0 is
    perfect; 0.0 matches always predicting the batch mean; negative values are worse
    than that baseline.
    """
    ss_res = (z_true - z_hat).pow(2).sum()
    ss_tot = (z_true - z_true.mean(dim=0, keepdim=True)).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


def summarize(z_hat: torch.Tensor, z_true: torch.Tensor) -> dict:
    """z_hat, z_true: (N, D). Medians for cosine_sim/relative_l2 match
    progress_metrics.py's own median-over-clips convention (robust to outlier clips);
    means are also reported since they're what r2_score is effectively an analogue of."""
    cos = cosine_sim(z_hat, z_true)
    rel = relative_l2(z_hat, z_true)
    return {
        "cosine_sim_median": cos.median().item(),
        "cosine_sim_mean": cos.mean().item(),
        "relative_l2_median": rel.median().item(),
        "relative_l2_mean": rel.mean().item(),
        "r2": r2_score(z_hat, z_true),
    }
