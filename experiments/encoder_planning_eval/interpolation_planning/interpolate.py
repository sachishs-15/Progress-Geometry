"""Training-free baseline: z_t_hat = z1 + alpha*(zT - z1), the direct test of whether an
encoder's latent space is already "straight" between start and goal (see ../README.md).
"""
import torch


def linear_interpolate(z1: torch.Tensor, zT: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """z1, zT: (..., D). alpha: (...,) or a python scalar in [0, 1]. Returns (..., D)."""
    if torch.is_tensor(alpha):
        alpha = alpha.unsqueeze(-1)
    return z1 + alpha * (zT - z1)


def interpolate_sequence(z1: torch.Tensor, zT: torch.Tensor, t_len: int) -> torch.Tensor:
    """Whole-clip convenience for PlanningTransformer's per-clip batching: z1, zT: (B, D)
    -> (B, T, D), linearly interpolated at every position (including 0/T-1, which land
    back on exactly z1/zT). alpha[t] = t/(T-1)."""
    alpha = torch.linspace(0, 1, t_len, device=z1.device, dtype=z1.dtype)  # (T,)
    return z1.unsqueeze(1) + alpha.view(1, -1, 1) * (zT - z1).unsqueeze(1)
