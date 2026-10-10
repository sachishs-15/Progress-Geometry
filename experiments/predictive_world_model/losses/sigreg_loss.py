"""SIGReg (Sketched Isotropic Gaussian Regularization), from LeJEPA (arXiv:2511.08544,
github.com/rbalestr-lab/lejepa) -- a principled alternative to VICReg's std/cov
heuristics: instead of separately bounding per-dimension variance and decorrelating
dimensions, directly tests whether the embedding distribution is an isotropic standard
Gaussian, via the Cramer-Wold device (a distribution is standard Gaussian in D dimensions
iff every 1-D projection of it is standard Gaussian).

Mechanism: project the batch onto `num_slices` random unit directions ("sketching"),
then for each projection run the Epps-Pulley normality test -- compare the projected
values' empirical characteristic function against the standard normal's phi(t) =
exp(-t^2/2) at `n_points` quadrature points over [0, t_max] (only t >= 0 needed; cos is
even and sin is odd, so the -t contributions are already folded into the doubled
trapezoid weights below), and average the resulting per-slice statistic over slices.

Formula and defaults (t_max=3, n_points=17, num_slices=1024) verified directly against
lejepa/univariate/epps_pulley.py's EppsPulley class -- this is that class's forward()
inlined as a stateless function (no persistent buffers/module needed: t_max/n_points
never change per-call, and re-deriving the tiny 17-point grid each call is negligible
next to the num_slices-wide projection matmul), with a fresh random projection matrix
drawn every call rather than a fixed one, matching the "sketch" being re-randomized each
training step rather than fixed once.
"""
import torch


def sigreg_loss(z, num_slices=1024, t_max=3.0, n_points=17, grouped=False):
    """z: (..., D) -- flattened to (N, D) internally, one row per frame/token whose
    marginal distribution across the batch should be an isotropic standard Gaussian.

    grouped=True: z is (G, N, D) instead -- G independent sample sets, each tested on its
    own (N rows) with the SAME random projections, and the per-group statistics averaged.
    This is how the regularizer is applied along one axis only (e.g. LeWM's per-timestep
    SIGReg: G=T groups of N=B clips); see world_model_loss._group. G=1 is identical to the
    ungrouped call."""
    x = z if grouped else z.reshape(1, -1, z.shape[-1])
    _, n, d = x.shape

    a = torch.randn(d, num_slices, device=x.device, dtype=x.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-8)
    proj = x @ a  # (G, N, num_slices)

    t = torch.linspace(0, t_max, n_points, device=x.device, dtype=x.dtype)
    dt = t_max / (n_points - 1)
    weights = torch.full((n_points,), 2 * dt, device=x.device, dtype=x.dtype)
    weights[0] = dt
    weights[-1] = dt  # half-weight at the endpoints (trapezoid rule)
    phi = torch.exp(-0.5 * t.square())  # standard normal's characteristic function, real
    weights = weights * phi

    x_t = proj.unsqueeze(-1) * t  # (G, N, num_slices, n_points)
    cos_mean = torch.cos(x_t).mean(dim=1)  # (G, num_slices, n_points)
    sin_mean = torch.sin(x_t).mean(dim=1)

    err = (cos_mean - phi).square() + sin_mean.square()  # |ECF(t) - phi(t)|^2
    per_slice = err @ weights  # (G, num_slices) -- weighted integral over t
    return per_slice.mean() * n  # scaled by per-group N, matching the reference
