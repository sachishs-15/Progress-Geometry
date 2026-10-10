"""Closed-form linear 'oracle' encoders: what the best LINEAR map of frozen features can reach.

A trained encoder head is a (nonlinear) learned map of the same frozen features. If even the
analytic optimum of a linear map cannot reach a metric value, a claimed gain beyond it must come
from non-linearity or from changing what information is kept; if a trained model falls short of it,
there is headroom a better optimiser could in principle recover.

fit_sfa            Slow Feature Analysis (Wiskott & Sejnowski 2002), closed form.
                   Minimise  E||W^T d^k z||^2  s.t.  W^T C W = I   (generalised eigenproblem),
                   d^k = k-th temporal difference (k=1 slowness, k=2 'curvature'/acceleration).
                   C = pooled frame covariance ('pooled': keeps clip identity, so the output is
                   whitened and cannot rank-collapse) or within-clip covariance ('within').
time_readout       Ridge regression of the frame index on within-clip-centred frames: the best
                   LINEAR read-out of 'how far along' from a single frame. Its per-clip Spearman with
                   t is a decodability ceiling for the unsupervised chord-projection used by
                   time_progress_spearman.
"""
import numpy as np
from scipy import linalg


def _cov(x):
    x = x - x.mean(0, keepdims=True)
    return x.T @ x / (len(x) - 1)


def fit_sfa(ztr, k_out=128, order=1, whiten="pooled", ridge=1e-6):
    """ztr: (N,T,D) train latents. Returns dict(mu, W) with W: (D, k_out) whitened slow/straight axes."""
    n, t, d = ztr.shape
    flat = ztr.reshape(-1, d)
    mu = flat.mean(0)
    if whiten == "pooled":
        c = _cov(flat)
    else:
        c = _cov((ztr - ztr.mean(1, keepdims=True)).reshape(-1, d))
    dz = np.diff(ztr, n=order, axis=1).reshape(-1, d)
    a = dz.T @ dz / len(dz)
    c = c + ridge * np.trace(c) / d * np.eye(d)
    w, v = linalg.eigh(a, c)  # ascending: slowest/straightest first, v^T c v = I
    return {"mu": mu, "W": v[:, :k_out], "eig": w[:k_out]}


def apply(model, z):
    return (z - model["mu"]) @ model["W"]


def fit_time_readout(ztr, ridge=1e3):
    n, t, d = ztr.shape
    zc = (ztr - ztr.mean(1, keepdims=True)).reshape(-1, d)
    y = np.tile(np.linspace(-1, 1, t), n)
    w = np.linalg.solve(zc.T @ zc + ridge * np.eye(d), zc.T @ y)
    return w


def readout_spearman(w, z):
    from scipy.stats import rankdata
    s = np.einsum("ntd,d->nt", z - z.mean(1, keepdims=True), w)
    rs = rankdata(s, axis=1)
    rt = rankdata(np.arange(z.shape[1]))
    rs, rt = rs - rs.mean(1, keepdims=True), rt - rt.mean()
    return (rs * rt).sum(1) / np.sqrt((rs ** 2).sum(1) * (rt ** 2).sum())
