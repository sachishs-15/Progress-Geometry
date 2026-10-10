"""Split each clip's within-clip variation into smooth temporal signal and per-frame noise.

Model per clip: z_t = s(t) + e_t, s smooth in t, e_t zero-mean and non-persistent (uncorrelated
across frames). "Noise" here means anything that changes frame-to-frame without being part of
the smooth trajectory: motion blur, compression, hand pose flicker, camera shake, encoder
jitter. This is what temporal-straightness metrics are most sensitive to.

Two estimators (they bracket each other):
  difference-based   sigma2_k = mean ||Delta^k z||^2 / C(2k,k).  For i.i.d. noise this is exact;
                     smooth signal inflates it, so it is an UPPER bound on noise, converging
                     to the truth as k grows until the k-th difference of the signal vanishes.
  polynomial fit     residual variance of a degree-d fit in t, with d chosen by leave-one-out
                     CV (the LOO error itself is noise + fit variance + bias, so it is only
                     used to pick d). Unbiased for the noise variance when the degree-d
                     polynomial captures the signal; biased UP when it does not.
Both are validated against known-noise synthetic data in tests/test_signal_noise.py.
"""
from math import comb

import numpy as np


def diff_noise_var(z, order=2):
    """Per-clip estimate of per-dimension-summed noise variance E||e_t||^2. z: (N,T,D)."""
    d = np.diff(z, n=order, axis=1)
    return (d ** 2).sum(-1).mean(1) / comb(2 * order, order)


def total_within_var(z):
    """Per-clip mean ||z_t - clip mean||^2 (biased by (T-1)/T w.r.t. unbiased variance)."""
    return ((z - z.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)


def _hat(t, deg):
    x = np.linspace(-1, 1, t)
    b = np.stack([x ** k for k in range(deg + 1)], axis=1)
    return b @ np.linalg.pinv(b)


def poly_loo_error(z, deg):
    """Per-clip mean over frames of ||z_t - z_hat_{-t}||^2 for a degree-`deg` polynomial fit
    (closed-form leave-one-out via the hat matrix). Returns (N,)."""
    t = z.shape[1]
    h = _hat(t, deg)
    res = z - np.einsum("st,ntd->nsd", h, z)
    lev = np.diag(h)
    loo = res / (1 - lev)[None, :, None]
    return (loo ** 2).sum(-1).mean(1)


def poly_resid_var(z, deg):
    """Per-clip unbiased residual variance sum_t ||z_t - fit_t||^2 / (T - deg - 1), summed over
    dimensions: an estimate of E||e_t||^2 when the degree-`deg` polynomial captures the signal."""
    t = z.shape[1]
    h = _hat(t, deg)
    res = z - np.einsum("st,ntd->nsd", h, z)
    return (res ** 2).sum((1, 2)) / (t - deg - 1)


def best_degree(z, degs=range(0, 8)):
    errs = {d: float(poly_loo_error(z, d).mean()) for d in degs}
    return min(errs, key=errs.get), errs


def summary(z):
    """Dict of scalar diagnostics for an encoder's (N,T,D) latents."""
    tot = total_within_var(z)
    out = {"within_var": float(tot.mean())}
    for k in (1, 2, 3, 4):
        nv = diff_noise_var(z, k)
        out[f"noise_frac_diff{k}"] = float(np.clip(nv / tot, 0, 1).mean())
    deg, errs = best_degree(z)
    out["best_poly_degree"] = deg
    out["noise_frac_loo_best"] = float(min(errs.values()) / tot.mean())
    for d in (1, 2, 3):
        out[f"noise_frac_poly{d}"] = float((poly_resid_var(z, d) / tot).mean())
    return out
