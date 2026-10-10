"""Reference trajectories that anchor a metric's scale, built from the SAME latents under test.

Every function maps real latents z (N, T, D) -> a transformed (N, T, D) array. Scoring the
transformed array with the unchanged metric gives a reference value:

FLOORS (what a trajectory with no usable temporal signal scores)
  shuffle_time      each clip's frames in random order. Keeps every clip's frame set (content,
                    per-frame noise, clip identity) and removes order. The permutation null for
                    every order-sensitive metric.
  shuffle_pool      each clip is T frames drawn from T different clips. Removes order AND clip
                    identity (floor for within/across, cross-clip and probe-style metrics).
  iid_gauss         i.i.d. N(mu, Sigma) frames with the pooled frame covariance. Parametric noise
                    with the encoder's own marginal geometry; no clip structure.
  static_gauss      per-clip mean + i.i.d. Gaussian residual (pooled within-clip covariance):
                    "clip identity + noise, zero dynamics".
DECOYS (smooth but undirected -- what the metric should NOT reward)
  brownian_signflip real step vectors with independent random signs, integrated from z_1.
                    Same step magnitudes/axes as the data, no step-to-step coherence.
MIRROR
  reverse           time-reversed clip (signed order metrics should flip sign).
CEILINGS (what is reachable if noise and/or curvature were removed)
  line_endpoints    exact straight line between the clip's real first and last frame. Formal
                    maximum of every straightness/ordering metric (before L2 normalisation).
  line_fit          least-squares straight line through the clip (noise-free linear trend).
  poly_smooth       degree-k polynomial fit in time (default cubic): the clip's real smooth
                    dynamics, including genuine curvature, with per-frame noise removed.
"""
import numpy as np


def _perm_idx(rng, n, t):
    return np.argsort(rng.random((n, t)), axis=1)


def shuffle_time(z, rng):
    idx = _perm_idx(rng, *z.shape[:2])
    return np.take_along_axis(z, idx[:, :, None], axis=1)


def shuffle_pool(z, rng):
    n, t, _ = z.shape
    return z[rng.integers(0, n, (n, t)), rng.integers(0, t, (n, t))]


def _gauss_sampler(cov, rng):
    w, v = np.linalg.eigh(cov)
    a = v * np.sqrt(np.clip(w, 0, None))
    return lambda shape: rng.standard_normal(shape + (cov.shape[0],)) @ a.T


def iid_gauss(z, rng):
    n, t, d = z.shape
    flat = z.reshape(-1, d)
    mu, cov = flat.mean(0), np.cov(flat, rowvar=False)
    return mu + _gauss_sampler(cov, rng)((n, t))


def static_gauss(z, rng):
    n, t, d = z.shape
    mean_c = z.mean(1, keepdims=True)
    res = (z - mean_c).reshape(-1, d)
    cov = res.T @ res / (len(res) - n) if len(res) > n else np.cov(res, rowvar=False)
    return mean_c + _gauss_sampler(cov, rng)((n, t))


def brownian_signflip(z, rng):
    steps = z[:, 1:] - z[:, :-1]
    signs = rng.choice([-1.0, 1.0], size=steps.shape[:2])[:, :, None]
    return np.concatenate([z[:, :1], z[:, :1] + np.cumsum(steps * signs, axis=1)], axis=1)


def reverse(z, rng=None):
    return z[:, ::-1].copy()


def line_endpoints(z, rng=None):
    t = z.shape[1]
    a = np.linspace(0, 1, t)[None, :, None]
    return z[:, :1] + a * (z[:, -1:] - z[:, :1])


def _poly_basis(t, deg):
    x = np.linspace(-1, 1, t)
    return np.stack([x ** k for k in range(deg + 1)], axis=1)  # (T, deg+1)


def poly_smooth(z, rng=None, deg=3):
    b = _poly_basis(z.shape[1], deg)
    proj = b @ np.linalg.pinv(b)  # (T, T) projection onto polynomials in t
    return np.einsum("st,ntd->nsd", proj, z)


def line_fit(z, rng=None):
    return poly_smooth(z, rng, deg=1)


FLOORS = {"shuffle_time": shuffle_time, "shuffle_pool": shuffle_pool,
          "iid_gauss": iid_gauss, "static_gauss": static_gauss}
DECOYS = {"brownian_signflip": brownian_signflip}
MIRRORS = {"reverse": reverse}
CEILINGS = {"line_endpoints": line_endpoints, "line_fit": line_fit, "poly_smooth": poly_smooth}
ALL = {**FLOORS, **DECOYS, **MIRRORS, **CEILINGS}


# ---- smoothing decoys: "cheap" ways to move the metrics without learning anything ------------
def cummean(z, rng=None):
    """Causal cumulative mean z'_t = mean(z_1..z_t). Content-free in the sense of adding no
    information, yet it scores ~1.0 on the progress metrics (see configs/cls_residual_mlp_*)."""
    return np.cumsum(z, axis=1) / np.arange(1, z.shape[1] + 1)[None, :, None]


def ema(z, rng=None, alpha=0.5):
    """Causal exponential moving average, z'_t = alpha*z_t + (1-alpha)*z'_{t-1}."""
    out = np.empty_like(z)
    out[:, 0] = z[:, 0]
    for t in range(1, z.shape[1]):
        out[:, t] = alpha * z[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def gauss_smooth(z, rng=None, sigma=1.0):
    """Non-causal Gaussian temporal smoothing along the 16 frames (reflect padding)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(z, sigma=sigma, axis=1, mode="nearest")


SMOOTHERS = {"cummean": cummean, "ema0.5": lambda z, rng=None: ema(z, rng, 0.5),
             "gauss1": lambda z, rng=None: gauss_smooth(z, rng, 1.0),
             "gauss2": lambda z, rng=None: gauss_smooth(z, rng, 2.0)}


# ---- information-free reparametrisations of the same latents ---------------------------------
def rand_proj(z, rng, d=128):
    """Random Gaussian projection to d dims (scaled to preserve norms in expectation)."""
    a = rng.standard_normal((z.shape[-1], d)) / np.sqrt(d)
    return z @ a


def pca_proj(z, rng=None, d=128, whiten=False):
    """Project onto the top-d principal axes of the pooled frames (fit on the same data)."""
    flat = z.reshape(-1, z.shape[-1])
    mu = flat.mean(0)
    w, v = np.linalg.eigh(np.cov(flat, rowvar=False))
    idx = np.argsort(w)[::-1][:d]
    out = (z - mu) @ v[:, idx]
    return out / np.sqrt(w[idx]) if whiten else out


REPARAMS = {"rand_proj128": rand_proj, "pca128": pca_proj,
            "pca128_whiten": lambda z, rng=None: pca_proj(z, rng, 128, True)}
