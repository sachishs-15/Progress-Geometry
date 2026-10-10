"""Controlled latent trajectories with KNOWN drift-to-noise ratio and curvature.

Generative model per clip (D dims, T frames, tau = t/(T-1)):
    z_t = c + gamma(tau) + sigma * eps_t,       eps_t ~ N(0, I)  i.i.d.
    straight signal   gamma(tau) = a * tau * u                       (u a random unit vector)
    arc signal        gamma(tau) = R*(cos(th*tau) - 1) e1 + R*sin(th*tau) e2,  total turning angle th,
                      arc length fixed to a  (R = a/th)
Drift-to-noise ratio  r = ||v||^2 / (D sigma^2),  v = a u / (T-1) the per-step drift.

Closed forms for the straight signal + iid noise model (high-D limit), checked by simulation in
tests/test_synthetic_latent.py:
    mean_velocity_cosine_raw   (r - 1) / (r + 2)
    trajectory_straightness_raw  sqrt(((T-1)^2 r + 2) / ((T-1)^2 (r + 2)))
    direction_reversal_fraction  P(cos < 0) -> 1 for r<1, 0 for r>1 (sharp in high D)
They invert to an 'equivalent r' for any observed metric value (infer_r_* below); disagreement between
the r inferred from different metrics shows how far real clips are from 'straight line + white noise'.
"""
import numpy as np

from metrics_batch import per_clip_metrics


def simulate(n, t, d, r, theta=0.0, seed=0, a=None):
    """r: drift-to-noise ratio; theta=0 straight, else total turning angle (radians) of an arc."""
    rng = np.random.default_rng(seed)
    sigma = 1.0
    a = a if a is not None else (t - 1) * np.sqrt(r * d) * sigma  # chord for r with sigma=1: ||v|| = a/(T-1)
    tau = np.linspace(0, 1, t)[None, :, None]
    c = 20.0 * rng.standard_normal((n, 1, d)) / np.sqrt(d) * 5  # clip offset, large vs noise (clip identity)
    if theta == 0:
        u = rng.standard_normal((n, 1, d))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        sig = a * tau * u
    else:
        q = np.linalg.qr(rng.standard_normal((n, d, 2)))[0]  # (n, d, 2) orthonormal e1, e2
        rad = a / theta
        th = theta * tau[0, :, 0]
        arc = np.stack([rad * (np.cos(th) - 1), rad * np.sin(th)], -1)  # (t, 2)
        sig = np.einsum("tk,ndk->ntd", arc, q)
    return c + sig + sigma * rng.standard_normal((n, t, d))


def vel_cos_theory(r):
    return (r - 1) / (r + 2)


def straight_theory(r, t=16):
    k = (t - 1) ** 2
    return np.sqrt((k * r + 2) / (k * (r + 2)))


def infer_r_velcos(c):
    return (1 + 2 * c) / (1 - c)


def infer_r_straight(s, t=16):
    k = (t - 1) ** 2
    return 2 * (s ** 2 * k - 1) / (k * (1 - s ** 2))


def curves(rs, n=300, t=16, d=384, theta=0.0, seed=0):
    out = []
    for r in rs:
        pc, keep = per_clip_metrics(simulate(n, t, d, r, theta, seed))
        out.append({"r": r, **{k: float(v[keep].mean()) for k, v in pc.items()}})
    return out
