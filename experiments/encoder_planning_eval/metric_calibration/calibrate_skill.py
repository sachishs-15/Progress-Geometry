"""Calibrate one-step prediction skill, skill = 1 - model_error / copy_error.

The README calls 1.0 'perfect'. It is not reachable: the target frame carries per-frame noise that no
predictor can foresee, so the best possible skill is  1 - sigma^2 / copy_error  (noise_ceiling below,
sigma^2 from the difference estimator in signal_noise.py). Also, 'copy' is a weak baseline when
frames are noisy, since averaging past frames already denoises. Ladder, all in the SAME latent space
and scored with the SAME squared error:

  copy            z_{k+1} = z_k                                   (skill 0 by definition)
  cummean         mean(z_1..z_k)       free denoiser, no learning
  ema0.5          exponential average
  linear_extrap   2 z_k - z_{k-1}
  ridge_ar{1,3}   ridge regression on the previous 1 / 3 frames, fit on the train latents of the
                  same encoder: the learnable LINEAR reference
  noise_ceiling   1 - sigma^2/copy_error
  model           the trained predictor (checkpoints only)
Positions: predicting z_{k+1} for k = 3..T-1 (all methods defined there).
"""
import numpy as np

import signal_noise as sn


def _err(pred, target):
    return ((pred - target) ** 2).mean(-1)  # (N, K) mean over dims


def fit_ridge_ar(ztr, order, ridge=1e-2):
    n, t, d = ztr.shape
    xs, ys = [], []
    for k in range(order - 1, t - 1):  # context frames k-order+1..k -> target k+1
        xs.append(np.concatenate([ztr[:, k - j] for j in range(order)], 1))
        ys.append(ztr[:, k + 1])
    x, y = np.concatenate(xs), np.concatenate(ys)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    g = xc.T @ xc
    w = np.linalg.solve(g + ridge * np.trace(g) / len(g) * np.eye(len(g)), xc.T @ yc)
    return w, xm, ym


def ridge_ar_predict(model, z, order):
    w, xm, ym = model
    n, t, d = z.shape
    preds = np.full((n, t - 1, d), np.nan)
    for k in range(order - 1, t - 1):
        x = np.concatenate([z[:, k - j] for j in range(order)], 1)
        preds[:, k] = (x - xm) @ w + ym
    return preds


def ladder(zva, ztr, model_pred=None, k0=3):
    """Per-clip, per-position squared errors for each method on positions k0..T-1.
    zva/ztr (N,T,D); model_pred (N,T-1,D) teacher-forced predictions of z[:,1:] (index k predicts k+1, 0-based)."""
    n, t, d = zva.shape
    tgt = zva[:, 1:]  # target for context ending at frame k (0-based k=0..T-2)
    sl = slice(k0 - 1, None)  # 0-based k >= k0-1  <=> 1-based k >= k0
    cm = np.cumsum(zva, 1) / np.arange(1, t + 1)[None, :, None]
    e = np.empty_like(zva)
    e[:, 0] = zva[:, 0]
    for k in range(1, t):
        e[:, k] = 0.5 * zva[:, k] + 0.5 * e[:, k - 1]
    preds = {
        "copy": zva[:, :-1],
        "cummean": cm[:, :-1],
        "ema0.5": e[:, :-1],
        "linear_extrap": np.concatenate([zva[:, :1], 2 * zva[:, 1:-1] - zva[:, :-2]], 1),
    }
    for o in (1, 3):
        preds[f"ridge_ar{o}"] = ridge_ar_predict(fit_ridge_ar(ztr, o), zva, o)
    if model_pred is not None:
        preds["model"] = model_pred
    errs = {k: _err(v, tgt)[:, sl] for k, v in preds.items()}
    nv = sn.diff_noise_var(zva, 4) / d  # per-dim noise variance, per clip
    return errs, nv


def summarize(errs, nv, boots=1000, seed=0):
    copy = errs["copy"]
    n, k = copy.shape
    w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=boots).astype(np.float64)
    out = {}
    cs = copy.sum(1)
    for name, e in errs.items():
        es = e.sum(1)
        skill = 1 - es.sum() / cs.sum()
        sb = 1 - (w @ es) / (w @ cs)
        out[name] = {"skill": float(skill), "lo": float(np.quantile(sb, .025)), "hi": float(np.quantile(sb, .975)),
                     "mse": float(e.mean())}
    out["noise_ceiling"] = {"skill": float(1 - (nv * k).sum() / cs.sum())}
    if "model" in errs:
        for ref in ("cummean", "ridge_ar3"):
            d = (1 - (w @ errs["model"].sum(1)) / (w @ cs)) - (1 - (w @ errs[ref].sum(1)) / (w @ cs))
            out[f"model_minus_{ref}"] = {"skill": float(out["model"]["skill"] - out[ref]["skill"]),
                                         "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
    return out
