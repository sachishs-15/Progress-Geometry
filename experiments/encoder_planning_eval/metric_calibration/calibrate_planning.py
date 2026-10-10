"""Calibrate the interpolation-planning metrics (cosine, relative L2, R^2).

Question per encoder: given (z_1, z_T), how well are the 14 intermediate latents recovered, and
how much of that is just "z_1 and z_T identify the clip" rather than "the path between them is
recoverable"?  Every predictor below is scored with the unchanged metrics on the same examples.

  floor      mean            train mean of intermediates (R^2 ~ 0 by construction)
             copy_z1/zT      static guesses from one endpoint
             mid_const       (z_1+z_T)/2 at every t: best temporally-STATIC guess from the endpoints
  metric     interp          z_1 + alpha_t (z_T - z_1)        <- what the paper reports
  ceilings   ridge_resid     interp + ridge(z_1,z_T) residual, fit on the TRAIN split. Convex, shrinks
                             toward interp, so unlike the MLP/transformer predictors it cannot do
                             much worse than interp. The achievable learned reference.
             neighbors       (z_{t-1}+z_{t+1})/2   smoothness oracle (sees the adjacent frames)
             denoised        cubic fit of the full clip; error = frame noise only
             noise_ceiling   R^2 if only the per-frame noise were irreducible (difference estimator)
Skill scores use mid_const as the reference:  skill = 1 - MSE(method) / MSE(mid_const).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise  # noqa: E402
import references as R  # noqa: E402
import signal_noise as sn  # noqa: E402

FEATS = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_{}.pt"


def load(enc, split):
    d = torch.load(FEATS.format(enc, split), map_location="cpu")
    return d["features"].numpy().astype(np.float64), d["video_paths"]


def interp(z1, zT, t):
    a = np.linspace(0, 1, t)[None, :, None]
    return z1[:, None] + a * (zT - z1)[:, None]


def ridge_resid_fit(ztr, alphas=(1e2, 1e3, 1e4, 1e5, 1e6), seed=0):
    """Ridge from centered [z1, zT] to the residual of the interpolation at t=1..T-2.
    alpha picked on a 20% clip hold-out of the TRAIN split, then refit on all train clips."""
    n, t, d = ztr.shape
    x = np.concatenate([ztr[:, 0], ztr[:, -1]], axis=1)
    mu = x.mean(0)
    xc = x - mu
    y = (ztr - interp(ztr[:, 0], ztr[:, -1], t))[:, 1:-1].reshape(n, -1)
    ymu = y.mean(0)
    yc = y - ymu
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    va, tr = perm[: n // 5], perm[n // 5:]

    def solve(idx, alpha):
        g = xc[idx].T @ xc[idx]
        w, v = np.linalg.eigh(g)
        b = v.T @ (xc[idx].T @ yc[idx])
        return v @ (b / (w + alpha)[:, None])

    g = xc[tr].T @ xc[tr]
    w, v = np.linalg.eigh(g)
    b = v.T @ (xc[tr].T @ yc[tr])
    best, best_err = None, np.inf
    for a in alphas:
        wmat = v @ (b / (w + a)[:, None])
        err = ((xc[va] @ wmat - yc[va]) ** 2).mean()
        if err < best_err:
            best, best_err = a, err
    return {"W": solve(np.arange(n), best), "mu": mu, "ymu": ymu, "alpha": best}


def ridge_resid_predict(model, z):
    n, t, d = z.shape
    x = np.concatenate([z[:, 0], z[:, -1]], axis=1) - model["mu"]
    r = (x @ model["W"] + model["ymu"]).reshape(n, t - 2, d)
    return interp(z[:, 0], z[:, -1], t)[:, 1:-1] + r


def per_example_stats(pred, true, mu):
    """Per-clip sums over the 14 intermediates so metrics can be bootstrapped over clips."""
    cos = (pred * true).sum(-1) / (np.linalg.norm(pred, axis=-1) * np.linalg.norm(true, axis=-1) + 1e-8)
    rel = np.linalg.norm(pred - true, axis=-1) / (np.linalg.norm(true, axis=-1) + 1e-8)
    return {"cos": cos, "rel": rel, "ss_res": ((pred - true) ** 2).sum(-1),
            "ss_tot": ((true - mu) ** 2).sum(-1)}


def summarize(st, boots=1000, seed=0):
    """Returns dict of scalar metrics and bootstrap CIs on R^2 and cosine mean."""
    n = st["cos"].shape[0]
    w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=boots).astype(np.float64)
    res_c, tot_c = st["ss_res"].sum(1), st["ss_tot"].sum(1)
    r2 = 1 - res_c.sum() / tot_c.sum()
    r2b = 1 - (w @ res_c) / (w @ tot_c)
    cosc = st["cos"].mean(1)
    cb = (w @ cosc) / n
    return {"r2": float(r2), "r2_lo": float(np.quantile(r2b, .025)), "r2_hi": float(np.quantile(r2b, .975)),
            "cos_mean": float(cosc.mean()), "cos_median": float(np.median(st["cos"])),
            "cos_lo": float(np.quantile(cb, .025)), "cos_hi": float(np.quantile(cb, .975)),
            "rel_l2_median": float(np.median(st["rel"])), "mse": float(res_c.sum() / (n * st["ss_res"].shape[1])),
            "res_c": res_c, "tot_c": tot_c}


def calibrate(enc, boots=1000, with_ridge=True):
    zva, paths = load(enc, "validation")
    try:
        ztr, _ = load(enc, "train")
    except FileNotFoundError:
        ztr = None
    return calibrate_arrays(enc, zva, ztr, boots, with_ridge)


def calibrate_arrays(enc, zva, ztr=None, boots=1000, with_ridge=True):
    """Same as calibrate() for latents already in memory: zva (N,T,D) val, ztr (M,T,D) train or None."""
    n, t, d = zva.shape
    z1, zT = zva[:, 0], zva[:, -1]
    true = zva[:, 1:-1]
    mu_global = true.reshape(-1, d).mean(0)
    mu_train = ztr[:, 1:-1].reshape(-1, d).mean(0) if ztr is not None else mu_global

    preds = {
        "mean": np.broadcast_to(mu_train, true.shape),
        "copy_z1": np.broadcast_to(z1[:, None], true.shape),
        "copy_zT": np.broadcast_to(zT[:, None], true.shape),
        "mid_const": np.broadcast_to(((z1 + zT) / 2)[:, None], true.shape),
        "interp": interp(z1, zT, t)[:, 1:-1],
        "neighbors": (zva[:, :-2] + zva[:, 2:]) / 2,
        "denoised_cubic": R.poly_smooth(zva, deg=3)[:, 1:-1],
    }
    if with_ridge and ztr is not None:
        m = ridge_resid_fit(ztr)
        preds["ridge_resid"] = ridge_resid_predict(m, zva)
    rows, sums = [], {}
    for name, p in preds.items():
        s = summarize(per_example_stats(p, true, mu_global), boots)
        sums[name] = s
        rows.append({"encoder": enc, "predictor": name, **{k: v for k, v in s.items() if not k.endswith("_c")}})
    mse_mid = sums["mid_const"]["mse"]
    for r in rows:
        r["skill_vs_mid_const"] = 1 - r["mse"] / mse_mid
        r["skill_vs_interp"] = 1 - r["mse"] / sums["interp"]["mse"]
    # paired bootstrap: interp vs mid_const and ridge vs interp, in R^2
    n_ = len(zva)
    w = np.random.default_rng(1).multinomial(n_, np.full(n_, 1 / n_), size=boots).astype(np.float64)

    def r2b(name):
        s = sums[name]
        return 1 - (w @ s["res_c"]) / (w @ s["tot_c"])

    pair = {"interp_minus_mid": r2b("interp") - r2b("mid_const")}
    if "ridge_resid" in sums:
        pair["ridge_minus_interp"] = r2b("ridge_resid") - r2b("interp")
    for k, v in pair.items():
        base = {"interp_minus_mid": sums["interp"]["r2"] - sums["mid_const"]["r2"],
                "ridge_minus_interp": (sums["ridge_resid"]["r2"] - sums["interp"]["r2"]) if "ridge_resid" in sums else 0}[k]
        rows.append({"encoder": enc, "predictor": "PAIR_" + k, "r2": float(base), "r2_lo": float(np.quantile(v, .025)),
                     "r2_hi": float(np.quantile(v, .975))})
    # noise ceiling in R^2: only per-frame noise (diff-4 estimator) is unpredictable
    nv = sn.diff_noise_var(zva, 4)  # per clip, summed over dims
    tot_c = sums["interp"]["tot_c"]
    rows.append({"encoder": enc, "predictor": "noise_ceiling", "r2": float(1 - (nv * (t - 2)).sum() / tot_c.sum())})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for e in a.encoders:
        rows += calibrate(e, a.boots)
        print("done", e, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
