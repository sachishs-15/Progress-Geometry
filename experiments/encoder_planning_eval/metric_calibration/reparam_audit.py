"""Which metrics survive information-preserving reparametrisations of the SAME features?

Applies maps that add or remove no information to raw SSv2 features and re-scores the whole suite.
A metric that moves under these maps is measuring parametrisation, not representation quality.

  identity | scale x0.1 | scale x10 | rotation (random orthogonal) | standardise (per-dim z-score)
  pca_whiten (full-rank, cov = I) | layernorm (per-frame, the encoder output norm in train configs)
  layernorm+pca_whiten
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_planning as cpl  # noqa: E402
import calibrate_probes as cpr  # noqa: E402
from metrics_batch import dataset_metrics, per_clip_metrics  # noqa: E402


def layernorm(z):
    m = z.mean(-1, keepdims=True)
    s = z.std(-1, keepdims=True)
    return (z - m) / (s + 1e-5)


def smooth_maps():
    """Information-reducing temporal smoothers applied to the raw frames (free baselines for trajectory gains)."""
    import references as R
    out = {f"gauss{sg:g}": (lambda z, sg=sg: R.gauss_smooth(z, None, sg)) for sg in (0.25, 0.5, 1.0, 2.0)}
    out["ema0.5"] = lambda z: R.ema(z, None, 0.5)
    out["cummean"] = lambda z: R.cummean(z)
    return out


def maps(ztr):
    flat = ztr.reshape(-1, ztr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    w, v = np.linalg.eigh(np.cov(flat, rowvar=False))
    wh = v / np.sqrt(np.clip(w, 1e-12, None))
    d = ztr.shape[-1]
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))
    return {
        "identity": lambda z: z,
        "scale_x0.1": lambda z: 0.1 * z,
        "scale_x10": lambda z: 10 * z,
        "rotation": lambda z: z @ q,
        "standardise": lambda z: (z - mu) / sd,
        "pca_whiten": lambda z: (z - mu) @ wh,
        "layernorm": layernorm,
        "layernorm+pca_whiten": lambda z, f=None: (layernorm(z) - layernorm(ztr).reshape(-1, d).mean(0)) @ _wh(layernorm(ztr)),
    }


def _wh(z):
    flat = z.reshape(-1, z.shape[-1])
    w, v = np.linalg.eigh(np.cov(flat, rowvar=False))
    return v / np.sqrt(np.clip(w, 1e-12, None))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", choices=["linear", "smooth"], default="linear")
    a = ap.parse_args()
    tr_m, va_m, classes, ztr, zva = cpr.load_split(a.encoder)
    out = {}
    for name, f in (maps(ztr) if a.group == "linear" else smooth_maps()).items():
        t, v = f(ztr), f(zva)
        pc, keep = per_clip_metrics(v)
        r = {"trajectory": {k: float(x[keep].mean()) for k, x in pc.items()}, "dataset": dataset_metrics(v)}
        rows = cpl.calibrate_arrays(name, v, None, boots=100, with_ridge=False)
        r["planning"] = {x["predictor"]: {k: x[k] for k in ("r2", "cos_mean", "cos_median") if k in x} for x in rows}
        p = cpr.probe_suite(t, tr_m, v, va_m, classes, light=True)
        p.pop("_pred_linear_seed0"), p.pop("_pred_knn_seed0")
        r["probes"] = p
        out[name] = r
        print("done", name, flush=True)
    json.dump(out, open(a.out, "w"), default=float, indent=1)
