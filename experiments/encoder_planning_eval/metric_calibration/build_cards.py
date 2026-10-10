"""Assemble the per-metric calibration card (trajectory metrics) from the result files.

Columns
  real         DINOv2-S CLS on SSv2 val (the reference encoder), clip-bootstrap 95% CI
  floor        within-clip frame shuffle (no temporal order); iid = i.i.d. Gaussian frames
  decoy        random walk with the clip's own step vectors (smooth but undirected)
  gauss1/cum   free temporal smoothers of the SAME features (Gaussian sigma=1 frame / causal cumulative mean)
  ceiling      straight line through each clip's real endpoints (formal maximum)
  pos          (real - floor) / (ceiling - floor): where the encoder sits in the formal range
  se_clip      unpaired bootstrap SE of one model's mean;  se_pair: paired SE of (model - raw) on the same clips
  sd_seed      sd over 5 training seeds (VICReg sw0.1, CLS family);  sd_epoch: within-run epoch 29/39/49
  mdd          minimum detectable difference, one trained model vs raw, 80% power:  2.8*sqrt(se_pair^2 + sd_seed^2)
  steps        (ceiling - floor) / mdd : how many MDD-sized steps fit in the formal range (resolution)
  icc          per-clip test-retest reliability under frame-sampling jitter (DINOv2)
  win_lo/hi    drift-to-noise ratios r at which the metric reaches 10% / 90% of its range on straight+white-noise
               trajectories (sensitive window); real encoders sit at r ~ 0.002-0.5, see README
  d_whiten     change under free PCA-whitening of the raw features, in units of mdd
  d_gauss1     change under sigma=1 frame smoothing, in fractions of the formal range
  rev_sym      value on the time-reversed clip equals the forward value (arrow-of-time blind)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import noise
from metrics_batch import PER_CLIP_KEYS

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DIRECTION = {"direction_reversal_fraction": -1, "norm_turning_point_fraction": -1, "adjacent_cosine_distance": 0,
             "start_end_displacement": 0, "norm_growth": 0, "norm_std_within_clip": 0, "adjacent_l2_distance": 0,
             "within_clip_mean_pairwise_l2": 0}


def build(enc="dinov2"):
    tc = pd.read_csv(RES / "trajectory_calibration.csv")
    g = lambda ref, m: float(tc[(tc.encoder == enc) & (tc.ref == ref) & (tc.metric == m)].value.iloc[0])
    boot = tc[(tc.encoder == enc) & (tc.ref == "REAL") & tc.metric.str.endswith("__boot")].set_index("metric")
    jr = pd.read_csv(RES / "jitter_reliability.csv")
    jr = jr[jr.encoder == enc].set_index("metric")
    sl = pd.read_csv(RES / "synthetic_latent_curves.csv")
    sl = sl[sl.theta == 0].sort_values("r")
    rp = json.load(open(RES / "reparam_audit_dinov2.json"))
    sm = json.load(open(RES / "smooth_audit_dinov2.json"))
    raw = np.load(RES / "noise_study/RAW_dinov2cls.perclip.npz")
    seeds = [np.load(RES / f"noise_study/vicreg_s{s}.perclip.npz") for s in range(1, 6)]
    ep = {m: [] for m in PER_CLIP_KEYS}
    for s in range(1, 6):
        for m in PER_CLIP_KEYS:
            ep[m].append(np.std([json.load(open(RES / f"noise_study/vicreg_s{s}{suf}.json"))["trajectory"][m]
                                 for suf in ("_epoch29", "_epoch39", "")], ddof=1))
    keep = raw["keep"] & np.all([s["keep"] for s in seeds], 0)
    rows = []
    for m in PER_CLIP_KEYS:
        floor, ceil = g("shuffle_time", m), g("line_endpoints", m)
        real = g("REAL", m)
        se_pair = noise.paired_diff_bootstrap(seeds[0][m], raw[m], B=1000, keep=keep)["se"]
        sd_seed = float(np.std([s[m][s["keep"]].mean() for s in seeds], ddof=1))
        mdd = noise.mdd(np.hypot(se_pair, sd_seed))
        y = sl[m].values
        lo, hi = y.min(), y.max()
        def cross(frac):
            tgt = lo + frac * (hi - lo)
            o = np.argsort(y)
            return float(np.exp(np.interp(tgt, y[o], np.log(sl.r.values)[o])))

        rng = ceil - floor
        rows.append({
            "metric": m, "real": real, "ci_lo": boot.loc[m + "__boot", "ci_lo"], "ci_hi": boot.loc[m + "__boot", "ci_hi"],
            "floor": floor, "iid": g("iid_gauss", m), "decoy_brownian": g("brownian_signflip", m),
            "gauss1": g("gauss1", m), "cummean": g("cummean", m), "ceiling": ceil,
            "pos": (real - floor) / rng if abs(rng) > 1e-9 else np.nan,
            "se_clip": float(boot.loc[m + "__boot", "sd_rep"]), "se_pair": se_pair, "sd_seed": sd_seed,
            "sd_epoch": float(np.mean(ep[m])), "mdd": mdd, "steps": abs(rng) / mdd,
            "icc": float(jr.loc[m, "icc"]), "win_lo": cross(0.1), "win_hi": cross(0.9),
            "d_whiten_mdd": (rp["pca_whiten"]["trajectory"][m] - rp["identity"]["trajectory"][m]) / mdd,
            "d_gauss1_range": (sm["gauss1"]["trajectory"][m] - rp["identity"]["trajectory"][m]) / rng if abs(rng) > 1e-9 else np.nan,
            "rev_sym": abs(g("reverse", m) - real) < 0.01 * max(1e-9, abs(real)) + 1e-3,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv(RES / "metric_card_trajectory_dinov2.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 100)
    print(df.round(4).to_string(index=False))
