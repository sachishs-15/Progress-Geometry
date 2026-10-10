"""Is system A really better than baseline B?  One call, every guard.

Both systems must be scored on the SAME val clips (same order).  For each trajectory metric the table gives

  delta        mean(A) - mean(B) with a PAIRED clip-bootstrap 95% CI (clip-sampling noise)
  range        floor..ceiling computed on B's own latents: floor = within-clip frame shuffle (no order),
               ceiling = straight line through each clip's real endpoints (formal maximum). Two denoised
               references are listed for scale (cubic fit / Gaussian sigma=1 smoothing of B): what B scores
               if its per-frame noise were removed; they are NOT ceilings (other smoothers can exceed them).
  frac         delta / (ceiling - floor): the effect as a fraction of the formal range (sign = improvement)
  smooth_eq    sigma (frames) of Gaussian temporal smoothing of B that reproduces A's value, i.e. 'A is as good as
               B averaged over +-sigma frames'. Small sigma = the gain is the size of a trivial filter.
  whiten_eq    where A falls between B (0) and PCA-whitened B (1). ~1 means the change is what a free linear
               whitening of B already does.
  verdict      REAL / WITHIN-NOISE / SMALL / WORSE / ... see classify()

Plus collapse and planning guards (effective rank, norm, interpolation R^2 and its static-midpoint part).
Training-seed noise is NOT visible from one pair of models: pass --seed-sd-json (a metric->sd map measured from replicate
runs, see results/noise_study) to add it in quadrature; otherwise the CI only covers clip sampling.

    python compare.py --a ckpt:/path/last.ckpt --b raw:dinov2 --out results/compare_x.csv
    python compare.py --a npy:a.npy --b npy:b.npy
Source spec:  raw:<encoder>  (SSv2 val features of a registry encoder)
              ckpt:<path>    (CLS-family ProceduralWorldModel on dinov2 CLS features; config.yaml beside it)
              npy:<file>     ((N,T,D) array saved with np.save)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import noise  # noqa: E402
import references as R  # noqa: E402
from metrics_batch import dataset_metrics, per_clip_metrics  # noqa: E402

HIGHER_BETTER = {"time_progress_spearman": 1, "pc1_time_spearman": 1, "monotonic_progress_fraction": 1,
                 "progress_efficiency": 1, "adjacent_temporal_nn_fraction": 1, "distance_to_goal_spearman": 1,
                 "trajectory_straightness": 1, "trajectory_straightness_raw": 1, "mean_velocity_cosine": 1,
                 "mean_velocity_cosine_raw": 1, "mean_chord_alignment_raw": 1, "direction_reversal_fraction": -1}
SIGMAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def load_source(spec):
    kind, _, arg = spec.partition(":")
    if kind == "npy":
        return np.load(arg).astype(np.float64)
    if kind == "raw":
        f = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_validation.pt".format(arg)
        return torch.load(f, map_location="cpu")["features"].numpy().astype(np.float64)
    if kind == "ckpt":
        from eval_common.loading import encode_world_model, load_world_model
        model, _, _ = load_world_model(arg, str(Path(arg).parent / "config.yaml"), torch.device("cpu"))
        f = torch.load("/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt",
                       map_location="cpu")["features"]
        return encode_world_model(model, f.float(), torch.device("cpu"), 256).astype(np.float64)
    raise SystemExit(f"bad source {spec}")


def whiten(z):
    return R.pca_proj(z, None, z.shape[-1], whiten=True)


def smooth_equivalent(zb, metric, target, rng):
    """sigma at which Gaussian smoothing of B matches `target`; None if outside the swept range."""
    vals = []
    for s in SIGMAS:
        pc, keep = per_clip_metrics(R.gauss_smooth(zb, None, s))
        vals.append(pc[metric][keep].mean())
    v0 = per_clip_metrics(zb)
    base = v0[0][metric][v0[1]].mean()
    xs, ys = np.array((0.0,) + SIGMAS), np.array([base] + vals)
    d = np.sign(target - base)
    if d == 0:
        return 0.0
    if not (ys.min() <= target <= ys.max()):
        return None if (target - base) * (ys[-1] - base) > 0 else None
    o = np.argsort(ys)
    return float(np.interp(target, ys[o], xs[o]))


def classify(delta, lo, hi, direction, frac, min_frac, whiten_pos):
    imp = direction * delta
    if lo <= 0 <= hi:
        return "WITHIN-NOISE"
    if imp < 0:
        return "WORSE"
    if abs(frac) < min_frac:
        return "SIG-BUT-SMALL"
    if whiten_pos is not None and 0.8 <= whiten_pos <= 1.2:
        return "EXPLAINED-BY-WHITENING"
    return "REAL"


def compare(za, zb, seed_sd=None, min_frac=0.05, boots=2000, shuffles=5, seed=0):
    rng = np.random.default_rng(seed)
    pa, ka = per_clip_metrics(za)
    pb, kb = per_clip_metrics(zb)
    keep = ka & kb
    fl = [per_clip_metrics(R.shuffle_time(zb, rng)) for _ in range(shuffles)]
    ce = per_clip_metrics(R.line_endpoints(zb))
    dn_cubic = per_clip_metrics(R.poly_smooth(zb, deg=3))
    dn_gauss = per_clip_metrics(R.gauss_smooth(zb, None, 1.0))
    zw = whiten(zb)
    pw, kw = per_clip_metrics(zw)
    rows = []
    for m, direction in HIGHER_BETTER.items():
        d = noise.paired_diff_bootstrap(pa[m], pb[m], B=boots, keep=keep, seed=seed)
        se = d["se"]
        lo, hi = d["lo"], d["hi"]
        if seed_sd and m in seed_sd:
            se = float(np.hypot(se, seed_sd[m]))
            lo, hi = d["diff"] - 1.96 * se, d["diff"] + 1.96 * se
        floor = float(np.mean([f[0][m][f[1]].mean() for f in fl]))
        ceil = float(ce[0][m][ce[1]].mean())
        frac = d["diff"] / (ceil - floor) * direction * np.sign(ceil - floor) if ceil != floor else np.nan
        bw = float(pb[m][kb].mean())
        ww = float(pw[m][kw].mean())
        wpos = (float(pa[m][ka].mean()) - bw) / (ww - bw) if abs(ww - bw) > max(0.005, 0.25 * abs(d["diff"])) else None
        rows.append({"metric": m, "B": bw, "A": float(pa[m][ka].mean()), "delta": d["diff"], "lo": lo, "hi": hi,
                     "mdd80": noise.mdd(se), "floor": floor, "ceiling": ceil, "frac_of_range": frac,
                     "denoised_cubic": float(dn_cubic[0][m][dn_cubic[1]].mean()),
                     "denoised_gauss1": float(dn_gauss[0][m][dn_gauss[1]].mean()),
                     "smooth_eq_sigma": smooth_equivalent(zb, m, float(pa[m][ka].mean()), rng),
                     "whiten_eq": wpos,
                     "verdict": classify(d["diff"], lo, hi, direction, frac, min_frac, wpos)})
    out = pd.DataFrame(rows)
    ga, gb = dataset_metrics(za), dataset_metrics(zb)
    guards = {"eff_rank_frac A/B": (ga["effective_rank_fraction"], gb["effective_rank_fraction"]),
              "mean_norm A/B": (ga["mean_norm"], gb["mean_norm"])}
    return out, guards


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seed-sd-json")
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--out")
    a = ap.parse_args()
    za, zb = load_source(a.a), load_source(a.b)
    sd = json.load(open(a.seed_sd_json)) if a.seed_sd_json else None
    df, guards = compare(za, zb, sd, a.min_frac)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(guards)
    if a.out:
        df.to_csv(a.out, index=False)
