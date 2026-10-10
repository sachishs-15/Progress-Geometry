"""Calibrate the trajectory / progress metric suite on real latents.

For each encoder, scores the REAL (N,T,D) latents and every reference trajectory in
references.py with the unchanged metric code, giving per metric:
    floor (shuffle_time null), decoys (random walk, smoothers), ceilings (line / denoised),
    and a clip-bootstrap CI on the real value.

    python calibrate_trajectory.py --encoders dinov2 vjepa2.1 --out results/trajectory_calibration.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise  # noqa: E402
import references as R  # noqa: E402
from metrics_batch import PER_CLIP_KEYS, dataset_metrics, per_clip_metrics  # noqa: E402

FEATS = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_validation.pt"
STOCH = {"shuffle_time", "shuffle_pool", "iid_gauss", "static_gauss", "brownian_signflip",
         "rand_proj128"}


def load(enc):
    return torch.load(FEATS.format(enc), map_location="cpu")["features"].numpy().astype(np.float64)


def refs():
    out = {"REAL": lambda z, rng: z}
    out.update(R.ALL)
    out["poly_smooth_deg2"] = lambda z, rng: R.poly_smooth(z, rng, 2)
    out["poly_smooth_deg3"] = lambda z, rng: R.poly_smooth(z, rng, 3)
    out["poly_smooth_deg4"] = lambda z, rng: R.poly_smooth(z, rng, 4)
    out.pop("poly_smooth")
    out.update(R.SMOOTHERS)
    out.update(R.REPARAMS)
    return out


def score(zz):
    pc, keep = per_clip_metrics(zz)
    ds = dataset_metrics(zz)
    m = {k: float(v[keep].mean()) for k, v in pc.items()}
    md = {k + "__median": float(np.median(v[keep])) for k, v in pc.items()}
    m.update(md)
    m.update(ds)
    d = ds["mean_pairwise_l2_across_clips"]
    m["within_over_across"] = m["within_clip_mean_pairwise_l2"] / d if d else np.nan
    return m, pc, keep


def run_encoder(args):
    enc, reps, boots, seed = args
    z = load(enc)
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in refs().items():
        n_rep = reps if name in STOCH else 1
        ms = [score(fn(z, rng))[0] for _ in range(n_rep)]
        for k in ms[0]:
            v = np.array([m[k] for m in ms])
            rows.append({"encoder": enc, "ref": name, "metric": k, "value": v.mean(),
                         "sd_rep": v.std(ddof=1) if n_rep > 1 else 0.0, "ci_lo": np.nan, "ci_hi": np.nan})
    _, pc, keep = score(z)
    for k in PER_CLIP_KEYS:
        s = noise.bootstrap_mean(pc[k], B=boots, keep=keep, seed=seed)
        lo, hi = noise.ci(s)
        rows.append({"encoder": enc, "ref": "REAL", "metric": k + "__boot", "value": float(pc[k][keep].mean()),
                     "sd_rep": float(s.std(ddof=1)), "ci_lo": lo, "ci_hi": hi})
    Path("results/perclip").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"results/perclip/{enc}.npz", keep=keep, **pc)
    print(f"done {enc}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(e, a.reps, a.boots, a.seed) for e in a.encoders]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_encoder, jobs) for r in rs]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)
