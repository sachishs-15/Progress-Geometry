"""Test-retest of clip features under frame-sampling jitter.

The stored SSv2 features use one fixed sampling of each video: 16 frames at
linspace(0, len-1, 16). Any other equally reasonable 16-frame sample of the same video is a
different 'measurement' of the same clip. This script re-extracts SSv2-val clips with

    base   the repo's linspace sampling (must reproduce the stored features -- checked)
    rep k  TSN-style: one uniformly random frame inside each of 16 equal segments

so per-clip metric reliability (and the dataset-level jitter of every metric) can be measured.

    python extract_jitter.py --encoder dinov2 --pooling cls --n-clips 600 --reps 4 --out results/jitter/dinov2.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "frozen_encoders"))
import models  # noqa: E402,F401  (registers plugins)
from common.base import build  # noqa: E402

MAN = ROOT / "data" / "chirality_in_action" / "SSv2" / "ssv2_validation.csv"
STORED = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_validation.pt"


def base_indices(n, t=16):
    return np.linspace(0, n - 1, t).astype(int)


def jitter_indices(n, rng, t=16):
    edges = np.linspace(0, n, t + 1)
    lo = np.floor(edges[:-1]).astype(int)
    hi = np.maximum(lo + 1, np.floor(edges[1:]).astype(int))
    return np.minimum(rng.integers(lo, hi), n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--pooling", default="cls")
    ap.add_argument("--n-clips", type=int, default=600)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = pd.read_csv(MAN)["video_path"].tolist()
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(len(paths), min(a.n_clips, len(paths)), replace=False))
    stored = torch.load(STORED.format(a.encoder), map_location="cpu", mmap=True)
    pos = {p: i for i, p in enumerate(stored["video_paths"])}

    enc = build(a.encoder, device="cuda", pooling=a.pooling)
    feats = {k: [] for k in ["base"] + [f"rep{r}" for r in range(a.reps)]}
    idx_log, kept = [], []
    with enc:
        for i in tqdm(sel):
            vp = paths[i]
            if vp not in pos:
                continue
            try:
                vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
                n = len(vr)
                sets = {"base": base_indices(n)}
                for r in range(a.reps):
                    sets[f"rep{r}"] = jitter_indices(n, rng)
                union = np.unique(np.concatenate(list(sets.values())))
                frames = dict(zip(union.tolist(), vr.get_batch(union).asnumpy()))
                out = {k: torch.from_numpy(enc.encode([Image.fromarray(frames[j]) for j in ix]).embeddings)
                       for k, ix in sets.items()}
            except Exception as e:  # skip undecodable videos, recorded by absence
                print("skip", vp, e)
                continue
            for k, v in out.items():
                feats[k].append(v)
            idx_log.append({k: v for k, v in sets.items()})
            kept.append(vp)

    res = {k: torch.stack(v) for k, v in feats.items()}
    ref = torch.stack([stored["features"][pos[p]] for p in kept]).float()
    d = (res["base"].float() - ref).abs()
    cos = torch.nn.functional.cosine_similarity(res["base"].float(), ref, dim=-1)
    print(f"base vs stored: max|diff|={d.max():.3e}  mean cos={cos.mean():.6f}  min cos={cos.min():.6f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": res, "video_paths": kept, "indices": idx_log, "stored": ref,
                "encoder": a.encoder, "pooling": a.pooling}, a.out)
    print("saved", a.out, {k: tuple(v.shape) for k, v in res.items()})


if __name__ == "__main__":
    main()
