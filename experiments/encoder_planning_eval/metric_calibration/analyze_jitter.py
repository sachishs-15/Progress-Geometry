"""Test-retest reliability of per-clip metrics under frame-sampling jitter (extract_jitter.py output)."""
import sys
import numpy as np, torch, pandas as pd
from metrics_batch import per_clip_metrics, PER_CLIP_KEYS

def icc1(x):
    """One-way random-effects ICC(1,1). x: (clips, k replicates)."""
    n, k = x.shape
    gm = x.mean()
    msb = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)

def run(enc):
    d = torch.load(f"results/jitter/{enc}.pt", weights_only=False)
    f = {k: v.numpy().astype(np.float64) for k, v in d["features"].items()}
    reps = [k for k in f if k.startswith("rep")]
    pcs = {k: per_clip_metrics(f[k]) for k in f}
    keep = np.all([pcs[k][1] for k in f], 0)
    rows = []
    for m in PER_CLIP_KEYS:
        X = np.stack([pcs[k][0][m][keep] for k in reps], 1)  # (clips, reps)
        base = pcs["base"][0][m][keep]
        rows.append({"encoder": enc, "metric": m, "icc": icc1(X), "mean_base": base.mean(), "mean_reps": X.mean(),
                     "shift_base_vs_reps": base.mean() - X.mean(), "sd_rep_means": X.mean(0).std(ddof=1),
                     "sd_within_clip": np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum() / (X.shape[0] * (X.shape[1] - 1))),
                     "sd_between_clip": X.mean(1).std(ddof=1), "n_clips": int(keep.sum())})
    return rows

if __name__ == "__main__":
    rows = [r for e in sys.argv[1:] for r in run(e)]
    df = pd.DataFrame(rows); df.to_csv("results/jitter_reliability.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    for e, g in df.groupby("encoder"):
        print(f"\n== {e}  (n={g.n_clips.iloc[0]} clips, 4 jittered samplings of each video)")
        print(g[["metric", "icc", "sd_within_clip", "sd_between_clip", "mean_base", "shift_base_vs_reps"]].round(4).to_string(index=False))
