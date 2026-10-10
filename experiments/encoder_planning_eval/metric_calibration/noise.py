"""Noise estimation and significance tests for evaluation metrics.

Four sources of variation, each needs its own tool -- a claim of "improvement" must beat all
that apply to the comparison being made:

  clip sampling     which clips are in the eval set.     -> bootstrap / paired bootstrap over clips
                    (cluster bootstrap when clips share a video / subject)
  metric internals  Monte-Carlo pairs, probe subsample,  -> repeat the metric with new internal seeds
                    classifier init
  training seed     weight init + data order             -> train K seeds; `seed_summary`, `welch`
  frame sampling    which 16 frames of the video         -> re-extract features with jittered frame
                                                            indices (see extract_jitter.py)

`mdd()` converts a standard error of a difference into a minimum detectable difference at the
chosen power, i.e. the smallest effect that this experimental design can reliably call real.
"""
import numpy as np
from scipy import stats

Z975 = 1.959964
Z80 = 0.841621


def _counts(rng, n, B, groups=None):
    """(B, n) non-negative resampling weights. Plain bootstrap if groups is None, else cluster
    bootstrap: whole groups (e.g. all clips of one video) are drawn with replacement."""
    if groups is None:
        return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    g = inv.max() + 1
    gc = rng.multinomial(g, np.full(g, 1.0 / g), size=B).astype(np.float64)  # (B, g)
    return gc[:, inv]


def bootstrap_mean(x, B=2000, groups=None, seed=0, keep=None):
    """Bootstrap distribution of mean(x) over clips. x: (N,) per-clip values."""
    x = np.asarray(x, dtype=np.float64)
    if keep is not None:
        x = x[keep]
        groups = None if groups is None else np.asarray(groups)[keep]
    w = _counts(np.random.default_rng(seed), len(x), B, groups)
    return (w @ x) / w.sum(1)


def bootstrap_median(x, B=2000, groups=None, seed=0, keep=None):
    x = np.asarray(x, dtype=np.float64)
    if keep is not None:
        x = x[keep]
        groups = None if groups is None else np.asarray(groups)[keep]
    w = _counts(np.random.default_rng(seed), len(x), B, groups)
    order = np.argsort(x)
    xs, ws = x[order], w[:, order]
    cum = np.cumsum(ws, axis=1)
    half = cum[:, -1:] / 2.0
    idx = (cum >= half).argmax(1)
    return xs[idx]


def ci(samples, level=0.95):
    a = (1 - level) / 2
    return tuple(np.quantile(samples, [a, 1 - a]))


def paired_diff_bootstrap(xa, xb, B=2000, groups=None, seed=0, keep=None, how="mean"):
    """Bootstrap of stat(xa) - stat(xb) on the SAME clips (paired). Much tighter than two
    independent CIs because clip difficulty cancels. Returns dict(diff, lo, hi, se, p)."""
    xa, xb = np.asarray(xa, np.float64), np.asarray(xb, np.float64)
    f = bootstrap_mean if how == "mean" else bootstrap_median
    sa = f(xa, B, groups, seed, keep)
    sb = f(xb, B, groups, seed, keep)  # same seed -> same resampled clips -> paired
    d = sa - sb
    stat = np.mean if how == "mean" else np.median
    k = slice(None) if keep is None else keep
    obs = float(stat(xa[k]) - stat(xb[k]))
    lo, hi = ci(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": obs, "lo": float(lo), "hi": float(hi), "se": float(d.std(ddof=1)), "p": float(max(p, 1.0 / B))}


def mdd(se_diff, alpha=0.05, power=0.80):
    """Minimum detectable difference for a two-sided test given the SE of the difference."""
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    return float((za + zb) * se_diff)


def seed_summary(values):
    """values: metric across K training seeds of one configuration."""
    v = np.asarray(values, dtype=np.float64)
    k = len(v)
    sd = v.std(ddof=1) if k > 1 else np.nan
    se = sd / np.sqrt(k) if k > 1 else np.nan
    tcrit = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan
    return {"k": k, "mean": float(v.mean()), "sd": float(sd), "se": float(se),
            "ci_lo": float(v.mean() - tcrit * se), "ci_hi": float(v.mean() + tcrit * se),
            "min": float(v.min()), "max": float(v.max())}


def welch(a, b):
    """Welch t-test between two sets of seed-level metric values. Returns diff, se, t, p, df."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    diff = a.mean() - b.mean()
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = diff / se
    return {"diff": float(diff), "se": float(se), "t": float(t), "df": float(df),
            "p": float(2 * stats.t.sf(abs(t), df)), "mdd80": mdd(se)}


def cohens_d(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def normalized(value, floor, ceiling):
    """Position inside [floor, ceiling] in [0,1] (can fall outside if the value does)."""
    return (value - floor) / (ceiling - floor)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (multiple metrics / pairs tested at once)."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
