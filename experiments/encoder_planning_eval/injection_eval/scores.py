"""Per-clip anomaly scores for a latent path, and the statistics used to compare clean
and perturbed clips. Every score is oriented so that a higher value means "more
anomalous"; an AUROC below 0.5 therefore means the perturbed clip looked more normal
than the clean one.

Geometry scores need only a (B, T, D) latent path and work for raw features too:
    jump_max, jump_spike   largest step between consecutive frames, and it relative to the
                           clip's median step
    cv_max, cv_mean        constant-velocity extrapolation error ||z[t+1] - 2 z[t] + z[t-1]||
    chord_dev              largest distance from the straight line start -> end, in units
                           of the mean step
    backslide              largest drop of progress below its running maximum, where
                           progress is the projection onto start -> end (1 = whole chord)
    step_cv, stall         variation of step sizes, and how small the smallest step is
                           relative to the median (1 = a perfectly still step)
Predictor scores need the world model's one-step prediction error per transition:
    pe_max, pe_mean, pe_spike
"""
import numpy as np
from scipy.stats import rankdata

EPS = 1e-8
GEOMETRY_SCORES = ("jump_max", "jump_spike", "cv_max", "cv_mean", "chord_dev", "backslide",
                   "step_cv", "stall")
PREDICTOR_SCORES = ("pe_max", "pe_mean", "pe_spike")


def geometry_scores(z):
    """z: (B, T, D) numpy. Returns ({score: (B,)}, step sizes (B, T-1))."""
    z = np.asarray(z, dtype=np.float64)
    d = np.linalg.norm(np.diff(z, axis=1), axis=-1)
    med = np.median(d, axis=1)
    out = {"jump_max": d.max(1), "jump_spike": d.max(1) / (med + EPS)}
    cv = np.linalg.norm(z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2], axis=-1)
    out["cv_max"], out["cv_mean"] = cv.max(1), cv.mean(1)
    rel = z - z[:, :1]
    u = z[:, -1] - z[:, 0]
    p = (rel * u[:, None]).sum(-1) / ((u * u).sum(-1)[:, None] + EPS)
    dev = np.linalg.norm(rel - p[..., None] * u[:, None], axis=-1)
    out["chord_dev"] = dev.max(1) / (d.mean(1) + EPS)
    out["backslide"] = (np.maximum.accumulate(p, axis=1) - p).max(1)
    out["step_cv"] = d.std(1) / (d.mean(1) + EPS)
    out["stall"] = 1 - d.min(1) / (med + EPS)
    return out, d


def predictor_scores(pe):
    """pe: (B, T-1) one-step prediction error per transition."""
    pe = np.asarray(pe, dtype=np.float64)
    return {"pe_max": pe.max(1), "pe_mean": pe.mean(1),
            "pe_spike": pe.max(1) / (np.median(pe, axis=1) + EPS)}


def auroc(pert, clean):
    """P(perturbed score > clean score), ties counted half."""
    pert, clean = np.asarray(pert, float), np.asarray(clean, float)
    n1, n0 = len(pert), len(clean)
    r = rankdata(np.concatenate([pert, clean]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_win(pert, clean):
    pert, clean = np.asarray(pert, float), np.asarray(clean, float)
    return float(np.mean((pert > clean) + 0.5 * (pert == clean)))


def compare(pert, clean, n_boot=200, seed=0):
    """AUROC (pooled), bootstrap 95% CI over clips, and the paired win rate. Clips whose
    score is not finite in either condition are dropped from both."""
    pert, clean = np.asarray(pert, float), np.asarray(clean, float)
    ok = np.isfinite(pert) & np.isfinite(clean)
    pert, clean = pert[ok], clean[ok]
    n = len(pert)
    if n < 10:
        return {"auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "paired_win": np.nan, "n": n}
    rng = np.random.default_rng(seed)
    boots = [auroc(pert[i], clean[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": auroc(pert, clean), "ci_lo": float(lo), "ci_hi": float(hi),
            "paired_win": paired_win(pert, clean), "n": n}


def localization(per_transition, seams, tolerance=1):
    """Share of clips whose largest per-transition value lands within `tolerance` of a seam,
    and the share expected if the peak were placed uniformly at random.
    per_transition: (B, T-1); seams: list of seam lists, one per clip (None = clip invalid)."""
    hits, chance = [], []
    n_trans = per_transition.shape[1]
    for row, s in zip(per_transition, seams):
        if not s or not np.isfinite(row).all():
            continue
        near = {t + k for t in s for k in range(-tolerance, tolerance + 1) if 0 <= t + k < n_trans}
        hits.append(int(np.argmax(row)) in near)
        chance.append(len(near) / n_trans)
    if not hits:
        return {"hit": np.nan, "chance": np.nan, "n": 0}
    return {"hit": float(np.mean(hits)), "chance": float(np.mean(chance)), "n": len(hits)}
