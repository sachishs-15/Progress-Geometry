"""Vectorised per-clip trajectory metrics over (N, T, D) latents.

Numerically equivalent to `progress_metrics/compute_progress_metrics.py::compute_metrics_dict`
(which loops clip by clip and takes ~minutes per encoder), but ~100x faster, so a metric can
be re-evaluated on hundreds of null / bootstrap / oracle replicates.
`tests/test_metrics_batch.py` checks the two agree on real SSv2 latents.

Returns per-clip arrays (not summaries), so callers can bootstrap over clips. Clips dropped by
LiFT's `clip_progress_metrics` (T<3 or zero chord after normalisation) are marked in `keep`;
metrics that the reference code maps None -> 0.0 inside a kept clip are mapped the same way.
"""
import numpy as np
from scipy.stats import rankdata

EPS = 1e-8
STEP_THRESH = 1e-6

PER_CLIP_KEYS = [
    # LiFT (on L2-normalised frames)
    "trajectory_straightness", "time_progress_spearman", "progress_efficiency",
    "monotonic_progress_fraction", "adjacent_temporal_nn_fraction",
    "adjacent_cosine_distance", "start_end_displacement",
    # project additions
    "pc1_time_spearman", "distance_to_goal_spearman",
    "mean_velocity_cosine", "mean_chord_alignment",
    "mean_velocity_cosine_raw", "mean_chord_alignment_raw",
    "trajectory_straightness_raw", "norm_growth", "norm_std_within_clip",
    "adjacent_l2_distance", "norm_turning_point_fraction",
    "direction_reversal_fraction", "within_clip_mean_pairwise_l2",
]


def _l2n(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + EPS)


def _spearman_rows(x_rank_axis, y):
    """Spearman between a fixed 1-D index vector and each row of y (N, T). nan if y row is constant."""
    ry = rankdata(y, axis=1)
    rx = rankdata(x_rank_axis)
    ry = ry - ry.mean(1, keepdims=True)
    rx = rx - rx.mean()
    den = np.sqrt((ry ** 2).sum(1) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ry * rx).sum(1) / den


def _vel_cos(z, normalize, valid_out=False):
    if normalize:
        z = _l2n(z)
    steps = z[:, 1:] - z[:, :-1]
    v1, v2 = steps[:, :-1], steps[:, 1:]
    n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    valid = (n1 > STEP_THRESH) & (n2 > STEP_THRESH)
    cos = (v1 * v2).sum(-1) / (n1 * n2 + EPS)
    return cos, valid


def _masked_mean(x, valid):
    cnt = valid.sum(1)
    s = np.where(valid, x, 0.0).sum(1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def per_clip_metrics(z):
    """z: (N, T, D) float array. Returns (dict key -> (N,) array, keep (N,) bool)."""
    z = np.asarray(z, dtype=np.float64)
    n, t, _ = z.shape
    out = {}

    # ---- LiFT block, on normalised frames -------------------------------------------
    zn = _l2n(z)
    steps = zn[:, 1:] - zn[:, :-1]
    step_norms = np.linalg.norm(steps, axis=-1)
    path = step_norms.sum(1)
    sed = np.linalg.norm(zn[:, -1] - zn[:, 0], axis=-1)
    out["trajectory_straightness"] = np.where(path > EPS, sed / np.maximum(path, EPS), 0.0)
    out["start_end_displacement"] = sed
    out["adjacent_cosine_distance"] = (1.0 - (zn[:, :-1] * zn[:, 1:]).sum(-1)).mean(1)

    u = zn[:, -1] - zn[:, 0]
    u_sq = (u ** 2).sum(-1)
    keep = u_sq >= 1e-10
    p = ((zn - zn[:, :1]) * u[:, None, :]).sum(-1) / np.where(keep, u_sq, 1.0)[:, None]
    tidx = np.arange(t)
    out["time_progress_spearman"] = _spearman_rows(tidx, p)
    p_steps = p[:, 1:] - p[:, :-1]
    tv = np.abs(p_steps).sum(1)
    out["progress_efficiency"] = np.where(tv > EPS, 1.0 / np.maximum(tv, EPS), 0.0)
    out["monotonic_progress_fraction"] = (p_steps >= -EPS).mean(1)
    sims = zn @ zn.transpose(0, 2, 1)
    sims[:, np.arange(t), np.arange(t)] = -np.inf
    nn_idx = sims.argmax(-1)
    out["adjacent_temporal_nn_fraction"] = (np.abs(nn_idx - tidx[None]) == 1).mean(1)

    # ---- project block -----------------------------------------------------------------
    zc = zn - zn.mean(1, keepdims=True)
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    proj = np.einsum("ntd,nd->nt", zc, vt[:, 0])
    flip = proj[:, -1] < proj[:, 0]
    proj = np.where(flip[:, None], -proj, proj)
    pc1 = _spearman_rows(tidx, proj)
    out["pc1_time_spearman"] = np.where(s[:, 0] < EPS, 0.0, pc1)

    dist = np.linalg.norm(z[:, :-1] - z[:, -1:], axis=-1)
    d2g = _spearman_rows(np.arange(t - 1, 0, -1), dist)
    out["distance_to_goal_spearman"] = np.where(np.ptp(dist, axis=1) == 0, 0.0, d2g)

    for name, normalize in (("", True), ("_raw", False)):
        cos, valid = _vel_cos(z, normalize)
        out[f"mean_velocity_cosine{name}"] = _masked_mean(cos, valid)
        zz = _l2n(z) if normalize else z
        st = zz[:, 1:] - zz[:, :-1]
        sn = np.linalg.norm(st, axis=-1)
        chord = zz[:, -1] - zz[:, 0]
        cn = np.linalg.norm(chord, axis=-1)
        ccos = np.einsum("ntd,nd->nt", st, chord) / (sn * cn[:, None] + EPS)
        valid = (sn > STEP_THRESH) & (cn[:, None] >= STEP_THRESH)
        out[f"mean_chord_alignment{name}"] = _masked_mean(ccos, valid)

    raw_steps = z[:, 1:] - z[:, :-1]
    raw_sn = np.linalg.norm(raw_steps, axis=-1)
    raw_path = raw_sn.sum(1)
    out["trajectory_straightness_raw"] = np.where(
        raw_path > EPS, np.linalg.norm(z[:, -1] - z[:, 0], axis=-1) / np.maximum(raw_path, EPS), 0.0)
    norms = np.linalg.norm(z, axis=-1)
    out["norm_growth"] = np.where(norms[:, 0] >= EPS, norms[:, -1] / np.maximum(norms[:, 0], EPS), 0.0)
    out["norm_std_within_clip"] = norms.std(1)
    out["adjacent_l2_distance"] = raw_sn.mean(1)

    dn = np.diff(norms, axis=1)
    sgn = np.sign(dn)
    ok = np.abs(dn) >= EPS
    # reference drops |d|<eps steps then compares adjacent survivors; with continuous data none are dropped
    if ok.all():
        out["norm_turning_point_fraction"] = (sgn[:, 1:] != sgn[:, :-1]).mean(1)
    else:
        tp = np.zeros(n)
        for i in range(n):
            sg = sgn[i][ok[i]]
            tp[i] = float((sg[1:] != sg[:-1]).mean()) if len(sg) >= 2 else 0.0
        out["norm_turning_point_fraction"] = tp

    cos_raw, valid_raw = _vel_cos(z, False)
    out["direction_reversal_fraction"] = _masked_mean((cos_raw < 0).astype(float), valid_raw)

    iu = np.triu_indices(t, 1)
    g = z @ z.transpose(0, 2, 1)  # Gram form avoids an (N,T,T,D) tensor
    sq = np.diagonal(g, axis1=1, axis2=2)
    d2 = sq[:, :, None] + sq[:, None, :] - 2 * g
    out["within_clip_mean_pairwise_l2"] = np.sqrt(np.maximum(d2, 0))[:, iu[0], iu[1]].mean(1)

    for k, v in out.items():
        out[k] = np.nan_to_num(v, nan=0.0)  # reference: spearman nan would propagate; none observed on real data
    return out, keep


def dataset_metrics(z, max_pairs=200_000, seed=0):
    """Whole-dataset collapse diagnostics (same definitions as compute_progress_metrics)."""
    z = np.asarray(z, dtype=np.float64)
    n, t, d = z.shape
    zp = z.reshape(-1, d)
    zc = zp - zp.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(zc.T @ zc)  # sigma_k^2; same spectrum as the SVD, much cheaper
    s = np.sqrt(np.clip(lam, 0, None))
    s = s[s > 1e-12]
    p = s / s.sum()
    er = float(np.exp(-(p * np.log(p)).sum()))
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    k = i != j
    i, j = i[k], j[k]
    a = rng.integers(0, t, len(i))
    b = rng.integers(0, t, len(i))
    return {
        "effective_rank": er,
        "effective_rank_fraction": er / d,
        "dead_dim_fraction": float((zp.std(0) < 1.0).mean()),
        "mean_norm": float(np.linalg.norm(zp, axis=1).mean()),
        "mean_pairwise_l2_across_clips": float(np.linalg.norm(z[i, a] - z[j, b], axis=-1).mean()),
    }


def summarize(per_clip, keep, how="mean"):
    f = np.mean if how == "mean" else np.median
    return {k: float(f(v[keep])) for k, v in per_clip.items()}
