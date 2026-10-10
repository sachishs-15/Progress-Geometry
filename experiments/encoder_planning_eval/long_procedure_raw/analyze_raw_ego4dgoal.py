#!/usr/bin/env python
"""Training-free long-procedure analyses on raw DINOv2 Ego4D Goal-Step features.

This script does not construct or load project-trained checkpoints. It consumes the
existing frozen DINOv2 feature cache (N, 16, 384) and its CSV manifest, assembles sparse
ordered clip sequences by video_uid, and writes tables/plots to a new results folder.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except (TypeError, ValueError):
        return default


def cosine(a, b, eps=1e-12):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= eps or nb <= eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def clip_geometry(z):
    """Raw-space chord efficiency and consecutive-velocity cosine for one clip."""
    z = np.asarray(z, dtype=np.float64)
    v = np.diff(z, axis=0)
    path = float(np.linalg.norm(v, axis=1).sum())
    chord = float(np.linalg.norm(z[-1] - z[0]))
    straightness = chord / path if path > 1e-12 else np.nan
    norms = np.linalg.norm(v, axis=1)
    valid = (norms[:-1] > 1e-12) & (norms[1:] > 1e-12)
    turns = np.sum(v[:-1] * v[1:], axis=1) / (norms[:-1] * norms[1:] + 1e-12)
    turn_cos = float(np.mean(turns[valid])) if valid.any() else np.nan
    # Rank of position projected onto the start-to-end chord (temporal ordinality).
    chord_vec = z[-1] - z[0]
    denom = float(chord_vec @ chord_vec)
    if denom > 1e-12:
        progress = ((z - z[0]) @ chord_vec) / denom
        ordinal = float(spearmanr(np.arange(len(z)), progress).statistic)
    else:
        ordinal = np.nan
    return {"chord_efficiency": straightness, "velocity_turn_cosine": turn_cos,
            "path_length": path, "chord_length": chord, "time_progress_spearman": ordinal}


def aggregate_bootstrap(values, seed=17, n_boot=4000):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(x):
        return {"n_videos": 0, "mean": None, "median": None, "ci95_low": None,
                "ci95_high": None}
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return {"n_videos": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "ci95_low": float(np.quantile(boots, .025)),
            "ci95_high": float(np.quantile(boots, .975))}


def group_relevance(frame):
    vals = [str(x).strip().lower() for x in frame["parent_is_relevant"]
            if pd.notna(x) and str(x).strip()]
    if not vals:
        return "unlabeled"
    return Counter(vals).most_common(1)[0][0]


def ordered_groups(frame):
    """One event per annotated parent index; preserve unlabeled clips outside this table."""
    known = frame[frame["parent_step_idx_num"].notna()].copy()
    out = []
    for (uid, parent_idx), g in known.groupby(["video_uid", "parent_step_idx_num"], sort=False):
        g = g.sort_values("start_time")
        start = pd.to_numeric(g["parent_start_time"], errors="coerce").dropna()
        end = pd.to_numeric(g["parent_end_time"], errors="coerce").dropna()
        event_start = float(start.min()) if len(start) else float(g["start_time"].min())
        event_end = float(end.max()) if len(end) else float(g["end_time"].max())
        out.append({
            "video_uid": uid,
            "parent_step_idx": int(parent_idx),
            "start_time": event_start,
            "end_time": event_end,
            "relevance": group_relevance(g),
            "parent_category": first_nonempty(g.get("parent_step_category", pd.Series(dtype=object))),
            "parent_description": first_nonempty(g.get("parent_step_description", pd.Series(dtype=object))),
            "n_clips": int(len(g)),
            "clip_rows": g.index.to_list(),
            "embedding": g["embedding"].to_list(),
            "step_embedding": np.mean(np.stack(g["embedding"].to_list()), axis=0),
        })
    if not out:
        return out
    df = pd.DataFrame(out).sort_values(["video_uid", "start_time", "parent_step_idx"])
    return df.to_dict("records")


def first_nonempty(series):
    for x in series:
        if pd.notna(x) and str(x).strip():
            return str(x)
    return ""


def path_score(vectors):
    v = np.asarray(vectors, dtype=np.float64)
    if len(v) < 2:
        return np.nan
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return float(np.nanmean(np.sum(vn[:-1] * vn[1:], axis=1)))


def signed_order_score(vectors, axis=None):
    """Signed progress correlation; unlike adjacency similarity this changes under reversal."""
    v = np.asarray(vectors, dtype=np.float64)
    if len(v) < 3:
        return np.nan
    if axis is None:
        axis = v[-1] - v[0]
    if np.linalg.norm(axis) <= 1e-12:
        return np.nan
    return float(spearmanr(np.arange(len(v)), v @ axis).statistic)


def rank_metrics(scores, target):
    order = np.argsort(-np.asarray(scores), kind="stable")
    rank = int(np.where(order == target)[0][0]) + 1
    return rank, {"reciprocal_rank": 1.0 / rank, "recall_at_1": float(rank <= 1),
                  "recall_at_5": float(rank <= 5), "rank": rank,
                  "n_candidates": int(len(scores))}


def make_plots(clip_df, step_records, out_dir, max_plots=8, seed=4):
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    embs = np.stack(clip_df["embedding"].to_list()).astype(np.float32)
    # PCA is used for visualization only; every quantitative metric stays in raw space.
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    xy = pca.fit_transform(embs)
    clip_df["pca1"] = xy[:, 0]
    clip_df["pca2"] = xy[:, 1]
    by_video_steps = {}
    for e in step_records:
        by_video_steps.setdefault(e["video_uid"], []).append(e)
    counts = clip_df.groupby("video_uid").size().sort_values(ascending=False)
    selected = []
    for uid in counts.index:
        nsteps = len(by_video_steps.get(uid, []))
        if nsteps >= 3:
            selected.append(uid)
        if len(selected) >= max_plots:
            break
    for uid in selected:
        g = clip_df[clip_df.video_uid == uid].sort_values("start_time")
        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        markers = {"essential": "o", "optional": "^", "irrelevant": "s",
                   "unsure": "D", "unlabeled": "x"}
        relevance = g["parent_is_relevant"].fillna("unlabeled").astype(str).str.lower()
        known = g["parent_step_idx_num"].notna()
        lo = float(g.loc[known, "parent_step_idx_num"].min()) if known.any() else 0.0
        hi = float(g.loc[known, "parent_step_idx_num"].max()) if known.any() else 1.0
        if lo == hi:
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
        for rel, marker in markers.items():
            mask = relevance.eq(rel)
            if not mask.any():
                continue
            with_step = mask & known
            without_step = mask & ~known
            if with_step.any():
                ax.scatter(g.loc[with_step, "pca1"], g.loc[with_step, "pca2"],
                           c=g.loc[with_step, "parent_step_idx_num"], cmap="viridis", norm=norm,
                           marker=marker, s=42, edgecolors="black", linewidths=.25,
                           label=f"{rel} (step annotated)")
            if without_step.any():
                ax.scatter(g.loc[without_step, "pca1"], g.loc[without_step, "pca2"],
                           color="#999999", marker=marker, s=35,
                           label=f"{rel} (step missing)")
        pts = g[["pca1", "pca2"]].to_numpy()
        if len(pts) > 1:
            gap_s = g.start_time.to_numpy()[1:] - g.end_time.to_numpy()[:-1]
            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", alpha=.4,
                        linewidth=.8, linestyle="--" if gap_s[i] > 1.0 else "-")
            ax.scatter([pts[0, 0]], [pts[0, 1]], marker="*", color="green", s=100, label="first clip")
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], marker="P", color="red", s=70, label="last clip")
        if known.any():
            sm = ScalarMappable(norm=norm, cmap="viridis")
            fig.colorbar(sm, ax=ax, label="parent step index (color)")
        ax.set_title(f"{uid}: raw DINOv2 clip means, ordered by time\n"
                     f"{len(g)} observed clips; dashed edges span >1s gaps; marker shape shows relevance")
        ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / f"procedure_{uid}.png", dpi=160)
        plt.close(fig)
    return pca.explained_variance_ratio_.tolist(), selected


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # load on CPU only; no model/checkpoint is instantiated
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    video_paths = payload["video_paths"]
    if features.ndim != 3:
        raise ValueError(f"Expected raw cls features (N,T,D), got {features.shape}")
    manifest = pd.read_csv(args.manifest)
    if manifest["video_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate video_path values")
    idx = {p: i for i, p in enumerate(video_paths)}
    manifest["feature_index"] = manifest["video_path"].map(idx)
    missing = int(manifest["feature_index"].isna().sum())
    manifest = manifest[manifest["feature_index"].notna()].copy()
    manifest["feature_index"] = manifest["feature_index"].astype(int)
    if missing:
        raise ValueError(f"{missing} manifest rows had no exact video_path match")
    if len(manifest) != len(features):
        print(f"Note: {len(manifest)} manifest rows matched from {len(features)} feature rows")
    # raw per-frame DINOv2 embeddings; equal-weight frame mean is the clip descriptor
    manifest["embedding"] = [features[i].mean(axis=0).astype(np.float32)
                             for i in manifest["feature_index"]]
    manifest["parent_step_idx_num"] = pd.to_numeric(manifest["parent_step_idx"], errors="coerce")
    for col in ("start_time", "end_time", "parent_start_time", "parent_end_time"):
        manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    manifest = manifest.sort_values(["video_uid", "start_time", "end_time"]).reset_index(drop=True)
    manifest["clip_order"] = manifest.groupby("video_uid").cumcount()

    # Experiment 1: procedure sequences, clip counts, and gaps.
    video_rows = []
    for uid, g in manifest.groupby("video_uid", sort=False):
        g = g.sort_values("start_time")
        starts, ends = g.start_time.to_numpy(), g.end_time.to_numpy()
        gaps = starts[1:] - ends[:-1]
        video_rows.append({"video_uid": uid, "n_clips": len(g),
                           "n_annotated_parent_steps": int(g.parent_step_idx_num.nunique()),
                           "n_unlabeled_parent_clips": int(g.parent_step_idx_num.isna().sum()),
                           "observed_start_sec": float(np.nanmin(starts)),
                           "observed_end_sec": float(np.nanmax(ends)),
                           "positive_gap_count": int(np.sum(gaps > 0)),
                           "median_positive_gap_sec": float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0,
                           "max_positive_gap_sec": float(np.max(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0,
                           "total_positive_gap_sec": float(np.maximum(gaps, 0).sum())})
    video_df = pd.DataFrame(video_rows)
    video_df.to_csv(out_dir / "video_sequence_summary.csv", index=False)

    # Experiment 2: within-clip geometry and within/across-parent clip transitions.
    clip_metrics = [clip_geometry(features[i]) for i in manifest.feature_index]
    clip_metric_df = pd.DataFrame(clip_metrics)
    for c in clip_metric_df:
        manifest[c] = clip_metric_df[c].to_numpy()
    manifest[["video_uid", "clip_id", "clip_order", "start_time", "end_time",
              "parent_step_idx", "parent_is_relevant", "chord_efficiency",
              "velocity_turn_cosine", "path_length", "chord_length",
              "time_progress_spearman"]].to_csv(out_dir / "clip_geometry.csv", index=False)

    trans_rows = []
    for uid, g in manifest.groupby("video_uid", sort=False):
        g = g.sort_values("start_time")
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            same = (np.isfinite(a["parent_step_idx_num"]) and np.isfinite(b["parent_step_idx_num"])
                    and a["parent_step_idx_num"] == b["parent_step_idx_num"])
            trans_rows.append({"video_uid": uid,
                               "transition_type": ("within_parent" if same else
                                                   "parent_change" if (np.isfinite(a["parent_step_idx_num"]) and np.isfinite(b["parent_step_idx_num"]))
                                                   else "unknown_parent"),
                               "gap_sec": max(0.0, float(b["start_time"] - a["end_time"])),
                               "feature_l2": float(np.linalg.norm(b["embedding"] - a["embedding"])),
                               "feature_cosine": cosine(a["embedding"], b["embedding"]),
                               "same_parent": bool(same)})
    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(out_dir / "clip_transitions.csv", index=False)
    transition_summary = []
    for typ, g in trans_df.groupby("transition_type"):
        for gap_label, gg in (("all", g), ("gap_le_1s", g[g.gap_sec <= 1.0])):
            if len(gg):
                transition_summary.append({"transition_type": typ, "gap_filter": gap_label,
                                           "n_transitions": len(gg),
                                           "median_gap_sec": float(gg.gap_sec.median()),
                                           "median_feature_l2": float(gg.feature_l2.median()),
                                           "median_feature_cosine": float(gg.feature_cosine.median())})
    pd.DataFrame(transition_summary).to_csv(out_dir / "transition_summary.csv", index=False)

    step_records = ordered_groups(manifest)
    step_df = pd.DataFrame([{k: v for k, v in e.items() if k not in ("embedding", "clip_rows")}
                            for e in step_records])
    # store descriptors for reuse without copying the source 16-frame feature tensor
    if step_records:
        np.savez_compressed(out_dir / "parent_step_embeddings.npz",
                            video_uid=np.array([e["video_uid"] for e in step_records]),
                            parent_step_idx=np.array([e["parent_step_idx"] for e in step_records]),
                            embeddings=np.stack([e["step_embedding"] for e in step_records]).astype(np.float32))
    step_df.drop(columns=[c for c in ("step_embedding",) if c in step_df]).to_csv(
        out_dir / "parent_steps.csv", index=False)

    # Per-parent summaries: within-step dispersion and available multi-clip straightness.
    within_step_rows = []
    for e in step_records:
        vectors = np.stack(e["embedding"])
        center = e["step_embedding"]
        center_cos = [cosine(v, center) for v in vectors]
        summary = {"video_uid": e["video_uid"], "parent_step_idx": e["parent_step_idx"],
                   "relevance": e["relevance"], "n_clips": e["n_clips"],
                   "within_step_mean_cosine_to_centroid": float(np.nanmean(center_cos)),
                   "within_step_median_feature_l2_to_centroid": float(np.median(np.linalg.norm(vectors-center, axis=1))),
                   "median_member_clip_chord_efficiency": float(np.nanmedian(
                       manifest.loc[e["clip_rows"], "chord_efficiency"].to_numpy(dtype=float)))}
        # A parent-level path statistic is only defined for >=3 observed clips. It describes
        # sparse sampled clips (including gaps), so is clearly labeled as such.
        if len(vectors) >= 3:
            summary["sparse_parent_path_efficiency"] = float(
                np.linalg.norm(vectors[-1]-vectors[0]) /
                max(np.linalg.norm(np.diff(vectors, axis=0), axis=1).sum(), 1e-12))
        else:
            summary["sparse_parent_path_efficiency"] = np.nan
        within_step_rows.append(summary)
    within_step_df = pd.DataFrame(within_step_rows)
    within_step_df.to_csv(out_dir / "parent_step_geometry.csv", index=False)
    parent_geometry_summary = {}
    for min_clips in (2, 3):
        subset = within_step_df[within_step_df.n_clips >= min_clips]
        parent_geometry_summary[str(min_clips)] = {
            "n_parent_steps": int(len(subset)),
            "median_within_step_cosine_to_centroid": float(subset.within_step_mean_cosine_to_centroid.median()) if len(subset) else None,
            "median_feature_l2_to_centroid": float(subset.within_step_median_feature_l2_to_centroid.median()) if len(subset) else None,
            "median_sparse_path_efficiency": float(subset.sparse_parent_path_efficiency.median()) if min_clips >= 3 and len(subset) else None}

    # Experiment 3: correct order vs reversed and within-video shuffled parent events.
    rng = np.random.default_rng(args.seed)
    order_rows = []
    by_uid = {}
    for e in step_records:
        by_uid.setdefault(e["video_uid"], []).append(e)
    for uid, all_events in by_uid.items():
        all_events.sort(key=lambda e: (e["start_time"], e["parent_step_idx"]))
        for scope, events in (("all_parent_events", all_events),
                              ("essential_only", [e for e in all_events if e["relevance"] == "essential"])):
            if len(events) < 3:
                continue
            vectors = [e["step_embedding"] for e in events]
            real = signed_order_score(vectors)
            axis = vectors[-1] - vectors[0]
            reverse = signed_order_score(vectors[::-1], axis=axis)
            shuffled = []
            base = np.arange(len(vectors))
            for _ in range(args.permutations):
                perm = rng.permutation(base)
                shuffled.append(signed_order_score([vectors[i] for i in perm], axis=axis))
            shuffled = np.asarray(shuffled)
            order_rows.append({"video_uid": uid, "scope": scope, "n_parent_steps": len(events),
                               "real_signed_progress_score": real,
                               "reverse_signed_progress_score": reverse,
                               "real_beats_reverse": float(real > reverse + 1e-9),
                               "real_beats_random_fraction": float(np.mean(real > shuffled + 1e-9)),
                               "real_random_percentile": float((np.sum(shuffled <= real) + 1) / (len(shuffled) + 1))})
    order_df = pd.DataFrame(order_rows)
    order_df.to_csv(out_dir / "step_order_test_per_video.csv", index=False)

    # Experiment 4: no-training next essential step retrieval within a video.
    retrieval_rows = []
    for uid, events in by_uid.items():
        events.sort(key=lambda e: (e["start_time"], e["parent_step_idx"]))
        essentials = [i for i, e in enumerate(events) if e["relevance"] == "essential"]
        for pos, cur_idx in enumerate(essentials[:-1]):
            target_idx = essentials[pos + 1]
            cand_idx = [j for j in range(cur_idx + 1, len(events))]
            if len(cand_idx) < 2:
                continue
            cur = events[cur_idx]["step_embedding"]
            cand = np.stack([events[j]["step_embedding"] for j in cand_idx])
            cosine_scores = np.array([cosine(cur, x) for x in cand])
            target_pos = cand_idx.index(target_idx)
            rank_cos, m_cos = rank_metrics(cosine_scores, target_pos)
            rank_earliest = target_pos + 1
            n_cand = len(cand_idx)
            random_rr = sum(1.0 / r for r in range(1, n_cand + 1)) / n_cand
            row = {"video_uid": uid, "query_parent_step_idx": events[cur_idx]["parent_step_idx"],
                   "target_parent_step_idx": events[target_idx]["parent_step_idx"],
                   "query_end_sec": events[cur_idx]["end_time"],
                   "target_start_sec": events[target_idx]["start_time"],
                   "n_candidates": m_cos["n_candidates"], "rank_cosine_current": rank_cos,
                   "rr_random_expected": random_rr,
                   "r1_random_expected": 1.0 / n_cand,
                   "r5_random_expected": min(5, n_cand) / n_cand,
                   "rr_cosine_current": m_cos["reciprocal_rank"],
                   "r1_cosine_current": m_cos["recall_at_1"],
                   "r5_cosine_current": m_cos["recall_at_5"],
                   "rank_earliest_future_baseline": rank_earliest,
                   "rr_earliest_future_baseline": 1.0 / rank_earliest,
                   "r1_earliest_future_baseline": float(rank_earliest <= 1),
                   "r5_earliest_future_baseline": float(rank_earliest <= 5)}
            if pos > 0:
                prev = events[essentials[pos - 1]]["step_embedding"]
                query = cur + (cur - prev)
                extrap_scores = np.array([cosine(query, x) for x in cand])
                rank_ext, m_ext = rank_metrics(extrap_scores, target_pos)
                row.update({"rank_linear_extrapolation": rank_ext,
                            "rr_linear_extrapolation": m_ext["reciprocal_rank"],
                            "r1_linear_extrapolation": m_ext["recall_at_1"],
                            "r5_linear_extrapolation": m_ext["recall_at_5"]})
            retrieval_rows.append(row)
    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df.to_csv(out_dir / "next_step_retrieval_queries.csv", index=False)

    # Aggregated report, grouping bootstrap at video level wherever possible.
    geometry_summary = {}
    for label, g in manifest.groupby("parent_is_relevant", dropna=False):
        key = str(label) if pd.notna(label) else "unlabeled"
        geometry_summary[key] = {
            "n_clips": int(len(g)),
            "chord_efficiency_median": float(g.chord_efficiency.median()),
            "velocity_turn_cosine_median": float(g.velocity_turn_cosine.median()),
            "path_length_median": float(g.path_length.median())}
    retrieval_by_video = {}
    if len(retrieval_df):
        for metric in ("rr_cosine_current", "r1_cosine_current", "r5_cosine_current",
                       "rr_random_expected", "r1_random_expected", "r5_random_expected",
                       "rr_earliest_future_baseline", "r1_earliest_future_baseline",
                       "r5_earliest_future_baseline", "rr_linear_extrapolation",
                       "r1_linear_extrapolation", "r5_linear_extrapolation"):
            if metric not in retrieval_df:
                continue
            per_v = retrieval_df.groupby("video_uid")[metric].mean()
            retrieval_by_video[metric] = aggregate_bootstrap(per_v.to_numpy(), seed=args.seed)
    order_summary = {}
    if len(order_df):
        for scope, scoped in order_df.groupby("scope"):
            order_summary[scope] = {}
            for metric in ("real_beats_reverse", "real_beats_random_fraction", "real_random_percentile"):
                order_summary[scope][metric] = aggregate_bootstrap(scoped[metric].to_numpy(), seed=args.seed)
    trans_summary = {f"{r['transition_type']}__{r['gap_filter']}": r for r in transition_summary}
    summary = {
        "dataset": f"Ego4D Goal-Step {args.split}",
        "features": str(args.features), "manifest": str(args.manifest),
        "method": "raw frozen DINOv2 CLS features; equal mean of 16 frames per clip; no trained checkpoint/model",
        "n_feature_rows": int(len(features)), "n_matched_clips": int(len(manifest)),
        "feature_dim": int(features.shape[-1]), "frames_per_clip": int(features.shape[1]),
        "n_videos": int(manifest.video_uid.nunique()),
        "n_clips_with_parent_step": int(manifest.parent_step_idx_num.notna().sum()),
        "n_clips_without_parent_step": int(manifest.parent_step_idx_num.isna().sum()),
        "n_clips_without_parent_relevance_label": int(manifest.parent_is_relevant.isna().sum()),
        "parent_relevance_counts": {str(k) if pd.notna(k) else "unlabeled": int(v)
                                     for k, v in manifest.parent_is_relevant.value_counts(dropna=False).items()},
        "video_clip_count_median": float(video_df.n_clips.median()),
        "video_clip_count_p90": float(video_df.n_clips.quantile(.9)),
        "total_positive_unobserved_gap_hours": float(video_df.total_positive_gap_sec.sum()/3600),
        "pca_variance_ratio_visualization_only": None,
        "geometry_by_parent_relevance": geometry_summary,
        "parent_step_geometry_multi_clip": parent_geometry_summary,
        "within_vs_boundary_transitions": trans_summary,
        "order_test_video_macro": order_summary,
        "next_step_retrieval_video_macro": retrieval_by_video,
        "caveats": [
            "Each video is represented by its available annotated clips, not continuous frame coverage.",
            "PCA is visualization only; quantitative analyses use original 384-D feature space.",
            "The order score is a signed projection on the true start-to-end feature axis; it can reflect scene continuity and is not proof of procedural understanding.",
            "Retrieval candidates are later annotated parent steps in the same video; unlabeled parent events are included as distractors.",
            "Bootstrap intervals resample videos, not clips."
        ]}
    pca_variance, plot_ids = make_plots(manifest, step_records, out_dir,
                                        max_plots=args.max_plots, seed=args.seed)
    summary["pca_variance_ratio_visualization_only"] = pca_variance
    summary["plotted_video_uids"] = plot_ids
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    write_report(summary, out_dir / "FINDINGS.md")
    print(f"Wrote raw-feature analysis for {len(manifest)} clips / {manifest.video_uid.nunique()} videos to {out_dir}")


def fmt_ci(d):
    if not d or d.get("mean") is None:
        return "not enough eligible videos"
    return f"{d['mean']:.3f} (95% video-bootstrap CI {d['ci95_low']:.3f}–{d['ci95_high']:.3f}; n={d['n_videos']})"


def write_report(s, path):
    lines = [
        "# Raw DINOv2 long-procedure analysis",
        "",
        "## Setup",
        "",
        f"- Data: {s['dataset']}; {s['n_matched_clips']:,} clips across {s['n_videos']:,} videos.",
        f"- Features: raw frozen DINOv2, {s['frames_per_clip']} sampled frames per clip, "
        f"{s['feature_dim']}-D; clip descriptor is the untrained frame mean.",
        "- No project-trained checkpoints or predictive models were used. PCA is only used for plots.",
        f"- Parent-step index: {s['n_clips_with_parent_step']:,} clips labeled; "
        f"{s['n_clips_without_parent_step']:,} without an index. Relevance is unlabeled on "
        f"{s['n_clips_without_parent_relevance_label']:,} clips.",
        f"- Sum of positive gaps between observed clips: {s['total_positive_unobserved_gap_hours']:.1f} hours.",
        "",
        "## Results",
        "",
        "### Geometry within clips",
        "",
        "Raw chord efficiency is endpoint distance divided by total feature-path length (1 means a straight path).",
    ]
    for rel, m in s["geometry_by_parent_relevance"].items():
        lines.append(f"- **{rel}:** n={m['n_clips']:,}; median chord efficiency="
                     f"{m['chord_efficiency_median']:.3f}; median velocity-turn cosine="
                     f"{m['velocity_turn_cosine_median']:.3f}.")
    lines += ["", "### Geometry across multiple clips in a parent step", "",
              "Singleton parent steps are excluded from within-step cohesion. Sparse path efficiency includes time gaps between observed clips."]
    for min_clips, m in s["parent_step_geometry_multi_clip"].items():
        path_text = f"; median sparse path efficiency={m['median_sparse_path_efficiency']:.3f}" if m["median_sparse_path_efficiency"] is not None else ""
        lines.append(f"- **At least {min_clips} clips:** n={m['n_parent_steps']:,}; median cosine to step centroid={m['median_within_step_cosine_to_centroid']:.3f}; median L2 dispersion={m['median_feature_l2_to_centroid']:.3f}{path_text}.")
    lines += ["", "### Transitions between observed clips", ""]
    for typ, m in s["within_vs_boundary_transitions"].items():
        lines.append(f"- **{typ} ({m['gap_filter']}):** n={m['n_transitions']:,}; median gap={m['median_gap_sec']:.1f}s; "
                     f"median feature distance={m['median_feature_l2']:.3f}; "
                     f"median cosine={m['median_feature_cosine']:.3f}.")
    lines += ["", "### Step-order diagnostic", "",
              "Score is Spearman correlation between event order and projection on the chronological start-to-end feature axis; shuffled sequences use that same axis.", ""]
    for scope, metrics in s["order_test_video_macro"].items():
        lines.append(f"- **{scope}:**")
        for key, d in metrics.items():
            lines.append(f"  - `{key}`: {fmt_ci(d)}")
    lines += ["", "### Next-essential-step retrieval", "",
              "Candidates are later parent-step events from the same video; the immediate next essential event is the target. The chronological baseline ranks the earliest future event first.", ""]
    for key, d in s["next_step_retrieval_video_macro"].items():
        lines.append(f"- `{key}`: {fmt_ci(d)}")
    lines += ["", "Random rank expectation is computed for each query's candidate count, then averaged by video.", "",
              "## Interpretation limits", ""]
    lines += [f"- {x}" for x in s["caveats"]]
    lines += ["- These results establish a frozen-feature baseline. They cannot show that an induced-straightness objective improves performance; that requires comparing a trained representation later.",
               "- Parent-step geometry uses sparse annotated clips and must be interpreted together with the reported time gaps.",
               ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--features", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--max-plots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.features = args.features or f"/mnt/rdata7/proprogress/data/ego4dgoal/precomputed_features/dinov2/ego4dgoal_{args.split}.pt"
    args.manifest = args.manifest or f"/home/sachish/proc-planning/data/ego4dgoal/ego4dgoal_{args.split}.csv"
    args.out = args.out or str(root / "results" / f"{args.split}_raw_dinov2")
    run(args)


if __name__ == "__main__":
    main()
