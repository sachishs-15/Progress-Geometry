"""P-JEPA Appendix A.3-style trajectory straightness/ordinality metrics for
ProceduralWorldModel, on the same SSv2 data and via the same clip_progress_metrics
function as the LiFT model family's own results
(external/chirality_in_action/cia_full_dataset/results/ssv2_progress_*), so the numbers
are directly comparable.

One of the two encoder-testing analyses under experiments/encoder_planning_eval/ (see
its top-level README) -- lives here rather than under experiments/predictive_world_model/
because its real subject is encoder comparison (mostly via --raw, no training at all);
checkpoint mode still works by reaching across to predictive_world_model/models/ (see the
sys.path insert below), so a trained ProceduralWorldModel checkpoint can be scored too.

clip_progress_metrics needs a 2D [T, D] per-frame vector per clip. The quantity fed to
it is model.aggregate(model.encoder(x)) — for token_mode='cls' or projector_type=
'global' this is a no-op (z is already 1 token/frame, just squeezed to [T, D]); for
projector_type='channel', raw model.encoder(x) is a 4D (B,T,196,8) per-PATCH grid, not
usable here directly, so aggregate() runs it through the model's own learned
AggregationHead (the same one used for the curvature/VICReg regularizers during
training, see models/aggregation_head.py) to collapse it to one representative vector
per frame first — the same "per-frame z" LiFT's PredictiveWorldModel produces natively,
just recovered via the aggregation head for spatial latents instead of being the raw
encoder output.

Usage (trained checkpoint -- path is relative to this file's own directory, reaching
back into predictive_world_model/'s checkpoints/):
    python compute_progress_metrics.py \\
        --checkpoint ../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/best.ckpt \\
        --config ../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/config.yaml \\
        --features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt \\
        --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \\
        --out results/ssv2_progress_k400.json

Usage (raw baselines, no model/checkpoint at all -- just reduces whatever a --features
file already contains): --raw cls needs a cls-shaped (N,T,D) file and uses it untouched;
--raw mean/flatten need a patch-shaped (N,T,P,D) file and either mean-pool or concatenate
the P patch vectors per frame (flatten is the un-learned counterpart of AggregationHead's
MLP -- see models/aggregation_head.py). "a specific encoder itself" is just a matter of
pointing --features at that encoder's own precomputed file; --raw is what lets you read it
without a trained ProceduralWorldModel:
    python compute_progress_metrics.py --raw cls \\
        --features .../precomputed_features/dinov2_cls/ssv2_train.pt \\
        --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \\
        --out results/ssv2_progress_raw_cls.json

    python compute_progress_metrics.py --raw flatten \\
        --features .../precomputed_features/dinov2_patch/ssv2_train.pt \\
        --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \\
        --out results/ssv2_progress_raw_flatten.json

On top of clip_progress_metrics' own 7 metrics, this script adds its own (see each one's
own docstring below) -- NOT added to the vendored clip_progress_metrics itself, so the
original 7 stay byte-for-byte comparable to LiFT's own numbers:
- pc1_time_spearman
- distance_to_goal_spearman -- does raw distance to the clip's last frame shrink as the
  goal approaches (the distance-fidelity check from Temporal Straightening for Latent
  Planning)
- mean_velocity_cosine / mean_chord_alignment, each on clip_progress_metrics' own
  unit-sphere-normalized z by default (normalize=True), matching this script's own
  convention -- and their normalize=False counterparts (reported as
  mean_velocity_cosine_raw / mean_chord_alignment_raw), which operate on RAW z instead,
  matching what world_model_loss.py's straightening_loss/chord_loss actually see during
  training (the two are genuinely different quantities, not rescaled versions of each
  other -- the unit-sphere projection is a nonlinear transform on the step-vector
  geometry, not just a scale change)
- trajectory_straightness_raw, the same ratio as trajectory_straightness but on raw z
- norm_growth, norm_std_within_clip, adjacent_l2_distance -- raw-magnitude diagnostics
  invisible to every normalized/angle-only metric above, catching a model that encodes
  progress via vector length rather than direction
- norm_turning_point_fraction, direction_reversal_fraction -- oscillation diagnostics:
  how OFTEN magnitude/direction reverse trend, not just how much (norm_std_within_clip)
  or on average (mean_velocity_cosine_raw) -- distinguishes one smooth drift from many
  small back-and-forth reversals that could otherwise average out to a similar number
- effective_rank / effective_rank_fraction, dead_dim_fraction -- WHOLE-DATASET collapse
  diagnostics (not per-clip, not in per_clip/summary_median/summary_mean -- top-level
  keys in the output instead), checking whether DIFFERENT clips actually produce
  different representations, which nothing else in this file measures (every other
  metric only looks at movement within one clip's own trajectory). Rank-based, so needs
  M >> D to mean anything -- sound at the whole-dataset pooling level used here, NOT
  meaningful per-clip (T=16 frames can't support a rank estimate against D=128-384 dims).
- within_clip_mean_pairwise_l2 (per-clip) / mean_pairwise_l2_across_clips (WHOLE-DATASET,
  top-level key like effective_rank) -- the per-clip collapse check that doesn't need a
  rank estimate: all-pairs spread within one clip's own T frames, vs. the population
  scale (mean distance between DIFFERENT clips' representative vectors) it should be
  read against. A small within_clip value only means "collapsed" relative to the across-
  clips number: on its own it's an unscaled distance.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_common.loading import (encode_lift, encode_world_model, load_lift_model,  # noqa: E402
                            load_matched_features, load_world_model, reduce_raw)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "external" / "chirality_in_action" / "LiFT" / "visualization"))
from progress_metrics import clip_progress_metrics  # noqa: E402


def pc1_time_spearman(z, eps=1e-8):
    """Like clip_progress_metrics' own time_progress_spearman, but the 1-D axis frames
    get projected onto is PC1 (the direction of maximum variance across the clip's T
    frames, via SVD) instead of the fixed z_T - z_1 endpoint direction.

    time_progress_spearman is anchored to two frames (start, end) -- noisy if either one
    happens to be an outlier. PC1 is fit from all T frames at once, so it's the axis the
    trajectory actually moves along the most, which may or may not line up with the raw
    endpoint-to-endpoint direction.

    z: [T, latent_dim], one clip's RAW (not yet normalized) per-frame latents -- same
    convention as clip_progress_metrics' own `z` argument. Returns a float in [-1, 1], or
    None if T < 3 or the clip has ~zero variance in every direction (fully static, same
    degeneracy clip_progress_metrics itself guards against).
    """
    t = z.shape[0]
    if t < 3:
        return None
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)  # match clip_progress_metrics' normalization

    z_centered = z - z.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(z_centered, full_matrices=False)
    if s[0] < eps:
        return None
    proj = z_centered @ vt[0]  # [T], signed projection onto PC1

    # PC1's sign is arbitrary (SVD doesn't orient it) -- flip so that later frames project
    # higher on average, matching time_progress_spearman's "positive = forward progress"
    # convention rather than reporting an arbitrarily-signed correlation.
    if proj[-1] < proj[0]:
        proj = -proj

    time_idx = np.arange(t)
    return float(spearmanr(time_idx, proj).statistic)


def distance_to_goal_spearman(z):
    """Spearman correlation between each frame's RAW Euclidean distance to the clip's
    last frame, ‖z_t − z_T‖, and the number of frames remaining, T − t (t = 1..T−1).
    1.0 = distance to the goal shrinks steadily as the goal gets closer, i.e. latent
    distance is a usable "steps to go" signal for a planner that minimises distance to a
    goal latent; ~0 = distance says nothing about time left (e.g. i.i.d. per-frame noise).
    The distance-fidelity check of Temporal Straightening for Latent Planning, adapted
    from ground-truth geodesic distance to frames remaining. Raw z, since a planner uses
    raw Euclidean distance.

    z: [T, latent_dim], RAW per-frame latents. Returns a float in [-1, 1], or None if
    T < 3 or all distances are equal.
    """
    t = z.shape[0]
    if t < 3:
        return None
    dist = np.linalg.norm(z[:-1] - z[-1], axis=1)
    if np.ptp(dist) == 0:
        return None
    return float(spearmanr(np.arange(t - 1, 0, -1), dist).statistic)


def mean_velocity_cosine(z, step_thresh=1e-6, eps=1e-8, normalize=True):
    """Average cos(v_t, v_{t+1}) between every pair of CONSECUTIVE step/velocity vectors
    (v_t = z[t+1] - z[t]) -- "local straightness" in the sense of how consistently the
    trajectory's own direction of travel turns from one step to the next. This is a
    genuinely different quantity from clip_progress_metrics' own adjacent_cosine_distance,
    which compares consecutive FRAME vectors (z_t vs z_{t+1}) directly, not their
    difference -- e.g. two frames far apart on the unit sphere but moving in a locally
    straight line still average out that distinction. Same formula as
    predictive_world_model/losses/world_model_loss.py's straightening_loss (the training
    objective this whole project optimizes for), reported here as a raw metric (cosine
    similarity, 1.0=perfectly straight) rather than a loss (1 - cosine).

    normalize=True (default) matches clip_progress_metrics' own unit-sphere convention,
    for comparability with the other metrics in this file. normalize=False skips that
    projection, so the cosine is computed on the SAME (raw, unnormalized) vectors
    straightening_loss itself sees during training -- the unit-sphere projection is a
    nonlinear transform that changes step-vector geometry, not just their scale, so
    normalize=True and normalize=False are genuinely different quantities, not just
    rescaled versions of each other; normalize=False is the one that actually matches
    what straightening_loss optimizes.

    z: [T, latent_dim], RAW (not yet normalized) per-frame latents. Returns a float in
    [-1, 1], or None if T < 3 or every step pair has a near-zero-norm step (masked via
    step_thresh, matching straightening_loss's own guard against meaningless cosines).
    """
    t = z.shape[0]
    if t < 3:
        return None
    if normalize:
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)  # match clip_progress_metrics' normalization
    steps = z[1:] - z[:-1]  # [T-1, D]
    v1, v2 = steps[:-1], steps[1:]  # [T-2, D] consecutive step PAIRS
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid = (norm1 > step_thresh) & (norm2 > step_thresh)
    if not valid.any():
        return None
    cos_sim = np.sum(v1 * v2, axis=1) / (norm1 * norm2 + eps)
    return float(cos_sim[valid].mean())


def mean_chord_alignment(z, step_thresh=1e-6, eps=1e-8, normalize=True):
    """Average cos(v_t, chord) between EVERY step vector (v_t = z[t+1] - z[t], all T-1 of
    them) and the GLOBAL chord z[-1] - z[0] (start-to-goal, the same vector
    start_end_displacement's length comes from) -- "does each individual step point
    toward the eventual destination," a stricter, more direct notion than
    time_progress_spearman's rank correlation of the scalar PROJECTION onto that same
    chord (which only cares whether the projection is monotonic, not whether each step's
    full direction, including any sideways drift, actually points along the chord).

    normalize=True (default) matches clip_progress_metrics' own unit-sphere convention;
    normalize=False computes the same formula on raw z instead, matching
    world_model_loss.py's chord_loss (see mean_velocity_cosine's docstring for why the
    two are genuinely different quantities, not just rescaled versions of each other).

    z: [T, latent_dim], RAW (not yet normalized) per-frame latents. Returns a float in
    [-1, 1], or None if T < 2, the chord itself is ~zero (degenerate start==end, same
    case clip_progress_metrics guards its own projection metrics against), or every step
    has near-zero norm.
    """
    t = z.shape[0]
    if t < 2:
        return None
    if normalize:
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)  # match clip_progress_metrics' normalization
    chord = z[-1] - z[0]
    chord_norm = np.linalg.norm(chord)
    if chord_norm < step_thresh:
        return None
    steps = z[1:] - z[:-1]  # [T-1, D], every step
    step_norms = np.linalg.norm(steps, axis=1)
    valid = step_norms > step_thresh
    if not valid.any():
        return None
    cos_sim = (steps @ chord) / (step_norms * chord_norm + eps)
    return float(cos_sim[valid].mean())


def trajectory_straightness_raw(z, eps=1e-8):
    """Same ratio as clip_progress_metrics' own trajectory_straightness
    (start_end_displacement / path_length), but on RAW z -- no unit-sphere projection.
    Projecting onto the sphere first changes path_length and start_end_displacement by
    different (position-dependent) amounts, so this is not just a rescaled version of the
    normalized number; it answers "is the path straight in the space the model actually
    predicts in," which is what pred_loss/straighten_weight actually see.

    z: [T, latent_dim], RAW per-frame latents. Returns a float in [0, 1] (0.0 if the path
    has ~zero length, matching clip_progress_metrics' own degenerate-fallback
    convention), or None if T < 2.
    """
    t = z.shape[0]
    if t < 2:
        return None
    steps = z[1:] - z[:-1]
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    start_end_displacement = float(np.linalg.norm(z[-1] - z[0]))
    return start_end_displacement / path_length if path_length > eps else 0.0


def norm_growth(z, eps=1e-8):
    """‖z_T‖ / ‖z_1‖ -- does the trajectory's raw magnitude grow or shrink over the clip.
    Invisible to every metric above that normalizes first: a model could satisfy those
    perfectly while using vector length, rather than direction, to encode progress. >1
    means the latent grows over the clip, <1 means it shrinks, ~1 means magnitude is
    roughly constant (the case where normalizing first, as clip_progress_metrics does,
    loses the least information).

    z: [T, latent_dim], RAW per-frame latents. Returns a float, or None if T < 2 or the
    first frame's norm is ~zero.
    """
    t = z.shape[0]
    if t < 2:
        return None
    start_norm = float(np.linalg.norm(z[0]))
    if start_norm < eps:
        return None
    return float(np.linalg.norm(z[-1])) / start_norm


def norm_std_within_clip(z):
    """Std of ‖z_t‖ across a clip's T frames -- how much raw magnitude fluctuates
    frame-to-frame, independent of direction. A companion sanity check to norm_growth
    (which only looks at the two endpoints): a clip could have norm_growth~1 while still
    swinging in magnitude in the middle of the clip.

    z: [T, latent_dim], RAW per-frame latents. Returns a float, or None if T < 2.
    """
    if z.shape[0] < 2:
        return None
    return float(np.linalg.norm(z, axis=1).std())


def adjacent_l2_distance(z):
    """Mean raw ‖z_{t+1} - z_t‖ across a clip -- the scale-sensitive companion to
    clip_progress_metrics' own adjacent_cosine_distance (which is angle-only, on
    unit-normalized frames). This is the same quantity pred_loss is scored against at the
    single-step level (pred_loss uses L1/L2 on z_pred vs. z, not cosine), so it's directly
    interpretable in the training objective's own units.

    z: [T, latent_dim], RAW per-frame latents. Returns a float, or None if T < 2.
    """
    if z.shape[0] < 2:
        return None
    return float(np.linalg.norm(z[1:] - z[:-1], axis=1).mean())


def norm_turning_point_fraction(z, eps=1e-8):
    """Fraction of interior steps where ‖z_t‖'s trend reverses (grows-then-shrinks or
    shrinks-then-grows). 0.0 = magnitude moves monotonically the whole clip. Chance
    levels (verified on synthetic data): ~0.5 for a random walk (independent increments),
    ~0.667 for i.i.d. per-frame noise around a fixed point (2/3 of interior points of an
    i.i.d. sequence are local extrema). Unlike norm_std_within_clip (HOW MUCH ‖z_t‖
    varies), this measures how OFTEN it changes direction.

    z: [T, latent_dim], RAW per-frame latents. Returns a float in [0, 1], or None if
    T < 3. Steps with |Δ‖z‖| < eps are dropped rather than assigned a direction.
    """
    t = z.shape[0]
    if t < 3:
        return None
    dnorm = np.diff(np.linalg.norm(z, axis=1))
    sign = np.sign(dnorm[np.abs(dnorm) >= eps])
    if len(sign) < 2:
        return 0.0
    return float((sign[1:] != sign[:-1]).mean())


def direction_reversal_fraction(z, step_thresh=1e-6, eps=1e-8):
    """Fraction of consecutive step-vector pairs (v_t = z[t+1]-z[t], v_t+1 = z[t+2]-z[t+1])
    with cos(v_t, v_t+1) < 0 -- i.e. the trajectory's direction of travel turns by more
    than 90 degrees from one step to the next. mean_velocity_cosine_raw reports the
    AVERAGE of these cosines, which a mix of mild realignments and sharp reversals can
    both produce; this instead counts how often a genuine >90-degree reversal happens, so
    it distinguishes "consistently drifting off-course" from "actually zig-zagging back
    and forth" even when the two would average to a similar mean_velocity_cosine_raw.
    Chance levels (verified on synthetic data): ~0.5 for a random walk, ~1.0 for i.i.d.
    per-frame noise around a fixed point (consecutive steps x2-x1, x3-x2 share x2 with
    opposite sign, so their cosine concentrates at -0.5 in high dimensions).

    z: [T, latent_dim], RAW per-frame latents. Returns a float in [0, 1], or None if
    T < 3 or every step pair has a near-zero-norm step (same masking convention as
    mean_velocity_cosine).
    """
    t = z.shape[0]
    if t < 3:
        return None
    steps = z[1:] - z[:-1]
    v1, v2 = steps[:-1], steps[1:]
    norm1, norm2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    valid = (norm1 > step_thresh) & (norm2 > step_thresh)
    if not valid.any():
        return None
    cos_sim = np.sum(v1 * v2, axis=1) / (norm1 * norm2 + eps)
    return float((cos_sim[valid] < 0).mean())


def within_clip_mean_pairwise_l2(z):
    """Mean raw L2 distance over EVERY pair of frames within a clip (all T(T-1)/2 pairs,
    not just adjacent ones like adjacent_l2_distance) -- a per-clip collapse check that
    needs no dimensionality/rank estimate, unlike effective_rank, which is unreliable at
    T=16 samples (rank capped at T-1=15 regardless of whether the representation is
    actually healthy or collapsed -- a per-clip effective_rank couldn't tell the two
    apart). Catches a pattern adjacent_l2_distance alone can miss: a clip that jitters a
    lot step-to-step but keeps returning near the same spot over the whole clip (large
    adjacent steps, small overall spread) is a more degenerate pattern than genuine
    drift, and only shows up in the ALL-pairs distance, not the adjacent-only one.

    Read this against mean_pairwise_l2_across_clips (this file's whole-dataset
    counterpart, computed once in __main__) rather than in isolation -- a small absolute
    value here only means "collapsed" relative to how far apart frames from DIFFERENT
    clips typically sit; on its own it's just an unscaled distance number.

    z: [T, latent_dim], RAW per-frame latents. Returns a float, or None if T < 2.
    """
    t = z.shape[0]
    if t < 2:
        return None
    diffs = z[:, None, :] - z[None, :, :]  # (T, T, D)
    dists = np.linalg.norm(diffs, axis=-1)  # (T, T)
    iu = np.triu_indices(t, k=1)
    return float(dists[iu].mean())


# effective_rank/dead_dim_fraction are WHOLE-DATASET statistics, unlike every metric
# above: they need the full pooled (M, D) matrix (every frame of every clip in the eval
# set) at once, not a single clip's (T, D). Computed once in __main__ after the per-clip
# loop, not folded into per_clip.

def effective_rank(z, eps=1e-12):
    """RankMe (Garrido et al., "RankMe: Assessing the Downstream Performance of
    Pretrained Self-Supervised Representations by Their Rank," 2023): exp(entropy of the
    normalized singular-value distribution) of a (M, D) pooled embedding matrix -- the
    standard label-free collapse diagnostic in the VICReg/SSL literature, and the natural
    eval-time check for what vicreg_cov_weight is trying to enforce at train time
    (decorrelated dimensions -> singular values spread evenly -> high effective rank).
    Ranges from 1 (total collapse -- every row is effectively the same point, or the
    whole matrix lies along a single direction) to min(M, D) (every dimension
    contributes roughly equally, no collapse at all).

    z: (M, D), RAW (not yet normalized -- scale genuinely matters here; normalizing
    first would silently equalize away exactly the kind of per-dimension scale collapse
    this is meant to catch) per-frame latents, every frame of every clip in the eval set
    pooled together.
    Returns (effective_rank: float in [1, min(M,D)], effective_rank_fraction: float in
    [0, 1] = effective_rank / D -- normalized so it's comparable across encoders/
    checkpoints with different latent_dim).
    """
    z = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0, 0.0
    p = s / s.sum()
    rank = float(np.exp(-np.sum(p * np.log(p))))
    return rank, rank / z.shape[1]


def dead_dim_fraction(z, std_thresh=1.0):
    """Fraction of the D latent dimensions whose std across the WHOLE pooled eval set is
    below std_thresh. The default 1.0 matches world_model_loss.py's vicreg_std_loss hinge
    (relu(1 - std_x)), so this reads as "fraction of dims below the std VICReg's own
    regularizer targets". Absolute threshold, so it's scale-dependent: most meaningful for
    VICReg-trained checkpoints, where std~1 is the target; for other encoders read it with
    their typical per-dim std in mind.

    z: (M, D) pooled RAW per-frame latents (same convention as effective_rank).
    Returns a float in [0, 1].
    """
    return float((z.std(axis=0) < std_thresh).mean())


def mean_pairwise_l2_across_clips(z_all, max_pairs=200_000, seed=0):
    """Mean raw L2 distance between a random FRAME of one clip and a random FRAME of a
    DIFFERENT clip -- the yardstick within_clip_mean_pairwise_l2 is read against, and
    deliberately frame-to-frame so the two are like-for-like: within/across ~1.0 means a
    clip's own frames are as far apart as frames of unrelated clips (no clip identity),
    << 1 means frames cluster by clip. (Comparing against clip CENTROIDS instead would
    shrink the across-clip side by averaging out per-frame noise -- by sqrt(T)=4x for
    i.i.d. frames -- making within/across ~4 even with no clip structure at all.)

    Monte Carlo estimate over max_pairs random (clip_i, frame_a, clip_j, frame_b) draws
    (i != j, fixed seed) rather than all pairs, which would be O((N*T)^2).

    z_all: (N, T, D), RAW per-frame latents for every clip in the eval set.
    Returns a float.
    """
    n, t, _ = z_all.shape
    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    a_idx = rng.integers(0, t, size=len(i_idx))
    b_idx = rng.integers(0, t, size=len(i_idx))
    return float(np.linalg.norm(z_all[i_idx, a_idx] - z_all[j_idx, b_idx], axis=-1).mean())


def compute_metrics_dict(z_all, out_df=None):
    """Everything __main__ does with an already-encoded (N,T,D) z_all: per-clip metrics,
    summary_median/summary_mean, and whole-dataset collapse diagnostics. Pulled out as its
    own function so a caller that already has z_all in memory (e.g. a checkpoint's own
    encoder output, pooled some non-standard way) can get the full metrics dict without
    ever writing z_all to a --features file on disk. out_df, if given, must have the same
    row order as z_all and may carry 'chiral_triplet_id'/'verb' columns to tag per_clip
    entries with, same as the CLI path does via load_matched_features.
    Returns the same dict __main__ used to build directly into `result` (n_clips,
    effective_rank, effective_rank_fraction, dead_dim_fraction, mean_norm,
    mean_pairwise_l2_across_clips, summary_median, summary_mean, per_clip) -- everything
    except the checkpoint/features/manifest path metadata, which only make sense for a
    CLI invocation with real file paths behind them.
    """
    has_triplet = out_df is not None and "chiral_triplet_id" in out_df.columns
    has_verb = out_df is not None and "verb" in out_df.columns

    per_clip = []
    for i in range(len(z_all)):
        m = clip_progress_metrics(z_all[i])
        if m is None:
            continue
        pc1_spearman = pc1_time_spearman(z_all[i])
        m["pc1_time_spearman"] = pc1_spearman if pc1_spearman is not None else 0.0
        d2g = distance_to_goal_spearman(z_all[i])
        m["distance_to_goal_spearman"] = d2g if d2g is not None else 0.0
        vel_cos = mean_velocity_cosine(z_all[i])
        m["mean_velocity_cosine"] = vel_cos if vel_cos is not None else 0.0
        chord_align = mean_chord_alignment(z_all[i])
        m["mean_chord_alignment"] = chord_align if chord_align is not None else 0.0
        vel_cos_raw = mean_velocity_cosine(z_all[i], normalize=False)
        m["mean_velocity_cosine_raw"] = vel_cos_raw if vel_cos_raw is not None else 0.0
        chord_align_raw = mean_chord_alignment(z_all[i], normalize=False)
        m["mean_chord_alignment_raw"] = chord_align_raw if chord_align_raw is not None else 0.0
        straightness_raw = trajectory_straightness_raw(z_all[i])
        m["trajectory_straightness_raw"] = straightness_raw if straightness_raw is not None else 0.0
        n_growth = norm_growth(z_all[i])
        m["norm_growth"] = n_growth if n_growth is not None else 0.0
        m["norm_std_within_clip"] = norm_std_within_clip(z_all[i]) or 0.0
        m["adjacent_l2_distance"] = adjacent_l2_distance(z_all[i]) or 0.0
        turn_frac = norm_turning_point_fraction(z_all[i])
        m["norm_turning_point_fraction"] = turn_frac if turn_frac is not None else 0.0
        rev_frac = direction_reversal_fraction(z_all[i])
        m["direction_reversal_fraction"] = rev_frac if rev_frac is not None else 0.0
        m["within_clip_mean_pairwise_l2"] = within_clip_mean_pairwise_l2(z_all[i]) or 0.0
        if has_triplet:
            m["triplet_id"] = str(out_df.iloc[i]["chiral_triplet_id"])
        if has_verb:
            m["verb"] = str(out_df.iloc[i]["verb"])
        per_clip.append(m)
    print(f"Computed metrics for {len(per_clip)}/{len(z_all)} clips (degenerate ones skipped)")

    keys = [k for k in per_clip[0] if k not in ("triplet_id", "verb")]
    summary_median = {k: float(np.median([c[k] for c in per_clip])) for k in keys}
    summary_mean = {k: float(np.mean([c[k] for c in per_clip])) for k in keys}

    z_pooled = z_all.reshape(-1, z_all.shape[-1])
    eff_rank, eff_rank_frac = effective_rank(z_pooled)
    dead_frac = dead_dim_fraction(z_pooled)
    mean_norm_val = float(np.linalg.norm(z_pooled, axis=1).mean())
    print(f"effective_rank: {eff_rank:.2f} / {z_pooled.shape[1]} dims "
          f"({eff_rank_frac:.4f} fraction), dead_dim_fraction: {dead_frac:.4f}")
    print(f"mean_norm: {mean_norm_val:.4f}")

    pairwise_across = mean_pairwise_l2_across_clips(z_all)
    print(f"mean_pairwise_l2_across_clips: {pairwise_across:.4f}")

    return {
        "n_clips": len(per_clip),
        "effective_rank": eff_rank,
        "effective_rank_fraction": eff_rank_frac,
        "dead_dim_fraction": dead_frac,
        "mean_norm": mean_norm_val,
        "mean_pairwise_l2_across_clips": pairwise_across,
        "summary_median": summary_median,
        "summary_mean": summary_mean,
        "per_clip": per_clip,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="trained checkpoint (mutually exclusive with --raw): "
                                              "ProceduralWorldModel when given together with --config, or an "
                                              "external LiFT checkpoint (e.g. ggwirp95-...ckpt) when --config is "
                                              "omitted -- --features must be plain DINOv2 CLS features either way "
                                              "(LiFT's own DINOv2ForVideo backbone, same model_id, produces what "
                                              "compute_lift_embeddings expects directly, no separate extraction)")
    parser.add_argument("--config", help="config.yaml written by train.py alongside a ProceduralWorldModel "
                                          "--checkpoint -- omit for a LiFT --checkpoint instead")
    parser.add_argument("--raw", choices=["cls", "mean", "flatten"],
                         help="skip the trained model entirely and reduce --features directly: "
                              "'cls' uses an already (N,T,D) file untouched; 'mean'/'flatten' "
                              "mean-pool or concatenate the patch dim of an (N,T,P,D) file. "
                              "Mutually exclusive with --checkpoint/--config.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.raw is not None:
        if args.checkpoint or args.config:
            raise SystemExit("--raw is mutually exclusive with --checkpoint/--config")
    elif not args.checkpoint:
        raise SystemExit("must pass either --raw, --checkpoint + --config (ProceduralWorldModel), "
                          "or --checkpoint alone (a LiFT checkpoint)")

    device = torch.device(args.device)
    ckpt_epoch = None
    df = pd.read_csv(args.manifest, dtype=str)
    features, out_df = load_matched_features(args.features, df, args.path_col)
    if args.raw is not None:
        print(f"--raw={args.raw}: no model/checkpoint, reducing --features directly")
        z_all = reduce_raw(features, args.raw)
    elif args.config:
        # aggregate() is a no-op for cls/global and the learned agg_head for channel mode
        # (whose raw encoder output is a per-PATCH grid, not usable here).
        model, _, ckpt_epoch = load_world_model(args.checkpoint, args.config, device)
        z_all = encode_world_model(model, features, device, args.batch_size)
    else:  # --checkpoint without --config -> LiFT; scores its reconstructed per-frame trajectory
        z_all = encode_lift(load_lift_model(args.checkpoint, device), features, device, args.batch_size)

    metrics = compute_metrics_dict(z_all, out_df)

    result = {
        "checkpoint": args.checkpoint if args.raw is None else f"raw:{args.raw}",
        "checkpoint_epoch": ckpt_epoch,
        "features": args.features,
        "manifest": args.manifest,
        **metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}")
    print("\nSummary (median):")
    for k, v in result["summary_median"].items():
        print(f"  {k}: {v:.4f}")
