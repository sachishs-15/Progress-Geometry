# Progress metrics

## What these metrics are for

A procedural video should look like *progress* in latent space: as the clip goes on, the
per-frame latent `z_1 ... z_T` should move steadily from "start state" towards "end
state". It should not sit still, wander randomly, or bounce back and forth. The
straightening objective in `predictive_world_model/` (`straighten_weight`, `chord_weight`)
is trying to make trajectories behave like that. These metrics check whether they
actually do. They also check that the encoder didn't cheat, for example by collapsing
the representation so nothing moves, or by encoding progress in vector length instead of
direction.

The metrics fall into six groups, each answering one question:

| group | question |
|---|---|
| [Temporal ordering](#1-temporal-ordering) | Does the trajectory's position track time? |
| [Path shape](#2-path-shape--straightness) | Is the path straight, or does it curve and zig-zag? |
| [Step size](#3-step-size) | How far does the latent move per frame? |
| [Magnitude](#4-magnitude) | Is `‖z‖` changing, i.e. is progress hidden in vector length? |
| [Oscillation](#5-oscillation) | How often do magnitude and direction reverse? |
| [Collapse](#6-collapse) | Do different clips and frames still get different representations? |

Several of these metrics come from P-JEPA Appendix A.3. The original 7 are computed by
LiFT's vendored `clip_progress_metrics`, so they stay directly comparable to the LiFT
model family's numbers. Everything else is this project's own addition, in
`compute_progress_metrics.py`.

## Conventions

- **Input.** One `(T, D)` per-frame latent per clip (`T=16`). In checkpoint mode this is
  `model.aggregate(model.encoder(x))`, the same `z_agg` the straightening and VICReg
  losses see during training. In `--raw` mode it's the precomputed features themselves
  (`cls`, or patch tokens reduced by `mean`/`flatten`).
- **Normalized vs raw.** LiFT's 7 metrics, `pc1_time_spearman` and the plain
  `mean_velocity_cosine` / `mean_chord_alignment` first L2-normalize every frame onto the
  unit sphere. Metrics ending in `_raw`, plus every magnitude, oscillation and collapse
  metric, use unnormalized `z`. The raw versions are the ones that match what the
  training losses actually optimize (see [Path shape](#2-path-shape--straightness)).
- **Per-clip metrics** are computed once per clip, always as a mean over that clip's
  frames or steps. Across clips they're reported as both `summary_median` and
  `summary_mean` in the output JSON. Medians are the convention, since they're robust to
  outlier clips.
- **Whole-dataset metrics** (`effective_rank`, `effective_rank_fraction`,
  `dead_dim_fraction`, `mean_pairwise_l2_across_clips`) need every clip at once. They
  appear as top-level keys in the output JSON, not in `summary_*`.
- **Degenerate clips.** A clip is dropped entirely if LiFT's `clip_progress_metrics`
  returns `None` (`T < 3`, or first frame == last frame after normalization). Inside a
  kept clip, any of this project's metrics that returns `None` is stored as `0.0`.

### Reference values

These were checked on synthetic trajectories in `D=128`, `T=16`. Use them to tell whether
a number is good, bad or just noise.

| metric | perfect straight line | random walk | i.i.d. noise per frame | static clip |
|---|---|---|---|---|
| `time_progress_spearman` | 1.0 | ~1.0 | ~0.33 | dropped |
| `adjacent_temporal_nn_fraction` | 1.0 | ~1.0 | **0.125** (chance) | dropped |
| `distance_to_goal_spearman` | 1.0 | ~1.0 | **~0.0** | 0.0 |
| `trajectory_straightness_raw` | 1.0 | ~0.25 | ~0.07 | 0.0 |
| `mean_velocity_cosine_raw` | 1.0 | ~0.0 | **-0.5** | — |
| `direction_reversal_fraction` | 0.0 | ~0.5 | **~1.0** | — |
| `norm_turning_point_fraction` | 0.0 | ~0.5 | **~0.667** | 0.0 |
| `within / across` pairwise ratio | — | — | **1.0** (no clip identity) | 0.0 |

The i.i.d.-noise column is the most useful one for reading real encoders. A trajectory
made of a slow trend plus per-frame noise lands between the "straight line" and "i.i.d.
noise" columns. It sits closer to the noise column the more the noise dominates.

---

## 1. Temporal ordering

These ask whether the trajectory moves forward in time: are later frames further along
than earlier ones?

### `time_progress_spearman` (LiFT)
**Summary:** does position along the start→end direction increase with time? 1.0 means
perfect rank order.

**How it's computed:**
1. Normalize each frame onto the unit sphere.
2. Let `u = z_T − z_1` (the chord).
3. Give each frame a progress value `p_t = (z_t − z_1)·u / ‖u‖²`. This makes `p_1 = 0`
   and `p_T = 1`.
4. Report the Spearman rank correlation between `p_t` and the frame index `t`.

### `pc1_time_spearman` (project)
**Summary:** same as `time_progress_spearman`, but the progress axis is the clip's first
principal component instead of the start→end chord. It doesn't depend on two possibly
noisy endpoint frames.

**How it's computed:**
1. Normalize each frame, then center the clip's frames.
2. Take PC1 from an SVD and project every frame onto it.
3. Flip the sign so the last frame projects higher than the first.
4. Report the Spearman correlation of the projection with `t`.

### `monotonic_progress_fraction` (LiFT)
**Summary:** fraction of steps where progress `p_t` doesn't go backwards.

**How it's computed:** the mean of `(p_{t+1} − p_t ≥ 0)` over the `T−1` steps.

### `progress_efficiency` (LiFT)
**Summary:** penalizes back-and-forth motion along the progress axis. 1.0 means `p` goes
from 0 to 1 without ever retreating.

**How it's computed:** `1 / Σ_t |p_{t+1} − p_t|`. The total variation is at least 1
because `p` goes from 0 to 1, so this lies in (0, 1].

### `adjacent_temporal_nn_fraction` (LiFT)
**Summary:** for each frame, is its nearest neighbor in latent space an adjacent frame
(`t±1`)? This checks local temporal ordering without relying on any global axis.

**How it's computed:** on normalized frames, find each frame's most cosine-similar other
frame in the same clip. Report the fraction of frames whose nearest neighbor is at `t−1`
or `t+1`. Chance level is 0.125 when `T=16`.

### `distance_to_goal_spearman` (project)
**Summary:** does the latent distance to the clip's final frame shrink steadily as the
end gets closer? A planner steers by minimizing distance to a goal latent, so this checks
whether that distance is a usable "steps to go" signal. This is the distance-fidelity
check from *Temporal Straightening for Latent Planning*, which compares latent distance
with the true shortest-path distance. Here, frames remaining stands in for that distance.

**How it's computed:**
1. For each frame `t = 1 … T−1`, take the raw Euclidean distance `‖z_t − z_T‖` to the
   last frame.
2. Report the Spearman correlation between those distances and the frames remaining,
   `T − t`.

It uses raw `z` because a planner uses raw distance. A value of 1.0 means the distance
falls monotonically toward the goal. About 0 means distance says nothing about how much
time is left, which is what i.i.d. per-frame noise gives.

---

## 2. Path shape / straightness

These ask whether the trajectory takes a direct route or curves and zig-zags. Each one
comes in a normalized form (unit sphere, comparable to LiFT) and a `_raw` form
(unnormalized, matching the training losses). Projecting onto the sphere is nonlinear,
so the two forms are not rescaled versions of each other. For example, a perfectly
straight raw line scores `trajectory_straightness = 0.992` but
`trajectory_straightness_raw = 1.0`, because the line becomes an arc on the sphere.

### `trajectory_straightness` (LiFT) / `trajectory_straightness_raw` (project)
**Summary:** straight-line distance from start to end divided by the total path length.
1.0 means a perfectly direct path.

**How it's computed:** `‖z_T − z_1‖ / Σ_t ‖z_{t+1} − z_t‖`. It's 0.0 if the path length is
about zero.

### `mean_velocity_cosine` / `mean_velocity_cosine_raw` (project)
**Summary:** local straightness. Does each step point the same way as the step before
it? The `_raw` version equals `1 − straightening_loss`, the quantity that
`straighten_weight` optimizes.

**How it's computed:**
1. Take the steps `v_t = z_{t+1} − z_t`.
2. Report the mean of `cos(v_t, v_{t+1})` over consecutive step pairs.
3. Pairs where either step's norm is below `1e-6` are masked out, the same as in the loss.

### `mean_chord_alignment` / `mean_chord_alignment_raw` (project)
**Summary:** does each step point toward the clip's final state? The `_raw` version
equals `1 − chord_loss`.

**How it's computed:** the mean of `cos(v_t, z_T − z_1)` over all `T−1` steps. Unlike
`time_progress_spearman`, which only checks that the projection onto the chord is
ordered, this also penalizes sideways drift.

### `start_end_displacement` (LiFT)
**Summary:** how far the clip travels overall on the unit sphere. The maximum is 2.0.

**How it's computed:** `‖z_T − z_1‖` on normalized frames.

---

## 3. Step size

### `adjacent_cosine_distance` (LiFT)
**Summary:** how much the representation's direction changes per frame.

**How it's computed:** the mean of `1 − cos(z_t, z_{t+1})` over consecutive frames.

### `adjacent_l2_distance` (project)
**Summary:** raw per-frame step size, in the same units `pred_loss` is trained in. It
depends on scale, so compare it within one encoder or checkpoint family, not across
them.

**How it's computed:** the mean of `‖z_{t+1} − z_t‖` on raw `z`.

---

## 4. Magnitude

Every normalized metric is blind to `‖z‖`. These metrics check whether that blindness
hides anything. If magnitude barely changes, normalizing loses nothing.

### `norm_growth` (project)
**Summary:** does the vector get longer or shorter over the clip? A value near 1.0 means
the length is stable end to end.

**How it's computed:** `‖z_T‖ / ‖z_1‖`.

### `norm_std_within_clip` (project)
**Summary:** how much `‖z_t‖` wobbles within the clip, including the middle frames that
`norm_growth` doesn't look at.

**How it's computed:** the standard deviation of `‖z_t‖` over the clip's `T` frames.
Read it relative to the typical `‖z‖`. For example, raw DINOv2 CLS has `‖z‖ ≈ 23.5`, so
a within-clip std of 0.5 is about a 2% wobble.

---

## 5. Oscillation

These separate one smooth drift from many small back-and-forth reversals. The two can
produce similar averages in sections 2 and 4.

### `direction_reversal_fraction` (project)
**Summary:** fraction of consecutive steps where the direction of travel turns by more
than 90°.

**How it's computed:** the mean of `cos(v_t, v_{t+1}) < 0` over consecutive step pairs,
on raw `z`, using the same masking as `mean_velocity_cosine`. It's about 0.5 for a random
walk and about 1.0 for i.i.d. per-frame noise. That's because consecutive steps
`x_2 − x_1` and `x_3 − x_2` share `x_2` with opposite signs, so their cosine concentrates
at −0.5.

### `norm_turning_point_fraction` (project)
**Summary:** how often `‖z_t‖` switches between growing and shrinking.

**How it's computed:**
1. Take the signs of the consecutive differences of `‖z_t‖`.
2. Drop steps where the absolute difference is below `1e-8`.
3. Report the fraction of adjacent sign pairs that differ.

It's about 0.5 for a random walk and about 0.667 for i.i.d. noise, since 2/3 of the
interior points of an i.i.d. sequence are local extrema.

---

## 6. Collapse

Everything above looks at one clip's own trajectory. None of it can tell whether the
encoder has collapsed, for example by mapping everything into a few directions or
shrinking the whole space. A collapsed trajectory can look "straight" trivially: if
nothing moves, there's nothing to zig-zag. So read this section before trusting any
straightness gain.

Collapse comes in two forms, and each is caught by a different pair of metrics:
- **Rank collapse:** variance is squeezed into a few directions. `effective_rank`
  catches this, and is scale-invariant. `dead_dim_fraction` checks each dimension
  against VICReg's std = 1 target.
- **Scale and identity collapse:** everything shrinks, or clips stop being
  distinguishable from each other. The pairwise L2 pair catches this.

### `effective_rank` / `effective_rank_fraction` (project, whole-dataset)
**Summary:** how many dimensions the representation effectively uses, following RankMe
(Garrido et al., 2023). The value runs from 1 (a single direction) to `D` (all
dimensions used evenly).

**How it's computed:**
1. Pool every frame of every clip into one `(N·T, D)` matrix and center it.
2. Take the singular values `σ_k` and normalize them: `p_k = σ_k / Σσ`.
3. `effective_rank = exp(−Σ p_k log p_k)`.
4. `effective_rank_fraction = effective_rank / D`, which is the value to compare across
   different `D`.

It is scale-invariant, so it can't see uniform shrinkage. It needs many more samples than
dimensions (`M ≫ D`), which is why it's computed over the whole dataset and never per
clip. With `T=16` frames, a per-clip rank would be capped at 15 whether or not anything
had collapsed. One deviation from RankMe: the matrix is centered first. Otherwise the
mean direction dominates `σ_1`.

### `dead_dim_fraction` (project, whole-dataset)
**Summary:** fraction of dimensions whose spread across the dataset is below the std = 1
target that VICReg's regularizer enforces.

**How it's computed:** compute each dimension's standard deviation over the pooled
`(N·T, D)` matrix, then report the fraction of dimensions with std < 1.0. The threshold
is the same as `vicreg_std_loss`'s hinge, `relu(1 − std)`.

It uses an absolute threshold, so it depends on scale. It's most meaningful for
VICReg-trained checkpoints, where std ≈ 1 is the target. For other encoders, read it
against that encoder's typical per-dimension std. A dimension sitting right at the
target (std ≈ 1) will fall on either side of the threshold. The metric is also
axis-aligned, so rank collapse along rotated directions shows up in `effective_rank`
but not here.

### `within_clip_mean_pairwise_l2` (project, per-clip)
**Summary:** how spread out one clip's own frames are. This is the per-clip collapse
check, and it needs no rank estimate.

**How it's computed:** the mean raw `‖z_a − z_b‖` over all `T(T−1)/2` frame pairs in the
clip (120 pairs when `T=16`). Unlike `adjacent_l2_distance`, it also catches clips that
jitter step to step but keep coming back to the same spot. On its own it's an unscaled
distance, so read it against `mean_pairwise_l2_across_clips`.

### `mean_pairwise_l2_across_clips` (project, whole-dataset)
**Summary:** how far apart frames from *different* clips typically are. This is the
yardstick for the within-clip number.

**How it's computed:** the mean raw `‖z_{i,a} − z_{j,b}‖` over 200,000 random draws of
clip `i ≠ j` and frame indices `a`, `b`, with a fixed seed. This is a Monte Carlo
estimate, because computing every pair would be `O((N·T)²)`.

It is deliberately frame-to-frame, to match the within-clip metric. An earlier version
compared clip *centroids*, which averages away per-frame noise. That made the ratio about
4 (= √T) even when there was no clip structure at all.

**Reading the within/across ratio** (`summary_median.within_clip_mean_pairwise_l2 /
mean_pairwise_l2_across_clips`):
- **About 1.0:** a clip's frames are as far apart as frames of unrelated clips, so there
  is no clip identity.
- **Much less than 1:** frames cluster by clip, which is expected and healthy.
- **Near 0:** the clip is static relative to the dataset's spread.

For absolute scale collapse, compare `mean_pairwise_l2_across_clips` across checkpoints
of the same `D`.

---

## Known limitations

- **Purely radial motion is dropped, not flagged.** A clip that only changes `‖z‖`
  normalizes to a single point, so LiFT's metrics return `None` and the whole clip is
  skipped, including its `norm_growth`. That is exactly the case `norm_growth` exists to
  catch. It hasn't happened on any run so far: 1430/1430 SSv2 clips and 6732/6732
  Ego4D GoalStep clips were kept.
- **`None` becomes `0.0` inside a kept clip.** For `direction_reversal_fraction` this
  reads as "perfectly smooth". It affects 2–3 of 1430 SSv2 clips per run, which is
  negligible for the median.
- **Scale-dependent metrics aren't comparable across encoders.** `adjacent_l2_distance`,
  `norm_std_within_clip` and both pairwise L2 metrics depend on scale. Compare them within
  one checkpoint family, or use the within/across ratio.

## Usage

`compute_progress_metrics.py` scores one encoder or checkpoint at a time:
- `--raw cls/mean/flatten` reduces a precomputed features file directly, with no model.
- `--checkpoint` + `--config` scores a trained `ProceduralWorldModel` instead.

Full flags are in the script's docstring.

`run_progress_metrics_batch.py` runs a JSON spec of experiments as parallel subprocesses
and collects them into one CSV.

```bash
cd experiments/encoder_planning_eval/progress_metrics
/home/sachish/.conda/envs/lift/bin/python run_progress_metrics_batch.py \
    --spec progress_metrics_experiments.json \
    --csv-out results/progress_metrics_summary.csv --workers 4
```

Single checkpoint:
```bash
/home/sachish/.conda/envs/lift/bin/python compute_progress_metrics.py \
    --checkpoint ../../predictive_world_model/checkpoints/<run>/best.ckpt \
    --config ../../predictive_world_model/checkpoints/<run>/config.yaml \
    --features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt \
    --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \
    --out results/<run>.json
```
