# Encoder planning eval

Tests whether different pretrained visual encoders' latent spaces support "planning"
between a start frame and a goal frame — **without training on actions** — using
frozen, precomputed per-frame features only (the same `.pt` files
`predictive_world_model/` trains on: `{"features": (N,T,D), "video_paths": [str]*N}`).
The encoder is the only thing that varies across a comparison; everything else (task
definition, predictor architecture where one is used, metrics) is held fixed, so any
difference is attributable to the encoder's own latent geometry.

Sibling to `../predictive_world_model/` (which trains a world model on top of an
encoder) rather than part of it, because nothing here trains or modifies an encoder —
every analysis in this folder just asks "what does this encoder's frozen latent space
already look like."

## Testing types

| folder | question | needs |
|---|---|---|
| `progress_metrics/` | Is each clip's trajectory ordered in time and straight? Is progress hidden in vector length? Has the space collapsed? | val latents only |
| `representation_probes/` | Does the space still encode *which* action or step, and *which direction* it goes? Do same-action clips move alike? How much of DINOv2 survives? | labels + train split |
| `world_model_eval/` | How much better than "nothing changes" does the trained predictor predict, one step and several steps ahead? | a trained checkpoint's predictor |
| `interpolation_planning/` | Can intermediate latents be recovered from just start + goal? | val latents (+ a trained interpolator) |
| `metric_calibration/` | What are each metric's floor, ceiling and noise, and is a gain over a baseline real? | val latents; seed-replicate checkpoints for training noise |

Shared checkpoint and feature loading lives in `eval_common/loading.py`. It rebuilds a
`ProceduralWorldModel` from its `config.yaml`, memory-maps single or sharded feature
files, and encodes them to per-frame latents. It's named `eval_common` because LiFT's
`visualization/` folder already has a `common` module that `progress_metrics` imports.

### `progress_metrics/`
Does the encoder's per-frame latent trajectory `z_1...z_T` move through the clip in a
temporally ordered, roughly-straight way? It covers P-JEPA Appendix A.3-style ordinality
metrics, scored via LiFT's own `clip_progress_metrics` so the numbers are directly
comparable to the LiFT model family's. It also includes this project's additions:
raw-space straightness, distance-to-goal, magnitude, oscillation and collapse. See its
README.

### `representation_probes/`
Frozen-feature probes, fit on train and scored on val:
- k-NN and linear semantic probes (SSv2 verb, Ego4D GoalStep step family)
- SSv2 chirality, following Chirality in Action: one ridge classifier per chiral
  triplet on all 16 frame latents concatenated, plus whether reversing a clip flips its
  prediction
- cross-clip consistency: change-direction agreement, and TCC Kendall's tau on
  mean-centered clips
- reconstruction R² back to raw DINOv2 through an MLP decoder

See its README.

### `world_model_eval/`
Prediction skill of a trained checkpoint's predictor, `1 − model_error / copy_error`,
teacher-forced one step ahead and as an open-loop rollout. See its README.

### `interpolation_planning/`
Operationalizes "planning" directly: given only `z_1` (start) and `z_T` (goal) and a
target step index `t`, predict the intermediate latent `ẑ_t`, then compare it to the
true `z_t`. Two evaluation modes:

- **Interpolation (training-free)** — `ẑ_t = z_1 + α·(z_T - z_1)`, `α = (t-1)/(T-1)`.
  A direct test of whether the encoder's latent space is already "straight" between
  start and goal: if this alone reconstructs the true intermediates well, straightening
  is doing real work for planning in that space.
- **Trained predictor (best case)** — a small, fixed-architecture model learns to fill
  in every intermediate latent from `(z_1, z_T)` (`train_predictor.py`, frozen features
  only — nothing about the encoder itself is ever trained). Gives a best-case number and
  measures how much a learned model gains over simple interpolation. `--predictor-type`
  selects the architecture, registry-driven (`predictors/predictor.py`'s
  `PREDICTOR_REGISTRY`, so adding a new variant needs no other code changes):
    - `mlp` — a feedforward net applied independently to each intermediate position,
      conditioned on `(z_1, z_T, sinusoidal(t/T))`. No interaction between the positions
      being filled in.
    - `transformer` (default) — a bidirectional `TransformerEncoder` over the whole
      length-`T` sequence, with every intermediate position replaced by one shared
      learned mask token and refined via full (non-causal) self-attention — directly
      mirrors `external/chirality_in_action/LiFT/lift/model.py`'s
      `MaskedLatentPredictor` + `MaskedJEPAWorldModel` masking convention (P-JEPA,
      arXiv:2606.23256), specialized to this task's *fixed* mask (always exactly the
      intermediate positions, never LiFT's random ratio). Every filled-in position can
      attend to every other one (including other intermediate positions), which the MLP
      variant cannot do.

  Every variant shares the same interface (`forward(z1, zT, t_len) -> (B,T,D)`), so
  `eval_planning.py`/`run_planning_batch.py` work unchanged regardless of which type a
  checkpoint was trained with — the checkpoint itself records its own `predictor_type`
  and architecture kwargs.

The gap between the two tells you what kind of encoder you have: one where
interpolation already nails the intermediates is one where straightening does real work
for planning; one where the trained predictor is needed to close a large gap is one
where straightening alone isn't sufficient. `eval_planning.py` reports both directly
(`gap_closed_by_training` in its output when a predictor checkpoint is given).

All three metrics (`cosine_sim`, `relative_l2`, `r2`) are scale-invariant, so they stay
comparable across encoders with very different raw `‖z‖` (unlike raw L2 — see
`predictive_world_model/models/norms.py`'s discussion of how much that varies). Here
`r2` is the global vector R²

```text
1 - Σᵢ ‖z_trueᵢ - z_hatᵢ‖² / Σᵢ ‖z_trueᵢ - mean(z_true)‖².
```

It sums residual and total variation across latent dimensions before taking the ratio.
Thus dimensions are weighted by their target variance; this is not the uniform average
of independently computed per-dimension R² values.

## Current results

`interpolation_planning/results/planning_summary.csv` (interpolation-only sweep, SSv2
validation, run 2026-09-10 — no predictor trained yet for any encoder):

| encoder | cosine_sim (median) | relative_l2 (median) | R² |
|---|---|---|---|
| dinov2_vits14 | 0.878 | 0.483 | 0.657 |
| dinov2_large | 0.891 | 0.458 | 0.689 |
| dinov3 | 0.908 | 0.422 | 0.716 |
| vjepa2 | 0.963 | 0.270 | 0.650 |
| vjepa2.1 | 0.995 | 0.099 | 0.793 |

vjepa2.1's latent space is by far the straightest under this test — but note this is
the same encoder whose `progress_metrics` numbers showed unusually small adjacent-frame
cosine distance too (see `progress_metrics/results/progress_metrics_summary.csv`), which
is also consistent with a *less discriminative* (lower frame-to-frame variance) latent
space rather than a more *genuinely temporally-ordered* one — high interpolation
accuracy alone doesn't distinguish those two explanations. Worth checking against the
trained-predictor number (if training also can't beat interpolation by much, and
absolute error is low mostly because targets themselves are close together, that
favors the low-variance explanation) before reading this as "vjepa2.1 is the best
planning substrate."

## Not yet run

- No predictor (`mlp` or `transformer`) has been trained to convergence for any encoder
  yet — `results/planning_summary.csv` is `interp_only` mode for all five, so
  `gap_closed_by_training` isn't populated. Both `train_predictor.py --predictor-type
  {mlp,transformer}` paths are smoke-tested (a couple epochs each, dinov2_vits14) and
  work end-to-end, but neither has had a real (~100-epoch) run yet. `train_predictor.py`
  needs a GPU run per encoder (SSv2 train/validation split, same convention as the rest
  of this project); `planning_experiments.example.json`'s entries can each get a
  `predictor_checkpoint` key once trained, then `run_planning_batch.py` reruns to add
  the trained-predictor columns.
