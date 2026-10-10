# Ordinal objectives

**Which training objective actually arranges a latent space ordinally?**

Trains one encoder, many times, on frozen per-frame features — changing *only the
objective* — and scores every result with the ordinality metrics
`../encoder_planning_eval/` already defines. Thirteen objectives from the progress
prediction, temporal self-supervision, video alignment and reward-learning literatures,
all selectable from a config file, plus the controls needed to read the table honestly.

Sibling to `../predictive_world_model/` (which trains one world model very well) and to
`../encoder_planning_eval/` (which scores encoders without training anything). This
project is the third leg: it *trains* encoders, but only in order to *compare
objectives*, and it deliberately reuses the other two rather than re-implementing them —
its encoder is imported from the first and its evaluation is run by the second, unmodified
(see `pwm_bridge.py`).

## What is held fixed

Everything except the objective, which is the entire design:

| | |
|---|---|
| Encoder | `predictive_world_model`'s `ClsMlpEncoder` — frozen DINOv2 ViT-S/14 CLS features → 2-layer MLP → `ClipRMSNorm`. **Imported, not copied.** |
| Data | SSv2, 16 frames evenly sampled per video, `ssv2_train` (12,216 clips) / `ssv2_validation` (1,430) |
| Optimizer | AdamW, `lr=1e-4` (encoder) / `5e-4` (head), `wd=1e-5`, `grad_clip=1.0`, 100 epochs, batch 128 |
| Evaluation | `../encoder_planning_eval/progress_metrics/compute_progress_metrics.py`, unchanged |
| Varies | the objective, and its own hyperparameters |

`norm_type: clip_rmsnorm` rather than LayerNorm is a deliberate project-wide choice: one
RMS scale per clip applied identically to every frame in it is the only option that
cannot distort within-clip straightness, and straightness is half of what is being
measured. See `../predictive_world_model/models/norms.py`.

## The objectives

Grouped by what they are actually supervised *by* — which is the axis the comparison is
really about, and the reason the table has controls in it.

### Absolute progress

| config | objective | paper |
|---|---|---|
| `progress_regression.yaml` | regress `t/(T-1)` per frame | ProgressNet, [Am I Done?](https://arxiv.org/abs/1705.01781) |
| `progress_regression_bo.yaml` | + Boundary Observant loss (L1 weighted up near phase boundaries) | same, Eqs. 2–3 |
| `ordinal_ce.yaml` | 16-way classification over progress bins | [Ordinal Entropy](https://arxiv.org/abs/2301.08915)'s premise |
| `ordinal_soft.yaml` | SORD-style unimodal soft labels | |
| `ordinal_coral.yaml` | CORAL: cumulative binary tasks, one shared projection | Cao et al. |
| `ordinal_corn.yaml` | CORN: same decomposition, conditional, unshared heads | Shi et al. |

`ordinal_ce` and `progress_regression` see **exactly the same information**, so any gap
between them isolates the loss form — which is the specific claim Ordinal Entropy makes.

### Relative order

| config | objective | paper |
|---|---|---|
| `pairwise_rank.yaml` | RankNet / margin loss on "which frame is later?" | learning-to-rank |
| `pairwise_rank_cross.yaml` | ... also across clips in the batch (one global progress axis) | |
| `shuffle_sort.yaml` | transformer predicts each shuffled frame's original rank | [Made to Order](https://arxiv.org/abs/2404.16828) |
| `shuffle_verify.yaml` | paper triplets: is `(b,middle,d)` temporally valid? (25% positive) | [Shuffle and Learn](https://arxiv.org/abs/1603.08561) |
| `shuffle_permutation.yaml` | classify which of 4! permutations was applied | OPN / VCOP |

`pairwise_rank` never sees an absolute progress value, only the sign of a difference — it
is the control for whether the absolute clock is doing any work at all.

The current `shuffle_verify.yaml` uses the paper's five-frame construction and three-frame
classifier. Existing `shuffle_verify_k8` and `ego_shuffle_verify` checkpoints and result
rows predate that correction: they used arbitrary permutations of eight frames and must
not be reported as Shuffle and Learn results. The corrected config writes to
`shuffle_verify_k3_paper` and needs a new training run.

### Temporal metric structure

| config | objective | paper |
|---|---|---|
| `rank_n_contrast.yaml` | contrast by *rank* of label distance to the anchor | [Rank-N-Contrast](https://arxiv.org/abs/2210.01189) |
| `tcn.yaml` | triplet: temporal neighbours near, distant frames far | [TCN](https://arxiv.org/abs/1704.06888) |
| `scl.yaml` | similarity profile must match a Gaussian prior over timestamp distance | [CARL](https://arxiv.org/abs/2203.14957) |
| `tcc.yaml` / `tcc_classification.yaml` | soft nearest neighbour in another video must cycle back | [TCC](https://arxiv.org/abs/1904.07846) |
| `vip.yaml` | goal-conditioned value: `-‖φ(o)−φ(g)‖` is an optimal value function | [VIP](https://arxiv.org/abs/2210.00030) |

Rank-N-Contrast is the one whose *optimum is proven* to order features by a continuous
label, which is verbatim what `time_progress_spearman` measures. A result where it does
not win is the most informative row in the table.

VIP is the one whose claim ("distance to the goal falls monotonically") is about metric
structure along the trajectory, which is what `../encoder_planning_eval/interpolation_planning/`
tests rather than what `progress_metrics` tests — expect its two evaluations to disagree.

### Deliberate floors

`arrow_of_time.yaml` (one bit per clip: forwards or backwards) and `pace.yaml` (classify
playback stride) are temporally supervised but not *ordinally* supervised. If they score
near the explicitly ordinal objectives, that is evidence about the **metric** being easy
to satisfy at least as much as about the objectives — worth establishing before reading
any other row.

### Controls

| config | what it isolates |
|---|---|
| `next_step.yaml` | `../predictive_world_model`'s objective, pure prediction, no regularizers |
| `next_step_pwm_recipe.yaml` | ... plus its curvature + VICReg recipe, re-run in this harness |
| `straighten_only.yaml` | the curvature regularizer *alone* (`objective: none`) |
| baseline (no training) | raw frozen DINOv2, added automatically by `make_eval_spec.py` |

`next_step`'s predictor and loss are **imported** from `predictive_world_model`, not
reimplemented — a control that is a reimplementation of the thing it controls for is not
a control.

## The crop confound

Read this before believing any absolute-progress row.

Every clip in this dataset is 16 frames spread evenly across the whole video, so frame
*i* is at progress *i*/15 **every single time**. An objective supervised on that target
can partly succeed by learning which of 16 slots a frame occupies, rather than anything
about progress. This does not affect the ranking/ordering objectives nearly as much, so
it would bias the comparison in a specific direction.

`crop_size` / `crop_label` are the control:

```bash
./train.sh configs/progress_regression.yaml \
    --checkpoint-dir checkpoints/pr_crop8_local --crop-size 8 --crop-label local
```

- `global` — a span starting at frame *s* keeps the video's clock, `(s+i)/(T-1)`. The
  same content always has the same label, so absolute progress must come from appearance.
- `local` — labels renormalized to [0,1] within the crop. Slot 0 is a different true
  progress value every example, so the slot layout carries no information at all.

`configs/progress_regression_crop.yaml` is the worked example. The same two flags apply
unchanged to every other objective.

## Usage

```bash
# one objective
./train.sh configs/rank_n_contrast.yaml

# a hyperparameter override of one
./train.sh configs/rank_n_contrast.yaml \
    --checkpoint-dir checkpoints/rnc_t0.5 --objective-kwargs '{"temperature": 0.5}'

# the whole comparison: 25 runs (23 configs + 2 RnC temperature brackets),
# round-robin over every visible GPU
python run_sweep.py --spec sweeps/full_comparison.json --workers 4

# a hyperparameter sweep
python run_sweep.py --spec sweeps/hyperparams.example.json --workers 2
```

Uses the `lift` conda env (`/home/sachish/.conda/envs/lift/bin/python`); `train.sh` calls
it by full path, since non-interactive PBS shells don't source the conda-init hook.

Outputs per run, under `--checkpoint-dir`: `config.yaml` (the fully resolved run),
`train_log.csv` (per-epoch losses **and** held-out ordinality metrics), `train.log`,
`best.ckpt` / `last.ckpt`.

### Scoring

```bash
python make_eval_spec.py                       # discovers checkpoints/, writes the spec
cd ../encoder_planning_eval/progress_metrics
/home/sachish/.conda/envs/lift/bin/python run_progress_metrics_batch.py \
    --spec ordinal_objectives.json \
    --csv-out results/ordinal_objectives_summary.csv --workers 4
```

Nothing here reimplements the evaluation. `train.py` writes checkpoints whose `"model"`
entry holds only `encoder.*` weights, which is exactly what `compute_progress_metrics.py`
loads once it drops `predictor.*` — the objective's own head lives under a separate
`"head"` key the evaluator never reads, so head capacity can never leak into the scored
model. `tests/test_objectives.py` asserts this contract against the real evaluation
script, so it fails loudly if that side ever changes.

The same checkpoints also drop straight into
`../encoder_planning_eval/interpolation_planning/` for the start→goal planning test,
which is a genuinely different question and the one VIP is aimed at.

## Hazards found and handled

These were found by auditing and measuring, not by reading; each is fixed at its source
and listed here because the fix encodes a fact about this setup that is easy to undo.

**TCC needs normalized latents.** Its two softmaxes are over `-‖·‖²/temperature`, and
this encoder's raw `‖z‖` is ~19 in 384 dimensions — squared distances in the hundreds,
so both softmaxes saturate to one-hot, `sigma² -> 0`, and the regression loss (which
divides by it) hit 6.7e4 with a gradient norm of 4.2e6 on epoch 0. `normalize: true`
(the default) bounds squared distances to [0, 4] and makes `temperature` mean something;
it also matches the evaluation, which L2-normalizes per frame before computing any metric.

**Crop randomness must not live in the dataset object.** A per-dataset
`random.Random(seed)` is cloned into every DataLoader worker and never reseeded, which
made crop offsets byte-identical in every epoch and made worker 1 mirror worker 0 —
`crop_size`, the entire confound control, was not random across epochs. `clip_data.py`
draws from the module-level `random`, which PyTorch reseeds per worker per epoch.

**SCL's two temporal views must be disjoint.** CARL's views may share a timestamp because
augmentation makes the two copies differ; with frozen features they would be bit-identical,
cosine similarity exactly 1, at the temporal distance the Gaussian prior weights most.
Two independent 8-subsets of 16 frames share 4 on average, so half the strongest targets
would be free. `disjoint: true` partitions the frames instead.

**`huber` was silently an alias for `l2`.** Targets and sigmoid predictions both live in
[0,1], so `|error| <= 1` always and `F.huber_loss`'s default `delta=1.0` never leaves the
quadratic regime — bitwise identical to `0.5*MSE`. `huber_delta` now defaults to 0.1.

**Rank-N-Contrast's tie handling needs a tolerance.** Label distances that are
mathematically equal differ in the last ulp in float32 (4 distinct values among the 15
gap-1 pairs of a 16-frame clip), so an exact `!=` split tie blocks that should be one and
dropped genuinely-tied samples out of the denominator. `tie_tol=1e-6` against a smallest
real gap of `1/(T-1) ≈ 0.067`.

**Rank-N-Contrast must not L2-normalize its features.** The reference applies
`F.normalize` only in its cosine branch and feeds *raw* distances to l1/l2; this file
originally normalized by default, which silently guts the objective at the paper's own
`temperature=2`. Measured on one real batch with this encoder:

| | similarity range | logit spread | loss (ceiling ln 2048 = 7.63) | summed encoder grad |
|---|---|---|---|---|
| `normalize: true` | [-1.49, 0] | 0.74 nats | 6.79 | 4.8 |
| `normalize: false` | [-32.3, 0] | 16.1 nats | 8.18 | 932 |

A 0.74-nat spread over 2048 candidates is a near-uniform softmax — the loss sits just
below the uniform ceiling and almost no gradient reaches the encoder. `normalize: false`
is now the default. Note the paper's `tau=2` is calibrated to *its* feature scale, not to
`clip_rmsnorm`'s `‖z‖≈19.5`, so `sweeps/full_comparison.json` brackets it at 0.5 and 8.

**Checkpoint selection must not peek at the reported split.** `make_eval_spec.py` scores
the table on `ssv2_validation`; selecting `best.ckpt` on validation ordinality would
choose every row by its own reported metric, and would flatter noisier objectives more
than stable ones. `monitor_source: train` (the default) keeps selection off that split.
Use `val` only if you then report `last.ckpt`.

**Head size is not, and cannot be, equal across objectives.** It is held fixed within a
family (every sequence head is `d_model=256, num_layers=2`) but ranges from 0 parameters
(`rank_n_contrast`, `scl`, `tcn`, `tcc`, `vip` — the loss applies to `z` directly)
through ~150K (`progress_regression`, `ordinal_*`, `pairwise_rank`) to ~1.68M
(`shuffle_*`, `arrow_of_time`, `pace`) and 3.55M for `next_step`'s predictor — against an
encoder of ~296K. A large head can drive its loss down with less help from the encoder,
which is exactly the failure this project measures, spread unevenly across the table.
Unfixable by construction; mitigated by the evaluation never seeing the head, and by
`head_params` being printed at startup and stored in every checkpoint. Read it alongside
each row, and use `--head-lr` if a large head needs slowing down.

## Open hazards to watch during the real runs

**VIP contracts.** Its first term pulls every frame toward the goal and only the
log-mean-exp term pushes back; after one epoch `start_end_displacement` had fallen to
0.13 and `adjacent_cosine_distance` to 0.003. A collapsed trajectory scores *well* on
smoothness metrics while being worthless, so `z_norm` and `ord_start_end_displacement`
in `train_log.csv` are the columns to watch. `aux_weights: {vicreg_std: 1.0,
vicreg_cov: 0.04}` is the fix if it persists, and composes with any objective here.

**The set-based objectives start at chance, and may stay there.** In the legacy runs,
`shuffle_verify_k8` and `arrow_of_time` sit at 0.50 accuracy after one epoch and
`shuffle_permutation` at
0.04 (= 1/24). That is not yet a problem, but it is the failure mode to watch for these
three specifically: the encoder is strictly per-frame, so all of their signal has to be
readable off a bag of independently-computed latents by the sequence head alone. If their
accuracy is still at chance after 100 epochs, those rows carry no information about the
objective — the encoder received no gradient worth the name — and should be reported as
"did not learn", not as "this objective does not produce ordinal structure".

## Checkpoint selection

A training loss is **not comparable across objectives** (VIP's is a log-mean-exp of
distances, ordinal classification's is nats, ranking's is a margin), so "best = lowest
loss" would silently mean something different in every row. Instead, `--eval-every`
(default 5) scores held-out full-length clips with LiFT's own `clip_progress_metrics` —
the same function the final table uses — and `best.ckpt` is selected on
`--monitor-metric` (default `time_progress_spearman`). Model selection and evaluation
therefore agree by construction. `--eval-every 0` reverts to loss-based selection.

The monitor always runs on **uncropped** clips regardless of `--crop-size`: the metric is
defined on a whole trajectory, and measuring it on random crops would confound "the space
got more ordinal" with "this epoch drew easier crops".

## Adding an objective

One file in `objectives/` implementing `forward(z, progress, frame_index, z_pair,
progress_pair, frame_index_pair) -> (loss, metrics)`, plus one line in
`objectives/__init__.py`'s `OBJECTIVE_REGISTRY`. No change to `train.py`, the config
schema, or the evaluation path. `train.py` validates `objective_kwargs` against the
class's own signature, so a config typo names the valid keys instead of being silently
ignored. Same registry convention as `../predictive_world_model/models/norms.py` and
`../encoder_planning_eval/interpolation_planning/predictors/`.

## Relationship to `../temporal_ordering/`

That project already asks a version of this question (raw vs. timestamp-regression vs.
shuffle), but on its own `.npz` cache, its own splits, and a linear-probe protocol rather
than these ordinality metrics — so its numbers are not comparable to anything in
`encoder_planning_eval`'s tables. Its two trained conditions are re-implemented here
(`progress_regression`, `shuffle_sort`) so they sit in the same table as the other
eleven. **Nothing in that folder is modified or imported.**

## Results

33 objectives/variants, 100 epochs each, SSv2 (`ssv2_train` 12,216 clips -> `ssv2_validation`
1,430), single seed, PBS jobs 121426 + 121456 (7h12m total). Scored by
`../encoder_planning_eval/progress_metrics/compute_progress_metrics.py`, unmodified.

### The headline

**On held-out clips, not one objective beat frozen DINOv2. On training clips, thirteen did.**

| | mean `time_progress_spearman` |
|---|---|
| frozen DINOv2, train split | 0.9119 |
| best objective (`scl`), train split | **0.9578** |
| frozen DINOv2, validation split | **0.9040** |
| best objective (`shuffle_sort`), validation split | 0.9056 (+0.0017 ± 0.0017, n.s.) |

So these objectives demonstrably *can* make a latent space more ordinal — they do it on the
clips they train on, by margins of up to +0.046 over the frozen features. None of it
transfers. The excess generalization gap (over the 0.0132 the untrained encoder shows
between the same two splits) tracks how much each gained on train: `scl` +0.060,
`shuffle_sort` +0.031, `tcn` +0.034.

`ordinal_entropy` is the extreme case at **+0.328** — 0.9007 on train, 0.5725 on validation.

### Validation, paired against the untrained encoder

Every run starts from a randomly-initialized `ClsMlpEncoder`, which scores 0.8902 — i.e.
the random projection *destroys* 0.0138 of the ordinality DINOv2 already had. That, not
raw DINOv2, is the zero point for "did training help".

| run | mean ± sem | Δ vs untrained init | | |
|---|---|---|---|---|
| `shuffle_sort_k8` | 0.9056 ± 0.0031 | +0.0154 | *** | |
| **frozen DINOv2 / `untrained_identity`** | **0.9040 ± 0.0031** | **+0.0138** | *** | ceiling |
| `ordinal_soft16` | 0.9028 ± 0.0030 | +0.0126 | *** | |
| `next_step_pwm_recipe` | 0.9026 ± 0.0032 | +0.0124 | *** | |
| `ordinal_ce16` | 0.9025 ± 0.0031 | +0.0123 | *** | |
| `rank_n_contrast_t0.5` | 0.9013 ± 0.0032 | +0.0111 | *** | **collapsed** |
| `vip_g098` | 0.9002 ± 0.0032 | +0.0101 | *** | **collapsed** |
| ... 10 rows indistinguishable from init ... | | | ns | |
| **`untrained_random_init`** | **0.8902 ± 0.0034** | (reference) | | zero point |
| `tcn_w1_m4` | 0.8799 ± 0.0038 | −0.0103 | *** | |
| `soft_dtw_lav` | 0.8786 ± 0.0037 | −0.0116 | *** | **collapsed** |
| `baseline_pwm_cls_mlp_k400` | 0.8523 ± 0.0042 | −0.0379 | *** | **collapsed** |
| `straighten_only` | 0.7514 ± 0.0067 | −0.1388 | *** | |
| `ordinal_entropy` | 0.5725 ± 0.0070 | −0.3177 | *** | |

Full tables: `results/val_paired_vs_init.csv`, `results/train_paired_vs_init.csv`.

### Three things that matter more than the ranking

**The reported median is quantized and cannot resolve these objectives.**
`time_progress_spearman` is a Spearman correlation over T=16 points, so it only takes
values `1 - k/680`; its median across clips lands on those same values. The 32-row summary
CSV collapsed onto 14 distinct medians with gaps of exactly 2/680 = 0.0029 — larger than
most of the real differences. `analyze_results.py` exists for this: per-clip means, paired
against a reference on the same clips, Wilcoxon + Holm.

**The metric is blind to collapse.** It L2-normalizes each frame and projects onto the
clip's own chord, so it is scale-invariant by construction. `rank_n_contrast_t0.5` and
`vip_g098` both score *significantly above* the untrained encoder while having contracted
the trajectory 10-20x (`start_end_displacement` 0.077 and 0.045 vs 0.80). Any ranking on
this metric alone is untrustworthy without reading `start_end_displacement` beside it.

**Eleven runs selected epoch 0.** `best.ckpt` is chosen on held-out ordinality, so the
selected epoch says whether training ever helped. Three groups: improved throughout
(`shuffle_sort` ep90, `ordinal_ce` ep80, `ordinal_corn` ep75, `ordinal_soft` ep70,
`progress_regression` ep50); peaked early then degraded (ep5-35: `coral`,
`rank_n_contrast`, `vip`, `scl`, `pairwise_rank`); and degraded from the first epoch
(`arrow_of_time`, `pace`, `progressor` x2, legacy `shuffle_verify_k8`, `soft_dtw`, `tcc` x2,
`next_step_l1`, `straighten_only`).

### Head capacity

The ablation (1.68M -> 25K params) does **not** support the "a big head absorbs the work
the encoder should do" hypothesis this project was built around:

| | full head | tiny head |
|---|---|---|
| `shuffle_sort` | +0.0154 *** | +0.0007 ns |
| `arrow_of_time` | +0.0042 ns | +0.0012 ns |
| `pace` | +0.0000 ns | +0.0024 ns |

Only `shuffle_sort` shows any head effect, and it runs the *opposite* way: the capable
head produced the more ordinal encoder. For `arrow_of_time` and `pace` head size is
irrelevant because neither learns anything at either size — consistent with their intended
role as floors.

### What this experiment cannot answer

Frozen DINOv2 sits at 0.904 of a metric whose ceiling is 1.0 and whose median resolution
is 0.003, and a 296K MLP over 12K clips has no headroom to add ordinality — only to lose
it, or to memorise the training split. The honest conclusion is not "these objectives do
not work" but **"this metric and this encoder cannot tell whether they work."**

Three ways to get an experiment that can: score on
`../encoder_planning_eval/interpolation_planning/` instead, where collapse is penalised
rather than invisible; use a dataset where DINOv2 is not already near ceiling; or give the
encoder enough capacity that fitting the objective is not trivial. The first is cheapest —
the checkpoints already drop into it unchanged.

## Not yet run

- **Seed replication.** Everything above is one seed. The large effects (`ordinal_entropy`,
  `straighten_only`, the collapses, the train/val gaps) are far beyond their standard
  errors and will hold; the ordering *within* the cluster near frozen DINOv2 will not be
  trustworthy until the top rows are rerun at seeds 1-2.
- **`interpolation_planning`.** Not run on any of these checkpoints. It is the evaluation
  that penalises collapse, and the one `vip` and `progressor` are actually aimed at.
- **Tier-3 leftovers.** OPEL / Gromov-Wasserstein OT and Drop-DTW are deliberately not
  implemented: their distinctive claim is handling order variation and background frames,
  which 16-frame single-action SSv2 clips give them nothing to exercise. They would be
  worth it on CrossTask or Ego4D-GoalStep.

### Questions this run did answer

- **Does `rank_n_contrast` win, as its theory says?** No. It collapses at low temperature
  (τ=0.5 and τ=2 both), and at τ=8 it is indistinguishable from the untrained encoder.
- **Does `ordinal_ce` beat `progress_regression`, per Ordinal Entropy's claim?** Marginally
  and not significantly: 0.9025 vs 0.9006 on validation. And adding the Ordinal Entropy
  regularizer itself is by far the *worst* result in the table (−0.3177), which is the
  opposite of what the claim predicts.
- **Do absolute-progress objectives survive cropping?** `progress_regression` 0.9006 →
  `progress_regression_crop8_local` 0.8942, dropping from significant to n.s. versus the
  untrained encoder. So yes, partly — the crop removes most of the gain, as designed.
- **Do `arrow_of_time` / `pace` sit close to the ordinal objectives?** Yes — both are
  statistically indistinguishable from the untrained encoder, and so are ten other
  objectives. That was the floor test, and much of the table failed to clear it.
