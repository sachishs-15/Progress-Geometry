# Does temporal shuffling produce more ordinal procedural representations?

Tests whether training a representation to recover the order of temporally-shuffled
timesteps makes it more predictive of procedural progress than (a) doing nothing to the
raw representation, or (b) a plain per-frame timestamp-regression control. Full
motivation and spec: see the conversation that produced this experiment; the short
version is section headers below map 1:1 onto the three trained conditions.

## The three conditions

| | Model A — `baseline` | Model B — `timestamp` | Model C — `shuffle_k8` |
|---|---|---|---|
| Representation | raw feature `z_t` | frozen encoder trained to regress `t/(T-1)` per-frame (no cross-frame interaction) | frozen encoder trained (with a small Transformer) to recover the original order of K=8 shuffled frames |
| Probe | `nn.Linear(D, 1)` | same | same |
| Optimizer / epochs / batch size / split / progress labels | identical across all three |

Only the representation-learning objective differs; everything downstream (probe
architecture, optimizer, split, evaluation) is byte-for-byte identical, per the
critical-evaluation-protocol requirement — see `train/train_progress_probe.py`.

## Dataset-agnostic by design

Every dataset (CrossTask, the synthetic smoke-test set below, or a future V-JEPA feature
cache) is converted by a `scripts/prepare_*.py` script into **one unified on-disk cache
format** (`data/cache.py`): a `manifest.json` + one `.npz` per video. Every downstream
module — splitting, training, evaluation — only ever talks to `CachedFeatureDataset`.
**Swapping datasets is a one-line change**: `dataset.cache_dir` in a config YAML. No code
changes needed.

## Status: synthetic smoke test, not the real result yet

Per instruction, CrossTask has **not been downloaded**. `scripts/prepare_crosstask.py` is
fully implemented against the verified official file format (see its docstring — format
was checked directly against `DmZhukov/CrossTask`'s `data.py`/`args.py` on GitHub, not by
downloading the 30GB+ dataset) and is ready to run, but hasn't been.

What HAS been run end-to-end, right now, in this repo: all three conditions above trained
and evaluated on a **synthetic** dataset (`scripts/prepare_synthetic.py`) that mimics
CrossTask's shape exactly (3200-D/sec features, per-video step boundaries) but is
generated from random smooth per-task trajectories, not real instructional videos. This
shows that the end-to-end mechanics run, but the timestamp condition in that historical
run used the incorrect targets described below. It is **not** a test of the actual
research hypothesis. The real answer requires retraining the corrected controls and then
running the same protocol after `prepare_crosstask.py`.

The table below is retained as a historical smoke-test artifact. Its `timestamp` row was
generated before sampled source indices were preserved, so that run learned rank within a
random K-frame subset rather than the documented absolute target `idx/(T-1)`. Retrain the
timestamp representation, its probe, and the combined evaluation before using that row in
any comparison. The baseline and shuffle objectives were not affected by that target bug.

Historical results from `results/results_table.csv`:

| Representation | test MAE ↓ | test RMSE ↓ | test R² ↑ | test Spearman ↑ | test Pairwise Order Acc ↑ | test Distance–Progress Spearman ↑ |
|---|---:|---:|---:|---:|---:|---:|
| baseline (raw) | 0.082 | 0.103 | 0.876 | 0.949 | 0.907 | 0.039 |
| timestamp (legacy rank-target run; rerun required) | 0.012 | 0.018 | 0.996 | 0.998 | 0.988 | 0.112 |
| shuffle_k8 | 0.015 | 0.020 | 0.995 | 0.998 | 0.986 | **0.966** |

`test_distance_progress_spearman` is the primary research metric, with
`test_pairwise_ordering_accuracy` and `test_spearman` as supporting ordinality checks.
MAE/RMSE/R² measure metric accuracy rather than ordinality. In this historical table the
shuffle-to-baseline comparison remains informative as a smoke test, but no conclusion may
use the legacy timestamp row. The synthetic generator's progress signal is smooth by
construction, so the corrected experiment must still be rerun on real data.

## Setup

Uses the existing `frozen-encoders` conda env (`/home/sachish/.conda/envs/frozen-encoders`),
which already has torch/numpy/scipy/scikit-learn/pandas/matplotlib/pyyaml. All commands
below assume `cd experiments/temporal_ordering` and that env's python on PATH (or call it
by full path as shown).

## Reproduce the synthetic smoke test (no download required)

```bash
python scripts/prepare_synthetic.py --config configs/synthetic_data.yaml

python scripts/train_baseline.py  --config configs/baseline.yaml
python scripts/train_timestamp.py --config configs/timestamp.yaml
python scripts/train_shuffle.py   --config configs/shuffle_k8.yaml

python scripts/evaluate.py --experiments baseline timestamp shuffle_k8
```

Outputs: `runs/<name>/{config.yaml,metrics.json,checkpoints}`, `splits/synthetic/*.txt`,
`results/results_table.{csv,json}`, `plots/{pca,distance_progress}_<name>.png`.

## Run the real experiment on CrossTask

```bash
# Downloads crosstask_release.zip + crosstask_features.zip (~30GB) and converts the
# 18 primary tasks into the unified cache. NOT run yet — do this when ready to spend the
# disk/bandwidth budget. --raw-dir can point at an already-extracted copy instead.
python scripts/prepare_crosstask.py --out data_cache/crosstask

# Point every config at the real cache instead of the synthetic one (either edit
# dataset.cache_dir: data_cache/crosstask in each configs/*.yaml, or copy the configs/
# dir and change just that one field — same three commands otherwise):
python scripts/train_baseline.py  --config configs/baseline.yaml
python scripts/train_timestamp.py --config configs/timestamp.yaml
python scripts/train_shuffle.py   --config configs/shuffle_k8.yaml
python scripts/evaluate.py --experiments baseline timestamp shuffle_k8
```

Delete `splits/<dataset_name>/` first if you want splits regenerated (they're cached and
reused automatically otherwise, which is the point — every experiment on a given dataset
uses the exact same train/val/test video IDs).

## Alternative dataset: Something-Something V2 (short clips)

CrossTask's videos are full-length instructional videos (several minutes); if you want a
dataset made of genuinely **short** clips instead (2-6s each, one templated action per
clip — e.g. "Pushing something from left to right"), `scripts/prepare_ssv2.py` converts
Something-Something V2 into the same unified cache. Unlike CrossTask, SSv2 ships raw
video, not pre-extracted features, so this script reuses `experiments/frozen_encoders`'s
existing DINOv2/DINOv3/CLIP/SigLIP/V-JEPA2 plugins to encode sampled frames itself —
`feature_dim` in the resulting manifest comes from whichever encoder you pick, not a
hardcoded 3200, so nothing downstream needs to change.

**Not run yet** — SSv2 requires a registered account on Qualcomm's AI-datasets portal
(no scriptable public URL like CrossTask), so getting the files onto disk is a manual
step. There are also no procedural sub-step annotations for SSv2 (one action per clip),
so only `dataset.progress_type: timestamp` is valid for it, not `step`.

```bash
# After manually downloading + extracting SSv2 to, say, data_cache/_ssv2_raw/
# (see the script's docstring for the exact expected file layout):
python scripts/prepare_ssv2.py --config configs/ssv2_data.yaml

# Same three commands as always, pointed at the SSv2 cache (edit dataset.cache_dir:
# data_cache/ssv2 in configs/*.yaml, or copy the configs/ dir):
python scripts/train_baseline.py  --config configs/baseline.yaml
python scripts/train_timestamp.py --config configs/timestamp.yaml
python scripts/train_shuffle.py   --config configs/shuffle_k8.yaml
python scripts/evaluate.py --experiments baseline timestamp shuffle_k8
```

`configs/ssv2_data.yaml` defaults to 20 clips/template (SSv2 has ~220k clips total across
~174 templates) — raise `max_videos_per_template` once the pipeline's confirmed working
on a subset.

## Ablations (section 18 of the spec)

```bash
# K sweep — 4 / 8 / 16 shuffled frames
python scripts/train_shuffle.py --config configs/shuffle_k4.yaml
python scripts/train_shuffle.py --config configs/shuffle_k16.yaml

# mild vs strong shuffle (shuffle_strength in the config; implemented as one continuous
# noisy-key-sigma knob — see losses/ordering_loss.py::shuffle_permutation)
python scripts/train_shuffle.py --config configs/shuffle_k8_mild.yaml

# training-data-size sweep — add `train_frac: 0.25` (etc.) to any config, or:
python -c "
import sys; sys.path.insert(0, '.')
from utils.config import load_config
import yaml
cfg = load_config('configs/shuffle_k8.yaml')
cfg['experiment_name'] = 'shuffle_k8_frac25'
cfg['train_frac'] = 0.25
yaml.safe_dump(cfg, open('configs/shuffle_k8_frac25.yaml', 'w'))
"
python scripts/train_shuffle.py --config configs/shuffle_k8_frac25.yaml
```

Then add the new experiment name(s) to the `evaluate.py --experiments` list.

## Step-level progress (section 16)

Set `dataset.progress_type: step` in a config instead of `timestamp` — every video in
the cache already carries per-second `step_labels` (both `prepare_synthetic.py` and
`prepare_crosstask.py` write them; CrossTask primary-task videos have them from the
official annotations). No other code changes needed — `data/progress.py` handles both
definitions behind the same interface.

## Notable implementation details / fixes made while validating the pipeline

- **Probe input standardization is required, not optional.** Raw CrossTask-scale features
  (3200-D, unnormalized) blow up plain `nn.Linear` training; `ProgressProbe` applies a
  fixed (non-trainable) per-dimension standardization computed from the training split
  before its one trainable `nn.Linear(D, 1)` — standard practice for any frozen-feature
  linear probe, not a change to the model's capacity. Critically, the standardization
  **floors the std** (like scikit-learn's `StandardScaler` does for zero-variance
  columns) rather than just adding a small epsilon: a training-set dimension that's ~0 for
  every training example (e.g. a dead ReLU unit) but not exactly 0 on a held-out video
  turns into a 10^6x noise amplifier under naive `std + 1e-6` division. This was caught
  by the `timestamp` condition's test RMSE exploding to ~99 despite clean training curves
  — see `models/progress_probe.py`.
- **Best-val-checkpoint selection for the probe.** With a fixed epoch budget and a highly
  progress-predictive representation (expected for `timestamp`/`shuffle`, whose encoders
  are directly trained around temporal position), an unregularized linear probe can drive
  train loss to ~0 within a handful of epochs and then overfit further in the remaining
  ones. `train_progress_probe.py` tracks val MSE every epoch and restores the best
  checkpoint — applied identically to all three conditions, so it strengthens rather than
  weakens the controlled comparison.
- **Probe learning rate matters a lot at D=3200.** `lr=0.01` (a reasonable default for the
  512-D encoder representations) oscillates and fails to converge on raw 3200-D input;
  `lr=0.001` is used everywhere for comparability.
