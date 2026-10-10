# Mistake-injection evaluation

Does a world model's latent path, or its predictor, flag clips that have been deliberately
broken, and does it do so better than the raw features it started from? The clips come from
datasets with no real mistakes (K400, SSv2), so the mistakes are injected: reorder frames,
cut a block out, splice in another clip, stall, change pace.

This tests whether the geometry and the one-step predictor react to the kinds of detour
that procedural mistakes would produce. It does not measure performance on real human
errors, which needs a procedural dataset.

## Readouts

| readout | latent path used for the geometry scores | predictor scores |
|---|---|---|
| `raw` | input features, mean-pooled over the 196 patches | none |
| `<name>` | encoder tokens (196 patches), mean-pooled over patches | one-step error in the predictor's own token space, using the checkpoint's `pred_loss` |

## Injections

Frame selection and reordering of precomputed features are equivalent to changing the
frames before a frame-wise encoder. Feature-space `slow_down` and `appearance` are
approximations; RGB-input runs apply their changes to pixels before encoding. Sequences
keep all 16 frames except `drop` and `speed_up`, which remove or
skip frames and give 12; each injection is compared with a clean version of the same length.

| injection | mimics | what changes |
|---|---|---|
| `reverse` | ordering mistake, correction | the clip played backwards |
| `swap` | ordering mistake | two 4-frame blocks exchanged |
| `shuffle` | ordering mistake | the four 4-frame blocks permuted |
| `drop` | omission | 4 consecutive frames removed (12 frames) |
| `splice_other` | addition, wrong action | a 4-frame block replaced by a clip from another class |
| `splice_same` | addition, hard case | the same, from a clip of the same class |
| `freeze` | stall | one frame held for 4 frames, then a jump |
| `speed_up` | timing | pace doubles after frame 8 (12 frames) |
| `slow_down` | timing | pace halves after frame 8, using interpolated features |
| `appearance` | negative control | every frame shifted by a constant feature offset |
| `null_split` | calibration | two random halves of the clean clips; should be 0.5 |

`appearance` is a feature-space stand-in for a colour or camera change. Real blur or colour
jitter would need the perturbed frames re-encoded through DINOv2.

## Scores

Higher always means more anomalous, so an AUROC under 0.5 means the perturbed clip looked more
normal than the clean one.

- Geometry (work on any latent path, raw features included): `jump_max`, `jump_spike`
  (largest step, and relative to the median step), `cv_max`, `cv_mean` (constant-velocity
  extrapolation error), `chord_dev` (largest distance from the start-to-end line),
  `backslide` (progress falling below its running maximum), `step_cv`, `stall`.
- Predictor (checkpoints only): `pe_max`, `pe_mean`, `pe_spike`.

The headline table uses one predictor score and one geometry score per injection, chosen in
advance (`HEADLINE` in `run_injection_eval.py`). The full CSV has every score.

## Metrics

- AUROC of perturbed vs clean clips, with a 95% bootstrap CI over clips; paired win rate.
- Seam localization: how often the largest per-transition value lands within one step of the
  seam, next to the rate expected by chance.
- Predictor skill on clean clips, `1 - model error / copy-previous-frame error`. When this is
  at or below 0 the predictor's error carries little information, whatever the AUROC says.

## Running

```bash
conda activate lift
python run_injection_eval.py                      # raw, vicreg3 best + last, K400 and SSv2, 500 clips each
python run_injection_eval.py --model name=RUN_DIR@epoch29.ckpt --dataset ssv2 --n-clips 200
python tests/test_injections.py && python tests/test_scores.py && python tests/test_readout.py && python tests/test_combine.py
python tests/test_egoper_style.py
```

K400 clips are drawn from the 10% training subset the checkpoints were trained on, stratified
over 50 classes; SSv2 clips come from the validation split, stratified over its 22 verb classes.
Outputs go to `results/<name>/`: `summary.md`, `<dataset>_auroc.csv`,
`<dataset>_localization.csv`, `scores.npz`, `injection_eval.png`, `config.json`.

`combine_results.py` requires matching sampling seed and clip count across runs, and
rejects duplicate readout names for the same dataset. Older result configs do not record
which feature file or pixel cache was used, so verify their data source before merging.

## EgoPER-style evaluation on synthetic clips

`run_egoper_style_eval.py` splits the sampled source clips by class into clean-only
calibration (about 20%) and held-out test (about 80%) sets before injecting errors.
SSv2 is the default because the VICREG3 checkpoint used here was trained on the K400
10% feature subset; K400 results from this checkpoint are training-exposed.
The model is not retrained. For each preselected headline score and injection, it uses
the 95th percentile of clean calibration scores as the test threshold. The test set
pairs each edited clip with its same-length clean reference; segment F1 therefore
reflects 50% mistake prevalence. The report also gives threshold-free clip AUROC,
framewise AUROC and average precision for splice/freeze, and seam localization.
The segment table has separate geometry and predictor rows for model readouts.
`appearance` and `blur` are benign controls; their threshold crossings are reported
as control flag rates rather than mistake recall.

```bash
python run_egoper_style_eval.py --dataset ssv2 --n-clips 500 \
    --out-dir results/egoper_style_ssv2
python run_egoper_style_eval.py --rescore results/egoper_style_ssv2
```

For patch models, `--geometry-pooling concat` flattens encoder patch tokens in
patch order before computing geometry scores; the default `mean` averages patches.
This does not change predictor error. Use `--skip-predictor` for a faster geometry-only
run when comparing pooling methods, and pin both runs to the same checkpoint file.

Outputs are `segment_metrics.csv`, `frame_metrics.csv`, `localization.csv`,
`per_clip_scores.npz`, `test_seams.json`, `protocol.json`, and `summary.md`.
Framewise scores assign transition `t` to its target frame `t+1`. For held frames,
`negative_step` measures the stall and positive step size measures the resume boundary.
The data have synthetic edits and no procedural step labels, so EgoPER omission metrics
are undefined and these results are not directly comparable with EgoPER benchmark scores.
