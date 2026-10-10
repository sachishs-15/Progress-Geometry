# World-model evaluation

## What this is for

The world model's job is to predict the next latent from the ones before it. Raw
prediction error can't tell whether it does that job well. If training shrinks the
latent space so frames barely move, every prediction error gets small even though the
predictor has learned nothing. This folder scores the predictor *relative to* the
simplest possible guess, "nothing changes". Under exact collapse that baseline has zero
error, so relative skill is reported as undefined rather than as NaN or a misleading score.

## Prediction skill

**Summary:** how much better than "the next frame equals the current frame" the
predictor is.

`skill = 1 − model_error / copy_error`

| skill | meaning |
|---|---|
| 1.0 | perfect prediction |
| 0.0 | no better than assuming nothing changes |
| < 0 | worse than assuming nothing changes |

The copy baseline's error shrinks along with the latent space, so the ratio can't be
gamed by near-collapse. If its summed error is exactly zero, `skill` is `null`,
`degenerate_copy_baseline` is true, and the output reports valid/degenerate clip counts.

**How it's computed:**
- **Space and error measure.** Errors are measured where the predictor works: the
  encoder's output, which is per-patch tokens `(T, 196, 8)` for patch-channel
  checkpoints and per-frame vectors for cls checkpoints. They use the checkpoint's own
  training loss from `config.yaml` (`pred_loss`: `l2` = mean squared difference, `l1` =
  mean absolute difference), averaged over tokens and dimensions. So `model_error` is on
  the same scale as the `train_pred` column in `train_log.csv`.
- **One-step (teacher-forced).** For every `t`, predict `z_{t+1}` from the true
  `z_1 … z_t`. This is exactly what `pred_loss` trains. The copy baseline predicts
  `z_{t+1} = z_t`.
- **Open-loop rollout.** Give the predictor the true first `--context` frames (default
  4). Then repeatedly append its own prediction as the next input frame, until the clip
  ends (12 more frames when `T=16`). Horizon `h` compares the `h`-th predicted frame with
  the truth. The copy baseline holds the last true context frame fixed. This tests
  whether the model can imagine several steps ahead, which is what planning needs.
- **Reported numbers.**
  - `skill`: computed from errors summed over all clips.
  - `skill_per_clip_median`: the median of each clip's own skill.
  - Both raw errors, so the scale is visible.

**Verified on synthetic data:** a predictor that copies its input scores exactly 0 at
every horizon. A perfect predictor of constant-velocity motion scores exactly 1.

**Only for trained checkpoints.** Raw encoders have no predictor.

## Current results (SSv2 val, 1430 clips, errors are mean squared differences)

| | one-step skill | rollout h=1 | h=2 | h=4 | h=8 | h=12 |
|---|---|---|---|---|---|---|
| sw0.5 (`pwm_patch_channel_k400_10pct_vicreg_sw0.5_window0_batch16`) | 0.033 | 0.019 | 0.078 | 0.152 | 0.215 | 0.192 |
| sw2.0 (`pwm_patch_channel_k400_10pct_vicreg_sw2_window0_batch16`) | 0.055 | 0.047 | 0.084 | 0.151 | 0.216 | 0.199 |

One frame ahead, both predictors are only 3–6% better than copying the current frame.
Their measured one-step error (about 0.14) matches the training loss, so this is not a
measurement artifact. The advantage grows with horizon to about 20% at 8 frames ahead,
because the copy baseline drifts further from the truth over time while the model's
predictions stay closer.

## Usage

```bash
/home/sachish/.conda/envs/lift/bin/python compute_prediction_skill.py \
    --checkpoint ../../predictive_world_model/checkpoints/<run>/best.ckpt \
    --config ../../predictive_world_model/checkpoints/<run>/config.yaml \
    --features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt \
    --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \
    --out results/<run>_skill.json
```

Flags: `--context` (true frames before the rollout, default 4), `--batch-size`
(default 8), `--device`.
