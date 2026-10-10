# Predictive World Model (procedural video, no actions)

Generalizes `external/temporal-straightening`'s architecture (encoder -> predictor,
trained mainly on latent-space prediction error, with curvature/VICReg as auxiliary
regularizers) to procedural video with no robotic action to condition on — closely
related to LiFT's own `PredictiveWorldModel`
(`external/chirality_in_action/LiFT/lift/model.py`), but built as its own
self-contained project so it can grow independently: patch-token input, CNN
projectors, and a block-causal predictor, none of which LiFT's variant has.

## Architecture

- `models/encoder.py` — `ClsMlpEncoder` (`token_mode=cls`): DINOv2 CLS features (1
  token/frame already) -> optional MLP projector -> `latent_dim`, followed by a
  configurable norm (`--norm-type`: `layernorm` / `clip_rmsnorm` / `batchnorm` / `none`
  — see `models/norms.py` for the tradeoffs).
- `models/patch_encoder.py` / `models/projectors.py` — `PatchProjectorEncoder`
  (`token_mode=patch`): raw DINOv2 patch tokens through a CNN projector, `global`
  (pools to 1 token/frame) or `channel` (keeps the full patch grid, only crushes channel
  depth), ported from `external/temporal-straightening`.
- `models/pixel_encoder.py` — `SmallResnetPixelEncoder` (`token_mode=raw`): raw RGB
  frames (no DINOv2 at all) through a small CNN trained from scratch, jointly with
  everything downstream of it -- the only `token_mode` where the backbone itself has
  gradients. Everything after the CNN is identical to `token_mode=cls`: same
  `ENCODER_REGISTRY` (`mlp`/`residual_mlp`), same `CausalPredictor`. See
  `configs/cls_residual_mlp_raw.yaml`, the raw-pixel counterpart to
  `configs/cls_residual_mlp_paper_window0.yaml`.
- `models/predictor.py` — `CausalPredictor`: causal transformer, `z_pred[:, t]`
  predicts `z[:, t+1]`. Used for `token_mode=cls` and `token_mode=patch`
  (`projector_type=global`).
- `models/block_causal_predictor.py` — `BlockCausalPredictor`: same idea, but dense
  within a frame's own patches and causal across frames — used for `token_mode=patch`
  (`projector_type=channel`), which keeps more than 1 token/frame.
- `losses/world_model_loss.py` — `WorldModelLoss`: configurable L1/L2 prediction loss +
  optional curvature (`straighten_weight`) + optional VICReg std/cov regularizers.

Both `token_mode=cls` (`configs/cls_mlp*.yaml`) and `token_mode=patch`
(`configs/patch_global.yaml`, `configs/patch_channel.yaml`) are implemented and covered
by `tests/test_model.py`. `cls` is the mode that has actually been run on real data so
far (K400/SSv2, see `checkpoints/`); `patch` has no real-dataset runs yet — see "Not yet
run" below.

## Precomputed-or-live features

`data/feature_provider.py`'s `FeatureProvider` checks, per video, in order:
1. A bulk precomputed `.pt` (`--features`, same format as LiFT's
   `precompute_features.py`: `{"features": (N,T,D), "video_paths": [str]*N}`) — pure
   dict lookup, no encoder touched.
2. A per-video on-disk live-cache (`--live-cache-dir`, default
   `<checkpoint-dir>/live_cache`) of anything this provider computed itself in a
   previous run. Cache keys and stored metadata include frame count, encoder/checkpoint,
   token mode, and patch/raw image sizes, so changed extraction settings cannot reuse a
   stale entry.
3. An actual DINOv2 forward pass (lazy-loaded on first miss), written to tier 2 before
   being returned.

So `--features` is entirely optional: omit it to run purely on-the-fly (e.g. while
smoke-testing against a handful of videos with nothing precomputed), or point it at an
existing bulk file and only videos missing from it get computed lazily. This is the same
`vit_small_patch14_reg4_dinov2.lvd142m`, T=16-frame, CLS-only setup LiFT's own
`precompute_features.py` produces, so an existing LiFT features file for a dataset
(e.g. SSv2) drops in directly via `--features`.

## Usage

```bash
conda activate lift   # has decord/timm/DINOv2 deps

python train.py --config configs/cls_mlp.yaml \
    --train-manifest /path/to/ssv2_train.csv \
    --val-manifest /path/to/ssv2_val.csv \
    --features /path/to/ssv2_train_raw.pt \
    --checkpoint-dir checkpoints/ssv2_cls_mlp
```

Manifest: any CSV with a `video_path` column (`--path-col` to use a different name) —
same convention as LiFT's manifests.

Outputs under `--checkpoint-dir`: `config.yaml` (resolved args), `train_log.csv`
(per-epoch losses), `last.ckpt` / `best.ckpt`, `live_cache/` (on-the-fly features
computed during this run, reused on any future run pointed at the same
`--live-cache-dir`).

## Smoke test (no real dataset needed)

```bash
printf 'video_path\n%s\n%s\n' \
  external/chirality_in_action/LiFT/assets/folding_paper.mp4 \
  external/chirality_in_action/LiFT/assets/horsejump-high.mp4 > /tmp/smoke.csv

python train.py --config configs/cls_mlp.yaml --train-manifest /tmp/smoke.csv \
    --checkpoint-dir /tmp/smoke_ckpt --epochs 2 --batch-size 2 --device cpu
```

No `--features` passed — both videos are computed on-the-fly and cached under
`/tmp/smoke_ckpt/live_cache/`.

## Evaluation

Encoder/checkpoint evaluation now lives in `../encoder_planning_eval/` (a sibling
project — see its README), not here, since its real subject is comparing *encoders*
rather than training this world model:

- `../encoder_planning_eval/progress_metrics/compute_progress_metrics.py` — loads a
  trained `ProceduralWorldModel` checkpoint + its `config.yaml` (reaching back into this
  project's `models/`, see the sys.path insert at its top), runs the encoder over a
  precomputed features file, and reports P-JEPA Appendix A.3-style straightness/
  ordinality metrics via LiFT's own `clip_progress_metrics`. Pass `--raw {cls,mean,flatten}`
  instead of `--checkpoint`/`--config` to score an untrained baseline directly off a
  precomputed features file, with no `ProceduralWorldModel` involved at all.
- `../encoder_planning_eval/interpolation_planning/` — a second, independent test:
  whether an encoder supports predicting intermediate frames from just the start+goal
  latents (linear interpolation vs. a small trained predictor).

## Cluster usage

`train.sh` wraps `train.py` for non-interactive PBS shells (calls the `lift` conda env's
python by full path rather than relying on `conda activate`) and tees output to
`<checkpoint-dir>/train.log`. `train.pbs` / `train_sweep.pbs` are qsub templates built
on top of it — see the comments in each for node pinning / mount-wait rationale.

## Not yet run

- `token_mode=patch` (both `projector_type=global` and `channel`) is implemented and
  tested (see "Architecture" above) but has not been validated on a real dataset yet —
  no `checkpoints/` runs use it. SSv2 planned first, given patch-mode's much larger
  per-frame storage cost makes a small dataset the sensible starting point.
- `norm_type=clip_rmsnorm` full-scale (500-epoch/241k-clip K400) validation — matched or
  beat plain LayerNorm at a smaller ablation scale (`checkpoints/ablation_10pct_norm_cliprms`),
  full-scale run is `configs/cls_mlp_k400_cliprmsnorm.yaml`.
- `norm_type=batchnorm` — implemented (`configs/cls_mlp_k400_batchnorm.yaml`) but not
  yet benchmarked at any scale.
