# Metric calibration for `encoder_planning_eval`

*Floors, ceilings and noise for every metric in the suite, and a protocol for saying an improvement is real.*
Run 2026-10-09. Unless stated, numbers are **SSv2 validation (1,430 clips, T = 16)** with **DINOv2-S CLS** as the reference
encoder. Everything here is reproducible from this directory (see §13).

---

## 0. Bottom line

1. **Noise is not your problem; validity is.** For the CLS-family models the largest noise source (clip sampling) moves a
   trajectory metric by 0.1–1% of its range; training-seed sd is smaller still (≤ 0.0015 for the regularised configs). Meanwhile *free* operations on
   the same raw features (PCA whitening: 1–4% of range; a 1-frame Gaussian smoothing: 9–92%) move the same metrics much further. A "significant"
   improvement is easy to get and tells you little until it is compared with those free baselines.
2. **The trajectory metrics reward smoothness, not progress.** Time-reversing a clip leaves nine of the ten ordering/straightness metrics *exactly* unchanged
   (only `distance_to_goal` moves, by ≤ 0.05); a random walk built from the clip's own step vectors outscores real DINOv2 on every one of them; a cumulative mean
   (carries no new information) scores ≈ 1.0; smoothing the raw features with σ = 1 frame beats every trained model you have.
3. **Whitening explains most of what VICReg does.** A closed-form PCA whitening of the raw features closely reproduces the
   VICReg(sw 0.1) change on time_progress, straightness, velocity-cosine, direction-reversal and the linear probe, and has the same sign and similar size on
   planning R² (−0.11 whitened vs −0.18 trained) and effective rank (+0.13 vs +0.09). An *untrained* copy of the architecture reproduces raw DINOv2. So "trained − raw" is the training effect, and "trained − whitened raw" is the part
   that is not just spectrum flattening.
4. **The ceilings in the README are not reachable.** Prediction skill 1.0 → noise ceiling ≈ 0.60 (and a 3-tap linear model gets 0.12
   while the trained predictors get 0.05–0.10). Interpolation R² 1.0 → a learned ridge/kNN predictor adds at most +0.01–0.03
   over plain interpolation. In my synthetic sweep, frozen encoders do not exceed straightness ≈ 0.20 (DINOv2) / 0.34 (V-JEPA 2.1) even on perfectly clean
   constant-velocity motion with a large object.
5. **Interpolation R² is 92% "which clip is this".** A static midpoint (no time at all) gets R² = 0.604 of the reported 0.657. The
   encoder ranking by interpolation R² is the ranking by static midpoint R² (Pearson 0.991 across 10 encoders); the temporal part is
   +0.035…+0.066 for every encoder, and in the middle of the clip (t = 6…10) interpolation is no better than a constant.
6. **Metrics with a floor far from zero or a ceiling you cannot reach need re-anchoring.** `time_progress_spearman` floors at 0.31 and
   saturates at r ≳ 0.03; `adjacent_temporal_nn` floors at 0.125; `direction_reversal` floors at 0.96 (0.99 on Ego4D — real data is *on the floor*).
7. **Per-clip reliability is poor for four metrics** (ICC 0.3–0.66: `monotonic_progress_fraction`, `adjacent_temporal_nn_fraction`,
   `direction_reversal_fraction`, `norm_turning_point_fraction`): fine for dataset means, not for per-clip or per-class use.
8. **Probes are the noisy ones.** A single balanced accuracy on 1,430 clips has SE ≈ 0.025 (95% CI ±0.05), and paired comparison only
   tightens that to ≈ 0.027, so a single-seed comparison cannot detect differences below ≈ 0.075.
9. **One of your claims does not hold:** "straightness hasn't peaked by sw 2.0": sw 1.5 → 2.0 is flat (Δ = −0.0007, CI [−0.0039, +0.0025]) and velocity-cosine
   drops significantly (§10).

**Rule of thumb that falls out of this:** report every gain as (i) a paired Δ with a CI, (ii) a fraction of the [floor, ceiling] range,
(iii) *relative to untrained architecture, whitened raw and σ-matched smoothing*, and (iv) alongside a metric that smoothing destroys
(chirality, prediction skill). §11 makes this a checklist; `compare.py` does it in one call.

---

## 1. What "floor", "ceiling" and "noise" mean here

| role | construction (all computed on the *same latents* as the metric under test) | answers |
|---|---|---|
| **floor** | `shuffle_time`: each clip's frames in random order (keeps content + per-frame noise, removes order). For clip-identity metrics also `shuffle_pool` (frames from different clips), `iid_gauss` (matched covariance) | what a trajectory with no temporal signal scores |
| **decoy** | `brownian_signflip`: the clip's real step vectors with random signs | smooth but undirected motion — what the metric should *not* reward |
| **free baselines** | Gaussian temporal smoothing (σ = 0.25…2 frames), EMA, causal cumulative mean; PCA-whitening, rotation, scale, LayerNorm; *untrained* copy of the architecture | what a gain must beat to be worth reporting |
| **formal ceiling** | straight line through the clip's real endpoints | the metric's maximum |
| **signal references** | denoised clip (cubic fit, Gaussian σ = 1); non-persistent-noise fraction by difference estimators | what the metric would give if per-frame noise were removed |
| **learnable ceilings** | closed-form linear oracles (SFA); ridge / kNN predictors; 3-tap AR; closed-form noise ceiling | what a *learned* map can reach |
| **empirical encoder ceiling** | frozen encoders on synthetic clips with *known* constant-velocity motion, no noise | what any frozen encoder can score on ideal input |

| noise source | how it was measured | where |
|---|---|---|
| clip sampling | bootstrap over clips; **paired** bootstrap for differences; cluster bootstrap by video on Ego4D | §8, `noise.py` |
| training seed | 5 seeds × {no-reg, VICReg sw 0.1, VICReg sw 1, SigReg} + 5 untrained inits (CLS family, SSv2-trained, 50 epochs) | §8, §9 |
| epoch | epochs 29 / 39 / 49 of the same run | §8 |
| frame sampling | re-extracted 700 SSv2-val clips × 4 random 16-frame samplings (DINOv2, DINOv3, CLIP); base sampling verified to reproduce your stored features bit-for-bit | §2.5 |
| metric internals | probe subsample seeds, TCC pair seeds, Monte-Carlo pairs | §5 |

---

## 2. Trajectory / progress metrics

![reference ladder](results/figures/fig1_reference_ladder.png)

### 2.1 The calibration card (DINOv2-S CLS)

`position` = (real − floor)/(ceiling − floor). `MDD` = minimum detectable difference for **one trained model vs raw, 80% power, 5%
two-sided**, from the paired clip SE and the training-seed sd. `range/MDD` is how many MDD-sized steps fit in the formal range. `whiten`/`smooth` columns show what free
baselines do to the metric (in MDDs / in % of formal range).

| metric | floor | DINOv2-S | random walk | smooth σ=1 | ceiling | position | MDD | range/MDD | ICC | sensitive window r (10–90%) | whiten (×MDD) | smooth σ=1 (% range) | reversal-blind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| time_progress_spearman | 0.307 | 0.904 | 0.984 | 0.966 | 1.000 | 0.861 | 0.005 | 150.176 | 0.846 | 0.00112 – 0.00527 | 3.553 | 8.954 | yes |
| pc1_time_spearman | 0.123 | 0.697 | 0.946 | 0.780 | 1.000 | 0.654 | 0.013 | 67.560 | 0.837 | 0.00109 – 0.00375 | 1.886 | 9.554 | yes |
| monotonic_progress_fraction | 0.562 | 0.733 | 0.879 | 0.899 | 1.000 | 0.391 | 0.006 | 76.006 | 0.506 | 0.00191 – 0.0229 | 3.827 | 37.875 | yes |
| progress_efficiency | 0.225 | 0.720 | 0.934 | 0.943 | 1.000 | 0.638 | 0.005 | 145.066 | 0.812 | 0.0016 – 0.0164 | 7.500 | 28.807 | yes |
| adjacent_temporal_nn_fraction | 0.125 | 0.787 | 0.907 | 0.999 | 1.000 | 0.757 | 0.005 | 164.365 | 0.513 | 0.00282 – 0.122 | 2.758 | 24.230 | yes |
| distance_to_goal_spearman | 0.003 | 0.611 | 0.954 | 0.762 | 1.000 | 0.610 | 0.012 | 83.475 | 0.917 | 0.00116 – 0.00708 | 0.622 | 15.152 | no |
| trajectory_straightness | 0.067 | 0.135 | 0.241 | 0.304 | 0.968 | 0.075 | 0.001 | 914.054 | 0.965 | 0.0445 – 9.24 | -11.000 | 18.792 | yes |
| mean_velocity_cosine_raw | -0.468 | -0.263 | 0.000 | 0.654 | 1.000 | 0.140 | 0.002 | 937.845 | 0.773 | 0.216 – 17.7 | -11.687 | 62.438 | yes |
| mean_chord_alignment_raw | 0.062 | 0.115 | 0.238 | 0.281 | 1.000 | 0.057 | 0.001 | 1112.801 | 0.868 | 0.0446 – 9.25 | -4.020 | 17.674 | yes |
| direction_reversal_fraction | 0.963 | 0.894 | 0.499 | 0.005 | 0.000 | 0.071 | 0.005 | 179.027 | 0.664 | step at r≈1 | 6.671 | 92.365 | yes |

Reading it:
* **`time_progress_spearman`** looks near-saturated (position 0.86) but its floor is **0.31, not 0**: `p_1 = 0` and `p_T = 1` are fixed by construction, so
  even i.i.d. noise scores 0.33. Smoothing σ = 1 → 0.966; cumulative mean → 0.981.
* **`trajectory_straightness`** has the *largest usable range*: real 0.135 vs floor 0.067 (= 1/15, the value for white noise) vs line 0.968. Real encoders use 7.5% of it.
* **`mean_velocity_cosine`** floors at −0.5 (white noise) and 0 for a random walk; real −0.26.
* **`direction_reversal_fraction`** floors at 0.96–1.0; real 0.89. On Ego4D-val real = 0.965 vs floor 0.988, i.e. *on the floor*.
* **`distance_to_goal_spearman`** is the one ordering metric with a true zero floor (0.003) and good clip-level reliability (ICC 0.92).
* Every metric in this table except `distance_to_goal` is **blind to the arrow of time** (reverse a clip → same value; table column "reversal-blind").

### 2.2 What the metrics can and cannot see

**(a) A random walk beats real DINOv2.** Integrating the clip's *own* step vectors with random signs gives time_progress 0.984, straightness 0.241,
velocity-cosine 0.000, direction-reversal 0.499, adjacent-NN 0.907, distance-to-goal 0.954 — all "better" than real DINOv2 (0.904, 0.135, −0.263, 0.894, 0.787, 0.611).
Real clips are *less* coherent than a random walk because ≈ 38% of within-clip variance is **non-persistent frame-to-frame noise**
(difference-estimator, converged at order 3–4; DINOv2 family 0.36–0.40, V-JEPA 2/2.1 0.25, CLIP/SigLIP 0.45–0.46, your synthetic `dinov2_straightline` features 0.000, which validates the estimator).

**(b) Free smoothing dominates.** σ = 1 frame Gaussian smoothing of the *raw* features:

| | raw | σ=0.5 | σ=1 | σ=2 | EMA 0.5 | cum. mean | trained VICReg sw0.1 | trained VICReg sw1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| time_progress | 0.904 | 0.927 | 0.966 | 0.991 | 0.968 | 0.981 | 0.915 | 0.906 |
| straightness | 0.135 | 0.174 | 0.304 | 0.484 | 0.248 | 0.466 | 0.124 | 0.130 |
| velocity-cos (raw) | −0.263 | −0.015 | 0.654 | 0.909 | 0.184 | 0.535 | −0.280 | −0.262 |
| direction reversal | 0.894 | 0.517 | 0.005 | 0.000 | 0.266 | 0.049 | 0.931 | 0.906 |
| chirality accuracy | 0.799 | 0.798 | 0.771 | 0.744 | 0.779 | 0.694 | 0.786 | 0.792 |
| linear probe c+c | 0.524 | 0.526 | 0.529 | 0.534 | 0.508 | 0.451 | 0.418 | 0.460 |

The cost of smoothing shows up only in the metrics that need *change* information (chirality −0.03 at σ = 1, −0.10 for the cumulative mean). So **a trajectory-metric gain is only
evidence of better representation if chirality (or prediction skill) does not drop** — Fig. 5 draws this frontier.

![trade-off frontier](results/figures/fig5_tradeoff_frontier.png)

**(c) Saturation and the sensitive window.** On straight-line + white-noise trajectories with drift-to-noise ratio r (closed forms, validated by simulation in
`tests/test_synthetic_latent.py`: velocity-cosine = (r−1)/(r+2), straightness = √((k·r+2)/(k·(r+2))) with k = (T−1)²):
`time_progress`, `distance_to_goal`, `monotonic`, `efficiency` are ≥ 0.98 of their maximum by **r ≈ 0.03** and cannot discriminate beyond (10→90% windows: time_progress r ∈ [0.001, 0.005], distance-to-goal [0.001, 0.007]);
`adjacent_NN` covers r ≈ 0.003–0.12; straightness r ≈ 0.04–9 and velocity-cosine r ≈ 0.2–18 (the only two with room above real encoders); `direction_reversal` is a step function at r ≈ 1 and sits on its plateau for r < 0.5.
Real DINOv2 maps to r = 0.002 (time_progress), 0.03 (straightness), 0.37 (velocity-cosine): the three metrics disagree by more than two orders of magnitude, so real
trajectories are *not* "straight line + white noise" — they are curved *and* noisy.

![SNR response](results/figures/fig3_snr_response.png)

**(d) No linear map can fix this.** Closed-form Slow-Feature-Analysis (the optimal linear straightener at a fixed dimension, whitened so it cannot rank-collapse) on the train split, evaluated on val:
over k = 384, 128, 32 and pooled- or within-clip whitening, straightness 0.124–0.151, velocity-cosine −0.28…−0.22, direction-reversal 0.78–0.93, time_progress 0.84–0.92 (raw: 0.135, −0.263, 0.894, 0.904); first- and second-order SFA give the same.
The per-frame noise is spread across many directions, so any improvement must come from a *nonlinear* per-frame map or from mixing frames in time.

**(e) "Progress" is clip-local.** A supervised *linear* read-out of frame index from a single frame reaches mean per-clip Spearman 0.23 (median 0.33); a quick untuned MLP, 0.14–0.17 (a ceiling for *those* readouts, not for every possible one).
Absolute position-in-task is barely decodable from one DINOv2 frame, so `time_progress_spearman` = 0.90 measures monotone drift along each clip's own start→end chord, not "how far through the task".

### 2.3 What a frozen encoder can reach on ideal input (synthetic, known motion)

I rendered 200 clips per condition (224², textured background, one object; `synth_videos.py`) and encoded them with your registry encoders (re-extraction pipeline
verified bit-exact against stored features). `static_noise` and `iid_scenes` reproduce the predicted floors almost exactly (straightness 0.066, velocity-cosine −0.49, adjacent-NN 0.12 — the
floor is a property of the pipeline, not of my null).

**time_progress_spearman**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.305 | 0.334 | 0.843 | 0.584 | 0.859 | 0.991 | 0.990 |
| dinov2_base | 0.307 | 0.323 | 0.753 | 0.526 | 0.778 | 0.990 | 0.990 |
| dinov2_large | 0.317 | 0.319 | 0.765 | 0.514 | 0.753 | 0.992 | 0.990 |
| clip | 0.303 | 0.322 | 0.852 | 0.632 | 0.881 | 0.963 | 0.982 |
| siglip | 0.317 | 0.342 | 0.839 | 0.728 | 0.836 | 0.922 | 0.985 |
| levljepa | 0.314 | 0.344 | 0.815 | 0.819 | 0.850 | 0.993 | 0.988 |
| vjepa2.1 | 0.325 | 0.320 | 0.931 | 0.625 | 0.917 | 0.999 | 0.991 |
| pixel | 0.282 | 0.334 | 0.992 | 0.865 | 0.992 | 1.000 | 0.989 |

**trajectory_straightness**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.066 | 0.066 | 0.124 | 0.090 | 0.135 | 0.339 | 0.573 |
| dinov2_base | 0.066 | 0.066 | 0.128 | 0.085 | 0.138 | 0.298 | 0.532 |
| dinov2_large | 0.065 | 0.066 | 0.154 | 0.086 | 0.158 | 0.300 | 0.462 |
| clip | 0.066 | 0.066 | 0.110 | 0.083 | 0.123 | 0.255 | 0.420 |
| siglip | 0.066 | 0.066 | 0.120 | 0.086 | 0.129 | 0.173 | 0.373 |
| levljepa | 0.067 | 0.066 | 0.098 | 0.092 | 0.111 | 0.197 | 0.404 |
| vjepa2.1 | 0.066 | 0.066 | 0.161 | 0.079 | 0.165 | 0.503 | 0.689 |
| pixel | 0.064 | 0.067 | 0.213 | 0.132 | 0.213 | 0.368 | 0.334 |

**mean_velocity_cosine_raw**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | -0.482 | -0.493 | -0.379 | -0.419 | -0.250 | -0.184 | 0.000 |
| dinov2_base | -0.484 | -0.493 | -0.362 | -0.440 | -0.271 | -0.132 | 0.000 |
| dinov2_large | -0.478 | -0.490 | -0.278 | -0.426 | -0.205 | -0.029 | 0.000 |
| clip | -0.486 | -0.496 | -0.443 | -0.456 | -0.261 | -0.205 | 0.000 |
| siglip | -0.490 | -0.491 | -0.354 | -0.446 | -0.240 | 0.013 | 0.000 |
| levljepa | -0.489 | -0.495 | -0.473 | -0.437 | -0.287 | -0.259 | 0.000 |
| vjepa2.1 | -0.485 | -0.493 | -0.171 | -0.461 | -0.152 | 0.231 | 0.000 |
| pixel | -0.405 | -0.487 | 0.592 | -0.240 | 0.591 | 0.570 | 0.000 |

**direction_reversal_fraction**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.948 | 0.984 | 0.886 | 0.907 | 0.770 | 0.705 | 0.000 |
| dinov2_base | 0.944 | 0.990 | 0.824 | 0.919 | 0.762 | 0.674 | 0.000 |
| dinov2_large | 0.917 | 0.977 | 0.733 | 0.876 | 0.660 | 0.499 | 0.000 |
| clip | 0.975 | 0.997 | 0.970 | 0.976 | 0.808 | 0.749 | 0.000 |
| siglip | 0.990 | 0.996 | 0.926 | 0.979 | 0.808 | 0.463 | 0.000 |
| levljepa | 0.986 | 1.000 | 0.974 | 0.979 | 0.814 | 0.845 | 0.000 |
| vjepa2.1 | 0.978 | 0.994 | 0.714 | 0.968 | 0.688 | 0.284 | 0.000 |
| pixel | 0.730 | 0.968 | 0.000 | 0.714 | 0.010 | 0.000 | 0.000 |

Object-size sweep (clean constant-velocity motion; bigger object = stronger signal):

**time_progress_spearman**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.662 | 0.736 | 0.889 | 0.919 | 0.944 |
| dinov2 | 0.582 | 0.653 | 0.860 | 0.933 | 0.973 |
| dinov3 | 0.527 | 0.613 | 0.831 | 0.948 | 0.979 |
| pixel | 0.814 | 0.939 | 0.997 | 1.000 | 1.000 |
| vjepa2.1 | 0.589 | 0.707 | 0.948 | 0.993 | 0.954 |

**trajectory_straightness**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.092 | 0.094 | 0.113 | 0.170 | 0.179 |
| dinov2 | 0.097 | 0.101 | 0.128 | 0.182 | 0.204 |
| dinov3 | 0.079 | 0.081 | 0.104 | 0.153 | 0.188 |
| pixel | 0.104 | 0.146 | 0.222 | 0.343 | 0.436 |
| vjepa2.1 | 0.102 | 0.119 | 0.170 | 0.244 | 0.336 |

**mean_velocity_cosine_raw**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | -0.443 | -0.483 | -0.409 | -0.303 | -0.094 |
| dinov2 | -0.407 | -0.445 | -0.355 | -0.232 | -0.105 |
| dinov3 | -0.468 | -0.509 | -0.441 | -0.328 | -0.153 |
| pixel | 0.050 | 0.426 | 0.610 | 0.637 | 0.623 |
| vjepa2.1 | -0.324 | -0.280 | -0.139 | 0.019 | 0.289 |

**direction_reversal_fraction**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.943 | 0.965 | 0.963 | 0.896 | 0.625 |
| dinov2 | 0.899 | 0.887 | 0.870 | 0.790 | 0.652 |
| dinov3 | 0.930 | 0.941 | 0.951 | 0.909 | 0.749 |
| pixel | 0.461 | 0.005 | 0.000 | 0.000 | 0.000 |
| vjepa2.1 | 0.766 | 0.776 | 0.708 | 0.446 | 0.106 |

* On **clean** motion DINOv2 scores time_progress 0.84, straightness 0.124, velocity-cosine −0.38, direction-reversal 0.89 — *close to real SSv2* (0.90 / 0.135 / −0.26 / 0.89). So video noise is not the main reason these values are low; a large part is how the encoder responds to smooth input (signal strength also matters, see the size sweep).
* Raw pixels (24×24) score straightness 0.21 (default object) to 0.44 (90 px), velocity-cosine +0.59: encoders **add** nonlinear jitter relative to pixels.
* Even with a 90-px object: DINOv2 straightness 0.20, V-JEPA 2.1 0.34, pixels 0.44. These are the empirical "best a frozen encoder can do on perfect motion" values. Your trained models' 0.20–0.21 (collapsed no-reg models, patch sw ≥ 1.0) sit **at that level**, consistent with those gains being partly collapse (their effective-rank fractions are 0.43 and 0.20–0.63).
* Edge cases: (i) clips whose first and last frames are identical (`out_and_back`) have a zero chord and are **silently dropped** (0 of 200 kept, every encoder) — a "pick up and put back" procedure behaves this way; (ii) clips with repeated frames make velocity-cosine undefined and the code maps it to **0.0** (`step_change`), which reads as "straight-ish" rather than "undefined".

![examples](results/synth/examples.png)

### 2.4 Across encoders (SSv2 val)

| encoder | time_prog. | straight. | vel.cos | dir.rev. | adj.NN | dist→goal |
|---|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.904 | 0.135 | -0.263 | 0.894 | 0.787 | 0.611 |
| dinov2_base | 0.898 | 0.134 | -0.272 | 0.902 | 0.768 | 0.604 |
| dinov2_large | 0.890 | 0.137 | -0.267 | 0.872 | 0.755 | 0.602 |
| dinov2_giant | 0.879 | 0.147 | -0.244 | 0.825 | 0.751 | 0.611 |
| dinov3 | 0.911 | 0.133 | -0.260 | 0.901 | 0.804 | 0.609 |
| clip | 0.873 | 0.121 | -0.293 | 0.906 | 0.742 | 0.569 |
| siglip | 0.882 | 0.118 | -0.311 | 0.933 | 0.725 | 0.568 |
| levljepa | 0.891 | 0.126 | -0.283 | 0.905 | 0.758 | 0.603 |
| vjepa2 | 0.883 | 0.178 | -0.132 | 0.635 | 0.763 | 0.617 |
| vjepa2.1 | 0.893 | 0.172 | -0.162 | 0.758 | 0.856 | 0.629 |
| *floor (shuffled), range over encoders* | 0.26 to 0.31 | 0.07 | -0.47 to -0.45 | 0.89 to 0.98 | 0.12 to 0.13 | -0.00 to 0.00 |
| *σ=1 smoothing of the same features, range* | 0.94 to 0.97 | 0.29 to 0.35 | 0.63 to 0.71 | 0.00 to 0.02 | 0.99 to 1.00 | 0.74 to 0.77 |

The whole spread across the ten encoders (e.g. time_progress 0.873–0.911, i.e. 0.04) is smaller than the shift σ = 1 smoothing produces on any single one of them (+0.06 to +0.09); V-JEPA 2 / 2.1 (video-trained) stand out on straightness, velocity-cosine and direction-reversal.

### 2.5 Reliability at the clip level (frame-sampling jitter)

Four random 16-frame samplings of each of 700 videos; ICC(1,1) of each per-clip metric:

| metric | clip | dinov2 | dinov3 | clip-level use |
|---|---:|---:|---:|---:|
| within_clip_mean_pairwise_l2 | 0.989 | 0.994 | 0.994 | reliable |
| start_end_displacement | 0.957 | 0.982 | 0.979 | reliable |
| adjacent_l2_distance | 0.971 | 0.979 | 0.981 | reliable |
| adjacent_cosine_distance | 0.962 | 0.974 | 0.975 | reliable |
| trajectory_straightness_raw | 0.921 | 0.965 | 0.962 | reliable |
| trajectory_straightness | 0.925 | 0.965 | 0.963 | reliable |
| norm_std_within_clip | 0.924 | 0.964 | 0.946 | reliable |
| norm_growth | 0.894 | 0.942 | 0.932 | reliable |
| distance_to_goal_spearman | 0.886 | 0.917 | 0.919 | reliable |
| mean_chord_alignment | 0.789 | 0.869 | 0.876 | moderate |
| mean_chord_alignment_raw | 0.781 | 0.868 | 0.876 | moderate |
| time_progress_spearman | 0.778 | 0.846 | 0.826 | moderate |
| pc1_time_spearman | 0.821 | 0.837 | 0.834 | moderate |
| progress_efficiency | 0.775 | 0.812 | 0.812 | moderate |
| mean_velocity_cosine | 0.711 | 0.774 | 0.763 | moderate |
| mean_velocity_cosine_raw | 0.708 | 0.773 | 0.763 | moderate |
| direction_reversal_fraction | 0.619 | 0.664 | 0.654 | unreliable |
| adjacent_temporal_nn_fraction | 0.508 | 0.513 | 0.502 | unreliable |
| monotonic_progress_fraction | 0.476 | 0.506 | 0.553 | unreliable |
| norm_turning_point_fraction | 0.265 | 0.318 | 0.281 | unreliable |

The four "unreliable" metrics (ICC 0.27–0.66) are counts over 15 steps — *dataset means are fine, per-clip values are mostly sampling noise*.
Switching from the stored linspace sampling to random-in-segment sampling also shifts some dataset means (base − jittered: adjacent-NN −0.018, velocity-cosine −0.010), so **metrics are only comparable under the same frame-sampling protocol**.

**Ego4D has a large clustering effect.** The 6,732 val clips come from 134 videos: the clip-level bootstrap SE is too small by a factor 1.9–3.0 (design effect 3.5–8.7×; e.g. time_progress SE 0.0034 → 0.0076). Use `noise._counts(..., groups=video_uid)`.

---

## 3. Collapse / spread metrics

* **`effective_rank_fraction` can be set to ≈ 1.0 by a free linear map.** Raw DINOv2 0.812 → PCA-whitened 0.943 (train-fitted, applied to val); fitted on val, 1.000. Its finite-sample ceiling for isotropic Gaussian noise at your shapes is 0.998 (D = 384), 0.999 (D = 128).
  It is also **insensitive to a dominant direction**: one direction with 900× the variance of the rest (plus unit noise in the other 383 dims) still scores 0.83, above raw DINOv2. Because it normalises singular values, not variances.
  So "ER fraction above DINOv2" does not certify a healthy representation. As a *collapse alarm* (ER falling far below raw, e.g. no-reg 0.431) it still works.
* **Scale-dependent metrics** — `dead_dim_fraction` (0.23 at raw; 1.00 when features are ×0.1; 0.00 at ×10), `adjacent_l2_distance`, `within_clip_mean_pairwise_l2`, `mean_pairwise_l2_across_clips`, `norm_std_within_clip` — move 10× under a global rescale. Compare only within one checkpoint family (as your README says) or as ratios.
* **`within/across` ratio**: scale-free by construction (a ratio of two L2 means; ≈ 1 → no clip identity). I did not calibrate it further.
* Use an **information-retention check that is invariant to invertible linear maps** (reconstruction R² back to raw, or linear CKA with raw) as the collapse guard, together with the ER alarm.

---

## 4. Interpolation planning (`cosine_sim`, `relative_l2`, `R²`)

![planning ladder](results/figures/fig2_planning_ladder.png)

| encoder | R² static midpoint | R² interp (reported) | temporal part [95% CI] | R² ridge | R² adjacent-frame oracle | R² noise ceiling | cos: mean-predictor | cos: interp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vjepa2.1 | 0.750 | 0.793 | +0.043 [0.040, 0.046] | 0.807 | 0.938 | 0.962 | 0.960 | 0.995 |
| dinov2_giant | 0.679 | 0.730 | +0.051 [0.048, 0.055] | 0.732 | 0.882 | 0.924 | 0.276 | 0.910 |
| dinov3 | 0.675 | 0.716 | +0.041 [0.038, 0.044] | 0.732 | 0.876 | 0.920 | 0.450 | 0.908 |
| dinov2_large | 0.639 | 0.689 | +0.050 [0.047, 0.053] | 0.674 | 0.855 | 0.906 | 0.311 | 0.891 |
| dinov2_base | 0.607 | 0.659 | +0.052 [0.049, 0.055] | 0.647 | 0.839 | 0.895 | 0.334 | 0.880 |
| dinov2 | 0.604 | 0.657 | +0.053 [0.050, 0.056] | 0.660 | 0.846 | 0.900 | 0.352 | 0.878 |
| vjepa2 | 0.584 | 0.650 | +0.066 [0.062, 0.070] | 0.677 | 0.894 | 0.934 | 0.827 | 0.963 |
| siglip | 0.610 | 0.645 | +0.035 [0.033, 0.038] | – | 0.826 | 0.887 | 0.794 | 0.943 |
| clip | 0.570 | 0.608 | +0.037 [0.034, 0.040] | – | 0.814 | 0.879 | 0.753 | 0.924 |
| levljepa | 0.489 | 0.546 | +0.057 [0.053, 0.061] | 0.567 | 0.783 | 0.859 | 0.759 | 0.912 |

* **What 0.657 is made of (DINOv2-S):** predict the dataset mean → −0.016; copy z₁ → 0.414; **static midpoint (z₁+z_T)/2 at every t → 0.604**; interpolation → 0.657. The temporal part is **+0.053** [0.050, 0.056]; across all ten encoders +0.035…+0.066.
* **The encoder ranking is a static-identity ranking**: Pearson(interp R², static-midpoint R²) = 0.991 over 10 encoders; the static part spans 0.49–0.75, the temporal part 0.035–0.066 and is uncorrelated with the headline (Spearman −0.14). Skill of interpolation over the static midpoint (1 − MSE/MSE_mid): clip 0.086, siglip 0.090, levljepa 0.112, dinov3 0.126, dinov2 0.133, dinov2_large 0.139, vjepa2 0.159, dinov2_giant 0.160, vjepa2.1 0.171.
* **Per position**: gain over the static midpoint is +0.134 at t = 1, +0.121 at t = 14, and ≈ 0 for t = 6…10 (+0.012, +0.002, −0.001, +0.003, +0.017). The "straight line between start and goal" premise is supported only next to the endpoints, which is autocorrelation, not global straightness.
* **Cosine has a floor set by the dataset mean.** Predicting the dataset mean gives median cosine 0.960 for V-JEPA 2.1 (0.35 for DINOv2-S). V-JEPA 2.1's 0.995 therefore sits at 0.88 of [floor, 1] vs DINOv2-S 0.81, and CLIP/SigLIP/LeVLJEPA 0.64–0.72. On the README's open question: most of the 0.995 *is* the common-mode floor, but V-JEPA 2.1 also has the highest skill of interpolation over a static midpoint (0.171 vs 0.133 for DINOv2-S), so it is not purely an artefact.
* **Learnable ceiling ≈ interpolation + 0.01–0.03.** Ridge (shrunk toward interpolation) on train, evaluated on val: −0.015 … +0.027 R² vs interpolation (CIs in the table). A kNN residual predictor on DINOv2: +0.011 at best. Your trained transformers are *below* interpolation (R² 0.619 vs 0.657 on DINOv2-S, `gap_closed_by_training` −0.27…−0.44); they are not an upper bound.
  Adjacent-frame oracle (0.78–0.94) and the noise ceiling (0.86–0.96) cannot be approached from the endpoints alone; my estimate of the real ceiling from (z₁, z_T) alone is ≈ 0.66–0.69 for DINOv2-S.
* **R² is not reparametrisation-invariant**: PCA-whitening the *same* information changes interpolation R² 0.657 → 0.545 and cosine 0.838 → 0.735 (§9). Do not compare R² across encoders or between raw and re-normalised latents.

---

## 5. Representation probes

| encoder | chance | perm-null | linear c+c | val-boot 95% CI | sd over 5 train subsamples | ceiling (all train) | chirality | chir. CI | chir. order-blind | chir. shuffled-label | TCC τ | τ null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.045 | 0.045±0.012 | 0.524 | [0.475, 0.572] | 0.012 | 0.540 | 0.799 | [0.766, 0.829] | 0.523 | 0.502 | 0.076 | 0.012 |
| dinov2_base | 0.045 | 0.046±0.014 | 0.464 | [0.414, 0.515] | 0.022 | 0.499 | 0.811 | [0.778, 0.843] | 0.512 | 0.506 | 0.094 | -0.002 |
| dinov2_large | 0.045 | 0.047±0.011 | 0.459 | [0.415, 0.509] | 0.021 | 0.473 | 0.810 | [0.780, 0.837] | 0.493 | 0.502 | 0.078 | 0.011 |
| dinov3 | 0.045 | 0.044±0.010 | 0.461 | [0.418, 0.508] | 0.011 | 0.483 | 0.793 | [0.763, 0.823] | 0.531 | 0.505 | 0.090 | 0.006 |
| levljepa | 0.045 | 0.044±0.008 | 0.367 | [0.323, 0.415] | 0.009 | 0.430 | 0.715 | [0.682, 0.749] | 0.515 | 0.507 | 0.056 | 0.002 |
| vjepa2.1 | 0.045 | 0.044±0.009 | 0.541 | [0.494, 0.593] | 0.004 | 0.580 | 0.787 | [0.750, 0.817] | 0.509 | 0.500 | 0.163 | 0.018 |
| vjepa2 | 0.045 | 0.048±0.014 | 0.524 | [0.480, 0.570] | 0.006 | 0.570 | 0.783 | [0.748, 0.815] | 0.493 | 0.509 | 0.143 | 0.020 |

* **Floors are clean**: label-permuted linear probe 0.044–0.048 ± 0.01 (= 1/22); chirality label-shuffle null 0.50 ± 0.02–0.03; TCC τ null (labels shuffled) −0.002…0.020.
* **Chirality is genuinely order-dependent**: randomly permuting each clip's frames (train and test) drops it to 0.49–0.53. It is also insensitive to feature scale (×0.1 / ×10: ≤ 0.025 change), so the unscaled-ridge worry in the code does not bite; the scale-free RidgeCV version gives 0.71–0.82, within ±0.02 of the repo version.
* **Noise.** Single-model bootstrap SE of balanced accuracy 0.023–0.026 (95% CI ±0.05); paired differences between encoders have SE 0.025–0.030 (unpaired 0.033–0.036), so the paired MDD is ≈ 0.075. Training-subsample sd (5 seeds of the ≤200-per-class subsample): 0.004–0.022. Chirality CI ±0.03.
  → A difference of 0.04 between two probe numbers (e.g. 0.4802 vs 0.5242 in your tables) is **not** distinguishable from noise with this val set.
* **Ceilings.** Linear probe on *all* train clips: +0.01 to +0.06 over the subsampled probe; richer temporal statistics do not help. So the 22-way verb probe is near saturation for frozen features at ≈ 0.47–0.58.
* **TCC τ has an empirical range**: aligning a clip with a *different sampling of itself* gives τ = 0.957–0.973; aligning different videos 0.01 ± 0.02. Your values (0.056–0.163) are 6–17% of that range; seed sd 0.004–0.010.
* **Probes are not reparametrisation-invariant either**: PCA-whitened raw has linear-probe 0.422 (raw 0.524), change-gap 0.017 (0.039), TCC 0.034 (0.066), while kNN *rises* (0.343 → 0.353). Opposite directions for two probes of the same label tell you the effect is geometry, not information.

---

## 6. World-model prediction skill

![skill ladder](results/figures/fig6_skill_ladder.png)

| system | skill | note |
|---|---:|---:|
| copy last frame | 0.000 | by definition |
| EMA(0.5) of past frames (no learning) | 0.079 | free denoiser |
| ridge AR(1), raw latents | 0.074 | learned, linear |
| ridge AR(3), raw latents | 0.120 | learned, linear |
| noise ceiling, raw latents | 0.599 | upper bound on any predictor |
| trained predictor, no reg (n=5) | 0.053 | sd 0.0118; ridge AR(3) in its own latents 0.074; noise ceiling 0.728 |
| trained predictor, VICReg sw0.1 (n=5) | 0.103 | sd 0.0014; ridge AR(3) in its own latents 0.139; noise ceiling 0.584 |
| trained predictor, SigReg (n=5) | 0.055 | sd 0.0006; ridge AR(3) in its own latents 0.131; noise ceiling 0.590 |
| trained predictor, VICReg sw1 (n=5) | 0.079 | sd 0.0022; ridge AR(3) in its own latents 0.124; noise ceiling 0.596 |

* `skill = 1 − error/copy_error` has a **noise ceiling of ≈ 0.60** here (target-frame noise is unpredictable; derived from the difference estimator: ceiling = 1 − σ²/copy_error). "1.0 = perfect" is not attainable.
* **Free baselines that need no learning** reach +0.079 (EMA 0.5); a **3-tap linear model fit on the same latents reaches +0.120**; every trained predictor (0.053–0.103) is *below* the linear model in its own latent space.
  Report skill relative to the AR(3) baseline, and as a fraction of [AR(3), noise ceiling].
* The cumulative mean that maxes out the progress metrics is a terrible predictor (skill −0.41): prediction skill and the progress metrics pull in opposite directions under smoothing, which makes skill (together with chirality) the natural second axis for any "gain" claim.
* Seed noise on skill: sd 0.0006–0.012 (n = 5 per config; the collapsed no-reg config is the noisy one).

---

## 7. Injection-eval (AUROC)

This is the best-calibrated part already: AUROC has an explicit floor (0.5), `null_split` and `appearance` controls come out at 0.500, bootstrap CIs are reported, and thresholds come from clean calibration clips. What I checked (from the existing result files, not re-run):
* `reverse` AUROC is 0.49999 for `cv_mean`: that score is reversal-symmetric by construction, like the progress metrics.
* `freeze` AUROC 0.995 is probably near-trivial (a held frame produces a zero step; I did not test this).
* No ceiling is defined. A supervised detector trained on perturbed vs clean *train* clips with the same features would give the achievable AUROC for each edit (not run here).
* Thresholds from 200 calibration clips (95th percentile) carry their own sampling noise that the clip bootstrap does not include.

---

## 8. The noise hierarchy

![noise vs free effects](results/figures/fig4_noise_vs_free_effects.png)

For the trajectory metrics of one trained CLS model compared with raw on the same clips (DINOv2-S):

| metric | unpaired clip SE (one model) | paired clip SE (model − raw) | training-seed sd | epoch 29→49 sd | MDD (80%) |
|---|---:|---:|---:|---:|---:|
| time_progress_spearman | 0.0030 | 0.0014 | 0.0009 | 0.0002 | 0.0046 |
| trajectory_straightness | 0.0012 | 0.0003 | 0.0000 | 0.0004 | 0.0010 |
| mean_velocity_cosine_raw | 0.0020 | 0.0005 | 0.0002 | 0.0006 | 0.0016 |
| direction_reversal_fraction | 0.0029 | 0.0018 | 0.0007 | 0.0009 | 0.0054 |
| adjacent_temporal_nn | 0.0035 | 0.0017 | 0.0008 | 0.0004 | 0.0053 |
| distance_to_goal | 0.0100 | 0.0040 | 0.0015 | 0.0013 | 0.0119 |

* **Paired is ≈ 1.6–4× tighter than unpaired** for trajectory metrics; for probe accuracy it is only ~1.2× tighter.
* **Clip sampling is the largest term for most metrics, then seed.** For straightness and velocity-cosine the epoch-to-epoch drift (0.0004–0.0006) is as large as the paired clip SE and larger than the seed sd, so pin the epoch you report. With K seeds the seed term shrinks by √K; the clip term does not.
* Everything above is for the **CLS family** (strictly per-frame residual-MLP encoder with `encoder_lr = 1e-5`, which stays close to identity: the untrained copy already matches raw). The patch-channel family with a learned aggregation head moves much more, so its seed sd is likely larger.
  I did not train patch seeds (≈ 9.5 h each on a 48-GB GPU); use the same `noise.seed_summary` on 3 seeds of one patch config before trusting differences < ~0.01 in that family.
* **Probe metrics** have absolute noise an order of magnitude or more larger than trajectory metrics: seed sd 0.005–0.018 (untrained inits vary by ±0.018 on the linear probe), clip SE 0.025.

---

## 9. What training actually did (controls)

Five seeds each (mean ± sd); all on SSv2 val; "untrained" is the same architecture at random init:

| metric | raw DINOv2 | PCA-whitened raw | untrained | no reg | VICReg sw0.1 | VICReg sw1 | SigReg |
|---|---:|---:|---:|---:|---:|---:|---:|
| time_progress_spearman | 0.9040 | 0.9204 | 0.9035 ± 0.0002 | 0.5268 ± 0.0018 | 0.9152 ± 0.0009 | 0.9060 ± 0.0006 | 0.9098 ± 0.0005 |
| trajectory_straightness | 0.1348 | 0.1240 | 0.1347 ± 0.0001 | 0.2096 ± 0.0012 | 0.1241 ± 0.0000 | 0.1296 ± 0.0002 | 0.1288 ± 0.0001 |
| mean_velocity_cosine_raw | -0.2627 | -0.2810 | -0.2639 ± 0.0001 | -0.0714 ± 0.0041 | -0.2803 ± 0.0002 | -0.2620 ± 0.0004 | -0.2721 ± 0.0001 |
| direction_reversal_fraction | 0.8941 | 0.9300 | 0.8967 ± 0.0006 | 0.5423 ± 0.0031 | 0.9310 ± 0.0007 | 0.9061 ± 0.0011 | 0.9160 ± 0.0003 |
| adjacent_temporal_nn | 0.7873 | 0.8020 | 0.7870 ± 0.0008 | 0.5206 ± 0.0066 | 0.7969 ± 0.0008 | 0.7976 ± 0.0007 | 0.7916 ± 0.0004 |
| distance_to_goal | 0.6110 | 0.6185 | 0.6038 ± 0.0008 | 0.3963 ± 0.0040 | 0.5763 ± 0.0015 | 0.5720 ± 0.0013 | 0.5880 ± 0.0007 |
| planning R² (interp) | 0.6570 | 0.5451 | 0.6484 ± 0.0003 | 0.2794 ± 0.0047 | 0.4815 ± 0.0004 | 0.4775 ± 0.0006 | 0.5650 ± 0.0006 |
| planning R² (static midpoint) | 0.6042 | 0.4854 | 0.5962 ± 0.0004 | 0.1558 ± 0.0051 | 0.4236 ± 0.0005 | 0.4151 ± 0.0008 | 0.5092 ± 0.0007 |
| effective_rank_fraction | 0.8120 | 0.9425 | 0.8070 ± 0.0003 | 0.4311 ± 0.0146 | 0.9059 ± 0.0001 | 0.8823 ± 0.0001 | 0.8574 ± 0.0003 |
| linear probe c+c | 0.5242 | 0.4221 | 0.5026 ± 0.0181 | 0.4548 ± 0.0070 | 0.4183 ± 0.0085 | 0.4602 ± 0.0047 | 0.4838 ± 0.0059 |
| kNN probe c+c | 0.3434 | 0.3526 | 0.3418 ± 0.0068 | 0.1510 ± 0.0074 | 0.3681 ± 0.0113 | 0.4174 ± 0.0095 | 0.3692 ± 0.0078 |
| chirality acc. | 0.7993 | 0.7718 | 0.7986 ± 0.0064 | 0.7535 ± 0.0084 | 0.7855 ± 0.0093 | 0.7917 ± 0.0096 | 0.7850 ± 0.0055 |
| change-direction gap | 0.0386 | 0.0171 | 0.0385 ± 0.0004 | 0.1405 ± 0.0050 | 0.0219 ± 0.0003 | 0.0287 ± 0.0007 | 0.0299 ± 0.0005 |
| TCC τ | 0.0661 | 0.0339 | 0.0626 ± 0.0033 | 0.0627 ± 0.0055 | 0.0546 ± 0.0047 | 0.0774 ± 0.0019 | 0.0495 ± 0.0047 |

* **Untrained ≈ raw** on every trajectory/planning/collapse metric (e.g. time_progress 0.9035 vs 0.9040, planning R² 0.648 vs 0.657): LayerNorm + a residual MLP at init is close to identity, so "trained − raw" is the training effect.
* **VICReg sw 0.1 ≈ PCA-whitened raw.** Trajectory metrics (0.915 / 0.124 / −0.280 / 0.931 vs whitened 0.920 / 0.124 / −0.281 / 0.930), planning R² (0.482 vs 0.545), linear probe (0.418 vs 0.422), effective rank (0.906 vs 0.943), change-gap (0.022 vs 0.017) all land on or near the whitened point. The regulariser's visible effect is spectrum flattening, which costs −0.17 interpolation R² and −0.11 linear-probe accuracy (vs raw) and buys +0.09 effective rank — none of which indicates a better representation.
* **The straightening weight has a real effect on top of whitening:** sw 1 vs sw 0.1 differs beyond seed noise on direction-reversal (0.906 vs 0.931), TCC τ (0.077 vs 0.055), kNN probe (0.417 vs 0.368) and change-gap (0.029 vs 0.022).
  Relative to *raw*, sw 1 is the only configuration with gains on kNN (+0.074, ≈ 1 MDD) and TCC τ (+0.011) while keeping chirality (0.792 vs 0.799, inside noise) — the one candidate here for a real, non-whitening effect. Its trajectory-metric changes vs raw are all small (|Δ| ≤ 0.04): slight gains on adjacent-NN (+0.010), monotonic fraction (+0.014) and efficiency (+0.023); slight losses on straightness (−0.005), direction-reversal (+0.012) and distance-to-goal (−0.039). Worth 3+ more seeds and a paired probe comparison before claiming it.
* **No regulariser** (collapse-like: ER 0.43): straightness +0.075 and velocity-cosine +0.19 but time_progress −0.38, adjacent-NN −0.27, planning R² −0.38, kNN −0.19. The classic "straightness through collapse" pattern; the metric cannot distinguish it from improvement without ER/probe guards.
* **SigReg vs VICReg**: distinguishable on nearly every metric against seed noise (planning R² 0.565 vs 0.482, linear probe 0.484 vs 0.418, ER 0.857 vs 0.906), but trajectory differences are ≤ 0.015; chirality is indistinguishable (0.785 vs 0.785).

---

## 10. Re-audit of statements in the write-up

Paired clip-bootstrap CIs on your own per-clip result files (patch-channel, K400-trained, sw sweep at epoch 49; clip noise only; patch-family seed noise would widen these):

| contrast | metric | Δ | 95% CI (paired, clips) | excludes 0 |
|---|---:|---:|---:|---:|
| sw0.5 − sw0.1 | time_progress_spearman | 0.0921 | [+0.0837, +0.1001] | yes |
| sw0.5 − sw0.1 | trajectory_straightness | 0.0466 | [+0.0449, +0.0483] | yes |
| sw0.5 − sw0.1 | mean_velocity_cosine | 0.1762 | [+0.1718, +0.1806] | yes |
| sw0.5 − sw0.1 | direction_reversal_fraction | -0.1599 | [-0.1684, -0.1520] | yes |
| sw1 − sw0.5 | time_progress_spearman | -0.0836 | [-0.0921, -0.0754] | yes |
| sw1 − sw0.5 | trajectory_straightness | 0.0561 | [+0.0528, +0.0594] | yes |
| sw1 − sw0.5 | mean_velocity_cosine | 0.1373 | [+0.1313, +0.1434] | yes |
| sw1 − sw0.5 | direction_reversal_fraction | -0.2715 | [-0.2806, -0.2632] | yes |
| sw1.5 − sw1 | time_progress_spearman | -0.0745 | [-0.0833, -0.0657] | yes |
| sw1.5 − sw1 | trajectory_straightness | 0.0155 | [+0.0126, +0.0183] | yes |
| sw1.5 − sw1 | mean_velocity_cosine | 0.0132 | [+0.0073, +0.0198] | yes |
| sw1.5 − sw1 | direction_reversal_fraction | -0.0236 | [-0.0314, -0.0156] | yes |
| sw2 − sw1.5 | time_progress_spearman | -0.0456 | [-0.0562, -0.0351] | yes |
| sw2 − sw1.5 | trajectory_straightness | -0.0007 | [-0.0039, +0.0025] | **no** |
| sw2 − sw1.5 | mean_velocity_cosine | -0.0187 | [-0.0263, -0.0112] | yes |
| sw2 − sw1.5 | direction_reversal_fraction | 0.0086 | [+0.0010, +0.0159] | yes |

| claim | verdict |
|---|---|
| time_progress peaks at sw 0.5 and declines through sw 1.0–2.0 | **supported**: Δ = +0.092 [0.084, 0.100] (0.1→0.5), then −0.084, −0.075, −0.046, all CIs exclude 0. Note this decline coincides with effective rank 0.92 → 0.20. |
| straightness climbs monotonically through sw 2.0, "hasn't peaked, would push further" | **not supported past sw 1.5**: sw 1.5 → 2.0 Δ = −0.0007 [−0.0039, +0.0025]; velocity-cosine *falls* (−0.019 [−0.026, −0.011]) and reversal rises. Plateau. |
| "VICReg and SigReg are indistinguishable to within noise on every metric" (agg-head-bypassed cls checkpoints, K400-trained) | **not testable on your files** (different training set / no seeds); in the *SSv2-trained* CLS replicates the two are distinguishable on 9/10 metrics (§9), though by small amounts on trajectory metrics. |
| green/red calls vs DINOv2-CLS / mean baselines (e.g. 0.9040 vs 0.9062, 0.4802 vs 0.5242) | single numbers without CIs; for trajectory metrics differences ≥ ~0.001–0.012 (metric-dependent MDD, §2.1) are resolvable, **for probes ≥ ~0.075 are needed** — most probe colourings are inside noise. |
| "effective_rank_fraction green" | a free whitening makes it green (0.943 vs 0.812); not evidence of anything but spectrum shape. |
| the pilot: STRAIGHT targets give R² 0.77–0.90, "the ceiling straightening could buy" | a STRAIGHT target *is* the linear-interpolation output, so a perfect predictor there is plain interpolation (R² = 1 exactly); 0.77–0.90 measures how well a network learns to interpolate, not headroom on real trajectories. The realistic ceiling on the real task is ≈ interp + 0.02 (§4). |

---

## 11. Protocol: how to say "this is a real improvement"

Use `compare.py` (below) or do the same by hand. A gain on metric *M* of system A over baseline B counts only if **all** of these hold:

1. **Paired, noise-aware.** Same clips, same frame sampling; paired clip bootstrap CI excludes 0 (cluster by video on Ego4D); with ≥ 3 training seeds, a Welch test on seed-level values *and* the clip CI both pass. Effect > MDD (table in §2.1; ≈ 0.075 for probes).
2. **Meaningful size.** Effect ≥ 5% of [floor, ceiling] (the threshold is a declared knob, `--min-frac`), and reported as that fraction.
3. **Beats the free baselines.** Better than (a) the *untrained* architecture, (b) *PCA-whitened* B, (c) B smoothed to the σ that matches A (`smooth_eq_sigma`; if σ_eq < 0.5 frames the "gain" is smaller than a trivial filter).
4. **Not collapse.** Information-retention guard (reconstruction R² or CKA with raw) within MDD of B; effective-rank used only as a one-sided alarm.
5. **Two-sided.** The gain must not come with a drop in a metric that smoothing destroys: chirality ≥ B − MDD, prediction skill ≥ AR(3) baseline.
6. **Not past the ceiling.** If a value exceeds the empirical frozen-encoder ceiling (e.g. straightness > 0.20 on DINOv2-sized features) or the learnable ceiling (§4, §6), look for collapse or temporal mixing before celebrating.

```bash
# one trained CLS-family checkpoint vs raw DINOv2 (paired CI, range fraction, smoothing- and whitening-equivalence)
python compare.py --a ckpt:/path/to/last.ckpt --b raw:dinov2 --seed-sd-json results/seed_sd_cls_family.json --out results/compare_x.csv
```

---

## 12. Per-metric verdicts

| metric | floor → ceiling usable? | main problem | recommendation |
|---|---|---|---|
| `time_progress_spearman` | floor 0.31; saturates at r ≳ 0.03 | reversal-blind; smoothing → 0.97+ | keep as a *sanity* metric, report position in [0.31, 1] |
| `pc1_time_spearman` | floor 0.12 | same family, noisier | redundant with the above |
| `monotonic_progress_fraction`, `adjacent_temporal_nn` | floors 0.56 / 0.125 | clip-level ICC 0.5 | dataset-level only |
| `progress_efficiency` | floor depends on the null (0.22 shuffled vs 0.50 i.i.d.) | 1/total-variation of a chord projection, likely unstable when the chord is short | drop or report median |
| `distance_to_goal_spearman` | floor 0.00, clean | smoothing-sensitive | **keep (most planner-relevant ordering metric)** |
| `trajectory_straightness`, `mean_chord_alignment` | floor 0.067 / 0.062, range ≈ 0.9 | dominated by smoothing; encoder ceiling ≈ 0.2 | **keep, primary**, with free-baseline controls |
| `mean_velocity_cosine(_raw)` | floor −0.47; sensitive for r ≈ 0.2–18 | same | **keep, primary** |
| `direction_reversal_fraction` | floor 0.96–1.0 (0.988 on Ego4D) | step function; ICC 0.66 | replace by velocity-cosine |
| `norm_turning_point_fraction` | floor 0.665 | ICC 0.32 | drop |
| `effective_rank_fraction` | ceiling 0.998; gameable | whitening → 1.0; blind to one dominant direction | one-sided collapse alarm only |
| `dead_dim_fraction`, L2 / norm metrics | scale-dependent | ×0.1 → 1.0 | ratios or within-family only |
| interpolation R² / cosine | R² floor 0; cosine floor = mean-predictor cosine (0.35–0.96) | 92% static, reparam-sensitive | report skill vs static midpoint per t; cosine relative to its mean-predictor floor |
| semantic probes | floor 1/22; ceiling ≈ all-train probe | CI ±0.05, MDD 0.075 | paired + ≥ 3 subsample seeds |
| chirality | floor 0.50; order-blind control passes | CI ±0.03 | **keep; the anti-smoothing axis** |
| TCC τ, change-gap | τ range [0, 0.97] | small values, small sd | keep, quote as fraction of range |
| prediction skill | ceiling ≈ 0.60, linear baseline 0.12 | trained predictors below the baseline | report vs AR(3) and the noise ceiling |
| injection AUROC | floor 0.5 verified | no ceiling, calibration noise | add supervised ceiling |

---

## 13. Limitations, and how to reproduce

**Limitations.** (1) Seed noise is measured for the CLS family only (§8); patch-family seeds are the main missing piece. (2) The SFA and per-frame time-readout ceilings are *linear / lightly tuned*: they bound what those function classes reach, not what any nonlinear per-frame map could. (3) Synthetic clips are my own renders (object on a muted multi-scale texture); absolute ceilings depend on signal strength (hence the size sweep) and V-JEPA 2 could not be included (32-GiB activation allocation exceeds the 24-GB GPUs; not re-run on 48-GB nodes). (4) Ego4D was only checked for clustering and the reference ladder. (5) The jitter study used 700 clips and three encoders. (6) SSv2 only for probes/planning.

**Code (this directory).** `metrics_batch.py` (vectorised suite, verified against `compute_metrics_dict`), `references.py`, `noise.py`, `signal_noise.py`, `oracles.py`,
`synthetic_latent.py`, `synth_videos.py`, `calibrate_{trajectory,planning,probes,skill}.py`, `eval_noise_study.py`, `reparam_audit.py`, `extract_{jitter,synth}.py`, `analyze_*.py`, `compare.py`, `build_cards.py`, `make_tables.py`, `make_figures.py`, `tests/`.
All results are in `results/`; this report is generated by `build_report.py`. Seed-replicate checkpoints are in `experiments/predictive_world_model/checkpoints/noise_study/` (20 runs).
Training used `train.pbs` unchanged (`qsub -V`, 4 CPUs/24 GB/1 GPU, 2 h walltime, no node pinning).

```bash
cd experiments/encoder_planning_eval/metric_calibration
python tests/test_metrics_batch.py && python tests/test_signal_noise.py && python tests/test_synthetic_latent.py   # lift env
python calibrate_trajectory.py --encoders dinov2 vjepa2.1 --out results/trajectory_calibration.csv
python calibrate_planning.py   --encoders dinov2 vjepa2.1 --out results/planning_calibration.csv
python compare.py --a ckpt:<last.ckpt> --b raw:dinov2
```
