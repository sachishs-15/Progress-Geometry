# Representation probes

## What these are for

`progress_metrics/` checks whether trajectories are ordered and straight, and whether
the space has collapsed. None of that says whether the representation still *means*
anything. A space can be full-rank and perfectly straight while no longer encoding which
action is happening, or which way it's going. These probes check that, following the
standard frozen-feature protocol from the self-supervised-learning literature: train a
light classifier on the frozen latents and score it on held-out data.

Everything is fit on the **train** split and scored on **val**. For Ego4D GoalStep that
matters twice over: val has only 134 videos, and train and val share no videos, so
fitting on train keeps the probe from matching clips by kitchen or person instead of by
step.

| probe | question |
|---|---|
| [Semantic](#1-semantic-probes) | Can you tell which action or step a clip shows? |
| [Chirality](#2-chirality--arrow-of-time-ssv2-only) | Can you tell which *direction* it goes (Folding vs Unfolding)? |
| [Cross-clip consistency](#3-cross-clip-consistency) | Do clips of the same action move the same way? |
| [Reconstruction](#4-reconstruction-r) | How much of the original DINOv2 feature survives the transformation? |

## Clip vectors

Each clip's per-frame latent `z` (`T=16` frames; the same `z` the progress metrics use)
becomes two vectors:
- **content** = mean over all frames. This is *what* is in the clip. It ignores frame
  order, so a clip and its reverse get the same content vector.
- **change** = mean of the last 4 frames − mean of the first 4 frames. This is *which
  way* the clip moved. Four frames are averaged at each end because per-frame noise is
  heavy (see `direction_reversal_fraction` in `progress_metrics/`).

The semantic probes run on content, change, and both concatenated. The chirality probe uses its own
clip vector instead: all 16 frame latents concatenated (see section 2).

## Labels

| dataset | label | classes | notes |
|---|---|---|---|
| SSv2 | `verb`, e.g. `Folding [something]` | 22 | chiral groups from `chiral_triplet_id` |
| Ego4D GoalStep | leaf step family: the `leaf_step_category` prefix before `:` | 15 | a class is kept only if it has ≥ 20 train and ≥ 5 val clips, which drops "Make coffee" and "Make sushi roll" |

For the semantic probes, train is subsampled to at most 200 clips per class, with a fixed
seed, and every encoder sees the same rows. The chirality probe uses all train clips of
each triplet, as Chirality in Action does.

---

## 1. Semantic probes

### k-NN
**Summary:** do clips of the same class sit near each other? It needs no training, so it
tests the geometry of the space directly.

**How it's computed:**
1. For each val clip, take its k = 20 nearest train clips by cosine similarity.
2. Predict their majority label.

These are the standard settings from DINO onwards. For the concatenated vector, content
and change are each L2-normalized first, so they count equally.

### Linear probe
**Summary:** can the classes be separated by a straight line?

**How it's computed:**
1. Standardize the features.
2. Fit logistic regression with balanced class weights. The regularization strength `C`
   is chosen from {0.01, 0.1, 1} by 3-fold cross-validation on train, scored by balanced
   accuracy.

**Reported:**
- `acc` and `balanced_acc`, the mean of per-class recall, which is robust to class
  imbalance.
- For the linear probe, `balanced_acc_ci95`: a 95% bootstrap interval on balanced accuracy, resampling whole
  videos for Ego4D and clips for SSv2.
- For Ego4D, also `balanced_acc_excl_general`: the same predictions scored only on clips
  whose true class isn't one of the three "General …" catch-all families.
- Chance levels, `1/K` and the majority-class rate, under `chance`.

## 2. Chirality / arrow of time (SSv2 only)

SSv2's chiral triplets pair an action with its time-reverse on the same kind of object:
Folding / Unfolding, Moving up / Moving down, or Opening / Closing a specific object.
Both sides share objects and hands, so telling them apart needs the representation to
encode *which way* things change.

### `concat_probe` (Chirality in Action protocol)
**Summary:** for each chiral triplet, can a classifier tell forward from reverse using
the clip's full sequence of frame latents? This follows Chirality in Action's
chiral-action-recognition evaluation (`external/chirality_in_action/LiFT/train_eval_probe.py`
and `lift/feat_utils.py`).

**How it's computed:**
1. **Clip vector:** concatenate all 16 frame latents into one `16 × D` vector, e.g. 6144
   values for 384-d features and 2048 for the 128-d checkpoint latents. Unlike a mean over
   frames, this keeps frame order.
2. **One classifier per triplet** (`chiral_triplet_id`; 16 in SSv2 val, where each
   Opening/Closing object is its own triplet):
   - Fit a fresh binary classifier to predict `chiral_label` (1 = forward, 0 = reverse)
     on **all** train clips of that triplet. This isn't subsampled, unlike the semantic
     probes.
   - Score accuracy on that triplet's val clips.
   - The default classifier is scikit-learn's `RidgeClassifier()` with default settings
     and no feature scaling, as in Chirality in Action. `--chiral-clf logistic` and
     `--chiral-clf kernel_svm` match its other two options.
   - Triplets with only one side in train are skipped.
3. **`mean_accuracy`:** the mean over triplets, each triplet weighted equally. Chance is
   about 0.5. `per_triplet` lists each triplet's verbs, accuracy and clip counts.
4. **`mean_reversal_flip`:**
   - For each triplet, take the val clips its classifier got right.
   - Reverse the order of their 16 frames. This is exact for per-frame encoders, where
     reversing the video just reverses `z`.
   - Re-predict, and count the fraction whose prediction flips to the other side.
   - Average over triplets.

   1.0 means every reversed clip is read as its time-reverse. Because the concatenation
   keeps order, this isn't guaranteed by construction.

## 3. Cross-clip consistency

Planning across videos needs clips of the same action to move the same way, not just
each clip to be internally ordered.

### `change_direction`
**Summary:** do same-label clips move in the same direction?

**How it's computed:** the mean cosine between the change vectors of pairs of val clips
from different videos. It's reported for same-label pairs (`within_label_cos`) and
different-label pairs (`between_label_cos`). `gap` is their difference, and 0 means
direction carries no label information.

### `tcc_kendall_tau`
**Summary:** can two clips of the same action be aligned frame by frame, in order? This
is the Kendall's tau measure from Temporal Cycle-Consistency (TCC; Dwibedi et al., 2019).

**How it's computed:**
1. Take 2000 random same-label clip pairs (A, B) from different videos.
2. Subtract each clip's own mean latent from all its frames.
3. Map each of A's 16 frames to its nearest frame of B by Euclidean distance.
4. Compute Kendall's tau between A's frame order and the matched indices.

1.0 means the clips align in order, and about 0 means there's no temporal alignment.

Step 2 matters. Two clips of the same verb still differ a lot in overall appearance
(object, background, lighting), and that difference dwarfs the movement within a clip.
Without centering, every frame of A just matches whichever frame of B looks most like A
overall. On raw DINOv2 patch features, 25% of pairs matched all 16 frames to a *single*
frame of B, and the median pair used only 2 distinct frames. Centering compares where
each frame sits *within its own clip's trajectory*. TCC gets the same effect by training
its encoder for alignment.

**Also reported:**
- `degenerate_fraction`: pairs where every frame of A matches one frame of B. Tau is
  undefined there and counted as 0.
- `median_distinct_matches`: how many of B's 16 frames get used.

**Verified on synthetic data:** clips that share a progression but carry a large per-clip
appearance offset give tau = 0.99 (15 of 16 frames used) and a large gap. i.i.d. noise
gives tau ≈ 0 and a gap ≈ 0.

## 4. Reconstruction R²

Output key: `reconstruction_r2`.

**Summary:** how much of the raw DINOv2 per-frame feature can an MLP decoder recover from
the latent? This is a lightweight version of the information-sufficiency check in
*Temporal Straightening for Latent Planning*, which trains an image decoder. An MLP
rather than a linear map separates "the information is gone" from "the information is
there but not linearly readable".

**How it's computed:**
1. Every frame is one sample. The input is its latent (128-d for the patch-channel
   checkpoints). The target is its raw DINOv2 feature: the mean over the 196 patch tokens
   for patch features (384-d).
2. Standardize inputs and targets using train statistics.
3. Train an MLP decoder: 2 hidden layers of 512 units with GELU, AdamW (learning rate
   1e-3, weight decay 1e-4), batch size 1024, mean-squared-error loss.
4. Early stopping: 10% of train *clips* are held out for it, not frames, so frames of one
   clip never sit on both sides of the split. Stop after 10 epochs without improvement
   (at most 200), and keep the best weights.
5. Report `R²` on val frames, in the original target units, variance-weighted over the
   384 outputs.

1.0 means nothing was lost.

**Checked on synthetic cases:**

| input | R² |
|---|---|
| the target itself | 0.995 |
| pure noise | −0.03 |
| a random 128-d projection of the target | 0.75 |

The random-projection value is the reference for what a 128-d latent could keep. (A
linear ridge decoder, used in an earlier version, gives 0.68 on the random projection.)

**Checkpoints only.** For a raw encoder the latent *is* the DINOv2 feature.

---

## Current results

**SSv2 val** (22 verbs; chance 0.045 uniform, 0.128 majority). All numbers are
balanced accuracy of the linear probe unless noted. The raw patch mean-pool row is the
fair baseline for the checkpoints, because it uses the same `dinov2_noreg` patch features
they were trained on.

| encoder | content | change | content+change (95% CI) | k-NN c+c | chirality acc | reversal flip | change gap | TCC τ | recon R² |
|---|---|---|---|---|---|---|---|---|---|
| raw DINOv2 CLS | 0.252 | 0.368 | 0.524 (0.48–0.57) | 0.343 | 0.799 | 0.715 | 0.039 | 0.066 | — |
| raw DINOv2 patch mean | 0.238 | 0.411 | 0.489 (0.44–0.53) | 0.371 | 0.776 | 0.734 | 0.063 | 0.130 | — |
| sw0.5 | 0.151 | 0.195 | 0.232 (0.20–0.27) | 0.210 | 0.627 | 0.562 | 0.018 | 0.038 | 0.074 |
| sw2.0 | 0.177 | 0.209 | 0.261 (0.22–0.31) | 0.214 | 0.634 | 0.572 | 0.092 | 0.095 | 0.112 |

Chirality accuracy and reversal flip come from the per-triplet concatenated-frame
ridge probe (mean over 16 triplets; chance about 0.5). Reconstruction R² uses the MLP decoder. A random 128-d projection of the DINOv2
feature gives 0.75, which is the reference for what a 128-d latent could keep. (The
earlier linear ridge decoder gave 0.028 for sw0.5 and 0.104 for sw2.0.)

**Ego4D GoalStep val** (raw DINOv2 CLS only; 15 step families; chance 0.067 uniform,
0.160 majority):

| | content | change | content+change (95% CI) | excluding "General …" (c+c) | change gap | TCC τ |
|---|---|---|---|---|---|---|
| raw DINOv2 CLS | 0.411 | 0.113 | 0.392 (0.34–0.42) | 0.427 | 0.001 | 0.003 |

## Usage

```bash
# raw DINOv2 CLS on SSv2
/home/sachish/.conda/envs/lift/bin/python compute_probes.py --dataset ssv2 --raw cls \
    --train-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt \
    --train-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \
    --val-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt \
    --val-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \
    --out results/ssv2_raw_dinov2_cls.json

# trained checkpoint: use --checkpoint/--config and the patch features it was trained on
/home/sachish/.conda/envs/lift/bin/python compute_probes.py --dataset ssv2 \
    --checkpoint ../../predictive_world_model/checkpoints/<run>/best.ckpt \
    --config ../../predictive_world_model/checkpoints/<run>/config.yaml \
    --train-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_train.pt \
    --train-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \
    --val-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt \
    --val-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \
    --out results/ssv2_<run>.json
```

- **Ego4D GoalStep:** `--dataset ego4d` with the `ego4dgoal` features and manifests.
  Only DINOv2 CLS features exist for Ego4D, so patch-channel checkpoints can't be probed
  on it yet.
- **Sharded features:** the loader memory-maps sharded feature files
  (`ssv2_train.pt.shard*.pt`) and reads only the clips it needs. Encoding runs 1500
  clips at a time, so even the full SSv2 train split (59 GB of patch features, needed
  by the chirality probe) never sits in memory at once.
- **Caching:** encoded latents are cached in `cache/`, one file each for the train
  subsample, the full train split and val, so reruns skip encoding.
- **Other flags:** `--chiral-clf` (`ridge` by default, or `logistic` / `kernel_svm`),
  `--max-train-per-class`, `--k`, `--edge-frames`, `--tcc-pairs`, `--bootstrap`, `--seed`.
