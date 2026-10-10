# Does temporal straightening fit Kinetics-400 and Ego4D GoalStep?

**Audit date:** 9 October 2026. **Project context:** procedural progress estimation and latent planning in `proc-planning`. This report separates *a video having a meaningful task progression* from *its frozen visual trajectory already being a straight path*. Those are different claims.

## Main conclusion

Straightening is defensible for a **task-conditioned, state-changing segment**, especially a leaf step such as peeling, cutting, adding, arranging, or serving. It is a poor blanket objective for every Kinetics clip or an entire Ego4D recording. A Kinetics label often names a repeated motion, one event, or travel through a scene. Ego4D long recordings often contain a real multi-step procedure, but revisit the same sink, counter, tools, and camera pose between different subgoals. Their latent trajectory is closer to a branching or piecewise sequence than one line.

The observed frozen geometry reinforces this distinction. On 9,937 usable, class-balanced Kinetics clips, the median start-to-end chord divided by path length is **0.104 for DINOv2 CLS** and **0.129 for V-JEPA 2.1 mean-patch** (1 is straight), although median time-order correlation is **0.824/0.832**. On 6,732 Ego4D GoalStep validation leaf clips, raw DINOv2 CLS chord efficiency is **0.099**, and across 128 whole-recording sequences of annotated clip centroids it is **0.036**. An unsupervised 64-D PCA projection modestly raises Kinetics chord efficiency, but does not make most trajectories line-like. An existing Ego4D straightening-only learned projector barely improves raw chord efficiency and sharply reduces representation rank and time-order quality.

![Manual review breakdown](manual_review_breakdown.png)

## What was inspected

| Evidence | Coverage | Sampling and interpretation |
|---|---:|---|
| Kinetics-400 raw videos | 100 clips, one video from each of 100 randomly selected action classes | Six evenly spaced frames per clip were visually reviewed. Because Kinetics classes are nearly balanced, this is useful for a class-level audit. It is still a sample, not a census of 241,258 clips. |
| Ego4D GoalStep leaf clips | 80 random validation clips from 55 recordings | Six frames per clip were reviewed. The validation manifest has 6,732 leaf clips from 134 recordings; this percentage sample is uniformly drawn from clips. |
| Ego4D whole recordings | 24 GoalStep validation recordings, spread across goal categories | Six widely spaced frames per recording and their goal labels were reviewed. This is a qualitative stage check; these long videos were not watched frame by frame. |
| Frozen feature geometry | 9,937 Kinetics clips per encoder; all 6,732 Ego4D validation leaf clips in DINOv2 CLS | Existing 16-frame feature files; 63 exact-static Kinetics entries were removed after checking that example source MP4s fail decoding. |
| Extra matched encoder check | 80 reviewed Ego4D leaf clips | V-JEPA 2.1 mean-patch features were extracted separately with the existing read-only precompute command. It is a small-sample comparison, not a full Ego4D feature store. |

The Kinetics manifest covers 241,258 training videos and 400 labels. The Ego4D GoalStep manifests cover **34,332 annotated leaf clips in 717 recordings** (27,600 train/583 recordings and 6,732 validation/134 recordings). Validation is **6,707/6,732 cooking clips**; this audit should not be generalized to all of Ego4D's activities. Median distinct parent-step counts are 16 per train recording and 17 per validation recording; 91.1% and 91.8% respectively have at least three annotated parent steps. These annotations establish *candidate task structure*, not automatically visible monotone state.

### Review rubric

An **S** clip has a visibly accumulating or persistent target-state change across the sampled window. **P** means a task or possible state change is present but six frames do not reveal a trustworthy progress variable (for example cleaning, heating, mixing, or a very short step). **T** is a one-shot event or travel with a temporal phase but no durable task-state coordinate. **R** is repeated or reversible motion without visible accumulating state. **U** is too weakly covered, dark, cut, or mismatched to the action label to judge. The strict support rate is S. T and R are the clear counterexamples for a global persistent-state straightening loss. P and U remain unresolved; assigning them to either side would overstate the video evidence.

| Dataset reviewed | S: visible state progress | P: possible/hidden | T: transient/travel | R: repeated/reversible | U: unassessable | Clear counterexamples T+R |
|---|---:|---:|---:|---:|---:|---:|
| Kinetics, 100 class-balanced clips | **12 (12%)** | 18 (18%) | 31 (31%) | 27 (27%) | 12 (12%) | **58 (58%)** |
| Ego4D GoalStep, 80 random leaf clips | **25 (31.3%)** | 34 (42.5%) | 2 (2.5%) | 14 (17.5%) | 5 (6.3%) | **16 (20%)** |

For Kinetics S, a simple 95% Wilson interval is **7.0–19.8%**; for T+R it is **48.2–67.2%**. For Ego4D, bootstrapping its 55 sampled recordings gives approximately **20.3–43.2%** for S and **11.1–28.6%** for T+R. Those intervals measure sampling uncertainty, not the subjectivity of a six-frame human judgment. In the separate 24 whole-recording review, **18/24** displayed multiple plausible stages and **6/24** did not show a clear state endpoint at the sampled times. That 18/24 is descriptive for a goal-spread sample; it is not a population estimate of globally straight videos.

### Annotated video examples

The saved sheets label each row with its original action and six sample times. The full per-video classifications and paths are in the linked CSVs below.

| Case | Observation in sampled video | Implication |
|---|---|---|
| [Kinetics 044: peeling apples](kinetics/sheet_04.jpg) | Peel is removed and discarded over the clip; the apple's surface changes persistently. | Strong candidate for a state-progress coordinate, though hand motion and camera still bend the frozen path. |
| [Kinetics 066: pumping gas](kinetics/sheet_06.jpg) | Nozzle inserted, fuel dispensed, and a volume display appears around 7.6 seconds. | A measurable monotone variable exists; a visual latent may need instrument or state attention. |
| [Kinetics 073: making pizza](kinetics/sheet_07.jpg) | More toppings appear on the base from about 0 to 9.5 seconds. | Accumulation is meaningful and local straightening is plausible. |
| [Kinetics 065: push-ups](kinetics/sheet_06.jpg) | Body repeatedly returns to an earlier pose. | A state-only image embedding cannot put repeated identical poses at increasing progress without history or a repetition count. |
| [Kinetics 080: answering questions](kinetics/sheet_08.jpg) | Interview cuts to a broadcast title card halfway through. | Straightening could learn the edit or graphic transition instead of an action state. |
| [Kinetics 051: laying bricks](kinetics/sheet_05.jpg) | Sample shows a brick-loading machine and a finished pavement shot, with no observed laying transition. | Label and coverage do not warrant a straightness target. |
| [Ego4D leaf 025: dicing celery](ego_leaf_uniform/sheet_02.jpg) | Chopped pieces accumulate by about 29 seconds. | An object-centric local progress coordinate makes sense. |
| [Ego4D leaf 039: forming round batter pieces](ego_leaf_uniform/sheet_03.jpg) | A tray fills with circular pieces by about 54 seconds. | Strong state accumulation despite repetitive hand motion. |
| [Ego4D leaf 011: stirring sauce](ego_leaf_uniform/sheet_01.jpg) | Repeated stirring around a nearly unchanged pan for about 20 seconds. | Motion phase cycles; any cooking progress is hidden from these sparse images. |
| [Ego4D whole 017: sushi roll](ego_whole/sheet_01.jpg) | Setup, work at the rolling mat, and later kitchen cleanup appear at widely separated times. | A multi-stage procedure exists, but revisits and intervening actions should be modeled piecewise. |
| [Ego4D whole 014: study/write](ego_whole/sheet_01.jpg) | Computer screens change across a long session, without a visible finished state. | Chronology is observable; task completion cannot be inferred from the sparse frames. |

## Frozen geometry: order is not straightness

`Straightness` below is raw Euclidean endpoint displacement divided by total 16-frame path length. `Time rho` is Spearman correlation between frame index and position on the *unit-normalized* start-to-end chord; it is endpoint-biased, so a shuffled-interior control is reported. `Velocity cosine` is the mean cosine between consecutive raw steps; negative values imply local reversals. `Interpolation R²` compares true intermediate vectors with linear start/end interpolation, normalized by within-clip variation; negative means the line reconstructs intermediates worse than the clip mean. `≥0.3` is an **illustrative operational threshold**, not a definition of semantic truth.

| Dataset / frozen representation | n | Median straightness | Median time rho | Median velocity cosine | Median interpolation R² | Straightness ≥0.3 | Joint rho ≥0.8 and straightness ≥0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kinetics DINOv2 CLS, 384-D | 9,937 | 0.104 | 0.824 | −0.335 | −0.378 | 0.7% | 0.4% |
| Kinetics V-JEPA 2.1 mean-patch, 768-D | 9,937 | 0.129 | 0.832 | −0.243 | −0.321 | 4.6% | 2.6% |
| Ego4D leaf DINOv2 CLS, 384-D | 6,732 | 0.099 | 0.650 | −0.356 | −0.668 | 0% | 0% |

Kinetics median time rho falls to **0.33 DINOv2 / 0.31 V-JEPA 2.1** when the 14 interior frames are shuffled while retaining the endpoints. Thus there is temporal information in the originals. But high time rho by itself does not imply a short Euclidean path. Even among the human-reviewed Kinetics clips, S median straightness was only **0.128 DINO / 0.164 V-JEPA**, versus **0.109 / 0.141** for R. These small and inconsistent gaps mean raw straightness is a weak semantic classifier on this review sample; an action may truly progress while camera and hand motion dominate, and a non-progressive shot can have an ordered appearance drift.

On the same 9,937 Kinetics clip centroids, nearest-neighbor retrieval of another video with the same action label is **42.1% DINOv2 CLS** versus **20.5% V-JEPA 2.1 mean-patch**, against about **0.25%** label-frequency chance in this class-balanced sample. V-JEPA's modest geometric advantage therefore does not imply more useful action separation under these particular pooling choices. This is an encoder comparison, not evidence that applying a straightening loss caused the difference.

![Frozen latent metric distributions](latent_metric_distributions.png)

![Four reviewed clips with actual latent progress and goal-distance curves](annotated_latent_examples.png)

### Matched V-JEPA 2.1 check on the reviewed Ego4D clips

The 80 randomly reviewed Ego4D clips were re-encoded with the project's V-JEPA 2.1 plugin, using the same 16 linearly sampled frames per clip and mean-pooled patch tokens. There were **80/80 successful extractions**. This paired subset is too small for the cross-recording topology analysis above, but it checks whether the semantic review looks materially different in this encoder.

| Frozen representation, same 80 clips | Median raw straightness | Median time rho | Median velocity cosine | Median interpolation R² | S straightness / R straightness |
|---|---:|---:|---:|---:|---:|
| DINOv2 CLS | 0.098 | 0.571 | −0.355 | −0.762 | 0.101 / 0.101 |
| V-JEPA 2.1 mean-patch | 0.122 | 0.579 | −0.266 | −0.885 | 0.132 / 0.123 |

V-JEPA 2.1 is somewhat less zig-zagged in this paired set, yet its straight-line interpolation is not better under the within-clip R² measure, and neither encoder's raw straightness cleanly separates visibly cumulative steps from repeated actions. The V-JEPA result is a **sample result only**; it should not be cited as a full Ego4D model comparison. The example plot above now overlays both encoders for the two Ego4D review clips as well as the two Kinetics clips.

### Does a different *unsupervised projected* space help?

A 64-D PCA was fitted to 2,500 Kinetics clip centroids per encoder and evaluated on the **7,437 held-out usable clips**. It retained about **63%** of DINOv2 and **80%** of V-JEPA 2.1 centroid variance. Matched held-out raw versus projected results:

| Encoder | Raw → PCA64 median straightness | Raw → PCA64 median time rho | Raw → PCA64 share straightness ≥0.3 | Raw → PCA64 median interpolation R² |
|---|---:|---:|---:|---:|
| DINOv2 CLS | 0.104 → 0.109 | 0.824 → 0.774 | 0.7% → 1.5% | −0.379 → −0.360 |
| V-JEPA 2.1 mean-patch | 0.129 → 0.145 | 0.832 → 0.788 | 4.5% → 9.4% | −0.321 → −0.291 |

This is some geometric gain from removing dimensions, but most trajectories remain far from lines, and temporal order weakens. A trained nonlinear, task-conditioned projection may do better. It must be judged jointly on semantic state, across-video transfer, and non-collapse. An unconstrained projection can make any trajectory look straighter by suppressing action information or mapping everything close to a point. A pure state-only projection cannot give a monotone coordinate to identical revisited states; it needs phase, history, or a count.

### Existing learned-projection control in this project

The repository already contains a directly relevant **straightening-only** DINOv2 CLS projector, trained on Ego4D GoalStep train clips and evaluated on the same 6,732 validation clips. Its [training configuration](../../experiments/ordinal_objectives/checkpoints/ego_straighten_only/config.yaml) uses a 384-D input and 384-D latent, with `straighten: 1.0` as its only objective and no anti-collapse term. The existing [raw baseline](../../experiments/encoder_planning_eval/progress_metrics/results/ego4dgoal_val/baseline_raw_dinov2.json) and [projector evaluation](../../experiments/encoder_planning_eval/progress_metrics/results/ego4dgoal_val/ego_straighten_only.json) were generated before this audit. This is a useful outcome of an actual learned projection, though it tests that specific loss and training run rather than all possible projected spaces.

| Validation metric, median unless stated | Frozen DINOv2 CLS | Straightening-only learned projector |
|---|---:|---:|
| Raw-coordinate chord efficiency | 0.099 | 0.104 |
| Clip-normalized chord efficiency | 0.099 | 0.095 |
| Time-order rho | 0.650 | 0.482 |
| Distance-to-goal rho | 0.254 | 0.104 |
| Raw adjacent-velocity cosine | −0.356 | −0.165 |
| Within-clip mean pairwise L2 | 15.73 | 8.08 |
| Effective rank across all validation frame latents, out of 384 | 293.9 | 16.6 |

The learned head makes successive raw velocity directions less opposed, but does **not** materially raise endpoint-to-path straightness. Clip-normalized straightness even falls slightly. Time order and goal-distance consistency worsen, while effective rank drops by about **94%**. The configuration explicitly allows a collapsed solution, so the rank result is a failure mode of *straightening alone*, not evidence that all learned projections fail. It is a strong reason to require anti-collapse and task-retention checks before crediting a straighter-looking trajectory as useful progress.

## How Ego4D clips and recordings are arranged in latent space

This section uses all **6,732 validation leaf clips**, each represented by the mean of its 16 frozen DINOv2 CLS vectors. Quantitative comparisons use the original 384-D space; the 2-D PCA figure retains only **20.1%** of variance and is illustrative. Action identity is the annotated leaf category (406 categories); parent-step and goal categories are coarser. The validation set is mostly cooking, so these are primarily distinctions *within cooking*.

| Nearest-neighbor or pair result | Measured value | Interpretation |
|---|---:|---|
| Nearest clip is from the **same recording** | **94.7%** | Recording, camera, kitchen, and participant appearance dominate local neighborhoods. |
| Same leaf category, nearest neighbor unrestricted | 39.7% | Inflated by the same-recording structure. |
| Same leaf category, **other recordings only** | **7.3%** versus 0.66% frequency baseline | Cross-recording action signal exists, but is much weaker than recording identity. |
| Same parent category, other recordings only | 6.2% versus 0.69% baseline | Coarser step semantics also transfer weakly. |
| Same goal category, other recordings only | 18.1% versus 4.7% baseline | Broad goal/context has some cross-recording separation. |
| Clip-centroid cosine, same leaf / different leaf, both cross-recording | **0.533 / 0.419** median | Same action clips occupy relatively similar *regions*, with substantial overlap. |
| Start-to-end **direction** cosine, same leaf / different leaf, both cross-recording | **0.005 / 0.001** median | There is effectively no shared global arrow of change for a leaf action. |

The arrangement is best described as **strong recording-specific clusters with weaker action/goal neighborhoods across recordings**. The same action in two kitchens does not trace parallel lines in the frozen CLS coordinates. A global action-vector translation assumption is therefore unsupported. A learned projection would need to factor out recording and viewpoint while preserving object state and action stage; action identity alone is not enough to specify start and goal states.

The cosine histograms use up to 100 sampled pairs per leaf category, always from different recordings, plus random different-category cross-recording pairs. Those pair summaries describe a category-spread comparison rather than the prevalence-weighted distribution of every possible pair.

![Cross-recording same-action location and direction distributions](ego_cross_video_action_geometry.png)

![Ego4D GoalStep centroid PCA by broad goal](ego_goal_pca.png)

### Whole-video trajectories

The long-recording analysis orders annotated leaf-clip centroids by timestamp; it does **not** densely encode every frame. In 128 validation recordings with at least three represented clips, median whole-recording chord efficiency is **0.036**, versus **0.029** for endpoint-preserving shuffled clip orders; only **8/128 (6.25%)** real sequences reach 0.2. Median time-order correlation on the video endpoint chord is **0.478**; **81/128 (63.3%)** beat the 95th percentile of 100 endpoint-preserving interior shuffles. Thus the real sequence is more coherent than random order, yet very far from a global line. The clip stream is sparse, and the endpoint chord can capture changes in viewpoint, lighting, or location. It does not establish that one straight path would be useful for planning.

The 24 whole-recording visual review agrees with a **hierarchy**: food preparation, cooking, plating, and cleanup are often recognizable stages, but each stage may involve circular hand motions and returns to the same workspace. Some videos do not show an observable completion state in sparse frames. A whole-video objective should work over subgoal transitions or ordered stage nodes, with separate local state coordinates, rather than penalizing every turn in the full raw visual trajectory.

## What this means for the project

1. **Choose the level of straightening before choosing a loss.** For GoalStep, start with short, state-changing leaf clips and parent-step transitions. Use the goal/step text, object masks, or before/after state to select examples. Do not apply a single full-video chord loss to 20–60 minute recordings. For Kinetics, filter to the relevant subset of cumulative transformations instead of assuming each of 400 action labels is a procedure.
2. **Separate state progress from action phase.** Peeling, slicing, adding, assembling, and serving have persistent state. Push-ups, music, stirring, and travel can have motion phase or position without accumulating task state. A planner may need both a persistent state vector and a cyclical/action-phase variable; a single monotone scalar cannot represent both.
3. **Condition on the task and guard against identity shortcuts.** Same-recording neighborhoods are extremely strong. Train/evaluate cross-recording, compare same action under different kitchens and different actions within the same kitchen, and prevent nuisance removal from collapsing object-state distinctions. Preserve an unprojected or reconstruction/probe branch if a straightening head discards too much semantic content.
4. **Evaluate planning utility, not a pretty line.** Report chord efficiency and time order together with goal-distance rank, interpolation R², step retrieval, action/category probes, effective rank, and within/across-video distances. Include reversed, shuffled, repeated, camera-cut, and static-video controls. The project's earlier next-essential-step retrieval baseline is a useful gate: a projected space should beat chronology and raw CLS on held-out recordings, not merely improve a geometric score.
5. **Use state/phase-aware local objectives.** A segment can have a monotone progress coordinate even while its full visual vector curves. Consider a task-conditioned progress head or waypoint graph with local straightening only between verified state anchors. Permit pauses and phase reversals within a step. For whole recordings, learn transitions between subgoal centroids and use temporal ordering as a separate constraint.

## Limits and reproducibility

- Manual categories are based on six sampled frames, plus action and GoalStep descriptions, not exhaustive video playback. Frames span roughly 3–97% of each video; contact-sheet timestamp labels are approximate. Brief actions between sample times may be missed. The individual annotations are in [Kinetics review](kinetics_review.csv), [Ego4D leaf review](ego_leaf_uniform_review.csv), and [whole-video review](ego_whole_review.csv); [contact sheets](kinetics/) and [Ego4D leaf sheets](ego_leaf_uniform/) permit independent reinspection. A second goal-spread 80-clip Ego4D sheet set is also saved for qualitative examples, but is not mixed into the random-sample percentages.
- The Kinetics frozen files had **63/10,000** sampled clips with exactly static 16-frame latents in both encoders. Direct checks of five corresponding MP4s failed with `moov atom not found`; these 63 were excluded from geometry and retrieval. The older project's [data-integrity note](../../docs/data_integrity.md) warns that some legacy precompute artifacts contain encoded replacement black frames. This audit's 0.63% observation is for the particular 10,000 sampled entries, not a new full-corpus failure estimate.
- DINOv2 CLS and V-JEPA 2.1 mean-pooled patch tokens have different architecture, dimensions, and pooling; V-JEPA 2.1's local plugin has **no CLS token**. The Ego4D V-JEPA extraction covers only 80 reviewed clips. Cross-recording arrangement statistics are DINOv2 CLS only.
- The percentage thresholds are diagnostic choices. In particular, high time rho is partly built into a start-to-end projection; the shuffled-interior null and goal-distance/interpolation checks are essential. PCA plots can distort global neighborhoods and cannot establish topology by themselves.
- Existing project code and source datasets were not modified. New scripts, review labels, feature sample, tables, and plots are confined to this report directory. Exact calculation outputs are [feature_analysis.json](feature_analysis.json), [reviewed_feature_analysis.json](reviewed_feature_analysis.json), [Kinetics per-video geometry](kinetics_dinov2_per_video.csv), [Ego4D per-clip geometry](ego_per_clip_geometry.csv), and [whole-video geometry](ego_whole_geometry.csv). Reproduction scripts: [make_review_sheets.py](make_review_sheets.py), [record_review.py](record_review.py), [analyze_features.py](analyze_features.py), and [analyze_reviewed.py](analyze_reviewed.py).
