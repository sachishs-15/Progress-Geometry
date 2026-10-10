**My assessment: local temporal straightening is a plausible regularizer, but the claim that a complete Ego4D Goal-Step procedure should follow one straight, goal-directed feature trajectory is poorly supported. The available trained run also fails to establish that straightening improves procedure understanding.**

This assessment uses fresh diagnostics on all 6,732 cached validation clips from 134 videos, dense 4-fps CLS and mean-patch features covering 904,881 timestamps per pooling, six visually inspected video examples, and checks of existing parent-step and trained-model evaluations. It does not train a new model or re-extract the features. 130 of the 134 videos have cooking goal categories; these findings concern the local step-annotation subset, rather than all scenarios in Ego4D.

The original perceptual-straightening work concerns a relative transformation of visual trajectories in human perception, not a guarantee of a globally straight path to a procedural goal. [Hénaff et al.](https://www.nature.com/articles/s41593-019-0377-4) LiFT describes its adaptation as loosely inspired by that work and evaluates time-sensitive representations, including opposite actions. That does not establish transfer to long procedural planning. [LiFT authors' project page](https://bpiyush.github.io/lift-website/)

**What the videos show.** The [contact sheet](results/video_contact_sheet.jpg) contains essential annotated clips selected near the median path efficiency for six description groups, restricted to 3–60 seconds. These are illustrative samples, not a prevalence estimate. Stirring potato salad repeatedly changes hand position while preserving the overall scene. Cutting dough increases the number of pieces while the hands repeatedly return to similar poses. Washing a cucumber includes approaching the sink and changing viewpoint. The onion-peeling clip ends after a large turn away from the work surface. Treating that last image as the semantic goal would primarily supervise viewpoint change.

The [cutting example](results/90e4acc5-28a2-4bd4-972c-c6f7e18e1cde__0061_geometry.png) makes the distinction particularly clear: endpoint-projection progress has interior Spearman correlation 0.864 with time, yet raw path efficiency is only 0.089. There can be a useful progress component inside a highly curved representation. Requiring the entire vector to straighten is a substantially stronger restriction than recovering that component.

**Fresh tests of the 16-frame cache.** Path efficiency is endpoint displacement divided by summed path length; 1 means a straight, forward path. All measurements use the original 384-dimensional space. Forty controls per clip shuffle only the fourteen interior frames, keeping endpoints fixed. Interpolation errors exclude endpoints, and progress correlations are also reported without endpoints to reduce automatic endpoint anchoring. Intervals bootstrap video-level means over 134 videos.

| Measurement | Observed result | Interpretation |
|---|---|---|
| Raw path efficiency | Median 0.0988; video-macro mean 0.1018 | Frozen trajectories are far from globally straight. |
| Efficiency gain over shuffled interiors | Mean +0.0198, 95% CI [0.0187, 0.0208] | Chronological order has real geometric structure. |
| Consecutive velocity cosine | Median −0.3557; mean gain over shuffle +0.1150 | Chronology is smoother than shuffle, while successive movements still strongly change direction. |
| Interior endpoint-projection progress correlation | Video-macro mean 0.3796, CI [0.3493, 0.4082]; shuffle approximately zero | There is temporal appearance information, with considerable nonmonotonicity. |
| Interior correlation of decreasing distance to final frame with time | Video-macro mean 0.2272 | Appearance distance to the endpoint is a weak progress proxy. |
| Uniform endpoint interpolation beats constant endpoint midpoint | 23.2% of clips; video-macro fraction 21.6% | Endpoints do not generally specify intermediate features through uniform linear interpolation. |

The interpolation result tests a stronger constant-speed assumption than the adjacent-velocity loss. It should not be treated as proof against every possible straightening objective. A nonlinear learned representation could behave differently. The cache has no extraction provenance metadata, although its entries are finite and no clip is constant across all sixteen frames. These checks cannot certify every frame's extraction history.

Essential and irrelevant leaves have nearly identical median path efficiencies (0.0981 and 0.0988). Their geometry remains similar within broad duration strata. This is descriptive evidence that generic geometric smoothness does not uniquely identify goal-relevant activity; the labels and group differences do not form a causal comparison.

**Time scale matters, but denser sampling does not remove the problem.** Clips range from 0.492 to 2,711.944 seconds, with median duration 10.932 seconds. Sixteen uniform samples therefore put consecutive observations about 0.729 seconds apart at the median, versus about 180.8 seconds in the longest clip. Applying the same curvature objective to both mixes very different prediction problems.

The dense cache uses no-register DINOv2 ViT-S/14, 196-pixel squash resizing, and 4-fps full-video sampling. The legacy clip cache uses another extraction setup; comparisons below vary spacing within the dense cache and do not attribute differences between caches to sampling.

| Dense frame spacing | Within-leaf CLS velocity cosine, video-macro mean |
|---|---:|
| 0.25 seconds | −0.3138 |
| 0.5 seconds | −0.3474 |
| 1 second | −0.3807 |
| 2 seconds | −0.4105 |
| 4 seconds | −0.4281 |

All three timestamps must belong to the same annotated leaf; zero-length movements are excluded. Mean-patch features give the same qualitative pattern. Excluding the twelve videos with more than 1% exactly duplicate adjacent embeddings leaves essentially the same curve. Within a leaf is not the same as within a single primitive movement, so this does not refute straightening at still finer scales. Negative velocity cosine can also arise from frame-dependent nuisance variation, not only physical reversal. [Lag plot](results/dense_lag_curves.png)

**Long procedures contain repeated steps and only partial order evidence.** Essential parent-step categories repeat in 80 of 120 videos with labeled essential parents. This establishes repeated category instances, not identical world states or zero semantic progress. The French-toast example alternates repeated toasting and serving instances. A procedure representation needs to preserve which instance has occurred and what has already been accomplished.

The earlier parent-order report compared sequences against shuffles including their endpoints, even though the chronological endpoints define its axis. A fresh control fixes those endpoints and scores interior events only. For essential sequences with at least five events, mean interior rank correlation is 0.493 across 71 videos. Real order beats 75.9% of shuffled interiors on average; 36.6% of videos have an uncorrected one-sided permutation p below 0.05. Short sequences have limited permutation-test power. Temporal signal survives the control, but the stronger original order percentages should not be interpreted as procedural understanding.

Using the previous retrieval queries, I also compared methods on the same 617 queries from 92 videos where extrapolation is defined. Mean reciprocal rank is 0.445 for raw linear extrapolation, 0.499 for current-step cosine, and 0.524 for selecting the earliest future candidate. Extrapolation minus earliest-future is −0.0793, with paired video-bootstrap CI [−0.1291, −0.0283]. The candidate pool already consists of future events from the same video; this is a retrieval diagnostic, not open-ended planning. It supplies no evidence that raw linear extrapolation is an effective procedure planner.

**The existing patch-world-model result is a failure case.** I read the saved full-validation evaluations for the 16-frame model trained with prediction, straightening weight 0.1, and token-level VICReg. These are existing artifacts, not a fresh checkpoint evaluation.

| Mean metric | Raw no-register mean-patch baseline | Trained aggregate representation |
|---|---:|---:|
| Raw-space path efficiency | 0.1067 | 0.1832 |
| Time/projection correlation | 0.6053 | 0.2796 |
| Adjacent temporal nearest-neighbor fraction | 0.5917 | 0.2783 |
| Effective rank / output dimensionality | 69.3% | 1.38% |

The trained aggregate has effective rank 1.76 in 128 dimensions. Straighter geometry coexists with severe dimensional concentration and worse temporal diagnostics. Because the baseline and trained output also differ in architecture and dimensionality, and there is no matched zero-straightening ablation here, this does not identify straightening as the cause. It does invalidate using improved straightness alone as evidence that this run learned useful procedural progress.

The adjacent-velocity loss is also exactly invariant to reversing time: reversal negates both vectors in each cosine and reverses the list of terms. Endpoint path efficiency has the same invariance. A directional prediction or ordered supervision objective must provide the arrow of time. Endpoints being chronological does not make Euclidean distance to them a semantic completion measure.

**What I would pursue.** Keep a history-aware task-state representation with goal/subgoal conditioning. Test weak local curvature regularization within fixed-duration windows, allowing changes of direction at step boundaries and preserving cyclic manipulation. A separate progress head can exploit a monotonic component without making every feature dimension encode progress. Its supervision should distinguish elapsed time from actual step or goal completion.

The decisive experiment is a matched prediction-and-anti-collapse baseline with straightening weight zero versus several nonzero weights, identical architecture, sampling, budget, and multiple seeds. Assess held-out next-step recognition or retrieval, step grounding, state preservation, and effective rank alongside curvature. Include chronology, temporal smoothing, and random-order controls. Only a downstream gain without loss of state information would validate straightening for this dataset and formulation.

The dataset's hierarchy, relevance labels, and completion/continuation annotations support this distinction between atomic motion and goal-oriented state. [Ego4D Goal-Step paper](https://papers.neurips.cc/paper_files/paper/2023/file/7a65606fa1a6849450550325832036e5-Paper-Datasets_and_Benchmarks.pdf)

Reproduction: run `analyze.py`, `parent_controls.py`, then `check_saved_results.py` with `/home/sachish/.conda/envs/lift/bin/python`, preferably setting `OPENBLAS_NUM_THREADS=2`. The `results/` directory contains clip diagnostics, dense per-video statistics and integrity checks, endpoint-controlled parent tests, illustrations, and compact JSON summaries. Original data, model code, and checkpoints were not changed.
