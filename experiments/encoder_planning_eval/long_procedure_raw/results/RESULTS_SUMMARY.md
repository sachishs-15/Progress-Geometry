# Raw DINOv2 long-procedure analysis: consolidated findings

## What was run

Only the existing frozen raw DINOv2 features and Ego4D Goal-Step annotations were used. No project-trained checkpoints or predictive models were used. Each clip descriptor is the mean of its 16 sampled 384-D frame features; PCA is only for visualization.

| Split | Videos | Annotated clips | Relevance unlabeled |
|---|---:|---:|---:|
| Validation | 134 | 6,732 | 2,388 |
| Train | 583 | 27,600 | 9,381 |

Every manifest row matched a feature row. Positive gaps between observed clips sum to 7.6 hours in validation and 35.2 hours in train. The sequences are sparse; dashed plot edges indicate gaps over one second.

## Findings

### 1. Raw clip trajectories are quite curved

For essential clips, median raw chord efficiency is 0.099 in validation and 0.100 in train (1.0 is a straight path). Median consecutive-velocity cosine is -0.359 and -0.355. The relevance groups have similar clip-level values.

For parent steps represented by at least three observed clips, median cosine to that step’s feature centroid is 0.928 on validation and 0.934 on train. Median sparse path efficiency is 0.342 and 0.328. This parent-step path metric includes missing intervals, so treat it as descriptive of sampled clips, not continuous motion.

### 2. Feature jumps are larger at known step boundaries

Among adjacent clips separated by no more than one second:

| Split | Transition | n | Median feature L2 | Median cosine |
|---|---|---:|---:|---:|
| Validation | within_parent | 2,474 | 10.135 | 0.875 |
| Validation | parent_change | 1,672 | 12.193 | 0.807 |
| Train | within_parent | 10,633 | 10.051 | 0.881 |
| Train | parent_change | 5,932 | 11.898 | 0.816 |

The two transition groups have similar gap distributions after filtering. The difference is descriptive; it suggests the frozen features register parent-step changes, but does not show that straightness causes better step recognition.

### 3. Actual order scores above shuffled order

The signed score correlates parent-step order with projection on the chronological start-to-end feature direction. Shuffled sequences use the same direction. Results are means over videos with 95% video-bootstrap intervals.

| Split | Scope | Beats reversed | Beats random shuffles | Videos |
|---|---|---:|---:|---:|
| Validation | all_parent_events | 0.967 [0.935, 0.992] | 0.918 [0.888, 0.946] | 123 |
| Validation | essential_only | 0.989 [0.968, 1.000] | 0.906 [0.876, 0.933] | 95 |
| Train | all_parent_events | 0.957 [0.940, 0.974] | 0.890 [0.876, 0.904] | 531 |
| Train | essential_only | 0.994 [0.986, 1.000] | 0.918 [0.906, 0.929] | 361 |

This is evidence that the frozen sequence has a temporal appearance signal. Camera, scene, or object-state changes could explain part of it, so it is not by itself evidence that the representation understands procedure order.

### 4. Feature similarity helps over random retrieval, but not reliably over chronology

For each query, the target is the next essential parent step; candidates are later parent events from the same video. The feature scorers do not receive timestamps. Scores are video-macro means with 95% video-bootstrap intervals.

| Split | Scorer | MRR | Recall@1 | Recall@5 |
|---|---|---:|---:|---:|
| Validation | Current-step cosine | 0.501 [0.456, 0.547] | 0.329 [0.274, 0.386] | 0.723 [0.669, 0.776] |
| Validation | Earliest-future baseline | 0.549 [0.503, 0.594] | 0.349 [0.292, 0.407] | 0.828 [0.778, 0.874] |
| Validation | Expected random rank | 0.316 [0.284, 0.349] | 0.139 [0.117, 0.162] | 0.517 [0.462, 0.573] |
| Validation | Linear extrapolation | 0.445 [0.396, 0.497] | 0.273 [0.213, 0.338] | 0.663 [0.602, 0.721] |
| Train | Current-step cosine | 0.520 [0.497, 0.544] | 0.353 [0.325, 0.383] | 0.734 [0.709, 0.760] |
| Train | Earliest-future baseline | 0.535 [0.513, 0.559] | 0.324 [0.294, 0.356] | 0.833 [0.808, 0.857] |
| Train | Expected random rank | 0.316 [0.301, 0.332] | 0.138 [0.127, 0.149] | 0.519 [0.493, 0.546] |
| Train | Linear extrapolation | 0.484 [0.459, 0.510] | 0.315 [0.285, 0.346] | 0.697 [0.667, 0.727] |

Cosine and extrapolation beat random ranking, but the earliest-future baseline has higher MRR and Recall@5 on both splits. Cosine has a small Recall@1 advantage over earliest-future on train only. The raw feature geometry offers some candidate signal, but does not consistently beat the order cue.

## Artifacts

- Split-level reports: `val_raw_dinov2/FINDINGS.md` and `train_raw_dinov2/FINDINGS.md`.
- Each split folder has the sequence, geometry, transition, order, and retrieval CSVs; reusable pooled step vectors; JSON summaries; and eight annotated PCA plots.
- Reproduction script and commands: [`../analyze_raw_ego4dgoal.py`](../analyze_raw_ego4dgoal.py) and [`../README.md`](../README.md).

## What this establishes

This is a raw-feature baseline for long procedural sequences. It does not establish that induced straightness improves long-horizon performance. The next controlled experiment should compare raw DINOv2 with a trained representation using the same retrieval queries and the earliest-future baseline, while checking for feature collapse.
