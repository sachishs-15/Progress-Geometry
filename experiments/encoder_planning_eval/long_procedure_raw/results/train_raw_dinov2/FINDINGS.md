# Raw DINOv2 long-procedure analysis

## Setup

- Data: Ego4D Goal-Step train; 27,600 clips across 583 videos.
- Features: raw frozen DINOv2, 16 sampled frames per clip, 384-D; clip descriptor is the untrained frame mean.
- No project-trained checkpoints or predictive models were used. PCA is only used for plots.
- Parent-step index: 27,600 clips labeled; 0 without an index. Relevance is unlabeled on 9,381 clips.
- Sum of positive gaps between observed clips: 35.2 hours.

## Results

### Geometry within clips

Raw chord efficiency is endpoint distance divided by total feature-path length (1 means a straight path).
- **essential:** n=16,384; median chord efficiency=0.100; median velocity-turn cosine=-0.355.
- **irrelevant:** n=719; median chord efficiency=0.097; median velocity-turn cosine=-0.351.
- **optional:** n=659; median chord efficiency=0.097; median velocity-turn cosine=-0.351.
- **unsure:** n=457; median chord efficiency=0.099; median velocity-turn cosine=-0.354.
- **unlabeled:** n=9,381; median chord efficiency=0.098; median velocity-turn cosine=-0.357.

### Geometry across multiple clips in a parent step

Singleton parent steps are excluded from within-step cohesion. Sparse path efficiency includes time gaps between observed clips.
- **At least 2 clips:** n=3,777; median cosine to step centroid=0.944; median L2 dispersion=6.307.
- **At least 3 clips:** n=2,467; median cosine to step centroid=0.934; median L2 dispersion=6.733; median sparse path efficiency=0.328.

### Transitions between observed clips

- **parent_change__all (all):** n=12,915; median gap=1.3s; median feature distance=13.139; median cosine=0.766.
- **parent_change__gap_le_1s (gap_le_1s):** n=5,932; median gap=0.2s; median feature distance=11.898; median cosine=0.816.
- **within_parent__all (all):** n=14,102; median gap=0.2s; median feature distance=10.374; median cosine=0.872.
- **within_parent__gap_le_1s (gap_le_1s):** n=10,633; median gap=0.1s; median feature distance=10.051; median cosine=0.881.

### Step-order diagnostic

Score is Spearman correlation between event order and projection on the chronological start-to-end feature axis; shuffled sequences use that same axis.

- **all_parent_events:**
  - `real_beats_reverse`: 0.957 (95% video-bootstrap CI 0.940–0.974; n=531)
  - `real_beats_random_fraction`: 0.890 (95% video-bootstrap CI 0.876–0.904; n=531)
  - `real_random_percentile`: 0.910 (95% video-bootstrap CI 0.896–0.924; n=531)
- **essential_only:**
  - `real_beats_reverse`: 0.994 (95% video-bootstrap CI 0.986–1.000; n=361)
  - `real_beats_random_fraction`: 0.918 (95% video-bootstrap CI 0.906–0.929; n=361)
  - `real_random_percentile`: 0.962 (95% video-bootstrap CI 0.953–0.970; n=361)

### Next-essential-step retrieval

Candidates are later parent-step events from the same video; the immediate next essential event is the target. The chronological baseline ranks the earliest future event first.

- `rr_cosine_current`: 0.520 (95% video-bootstrap CI 0.497–0.544; n=407)
- `r1_cosine_current`: 0.353 (95% video-bootstrap CI 0.325–0.383; n=407)
- `r5_cosine_current`: 0.734 (95% video-bootstrap CI 0.709–0.760; n=407)
- `rr_random_expected`: 0.316 (95% video-bootstrap CI 0.301–0.332; n=407)
- `r1_random_expected`: 0.138 (95% video-bootstrap CI 0.127–0.149; n=407)
- `r5_random_expected`: 0.519 (95% video-bootstrap CI 0.493–0.546; n=407)
- `rr_earliest_future_baseline`: 0.535 (95% video-bootstrap CI 0.513–0.559; n=407)
- `r1_earliest_future_baseline`: 0.324 (95% video-bootstrap CI 0.294–0.356; n=407)
- `r5_earliest_future_baseline`: 0.833 (95% video-bootstrap CI 0.808–0.857; n=407)
- `rr_linear_extrapolation`: 0.484 (95% video-bootstrap CI 0.459–0.510; n=354)
- `r1_linear_extrapolation`: 0.315 (95% video-bootstrap CI 0.285–0.346; n=354)
- `r5_linear_extrapolation`: 0.697 (95% video-bootstrap CI 0.667–0.727; n=354)

Random rank expectation is computed for each query's candidate count, then averaged by video.

## Interpretation limits

- Each video is represented by its available annotated clips, not continuous frame coverage.
- PCA is visualization only; quantitative analyses use original 384-D feature space.
- The order score is a signed projection on the true start-to-end feature axis; it can reflect scene continuity and is not proof of procedural understanding.
- Retrieval candidates are later annotated parent steps in the same video; unlabeled parent events are included as distractors.
- Bootstrap intervals resample videos, not clips.
- These results establish a frozen-feature baseline. They cannot show that an induced-straightness objective improves performance; that requires comparing a trained representation later.
- Parent-step geometry uses sparse annotated clips and must be interpreted together with the reported time gaps.
