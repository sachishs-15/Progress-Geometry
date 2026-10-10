# Raw DINOv2 long-procedure analysis

## Setup

- Data: Ego4D Goal-Step val; 6,732 clips across 134 videos.
- Features: raw frozen DINOv2, 16 sampled frames per clip, 384-D; clip descriptor is the untrained frame mean.
- No project-trained checkpoints or predictive models were used. PCA is only used for plots.
- Parent-step index: 6,732 clips labeled; 0 without an index. Relevance is unlabeled on 2,388 clips.
- Sum of positive gaps between observed clips: 7.6 hours.

## Results

### Geometry within clips

Raw chord efficiency is endpoint distance divided by total feature-path length (1 means a straight path).
- **essential:** n=3,977; median chord efficiency=0.099; median velocity-turn cosine=-0.359.
- **irrelevant:** n=222; median chord efficiency=0.099; median velocity-turn cosine=-0.342.
- **optional:** n=87; median chord efficiency=0.100; median velocity-turn cosine=-0.324.
- **unsure:** n=58; median chord efficiency=0.105; median velocity-turn cosine=-0.325.
- **unlabeled:** n=2,388; median chord efficiency=0.098; median velocity-turn cosine=-0.355.

### Geometry across multiple clips in a parent step

Singleton parent steps are excluded from within-step cohesion. Sparse path efficiency includes time gaps between observed clips.
- **At least 2 clips:** n=929; median cosine to step centroid=0.942; median L2 dispersion=6.351.
- **At least 3 clips:** n=559; median cosine to step centroid=0.928; median L2 dispersion=7.013; median sparse path efficiency=0.342.

### Transitions between observed clips

- **parent_change__all (all):** n=3,239; median gap=0.9s; median feature distance=13.312; median cosine=0.766.
- **parent_change__gap_le_1s (gap_le_1s):** n=1,672; median gap=0.2s; median feature distance=12.193; median cosine=0.807.
- **within_parent__all (all):** n=3,359; median gap=0.2s; median feature distance=10.434; median cosine=0.870.
- **within_parent__gap_le_1s (gap_le_1s):** n=2,474; median gap=0.1s; median feature distance=10.135; median cosine=0.875.

### Step-order diagnostic

Score is Spearman correlation between event order and projection on the chronological start-to-end feature axis; shuffled sequences use that same axis.

- **all_parent_events:**
  - `real_beats_reverse`: 0.967 (95% video-bootstrap CI 0.935–0.992; n=123)
  - `real_beats_random_fraction`: 0.918 (95% video-bootstrap CI 0.888–0.946; n=123)
  - `real_random_percentile`: 0.935 (95% video-bootstrap CI 0.906–0.960; n=123)
- **essential_only:**
  - `real_beats_reverse`: 0.989 (95% video-bootstrap CI 0.968–1.000; n=95)
  - `real_beats_random_fraction`: 0.906 (95% video-bootstrap CI 0.876–0.933; n=95)
  - `real_random_percentile`: 0.956 (95% video-bootstrap CI 0.933–0.974; n=95)

### Next-essential-step retrieval

Candidates are later parent-step events from the same video; the immediate next essential event is the target. The chronological baseline ranks the earliest future event first.

- `rr_cosine_current`: 0.501 (95% video-bootstrap CI 0.456–0.547; n=105)
- `r1_cosine_current`: 0.329 (95% video-bootstrap CI 0.274–0.386; n=105)
- `r5_cosine_current`: 0.723 (95% video-bootstrap CI 0.669–0.776; n=105)
- `rr_random_expected`: 0.316 (95% video-bootstrap CI 0.284–0.349; n=105)
- `r1_random_expected`: 0.139 (95% video-bootstrap CI 0.117–0.162; n=105)
- `r5_random_expected`: 0.517 (95% video-bootstrap CI 0.462–0.573; n=105)
- `rr_earliest_future_baseline`: 0.549 (95% video-bootstrap CI 0.503–0.594; n=105)
- `r1_earliest_future_baseline`: 0.349 (95% video-bootstrap CI 0.292–0.407; n=105)
- `r5_earliest_future_baseline`: 0.828 (95% video-bootstrap CI 0.778–0.874; n=105)
- `rr_linear_extrapolation`: 0.445 (95% video-bootstrap CI 0.396–0.497; n=92)
- `r1_linear_extrapolation`: 0.273 (95% video-bootstrap CI 0.213–0.338; n=92)
- `r5_linear_extrapolation`: 0.663 (95% video-bootstrap CI 0.602–0.721; n=92)

Random rank expectation is computed for each query's candidate count, then averaged by video.

## Interpretation limits

- Each video is represented by its available annotated clips, not continuous frame coverage.
- PCA is visualization only; quantitative analyses use original 384-D feature space.
- The order score is a signed projection on the true start-to-end feature axis; it can reflect scene continuity and is not proof of procedural understanding.
- Retrieval candidates are later annotated parent steps in the same video; unlabeled parent events are included as distractors.
- Bootstrap intervals resample videos, not clips.
- These results establish a frozen-feature baseline. They cannot show that an induced-straightness objective improves performance; that requires comparing a trained representation later.
- Parent-step geometry uses sparse annotated clips and must be interpreted together with the reported time gaps.
