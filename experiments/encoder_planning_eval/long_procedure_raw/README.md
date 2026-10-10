# Raw long-procedure analyses

Training-free analysis of Ego4D Goal-Step sequences using only the existing frozen raw DINOv2 CLS feature cache. No project-trained checkpoints are loaded and no model is fitted. Each source clip has 16 sampled frame features (384-D); this analysis mean-pools those frames to one clip descriptor and groups annotated clips by `video_uid` and `parent_step_idx`.

Run in the existing Lift environment:

```bash
/home/sachish/.conda/envs/lift/bin/python analyze_raw_ego4dgoal.py --split val --permutations 500
/home/sachish/.conda/envs/lift/bin/python analyze_raw_ego4dgoal.py --split train --permutations 500
```

The script writes to `results/{split}_raw_dinov2/` by default. The PCA is only for visualization. Numeric geometry and retrieval use the original 384-D feature space. Order and retrieval summaries bootstrap over video IDs. Retrieval ranks the next essential parent step among later parent-step events in the same video, with an earliest-future baseline and expected random-ranking scores.

Outputs:

- `FINDINGS.md`, `summary.json`: generated report and machine-readable summary.
- `video_sequence_summary.csv`: number of clips, annotation coverage, and observed gaps per video.
- `parent_steps.csv`, `parent_step_embeddings.npz`: ordered pooled step events and their raw descriptors.
- `clip_geometry.csv`, `parent_step_geometry.csv`: clip curvature/ordinality and parent-level dispersion.
- `clip_transitions.csv`, `transition_summary.csv`: within-parent versus parent-change feature jumps, also restricted to gaps of at most one second.
- `step_order_test_per_video.csv`: signed progress score versus reversed and shuffled step orders.
- `next_step_retrieval_queries.csv`: query-level retrieval ranks and baselines.
- `plots/`: PCA views of selected full procedures, with the line segments explicitly bridging unobserved time gaps.

The data are sparse clip sequences rather than continuous video features. Parent-level paths therefore summarize the observed clips and can cross missing intervals. Retrieval is a diagnostic benchmark: same-video candidates control goal identity but preserve same-scene and annotation-order cues. These experiments establish frozen-feature baselines; demonstrating a causal benefit of induced straightness still requires a later matched trained-checkpoint comparison.
