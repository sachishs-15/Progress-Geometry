# Precomputed-feature data integrity

The precompute pipeline previously replaced videos that failed to decode with 16 black
frames and saved their embeddings under the original video path. Direct inspection
confirmed that this affected existing artifacts:

- K400 10% DINOv2 no-register patch features: at least 2,364 failed paths from GPU shard
  3 are present in the 24,126-entry merged file (9.8% of the merged file). Inspected
  failed entries have identical features at all 16 timesteps, and features from distinct
  failed paths are exactly equal.
- Charades training DINOv2 features: the failed `SFOPU_3.2_10.1.mp4` entry is present and
  constant across all 16 timesteps.
- EPIC-Kitchens training and validation DINOv2 features: the logged failed paths are
  present as constant entries; duplicated manifest paths produced two entries in each
  affected split.

These legacy files must not be used for new training or evaluation. Regenerate them with
`scripts/local/precompute_features/precompute_features.py`. The corrected extractor skips
decode/encode failures, stores them in both `<output>.failures.csv` and the output's
`failures` field, and fingerprints extraction settings in every chunk and final file.
Legacy outputs without this provenance are rejected by the extractor when encountered at
the intended output path, and `FeatureProvider` warns when one is loaded directly.
