# VICREG clean: mean versus concatenated patch geometry

Same epoch-49 checkpoint, 85 calibration clips, and 312 held-out clips. Predictor computation was skipped for both runs because pooling does not affect it.
Each score uses the same definition in both readouts. Higher AUROC means better edited-versus-clean separation.
Delta = concat minus mean. The 95% confidence interval bootstraps source clips in paired fashion (1,000 draws).

| Alteration | Score | DINOv2 mean patch | Model mean | Model concat | Delta [95% CI] |
|---|---|---:|---:|---:|---:|
| reverse | `cv_mean` | 0.500 | 0.500 | 0.500 | +0.000 [-0.001, +0.001] |
| swap | `jump_spike` | 0.672 | 0.633 | 0.705 | +0.071 [+0.042, +0.104] |
| shuffle | `jump_spike` | 0.731 | 0.672 | 0.754 | +0.081 [+0.054, +0.108] |
| drop | `jump_spike` | 0.705 | 0.651 | 0.746 | +0.095 [+0.068, +0.123] |
| splice_other | `chord_dev` | 0.929 | 0.849 | 0.912 | +0.062 [+0.042, +0.084] |
| splice_same | `chord_dev` | 0.914 | 0.822 | 0.891 | +0.068 [+0.047, +0.091] |
| freeze | `stall` | 0.997 | 0.997 | 0.997 | +0.000 [+0.000, +0.000] |
| speed_up | `step_cv` | 0.618 | 0.604 | 0.628 | +0.024 [+0.005, +0.045] |
| slow_down | `step_cv` | 0.610 | 0.567 | 0.644 | +0.077 [+0.056, +0.101] |
| appearance | `jump_max` | 0.500 | 0.493 | 0.511 | +0.018 [+0.008, +0.027] |

The CSV also contains flag rates and FPR at thresholds from clean calibration clips. Appearance is a benign control.
All eight geometry scores are in each run’s `geometry_only_comparison.csv`. These results use synthetic SSv2 edits.
