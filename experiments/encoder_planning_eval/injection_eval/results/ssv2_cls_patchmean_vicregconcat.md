# SSv2 geometry: DINOv2 CLS, DINOv2 patch mean, VICREG clean patch concat

Same 85 calibration and 312 held-out source clips and the same synthetic injections. All columns use the same preset geometry score per row. AUROC is edited versus clean.

| Alteration | Score | DINOv2 CLS | DINOv2 patch mean | VICREG clean concat |
|---|---|---:|---:|---:|
| reverse | `cv_mean` | 0.500 | 0.500 | 0.500 |
| swap | `jump_spike` | 0.615 | 0.672 | 0.705 |
| shuffle | `jump_spike` | 0.682 | 0.731 | 0.754 |
| drop | `jump_spike` | 0.651 | 0.705 | 0.746 |
| splice_other | `chord_dev` | 0.899 | 0.929 | 0.912 |
| splice_same | `chord_dev` | 0.888 | 0.914 | 0.891 |
| freeze | `stall` | 0.997 | 0.997 | 0.997 |
| speed_up | `step_cv` | 0.590 | 0.618 | 0.628 |
| slow_down | `step_cv` | 0.627 | 0.610 | 0.644 |
| appearance | `jump_max` | 0.500 | 0.500 | 0.511 |

CLS features came from `dinov2/ssv2_validation.pt`; patch features came from `dinov2_noreg/ssv2_validation.pt`. They were aligned by video path because the files have different row orders.
Appearance is a benign control. These are synthetic SSv2 edits, not real mistake-phase results.
