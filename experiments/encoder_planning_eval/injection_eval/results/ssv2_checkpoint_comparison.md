# SSv2 checkpoint comparison

All runs use the same 85 calibration and 312 held-out source clips (seed 0). Values are clip-level AUROC: edited versus clean.
DINOv2 is raw patch tokens averaged per frame. The RGB run additionally uses a downsampled pixel baseline.
The geometry score and predictor score for each injection were specified by the evaluation runner.
The full threshold metrics and confidence intervals are in `ssv2_checkpoint_comparison.csv`.

| Injection | Score | DINOv2 geometry | VICREG3 predictor | VICREG clean predictor | RGB residual predictor |
|---|---|---:|---:|---:|---:|
| reverse | pe_mean | 0.500 | 0.501 | 0.506 | 0.504 |
| swap | pe_max | 0.672 | 0.734 | 0.730 | 0.711 |
| shuffle | pe_max | 0.731 | 0.741 | 0.742 | 0.730 |
| drop | pe_max | 0.705 | 0.659 | 0.657 | 0.654 |
| splice_other | pe_max | 0.929 | 0.983 | 0.982 | 0.999 |
| splice_same | pe_max | 0.914 | 0.980 | 0.974 | 0.999 |
| freeze | pe_max | 0.997 | 0.596 | 0.592 | 0.580 |
| speed_up | pe_mean | 0.618 | 0.554 | 0.554 | 0.578 |
| slow_down | pe_mean | 0.610 | 0.282 | 0.289 | 0.495 |
| appearance | pe_mean | 0.500 | 0.515 | 0.520 | 0.512 |

## Geometry-only headline scores

| Injection | Score | DINOv2 | VICREG3 latent | VICREG clean latent | RGB residual latent | Pixel baseline |
|---|---|---:|---:|---:|---:|---:|
| reverse | cv_mean | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| swap | jump_spike | 0.672 | 0.617 | 0.626 | 0.633 | 0.576 |
| shuffle | jump_spike | 0.731 | 0.684 | 0.671 | 0.706 | 0.630 |
| drop | jump_spike | 0.705 | 0.643 | 0.642 | 0.720 | 0.718 |
| splice_other | chord_dev | 0.929 | 0.861 | 0.858 | 0.775 | 0.809 |
| splice_same | chord_dev | 0.914 | 0.840 | 0.826 | 0.778 | 0.799 |
| freeze | stall | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| speed_up | step_cv | 0.618 | 0.596 | 0.602 | 0.624 | 0.622 |
| slow_down | step_cv | 0.610 | 0.561 | 0.565 | 0.369 | 0.522 |
| appearance | jump_max | 0.500 | 0.508 | 0.500 | 0.507 | 0.491 |

All eight geometry scores per injection are in each run’s `geometry_only_comparison.csv`.
Appearance is a benign control; the RGB run also has a blur control.
These are synthetic SSv2 edits, not real phase-level error annotations.
