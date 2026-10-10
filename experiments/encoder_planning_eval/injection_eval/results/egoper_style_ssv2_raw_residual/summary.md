# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 85; held-out source clips: 312.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.499 | 0.500 | 0.022 | 0.022 | 0.043 | nan |
| pixel | geometry | swap | mistake | jump_spike | 312 | 0.576 | 0.549 | 0.607 | 0.013 | 0.032 | 0.025 | nan |
| pixel | geometry | shuffle | mistake | jump_spike | 312 | 0.630 | 0.600 | 0.660 | 0.026 | 0.032 | 0.048 | nan |
| pixel | geometry | drop | mistake | jump_spike | 312 | 0.718 | 0.688 | 0.748 | 0.170 | 0.048 | 0.279 | nan |
| pixel | geometry | splice_other | mistake | chord_dev | 312 | 0.809 | 0.780 | 0.835 | 0.439 | 0.051 | 0.589 | nan |
| pixel | geometry | splice_same | mistake | chord_dev | 312 | 0.799 | 0.767 | 0.833 | 0.439 | 0.051 | 0.589 | nan |
| pixel | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.038 | 0.981 | nan |
| pixel | geometry | speed_up | mistake | step_cv | 312 | 0.622 | 0.598 | 0.641 | 0.077 | 0.032 | 0.139 | nan |
| pixel | geometry | slow_down | mistake | step_cv | 312 | 0.522 | 0.500 | 0.553 | 0.054 | 0.035 | 0.100 | nan |
| pixel | geometry | appearance | negative_control | jump_max | 312 | 0.491 | 0.487 | 0.491 | nan | 0.016 | nan | 0.013 |
| pixel | geometry | blur | negative_control | jump_max | 312 | 0.473 | 0.469 | 0.475 | nan | 0.016 | nan | 0.013 |
| raw_residual | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.499 | 0.500 | 0.038 | 0.038 | 0.071 | nan |
| raw_residual | predictor | reverse | mistake | pe_mean | 312 | 0.504 | 0.499 | 0.510 | 0.048 | 0.051 | 0.087 | nan |
| raw_residual | geometry | swap | mistake | jump_spike | 312 | 0.633 | 0.598 | 0.669 | 0.032 | 0.026 | 0.061 | nan |
| raw_residual | predictor | swap | mistake | pe_max | 312 | 0.711 | 0.695 | 0.734 | 0.135 | 0.016 | 0.234 | nan |
| raw_residual | geometry | shuffle | mistake | jump_spike | 312 | 0.706 | 0.672 | 0.738 | 0.058 | 0.026 | 0.107 | nan |
| raw_residual | predictor | shuffle | mistake | pe_max | 312 | 0.730 | 0.709 | 0.752 | 0.179 | 0.016 | 0.300 | nan |
| raw_residual | geometry | drop | mistake | jump_spike | 312 | 0.720 | 0.682 | 0.753 | 0.119 | 0.045 | 0.204 | nan |
| raw_residual | predictor | drop | mistake | pe_max | 312 | 0.654 | 0.637 | 0.676 | 0.087 | 0.010 | 0.158 | nan |
| raw_residual | geometry | splice_other | mistake | chord_dev | 312 | 0.775 | 0.741 | 0.802 | 0.369 | 0.029 | 0.528 | nan |
| raw_residual | predictor | splice_other | mistake | pe_max | 312 | 0.999 | 0.998 | 1.000 | 1.000 | 0.016 | 0.992 | nan |
| raw_residual | geometry | splice_same | mistake | chord_dev | 312 | 0.778 | 0.744 | 0.809 | 0.359 | 0.029 | 0.517 | nan |
| raw_residual | predictor | splice_same | mistake | pe_max | 312 | 0.999 | 0.997 | 1.000 | 0.987 | 0.016 | 0.986 | nan |
| raw_residual | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.019 | 0.990 | nan |
| raw_residual | predictor | freeze | mistake | pe_max | 312 | 0.580 | 0.564 | 0.595 | 0.051 | 0.016 | 0.096 | nan |
| raw_residual | geometry | speed_up | mistake | step_cv | 312 | 0.624 | 0.595 | 0.651 | 0.071 | 0.038 | 0.127 | nan |
| raw_residual | predictor | speed_up | mistake | pe_mean | 312 | 0.578 | 0.567 | 0.594 | 0.058 | 0.026 | 0.107 | nan |
| raw_residual | geometry | slow_down | mistake | step_cv | 312 | 0.369 | 0.343 | 0.393 | 0.013 | 0.019 | 0.025 | nan |
| raw_residual | predictor | slow_down | mistake | pe_mean | 312 | 0.495 | 0.476 | 0.510 | 0.048 | 0.051 | 0.087 | nan |
| raw_residual | geometry | appearance | negative_control | jump_max | 312 | 0.507 | 0.498 | 0.517 | nan | 0.022 | nan | 0.029 |
| raw_residual | predictor | appearance | negative_control | pe_mean | 312 | 0.512 | 0.501 | 0.525 | nan | 0.051 | nan | 0.054 |
| raw_residual | geometry | blur | negative_control | jump_max | 312 | 0.472 | 0.466 | 0.476 | nan | 0.022 | nan | 0.019 |
| raw_residual | predictor | blur | negative_control | pe_mean | 312 | 0.489 | 0.481 | 0.494 | nan | 0.051 | nan | 0.048 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| pixel | splice_other | step | 0.633 | 0.264 | 312 | 1248 |
| pixel | splice_same | step | 0.625 | 0.258 | 312 | 1248 |
| pixel | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| raw_residual | splice_other | step | 0.626 | 0.274 | 312 | 1248 |
| raw_residual | splice_other | pe | 0.718 | 0.379 | 312 | 1248 |
| raw_residual | splice_same | step | 0.628 | 0.271 | 312 | 1248 |
| raw_residual | splice_same | pe | 0.717 | 0.378 | 312 | 1248 |
| raw_residual | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| raw_residual | freeze | pe | 0.176 | 0.057 | 312 | 936 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| pixel | swap | step | 0.936 | 0.601 | 312 |
| pixel | swap | tokstep | 0.936 | 0.601 | 312 |
| pixel | shuffle | step | 0.888 | 0.459 | 312 |
| pixel | shuffle | tokstep | 0.888 | 0.459 | 312 |
| pixel | drop | step | 0.740 | 0.273 | 312 |
| pixel | drop | tokstep | 0.740 | 0.273 | 312 |
| pixel | splice_other | step | 0.987 | 0.400 | 312 |
| pixel | splice_other | tokstep | 0.987 | 0.400 | 312 |
| pixel | splice_same | step | 0.984 | 0.400 | 312 |
| pixel | splice_same | tokstep | 0.984 | 0.400 | 312 |
| pixel | freeze | step | 0.628 | 0.200 | 312 |
| pixel | freeze | tokstep | 0.628 | 0.200 | 312 |
| raw_residual | swap | step | 0.933 | 0.601 | 312 |
| raw_residual | swap | tokstep | 0.933 | 0.601 | 312 |
| raw_residual | swap | pe | 0.920 | 0.601 | 312 |
| raw_residual | shuffle | step | 0.904 | 0.459 | 312 |
| raw_residual | shuffle | tokstep | 0.904 | 0.459 | 312 |
| raw_residual | shuffle | pe | 0.913 | 0.459 | 312 |
| raw_residual | drop | step | 0.718 | 0.273 | 312 |
| raw_residual | drop | tokstep | 0.718 | 0.273 | 312 |
| raw_residual | drop | pe | 0.702 | 0.273 | 312 |
| raw_residual | splice_other | step | 1.000 | 0.400 | 312 |
| raw_residual | splice_other | tokstep | 1.000 | 0.400 | 312 |
| raw_residual | splice_other | pe | 1.000 | 0.400 | 312 |
| raw_residual | splice_same | step | 1.000 | 0.400 | 312 |
| raw_residual | splice_same | tokstep | 1.000 | 0.400 | 312 |
| raw_residual | splice_same | pe | 1.000 | 0.400 | 312 |
| raw_residual | freeze | step | 0.622 | 0.200 | 312 |
| raw_residual | freeze | tokstep | 0.622 | 0.200 | 312 |
| raw_residual | freeze | pe | 0.615 | 0.200 | 312 |
