# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 85; held-out source clips: 312.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.499 | 0.500 | 0.045 | 0.045 | 0.082 | nan |
| raw | geometry | swap | mistake | jump_spike | 312 | 0.672 | 0.641 | 0.701 | 0.026 | 0.026 | 0.049 | nan |
| raw | geometry | shuffle | mistake | jump_spike | 312 | 0.731 | 0.699 | 0.759 | 0.058 | 0.026 | 0.107 | nan |
| raw | geometry | drop | mistake | jump_spike | 312 | 0.705 | 0.678 | 0.729 | 0.106 | 0.026 | 0.187 | nan |
| raw | geometry | splice_other | mistake | chord_dev | 312 | 0.929 | 0.910 | 0.945 | 0.712 | 0.087 | 0.791 | nan |
| raw | geometry | splice_same | mistake | chord_dev | 312 | 0.914 | 0.892 | 0.931 | 0.683 | 0.087 | 0.772 | nan |
| raw | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.054 | 0.973 | nan |
| raw | geometry | speed_up | mistake | step_cv | 312 | 0.618 | 0.598 | 0.642 | 0.109 | 0.054 | 0.187 | nan |
| raw | geometry | slow_down | mistake | step_cv | 312 | 0.610 | 0.573 | 0.655 | 0.080 | 0.058 | 0.141 | nan |
| raw | geometry | appearance | negative_control | jump_max | 312 | 0.500 | 0.500 | 0.501 | nan | 0.022 | nan | 0.022 |
| vicreg_clean | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.500 | 0.501 | 0.058 | 0.058 | 0.103 | nan |
| vicreg_clean | predictor | reverse | mistake | pe_mean | 312 | 0.506 | 0.502 | 0.510 | 0.029 | 0.026 | 0.055 | nan |
| vicreg_clean | geometry | swap | mistake | jump_spike | 312 | 0.626 | 0.598 | 0.656 | 0.026 | 0.019 | 0.049 | nan |
| vicreg_clean | predictor | swap | mistake | pe_max | 312 | 0.730 | 0.709 | 0.751 | 0.160 | 0.035 | 0.268 | nan |
| vicreg_clean | geometry | shuffle | mistake | jump_spike | 312 | 0.671 | 0.642 | 0.706 | 0.051 | 0.019 | 0.096 | nan |
| vicreg_clean | predictor | shuffle | mistake | pe_max | 312 | 0.742 | 0.720 | 0.767 | 0.199 | 0.035 | 0.322 | nan |
| vicreg_clean | geometry | drop | mistake | jump_spike | 312 | 0.642 | 0.609 | 0.669 | 0.205 | 0.080 | 0.319 | nan |
| vicreg_clean | predictor | drop | mistake | pe_max | 312 | 0.657 | 0.639 | 0.679 | 0.109 | 0.029 | 0.192 | nan |
| vicreg_clean | geometry | splice_other | mistake | chord_dev | 312 | 0.858 | 0.832 | 0.883 | 0.561 | 0.096 | 0.677 | nan |
| vicreg_clean | predictor | splice_other | mistake | pe_max | 312 | 0.982 | 0.974 | 0.991 | 0.875 | 0.035 | 0.916 | nan |
| vicreg_clean | geometry | splice_same | mistake | chord_dev | 312 | 0.826 | 0.789 | 0.856 | 0.487 | 0.096 | 0.615 | nan |
| vicreg_clean | predictor | splice_same | mistake | pe_max | 312 | 0.974 | 0.964 | 0.983 | 0.833 | 0.035 | 0.892 | nan |
| vicreg_clean | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.042 | 0.980 | nan |
| vicreg_clean | predictor | freeze | mistake | pe_max | 312 | 0.592 | 0.578 | 0.608 | 0.090 | 0.035 | 0.160 | nan |
| vicreg_clean | geometry | speed_up | mistake | step_cv | 312 | 0.602 | 0.575 | 0.624 | 0.099 | 0.054 | 0.172 | nan |
| vicreg_clean | predictor | speed_up | mistake | pe_mean | 312 | 0.554 | 0.545 | 0.566 | 0.042 | 0.026 | 0.078 | nan |
| vicreg_clean | geometry | slow_down | mistake | step_cv | 312 | 0.565 | 0.535 | 0.601 | 0.054 | 0.032 | 0.100 | nan |
| vicreg_clean | predictor | slow_down | mistake | pe_mean | 312 | 0.289 | 0.273 | 0.304 | 0.000 | 0.026 | 0.000 | nan |
| vicreg_clean | geometry | appearance | negative_control | jump_max | 312 | 0.500 | 0.490 | 0.512 | nan | 0.048 | nan | 0.051 |
| vicreg_clean | predictor | appearance | negative_control | pe_mean | 312 | 0.520 | 0.505 | 0.535 | nan | 0.026 | nan | 0.026 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| raw | splice_other | step | 0.608 | 0.254 | 312 | 1248 |
| raw | splice_same | step | 0.627 | 0.263 | 312 | 1248 |
| raw | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| vicreg_clean | splice_other | step | 0.611 | 0.258 | 312 | 1248 |
| vicreg_clean | splice_other | pe | 0.665 | 0.287 | 312 | 1248 |
| vicreg_clean | splice_same | step | 0.617 | 0.259 | 312 | 1248 |
| vicreg_clean | splice_same | pe | 0.663 | 0.285 | 312 | 1248 |
| vicreg_clean | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| vicreg_clean | freeze | pe | 0.021 | 0.052 | 312 | 936 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| raw | swap | step | 0.910 | 0.601 | 312 |
| raw | swap | tokstep | 0.968 | 0.601 | 312 |
| raw | shuffle | step | 0.897 | 0.459 | 312 |
| raw | shuffle | tokstep | 0.933 | 0.459 | 312 |
| raw | drop | step | 0.651 | 0.273 | 312 |
| raw | drop | tokstep | 0.788 | 0.273 | 312 |
| raw | splice_other | step | 0.994 | 0.400 | 312 |
| raw | splice_other | tokstep | 0.997 | 0.400 | 312 |
| raw | splice_same | step | 0.987 | 0.400 | 312 |
| raw | splice_same | tokstep | 1.000 | 0.400 | 312 |
| raw | freeze | step | 0.593 | 0.200 | 312 |
| raw | freeze | tokstep | 0.660 | 0.200 | 312 |
| vicreg_clean | swap | step | 0.923 | 0.601 | 312 |
| vicreg_clean | swap | tokstep | 0.958 | 0.601 | 312 |
| vicreg_clean | swap | pe | 0.952 | 0.601 | 312 |
| vicreg_clean | shuffle | step | 0.862 | 0.459 | 312 |
| vicreg_clean | shuffle | tokstep | 0.926 | 0.459 | 312 |
| vicreg_clean | shuffle | pe | 0.933 | 0.459 | 312 |
| vicreg_clean | drop | step | 0.593 | 0.273 | 312 |
| vicreg_clean | drop | tokstep | 0.753 | 0.273 | 312 |
| vicreg_clean | drop | pe | 0.747 | 0.273 | 312 |
| vicreg_clean | splice_other | step | 0.994 | 0.400 | 312 |
| vicreg_clean | splice_other | tokstep | 1.000 | 0.400 | 312 |
| vicreg_clean | splice_other | pe | 1.000 | 0.400 | 312 |
| vicreg_clean | splice_same | step | 0.974 | 0.400 | 312 |
| vicreg_clean | splice_same | tokstep | 0.997 | 0.400 | 312 |
| vicreg_clean | splice_same | pe | 0.997 | 0.400 | 312 |
| vicreg_clean | freeze | step | 0.494 | 0.200 | 312 |
| vicreg_clean | freeze | tokstep | 0.628 | 0.200 | 312 |
| vicreg_clean | freeze | pe | 0.615 | 0.200 | 312 |
