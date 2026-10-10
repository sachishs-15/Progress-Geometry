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
| vicreg_clean | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.500 | 0.501 | 0.074 | 0.074 | 0.128 | nan |
| vicreg_clean | geometry | swap | mistake | jump_spike | 312 | 0.633 | 0.601 | 0.669 | 0.019 | 0.019 | 0.037 | nan |
| vicreg_clean | geometry | shuffle | mistake | jump_spike | 312 | 0.672 | 0.642 | 0.701 | 0.054 | 0.019 | 0.101 | nan |
| vicreg_clean | geometry | drop | mistake | jump_spike | 312 | 0.651 | 0.618 | 0.679 | 0.179 | 0.054 | 0.291 | nan |
| vicreg_clean | geometry | splice_other | mistake | chord_dev | 312 | 0.849 | 0.823 | 0.876 | 0.500 | 0.077 | 0.634 | nan |
| vicreg_clean | geometry | splice_same | mistake | chord_dev | 312 | 0.822 | 0.788 | 0.853 | 0.429 | 0.077 | 0.570 | nan |
| vicreg_clean | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.035 | 0.983 | nan |
| vicreg_clean | geometry | speed_up | mistake | step_cv | 312 | 0.604 | 0.577 | 0.628 | 0.077 | 0.038 | 0.138 | nan |
| vicreg_clean | geometry | slow_down | mistake | step_cv | 312 | 0.567 | 0.534 | 0.601 | 0.032 | 0.013 | 0.061 | nan |
| vicreg_clean | geometry | appearance | negative_control | jump_max | 312 | 0.493 | 0.483 | 0.505 | nan | 0.051 | nan | 0.058 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| raw | splice_other | step | 0.608 | 0.254 | 312 | 1248 |
| raw | splice_same | step | 0.627 | 0.263 | 312 | 1248 |
| raw | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| vicreg_clean | splice_other | step | 0.610 | 0.258 | 312 | 1248 |
| vicreg_clean | splice_same | step | 0.617 | 0.257 | 312 | 1248 |
| vicreg_clean | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |

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
| vicreg_clean | swap | tokstep | 0.962 | 0.601 | 312 |
| vicreg_clean | shuffle | step | 0.875 | 0.459 | 312 |
| vicreg_clean | shuffle | tokstep | 0.917 | 0.459 | 312 |
| vicreg_clean | drop | step | 0.583 | 0.273 | 312 |
| vicreg_clean | drop | tokstep | 0.731 | 0.273 | 312 |
| vicreg_clean | splice_other | step | 0.987 | 0.400 | 312 |
| vicreg_clean | splice_other | tokstep | 1.000 | 0.400 | 312 |
| vicreg_clean | splice_same | step | 0.974 | 0.400 | 312 |
| vicreg_clean | splice_same | tokstep | 0.997 | 0.400 | 312 |
| vicreg_clean | freeze | step | 0.481 | 0.200 | 312 |
| vicreg_clean | freeze | tokstep | 0.619 | 0.200 | 312 |
