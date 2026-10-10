# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 200; held-out source clips: 800.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | geometry | reverse | mistake | cv_mean | 800 | 0.500 | 0.500 | 0.500 | 0.041 | 0.041 | 0.076 | nan |
| raw | geometry | swap | mistake | jump_spike | 800 | 0.625 | 0.607 | 0.644 | 0.105 | 0.077 | 0.178 | nan |
| raw | geometry | shuffle | mistake | jump_spike | 800 | 0.687 | 0.669 | 0.706 | 0.161 | 0.077 | 0.260 | nan |
| raw | geometry | drop | mistake | jump_spike | 800 | 0.669 | 0.653 | 0.685 | 0.204 | 0.070 | 0.320 | nan |
| raw | geometry | splice_other | mistake | chord_dev | 800 | 0.931 | 0.920 | 0.941 | 0.632 | 0.046 | 0.754 | nan |
| raw | geometry | splice_same | mistake | chord_dev | 800 | 0.916 | 0.905 | 0.927 | 0.599 | 0.046 | 0.728 | nan |
| raw | geometry | freeze | mistake | stall | 800 | 0.995 | 0.992 | 0.998 | 1.000 | 0.056 | 0.973 | nan |
| raw | geometry | speed_up | mistake | step_cv | 800 | 0.595 | 0.582 | 0.610 | 0.135 | 0.062 | 0.225 | nan |
| raw | geometry | slow_down | mistake | step_cv | 800 | 0.646 | 0.631 | 0.668 | 0.084 | 0.049 | 0.148 | nan |
| raw | geometry | appearance | negative_control | jump_max | 800 | 0.500 | 0.500 | 0.500 | nan | 0.036 | nan | 0.036 |
| vicreg_clean | geometry | reverse | mistake | cv_mean | 800 | 0.500 | 0.500 | 0.500 | 0.056 | 0.056 | 0.101 | nan |
| vicreg_clean | geometry | swap | mistake | jump_spike | 800 | 0.680 | 0.662 | 0.704 | 0.099 | 0.065 | 0.170 | nan |
| vicreg_clean | geometry | shuffle | mistake | jump_spike | 800 | 0.734 | 0.718 | 0.754 | 0.152 | 0.065 | 0.251 | nan |
| vicreg_clean | geometry | drop | mistake | jump_spike | 800 | 0.705 | 0.686 | 0.723 | 0.180 | 0.049 | 0.293 | nan |
| vicreg_clean | geometry | splice_other | mistake | chord_dev | 800 | 0.905 | 0.892 | 0.917 | 0.556 | 0.037 | 0.698 | nan |
| vicreg_clean | geometry | splice_same | mistake | chord_dev | 800 | 0.891 | 0.876 | 0.903 | 0.545 | 0.037 | 0.689 | nan |
| vicreg_clean | geometry | freeze | mistake | stall | 800 | 0.995 | 0.992 | 0.998 | 1.000 | 0.052 | 0.974 | nan |
| vicreg_clean | geometry | speed_up | mistake | step_cv | 800 | 0.599 | 0.584 | 0.613 | 0.163 | 0.089 | 0.260 | nan |
| vicreg_clean | geometry | slow_down | mistake | step_cv | 800 | 0.675 | 0.657 | 0.696 | 0.076 | 0.050 | 0.135 | nan |
| vicreg_clean | geometry | appearance | negative_control | jump_max | 800 | 0.503 | 0.495 | 0.511 | nan | 0.050 | nan | 0.044 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| raw | splice_other | step | 0.623 | 0.262 | 800 | 3200 |
| raw | splice_same | step | 0.625 | 0.262 | 800 | 3200 |
| raw | freeze | negative_step | 0.999 | 0.978 | 800 | 2400 |
| vicreg_clean | splice_other | step | 0.625 | 0.265 | 800 | 3200 |
| vicreg_clean | splice_same | step | 0.629 | 0.264 | 800 | 3200 |
| vicreg_clean | freeze | negative_step | 0.999 | 0.978 | 800 | 2400 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| raw | swap | step | 0.909 | 0.597 | 800 |
| raw | swap | tokstep | 0.954 | 0.597 | 800 |
| raw | shuffle | step | 0.885 | 0.460 | 800 |
| raw | shuffle | tokstep | 0.924 | 0.460 | 800 |
| raw | drop | step | 0.651 | 0.273 | 800 |
| raw | drop | tokstep | 0.775 | 0.273 | 800 |
| raw | splice_other | step | 0.993 | 0.400 | 800 |
| raw | splice_other | tokstep | 0.996 | 0.400 | 800 |
| raw | splice_same | step | 0.994 | 0.400 | 800 |
| raw | splice_same | tokstep | 0.996 | 0.400 | 800 |
| raw | freeze | step | 0.569 | 0.200 | 800 |
| raw | freeze | tokstep | 0.675 | 0.200 | 800 |
| vicreg_clean | swap | step | 0.929 | 0.597 | 800 |
| vicreg_clean | swap | tokstep | 0.927 | 0.597 | 800 |
| vicreg_clean | shuffle | step | 0.910 | 0.460 | 800 |
| vicreg_clean | shuffle | tokstep | 0.915 | 0.460 | 800 |
| vicreg_clean | drop | step | 0.715 | 0.273 | 800 |
| vicreg_clean | drop | tokstep | 0.716 | 0.273 | 800 |
| vicreg_clean | splice_other | step | 0.994 | 0.400 | 800 |
| vicreg_clean | splice_other | tokstep | 0.995 | 0.400 | 800 |
| vicreg_clean | splice_same | step | 0.990 | 0.400 | 800 |
| vicreg_clean | splice_same | tokstep | 0.993 | 0.400 | 800 |
| vicreg_clean | freeze | step | 0.621 | 0.200 | 800 |
| vicreg_clean | freeze | tokstep | 0.632 | 0.200 | 800 |
