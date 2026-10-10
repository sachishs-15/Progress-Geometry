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
| vicreg3 | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.500 | 0.500 | 0.045 | 0.045 | 0.082 | nan |
| vicreg3 | predictor | reverse | mistake | pe_mean | 312 | 0.501 | 0.496 | 0.505 | 0.022 | 0.026 | 0.043 | nan |
| vicreg3 | geometry | swap | mistake | jump_spike | 312 | 0.617 | 0.583 | 0.644 | 0.061 | 0.035 | 0.111 | nan |
| vicreg3 | predictor | swap | mistake | pe_max | 312 | 0.734 | 0.713 | 0.755 | 0.135 | 0.022 | 0.233 | nan |
| vicreg3 | geometry | shuffle | mistake | jump_spike | 312 | 0.684 | 0.653 | 0.710 | 0.093 | 0.035 | 0.165 | nan |
| vicreg3 | predictor | shuffle | mistake | pe_max | 312 | 0.741 | 0.719 | 0.763 | 0.160 | 0.022 | 0.271 | nan |
| vicreg3 | geometry | drop | mistake | jump_spike | 312 | 0.643 | 0.617 | 0.672 | 0.131 | 0.058 | 0.221 | nan |
| vicreg3 | predictor | drop | mistake | pe_max | 312 | 0.659 | 0.636 | 0.680 | 0.090 | 0.022 | 0.161 | nan |
| vicreg3 | geometry | splice_other | mistake | chord_dev | 312 | 0.861 | 0.838 | 0.887 | 0.465 | 0.071 | 0.605 | nan |
| vicreg3 | predictor | splice_other | mistake | pe_max | 312 | 0.983 | 0.974 | 0.990 | 0.808 | 0.022 | 0.883 | nan |
| vicreg3 | geometry | splice_same | mistake | chord_dev | 312 | 0.840 | 0.814 | 0.865 | 0.465 | 0.071 | 0.605 | nan |
| vicreg3 | predictor | splice_same | mistake | pe_max | 312 | 0.980 | 0.970 | 0.988 | 0.776 | 0.022 | 0.863 | nan |
| vicreg3 | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.077 | 0.963 | nan |
| vicreg3 | predictor | freeze | mistake | pe_max | 312 | 0.596 | 0.580 | 0.613 | 0.064 | 0.022 | 0.118 | nan |
| vicreg3 | geometry | speed_up | mistake | step_cv | 312 | 0.596 | 0.568 | 0.618 | 0.071 | 0.035 | 0.128 | nan |
| vicreg3 | predictor | speed_up | mistake | pe_mean | 312 | 0.554 | 0.542 | 0.565 | 0.058 | 0.032 | 0.106 | nan |
| vicreg3 | geometry | slow_down | mistake | step_cv | 312 | 0.561 | 0.527 | 0.596 | 0.035 | 0.016 | 0.067 | nan |
| vicreg3 | predictor | slow_down | mistake | pe_mean | 312 | 0.282 | 0.265 | 0.302 | 0.000 | 0.026 | 0.000 | nan |
| vicreg3 | geometry | appearance | negative_control | jump_max | 312 | 0.508 | 0.502 | 0.515 | nan | 0.035 | nan | 0.042 |
| vicreg3 | predictor | appearance | negative_control | pe_mean | 312 | 0.515 | 0.506 | 0.523 | nan | 0.026 | nan | 0.038 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| raw | splice_other | step | 0.608 | 0.254 | 312 | 1248 |
| raw | splice_same | step | 0.627 | 0.263 | 312 | 1248 |
| raw | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| vicreg3 | splice_other | step | 0.604 | 0.248 | 312 | 1248 |
| vicreg3 | splice_other | pe | 0.682 | 0.296 | 312 | 1248 |
| vicreg3 | splice_same | step | 0.625 | 0.256 | 312 | 1248 |
| vicreg3 | splice_same | pe | 0.680 | 0.295 | 312 | 1248 |
| vicreg3 | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |
| vicreg3 | freeze | pe | 0.034 | 0.053 | 312 | 936 |

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
| vicreg3 | swap | step | 0.894 | 0.601 | 312 |
| vicreg3 | swap | tokstep | 0.965 | 0.601 | 312 |
| vicreg3 | swap | pe | 0.958 | 0.601 | 312 |
| vicreg3 | shuffle | step | 0.859 | 0.459 | 312 |
| vicreg3 | shuffle | tokstep | 0.926 | 0.459 | 312 |
| vicreg3 | shuffle | pe | 0.917 | 0.459 | 312 |
| vicreg3 | drop | step | 0.596 | 0.273 | 312 |
| vicreg3 | drop | tokstep | 0.747 | 0.273 | 312 |
| vicreg3 | drop | pe | 0.753 | 0.273 | 312 |
| vicreg3 | splice_other | step | 0.987 | 0.400 | 312 |
| vicreg3 | splice_other | tokstep | 0.997 | 0.400 | 312 |
| vicreg3 | splice_other | pe | 0.997 | 0.400 | 312 |
| vicreg3 | splice_same | step | 0.974 | 0.400 | 312 |
| vicreg3 | splice_same | tokstep | 0.987 | 0.400 | 312 |
| vicreg3 | splice_same | pe | 0.987 | 0.400 | 312 |
| vicreg3 | freeze | step | 0.503 | 0.200 | 312 |
| vicreg3 | freeze | tokstep | 0.635 | 0.200 | 312 |
| vicreg3 | freeze | pe | 0.644 | 0.200 | 312 |
