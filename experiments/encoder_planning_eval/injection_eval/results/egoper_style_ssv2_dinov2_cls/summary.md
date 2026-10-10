# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 85; held-out source clips: 312.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_cls | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.500 | 0.500 | 0.048 | 0.048 | 0.088 | nan |
| dinov2_cls | geometry | swap | mistake | jump_spike | 312 | 0.615 | 0.590 | 0.643 | 0.051 | 0.058 | 0.092 | nan |
| dinov2_cls | geometry | shuffle | mistake | jump_spike | 312 | 0.682 | 0.652 | 0.708 | 0.090 | 0.058 | 0.156 | nan |
| dinov2_cls | geometry | drop | mistake | jump_spike | 312 | 0.651 | 0.625 | 0.677 | 0.122 | 0.042 | 0.209 | nan |
| dinov2_cls | geometry | splice_other | mistake | chord_dev | 312 | 0.899 | 0.876 | 0.917 | 0.567 | 0.074 | 0.691 | nan |
| dinov2_cls | geometry | splice_same | mistake | chord_dev | 312 | 0.888 | 0.865 | 0.909 | 0.529 | 0.074 | 0.660 | nan |
| dinov2_cls | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.058 | 0.972 | nan |
| dinov2_cls | geometry | speed_up | mistake | step_cv | 312 | 0.590 | 0.574 | 0.609 | 0.096 | 0.067 | 0.165 | nan |
| dinov2_cls | geometry | slow_down | mistake | step_cv | 312 | 0.627 | 0.596 | 0.670 | 0.067 | 0.048 | 0.121 | nan |
| dinov2_cls | geometry | appearance | negative_control | jump_max | 312 | 0.500 | 0.500 | 0.501 | nan | 0.045 | nan | 0.045 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| dinov2_cls | splice_other | step | 0.604 | 0.262 | 312 | 1248 |
| dinov2_cls | splice_same | step | 0.624 | 0.276 | 312 | 1248 |
| dinov2_cls | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| dinov2_cls | swap | step | 0.926 | 0.601 | 312 |
| dinov2_cls | swap | tokstep | 0.926 | 0.601 | 312 |
| dinov2_cls | shuffle | step | 0.862 | 0.459 | 312 |
| dinov2_cls | shuffle | tokstep | 0.862 | 0.459 | 312 |
| dinov2_cls | drop | step | 0.647 | 0.273 | 312 |
| dinov2_cls | drop | tokstep | 0.647 | 0.273 | 312 |
| dinov2_cls | splice_other | step | 1.000 | 0.400 | 312 |
| dinov2_cls | splice_other | tokstep | 1.000 | 0.400 | 312 |
| dinov2_cls | splice_same | step | 1.000 | 0.400 | 312 |
| dinov2_cls | splice_same | tokstep | 1.000 | 0.400 | 312 |
| dinov2_cls | freeze | step | 0.561 | 0.200 | 312 |
| dinov2_cls | freeze | tokstep | 0.561 | 0.200 | 312 |
