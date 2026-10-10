# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 200; held-out source clips: 800.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_cls | geometry | reverse | mistake | cv_mean | 800 | 0.500 | 0.500 | 0.500 | 0.039 | 0.039 | 0.072 | nan |
| dinov2_cls | geometry | swap | mistake | jump_spike | 800 | 0.582 | 0.563 | 0.599 | 0.110 | 0.089 | 0.184 | nan |
| dinov2_cls | geometry | shuffle | mistake | jump_spike | 800 | 0.635 | 0.618 | 0.650 | 0.170 | 0.089 | 0.270 | nan |
| dinov2_cls | geometry | drop | mistake | jump_spike | 800 | 0.659 | 0.644 | 0.678 | 0.211 | 0.071 | 0.329 | nan |
| dinov2_cls | geometry | splice_other | mistake | chord_dev | 800 | 0.901 | 0.887 | 0.915 | 0.547 | 0.061 | 0.681 | nan |
| dinov2_cls | geometry | splice_same | mistake | chord_dev | 800 | 0.882 | 0.867 | 0.895 | 0.524 | 0.061 | 0.661 | nan |
| dinov2_cls | geometry | freeze | mistake | stall | 800 | 0.995 | 0.992 | 0.998 | 1.000 | 0.066 | 0.968 | nan |
| dinov2_cls | geometry | speed_up | mistake | step_cv | 800 | 0.586 | 0.573 | 0.600 | 0.147 | 0.085 | 0.239 | nan |
| dinov2_cls | geometry | slow_down | mistake | step_cv | 800 | 0.662 | 0.645 | 0.683 | 0.095 | 0.061 | 0.164 | nan |
| dinov2_cls | geometry | appearance | negative_control | jump_max | 800 | 0.500 | 0.500 | 0.500 | nan | 0.039 | nan | 0.039 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| dinov2_cls | splice_other | step | 0.626 | 0.271 | 800 | 3200 |
| dinov2_cls | splice_same | step | 0.626 | 0.275 | 800 | 3200 |
| dinov2_cls | freeze | negative_step | 0.999 | 0.978 | 800 | 2400 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| dinov2_cls | swap | step | 0.886 | 0.597 | 800 |
| dinov2_cls | swap | tokstep | 0.886 | 0.597 | 800 |
| dinov2_cls | shuffle | step | 0.863 | 0.460 | 800 |
| dinov2_cls | shuffle | tokstep | 0.863 | 0.460 | 800 |
| dinov2_cls | drop | step | 0.641 | 0.273 | 800 |
| dinov2_cls | drop | tokstep | 0.641 | 0.273 | 800 |
| dinov2_cls | splice_other | step | 0.998 | 0.400 | 800 |
| dinov2_cls | splice_other | tokstep | 0.998 | 0.400 | 800 |
| dinov2_cls | splice_same | step | 0.998 | 0.400 | 800 |
| dinov2_cls | splice_same | tokstep | 0.998 | 0.400 | 800 |
| dinov2_cls | freeze | step | 0.530 | 0.200 | 800 |
| dinov2_cls | freeze | tokstep | 0.530 | 0.200 | 800 |
