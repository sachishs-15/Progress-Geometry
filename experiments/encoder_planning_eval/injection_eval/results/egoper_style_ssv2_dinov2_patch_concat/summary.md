# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 85; held-out source clips: 312.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_patch_concat | geometry | reverse | mistake | cv_mean | 312 | 0.500 | 0.499 | 0.500 | 0.038 | 0.038 | 0.071 | nan |
| dinov2_patch_concat | geometry | swap | mistake | jump_spike | 312 | 0.634 | 0.601 | 0.666 | 0.032 | 0.048 | 0.059 | nan |
| dinov2_patch_concat | geometry | shuffle | mistake | jump_spike | 312 | 0.685 | 0.658 | 0.714 | 0.045 | 0.048 | 0.082 | nan |
| dinov2_patch_concat | geometry | drop | mistake | jump_spike | 312 | 0.728 | 0.704 | 0.758 | 0.141 | 0.035 | 0.240 | nan |
| dinov2_patch_concat | geometry | splice_other | mistake | chord_dev | 312 | 0.815 | 0.791 | 0.836 | 0.401 | 0.074 | 0.543 | nan |
| dinov2_patch_concat | geometry | splice_same | mistake | chord_dev | 312 | 0.804 | 0.779 | 0.830 | 0.385 | 0.074 | 0.527 | nan |
| dinov2_patch_concat | geometry | freeze | mistake | stall | 312 | 0.997 | 0.992 | 1.000 | 1.000 | 0.061 | 0.970 | nan |
| dinov2_patch_concat | geometry | speed_up | mistake | step_cv | 312 | 0.611 | 0.587 | 0.633 | 0.103 | 0.061 | 0.176 | nan |
| dinov2_patch_concat | geometry | slow_down | mistake | step_cv | 312 | 0.694 | 0.656 | 0.729 | 0.087 | 0.061 | 0.151 | nan |
| dinov2_patch_concat | geometry | appearance | negative_control | jump_max | 312 | 0.500 | 0.500 | 0.500 | nan | 0.038 | nan | 0.038 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| dinov2_patch_concat | splice_other | step | 0.628 | 0.270 | 312 | 1248 |
| dinov2_patch_concat | splice_same | step | 0.631 | 0.274 | 312 | 1248 |
| dinov2_patch_concat | freeze | negative_step | 0.999 | 0.979 | 312 | 936 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| dinov2_patch_concat | swap | step | 0.978 | 0.601 | 312 |
| dinov2_patch_concat | swap | tokstep | 0.968 | 0.601 | 312 |
| dinov2_patch_concat | shuffle | step | 0.936 | 0.459 | 312 |
| dinov2_patch_concat | shuffle | tokstep | 0.933 | 0.459 | 312 |
| dinov2_patch_concat | drop | step | 0.792 | 0.273 | 312 |
| dinov2_patch_concat | drop | tokstep | 0.788 | 0.273 | 312 |
| dinov2_patch_concat | splice_other | step | 0.997 | 0.400 | 312 |
| dinov2_patch_concat | splice_other | tokstep | 0.997 | 0.400 | 312 |
| dinov2_patch_concat | splice_same | step | 1.000 | 0.400 | 312 |
| dinov2_patch_concat | splice_same | tokstep | 1.000 | 0.400 | 312 |
| dinov2_patch_concat | freeze | step | 0.651 | 0.200 | 312 |
| dinov2_patch_concat | freeze | tokstep | 0.660 | 0.200 | 312 |
