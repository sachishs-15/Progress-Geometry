# EgoPER-style synthetic evaluation: ssv2

Calibration source clips: 200; held-out source clips: 800.
Thresholds use clean calibration clips only. Mistake test sets pair each edited clip with its clean reference, giving 50% mistake prevalence.
Appearance and blur are benign controls; control flag rate measures false alarms on these edits.
Frame metrics cover splice and freeze only. Localization covers the boundary edits. Omission accuracy and IoU are not defined without procedural step labels.

## Segment detection

| readout | channel | injection | role | score | n | auroc | ci_lo | ci_hi | recall | fpr | f1 | control_flag_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_patch_concat | geometry | reverse | mistake | cv_mean | 800 | 0.500 | 0.500 | 0.500 | 0.058 | 0.058 | 0.103 | nan |
| dinov2_patch_concat | geometry | swap | mistake | jump_spike | 800 | 0.618 | 0.601 | 0.639 | 0.040 | 0.052 | 0.073 | nan |
| dinov2_patch_concat | geometry | shuffle | mistake | jump_spike | 800 | 0.681 | 0.666 | 0.699 | 0.076 | 0.052 | 0.135 | nan |
| dinov2_patch_concat | geometry | drop | mistake | jump_spike | 800 | 0.709 | 0.693 | 0.726 | 0.215 | 0.072 | 0.334 | nan |
| dinov2_patch_concat | geometry | splice_other | mistake | chord_dev | 800 | 0.830 | 0.817 | 0.845 | 0.415 | 0.055 | 0.565 | nan |
| dinov2_patch_concat | geometry | splice_same | mistake | chord_dev | 800 | 0.811 | 0.796 | 0.824 | 0.406 | 0.055 | 0.556 | nan |
| dinov2_patch_concat | geometry | freeze | mistake | stall | 800 | 0.995 | 0.992 | 0.998 | 1.000 | 0.060 | 0.971 | nan |
| dinov2_patch_concat | geometry | speed_up | mistake | step_cv | 800 | 0.595 | 0.583 | 0.609 | 0.134 | 0.080 | 0.220 | nan |
| dinov2_patch_concat | geometry | slow_down | mistake | step_cv | 800 | 0.728 | 0.707 | 0.749 | 0.129 | 0.054 | 0.218 | nan |
| dinov2_patch_concat | geometry | appearance | negative_control | jump_max | 800 | 0.500 | 0.500 | 0.500 | nan | 0.039 | nan | 0.039 |

## Frame scores

| readout | injection | signal | frame_auc | frame_ap | n_videos | positive_frames |
|---|---|---|---|---|---|---|
| dinov2_patch_concat | splice_other | step | 0.630 | 0.271 | 800 | 3200 |
| dinov2_patch_concat | splice_same | step | 0.634 | 0.273 | 800 | 3200 |
| dinov2_patch_concat | freeze | negative_step | 0.999 | 0.978 | 800 | 2400 |

## Seam localization

| readout | injection | signal | hit | chance | n |
|---|---|---|---|---|---|
| dinov2_patch_concat | swap | step | 0.955 | 0.597 | 800 |
| dinov2_patch_concat | swap | tokstep | 0.954 | 0.597 | 800 |
| dinov2_patch_concat | shuffle | step | 0.915 | 0.460 | 800 |
| dinov2_patch_concat | shuffle | tokstep | 0.924 | 0.460 | 800 |
| dinov2_patch_concat | drop | step | 0.772 | 0.273 | 800 |
| dinov2_patch_concat | drop | tokstep | 0.775 | 0.273 | 800 |
| dinov2_patch_concat | splice_other | step | 0.996 | 0.400 | 800 |
| dinov2_patch_concat | splice_other | tokstep | 0.996 | 0.400 | 800 |
| dinov2_patch_concat | splice_same | step | 0.996 | 0.400 | 800 |
| dinov2_patch_concat | splice_same | tokstep | 0.996 | 0.400 | 800 |
| dinov2_patch_concat | freeze | step | 0.681 | 0.200 | 800 |
| dinov2_patch_concat | freeze | tokstep | 0.675 | 0.200 | 800 |
