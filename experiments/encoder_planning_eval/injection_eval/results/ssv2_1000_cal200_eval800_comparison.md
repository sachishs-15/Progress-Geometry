# SSv2 offline synthetic-edit evaluation: 1,000 source clips

200 clean calibration clips; 800 held-out evaluation clips; 22 action classes. Same paths and edits for all three reported readouts. Geometry scores only.

## Prespecified score for each edit

| Edit | Score | DINOv2 CLS | DINOv2 patch mean | VICREG patch concat |
| --- | --- | ---: | ---: | ---: |
| swap | jump_spike | 0.582 | 0.625 | 0.680 |
| shuffle | jump_spike | 0.635 | 0.687 | 0.734 |
| drop | jump_spike | 0.659 | 0.669 | 0.705 |
| speed_up | step_cv | 0.586 | 0.595 | 0.599 |
| slow_down | step_cv | 0.662 | 0.646 | 0.675 |
| splice_other | chord_dev | 0.901 | 0.931 | 0.905 |
| splice_same | chord_dev | 0.882 | 0.916 | 0.891 |
| freeze | stall | 0.995 | 0.995 | 0.995 |
| reverse | cv_mean | 0.500 | 0.500 | 0.500 |
| appearance | jump_max | 0.500 | 0.500 | 0.503 |

## Jump spike for every edit

| Edit | DINOv2 CLS | DINOv2 patch mean | VICREG patch concat |
| --- | ---: | ---: | ---: |
| swap | 0.582 | 0.625 | 0.680 |
| shuffle | 0.635 | 0.687 | 0.734 |
| drop | 0.659 | 0.669 | 0.705 |
| speed_up | 0.579 | 0.585 | 0.582 |
| slow_down | 0.707 | 0.675 | 0.716 |
| splice_other | 0.924 | 0.959 | 0.957 |
| splice_same | 0.910 | 0.948 | 0.949 |
| freeze | 0.679 | 0.677 | 0.725 |
| reverse | 0.500 | 0.500 | 0.500 |
| appearance | 0.500 | 0.500 | 0.493 |
