### ssv2: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| vicreg3 | 0.0606 | 0.0622 | 0.03 |
| vicreg3_last | 0.6346 | 0.0478 | -12.28 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### ssv2: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | raw geometry | vicreg3 geometry | vicreg3 predictor | vicreg3_last geometry | vicreg3_last predictor |
|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.49, 0.51] (pe_mean) |
| **swap** | clean16 | 0.67 [0.64, 0.70] (jump_spike) | 0.62 [0.60, 0.65] (jump_spike) | 0.73 [0.71, 0.74] (pe_max) | 0.63 [0.60, 0.65] (jump_spike) | 0.51 [0.50, 0.51] (pe_max) |
| **shuffle** | clean16 | 0.72 [0.70, 0.74] (jump_spike) | 0.68 [0.66, 0.71] (jump_spike) | 0.74 [0.72, 0.76] (pe_max) | 0.67 [0.65, 0.69] (jump_spike) | 0.51 [0.50, 0.52] (pe_max) |
| **drop** | clean12 | 0.70 [0.68, 0.72] (jump_spike) | 0.63 [0.61, 0.66] (jump_spike) | 0.65 [0.63, 0.67] (pe_max) | 0.65 [0.62, 0.67] (jump_spike) | 0.52 [0.51, 0.52] (pe_max) |
| **splice_other** | clean16 | 0.93 [0.91, 0.94] (chord_dev) | 0.86 [0.84, 0.88] (chord_dev) | 0.98 [0.97, 0.99] (pe_max) | 0.81 [0.78, 0.84] (chord_dev) | 0.64 [0.62, 0.66] (pe_max) |
| **splice_same** | clean16 | 0.92 [0.90, 0.93] (chord_dev) | 0.83 [0.81, 0.85] (chord_dev) | 0.97 [0.96, 0.98] (pe_max) | 0.78 [0.75, 0.81] (chord_dev) | 0.62 [0.60, 0.64] (pe_max) |
| **freeze** | clean16 | 1.00 [0.99, 1.00] (stall) | 1.00 [0.99, 1.00] (stall) | 0.61 [0.59, 0.62] (pe_max) | 1.00 [0.99, 1.00] (stall) | 0.49 [0.49, 0.50] (pe_max) |
| **speed_up** | clean12 | 0.62 [0.60, 0.64] (step_cv) | 0.59 [0.57, 0.61] (step_cv) | 0.55 [0.54, 0.56] (pe_mean) | 0.59 [0.57, 0.61] (step_cv) | 0.52 [0.51, 0.52] (pe_mean) |
| **slow_down** | clean16 | 0.62 [0.58, 0.64] (step_cv) | 0.57 [0.54, 0.59] (step_cv) | 0.29 [0.26, 0.30] (pe_mean) | 0.56 [0.54, 0.59] (step_cv) | 0.48 [0.47, 0.48] (pe_mean) |
| **appearance** | clean16 | 0.50 [0.50, 0.50] (jump_max) | 0.51 [0.51, 0.52] (jump_max) | 0.52 [0.51, 0.53] (pe_mean) | 0.50 [0.50, 0.51] (jump_max) | 0.49 [0.47, 0.52] (pe_mean) |
| **null_split** | other half | 0.51 [0.46, 0.56] (jump_max) | 0.50 [0.44, 0.55] (jump_max) | 0.51 [0.46, 0.56] (pe_mean) | 0.51 [0.45, 0.57] (jump_max) | 0.50 [0.45, 0.55] (pe_mean) |

### ssv2: does the largest per-transition value land within one step of the seam?

| injection | chance | raw step | raw tokstep | vicreg3 step | vicreg3 tokstep | vicreg3 pe | vicreg3_last step | vicreg3_last tokstep | vicreg3_last pe |
|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.92 | 0.95 | 0.89 | 0.95 | 0.95 | 0.90 | 0.94 | 0.55 |
| **shuffle** | 0.46 | 0.87 | 0.92 | 0.82 | 0.91 | 0.90 | 0.80 | 0.89 | 0.56 |
| **drop** | 0.27 | 0.70 | 0.83 | 0.64 | 0.78 | 0.77 | 0.61 | 0.76 | 0.28 |
| **splice_other** | 0.40 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.97 | 0.99 | 0.56 |
| **splice_same** | 0.40 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.96 | 0.99 | 0.50 |
| **freeze** | 0.20 | 0.58 | 0.71 | 0.48 | 0.66 | 0.64 | 0.48 | 0.65 | 0.17 |

### k400: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| vicreg3 | 0.0639 | 0.0713 | 0.10 |
| vicreg3_last | 0.3864 | 0.0438 | -7.82 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### k400: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | raw geometry | vicreg3 geometry | vicreg3 predictor | vicreg3_last geometry | vicreg3_last predictor |
|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (pe_mean) |
| **swap** | clean16 | 0.51 [0.50, 0.52] (jump_spike) | 0.51 [0.50, 0.52] (jump_spike) | 0.52 [0.52, 0.53] (pe_max) | 0.50 [0.49, 0.51] (jump_spike) | 0.50 [0.50, 0.50] (pe_max) |
| **shuffle** | clean16 | 0.52 [0.51, 0.53] (jump_spike) | 0.51 [0.51, 0.53] (jump_spike) | 0.52 [0.52, 0.53] (pe_max) | 0.51 [0.50, 0.52] (jump_spike) | 0.50 [0.50, 0.51] (pe_max) |
| **drop** | clean12 | 0.52 [0.51, 0.53] (jump_spike) | 0.52 [0.51, 0.53] (jump_spike) | 0.52 [0.52, 0.53] (pe_max) | 0.51 [0.50, 0.52] (jump_spike) | 0.50 [0.50, 0.51] (pe_max) |
| **splice_other** | clean16 | 0.85 [0.82, 0.87] (chord_dev) | 0.83 [0.80, 0.86] (chord_dev) | 0.85 [0.82, 0.87] (pe_max) | 0.83 [0.81, 0.86] (chord_dev) | 0.66 [0.64, 0.68] (pe_max) |
| **splice_same** | clean16 | 0.84 [0.81, 0.86] (chord_dev) | 0.82 [0.79, 0.84] (chord_dev) | 0.84 [0.81, 0.86] (pe_max) | 0.82 [0.79, 0.84] (chord_dev) | 0.65 [0.63, 0.67] (pe_max) |
| **freeze** | clean16 | 0.79 [0.77, 0.81] (stall) | 0.79 [0.77, 0.81] (stall) | 0.51 [0.50, 0.51] (pe_max) | 0.79 [0.77, 0.81] (stall) | 0.50 [0.50, 0.50] (pe_max) |
| **speed_up** | clean12 | 0.52 [0.51, 0.53] (step_cv) | 0.52 [0.51, 0.52] (step_cv) | 0.52 [0.51, 0.52] (pe_mean) | 0.51 [0.50, 0.52] (step_cv) | 0.50 [0.50, 0.50] (pe_mean) |
| **slow_down** | clean16 | 0.57 [0.55, 0.58] (step_cv) | 0.53 [0.52, 0.55] (step_cv) | 0.44 [0.43, 0.45] (pe_mean) | 0.53 [0.52, 0.54] (step_cv) | 0.50 [0.50, 0.50] (pe_mean) |
| **appearance** | clean16 | 0.50 [0.50, 0.50] (jump_max) | 0.52 [0.52, 0.53] (jump_max) | 0.63 [0.61, 0.64] (pe_mean) | 0.49 [0.48, 0.49] (jump_max) | 0.46 [0.44, 0.48] (pe_mean) |
| **null_split** | other half | 0.52 [0.48, 0.57] (jump_max) | 0.52 [0.48, 0.57] (jump_max) | 0.52 [0.47, 0.57] (pe_mean) | 0.52 [0.48, 0.57] (jump_max) | 0.53 [0.48, 0.58] (pe_mean) |

### k400: does the largest per-transition value land within one step of the seam?

| injection | chance | raw step | raw tokstep | vicreg3 step | vicreg3 tokstep | vicreg3 pe | vicreg3_last step | vicreg3_last tokstep | vicreg3_last pe |
|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.57 | 0.59 | 0.53 | 0.56 | 0.57 | 0.52 | 0.58 | 0.47 |
| **shuffle** | 0.46 | 0.43 | 0.48 | 0.39 | 0.45 | 0.44 | 0.40 | 0.44 | 0.32 |
| **drop** | 0.27 | 0.33 | 0.36 | 0.32 | 0.34 | 0.34 | 0.30 | 0.34 | 0.26 |
| **splice_other** | 0.40 | 0.84 | 0.83 | 0.81 | 0.84 | 0.83 | 0.82 | 0.83 | 0.54 |
| **splice_same** | 0.40 | 0.81 | 0.84 | 0.79 | 0.83 | 0.83 | 0.80 | 0.81 | 0.54 |
| **freeze** | 0.20 | 0.25 | 0.25 | 0.18 | 0.22 | 0.23 | 0.18 | 0.25 | 0.13 |
