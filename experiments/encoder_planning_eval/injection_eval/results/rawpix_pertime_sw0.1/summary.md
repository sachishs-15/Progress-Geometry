### k400: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| rawpix_pertime_sw0.1 | 0.0726 | 0.0846 | 0.14 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### k400: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | pixel geometry | rawpix_pertime_sw0.1 geometry | rawpix_pertime_sw0.1 predictor |
|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.51 [0.51, 0.52] (pe_mean) |
| **swap** | clean16 | 0.50 [0.49, 0.51] (jump_spike) | 0.51 [0.50, 0.52] (jump_spike) | 0.56 [0.55, 0.57] (pe_max) |
| **shuffle** | clean16 | 0.52 [0.51, 0.52] (jump_spike) | 0.52 [0.51, 0.53] (jump_spike) | 0.57 [0.56, 0.59] (pe_max) |
| **drop** | clean12 | 0.52 [0.51, 0.53] (jump_spike) | 0.52 [0.51, 0.53] (jump_spike) | 0.54 [0.53, 0.55] (pe_max) |
| **splice_other** | clean16 | 0.80 [0.78, 0.83] (chord_dev) | 0.79 [0.76, 0.81] (chord_dev) | 0.88 [0.86, 0.90] (pe_max) |
| **splice_same** | clean16 | 0.80 [0.78, 0.82] (chord_dev) | 0.79 [0.76, 0.81] (chord_dev) | 0.89 [0.87, 0.91] (pe_max) |
| **freeze** | clean16 | 0.79 [0.77, 0.81] (stall) | 0.79 [0.77, 0.81] (stall) | 0.52 [0.51, 0.53] (pe_max) |
| **speed_up** | clean12 | 0.51 [0.50, 0.52] (step_cv) | 0.51 [0.51, 0.52] (step_cv) | 0.53 [0.52, 0.54] (pe_mean) |
| **slow_down** | clean16 | 0.55 [0.54, 0.57] (step_cv) | 0.47 [0.46, 0.48] (step_cv) | 0.50 [0.49, 0.51] (pe_mean) |
| **appearance** | clean16 | 0.49 [0.48, 0.49] (jump_max) | 0.51 [0.51, 0.51] (jump_max) | 0.48 [0.47, 0.49] (pe_mean) |
| **blur** | clean16 | 0.49 [0.49, 0.49] (jump_max) | 0.49 [0.49, 0.50] (jump_max) | 0.50 [0.50, 0.50] (pe_mean) |
| **null_split** | other half | 0.53 [0.49, 0.58] (jump_max) | 0.53 [0.49, 0.58] (jump_max) | 0.53 [0.49, 0.59] (pe_mean) |

### k400: does the largest per-transition value land within one step of the seam?

| injection | chance | pixel step | pixel tokstep | rawpix_pertime_sw0.1 step | rawpix_pertime_sw0.1 tokstep | rawpix_pertime_sw0.1 pe |
|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.56 | 0.56 | 0.59 | 0.59 | 0.60 |
| **shuffle** | 0.46 | 0.42 | 0.42 | 0.46 | 0.46 | 0.45 |
| **drop** | 0.27 | 0.32 | 0.32 | 0.33 | 0.33 | 0.33 |
| **splice_other** | 0.40 | 0.80 | 0.80 | 0.82 | 0.82 | 0.83 |
| **splice_same** | 0.40 | 0.80 | 0.80 | 0.83 | 0.83 | 0.84 |
| **freeze** | 0.20 | 0.23 | 0.23 | 0.27 | 0.27 | 0.29 |

### ssv2: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| rawpix_pertime_sw0.1 | 0.0644 | 0.0584 | -0.10 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### ssv2: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | pixel geometry | rawpix_pertime_sw0.1 geometry | rawpix_pertime_sw0.1 predictor |
|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.51] (pe_mean) |
| **swap** | clean16 | 0.58 [0.55, 0.60] (jump_spike) | 0.65 [0.62, 0.68] (jump_spike) | 0.72 [0.70, 0.74] (pe_max) |
| **shuffle** | clean16 | 0.64 [0.61, 0.66] (jump_spike) | 0.72 [0.70, 0.75] (jump_spike) | 0.73 [0.71, 0.75] (pe_max) |
| **drop** | clean12 | 0.71 [0.69, 0.74] (jump_spike) | 0.74 [0.71, 0.76] (jump_spike) | 0.66 [0.64, 0.67] (pe_max) |
| **splice_other** | clean16 | 0.81 [0.78, 0.83] (chord_dev) | 0.79 [0.76, 0.81] (chord_dev) | 1.00 [0.99, 1.00] (pe_max) |
| **splice_same** | clean16 | 0.79 [0.76, 0.81] (chord_dev) | 0.77 [0.75, 0.80] (chord_dev) | 1.00 [0.99, 1.00] (pe_max) |
| **freeze** | clean16 | 1.00 [0.99, 1.00] (stall) | 1.00 [0.99, 1.00] (stall) | 0.59 [0.58, 0.61] (pe_max) |
| **speed_up** | clean12 | 0.62 [0.60, 0.65] (step_cv) | 0.63 [0.60, 0.65] (step_cv) | 0.57 [0.57, 0.59] (pe_mean) |
| **slow_down** | clean16 | 0.52 [0.49, 0.55] (step_cv) | 0.37 [0.35, 0.39] (step_cv) | 0.49 [0.48, 0.51] (pe_mean) |
| **appearance** | clean16 | 0.49 [0.49, 0.49] (jump_max) | 0.51 [0.50, 0.52] (jump_max) | 0.51 [0.50, 0.52] (pe_mean) |
| **blur** | clean16 | 0.47 [0.47, 0.47] (jump_max) | 0.47 [0.47, 0.48] (jump_max) | 0.49 [0.48, 0.50] (pe_mean) |
| **null_split** | other half | 0.48 [0.43, 0.54] (jump_max) | 0.51 [0.45, 0.56] (jump_max) | 0.50 [0.43, 0.55] (pe_mean) |

### ssv2: does the largest per-transition value land within one step of the seam?

| injection | chance | pixel step | pixel tokstep | rawpix_pertime_sw0.1 step | rawpix_pertime_sw0.1 tokstep | rawpix_pertime_sw0.1 pe |
|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.91 | 0.91 | 0.95 | 0.95 | 0.94 |
| **shuffle** | 0.46 | 0.87 | 0.87 | 0.89 | 0.89 | 0.88 |
| **drop** | 0.27 | 0.75 | 0.75 | 0.74 | 0.74 | 0.73 |
| **splice_other** | 0.40 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
| **splice_same** | 0.40 | 0.98 | 0.98 | 1.00 | 1.00 | 0.99 |
| **freeze** | 0.20 | 0.64 | 0.64 | 0.64 | 0.64 | 0.61 |
