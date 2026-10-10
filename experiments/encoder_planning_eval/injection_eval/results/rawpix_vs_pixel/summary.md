### ssv2: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| rawpix | 0.2010 | 0.1845 | -0.09 |
| rawpix_last | 0.4466 | 0.3702 | -0.21 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### ssv2: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | pixel geometry | rawpix geometry | rawpix predictor | rawpix_last geometry | rawpix_last predictor |
|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.49, 0.51] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.49, 0.51] (pe_mean) |
| **swap** | clean16 | 0.58 [0.55, 0.60] (jump_spike) | 0.59 [0.56, 0.63] (jump_spike) | 0.74 [0.72, 0.76] (pe_max) | 0.56 [0.54, 0.60] (jump_spike) | 0.74 [0.72, 0.76] (pe_max) |
| **shuffle** | clean16 | 0.64 [0.61, 0.66] (jump_spike) | 0.66 [0.63, 0.69] (jump_spike) | 0.76 [0.73, 0.77] (pe_max) | 0.63 [0.60, 0.66] (jump_spike) | 0.75 [0.73, 0.77] (pe_max) |
| **drop** | clean12 | 0.71 [0.69, 0.74] (jump_spike) | 0.72 [0.69, 0.75] (jump_spike) | 0.66 [0.64, 0.68] (pe_max) | 0.72 [0.69, 0.75] (jump_spike) | 0.65 [0.63, 0.67] (pe_max) |
| **splice_other** | clean16 | 0.81 [0.78, 0.83] (chord_dev) | 0.62 [0.60, 0.65] (chord_dev) | 0.99 [0.99, 1.00] (pe_max) | 0.58 [0.55, 0.60] (chord_dev) | 0.98 [0.97, 0.99] (pe_max) |
| **splice_same** | clean16 | 0.79 [0.76, 0.81] (chord_dev) | 0.61 [0.58, 0.64] (chord_dev) | 0.99 [0.98, 1.00] (pe_max) | 0.56 [0.53, 0.59] (chord_dev) | 0.98 [0.97, 0.98] (pe_max) |
| **freeze** | clean16 | 1.00 [0.99, 1.00] (stall) | 1.00 [0.99, 1.00] (stall) | 0.62 [0.61, 0.64] (pe_max) | 1.00 [0.99, 1.00] (stall) | 0.61 [0.59, 0.63] (pe_max) |
| **speed_up** | clean12 | 0.62 [0.60, 0.65] (step_cv) | 0.62 [0.60, 0.64] (step_cv) | 0.56 [0.55, 0.57] (pe_mean) | 0.62 [0.60, 0.64] (step_cv) | 0.54 [0.53, 0.56] (pe_mean) |
| **slow_down** | clean16 | 0.52 [0.49, 0.55] (step_cv) | 0.36 [0.33, 0.37] (step_cv) | 0.46 [0.45, 0.47] (pe_mean) | 0.36 [0.34, 0.38] (step_cv) | 0.47 [0.46, 0.48] (pe_mean) |
| **appearance** | clean16 | 0.49 [0.49, 0.49] (jump_max) | 0.50 [0.49, 0.51] (jump_max) | 0.50 [0.50, 0.51] (pe_mean) | 0.51 [0.50, 0.51] (jump_max) | 0.52 [0.51, 0.53] (pe_mean) |
| **blur** | clean16 | 0.47 [0.47, 0.47] (jump_max) | 0.47 [0.47, 0.48] (jump_max) | 0.49 [0.48, 0.50] (pe_mean) | 0.47 [0.47, 0.48] (jump_max) | 0.55 [0.53, 0.56] (pe_mean) |
| **null_split** | other half | 0.48 [0.43, 0.54] (jump_max) | 0.53 [0.46, 0.58] (jump_max) | 0.51 [0.45, 0.56] (pe_mean) | 0.52 [0.46, 0.58] (jump_max) | 0.49 [0.44, 0.55] (pe_mean) |

### ssv2: does the largest per-transition value land within one step of the seam?

| injection | chance | pixel step | pixel tokstep | rawpix step | rawpix tokstep | rawpix pe | rawpix_last step | rawpix_last tokstep | rawpix_last pe |
|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.91 | 0.91 | 0.93 | 0.93 | 0.94 | 0.93 | 0.93 | 0.93 |
| **shuffle** | 0.46 | 0.87 | 0.87 | 0.88 | 0.88 | 0.88 | 0.89 | 0.89 | 0.87 |
| **drop** | 0.27 | 0.75 | 0.75 | 0.76 | 0.76 | 0.75 | 0.75 | 0.75 | 0.72 |
| **splice_other** | 0.40 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 |
| **splice_same** | 0.40 | 0.98 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| **freeze** | 0.20 | 0.64 | 0.64 | 0.64 | 0.64 | 0.64 | 0.65 | 0.65 | 0.62 |

### k400: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| rawpix | 0.2012 | 0.2646 | 0.24 |
| rawpix_last | 0.3178 | 0.4470 | 0.29 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### k400: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | pixel geometry | rawpix geometry | rawpix predictor | rawpix_last geometry | rawpix_last predictor |
|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.51 [0.51, 0.52] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.52 [0.52, 0.53] (pe_mean) |
| **swap** | clean16 | 0.50 [0.49, 0.51] (jump_spike) | 0.49 [0.49, 0.50] (jump_spike) | 0.56 [0.55, 0.57] (pe_max) | 0.50 [0.49, 0.51] (jump_spike) | 0.60 [0.58, 0.62] (pe_max) |
| **shuffle** | clean16 | 0.52 [0.51, 0.52] (jump_spike) | 0.51 [0.51, 0.52] (jump_spike) | 0.57 [0.56, 0.58] (pe_max) | 0.52 [0.51, 0.52] (jump_spike) | 0.60 [0.59, 0.62] (pe_max) |
| **drop** | clean12 | 0.52 [0.51, 0.53] (jump_spike) | 0.51 [0.51, 0.53] (jump_spike) | 0.53 [0.52, 0.54] (pe_max) | 0.52 [0.51, 0.53] (jump_spike) | 0.54 [0.53, 0.55] (pe_max) |
| **splice_other** | clean16 | 0.80 [0.78, 0.83] (chord_dev) | 0.73 [0.71, 0.76] (chord_dev) | 0.77 [0.75, 0.79] (pe_max) | 0.72 [0.69, 0.74] (chord_dev) | 0.78 [0.76, 0.81] (pe_max) |
| **splice_same** | clean16 | 0.80 [0.78, 0.82] (chord_dev) | 0.73 [0.71, 0.76] (chord_dev) | 0.77 [0.75, 0.79] (pe_max) | 0.72 [0.69, 0.74] (chord_dev) | 0.78 [0.76, 0.80] (pe_max) |
| **freeze** | clean16 | 0.79 [0.77, 0.81] (stall) | 0.79 [0.77, 0.81] (stall) | 0.53 [0.52, 0.53] (pe_max) | 0.79 [0.77, 0.81] (stall) | 0.55 [0.54, 0.56] (pe_max) |
| **speed_up** | clean12 | 0.51 [0.50, 0.52] (step_cv) | 0.51 [0.50, 0.52] (step_cv) | 0.52 [0.51, 0.52] (pe_mean) | 0.51 [0.50, 0.51] (step_cv) | 0.52 [0.51, 0.52] (pe_mean) |
| **slow_down** | clean16 | 0.55 [0.54, 0.57] (step_cv) | 0.47 [0.47, 0.48] (step_cv) | 0.48 [0.48, 0.49] (pe_mean) | 0.47 [0.46, 0.48] (step_cv) | 0.49 [0.49, 0.50] (pe_mean) |
| **appearance** | clean16 | 0.49 [0.48, 0.49] (jump_max) | 0.50 [0.50, 0.51] (jump_max) | 0.57 [0.56, 0.58] (pe_mean) | 0.51 [0.50, 0.51] (jump_max) | 0.59 [0.58, 0.61] (pe_mean) |
| **blur** | clean16 | 0.49 [0.49, 0.49] (jump_max) | 0.50 [0.49, 0.50] (jump_max) | 0.50 [0.50, 0.50] (pe_mean) | 0.50 [0.50, 0.50] (jump_max) | 0.52 [0.52, 0.53] (pe_mean) |
| **null_split** | other half | 0.53 [0.49, 0.58] (jump_max) | 0.53 [0.49, 0.58] (jump_max) | 0.52 [0.47, 0.57] (pe_mean) | 0.53 [0.48, 0.57] (jump_max) | 0.51 [0.47, 0.56] (pe_mean) |

### k400: does the largest per-transition value land within one step of the seam?

| injection | chance | pixel step | pixel tokstep | rawpix step | rawpix tokstep | rawpix pe | rawpix_last step | rawpix_last tokstep | rawpix_last pe |
|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.56 | 0.56 | 0.57 | 0.57 | 0.58 | 0.59 | 0.59 | 0.64 |
| **shuffle** | 0.46 | 0.42 | 0.42 | 0.46 | 0.46 | 0.46 | 0.47 | 0.47 | 0.49 |
| **drop** | 0.27 | 0.32 | 0.32 | 0.35 | 0.35 | 0.36 | 0.38 | 0.38 | 0.42 |
| **splice_other** | 0.40 | 0.80 | 0.80 | 0.70 | 0.70 | 0.76 | 0.65 | 0.65 | 0.80 |
| **splice_same** | 0.40 | 0.80 | 0.80 | 0.71 | 0.71 | 0.77 | 0.66 | 0.66 | 0.80 |
| **freeze** | 0.20 | 0.23 | 0.23 | 0.26 | 0.26 | 0.29 | 0.29 | 0.29 | 0.34 |
