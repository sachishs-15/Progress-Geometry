### k400: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| rawpix | 0.3671 | 0.4610 | 0.20 |
| rawpix_last | 0.6239 | 0.8072 | 0.23 |
| rawpix_pertime_sw0.1 | 0.1194 | 0.1503 | 0.21 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### k400: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | pixel geometry | rawpix geometry | rawpix predictor | rawpix_last geometry | rawpix_last predictor | rawpix_pertime_sw0.1 geometry | rawpix_pertime_sw0.1 predictor |
|---|---|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.52 [0.52, 0.53] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.55 [0.54, 0.55] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.53 [0.53, 0.54] (pe_mean) |
| **swap** | clean16 | 0.49 [0.47, 0.50] (jump_spike) | 0.47 [0.46, 0.49] (jump_spike) | 0.65 [0.63, 0.67] (pe_max) | 0.47 [0.46, 0.49] (jump_spike) | 0.72 [0.70, 0.74] (pe_max) | 0.50 [0.48, 0.52] (jump_spike) | 0.63 [0.62, 0.64] (pe_max) |
| **shuffle** | clean16 | 0.53 [0.52, 0.55] (jump_spike) | 0.52 [0.50, 0.53] (jump_spike) | 0.66 [0.64, 0.68] (pe_max) | 0.52 [0.51, 0.53] (jump_spike) | 0.73 [0.71, 0.75] (pe_max) | 0.54 [0.53, 0.56] (jump_spike) | 0.64 [0.63, 0.66] (pe_max) |
| **drop** | clean12 | 0.55 [0.52, 0.57] (jump_spike) | 0.53 [0.51, 0.56] (jump_spike) | 0.57 [0.55, 0.59] (pe_max) | 0.54 [0.52, 0.56] (jump_spike) | 0.59 [0.57, 0.61] (pe_max) | 0.55 [0.53, 0.58] (jump_spike) | 0.56 [0.55, 0.58] (pe_max) |
| **splice_other** | clean16 | 0.75 [0.73, 0.78] (chord_dev) | 0.58 [0.56, 0.60] (chord_dev) | 0.92 [0.90, 0.94] (pe_max) | 0.53 [0.51, 0.55] (chord_dev) | 0.92 [0.90, 0.93] (pe_max) | 0.74 [0.72, 0.77] (chord_dev) | 0.99 [0.99, 1.00] (pe_max) |
| **splice_same** | clean16 | 0.74 [0.72, 0.76] (chord_dev) | 0.59 [0.57, 0.61] (chord_dev) | 0.91 [0.89, 0.93] (pe_max) | 0.54 [0.52, 0.56] (chord_dev) | 0.92 [0.90, 0.93] (pe_max) | 0.74 [0.72, 0.76] (chord_dev) | 0.99 [0.98, 0.99] (pe_max) |
| **freeze** | clean16 | 1.00 [0.99, 1.00] (stall) | 1.00 [0.99, 1.00] (stall) | 0.56 [0.55, 0.58] (pe_max) | 1.00 [0.99, 1.00] (stall) | 0.59 [0.57, 0.60] (pe_max) | 1.00 [0.99, 1.00] (stall) | 0.55 [0.54, 0.56] (pe_max) |
| **speed_up** | clean12 | 0.52 [0.51, 0.54] (step_cv) | 0.51 [0.49, 0.53] (step_cv) | 0.55 [0.55, 0.56] (pe_mean) | 0.51 [0.49, 0.53] (step_cv) | 0.55 [0.54, 0.56] (pe_mean) | 0.53 [0.51, 0.55] (step_cv) | 0.56 [0.55, 0.56] (pe_mean) |
| **slow_down** | clean16 | 0.65 [0.63, 0.67] (step_cv) | 0.43 [0.42, 0.45] (step_cv) | 0.45 [0.44, 0.46] (pe_mean) | 0.44 [0.42, 0.46] (step_cv) | 0.46 [0.46, 0.47] (pe_mean) | 0.41 [0.39, 0.43] (step_cv) | 0.50 [0.49, 0.51] (pe_mean) |
| **appearance** | clean16 | 0.49 [0.49, 0.49] (jump_max) | 0.51 [0.50, 0.51] (jump_max) | 0.51 [0.51, 0.52] (pe_mean) | 0.51 [0.50, 0.52] (jump_max) | 0.54 [0.53, 0.55] (pe_mean) | 0.51 [0.51, 0.52] (jump_max) | 0.54 [0.53, 0.54] (pe_mean) |
| **blur** | clean16 | 0.48 [0.48, 0.48] (jump_max) | 0.48 [0.48, 0.49] (jump_max) | 0.49 [0.48, 0.49] (pe_mean) | 0.49 [0.48, 0.50] (jump_max) | 0.53 [0.53, 0.54] (pe_mean) | 0.48 [0.48, 0.48] (jump_max) | 0.49 [0.49, 0.50] (pe_mean) |
| **null_split** | other half | 0.54 [0.49, 0.59] (jump_max) | 0.53 [0.48, 0.58] (jump_max) | 0.52 [0.47, 0.57] (pe_mean) | 0.53 [0.48, 0.59] (jump_max) | 0.52 [0.47, 0.57] (pe_mean) | 0.51 [0.46, 0.57] (jump_max) | 0.51 [0.46, 0.57] (pe_mean) |

### k400: does the largest per-transition value land within one step of the seam?

| injection | chance | pixel step | pixel tokstep | rawpix step | rawpix tokstep | rawpix pe | rawpix_last step | rawpix_last tokstep | rawpix_last pe | rawpix_pertime_sw0.1 step | rawpix_pertime_sw0.1 tokstep | rawpix_pertime_sw0.1 pe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.77 | 0.77 | 0.79 | 0.79 | 0.80 | 0.81 | 0.81 | 0.86 | 0.82 | 0.82 | 0.83 |
| **shuffle** | 0.46 | 0.73 | 0.73 | 0.77 | 0.77 | 0.77 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.82 |
| **drop** | 0.27 | 0.48 | 0.48 | 0.51 | 0.51 | 0.52 | 0.54 | 0.54 | 0.58 | 0.54 | 0.54 | 0.55 |
| **splice_other** | 0.40 | 0.88 | 0.88 | 0.89 | 0.89 | 0.91 | 0.86 | 0.86 | 0.93 | 0.96 | 0.96 | 0.99 |
| **splice_same** | 0.40 | 0.88 | 0.88 | 0.90 | 0.90 | 0.94 | 0.84 | 0.84 | 0.93 | 0.96 | 0.96 | 0.99 |
| **freeze** | 0.20 | 0.41 | 0.41 | 0.39 | 0.39 | 0.45 | 0.43 | 0.43 | 0.52 | 0.43 | 0.43 | 0.46 |
