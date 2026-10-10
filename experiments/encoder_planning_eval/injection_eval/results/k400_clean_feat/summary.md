### k400: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)

| readout | model error | copy error | skill |
|---|---|---|---|
| vicreg3 | 0.1127 | 0.1286 | 0.12 |
| vicreg3_last | 0.5705 | 0.0763 | -6.47 |

Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little about whether a clip is anomalous.

### k400: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips

| injection | reference | raw geometry | vicreg3 geometry | vicreg3 predictor | vicreg3_last geometry | vicreg3_last predictor |
|---|---|---|---|---|---|---|
| **reverse** | clean16 | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.50, 0.50] (pe_mean) | 0.50 [0.50, 0.50] (cv_mean) | 0.50 [0.49, 0.51] (pe_mean) |
| **swap** | clean16 | 0.51 [0.50, 0.53] (jump_spike) | 0.51 [0.50, 0.53] (jump_spike) | 0.57 [0.56, 0.58] (pe_max) | 0.50 [0.48, 0.51] (jump_spike) | 0.50 [0.50, 0.50] (pe_max) |
| **shuffle** | clean16 | 0.55 [0.54, 0.57] (jump_spike) | 0.54 [0.52, 0.55] (jump_spike) | 0.57 [0.56, 0.58] (pe_max) | 0.53 [0.51, 0.54] (jump_spike) | 0.51 [0.50, 0.51] (pe_max) |
| **drop** | clean12 | 0.58 [0.55, 0.60] (jump_spike) | 0.56 [0.54, 0.58] (jump_spike) | 0.56 [0.55, 0.57] (pe_max) | 0.56 [0.53, 0.57] (jump_spike) | 0.52 [0.51, 0.53] (pe_max) |
| **splice_other** | clean16 | 0.91 [0.89, 0.93] (chord_dev) | 0.88 [0.86, 0.90] (chord_dev) | 0.91 [0.89, 0.93] (pe_max) | 0.85 [0.83, 0.88] (chord_dev) | 0.65 [0.63, 0.68] (pe_max) |
| **splice_same** | clean16 | 0.87 [0.84, 0.88] (chord_dev) | 0.79 [0.76, 0.82] (chord_dev) | 0.83 [0.81, 0.85] (pe_max) | 0.78 [0.76, 0.81] (chord_dev) | 0.60 [0.59, 0.62] (pe_max) |
| **freeze** | clean16 | 1.00 [0.99, 1.00] (stall) | 1.00 [0.99, 1.00] (stall) | 0.52 [0.51, 0.52] (pe_max) | 1.00 [0.99, 1.00] (stall) | 0.49 [0.49, 0.49] (pe_max) |
| **speed_up** | clean12 | 0.55 [0.54, 0.57] (step_cv) | 0.54 [0.52, 0.56] (step_cv) | 0.56 [0.55, 0.57] (pe_mean) | 0.53 [0.51, 0.55] (step_cv) | 0.50 [0.50, 0.51] (pe_mean) |
| **slow_down** | clean16 | 0.71 [0.68, 0.73] (step_cv) | 0.60 [0.58, 0.63] (step_cv) | 0.33 [0.32, 0.34] (pe_mean) | 0.60 [0.58, 0.63] (step_cv) | 0.49 [0.49, 0.50] (pe_mean) |
| **appearance** | clean16 | 0.50 [0.50, 0.50] (jump_max) | 0.52 [0.51, 0.52] (jump_max) | 0.53 [0.52, 0.54] (pe_mean) | 0.51 [0.50, 0.51] (jump_max) | 0.50 [0.48, 0.52] (pe_mean) |
| **null_split** | other half | 0.52 [0.47, 0.57] (jump_max) | 0.52 [0.47, 0.57] (jump_max) | 0.54 [0.49, 0.59] (pe_mean) | 0.51 [0.47, 0.56] (jump_max) | 0.49 [0.45, 0.55] (pe_mean) |

### k400: does the largest per-transition value land within one step of the seam?

| injection | chance | raw step | raw tokstep | vicreg3 step | vicreg3 tokstep | vicreg3 pe | vicreg3_last step | vicreg3_last tokstep | vicreg3_last pe |
|---|---|---|---|---|---|---|---|---|---|
| **swap** | 0.60 | 0.77 | 0.83 | 0.73 | 0.78 | 0.78 | 0.70 | 0.77 | 0.58 |
| **shuffle** | 0.46 | 0.76 | 0.80 | 0.70 | 0.76 | 0.75 | 0.67 | 0.74 | 0.51 |
| **drop** | 0.27 | 0.48 | 0.53 | 0.44 | 0.48 | 0.48 | 0.41 | 0.48 | 0.32 |
| **splice_other** | 0.40 | 0.93 | 0.96 | 0.93 | 0.93 | 0.92 | 0.91 | 0.92 | 0.56 |
| **splice_same** | 0.40 | 0.92 | 0.95 | 0.89 | 0.91 | 0.90 | 0.88 | 0.90 | 0.55 |
| **freeze** | 0.20 | 0.40 | 0.44 | 0.34 | 0.41 | 0.39 | 0.33 | 0.40 | 0.17 |
