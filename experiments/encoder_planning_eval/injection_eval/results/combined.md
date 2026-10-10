### k400: AUROC of perturbed vs clean (0.5 = chance)

| injection | raw geometry | pixel geometry | vicreg3 geometry | vicreg3 predictor | rawpix geometry | rawpix predictor |
|---|---|---|---|---|---|---|
| **reverse** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.51 |
| **swap** | 0.51 | 0.50 | 0.51 | 0.52 | 0.49 | 0.56 |
| **shuffle** | 0.52 | 0.52 | 0.51 | 0.52 | 0.51 | 0.57 |
| **drop** | 0.52 | 0.52 | 0.52 | 0.52 | 0.51 | 0.53 |
| **splice_other** | 0.85 | 0.80 | 0.83 | 0.85 | 0.73 | 0.77 |
| **splice_same** | 0.84 | 0.80 | 0.82 | 0.84 | 0.73 | 0.77 |
| **freeze** | 0.79 | 0.79 | 0.79 | 0.51 | 0.79 | 0.53 |
| **speed_up** | 0.52 | 0.51 | 0.52 | 0.52 | 0.51 | 0.52 |
| **slow_down** | 0.57 | 0.55 | 0.53 | 0.44 | 0.47 | 0.48 |
| **appearance** | 0.50 | 0.49 | 0.52 | 0.63 | 0.50 | 0.57 |
| **blur** | - | 0.49 | - | - | 0.50 | 0.50 |
| **null_split** | 0.52 | 0.53 | 0.52 | 0.52 | 0.53 | 0.52 |
| *predictor skill* |  |  |  | 0.10 |  | 0.24 |

### ssv2: AUROC of perturbed vs clean (0.5 = chance)

| injection | raw geometry | pixel geometry | vicreg3 geometry | vicreg3 predictor | rawpix geometry | rawpix predictor |
|---|---|---|---|---|---|---|
| **reverse** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| **swap** | 0.67 | 0.58 | 0.62 | 0.73 | 0.59 | 0.74 |
| **shuffle** | 0.72 | 0.64 | 0.68 | 0.74 | 0.66 | 0.76 |
| **drop** | 0.70 | 0.71 | 0.63 | 0.65 | 0.72 | 0.66 |
| **splice_other** | 0.93 | 0.81 | 0.86 | 0.98 | 0.62 | 0.99 |
| **splice_same** | 0.92 | 0.79 | 0.83 | 0.97 | 0.61 | 0.99 |
| **freeze** | 1.00 | 1.00 | 1.00 | 0.61 | 1.00 | 0.62 |
| **speed_up** | 0.62 | 0.62 | 0.59 | 0.55 | 0.62 | 0.56 |
| **slow_down** | 0.62 | 0.52 | 0.57 | 0.29 | 0.36 | 0.46 |
| **appearance** | 0.50 | 0.49 | 0.51 | 0.52 | 0.50 | 0.50 |
| **blur** | - | 0.47 | - | - | 0.47 | 0.49 |
| **null_split** | 0.51 | 0.48 | 0.50 | 0.51 | 0.53 | 0.51 |
| *predictor skill* |  |  |  | 0.03 |  | -0.09 |
