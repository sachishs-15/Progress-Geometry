## card

| metric | floor | DINOv2-S | random walk | smooth σ=1 | ceiling | position | MDD | range/MDD | ICC | sensitive window r (10–90%) | whiten (×MDD) | smooth σ=1 (% range) | reversal-blind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| time_progress_spearman | 0.307 | 0.904 | 0.984 | 0.966 | 1.000 | 0.861 | 0.005 | 150.176 | 0.846 | 0.00112 – 0.00527 | 3.553 | 8.954 | yes |
| pc1_time_spearman | 0.123 | 0.697 | 0.946 | 0.780 | 1.000 | 0.654 | 0.013 | 67.560 | 0.837 | 0.00109 – 0.00375 | 1.886 | 9.554 | yes |
| monotonic_progress_fraction | 0.562 | 0.733 | 0.879 | 0.899 | 1.000 | 0.391 | 0.006 | 76.006 | 0.506 | 0.00191 – 0.0229 | 3.827 | 37.875 | yes |
| progress_efficiency | 0.225 | 0.720 | 0.934 | 0.943 | 1.000 | 0.638 | 0.005 | 145.066 | 0.812 | 0.0016 – 0.0164 | 7.500 | 28.807 | yes |
| adjacent_temporal_nn_fraction | 0.125 | 0.787 | 0.907 | 0.999 | 1.000 | 0.757 | 0.005 | 164.365 | 0.513 | 0.00282 – 0.122 | 2.758 | 24.230 | yes |
| distance_to_goal_spearman | 0.003 | 0.611 | 0.954 | 0.762 | 1.000 | 0.610 | 0.012 | 83.475 | 0.917 | 0.00116 – 0.00708 | 0.622 | 15.152 | no |
| trajectory_straightness | 0.067 | 0.135 | 0.241 | 0.304 | 0.968 | 0.075 | 0.001 | 914.054 | 0.965 | 0.0471 – 130 | -11.000 | 18.792 | yes |
| mean_velocity_cosine_raw | -0.468 | -0.263 | 0.000 | 0.654 | 1.000 | 0.140 | 0.002 | 937.845 | 0.773 | 0.216 – 17.7 | -11.687 | 62.438 | yes |
| mean_chord_alignment_raw | 0.062 | 0.115 | 0.238 | 0.281 | 1.000 | 0.057 | 0.001 | 1112.801 | 0.868 | 0.0446 – 9.25 | -4.020 | 17.674 | yes |
| direction_reversal_fraction | 0.963 | 0.894 | 0.499 | 0.005 | 0.000 | 0.071 | 0.005 | 179.027 | 0.664 | 6.29 – 0.00394 | 6.671 | 92.365 | yes |

## encoders

| encoder | time_prog. | straight. | vel.cos | dir.rev. | adj.NN | dist→goal |
|---|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.904 | 0.135 | -0.263 | 0.894 | 0.787 | 0.611 |
| dinov2_base | 0.898 | 0.134 | -0.272 | 0.902 | 0.768 | 0.604 |
| dinov2_large | 0.890 | 0.137 | -0.267 | 0.872 | 0.755 | 0.602 |
| dinov2_giant | 0.879 | 0.147 | -0.244 | 0.825 | 0.751 | 0.611 |
| dinov3 | 0.911 | 0.133 | -0.260 | 0.901 | 0.804 | 0.609 |
| clip | 0.873 | 0.121 | -0.293 | 0.906 | 0.742 | 0.569 |
| siglip | 0.882 | 0.118 | -0.311 | 0.933 | 0.725 | 0.568 |
| levljepa | 0.891 | 0.126 | -0.283 | 0.905 | 0.758 | 0.603 |
| vjepa2 | 0.883 | 0.178 | -0.132 | 0.635 | 0.763 | 0.617 |
| vjepa2.1 | 0.893 | 0.172 | -0.162 | 0.758 | 0.856 | 0.629 |
| *floor (shuffled), range over encoders* | 0.26–0.31 | 0.07–0.07 | -0.47–-0.45 | 0.89–0.98 | 0.12–0.13 | -0.00–0.00 |
| *σ=1 smoothing of the same features, range* | 0.94–0.97 | 0.29–0.35 | 0.63–0.71 | 0.00–0.02 | 0.99–1.00 | 0.74–0.77 |

## planning

| encoder | R² static midpoint | R² interp (reported) | temporal part [95% CI] | R² ridge | R² adjacent-frame oracle | R² noise ceiling | cos: mean-predictor | cos: interp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vjepa2.1 | 0.750 | 0.793 | +0.043 [0.040, 0.046] | 0.807 | 0.938 | 0.962 | 0.960 | 0.995 |
| dinov2_giant | 0.679 | 0.730 | +0.051 [0.048, 0.055] | 0.732 | 0.882 | 0.924 | 0.276 | 0.910 |
| dinov3 | 0.675 | 0.716 | +0.041 [0.038, 0.044] | 0.732 | 0.876 | 0.920 | 0.450 | 0.908 |
| dinov2_large | 0.639 | 0.689 | +0.050 [0.047, 0.053] | 0.674 | 0.855 | 0.906 | 0.311 | 0.891 |
| dinov2_base | 0.607 | 0.659 | +0.052 [0.049, 0.055] | 0.647 | 0.839 | 0.895 | 0.334 | 0.880 |
| dinov2 | 0.604 | 0.657 | +0.053 [0.050, 0.056] | 0.660 | 0.846 | 0.900 | 0.352 | 0.878 |
| vjepa2 | 0.584 | 0.650 | +0.066 [0.062, 0.070] | 0.677 | 0.894 | 0.934 | 0.827 | 0.963 |
| siglip | 0.610 | 0.645 | +0.035 [0.033, 0.038] | – | 0.826 | 0.887 | 0.794 | 0.943 |
| clip | 0.570 | 0.608 | +0.037 [0.034, 0.040] | – | 0.814 | 0.879 | 0.753 | 0.924 |
| levljepa | 0.489 | 0.546 | +0.057 [0.053, 0.061] | 0.567 | 0.783 | 0.859 | 0.759 | 0.912 |

## noise_study

| metric | raw DINOv2 | PCA-whitened raw | untrained | no reg | VICReg sw0.1 | VICReg sw1 | SigReg |
|---|---:|---:|---:|---:|---:|---:|---:|
| time_progress_spearman | 0.9040 | 0.9204 | 0.9035 ± 0.0002 | 0.5268 ± 0.0018 | 0.9152 ± 0.0009 | 0.9060 ± 0.0006 | 0.9098 ± 0.0005 |
| trajectory_straightness | 0.1348 | 0.1240 | 0.1347 ± 0.0001 | 0.2096 ± 0.0012 | 0.1241 ± 0.0000 | 0.1296 ± 0.0002 | 0.1288 ± 0.0001 |
| mean_velocity_cosine_raw | -0.2627 | -0.2810 | -0.2639 ± 0.0001 | -0.0714 ± 0.0041 | -0.2803 ± 0.0002 | -0.2620 ± 0.0004 | -0.2721 ± 0.0001 |
| direction_reversal_fraction | 0.8941 | 0.9300 | 0.8967 ± 0.0006 | 0.5423 ± 0.0031 | 0.9310 ± 0.0007 | 0.9061 ± 0.0011 | 0.9160 ± 0.0003 |
| adjacent_temporal_nn | 0.7873 | 0.8020 | 0.7870 ± 0.0008 | 0.5206 ± 0.0066 | 0.7969 ± 0.0008 | 0.7976 ± 0.0007 | 0.7916 ± 0.0004 |
| distance_to_goal | 0.6110 | 0.6185 | 0.6038 ± 0.0008 | 0.3963 ± 0.0040 | 0.5763 ± 0.0015 | 0.5720 ± 0.0013 | 0.5880 ± 0.0007 |
| planning R² (interp) | 0.6570 | 0.5451 | 0.6484 ± 0.0003 | 0.2794 ± 0.0047 | 0.4815 ± 0.0004 | 0.4775 ± 0.0006 | 0.5650 ± 0.0006 |
| planning R² (static midpoint) | 0.6042 | 0.4854 | 0.5962 ± 0.0004 | 0.1558 ± 0.0051 | 0.4236 ± 0.0005 | 0.4151 ± 0.0008 | 0.5092 ± 0.0007 |
| effective_rank_fraction | 0.8120 | 0.9425 | 0.8070 ± 0.0003 | 0.4311 ± 0.0146 | 0.9059 ± 0.0001 | 0.8823 ± 0.0001 | 0.8574 ± 0.0003 |
| linear probe c+c | 0.5242 | 0.4221 | 0.5026 ± 0.0181 | 0.4548 ± 0.0070 | 0.4183 ± 0.0085 | 0.4602 ± 0.0047 | 0.4838 ± 0.0059 |
| kNN probe c+c | 0.3434 | 0.3526 | 0.3418 ± 0.0068 | 0.1510 ± 0.0074 | 0.3681 ± 0.0113 | 0.4174 ± 0.0095 | 0.3692 ± 0.0078 |
| chirality acc. | 0.7993 | 0.7718 | 0.7986 ± 0.0064 | 0.7535 ± 0.0084 | 0.7855 ± 0.0093 | 0.7917 ± 0.0096 | 0.7850 ± 0.0055 |
| change-direction gap | 0.0386 | 0.0171 | 0.0385 ± 0.0004 | 0.1405 ± 0.0050 | 0.0219 ± 0.0003 | 0.0287 ± 0.0007 | 0.0299 ± 0.0005 |
| TCC τ | 0.0661 | 0.0339 | 0.0626 ± 0.0033 | 0.0627 ± 0.0055 | 0.0546 ± 0.0047 | 0.0774 ± 0.0019 | 0.0495 ± 0.0047 |

## probes

| encoder | chance | perm-null | linear c+c | val-boot 95% CI | sd over 5 train subsamples | ceiling (all train) | chirality | chir. CI | chir. order-blind | chir. shuffled-label | TCC τ | τ null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.045 | 0.045±0.012 | 0.524 | [0.475, 0.572] | 0.012 | 0.540 | 0.799 | [0.766, 0.829] | 0.523 | 0.502 | 0.076 | 0.012 |
| dinov2_base | 0.045 | 0.046±0.014 | 0.464 | [0.414, 0.515] | 0.022 | 0.499 | 0.811 | [0.778, 0.843] | 0.512 | 0.506 | 0.094 | -0.002 |
| dinov2_large | 0.045 | 0.047±0.011 | 0.459 | [0.415, 0.509] | 0.021 | 0.473 | 0.810 | [0.780, 0.837] | 0.493 | 0.502 | 0.078 | 0.011 |
| dinov3 | 0.045 | 0.044±0.010 | 0.461 | [0.418, 0.508] | 0.011 | 0.483 | 0.793 | [0.763, 0.823] | 0.531 | 0.505 | 0.090 | 0.006 |
| levljepa | 0.045 | 0.044±0.008 | 0.367 | [0.323, 0.415] | 0.009 | 0.430 | 0.715 | [0.682, 0.749] | 0.515 | 0.507 | 0.056 | 0.002 |
| vjepa2.1 | 0.045 | 0.044±0.009 | 0.541 | [0.494, 0.593] | 0.004 | 0.580 | 0.787 | [0.750, 0.817] | 0.509 | 0.500 | 0.163 | 0.018 |
| vjepa2 | 0.045 | 0.048±0.014 | 0.524 | [0.480, 0.570] | 0.006 | 0.570 | 0.783 | [0.748, 0.815] | 0.493 | 0.509 | 0.143 | 0.020 |

## skill

| system | skill | note |
|---|---:|---:|
| copy last frame | 0.000 | by definition |
| EMA(0.5) of past frames (no learning) | 0.079 | free denoiser |
| ridge AR(1), raw latents | 0.074 | learned, linear |
| ridge AR(3), raw latents | 0.120 | learned, linear |
| noise ceiling, raw latents | 0.599 | upper bound on any predictor |
| trained predictor, no reg (n=5) | 0.053 | sd 0.0118; ridge AR(3) in its own latents 0.074; noise ceiling 0.728 |
| trained predictor, VICReg sw0.1 (n=5) | 0.103 | sd 0.0014; ridge AR(3) in its own latents 0.139; noise ceiling 0.584 |
| trained predictor, SigReg (n=5) | 0.055 | sd 0.0006; ridge AR(3) in its own latents 0.131; noise ceiling 0.590 |
| trained predictor, VICReg sw1 (n=5) | 0.079 | sd 0.0022; ridge AR(3) in its own latents 0.124; noise ceiling 0.596 |

## synth

**time_progress_spearman**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.305 | 0.334 | 0.843 | 0.584 | 0.859 | 0.991 | 0.990 |
| dinov2_base | 0.307 | 0.323 | 0.753 | 0.526 | 0.778 | 0.990 | 0.990 |
| clip | 0.303 | 0.322 | 0.852 | 0.632 | 0.881 | 0.963 | 0.982 |
| siglip | 0.317 | 0.342 | 0.839 | 0.728 | 0.836 | 0.922 | 0.985 |
| pixel | 0.282 | 0.334 | 0.992 | 0.865 | 0.992 | 1.000 | 0.989 |

**trajectory_straightness**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.066 | 0.066 | 0.124 | 0.090 | 0.135 | 0.339 | 0.573 |
| dinov2_base | 0.066 | 0.066 | 0.128 | 0.085 | 0.138 | 0.298 | 0.532 |
| clip | 0.066 | 0.066 | 0.110 | 0.083 | 0.123 | 0.255 | 0.420 |
| siglip | 0.066 | 0.066 | 0.120 | 0.086 | 0.129 | 0.173 | 0.373 |
| pixel | 0.064 | 0.067 | 0.213 | 0.132 | 0.213 | 0.368 | 0.334 |

**mean_velocity_cosine_raw**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | -0.482 | -0.493 | -0.379 | -0.419 | -0.250 | -0.184 | 0.000 |
| dinov2_base | -0.484 | -0.493 | -0.362 | -0.440 | -0.271 | -0.132 | 0.000 |
| clip | -0.486 | -0.496 | -0.443 | -0.456 | -0.261 | -0.205 | 0.000 |
| siglip | -0.490 | -0.491 | -0.354 | -0.446 | -0.240 | 0.013 | 0.000 |
| pixel | -0.405 | -0.487 | 0.592 | -0.240 | 0.591 | 0.570 | 0.000 |

**direction_reversal_fraction**

| encoder | static_noise | iid_scenes | linear_clean | linear_both | accel_clean | fill_level | step_change |
|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.948 | 0.984 | 0.886 | 0.907 | 0.770 | 0.705 | 0.000 |
| dinov2_base | 0.944 | 0.990 | 0.824 | 0.919 | 0.762 | 0.674 | 0.000 |
| clip | 0.975 | 0.997 | 0.970 | 0.976 | 0.808 | 0.749 | 0.000 |
| siglip | 0.990 | 0.996 | 0.926 | 0.979 | 0.808 | 0.463 | 0.000 |
| pixel | 0.730 | 0.968 | 0.000 | 0.714 | 0.010 | 0.000 | 0.000 |

## audit

| contrast | metric | Δ | 95% CI (paired, clips) | excludes 0 |
|---|---:|---:|---:|---:|
| sw0.5 − sw0.1 | time_progress_spearman | 0.0921 | [+0.0837, +0.1001] | yes |
| sw0.5 − sw0.1 | trajectory_straightness | 0.0466 | [+0.0449, +0.0483] | yes |
| sw0.5 − sw0.1 | mean_velocity_cosine | 0.1762 | [+0.1718, +0.1806] | yes |
| sw0.5 − sw0.1 | direction_reversal_fraction | -0.1599 | [-0.1684, -0.1520] | yes |
| sw1 − sw0.5 | time_progress_spearman | -0.0836 | [-0.0921, -0.0754] | yes |
| sw1 − sw0.5 | trajectory_straightness | 0.0561 | [+0.0528, +0.0594] | yes |
| sw1 − sw0.5 | mean_velocity_cosine | 0.1373 | [+0.1313, +0.1434] | yes |
| sw1 − sw0.5 | direction_reversal_fraction | -0.2715 | [-0.2806, -0.2632] | yes |
| sw1.5 − sw1 | time_progress_spearman | -0.0745 | [-0.0833, -0.0657] | yes |
| sw1.5 − sw1 | trajectory_straightness | 0.0155 | [+0.0126, +0.0183] | yes |
| sw1.5 − sw1 | mean_velocity_cosine | 0.0132 | [+0.0073, +0.0198] | yes |
| sw1.5 − sw1 | direction_reversal_fraction | -0.0236 | [-0.0314, -0.0156] | yes |
| sw2 − sw1.5 | time_progress_spearman | -0.0456 | [-0.0562, -0.0351] | yes |
| sw2 − sw1.5 | trajectory_straightness | -0.0007 | [-0.0039, +0.0025] | **no** |
| sw2 − sw1.5 | mean_velocity_cosine | -0.0187 | [-0.0263, -0.0112] | yes |
| sw2 − sw1.5 | direction_reversal_fraction | 0.0086 | [+0.0010, +0.0159] | yes |

## amp

**time_progress_spearman**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.662 | 0.736 | 0.889 | 0.919 | 0.944 |
| dinov2 | 0.582 | 0.653 | 0.860 | 0.933 | 0.973 |
| dinov3 | 0.527 | 0.613 | 0.831 | 0.948 | 0.979 |
| pixel | 0.814 | 0.939 | 0.997 | 1.000 | 1.000 |
| vjepa2.1 | 0.589 | 0.707 | 0.948 | 0.993 | 0.954 |

**trajectory_straightness**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.092 | 0.094 | 0.113 | 0.170 | 0.179 |
| dinov2 | 0.097 | 0.101 | 0.128 | 0.182 | 0.204 |
| dinov3 | 0.079 | 0.081 | 0.104 | 0.153 | 0.188 |
| pixel | 0.104 | 0.146 | 0.222 | 0.343 | 0.436 |
| vjepa2.1 | 0.102 | 0.119 | 0.170 | 0.244 | 0.336 |

**mean_velocity_cosine_raw**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | -0.443 | -0.483 | -0.409 | -0.303 | -0.094 |
| dinov2 | -0.407 | -0.445 | -0.355 | -0.232 | -0.105 |
| dinov3 | -0.468 | -0.509 | -0.441 | -0.328 | -0.153 |
| pixel | 0.050 | 0.426 | 0.610 | 0.637 | 0.623 |
| vjepa2.1 | -0.324 | -0.280 | -0.139 | 0.019 | 0.289 |

**direction_reversal_fraction**

| encoder | r=6px | r=12px | r=24px | r=48px | r=90px |
|---|---:|---:|---:|---:|---:|
| clip | 0.943 | 0.965 | 0.963 | 0.896 | 0.625 |
| dinov2 | 0.899 | 0.887 | 0.870 | 0.790 | 0.652 |
| dinov3 | 0.930 | 0.941 | 0.951 | 0.909 | 0.749 |
| pixel | 0.461 | 0.005 | 0.000 | 0.000 | 0.000 |
| vjepa2.1 | 0.766 | 0.776 | 0.708 | 0.446 | 0.106 |

## reliability

| metric | clip | dinov2 | dinov3 | clip-level use |
|---|---:|---:|---:|---:|
| within_clip_mean_pairwise_l2 | 0.989 | 0.994 | 0.994 | reliable |
| start_end_displacement | 0.957 | 0.982 | 0.979 | reliable |
| adjacent_l2_distance | 0.971 | 0.979 | 0.981 | reliable |
| adjacent_cosine_distance | 0.962 | 0.974 | 0.975 | reliable |
| trajectory_straightness_raw | 0.921 | 0.965 | 0.962 | reliable |
| trajectory_straightness | 0.925 | 0.965 | 0.963 | reliable |
| norm_std_within_clip | 0.924 | 0.964 | 0.946 | reliable |
| norm_growth | 0.894 | 0.942 | 0.932 | reliable |
| distance_to_goal_spearman | 0.886 | 0.917 | 0.919 | reliable |
| mean_chord_alignment | 0.789 | 0.869 | 0.876 | moderate |
| mean_chord_alignment_raw | 0.781 | 0.868 | 0.876 | moderate |
| time_progress_spearman | 0.778 | 0.846 | 0.826 | moderate |
| pc1_time_spearman | 0.821 | 0.837 | 0.834 | moderate |
| progress_efficiency | 0.775 | 0.812 | 0.812 | moderate |
| mean_velocity_cosine | 0.711 | 0.774 | 0.763 | moderate |
| mean_velocity_cosine_raw | 0.708 | 0.773 | 0.763 | moderate |
| direction_reversal_fraction | 0.619 | 0.664 | 0.654 | unreliable |
| adjacent_temporal_nn_fraction | 0.508 | 0.513 | 0.502 | unreliable |
| monotonic_progress_fraction | 0.476 | 0.506 | 0.553 | unreliable |
| norm_turning_point_fraction | 0.265 | 0.318 | 0.281 | unreliable |
