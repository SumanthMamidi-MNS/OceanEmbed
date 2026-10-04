# R1 rigour: seeds, confidence intervals and paired comparisons (poc)

Test period 2024-01-01 .. 2024-12-15 (350 days), reference GLORYS reanalysis. Temperatures in °C; bias is method minus GLORYS; `skill` is `1 - MSE / MSE_climatology`; `anomaly corr` is the correlation of anomalies (climatology removed from both).

## Method and uncertainty

Seeds per method: OceanEmbed (pretrained encoder) n=5, OceanEmbed (no pretraining) n=5, Plain U-Net (no Transformer) n=5, Per-pixel MLP n=3, Ridge regression n=1, Climatology n=1. Ridge and climatology are deterministic (one evaluation, taken from the main run).

- **Seed spread** (`mean ± SD`): each seed's metric from its own run, sample SD over seeds (n - 1); for `oceanembed` every seed repeats pretraining as well as fine-tuning.
- **95 % CI over test days**: moving-block bootstrap over test days (overlapping blocks, no wrap); 2000 replicates, block length 42 days, 95 % percentile interval of the replicates. The per-day sufficient statistics are averaged over seeds and the metric is recomputed from the resampled days, so the interval reflects test-day sampling of the seed-mean model, not seed variation.
- **Paired difference** `A - B` (negative = A better): computed inside each replicate with the same resampled days for both methods.

### Block length

Daily errors are strongly autocorrelated. Block length = median over methods of the e-folding time (first lag with autocorrelation < 1/e) of the daily pooled 50-200 m MSE, capped at n_days/4: 42 days.

| daily pooled 50-200 m MSE of | e-folding time (days) | integrated autocorrelation time (days) |
|---|---|---|
| OceanEmbed (pretrained encoder) | 41 | 45.3 |
| OceanEmbed (no pretraining) | 39 | 37.6 |
| Plain U-Net (no Transformer) | 44 | 52.3 |
| Per-pixel MLP | 44 | 51.3 |
| Ridge regression | 42 | 54.2 |
| Climatology | 18 | 19.9 |

Half-width of the 95 % interval (°C) of the pooled 50-200 m RMSE, and of the key differences, as a function of block length (length 1 = ignoring autocorrelation):

| block length (days) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression | Climatology | oceanembed-scratch (diff) | scratch-unet (diff) | oceanembed-unet (diff) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0075 | 0.0071 | 0.0103 | 0.0119 | 0.0150 | 0.0136 | 0.0018 | 0.0057 | 0.0048 |
| 7 | 0.0196 | 0.0187 | 0.0280 | 0.0326 | 0.0377 | 0.0365 | 0.0046 | 0.0147 | 0.0123 |
| 15 | 0.0275 | 0.0262 | 0.0410 | 0.0459 | 0.0495 | 0.0434 | 0.0062 | 0.0201 | 0.0168 |
| 30 | 0.0384 | 0.0351 | 0.0561 | 0.0592 | 0.0643 | 0.0489 | 0.0081 | 0.0273 | 0.0218 |
| 42 * | 0.0446 | 0.0415 | 0.0635 | 0.0687 | 0.0675 | 0.0519 | 0.0080 | 0.0296 | 0.0238 |
| 45 | 0.0451 | 0.0417 | 0.0662 | 0.0695 | 0.0720 | 0.0549 | 0.0078 | 0.0300 | 0.0242 |

(* = block length used)

## The two questions

- **Does pretraining matter?** pooled 50-200 m RMSE, OceanEmbed (pretrained encoder) minus OceanEmbed (no pretraining): +0.013 °C, 95 % CI [0.004, 0.020] over test days (bootstrap p = 0.004). Seeds: 1.064 ± 0.016 (n=5) vs 1.052 ± 0.015 (n=5), seed ranges overlap; Welch t-test across seeds p = 0.236. Verdict: the day-resampling interval excludes zero but the seed ranges overlap, so the difference is not established independently of the training seed.
- **Does the Transformer add anything over a U-Net?** pooled 50-200 m RMSE, OceanEmbed (no pretraining) minus Plain U-Net (no Transformer): -0.072 °C, 95 % CI [-0.101, -0.043] over test days (bootstrap p = 0.001). Seeds: 1.052 ± 0.015 (n=5) vs 1.124 ± 0.031 (n=5), seed ranges do not overlap; Welch t-test across seeds p = 0.004. Verdict: distinguishable from zero on both counts.
- **Pretrained Transformer vs U-Net?** pooled 50-200 m RMSE, OceanEmbed (pretrained encoder) minus Plain U-Net (no Transformer): -0.060 °C, 95 % CI [-0.084, -0.036] over test days (bootstrap p = 0.001). Seeds: 1.064 ± 0.016 (n=5) vs 1.124 ± 0.031 (n=5), seed ranges do not overlap; Welch t-test across seeds p = 0.009. Verdict: distinguishable from zero on both counts.

## Pooled 50-200 m (the thermocline)

### whole domain

| method | seeds | RMSE seed mean ± SD | RMSE 95 % CI (days) | MAE | bias | anomaly corr [95 % CI] | skill vs clim [95 % CI] |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) | 5 | 1.064 ± 0.016 | [1.013, 1.099] | 0.805 ± 0.012 | 0.039 ± 0.048 | 0.651 ± 0.012 [0.61, 0.67] | 0.413 ± 0.018 [0.355, 0.436] |
| OceanEmbed (no pretraining) | 5 | 1.052 ± 0.015 | [1.003, 1.084] | 0.796 ± 0.011 | 0.013 ± 0.024 | 0.658 ± 0.010 [0.62, 0.67] | 0.427 ± 0.016 [0.371, 0.451] |
| Plain U-Net (no Transformer) | 5 | 1.124 ± 0.031 | [1.054, 1.180] | 0.850 ± 0.022 | 0.103 ± 0.063 | 0.610 ± 0.016 [0.55, 0.64] | 0.345 ± 0.036 [0.262, 0.387] |
| Per-pixel MLP | 3 | 1.094 ± 0.005 | [1.006, 1.146] | 0.828 ± 0.004 | 0.128 ± 0.008 | 0.625 ± 0.005 [0.58, 0.65] | 0.380 ± 0.006 [0.328, 0.427] |
| Ridge regression | 1 | 1.155 | [1.061, 1.204] | 0.875 | 0.057 | 0.559 [0.50, 0.62] | 0.310 [0.250, 0.376] |
| Climatology | 1 | 1.390 | [1.312, 1.421] | 1.036 | -0.074 | n/a | 0.000 [0.000, 0.000] |

### Arabian Sea

| method | seeds | RMSE seed mean ± SD | RMSE 95 % CI (days) | MAE | bias | anomaly corr [95 % CI] | skill vs clim [95 % CI] |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) | 5 | 1.099 ± 0.021 | [1.044, 1.136] | 0.838 ± 0.016 | 0.037 ± 0.044 | 0.550 ± 0.018 [0.46, 0.62] | 0.303 ± 0.026 [0.244, 0.367] |
| OceanEmbed (no pretraining) | 5 | 1.089 ± 0.022 | [1.033, 1.124] | 0.832 ± 0.019 | 0.006 ± 0.024 | 0.554 ± 0.020 [0.47, 0.62] | 0.316 ± 0.028 [0.258, 0.382] |
| Plain U-Net (no Transformer) | 5 | 1.136 ± 0.028 | [1.070, 1.179] | 0.867 ± 0.022 | 0.060 ± 0.052 | 0.515 ± 0.019 [0.42, 0.59] | 0.256 ± 0.036 [0.192, 0.329] |
| Per-pixel MLP | 3 | 1.083 ± 0.008 | [0.991, 1.148] | 0.824 ± 0.008 | 0.043 ± 0.015 | 0.552 ± 0.010 [0.51, 0.61] | 0.324 ± 0.011 [0.282, 0.397] |
| Ridge regression | 1 | 1.139 | [1.058, 1.192] | 0.872 | -0.025 | 0.476 [0.39, 0.56] | 0.252 [0.192, 0.334] |
| Climatology | 1 | 1.317 | [1.255, 1.381] | 0.991 | -0.242 | n/a | 0.000 [0.000, 0.000] |

### Bay of Bengal

| method | seeds | RMSE seed mean ± SD | RMSE 95 % CI (days) | MAE | bias | anomaly corr [95 % CI] | skill vs clim [95 % CI] |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) | 5 | 1.007 ± 0.024 | [0.937, 1.064] | 0.751 ± 0.015 | 0.045 ± 0.070 | 0.749 ± 0.013 [0.69, 0.76] | 0.565 ± 0.021 [0.466, 0.592] |
| OceanEmbed (no pretraining) | 5 | 0.988 ± 0.016 | [0.932, 1.034] | 0.735 ± 0.009 | 0.028 ± 0.031 | 0.758 ± 0.008 [0.69, 0.77] | 0.581 ± 0.013 [0.474, 0.611] |
| Plain U-Net (no Transformer) | 5 | 1.112 ± 0.046 | [1.007, 1.212] | 0.826 ± 0.031 | 0.188 ± 0.125 | 0.703 ± 0.016 [0.61, 0.73] | 0.469 ± 0.044 [0.335, 0.511] |
| Per-pixel MLP | 3 | 1.120 ± 0.021 | [1.015, 1.201] | 0.838 ± 0.013 | 0.283 ± 0.015 | 0.700 ± 0.010 [0.64, 0.72] | 0.462 ± 0.020 [0.366, 0.492] |
| Ridge regression | 1 | 1.186 | [1.049, 1.259] | 0.881 | 0.208 | 0.645 [0.56, 0.70] | 0.397 [0.329, 0.455] |
| Climatology | 1 | 1.527 | [1.383, 1.567] | 1.130 | 0.229 | n/a | 0.000 [0.000, 0.000] |

### All depths pooled (0-1000 m), whole domain

| method | seeds | RMSE seed mean ± SD | RMSE 95 % CI (days) | MAE | bias | anomaly corr [95 % CI] | skill vs clim [95 % CI] |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) | 5 | 0.783 ± 0.012 | [0.741, 0.820] | 0.550 ± 0.009 | -0.079 ± 0.018 | 0.630 ± 0.016 [0.59, 0.65] | 0.409 ± 0.018 [0.346, 0.449] |
| OceanEmbed (no pretraining) | 5 | 0.781 ± 0.007 | [0.741, 0.817] | 0.552 ± 0.007 | -0.106 ± 0.022 | 0.634 ± 0.008 [0.59, 0.66] | 0.413 ± 0.011 [0.347, 0.451] |
| Plain U-Net (no Transformer) | 5 | 0.813 ± 0.019 | [0.765, 0.856] | 0.565 ± 0.012 | -0.048 ± 0.023 | 0.594 ± 0.020 [0.54, 0.62] | 0.364 ± 0.030 [0.290, 0.410] |
| Per-pixel MLP | 3 | 0.786 ± 0.002 | [0.738, 0.816] | 0.545 ± 0.001 | -0.041 ± 0.003 | 0.617 ± 0.002 [0.58, 0.64] | 0.406 ± 0.004 [0.358, 0.449] |
| Ridge regression | 1 | 0.876 | [0.815, 0.924] | 0.628 | -0.139 | 0.498 [0.44, 0.56] | 0.262 [0.199, 0.327] |
| Climatology | 1 | 1.019 | [0.968, 1.051] | 0.716 | -0.219 | n/a | 0.000 [0.000, 0.000] |

## RMSE by depth, whole domain (seed mean ± SD)

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression | Climatology |
|---|---|---|---|---|---|---|
| 0 | 0.541 ± 0.016 | 0.566 ± 0.022 | 0.526 ± 0.018 | 0.506 ± 0.006 | 0.727 | 0.793 |
| 5 | 0.549 ± 0.014 | 0.576 ± 0.024 | 0.536 ± 0.016 | 0.515 ± 0.004 | 0.728 | 0.792 |
| 10 | 0.552 ± 0.018 | 0.577 ± 0.026 | 0.539 ± 0.017 | 0.508 ± 0.004 | 0.726 | 0.792 |
| 20 | 0.616 ± 0.018 | 0.633 ± 0.025 | 0.606 ± 0.015 | 0.566 ± 0.004 | 0.762 | 0.829 |
| 30 | 0.717 ± 0.020 | 0.731 ± 0.020 | 0.717 ± 0.017 | 0.672 ± 0.002 | 0.828 | 0.913 |
| 50 | 0.920 ± 0.018 | 0.921 ± 0.005 | 0.935 ± 0.016 | 0.922 ± 0.009 | 0.997 | 1.179 |
| 75 | 1.106 ± 0.017 | 1.104 ± 0.007 | 1.177 ± 0.026 | 1.191 ± 0.016 | 1.213 | 1.545 |
| 100 | 1.212 ± 0.017 | 1.203 ± 0.018 | 1.309 ± 0.046 | 1.274 ± 0.010 | 1.313 | 1.657 |
| 125 | 1.207 ± 0.033 | 1.186 ± 0.026 | 1.284 ± 0.048 | 1.216 ± 0.003 | 1.303 | 1.556 |
| 150 | 1.098 ± 0.030 | 1.073 ± 0.028 | 1.149 ± 0.038 | 1.101 ± 0.007 | 1.199 | 1.359 |
| 200 | 0.780 ± 0.008 | 0.758 ± 0.015 | 0.806 ± 0.020 | 0.781 ± 0.005 | 0.827 | 0.894 |
| 300 | 0.534 ± 0.008 | 0.517 ± 0.010 | 0.550 ± 0.017 | 0.526 ± 0.001 | 0.532 | 0.545 |
| 500 | 0.366 ± 0.007 | 0.350 ± 0.005 | 0.370 ± 0.012 | 0.354 ± 0.000 | 0.351 | 0.352 |
| 700 | 0.367 ± 0.007 | 0.359 ± 0.004 | 0.371 ± 0.009 | 0.358 ± 0.003 | 0.354 | 0.356 |
| 1000 | 0.414 ± 0.010 | 0.405 ± 0.007 | 0.414 ± 0.018 | 0.389 ± 0.003 | 0.380 | 0.380 |

## RMSE by depth, Arabian Sea (seed mean ± SD)

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression | Climatology |
|---|---|---|---|---|---|---|
| 0 | 0.591 ± 0.020 | 0.608 ± 0.027 | 0.570 ± 0.021 | 0.518 ± 0.007 | 0.786 | 0.868 |
| 5 | 0.603 ± 0.018 | 0.620 ± 0.031 | 0.582 ± 0.021 | 0.530 ± 0.004 | 0.790 | 0.872 |
| 10 | 0.614 ± 0.023 | 0.632 ± 0.031 | 0.596 ± 0.020 | 0.539 ± 0.005 | 0.797 | 0.881 |
| 20 | 0.689 ± 0.022 | 0.694 ± 0.026 | 0.673 ± 0.016 | 0.613 ± 0.005 | 0.847 | 0.936 |
| 30 | 0.801 ± 0.025 | 0.799 ± 0.014 | 0.793 ± 0.022 | 0.733 ± 0.003 | 0.928 | 1.040 |
| 50 | 0.995 ± 0.022 | 0.988 ± 0.015 | 0.993 ± 0.026 | 0.962 ± 0.014 | 1.071 | 1.254 |
| 75 | 1.093 ± 0.015 | 1.090 ± 0.015 | 1.112 ± 0.022 | 1.082 ± 0.012 | 1.113 | 1.389 |
| 100 | 1.222 ± 0.019 | 1.214 ± 0.028 | 1.269 ± 0.034 | 1.214 ± 0.007 | 1.210 | 1.476 |
| 125 | 1.257 ± 0.049 | 1.244 ± 0.037 | 1.320 ± 0.055 | 1.244 ± 0.012 | 1.300 | 1.467 |
| 150 | 1.148 ± 0.037 | 1.134 ± 0.032 | 1.202 ± 0.048 | 1.132 ± 0.011 | 1.231 | 1.327 |
| 200 | 0.820 ± 0.008 | 0.805 ± 0.017 | 0.849 ± 0.026 | 0.800 ± 0.008 | 0.852 | 0.896 |
| 300 | 0.602 ± 0.011 | 0.583 ± 0.013 | 0.619 ± 0.020 | 0.582 ± 0.003 | 0.589 | 0.596 |
| 500 | 0.413 ± 0.009 | 0.392 ± 0.005 | 0.414 ± 0.014 | 0.393 ± 0.000 | 0.390 | 0.387 |
| 700 | 0.412 ± 0.008 | 0.404 ± 0.004 | 0.413 ± 0.009 | 0.397 ± 0.002 | 0.396 | 0.396 |
| 1000 | 0.473 ± 0.012 | 0.461 ± 0.008 | 0.470 ± 0.020 | 0.439 ± 0.003 | 0.428 | 0.429 |

## RMSE by depth, Bay of Bengal (seed mean ± SD)

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression | Climatology |
|---|---|---|---|---|---|---|
| 0 | 0.423 ± 0.014 | 0.451 ± 0.012 | 0.420 ± 0.015 | 0.433 ± 0.003 | 0.572 | 0.625 |
| 5 | 0.422 ± 0.012 | 0.456 ± 0.015 | 0.423 ± 0.013 | 0.434 ± 0.003 | 0.565 | 0.613 |
| 10 | 0.414 ± 0.012 | 0.441 ± 0.016 | 0.411 ± 0.017 | 0.413 ± 0.003 | 0.549 | 0.601 |
| 20 | 0.446 ± 0.013 | 0.486 ± 0.026 | 0.449 ± 0.016 | 0.435 ± 0.004 | 0.547 | 0.593 |
| 30 | 0.528 ± 0.013 | 0.572 ± 0.041 | 0.545 ± 0.006 | 0.519 ± 0.003 | 0.580 | 0.631 |
| 50 | 0.783 ± 0.024 | 0.799 ± 0.023 | 0.830 ± 0.022 | 0.854 ± 0.007 | 0.846 | 1.059 |
| 75 | 1.132 ± 0.045 | 1.131 ± 0.010 | 1.291 ± 0.070 | 1.372 ± 0.037 | 1.380 | 1.807 |
| 100 | 1.197 ± 0.047 | 1.185 ± 0.023 | 1.387 ± 0.085 | 1.378 ± 0.038 | 1.482 | 1.954 |
| 125 | 1.117 ± 0.031 | 1.079 ± 0.037 | 1.226 ± 0.053 | 1.164 ± 0.014 | 1.308 | 1.717 |
| 150 | 1.008 ± 0.029 | 0.959 ± 0.031 | 1.055 ± 0.025 | 1.045 ± 0.013 | 1.141 | 1.426 |
| 200 | 0.703 ± 0.019 | 0.665 ± 0.011 | 0.725 ± 0.014 | 0.750 ± 0.012 | 0.783 | 0.898 |
| 300 | 0.376 ± 0.014 | 0.361 ± 0.006 | 0.388 ± 0.017 | 0.401 ± 0.003 | 0.405 | 0.434 |
| 500 | 0.248 ± 0.005 | 0.243 ± 0.007 | 0.256 ± 0.010 | 0.260 ± 0.001 | 0.255 | 0.267 |
| 700 | 0.251 ± 0.006 | 0.246 ± 0.006 | 0.265 ± 0.011 | 0.263 ± 0.005 | 0.252 | 0.261 |
| 1000 | 0.248 ± 0.008 | 0.246 ± 0.008 | 0.255 ± 0.013 | 0.254 ± 0.007 | 0.250 | 0.248 |

## RMSE by depth, whole domain: 95 % CI over test days

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression | Climatology |
|---|---|---|---|---|---|---|
| 0 | [0.48, 0.61] | [0.51, 0.64] | [0.48, 0.58] | [0.49, 0.53] | [0.65, 0.82] | [0.72, 0.87] |
| 5 | [0.48, 0.63] | [0.51, 0.65] | [0.49, 0.59] | [0.50, 0.54] | [0.65, 0.82] | [0.73, 0.87] |
| 10 | [0.49, 0.63] | [0.52, 0.65] | [0.50, 0.60] | [0.49, 0.54] | [0.65, 0.82] | [0.73, 0.87] |
| 20 | [0.57, 0.69] | [0.58, 0.71] | [0.57, 0.66] | [0.54, 0.61] | [0.70, 0.84] | [0.78, 0.90] |
| 30 | [0.67, 0.78] | [0.68, 0.80] | [0.68, 0.77] | [0.64, 0.73] | [0.79, 0.90] | [0.88, 0.97] |
| 50 | [0.87, 0.97] | [0.87, 0.98] | [0.88, 0.98] | [0.85, 0.98] | [0.96, 1.03] | [1.13, 1.22] |
| 75 | [1.05, 1.14] | [1.05, 1.14] | [1.10, 1.23] | [1.09, 1.25] | [1.10, 1.25] | [1.45, 1.57] |
| 100 | [1.14, 1.25] | [1.14, 1.24] | [1.22, 1.37] | [1.16, 1.34] | [1.18, 1.37] | [1.54, 1.70] |
| 125 | [1.13, 1.27] | [1.11, 1.24] | [1.19, 1.37] | [1.11, 1.28] | [1.17, 1.39] | [1.45, 1.60] |
| 150 | [1.02, 1.17] | [1.00, 1.14] | [1.06, 1.24] | [1.00, 1.18] | [1.08, 1.29] | [1.28, 1.40] |
| 200 | [0.74, 0.81] | [0.72, 0.78] | [0.77, 0.84] | [0.74, 0.81] | [0.78, 0.87] | [0.84, 0.93] |
| 300 | [0.51, 0.55] | [0.49, 0.53] | [0.53, 0.57] | [0.50, 0.54] | [0.50, 0.55] | [0.51, 0.56] |
| 500 | [0.34, 0.38] | [0.32, 0.37] | [0.34, 0.39] | [0.32, 0.37] | [0.32, 0.37] | [0.32, 0.37] |
| 700 | [0.33, 0.39] | [0.32, 0.38] | [0.33, 0.39] | [0.32, 0.38] | [0.32, 0.37] | [0.32, 0.38] |
| 1000 | [0.36, 0.45] | [0.35, 0.44] | [0.37, 0.45] | [0.33, 0.43] | [0.32, 0.42] | [0.32, 0.42] |

## Skill vs climatology by depth, whole domain (seed mean ± SD)

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression |
|---|---|---|---|---|---|
| 0 | 0.533 ± 0.027 | 0.490 ± 0.039 | 0.559 ± 0.030 | 0.592 ± 0.010 | 0.160 |
| 5 | 0.519 ± 0.025 | 0.471 ± 0.045 | 0.542 ± 0.027 | 0.577 ± 0.006 | 0.156 |
| 10 | 0.513 ± 0.032 | 0.469 ± 0.048 | 0.536 ± 0.028 | 0.588 ± 0.006 | 0.161 |
| 20 | 0.448 ± 0.032 | 0.416 ± 0.047 | 0.465 ± 0.026 | 0.534 ± 0.006 | 0.156 |
| 30 | 0.382 ± 0.034 | 0.358 ± 0.036 | 0.383 ± 0.030 | 0.457 ± 0.003 | 0.177 |
| 50 | 0.391 ± 0.023 | 0.390 ± 0.006 | 0.372 ± 0.022 | 0.389 ± 0.011 | 0.285 |
| 75 | 0.488 ± 0.015 | 0.490 ± 0.007 | 0.420 ± 0.026 | 0.405 ± 0.016 | 0.384 |
| 100 | 0.465 ± 0.015 | 0.473 ± 0.016 | 0.375 ± 0.043 | 0.409 ± 0.010 | 0.372 |
| 125 | 0.398 ± 0.034 | 0.419 ± 0.026 | 0.318 ± 0.051 | 0.390 ± 0.003 | 0.299 |
| 150 | 0.346 ± 0.036 | 0.376 ± 0.033 | 0.284 ± 0.048 | 0.343 ± 0.008 | 0.221 |
| 200 | 0.239 ± 0.016 | 0.281 ± 0.028 | 0.186 ± 0.040 | 0.236 ± 0.009 | 0.143 |
| 300 | 0.038 ± 0.027 | 0.099 ± 0.036 | -0.019 ± 0.061 | 0.069 ± 0.004 | 0.046 |
| 500 | -0.086 ± 0.043 | 0.008 ± 0.027 | -0.106 ± 0.073 | -0.013 ± 0.002 | 0.003 |
| 700 | -0.062 ± 0.041 | -0.017 ± 0.025 | -0.087 ± 0.056 | -0.010 ± 0.017 | 0.012 |
| 1000 | -0.186 ± 0.060 | -0.134 ± 0.040 | -0.185 ± 0.100 | -0.049 ± 0.014 | 0.001 |

## Anomaly correlation by depth, whole domain (seed mean ± SD)

| depth (m) | OceanEmbed (pretrained encoder) | OceanEmbed (no pretraining) | Plain U-Net (no Transformer) | Per-pixel MLP | Ridge regression |
|---|---|---|---|---|---|
| 0 | 0.65 ± 0.02 | 0.63 ± 0.02 | 0.65 ± 0.03 | 0.69 ± 0.01 | 0.33 |
| 5 | 0.63 ± 0.02 | 0.62 ± 0.01 | 0.64 ± 0.03 | 0.67 ± 0.01 | 0.33 |
| 10 | 0.62 ± 0.03 | 0.61 ± 0.02 | 0.63 ± 0.03 | 0.68 ± 0.01 | 0.33 |
| 20 | 0.57 ± 0.03 | 0.58 ± 0.02 | 0.57 ± 0.02 | 0.64 ± 0.00 | 0.31 |
| 30 | 0.54 ± 0.02 | 0.55 ± 0.01 | 0.52 ± 0.02 | 0.58 ± 0.01 | 0.36 |
| 50 | 0.61 ± 0.02 | 0.62 ± 0.01 | 0.58 ± 0.03 | 0.57 ± 0.01 | 0.54 |
| 75 | 0.70 ± 0.01 | 0.70 ± 0.01 | 0.65 ± 0.02 | 0.63 ± 0.01 | 0.63 |
| 100 | 0.70 ± 0.01 | 0.70 ± 0.01 | 0.65 ± 0.01 | 0.67 ± 0.01 | 0.63 |
| 125 | 0.66 ± 0.01 | 0.67 ± 0.01 | 0.63 ± 0.02 | 0.66 ± 0.00 | 0.57 |
| 150 | 0.62 ± 0.01 | 0.63 ± 0.01 | 0.59 ± 0.02 | 0.61 ± 0.01 | 0.49 |
| 200 | 0.54 ± 0.01 | 0.56 ± 0.01 | 0.51 ± 0.02 | 0.51 ± 0.01 | 0.39 |
| 300 | 0.32 ± 0.02 | 0.36 ± 0.02 | 0.28 ± 0.03 | 0.29 ± 0.00 | 0.22 |
| 500 | 0.15 ± 0.02 | 0.22 ± 0.01 | 0.16 ± 0.03 | 0.14 ± 0.01 | 0.10 |
| 700 | 0.18 ± 0.02 | 0.21 ± 0.02 | 0.16 ± 0.03 | 0.14 ± 0.01 | 0.11 |
| 1000 | 0.05 ± 0.03 | 0.07 ± 0.03 | 0.07 ± 0.04 | 0.08 ± 0.01 | 0.07 |

## Paired comparisons (A - B in RMSE, °C; negative = A better)

Bootstrap interval and p-value from the shared replicates; the seed columns compare the per-seed pooled RMSEs (ranges overlap = the spread of A and B intersect; Welch t-test needs at least 2 seeds on each side).

### Pooled 50-200 m, whole domain

| A - B | Δ RMSE | 95 % CI | excludes 0 | p (bootstrap) | seed means A / B | seed ranges overlap | Welch p |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) - OceanEmbed (no pretraining) | +0.013 | [0.004, 0.020] | yes | 0.004 | 1.064 / 1.052 | yes | 0.236 |
| OceanEmbed (pretrained encoder) - Plain U-Net (no Transformer) | -0.060 | [-0.084, -0.036] | yes | 0.001 | 1.064 / 1.124 | no | 0.009 |
| OceanEmbed (pretrained encoder) - Per-pixel MLP | -0.030 | [-0.058, 0.011] | no | 0.313 | 1.064 / 1.094 | yes | 0.012 |
| OceanEmbed (pretrained encoder) - Ridge regression | -0.090 | [-0.114, -0.040] | yes | 0.001 | 1.064 / 1.155 | no | n/a |
| OceanEmbed (pretrained encoder) - Climatology | -0.325 | [-0.348, -0.264] | yes | 0.001 | 1.064 / 1.390 | no | n/a |
| OceanEmbed (no pretraining) - Plain U-Net (no Transformer) | -0.072 | [-0.101, -0.043] | yes | 0.001 | 1.052 / 1.124 | no | 0.004 |
| OceanEmbed (no pretraining) - Ridge regression | -0.103 | [-0.130, -0.049] | yes | 0.001 | 1.052 / 1.155 | no | n/a |
| Plain U-Net (no Transformer) - Ridge regression | -0.030 | [-0.047, 0.014] | no | 0.371 | 1.124 / 1.155 | yes | n/a |
| Per-pixel MLP - Ridge regression | -0.060 | [-0.091, -0.017] | yes | 0.004 | 1.094 / 1.155 | no | n/a |

### Pooled 50-200 m, Arabian Sea

| A - B | Δ RMSE | 95 % CI | excludes 0 | p (bootstrap) | seed means A / B | seed ranges overlap | Welch p |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) - OceanEmbed (no pretraining) | +0.010 | [0.002, 0.019] | yes | 0.015 | 1.099 / 1.089 | yes | 0.495 |
| OceanEmbed (pretrained encoder) - Plain U-Net (no Transformer) | -0.037 | [-0.048, -0.023] | yes | 0.001 | 1.099 / 1.136 | yes | 0.047 |
| OceanEmbed (pretrained encoder) - Per-pixel MLP | +0.016 | [-0.033, 0.074] | no | 0.336 | 1.099 / 1.083 | yes | 0.180 |
| OceanEmbed (pretrained encoder) - Ridge regression | -0.040 | [-0.078, 0.012] | no | 0.154 | 1.099 / 1.139 | no | n/a |
| OceanEmbed (pretrained encoder) - Climatology | -0.218 | [-0.277, -0.167] | yes | 0.001 | 1.099 / 1.317 | no | n/a |
| OceanEmbed (no pretraining) - Plain U-Net (no Transformer) | -0.047 | [-0.061, -0.031] | yes | 0.001 | 1.089 / 1.136 | yes | 0.020 |
| OceanEmbed (no pretraining) - Ridge regression | -0.049 | [-0.087, -0.002] | yes | 0.041 | 1.089 / 1.139 | no | n/a |
| Plain U-Net (no Transformer) - Ridge regression | -0.003 | [-0.042, 0.044] | no | 0.940 | 1.136 / 1.139 | yes | n/a |
| Per-pixel MLP - Ridge regression | -0.056 | [-0.112, -0.013] | yes | 0.011 | 1.083 / 1.139 | no | n/a |

### Pooled 50-200 m, Bay of Bengal

| A - B | Δ RMSE | 95 % CI | excludes 0 | p (bootstrap) | seed means A / B | seed ranges overlap | Welch p |
|---|---|---|---|---|---|---|---|
| OceanEmbed (pretrained encoder) - OceanEmbed (no pretraining) | +0.019 | [-0.006, 0.047] | no | 0.323 | 1.007 / 0.988 | yes | 0.178 |
| OceanEmbed (pretrained encoder) - Plain U-Net (no Transformer) | -0.106 | [-0.167, -0.051] | yes | 0.001 | 1.007 / 1.112 | no | 0.004 |
| OceanEmbed (pretrained encoder) - Per-pixel MLP | -0.113 | [-0.159, -0.047] | yes | 0.001 | 1.007 / 1.120 | no | 0.001 |
| OceanEmbed (pretrained encoder) - Ridge regression | -0.178 | [-0.220, -0.070] | yes | 0.001 | 1.007 / 1.186 | no | n/a |
| OceanEmbed (pretrained encoder) - Climatology | -0.519 | [-0.550, -0.378] | yes | 0.001 | 1.007 / 1.527 | no | n/a |
| OceanEmbed (no pretraining) - Plain U-Net (no Transformer) | -0.125 | [-0.202, -0.049] | yes | 0.001 | 0.988 / 1.112 | no | 0.002 |
| OceanEmbed (no pretraining) - Ridge regression | -0.198 | [-0.263, -0.068] | yes | 0.001 | 0.988 / 1.186 | no | n/a |
| Plain U-Net (no Transformer) - Ridge regression | -0.073 | [-0.093, 0.015] | no | 0.237 | 1.112 / 1.186 | no | n/a |
| Per-pixel MLP - Ridge regression | -0.065 | [-0.078, -0.007] | yes | 0.008 | 1.120 / 1.186 | no | n/a |

### Anomaly correlation difference, pooled 50-200 m, whole domain

| A - B | Δ anomaly corr | 95 % CI | excludes 0 |
|---|---|---|---|
| OceanEmbed (pretrained encoder) - OceanEmbed (no pretraining) | -0.006 | [-0.014, 0.002] | no |
| OceanEmbed (pretrained encoder) - Plain U-Net (no Transformer) | +0.042 | [0.023, 0.070] | yes |
| OceanEmbed (pretrained encoder) - Per-pixel MLP | +0.026 | [-0.003, 0.046] | no |
| OceanEmbed (pretrained encoder) - Ridge regression | +0.092 | [0.037, 0.125] | yes |
| OceanEmbed (no pretraining) - Plain U-Net (no Transformer) | +0.049 | [0.022, 0.082] | yes |
| OceanEmbed (no pretraining) - Ridge regression | +0.098 | [0.042, 0.136] | yes |
| Plain U-Net (no Transformer) - Ridge regression | +0.049 | [-0.001, 0.074] | no |
| Per-pixel MLP - Ridge regression | +0.065 | [0.016, 0.107] | yes |

### Δ RMSE by depth, OceanEmbed (pretrained encoder) - OceanEmbed (no pretraining), whole domain (* = CI excludes 0)

| depth (m) | Δ RMSE | 95 % CI |
|---|---|---|
| 0 | -0.025* | [-0.036, -0.020] |
| 5 | -0.027* | [-0.037, -0.022] |
| 10 | -0.025* | [-0.035, -0.018] |
| 20 | -0.018* | [-0.025, -0.012] |
| 30 | -0.014* | [-0.024, -0.004] |
| 50 | -0.001 | [-0.017, 0.009] |
| 75 | +0.002 | [-0.008, 0.006] |
| 100 | +0.009 | [-0.011, 0.025] |
| 125 | +0.021* | [0.008, 0.041] |
| 150 | +0.025* | [0.015, 0.040] |
| 200 | +0.022* | [0.013, 0.032] |
| 300 | +0.017* | [0.012, 0.023] |
| 500 | +0.016* | [0.013, 0.020] |
| 700 | +0.008* | [0.004, 0.014] |
| 1000 | +0.009* | [0.004, 0.014] |

### Δ RMSE by depth, OceanEmbed (no pretraining) - Plain U-Net (no Transformer), whole domain (* = CI excludes 0)

| depth (m) | Δ RMSE | 95 % CI |
|---|---|---|
| 0 | +0.040* | [0.016, 0.064] |
| 5 | +0.040* | [0.018, 0.063] |
| 10 | +0.037* | [0.012, 0.064] |
| 20 | +0.027* | [0.003, 0.054] |
| 30 | +0.014 | [-0.007, 0.042] |
| 50 | -0.013 | [-0.024, 0.002] |
| 75 | -0.073* | [-0.119, -0.025] |
| 100 | -0.107* | [-0.151, -0.058] |
| 125 | -0.099* | [-0.136, -0.068] |
| 150 | -0.077* | [-0.109, -0.054] |
| 200 | -0.049* | [-0.067, -0.036] |
| 300 | -0.033* | [-0.044, -0.026] |
| 500 | -0.019* | [-0.024, -0.015] |
| 700 | -0.012* | [-0.021, -0.006] |
| 1000 | -0.009* | [-0.016, -0.004] |

### Δ RMSE by depth, OceanEmbed (pretrained encoder) - Plain U-Net (no Transformer), whole domain (* = CI excludes 0)

| depth (m) | Δ RMSE | 95 % CI |
|---|---|---|
| 0 | +0.015 | [-0.011, 0.035] |
| 5 | +0.013 | [-0.012, 0.035] |
| 10 | +0.013 | [-0.011, 0.032] |
| 20 | +0.010 | [-0.012, 0.031] |
| 30 | +0.000 | [-0.014, 0.020] |
| 50 | -0.014 | [-0.031, 0.005] |
| 75 | -0.072* | [-0.118, -0.027] |
| 100 | -0.098* | [-0.131, -0.064] |
| 125 | -0.077* | [-0.099, -0.055] |
| 150 | -0.052* | [-0.081, -0.028] |
| 200 | -0.027* | [-0.044, -0.016] |
| 300 | -0.015* | [-0.025, -0.010] |
| 500 | -0.003 | [-0.006, 0.002] |
| 700 | -0.004 | [-0.008, 0.001] |
| 1000 | +0.000 | [-0.007, 0.005] |

## Provenance

| method | parameters | best epoch per seed | val RMSE per seed (°C) |
|---|---|---|---|
| OceanEmbed (pretrained encoder) | 3,741,423 | 4, 5, 8, 5, 9 | 0.743, 0.738, 0.733, 0.738, 0.730 |
| OceanEmbed (no pretraining) | 3,741,423 | 4, 5, 4, 3, 5 | 0.742, 0.735, 0.738, 0.737, 0.730 |
| Plain U-Net (no Transformer) | 3,731,055 | 8, 5, 3, 6, 6 | 0.748, 0.732, 0.739, 0.736, 0.738 |
| Per-pixel MLP | 138,511 | 17, 20, 11 | 0.736, 0.734, 0.735 |
| Ridge regression | n/a | n/a | n/a |
| Climatology | n/a | n/a | n/a |

Figures: `figures/rmse_by_depth.png`, `figures/paired_differences.png`, `figures/skill_by_depth.png`. Numbers: `summary.json`.
