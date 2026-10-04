# Benchmark - every model family under one protocol (poc_long)

Reference: GLORYS reanalysis, harmonised to the 0.25 degree grid. Data: eleven training years (2011-2021), validation 2022, test 2023 and 2024. Metric: RMSE pooled over 50-200 m, whole test period and by basin and depth. Intervals: moving-block bootstrap over days, 95 % percentile intervals, replicates shared by every family so differences are paired. A difference is *established* only if paired interval excludes zero AND the per-seed RMSE ranges do not overlap (marked `*`). Seeds: 3 per trained family (ridge and climatology are deterministic).

## Input set: all seven inputs

Inputs: sst, sss, sla, currents, winds. Sources: R2 (`research/r2`): CNN + Transformer, per-pixel MLP, ridge, climatology; this stage: boosted trees, random forest, plain U-Net. Block bootstrap: 70-day blocks, 2000 replicates.

### Ranking, RMSE pooled over 50-200 m (degC)

Seed mean +- SD; the block-bootstrap interval of the seed-mean predictions in brackets.

| Rank | Family | 2023 | 2024 | 2023 + 2024 | Arabian Sea (both years) | Bay of Bengal (both years) |
|---|---|---|---|---|---|---|
| 1 | CNN + Transformer | 0.975 ± 0.007 [0.899, 1.032] | 1.004 ± 0.014 [0.936, 1.044] | 0.989 ± 0.010 [0.941, 1.032] | 1.018 ± 0.018 [0.957, 1.074] | 0.938 ± 0.007 [0.894, 0.969] |
| 2 | Boosted trees, LightGBM (per pixel) | 0.966 ± 0.003 [0.882, 1.032] | 1.013 ± 0.001 [0.921, 1.054] | 0.990 ± 0.002 [0.931, 1.040] | 0.983 ± 0.002 [0.920, 1.037] | 1.001 ± 0.002 [0.926, 1.083] |
| 3 | Per-pixel MLP | 0.984 ± 0.002 [0.902, 1.047] | 1.024 ± 0.008 [0.929, 1.063] | 1.004 ± 0.004 [0.945, 1.054] | 0.996 ± 0.004 [0.933, 1.049] | 1.015 ± 0.006 [0.945, 1.096] |
| 4 | Plain U-Net (no Transformer) | 0.992 ± 0.010 [0.913, 1.054] | 1.030 ± 0.025 [0.974, 1.063] | 1.011 ± 0.017 [0.964, 1.053] | 1.026 ± 0.028 [0.968, 1.077] | 0.985 ± 0.010 [0.949, 1.026] |
| 5 | Random forest (per pixel) | 1.030 ± 0.002 [0.913, 1.120] | 1.041 ± 0.001 [0.945, 1.082] | 1.035 ± 0.002 [0.965, 1.111] | 1.013 ± 0.001 [0.945, 1.076] | 1.076 ± 0.003 [0.972, 1.195] |
| 6 | Ridge regression | 1.270 [1.045, 1.401] | 1.138 [1.023, 1.182] | 1.207 [1.081, 1.365] | 1.150 [1.060, 1.259] | 1.306 [1.092, 1.567] |
| 7 | Climatology | 1.519 [1.194, 1.735] | 1.394 [1.317, 1.404] | 1.459 [1.294, 1.663] | 1.350 [1.209, 1.519] | 1.649 [1.396, 1.942] |

Winner by period and region (ties: families not established worse than the best):

| Period | whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|
| 2023 | tie: Boosted trees, LightGBM (per pixel) = CNN + Transformer | tie: Boosted trees, LightGBM (per pixel) = Per-pixel MLP | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) |
| 2024 | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) = Per-pixel MLP = Plain U-Net (no Transformer) = Random forest (per pixel) | tie: Boosted trees, LightGBM (per pixel) = Per-pixel MLP = Plain U-Net (no Transformer) | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) |
| 2023 + 2024 | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) = Per-pixel MLP = Plain U-Net (no Transformer) | Boosted trees, LightGBM (per pixel) (established) | CNN + Transformer (established) |

### Each family minus the CNN + Transformer (pooled 50-200 m, degC; positive = the family is worse)

| Family | 2023 | 2024 | 2023 + 2024 | Arabian Sea (both years) | Bay of Bengal (both years) |
|---|---|---|---|---|---|
| Boosted trees, LightGBM (per pixel) | -0.008 [-0.022, +0.006] | +0.010 [-0.023, +0.019] | +0.001 [-0.015, +0.013] | -0.035 [-0.064, -0.018] * | +0.063 [+0.004, +0.144] * |
| Per-pixel MLP | +0.010 [-0.000, +0.019] | +0.020 [-0.010, +0.025] | +0.015 [+0.001, +0.026] | -0.022 [-0.054, -0.002] * | +0.078 [+0.023, +0.152] * |
| Plain U-Net (no Transformer) | +0.017 [+0.010, +0.028] * | +0.026 [+0.012, +0.044] | +0.022 [+0.014, +0.032] | +0.009 [-0.005, +0.018] | +0.047 [+0.029, +0.080] * |
| Random forest (per pixel) | +0.056 [+0.012, +0.092] * | +0.037 [-0.001, +0.050] | +0.046 [+0.016, +0.083] * | -0.005 [-0.034, +0.017] | +0.138 [+0.053, +0.242] * |
| Ridge regression | +0.295 [+0.137, +0.382] * | +0.135 [+0.068, +0.158] * | +0.218 [+0.125, +0.343] * | +0.132 [+0.073, +0.200] * | +0.368 [+0.171, +0.616] * |
| Climatology | +0.545 [+0.296, +0.708] * | +0.390 [+0.321, +0.401] * | +0.470 [+0.339, +0.644] * | +0.332 [+0.238, +0.455] * | +0.711 [+0.478, +0.985] * |

### The best per-pixel model minus the CNN + Transformer, by basin

| Region | Best per-pixel family (chosen on both years) | 2023 | 2024 | 2023 + 2024 |
|---|---|---|---|---|
| whole domain | Boosted trees, LightGBM (per pixel) | -0.008 [-0.022, +0.006] | +0.010 [-0.023, +0.019] | +0.001 [-0.015, +0.013] |
| Arabian Sea | Boosted trees, LightGBM (per pixel) | -0.026 [-0.030, -0.009] * | -0.043 [-0.088, -0.017] * | -0.035 [-0.064, -0.018] * |
| Bay of Bengal | Boosted trees, LightGBM (per pixel) | +0.020 [-0.021, +0.039] | +0.106 [-0.012, +0.207] | +0.063 [+0.004, +0.144] * |

### RMSE by depth, both test years, whole domain (seed mean, degC)

| Family | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN + Transformer | 0.48 | 0.48 | 0.48 | 0.54 | 0.64 | 0.84 | 1.03 | 1.14 | 1.12 | 1.02 | 0.72 | 0.50 | 0.35 | 0.35 | 0.37 |
| Boosted trees, LightGBM (per pixel) | 0.50 | 0.50 | 0.49 | 0.55 | 0.65 | 0.84 | 1.06 | 1.14 | 1.11 | 1.00 | 0.73 | 0.49 | 0.34 | 0.33 | 0.36 |
| Per-pixel MLP | 0.49 | 0.49 | 0.48 | 0.54 | 0.65 | 0.86 | 1.07 | 1.15 | 1.12 | 1.02 | 0.74 | 0.50 | 0.34 | 0.33 | 0.36 |
| Plain U-Net (no Transformer) | 0.49 | 0.50 | 0.50 | 0.56 | 0.65 | 0.85 | 1.05 | 1.16 | 1.15 | 1.05 | 0.75 | 0.51 | 0.35 | 0.34 | 0.37 |
| Random forest (per pixel) | 0.55 | 0.56 | 0.55 | 0.60 | 0.69 | 0.89 | 1.12 | 1.20 | 1.15 | 1.04 | 0.74 | 0.50 | 0.34 | 0.34 | 0.36 |
| Ridge regression | 0.72 | 0.72 | 0.71 | 0.75 | 0.83 | 1.05 | 1.34 | 1.43 | 1.34 | 1.17 | 0.80 | 0.50 | 0.34 | 0.34 | 0.36 |
| Climatology | 0.81 | 0.81 | 0.81 | 0.84 | 0.94 | 1.22 | 1.62 | 1.76 | 1.65 | 1.41 | 0.93 | 0.53 | 0.34 | 0.34 | 0.36 |

### Training cost (per seed)

| Family | Training time | Size | Seeds | Hardware |
|---|---|---|---|---|
| CNN + Transformer | 19.3 min | 3,741,423 parameters | 3 | GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Boosted trees, LightGBM (per pixel) | 5.4 min | 5,781 trees | 3 | CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Per-pixel MLP | 0.3 min | 138,511 parameters | 3 | GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Plain U-Net (no Transformer) | 29.7 min | 3,731,055 parameters | 3 | GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU; CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Random forest (per pixel) | 4.2 min | 100 trees | 3 | CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Ridge regression | 3.0 min | 180 parameters | 1 | CPU |
| Climatology | n/a (no training) | none | 1 | - |

Figures: `figures/rmse_by_depth_all7.png`, `figures/ranking_all7.png`.

## Input set: SST + sea level

Inputs: sst, sla. Sources: `research/final_inputs`: CNN + Transformer (the final model); this stage: ridge, per-pixel MLP, boosted trees, random forest; climatology from R2. Block bootstrap: 69-day blocks, 2000 replicates.

### Ranking, RMSE pooled over 50-200 m (degC)

Seed mean +- SD; the block-bootstrap interval of the seed-mean predictions in brackets.

| Rank | Family | 2023 | 2024 | 2023 + 2024 | Arabian Sea (both years) | Bay of Bengal (both years) |
|---|---|---|---|---|---|---|
| 1 | CNN + Transformer | 0.970 ± 0.009 [0.896, 1.026] | 1.002 ± 0.003 [0.943, 1.040] | 0.986 ± 0.003 [0.942, 1.030] | 1.010 ± 0.004 [0.954, 1.064] | 0.943 ± 0.002 [0.900, 0.986] |
| 2 | Boosted trees, LightGBM (per pixel) | 0.970 ± 0.001 [0.885, 1.034] | 1.029 ± 0.001 [0.932, 1.069] | 0.999 ± 0.001 [0.939, 1.055] | 0.985 ± 0.001 [0.923, 1.035] | 1.025 ± 0.002 [0.935, 1.132] |
| 3 | Per-pixel MLP | 0.983 ± 0.001 [0.897, 1.049] | 1.038 ± 0.007 [0.934, 1.077] | 1.010 ± 0.003 [0.947, 1.069] | 0.993 ± 0.004 [0.930, 1.044] | 1.040 ± 0.004 [0.950, 1.147] |
| 4 | Random forest (per pixel) | 1.020 ± 0.002 [0.909, 1.103] | 1.049 ± 0.001 [0.952, 1.087] | 1.034 ± 0.001 [0.963, 1.107] | 1.010 ± 0.001 [0.943, 1.069] | 1.078 ± 0.002 [0.974, 1.203] |
| 5 | Ridge regression | 1.280 [1.042, 1.419] | 1.149 [1.028, 1.196] | 1.218 [1.083, 1.387] | 1.152 [1.060, 1.261] | 1.330 [1.102, 1.615] |
| 6 | Climatology | 1.519 [1.196, 1.731] | 1.394 [1.317, 1.406] | 1.459 [1.296, 1.662] | 1.350 [1.210, 1.517] | 1.649 [1.395, 1.940] |

Winner by period and region (ties: families not established worse than the best):

| Period | whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|
| 2023 | tie: Boosted trees, LightGBM (per pixel) = CNN + Transformer | tie: Boosted trees, LightGBM (per pixel) = Per-pixel MLP = CNN + Transformer | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) = Per-pixel MLP |
| 2024 | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) = Per-pixel MLP = Random forest (per pixel) | tie: Boosted trees, LightGBM (per pixel) = Per-pixel MLP | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) = Per-pixel MLP |
| 2023 + 2024 | tie: CNN + Transformer = Boosted trees, LightGBM (per pixel) | tie: Boosted trees, LightGBM (per pixel) = Per-pixel MLP | CNN + Transformer (established) |

### Each family minus the CNN + Transformer (pooled 50-200 m, degC; positive = the family is worse)

| Family | 2023 | 2024 | 2023 + 2024 | Arabian Sea (both years) | Bay of Bengal (both years) |
|---|---|---|---|---|---|
| Boosted trees, LightGBM (per pixel) | -0.000 [-0.020, +0.018] | +0.027 [-0.020, +0.040] | +0.013 [-0.009, +0.033] | -0.025 [-0.051, -0.011] * | +0.083 [+0.012, +0.167] * |
| Per-pixel MLP | +0.013 [-0.005, +0.028] | +0.035 [-0.016, +0.045] | +0.024 [+0.001, +0.045] * | -0.017 [-0.046, -0.001] * | +0.097 [+0.025, +0.180] * |
| Random forest (per pixel) | +0.049 [+0.008, +0.079] * | +0.047 [-0.002, +0.060] | +0.048 [+0.016, +0.081] * | -0.000 [-0.025, +0.016] | +0.136 [+0.050, +0.229] * |
| Ridge regression | +0.310 [+0.134, +0.404] * | +0.147 [+0.074, +0.174] * | +0.232 [+0.131, +0.364] * | +0.142 [+0.081, +0.209] * | +0.387 [+0.180, +0.638] * |
| Climatology | +0.549 [+0.300, +0.711] * | +0.392 [+0.327, +0.397] * | +0.473 [+0.342, +0.644] * | +0.340 [+0.246, +0.462] * | +0.706 [+0.475, +0.971] * |

### The best per-pixel model minus the CNN + Transformer, by basin

| Region | Best per-pixel family (chosen on both years) | 2023 | 2024 | 2023 + 2024 |
|---|---|---|---|---|
| whole domain | Boosted trees, LightGBM (per pixel) | -0.000 [-0.020, +0.018] | +0.027 [-0.020, +0.040] | +0.013 [-0.009, +0.033] |
| Arabian Sea | Boosted trees, LightGBM (per pixel) | -0.017 [-0.023, +0.005] | -0.034 [-0.071, -0.012] * | -0.025 [-0.051, -0.011] * |
| Bay of Bengal | Boosted trees, LightGBM (per pixel) | +0.029 [-0.031, +0.058] | +0.135 [-0.001, +0.221] | +0.083 [+0.012, +0.167] * |

### RMSE by depth, both test years, whole domain (seed mean, degC)

| Family | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN + Transformer | 0.48 | 0.49 | 0.49 | 0.55 | 0.65 | 0.84 | 1.02 | 1.13 | 1.12 | 1.02 | 0.73 | 0.50 | 0.34 | 0.34 | 0.37 |
| Boosted trees, LightGBM (per pixel) | 0.49 | 0.50 | 0.50 | 0.56 | 0.66 | 0.85 | 1.06 | 1.15 | 1.12 | 1.01 | 0.74 | 0.50 | 0.34 | 0.34 | 0.36 |
| Per-pixel MLP | 0.48 | 0.49 | 0.48 | 0.54 | 0.64 | 0.85 | 1.08 | 1.17 | 1.13 | 1.03 | 0.74 | 0.50 | 0.34 | 0.34 | 0.36 |
| Random forest (per pixel) | 0.54 | 0.54 | 0.54 | 0.59 | 0.68 | 0.88 | 1.11 | 1.20 | 1.16 | 1.04 | 0.74 | 0.50 | 0.34 | 0.34 | 0.36 |
| Ridge regression | 0.73 | 0.72 | 0.72 | 0.75 | 0.83 | 1.05 | 1.35 | 1.45 | 1.35 | 1.18 | 0.81 | 0.50 | 0.34 | 0.34 | 0.36 |
| Climatology | 0.81 | 0.81 | 0.81 | 0.84 | 0.94 | 1.22 | 1.62 | 1.76 | 1.65 | 1.41 | 0.93 | 0.53 | 0.34 | 0.34 | 0.36 |

### Training cost (per seed)

| Family | Training time | Size | Seeds | Hardware |
|---|---|---|---|---|
| CNN + Transformer | 18.6 min | 3,741,423 parameters | 3 | GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Boosted trees, LightGBM (per pixel) | 3.2 min | 5,607 trees | 3 | CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Per-pixel MLP | 0.9 min | 138,511 parameters | 3 | GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU; CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Random forest (per pixel) | 1.9 min | 100 trees | 3 | CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Ridge regression | 0.7 min | 180 parameters | 1 | CPU Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical cores |
| Climatology | n/a (no training) | none | 1 | - |

Figures: `figures/rmse_by_depth_sst_sla.png`, `figures/ranking_sst_sla.png`.

