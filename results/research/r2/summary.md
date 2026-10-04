# R2 - more training years, two test years (poc_long)

Test periods: 2023 (365 days, 2023-01-01 .. 2023-12-31); 2024 (350 days, 2024-01-01 .. 2024-12-15); pooled (715 days, 2023-01-01 .. 2024-12-15). Reference GLORYS. Intervals: moving-block bootstrap over the days of each period, 2000 replicates, blocks of 70 days, 95 % percentile. `*` = established (interval excludes zero AND the per-seed RMSE ranges do not overlap).

## Skill of every method

| Method | Seeds | Period | RMSE 50-200 m whole domain | Arabian Sea | Bay of Bengal | Skill 50-200 m | Anomaly corr. | Skill 500-1000 m |
|---|---|---|---|---|---|---|---|---|
| OceanEmbed (no pretraining) | 3 | 2023 | 0.975 ± 0.007 | 0.992 ± 0.013 | 0.941 ± 0.008 | 0.588 ± 0.006 | 0.764 ± 0.007 | -0.021 ± 0.043 |
| OceanEmbed (no pretraining) | 3 | 2024 | 1.004 ± 0.014 | 1.043 ± 0.024 | 0.934 ± 0.006 | 0.481 ± 0.015 | 0.688 ± 0.011 | -0.035 ± 0.051 |
| OceanEmbed (no pretraining) | 3 | pooled | 0.989 ± 0.010 | 1.018 ± 0.018 | 0.938 ± 0.007 | 0.540 ± 0.010 | 0.729 ± 0.008 | -0.028 ± 0.046 |
| Per-pixel MLP | 3 | 2023 | 0.984 ± 0.002 | 0.980 ± 0.003 | 0.986 ± 0.005 | 0.580 ± 0.001 | 0.758 ± 0.002 | 0.055 ± 0.000 |
| Per-pixel MLP | 3 | 2024 | 1.024 ± 0.008 | 1.012 ± 0.005 | 1.045 ± 0.014 | 0.460 ± 0.009 | 0.669 ± 0.006 | 0.015 ± 0.004 |
| Per-pixel MLP | 3 | pooled | 1.004 ± 0.004 | 0.996 ± 0.004 | 1.015 ± 0.006 | 0.527 ± 0.004 | 0.716 ± 0.003 | 0.035 ± 0.002 |
| Ridge regression | 1 | 2023 | 1.270 | 1.184 | 1.415 | 0.301 | 0.542 | 0.049 |
| Ridge regression | 1 | 2024 | 1.138 | 1.113 | 1.182 | 0.333 | 0.558 | 0.018 |
| Ridge regression | 1 | pooled | 1.207 | 1.150 | 1.306 | 0.315 | 0.547 | 0.033 |
| Climatology | 1 | 2023 | 1.519 | 1.372 | 1.765 | 0.000 | n/a | 0.000 |
| Climatology | 1 | 2024 | 1.394 | 1.326 | 1.519 | 0.000 | n/a | 0.000 |
| Climatology | 1 | pooled | 1.459 | 1.350 | 1.649 | 0.000 | n/a | 0.000 |
| OceanEmbed (no pretraining), last 2 years | 1 | 2023 | 1.184 | 1.112 | 1.308 | 0.393 | 0.621 | -0.137 |
| OceanEmbed (no pretraining), last 2 years | 1 | 2024 | 1.104 | 1.135 | 1.052 | 0.373 | 0.617 | -0.044 |
| OceanEmbed (no pretraining), last 2 years | 1 | pooled | 1.145 | 1.123 | 1.189 | 0.384 | 0.618 | -0.089 |
| OceanEmbed (no pretraining), last 5 years | 1 | 2023 | 1.022 | 1.015 | 1.027 | 0.548 | 0.732 | 0.010 |
| OceanEmbed (no pretraining), last 5 years | 1 | 2024 | 1.064 | 1.068 | 1.064 | 0.417 | 0.645 | 0.017 |
| OceanEmbed (no pretraining), last 5 years | 1 | pooled | 1.043 | 1.041 | 1.045 | 0.489 | 0.691 | 0.014 |

## Is the skill stable from year to year? (2024 minus 2023)

| Method | Region | Delta RMSE 50-200 m [95 % CI] | Delta skill |
|---|---|---|---|
| OceanEmbed (no pretraining) | whole domain | +0.029 [-0.058, +0.109] | -0.107 [-0.202, +0.041] |
| OceanEmbed (no pretraining) | Arabian Sea | +0.051 [-0.058, +0.159] | -0.096 [-0.210, +0.116] |
| OceanEmbed (no pretraining) | Bay of Bengal | -0.006 [-0.078, +0.064] | -0.094 [-0.205, +0.049] |
| Per-pixel MLP | whole domain | +0.039 [-0.075, +0.119] | -0.120 [-0.201, +0.047] |
| Per-pixel MLP | Arabian Sea | +0.032 [-0.099, +0.129] | -0.072 [-0.153, +0.151] |
| Per-pixel MLP | Bay of Bengal | +0.060 [-0.057, +0.169] | -0.161 [-0.277, -0.000] |
| Ridge regression | whole domain | -0.132 [-0.322, +0.077] | +0.032 [-0.060, +0.174] |
| Ridge regression | Arabian Sea | -0.071 [-0.249, +0.112] | +0.041 [-0.065, +0.233] |
| Ridge regression | Bay of Bengal | -0.233 [-0.505, +0.078] | +0.037 [-0.077, +0.175] |
| Climatology | whole domain | -0.126 [-0.377, +0.160] | +0.000 [+0.000, +0.000] |
| Climatology | Arabian Sea | -0.045 [-0.302, +0.264] | +0.000 [+0.000, +0.000] |
| Climatology | Bay of Bengal | -0.245 [-0.577, +0.096] | +0.000 [+0.000, +0.000] |

Bias (method minus GLORYS, whole domain, seed mean) by depth:

| Depth | OceanEmbed (no pretraining) 2023 | OceanEmbed (no pretraining) 2024 | Ridge regression 2023 | Ridge regression 2024 | Climatology 2023 | Climatology 2024 |
|---|---|---|---|---|---|---|
| 0 m | -0.13 | -0.21 | -0.36 | -0.50 | -0.43 | -0.62 |
| 5 m | -0.13 | -0.21 | -0.35 | -0.50 | -0.42 | -0.61 |
| 10 m | -0.13 | -0.21 | -0.34 | -0.49 | -0.41 | -0.61 |
| 20 m | -0.13 | -0.20 | -0.34 | -0.48 | -0.41 | -0.60 |
| 30 m | -0.16 | -0.23 | -0.33 | -0.48 | -0.41 | -0.62 |
| 50 m | -0.20 | -0.28 | -0.30 | -0.41 | -0.39 | -0.60 |
| 75 m | -0.22 | -0.12 | -0.25 | -0.11 | -0.36 | -0.38 |
| 100 m | -0.18 | +0.08 | -0.20 | +0.12 | -0.33 | -0.20 |
| 125 m | -0.08 | +0.18 | -0.10 | +0.19 | -0.23 | -0.12 |
| 150 m | +0.00 | +0.18 | -0.02 | +0.16 | -0.13 | -0.09 |
| 200 m | +0.04 | +0.08 | +0.03 | +0.06 | -0.03 | -0.09 |
| 300 m | +0.03 | +0.05 | +0.03 | +0.01 | +0.01 | -0.04 |
| 500 m | +0.01 | +0.03 | -0.00 | -0.02 | -0.02 | -0.05 |
| 700 m | +0.00 | +0.01 | -0.00 | -0.04 | -0.01 | -0.06 |
| 1000 m | -0.01 | +0.02 | -0.01 | -0.03 | -0.02 | -0.05 |

## Does the Transformer still beat the per-pixel MLP, and where?

RMSE(MLP) minus RMSE(Transformer), 50-200 m; positive = Transformer better.

| Period | Whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|
| 2023 | +0.010 [-0.000, +0.019] | -0.012 [-0.017, +0.002] | +0.045 [+0.004, +0.064] * |
| 2024 | +0.020 [-0.010, +0.025] | -0.031 [-0.087, +0.006] | +0.111 [+0.002, +0.203] * |
| pooled | +0.015 [+0.001, +0.026] | -0.022 [-0.054, -0.002] * | +0.078 [+0.023, +0.152] * |
| 2024, first run (5-year models) | +0.042 [-0.011, +0.061] | -0.006 [-0.067, +0.044] | +0.132 [+0.023, +0.196] * |

## Skill below about 300 m

RMSE of each method minus the climatology's, pooled 500-1000 m, whole domain (negative = better than climatology).

| Method | Period | Delta RMSE vs climatology [95 % CI] |
|---|---|---|
| OceanEmbed (no pretraining) | 2023 | +0.004 [-0.005, +0.017] |
| OceanEmbed (no pretraining) | 2024 | +0.006 [+0.002, +0.019] |
| OceanEmbed (no pretraining) | pooled | +0.005 [-0.003, +0.014] |
| Per-pixel MLP | 2023 | -0.010 [-0.013, -0.005] * |
| Per-pixel MLP | 2024 | -0.003 [-0.007, +0.007] |
| Per-pixel MLP | pooled | -0.006 [-0.010, -0.001] * |
| Ridge regression | 2023 | -0.008 [-0.012, -0.005] * |
| Ridge regression | 2024 | -0.003 [-0.008, +0.004] |
| Ridge regression | pooled | -0.006 [-0.009, -0.002] * |

## Eleven training years against the first run's five (same test year, 2024)

Same days: True; same scored points: True (n = 54320000 vs 54320000). RMSE differences are absolute; the two climatologies are fitted on different periods, so skill is relative to each run's own climatology.

| Method | Region | RMSE 50-200 m, long | first run | Delta (long - first) [95 % CI] | Delta skill |
|---|---|---|---|---|---|
| OceanEmbed (no pretraining) | whole domain | 1.004 ± 0.014 | 1.052 ± 0.015 | -0.048 [-0.059, -0.033] * | +0.054 |
| OceanEmbed (no pretraining) | Arabian Sea | 1.043 ± 0.024 | 1.089 ± 0.022 | -0.046 [-0.056, -0.029] | +0.065 |
| OceanEmbed (no pretraining) | Bay of Bengal | 0.934 ± 0.006 | 0.988 ± 0.016 | -0.053 [-0.070, -0.036] * | +0.040 |
| Per-pixel MLP | whole domain | 1.024 ± 0.008 | 1.094 ± 0.005 | -0.071 [-0.074, -0.049] * | +0.081 |
| Per-pixel MLP | Arabian Sea | 1.012 ± 0.005 | 1.083 ± 0.008 | -0.071 [-0.078, -0.043] * | +0.094 |
| Per-pixel MLP | Bay of Bengal | 1.045 ± 0.014 | 1.120 ± 0.021 | -0.075 [-0.083, -0.046] * | +0.065 |
| Ridge regression | whole domain | 1.138 | 1.155 | -0.016 [-0.029, +0.002] | +0.023 |
| Ridge regression | Arabian Sea | 1.113 | 1.139 | -0.025 [-0.048, +0.013] | +0.043 |
| Ridge regression | Bay of Bengal | 1.182 | 1.186 | -0.004 [-0.043, +0.023] | -0.002 |
| Climatology | whole domain | 1.394 | 1.390 | +0.004 [-0.018, +0.023] | +0.000 |
| Climatology | Arabian Sea | 1.326 | 1.317 | +0.010 [-0.017, +0.044] | +0.000 |
| Climatology | Bay of Bengal | 1.519 | 1.527 | -0.007 [-0.065, +0.031] | +0.000 |

## Learning curve (headline model, one seed, last N training years)

| Training years | Test period | RMSE 50-200 m | 95 % CI | Delta vs full period |
|---|---|---|---|---|
| 2 | 2023 | 1.184 | [1.046, 1.322] | +0.202 [+0.135, +0.291] |
| 2 | 2024 | 1.104 | [1.051, 1.136] | +0.084 [+0.072, +0.106] |
| 2 | pooled | 1.145 | [1.077, 1.243] | +0.145 [+0.099, +0.216] |
| 5 | 2023 | 1.022 | [0.916, 1.101] | +0.040 [+0.012, +0.077] |
| 5 | 2024 | 1.064 | [1.013, 1.091] | +0.044 [+0.025, +0.070] |
| 5 | pooled | 1.043 | [0.990, 1.098] | +0.042 [+0.025, +0.066] |
| 11 | 2023 | 0.982 | [0.901, 1.034] | +0.000 [+0.000, +0.000] |
| 11 | 2024 | 1.020 | [0.952, 1.058] | +0.000 [+0.000, +0.000] |
| 11 | pooled | 1.000 | [0.950, 1.050] | +0.000 [+0.000, +0.000] |

## Argo profiles (whole domain, 50-200 m pooled)

GLORYS assimilates Argo and the model is trained on GLORYS: not an independent error estimate; GLORYS-vs-Argo is the reference floor.

| Year | Method | Matchups | RMSE vs Argo | 95 % CI (profiles) | Bias |
|---|---|---|---|---|---|
| 2023 | OceanEmbed (no pretraining) | 37329 | 1.218 | [1.187, 1.249] | +0.474 |
| 2023 | Ridge regression | 37329 | 1.279 | [1.240, 1.319] | +0.331 |
| 2023 | Climatology | 37329 | 1.454 | [1.408, 1.504] | +0.173 |
| 2023 | GLORYS | 37329 | 1.022 | [0.993, 1.050] | +0.414 |
| 2024 | OceanEmbed (no pretraining) | 40059 | 1.381 | [1.348, 1.413] | +0.648 |
| 2024 | Ridge regression | 40059 | 1.457 | [1.421, 1.496] | +0.610 |
| 2024 | Climatology | 40059 | 1.573 | [1.530, 1.620] | +0.328 |
| 2024 | GLORYS | 40059 | 1.079 | [1.049, 1.112] | +0.494 |
| pooled | OceanEmbed (no pretraining) | 77388 | 1.304 | [1.281, 1.327] | +0.563 |
| pooled | Ridge regression | 77388 | 1.373 | [1.345, 1.401] | +0.474 |
| pooled | Climatology | 77388 | 1.516 | [1.483, 1.551] | +0.252 |
| pooled | GLORYS | 77388 | 1.051 | [1.031, 1.072] | +0.455 |

Figures: `figures/rmse_by_depth.png`, `long_vs_first_run_2024.png`, `years_bias_by_depth.png`, `learning_curve.png`, `argo_by_depth.png`. Numbers: `summary.json`.
