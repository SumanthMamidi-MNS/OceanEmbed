# R5 physical metrics - run `poc`

Test period 2024-01-01 .. 2024-12-15 (350 days), reference GLORYS. Intervals are 95 % moving-block bootstrap over days (block 41 days, 2000 replicates). Derived quantities are evaluated on the cells where every product and GLORYS have a defined value (one common sample).

## Derived quantities, whole domain

### Depth of the 20 degC isotherm (m)

Defined in 81.6 % of GLORYS ocean cell-days; 81.4 % in every product (the scored sample, 3,413,324 cell-days).

| Product | RMSE (m) | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 17.633 [16.246, 18.440] | 1.615 | - | 0.000 [0.000, 0.000] |
| Ridge regression | 14.983 [13.624, 15.675] | 3.121 | 0.552 [0.511, 0.600] | 0.278 [0.235, 0.330] |
| Per-pixel MLP (R1, seed 0) | 14.096 [12.871, 14.847] | 3.203 | 0.629 [0.595, 0.665] | 0.361 [0.310, 0.408] |
| OceanEmbed (no pretraining) | 13.832 [13.070, 14.318] | 2.744 | 0.648 [0.622, 0.670] | 0.385 [0.337, 0.424] |
| OceanEmbed (pretrained) | 14.552 [13.626, 15.177] | 2.713 | 0.608 [0.582, 0.630] | 0.319 [0.276, 0.352] |

### Depth of the 23 degC isotherm (m)

Defined in 82.9 % of GLORYS ocean cell-days; 82.1 % in every product (the scored sample, 3,445,121 cell-days).

| Product | RMSE (m) | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 16.478 [15.818, 16.931] | -0.317 | - | 0.000 [0.000, 0.000] |
| Ridge regression | 13.483 [12.466, 14.057] | 2.058 | 0.588 [0.535, 0.652] | 0.331 [0.274, 0.408] |
| Per-pixel MLP (R1, seed 0) | 12.616 [11.643, 13.218] | 2.737 | 0.666 [0.632, 0.705] | 0.414 [0.359, 0.478] |
| OceanEmbed (no pretraining) | 12.451 [11.761, 13.158] | 1.448 | 0.667 [0.604, 0.708] | 0.429 [0.340, 0.485] |
| OceanEmbed (pretrained) | 12.759 [12.078, 13.407] | 1.570 | 0.655 [0.610, 0.687] | 0.400 [0.335, 0.445] |

### Heat content 0-300 m (1e9 J m-2)

Defined in 80.0 % of GLORYS ocean cell-days; 80.0 % in every product (the scored sample, 3,355,800 cell-days).

| Product | RMSE (1e9 J m-2) | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.024 [0.962, 1.059] | -0.118 | - | 0.000 [0.000, 0.000] |
| Ridge regression | 0.807 [0.729, 0.858] | 0.003 | 0.610 [0.532, 0.683] | 0.379 [0.290, 0.466] |
| Per-pixel MLP (R1, seed 0) | 0.729 [0.671, 0.769] | 0.065 | 0.702 [0.667, 0.730] | 0.494 [0.442, 0.541] |
| OceanEmbed (no pretraining) | 0.721 [0.689, 0.742] | 0.013 | 0.712 [0.678, 0.733] | 0.504 [0.451, 0.538] |
| OceanEmbed (pretrained) | 0.760 [0.721, 0.783] | 0.006 | 0.678 [0.637, 0.708] | 0.449 [0.393, 0.489] |

### Mean temperature 0-300 m (heat content / (rho cp 300 m)) (degC)

Defined in 80.0 % of GLORYS ocean cell-days; 80.0 % in every product (the scored sample, 3,355,800 cell-days).

| Product | RMSE (degC) | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 0.835 [0.784, 0.863] | -0.096 | - | 0.000 [0.000, 0.000] |
| Ridge regression | 0.658 [0.594, 0.700] | 0.002 | 0.610 [0.532, 0.683] | 0.379 [0.290, 0.466] |
| Per-pixel MLP (R1, seed 0) | 0.594 [0.547, 0.627] | 0.053 | 0.702 [0.667, 0.730] | 0.494 [0.442, 0.541] |
| OceanEmbed (no pretraining) | 0.588 [0.561, 0.605] | 0.011 | 0.712 [0.678, 0.733] | 0.504 [0.451, 0.538] |
| OceanEmbed (pretrained) | 0.620 [0.588, 0.638] | 0.005 | 0.678 [0.637, 0.708] | 0.449 [0.393, 0.489] |

### Mean temperature 0-30 m (mixed-layer proxy) (degC)

Defined in 91.9 % of GLORYS ocean cell-days; 91.9 % in every product (the scored sample, 3,856,300 cell-days).

| Product | RMSE (degC) | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 0.790 [0.733, 0.868] | -0.500 | - | 0.000 [0.000, 0.000] |
| Ridge regression | 0.716 [0.643, 0.809] | -0.431 | 0.357 [0.289, 0.443] | 0.179 [0.097, 0.273] |
| Per-pixel MLP (R1, seed 0) | 0.508 [0.494, 0.541] | -0.244 | 0.686 [0.619, 0.742] | 0.587 [0.484, 0.656] |
| OceanEmbed (no pretraining) | 0.567 [0.516, 0.640] | -0.300 | 0.625 [0.578, 0.683] | 0.485 [0.386, 0.568] |
| OceanEmbed (pretrained) | 0.578 [0.519, 0.652] | -0.283 | 0.584 [0.529, 0.656] | 0.464 [0.367, 0.563] |

## D20, D23 and heat content by basin (RMSE)

**Depth of the 20 degC isotherm** (RMSE, m)

| Product | Arabian Sea | Bay of Bengal |
|---|---|---|
| Climatology | 18.689 [16.897, 20.149] | 15.613 [14.207, 16.094] |
| Ridge regression | 16.551 [15.115, 17.531] | 11.516 [10.119, 12.334] |
| Per-pixel MLP (R1, seed 0) | 15.792 [14.240, 16.762] | 10.256 [9.620, 10.813] |
| OceanEmbed (no pretraining) | 15.582 [14.727, 16.060] | 9.727 [8.868, 10.536] |
| OceanEmbed (pretrained) | 16.336 [15.243, 17.106] | 10.484 [9.744, 11.012] |

**Depth of the 23 degC isotherm** (RMSE, m)

| Product | Arabian Sea | Bay of Bengal |
|---|---|---|
| Climatology | 17.339 [16.599, 18.377] | 14.853 [13.450, 15.271] |
| Ridge regression | 14.587 [13.490, 15.497] | 10.958 [9.795, 11.459] |
| Per-pixel MLP (R1, seed 0) | 13.725 [12.528, 14.563] | 10.247 [9.398, 10.862] |
| OceanEmbed (no pretraining) | 13.916 [13.095, 14.724] | 8.732 [8.172, 9.488] |
| OceanEmbed (pretrained) | 14.403 [13.525, 15.211] | 8.847 [8.475, 9.342] |

**Heat content 0-300 m** (RMSE, 1e9 J m-2)

| Product | Arabian Sea | Bay of Bengal |
|---|---|---|
| Climatology | 1.025 [0.958, 1.099] | 1.032 [0.932, 1.051] |
| Ridge regression | 0.857 [0.775, 0.916] | 0.708 [0.600, 0.780] |
| Per-pixel MLP (R1, seed 0) | 0.784 [0.718, 0.836] | 0.617 [0.562, 0.664] |
| OceanEmbed (no pretraining) | 0.786 [0.747, 0.810] | 0.588 [0.551, 0.624] |
| OceanEmbed (pretrained) | 0.819 [0.772, 0.847] | 0.646 [0.596, 0.688] |

## Pooled 50-200 m temperature by season and basin

**whole domain**: RMSE (degC) [95 % CI] / anomaly correlation

| Season (days) | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| all year (350) | 1.39 [1.31, 1.42] / - | 1.15 [1.06, 1.20] / 0.56 | 1.10 [1.00, 1.15] / 0.62 | 1.04 [1.00, 1.07] / 0.67 | 1.08 [1.02, 1.11] / 0.64 |
| winter monsoon (Dec-Feb) (75) | 1.53 [1.32, 1.59] / - | 1.34 [1.16, 1.41] / 0.49 | 1.21 [1.12, 1.26] / 0.63 | 1.08 [1.06, 1.12] / 0.71 | 1.15 [1.13, 1.18] / 0.66 |
| pre-monsoon (Mar-May) (92) | 1.36 [1.30, 1.43] / - | 1.11 [1.04, 1.18] / 0.58 | 1.06 [1.01, 1.10] / 0.62 | 1.05 [1.04, 1.06] / 0.63 | 1.08 [1.05, 1.10] / 0.61 |
| summer monsoon (Jun-Sep) (122) | 1.29 [1.26, 1.34] / - | 1.05 [0.98, 1.13] / 0.58 | 0.97 [0.92, 1.05] / 0.66 | 1.00 [0.94, 1.08] / 0.65 | 1.00 [0.95, 1.08] / 0.65 |
| post-monsoon (Oct-Nov) (61) | 1.44 [1.34, 1.48] / - | 1.18 [1.15, 1.19] / 0.59 | 1.22 [1.16, 1.26] / 0.58 | 1.07 [1.04, 1.11] / 0.67 | 1.11 [1.08, 1.12] / 0.65 |

**Arabian Sea**: RMSE (degC) [95 % CI] / anomaly correlation

| Season (days) | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| all year (350) | 1.32 [1.26, 1.38] / - | 1.14 [1.06, 1.19] / 0.48 | 1.07 [0.98, 1.14] / 0.56 | 1.08 [1.03, 1.12] / 0.56 | 1.11 [1.05, 1.15] / 0.53 |
| winter monsoon (Dec-Feb) (75) | 1.32 [1.25, 1.35] / - | 1.23 [1.17, 1.25] / 0.32 | 1.12 [1.06, 1.31] / 0.50 | 1.13 [1.09, 1.19] / 0.50 | 1.16 [1.11, 1.28] / 0.46 |
| pre-monsoon (Mar-May) (92) | 1.29 [1.23, 1.39] / - | 1.10 [1.02, 1.20] / 0.42 | 1.03 [0.99, 1.04] / 0.53 | 1.13 [1.10, 1.14] / 0.39 | 1.16 [1.12, 1.18] / 0.35 |
| summer monsoon (Jun-Sep) (122) | 1.25 [1.20, 1.31] / - | 1.06 [0.97, 1.15] / 0.51 | 0.95 [0.91, 1.02] / 0.63 | 1.00 [0.96, 1.07] / 0.59 | 1.01 [0.96, 1.09] / 0.59 |
| post-monsoon (Oct-Nov) (61) | 1.47 [1.38, 1.55] / - | 1.24 [1.20, 1.27] / 0.56 | 1.28 [1.24, 1.31] / 0.54 | 1.10 [1.07, 1.18] / 0.66 | 1.16 [1.14, 1.19] / 0.64 |

**Bay of Bengal**: RMSE (degC) [95 % CI] / anomaly correlation

| Season (days) | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| all year (350) | 1.53 [1.38, 1.57] / - | 1.19 [1.05, 1.26] / 0.64 | 1.14 [1.02, 1.23] / 0.69 | 0.97 [0.91, 1.02] / 0.77 | 1.02 [0.96, 1.08] / 0.74 |
| winter monsoon (Dec-Feb) (75) | 1.87 [1.37, 1.97] / - | 1.52 [1.15, 1.66] / 0.59 | 1.38 [1.13, 1.51] / 0.69 | 0.99 [0.92, 1.01] / 0.84 | 1.15 [0.99, 1.25] / 0.78 |
| pre-monsoon (Mar-May) (92) | 1.49 [1.38, 1.55] / - | 1.13 [1.07, 1.17] / 0.63 | 1.13 [1.04, 1.20] / 0.68 | 0.90 [0.88, 0.92] / 0.78 | 0.94 [0.93, 0.95] / 0.76 |
| summer monsoon (Jun-Sep) (122) | 1.37 [1.35, 1.40] / - | 1.04 [0.99, 1.09] / 0.69 | 1.01 [0.93, 1.11] / 0.71 | 0.99 [0.88, 1.14] / 0.72 | 1.00 [0.91, 1.09] / 0.71 |
| post-monsoon (Oct-Nov) (61) | 1.39 [1.23, 1.47] / - | 1.07 [0.97, 1.12] / 0.64 | 1.09 [0.98, 1.16] / 0.66 | 1.00 [0.97, 1.01] / 0.69 | 1.01 [0.93, 1.04] / 0.70 |

## Eddy regime (abs(SLA) terciles), whole domain

- abs(SLA) < 0.083 m: 6,889,878 cell-day-depth samples
- 0.083 - 0.158 m: 7,100,827 cell-day-depth samples
- abs(SLA) >= 0.158 m: 6,934,395 cell-day-depth samples

Pooled 50-200 m RMSE (degC)

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | 1.43 [1.32, 1.49] | 1.23 [1.13, 1.28] | 1.16 [1.07, 1.19] | 1.07 [1.03, 1.09] | 1.11 [1.06, 1.14] |
| 0.083 - 0.158 m | 1.23 [1.14, 1.27] | 1.10 [1.01, 1.16] | 1.06 [0.97, 1.12] | 1.00 [0.96, 1.03] | 1.03 [0.97, 1.06] |
| abs(SLA) >= 0.158 m | 1.51 [1.38, 1.62] | 1.13 [1.03, 1.22] | 1.07 [0.98, 1.16] | 1.05 [0.97, 1.12] | 1.09 [0.99, 1.18] |

Anomaly correlation

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | - | 0.41 | 0.53 | 0.62 | 0.59 |
| 0.083 - 0.158 m | - | 0.44 | 0.53 | 0.59 | 0.57 |
| abs(SLA) >= 0.158 m | - | 0.53 | 0.60 | 0.62 | 0.59 |

Gain of OceanEmbed over a reference (RMSE_ref - RMSE_method, degC, positive = better)

| Method over reference | abs(SLA) < 0.083 m | 0.083 - 0.158 m | abs(SLA) >= 0.158 m | high - low |
|---|---|---|---|---|
| OceanEmbed (pretrained) over Climatology | 0.311 [0.206, 0.389] | 0.196 [0.155, 0.220] | 0.421 [0.314, 0.495] | 0.110 [-0.037, 0.264] |
| OceanEmbed (pretrained) over Ridge regression | 0.115 [0.034, 0.170] | 0.073 [0.032, 0.102] | 0.043 [-0.017, 0.086] | -0.073 [-0.161, 0.036] |
| OceanEmbed (pretrained) over Per-pixel MLP (R1, seed 0) | 0.042 [-0.041, 0.086] | 0.028 [-0.008, 0.057] | -0.015 [-0.063, 0.026] | -0.057 [-0.129, 0.030] |
| OceanEmbed (no pretraining) over Climatology | 0.353 [0.240, 0.435] | 0.225 [0.165, 0.260] | 0.457 [0.357, 0.536] | 0.104 [-0.041, 0.264] |
| OceanEmbed (no pretraining) over Ridge regression | 0.157 [0.059, 0.227] | 0.103 [0.041, 0.142] | 0.079 [0.022, 0.129] | -0.078 [-0.167, 0.032] |
| OceanEmbed (no pretraining) over Per-pixel MLP (R1, seed 0) | 0.084 [-0.010, 0.133] | 0.057 [-0.001, 0.101] | 0.021 [-0.024, 0.067] | -0.063 [-0.122, 0.013] |

`*`: the interval of the high-minus-low contrast excludes zero.

## Eddy regime (abs(SLA) terciles), Arabian Sea

- abs(SLA) < 0.083 m: 4,152,300 cell-day-depth samples
- 0.083 - 0.158 m: 4,589,132 cell-day-depth samples
- abs(SLA) >= 0.158 m: 4,527,768 cell-day-depth samples

Pooled 50-200 m RMSE (degC)

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | 1.22 [1.15, 1.27] | 1.14 [1.11, 1.18] | 1.13 [1.02, 1.20] | 1.10 [1.07, 1.12] | 1.13 [1.10, 1.15] |
| 0.083 - 0.158 m | 1.20 [1.07, 1.29] | 1.10 [1.00, 1.16] | 1.00 [0.92, 1.07] | 1.04 [0.98, 1.08] | 1.05 [0.98, 1.10] |
| abs(SLA) >= 0.158 m | 1.50 [1.39, 1.66] | 1.17 [1.07, 1.30] | 1.09 [0.99, 1.18] | 1.11 [1.03, 1.18] | 1.15 [1.04, 1.24] |

Anomaly correlation

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | - | 0.35 | 0.40 | 0.46 | 0.42 |
| 0.083 - 0.158 m | - | 0.38 | 0.54 | 0.51 | 0.50 |
| abs(SLA) >= 0.158 m | - | 0.49 | 0.58 | 0.56 | 0.53 |

Gain of OceanEmbed over a reference (RMSE_ref - RMSE_method, degC, positive = better)

| Method over reference | abs(SLA) < 0.083 m | 0.083 - 0.158 m | abs(SLA) >= 0.158 m | high - low |
|---|---|---|---|---|
| OceanEmbed (pretrained) over Climatology | 0.084 [0.016, 0.136] | 0.149 [0.062, 0.217] | 0.354 [0.246, 0.490] | 0.270 [0.187, 0.376] * |
| OceanEmbed (pretrained) over Ridge regression | 0.007 [-0.024, 0.051] | 0.049 [-0.004, 0.090] | 0.024 [-0.055, 0.117] | 0.017 [-0.074, 0.104] |
| OceanEmbed (pretrained) over Per-pixel MLP (R1, seed 0) | -0.002 [-0.111, 0.061] | -0.051 [-0.105, -0.004] | -0.061 [-0.100, -0.010] | -0.059 [-0.120, 0.038] |
| OceanEmbed (no pretraining) over Climatology | 0.121 [0.042, 0.181] | 0.165 [0.063, 0.242] | 0.390 [0.286, 0.525] | 0.270 [0.186, 0.366] * |
| OceanEmbed (no pretraining) over Ridge regression | 0.044 [0.011, 0.085] | 0.065 [-0.002, 0.108] | 0.060 [-0.025, 0.164] | 0.016 [-0.074, 0.104] |
| OceanEmbed (no pretraining) over Per-pixel MLP (R1, seed 0) | 0.035 [-0.085, 0.103] | -0.035 [-0.103, 0.026] | -0.024 [-0.070, 0.034] | -0.059 [-0.127, 0.040] |

`*`: the interval of the high-minus-low contrast excludes zero.

## Eddy regime (abs(SLA) terciles), Bay of Bengal

- abs(SLA) < 0.083 m: 2,599,593 cell-day-depth samples
- 0.083 - 0.158 m: 2,353,857 cell-day-depth samples
- abs(SLA) >= 0.158 m: 2,323,400 cell-day-depth samples

Pooled 50-200 m RMSE (degC)

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | 1.71 [1.50, 1.80] | 1.36 [1.16, 1.44] | 1.20 [1.07, 1.28] | 1.04 [0.97, 1.12] | 1.09 [0.99, 1.17] |
| 0.083 - 0.158 m | 1.29 [1.18, 1.35] | 1.11 [1.00, 1.20] | 1.17 [1.02, 1.29] | 0.94 [0.90, 0.97] | 1.00 [0.92, 1.06] |
| abs(SLA) >= 0.158 m | 1.53 [1.33, 1.63] | 1.04 [0.96, 1.10] | 1.04 [0.94, 1.11] | 0.92 [0.83, 1.01] | 0.96 [0.89, 1.05] |

Anomaly correlation

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| abs(SLA) < 0.083 m | - | 0.32 | 0.55 | 0.67 | 0.63 |
| 0.083 - 0.158 m | - | 0.49 | 0.50 | 0.66 | 0.61 |
| abs(SLA) >= 0.158 m | - | 0.63 | 0.66 | 0.72 | 0.70 |

Gain of OceanEmbed over a reference (RMSE_ref - RMSE_method, degC, positive = better)

| Method over reference | abs(SLA) < 0.083 m | 0.083 - 0.158 m | abs(SLA) >= 0.158 m | high - low |
|---|---|---|---|---|
| OceanEmbed (pretrained) over Climatology | 0.623 [0.416, 0.684] | 0.287 [0.213, 0.353] | 0.568 [0.414, 0.613] | -0.055 [-0.185, 0.115] |
| OceanEmbed (pretrained) over Ridge regression | 0.269 [0.081, 0.312] | 0.114 [0.049, 0.171] | 0.079 [0.005, 0.115] | -0.189 [-0.245, -0.031] * |
| OceanEmbed (pretrained) over Per-pixel MLP (R1, seed 0) | 0.112 [-0.025, 0.170] | 0.170 [0.071, 0.251] | 0.076 [0.009, 0.123] | -0.036 [-0.114, 0.090] |
| OceanEmbed (no pretraining) over Climatology | 0.678 [0.423, 0.773] | 0.349 [0.237, 0.426] | 0.608 [0.442, 0.678] | -0.071 [-0.219, 0.139] |
| OceanEmbed (no pretraining) over Ridge regression | 0.324 [0.067, 0.411] | 0.175 [0.073, 0.258] | 0.119 [0.030, 0.191] | -0.205 [-0.282, 0.002] |
| OceanEmbed (no pretraining) over Per-pixel MLP (R1, seed 0) | 0.167 [-0.028, 0.264] | 0.232 [0.098, 0.351] | 0.115 [0.055, 0.173] | -0.052 [-0.119, 0.116] |

`*`: the interval of the high-minus-low contrast excludes zero.

## Bay of Bengal by surface-salinity tercile (barrier-layer proxy, low salinity = fresh)

- SSS < 31.99: 1,947,055 cell-day-depth samples
- 31.99 - 32.89: 2,591,438 cell-day-depth samples
- SSS >= 32.89: 2,738,357 cell-day-depth samples

Pooled 50-200 m RMSE (degC)

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| SSS < 31.99 | 1.44 [1.12, 1.58] | 1.18 [0.94, 1.34] | 1.26 [1.01, 1.48] | 0.93 [0.84, 1.00] | 1.02 [0.86, 1.20] |
| 31.99 - 32.89 | 1.47 [1.28, 1.53] | 1.15 [0.98, 1.23] | 1.08 [0.97, 1.13] | 0.89 [0.85, 0.93] | 0.92 [0.85, 0.97] |
| SSS >= 32.89 | 1.64 [1.52, 1.68] | 1.23 [1.12, 1.30] | 1.10 [1.03, 1.21] | 1.06 [0.94, 1.22] | 1.10 [1.01, 1.21] |

Anomaly correlation

| Tercile | Climatology | Ridge regression | Per-pixel MLP (R1, seed 0) | OceanEmbed (no pretraining) | OceanEmbed (pretrained) |
|---|---|---|---|---|---|
| SSS < 31.99 | - | 0.56 | 0.57 | 0.75 | 0.69 |
| 31.99 - 32.89 | - | 0.63 | 0.70 | 0.79 | 0.77 |
| SSS >= 32.89 | - | 0.71 | 0.75 | 0.76 | 0.75 |

Gain of OceanEmbed over a reference (RMSE_ref - RMSE_method, degC, positive = better)

| Method over reference | SSS < 31.99 | 31.99 - 32.89 | SSS >= 32.89 | high - low |
|---|---|---|---|---|
| OceanEmbed (pretrained) over Climatology | 0.413 [0.219, 0.436] | 0.545 [0.409, 0.582] | 0.535 [0.383, 0.585] | 0.122 [0.066, 0.282] * |
| OceanEmbed (pretrained) over Ridge regression | 0.157 [0.045, 0.189] | 0.222 [0.105, 0.273] | 0.123 [0.001, 0.201] | -0.034 [-0.124, 0.107] |
| OceanEmbed (pretrained) over Per-pixel MLP (R1, seed 0) | 0.238 [0.130, 0.298] | 0.158 [0.088, 0.200] | 0.002 [-0.086, 0.105] | -0.236 [-0.338, -0.085] * |
| OceanEmbed (no pretraining) over Climatology | 0.510 [0.246, 0.595] | 0.577 [0.400, 0.643] | 0.575 [0.390, 0.653] | 0.065 [-0.028, 0.280] |
| OceanEmbed (no pretraining) over Ridge regression | 0.254 [0.081, 0.353] | 0.254 [0.098, 0.337] | 0.163 [0.009, 0.271] | -0.092 [-0.203, 0.089] |
| OceanEmbed (no pretraining) over Per-pixel MLP (R1, seed 0) | 0.335 [0.150, 0.485] | 0.190 [0.094, 0.241] | 0.042 [-0.076, 0.174] | -0.293 [-0.443, -0.061] * |

`*`: the interval of the high-minus-low contrast excludes zero.

### Is it a season effect? Share of each salinity tercile's samples by season

| Tercile | winter monsoon (Dec-Feb) | pre-monsoon (Mar-May) | summer monsoon (Jun-Sep) | post-monsoon (Oct-Nov) |
|---|---|---|---|---|
| SSS < 31.99 | 36 % | 9 % | 23 % | 32 % |
| 31.99 - 32.89 | 21 % | 38 % | 27 % | 15 % |
| SSS >= 32.89 | 12 % | 28 % | 51 % | 10 % |

Gain over the per-pixel MLP (RMSE_MLP - RMSE_method, degC) by salinity tercile, within each season

| Method | Season | SSS < 31.99 | 31.99 - 32.89 | SSS >= 32.89 |
|---|---|---|---|---|
| OceanEmbed (pretrained) | winter monsoon (Dec-Feb) | 0.31 [0.11, 0.37] | 0.18 [0.14, 0.32] | 0.07 [-0.04, 0.14] |
| OceanEmbed (pretrained) | pre-monsoon (Mar-May) | 0.30 [0.15, 0.53] | 0.21 [0.16, 0.26] | 0.14 [0.04, 0.28] |
| OceanEmbed (pretrained) | summer monsoon (Jun-Sep) | 0.24 [0.04, 0.32] | 0.11 [0.01, 0.20] | -0.10 [-0.20, 0.00] |
| OceanEmbed (pretrained) | post-monsoon (Oct-Nov) | 0.11 [0.05, 0.14] | 0.07 [-0.02, 0.18] | 0.04 [-0.02, 0.07] |
| OceanEmbed (no pretraining) | winter monsoon (Dec-Feb) | 0.54 [0.18, 0.66] | 0.32 [0.27, 0.43] | 0.11 [-0.05, 0.25] |
| OceanEmbed (no pretraining) | pre-monsoon (Mar-May) | 0.26 [0.13, 0.44] | 0.23 [0.17, 0.28] | 0.22 [0.09, 0.34] |
| OceanEmbed (no pretraining) | summer monsoon (Jun-Sep) | 0.23 [0.07, 0.29] | 0.09 [0.01, 0.16] | -0.07 [-0.27, 0.08] |
| OceanEmbed (no pretraining) | post-monsoon (Oct-Nov) | 0.13 [0.07, 0.20] | 0.05 [-0.03, 0.15] | 0.06 [-0.14, 0.10] |
