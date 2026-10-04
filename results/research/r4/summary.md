# R4 external benchmark: ARMOR3D - run `poc`

ARMOR3D `cmems_obs-mob_glo_phy_my_0.125deg_P1D-m` (version 202511), temperature `to`; 12 monthly files, 754 MB; days with a field: 350 of 350 (2024-01-01 .. 2024-12-15). Intervals: 95 % moving-block bootstrap over days (block 38 d, 2000 replicates).

**Independence:** ARMOR3D is built from satellite observations and in-situ profiles, Argo among them, so it is not independent of the floats used here (GLORYS assimilates them too); OceanEmbed sees no in-situ data at prediction time, but is trained on GLORYS.

## Against Argo (identical profile levels for every product)

Matchups kept: 40,052 of 40,059 (2,826 of 2,826 profiles); the rest have no ARMOR3D value in that cell / level.

### whole domain, pooled 50-200 m and all depths

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. | RMSE all depths |
|---|---|---|---|---|---|
| Climatology | 1.67 [1.55, 1.78] | 0.51 | - | 0.00 | 1.23 [1.14, 1.31] |
| Ridge regression | 1.53 [1.44, 1.60] | 0.68 | 0.51 | 0.16 | 1.13 [1.06, 1.19] |
| OceanEmbed (no pretraining) | 1.41 [1.39, 1.47] | 0.65 | 0.61 | 0.28 | 1.02 [1.00, 1.07] |
| OceanEmbed (pretrained) | 1.41 [1.38, 1.47] | 0.64 | 0.61 | 0.28 | 1.03 [0.99, 1.08] |
| ARMOR3D | 0.63 [0.57, 0.69] | 0.09 | 0.92 | 0.86 | 0.48 [0.43, 0.53] |
| GLORYS | 1.08 [1.02, 1.15] | 0.49 | 0.80 | 0.58 | 0.77 [0.72, 0.84] |
n = 16,241 (50-200 m), 40,052 (all depths).

### Arabian Sea, pooled 50-200 m and all depths

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. | RMSE all depths |
|---|---|---|---|---|---|
| Climatology | 1.52 [1.35, 1.68] | 0.33 | - | 0.00 | 1.19 [1.06, 1.30] |
| Ridge regression | 1.43 [1.29, 1.54] | 0.54 | 0.46 | 0.12 | 1.11 [1.01, 1.19] |
| OceanEmbed (no pretraining) | 1.38 [1.34, 1.43] | 0.55 | 0.53 | 0.18 | 1.03 [1.00, 1.07] |
| OceanEmbed (pretrained) | 1.38 [1.33, 1.44] | 0.55 | 0.52 | 0.18 | 1.04 [0.99, 1.10] |
| ARMOR3D | 0.58 [0.52, 0.62] | 0.07 | 0.93 | 0.86 | 0.46 [0.42, 0.51] |
| GLORYS | 1.00 [0.96, 1.04] | 0.42 | 0.79 | 0.57 | 0.75 [0.70, 0.81] |
n = 11,925 (50-200 m), 29,434 (all depths).

### Bay of Bengal, pooled 50-200 m and all depths

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. | RMSE all depths |
|---|---|---|---|---|---|
| Climatology | 2.02 [1.85, 2.19] | 1.01 | - | 0.00 | 1.34 [1.25, 1.44] |
| Ridge regression | 1.78 [1.65, 1.93] | 1.06 | 0.61 | 0.22 | 1.19 [1.11, 1.27] |
| OceanEmbed (no pretraining) | 1.51 [1.39, 1.67] | 0.92 | 0.73 | 0.44 | 1.00 [0.93, 1.10] |
| OceanEmbed (pretrained) | 1.51 [1.43, 1.64] | 0.90 | 0.72 | 0.44 | 1.00 [0.96, 1.08] |
| ARMOR3D | 0.75 [0.65, 0.89] | 0.18 | 0.91 | 0.86 | 0.51 [0.44, 0.59] |
| GLORYS | 1.27 [1.13, 1.44] | 0.69 | 0.79 | 0.61 | 0.83 [0.75, 0.94] |
n = 4,310 (50-200 m), 10,603 (all depths).

### RMSE by depth (whole domain, degC)

| Product | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.87 | 0.89 | 0.94 | 1.15 | 1.25 | 1.45 | 1.84 | 2.09 | 1.88 | 1.50 | 0.96 | 0.63 | 0.37 | 0.37 | 0.33 |
| Ridge regression | 0.78 | 0.80 | 0.85 | 1.07 | 1.15 | 1.27 | 1.59 | 1.92 | 1.79 | 1.43 | 0.92 | 0.61 | 0.37 | 0.36 | 0.33 |
| OceanEmbed (no pretraining) | 0.58 | 0.62 | 0.68 | 0.90 | 1.02 | 1.19 | 1.47 | 1.80 | 1.64 | 1.28 | 0.84 | 0.55 | 0.37 | 0.37 | 0.37 |
| OceanEmbed (pretrained) | 0.62 | 0.64 | 0.69 | 0.92 | 1.01 | 1.16 | 1.43 | 1.80 | 1.68 | 1.30 | 0.87 | 0.58 | 0.38 | 0.37 | 0.35 |
| ARMOR3D | 0.59 | 0.25 | 0.32 | 0.44 | 0.48 | 0.63 | 0.75 | 0.75 | 0.66 | 0.54 | 0.37 | 0.21 | 0.15 | 0.15 | 0.14 |
| GLORYS | 0.39 | 0.39 | 0.48 | 0.64 | 0.74 | 0.91 | 1.19 | 1.36 | 1.21 | 0.95 | 0.67 | 0.44 | 0.29 | 0.32 | 0.36 |

### Bias by depth (product minus Argo, whole domain, degC)

| Product | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | -0.55 | -0.60 | -0.56 | -0.47 | -0.40 | -0.23 | 0.30 | 0.89 | 0.92 | 0.67 | 0.36 | 0.20 | 0.03 | -0.05 | -0.02 |
| Ridge regression | -0.47 | -0.52 | -0.47 | -0.38 | -0.30 | -0.10 | 0.49 | 1.11 | 1.13 | 0.83 | 0.45 | 0.23 | 0.04 | -0.04 | -0.00 |
| OceanEmbed (no pretraining) | -0.31 | -0.37 | -0.34 | -0.26 | -0.26 | -0.09 | 0.47 | 1.09 | 1.06 | 0.80 | 0.40 | 0.20 | 0.04 | -0.05 | 0.02 |
| OceanEmbed (pretrained) | -0.35 | -0.38 | -0.34 | -0.25 | -0.20 | -0.08 | 0.42 | 1.06 | 1.09 | 0.77 | 0.42 | 0.23 | 0.07 | -0.02 | 0.02 |
| ARMOR3D | -0.03 | -0.00 | 0.00 | 0.02 | 0.02 | 0.05 | 0.07 | 0.14 | 0.12 | 0.12 | 0.06 | 0.03 | 0.01 | 0.01 | 0.02 |
| GLORYS | -0.01 | -0.05 | -0.02 | 0.08 | 0.15 | 0.26 | 0.50 | 0.77 | 0.67 | 0.46 | 0.26 | 0.10 | -0.00 | -0.04 | -0.03 |

### Paired RMSE differences, pooled 50-200 m (A - B; negative = A better)

| A - B | whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|
| OceanEmbed (pretrained) - ARMOR3D | 0.787 [0.735, 0.853] | 0.804 [0.755, 0.866] | 0.755 [0.680, 0.846] |
| OceanEmbed (no pretraining) - ARMOR3D | 0.787 [0.748, 0.839] | 0.802 [0.754, 0.863] | 0.759 [0.696, 0.835] |
| GLORYS - ARMOR3D | 0.452 [0.418, 0.485] | 0.425 [0.393, 0.459] | 0.518 [0.448, 0.581] |
| ARMOR3D - Ridge regression | -0.904 [-0.967, -0.803] | -0.853 [-0.943, -0.737] | -1.032 [-1.144, -0.901] |
| ARMOR3D - Climatology | -1.041 [-1.133, -0.924] | -0.946 [-1.081, -0.797] | -1.269 [-1.426, -1.075] |
| OceanEmbed (pretrained) - GLORYS | 0.336 [0.285, 0.399] | 0.379 [0.334, 0.439] | 0.237 [0.151, 0.350] |
| OceanEmbed (pretrained) - Ridge regression | -0.116 [-0.173, -0.018] | -0.049 [-0.109, 0.051] | -0.277 [-0.412, -0.101] |

### Does ARMOR3D benefit from nearby floats?

Local float density = other Argo profiles within +-3 days and +-2 degrees of the matchup profile (tercile edges 1.0 and 3.0; median 2.0). Pooled 50-200 m RMSE (degC).

| Density tercile | profiles | Climatology | Ridge regression | OceanEmbed (no pretraining) | OceanEmbed (pretrained) | ARMOR3D | GLORYS |
|---|---|---|---|---|---|---|---|
| few floats nearby | 392 | 1.96 [1.80, 2.11] | 1.72 [1.62, 1.83] | 1.55 [1.46, 1.63] | 1.57 [1.47, 1.64] | 0.75 [0.65, 0.83] | 1.33 [1.24, 1.46] |
| some | 1,303 | 1.68 [1.55, 1.86] | 1.59 [1.47, 1.71] | 1.47 [1.43, 1.53] | 1.46 [1.42, 1.52] | 0.65 [0.57, 0.75] | 1.11 [1.02, 1.19] |
| many floats nearby | 1,131 | 1.54 [1.38, 1.64] | 1.39 [1.27, 1.47] | 1.30 [1.24, 1.38] | 1.30 [1.26, 1.38] | 0.55 [0.51, 0.59] | 0.95 [0.91, 1.01] |

Difference-in-differences: (RMSE gap, high-density tercile) minus (gap, low one).

| Gap | high - low |
|---|---|
| ARMOR3D - OceanEmbed (pretrained) | 0.070 [-0.019, 0.126] |
| ARMOR3D - OceanEmbed (no pretraining) | 0.052 [-0.065, 0.142] |
| ARMOR3D - GLORYS | 0.186 [0.116, 0.274] |

## On the grid against GLORYS (cells with mask, GLORYS and ARMOR3D defined)

### whole domain

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.390 [1.315, 1.420] | -0.075 | - | 0.000 |
| Ridge regression | 1.154 [1.064, 1.205] | 0.057 | 0.559 | 0.310 |
| OceanEmbed (no pretraining) | 1.041 [0.999, 1.067] | 0.040 | 0.665 | 0.439 |
| OceanEmbed (pretrained) | 1.077 [1.025, 1.110] | 0.020 | 0.641 | 0.399 |
| ARMOR3D | 1.196 [1.116, 1.267] | -0.468 | 0.676 | 0.259 |

### Arabian Sea

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.317 [1.258, 1.383] | -0.242 | - | 0.000 |
| Ridge regression | 1.139 [1.061, 1.191] | -0.025 | 0.477 | 0.253 |
| OceanEmbed (no pretraining) | 1.082 [1.034, 1.115] | 0.043 | 0.559 | 0.325 |
| OceanEmbed (pretrained) | 1.112 [1.049, 1.153] | 0.042 | 0.533 | 0.288 |
| ARMOR3D | 1.185 [1.082, 1.283] | -0.505 | 0.652 | 0.191 |

### Bay of Bengal

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.526 [1.388, 1.570] | 0.229 | - | 0.000 |
| Ridge regression | 1.185 [1.052, 1.262] | 0.207 | 0.645 | 0.397 |
| OceanEmbed (no pretraining) | 0.968 [0.910, 1.020] | 0.036 | 0.768 | 0.597 |
| OceanEmbed (pretrained) | 1.020 [0.957, 1.082] | -0.019 | 0.742 | 0.553 |
| ARMOR3D | 1.215 [1.157, 1.257] | -0.411 | 0.700 | 0.366 |

RMSE by depth (whole domain, degC)

| Product | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.79 | 0.79 | 0.79 | 0.83 | 0.91 | 1.18 | 1.54 | 1.66 | 1.56 | 1.36 | 0.89 | 0.54 | 0.35 | 0.36 | 0.38 |
| Ridge regression | 0.72 | 0.72 | 0.72 | 0.76 | 0.83 | 1.00 | 1.21 | 1.31 | 1.30 | 1.20 | 0.83 | 0.53 | 0.35 | 0.35 | 0.38 |
| OceanEmbed (no pretraining) | 0.56 | 0.56 | 0.57 | 0.61 | 0.72 | 0.92 | 1.08 | 1.18 | 1.18 | 1.06 | 0.75 | 0.52 | 0.35 | 0.36 | 0.41 |
| OceanEmbed (pretrained) | 0.57 | 0.56 | 0.57 | 0.63 | 0.72 | 0.92 | 1.10 | 1.22 | 1.24 | 1.12 | 0.79 | 0.53 | 0.36 | 0.37 | 0.41 |
| ARMOR3D | 0.63 | 0.47 | 0.49 | 0.61 | 0.76 | 1.07 | 1.33 | 1.45 | 1.33 | 1.09 | 0.77 | 0.52 | 0.35 | 0.35 | 0.39 |

## On the grid against ARMOR3D (cells with mask, GLORYS and ARMOR3D defined)

### whole domain

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.403 [1.323, 1.452] | 0.393 | - | 0.000 |
| Ridge regression | 1.194 [1.138, 1.235] | 0.525 | 0.611 | 0.275 |
| OceanEmbed (no pretraining) | 1.147 [1.102, 1.180] | 0.508 | 0.653 | 0.331 |
| OceanEmbed (pretrained) | 1.181 [1.133, 1.227] | 0.487 | 0.622 | 0.291 |
| GLORYS | 1.196 [1.116, 1.267] | 0.468 | 0.676 | 0.272 |

### Arabian Sea

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.302 [1.192, 1.418] | 0.263 | - | 0.000 |
| Ridge regression | 1.155 [1.055, 1.230] | 0.480 | 0.573 | 0.214 |
| OceanEmbed (no pretraining) | 1.193 [1.125, 1.242] | 0.549 | 0.566 | 0.161 |
| OceanEmbed (pretrained) | 1.222 [1.145, 1.284] | 0.547 | 0.537 | 0.120 |
| GLORYS | 1.185 [1.082, 1.283] | 0.505 | 0.652 | 0.172 |

### Bay of Bengal

| Product | RMSE 50-200 m | bias | anomaly corr. | skill vs clim. |
|---|---|---|---|---|
| Climatology | 1.573 [1.452, 1.625] | 0.640 | - | 0.000 |
| Ridge regression | 1.262 [1.169, 1.334] | 0.619 | 0.651 | 0.356 |
| OceanEmbed (no pretraining) | 1.057 [0.993, 1.131] | 0.447 | 0.752 | 0.549 |
| OceanEmbed (pretrained) | 1.100 [1.066, 1.152] | 0.392 | 0.716 | 0.511 |
| GLORYS | 1.215 [1.157, 1.257] | 0.411 | 0.700 | 0.404 |

RMSE by depth (whole domain, degC)

| Product | 0 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 125 m | 150 m | 200 m | 300 m | 500 m | 700 m | 1000 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.87 | 0.79 | 0.77 | 0.78 | 0.83 | 1.04 | 1.50 | 1.82 | 1.66 | 1.33 | 0.80 | 0.38 | 0.24 | 0.24 | 0.22 |
| Ridge regression | 0.81 | 0.71 | 0.70 | 0.72 | 0.75 | 0.88 | 1.23 | 1.55 | 1.46 | 1.13 | 0.69 | 0.35 | 0.22 | 0.23 | 0.22 |
| OceanEmbed (no pretraining) | 0.72 | 0.57 | 0.56 | 0.62 | 0.71 | 0.92 | 1.16 | 1.49 | 1.39 | 1.07 | 0.65 | 0.35 | 0.24 | 0.25 | 0.26 |
| OceanEmbed (pretrained) | 0.73 | 0.58 | 0.57 | 0.64 | 0.75 | 0.96 | 1.18 | 1.52 | 1.45 | 1.09 | 0.68 | 0.37 | 0.25 | 0.26 | 0.26 |
| GLORYS | 0.63 | 0.47 | 0.49 | 0.61 | 0.76 | 1.07 | 1.33 | 1.45 | 1.33 | 1.09 | 0.77 | 0.52 | 0.35 | 0.35 | 0.39 |
