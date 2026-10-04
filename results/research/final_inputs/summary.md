# Input-set selection for the final model (poc_long)

**Decision: SST + sea level** (SST, sea level).

## Rule (validation year only)

a reduced input set is adopted only if its mean validation-year RMSE is lower than the full set's by more than the seed spread (the larger sample SD over seeds of the two sets); the lowest adopted set wins, otherwise all seven inputs are kept.

Criterion: validation-year (2022) RMSE of the kept checkpoint, pooled over all depths. Spread: max(SD over seeds of the full set, SD over seeds of the candidate).

## Validation (2022), the headline model trained from scratch

| Input set | Inputs | Val. RMSE per seed (°C) | Mean | SD | Mean over 50-200 m levels (unweighted) | Gain vs full (positive = better) | Seed spread | Adopted |
|---|---|---|---|---|---|---|---|---|
| All seven inputs | SST, salinity, sea level, currents, winds | 0.6996, 0.6967, 0.7003 | 0.6988 | 0.0019 | 0.9589 | reference | - | - |
| all but salinity and currents | SST, sea level, winds | 0.6897, 0.7032, 0.6991 | 0.6973 | 0.0069 | 0.9545 | +0.0015 | 0.0069 | no |
| SST + sea level | SST, sea level | 0.6874, 0.6964, 0.6924 | 0.6921 | 0.0045 | 0.9506 | +0.0068 | 0.0045 | yes |

Test periods: 2023 (365 days, 2023-01-01 .. 2023-12-31); 2024 (350 days, 2024-01-01 .. 2024-12-15); pooled (715 days, 2023-01-01 .. 2024-12-15). Reference GLORYS. Intervals: moving-block bootstrap over the days of each period (as R2), 2000 replicates, blocks of 70 days, 95 % percentile.

## Test-year scores of every candidate (reported after the decision)

RMSE over 50-200 m against GLORYS, mean ± SD over seeds.

| Input set | Seeds | Period | Whole domain | Arabian Sea | Bay of Bengal | Skill vs climatology | RMSE all depths |
|---|---|---|---|---|---|---|---|
| All seven inputs | 3 | 2023 | 0.975 ± 0.007 | 0.992 ± 0.013 | 0.941 ± 0.008 | 0.588 | 0.704 |
| All seven inputs | 3 | 2024 | 1.004 ± 0.014 | 1.043 ± 0.024 | 0.934 ± 0.006 | 0.481 | 0.736 |
| All seven inputs | 3 | pooled | 0.989 ± 0.010 | 1.018 ± 0.018 | 0.938 ± 0.007 | 0.540 | 0.720 |
| all but salinity and currents | 3 | 2023 | 0.970 ± 0.008 | 0.986 ± 0.010 | 0.940 ± 0.003 | 0.592 | 0.702 |
| all but salinity and currents | 3 | 2024 | 1.010 ± 0.016 | 1.043 ± 0.021 | 0.955 ± 0.009 | 0.474 | 0.741 |
| all but salinity and currents | 3 | pooled | 0.990 ± 0.005 | 1.015 ± 0.006 | 0.947 ± 0.004 | 0.539 | 0.721 |
| SST + sea level | 3 | 2023 | 0.970 ± 0.009 | 0.986 ± 0.008 | 0.939 ± 0.012 | 0.592 | 0.705 |
| SST + sea level | 3 | 2024 | 1.002 ± 0.003 | 1.035 ± 0.001 | 0.946 ± 0.009 | 0.483 | 0.736 |
| SST + sea level | 3 | pooled | 0.986 ± 0.003 | 1.010 ± 0.004 | 0.943 ± 0.002 | 0.543 | 0.720 |

Candidate minus full set, RMSE over 50-200 m (negative = candidate better, `*` established):

| Input set | Period | whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|---|
| all but salinity and currents | 2023 | -0.004 [-0.010, +0.001] | -0.006 [-0.012, +0.004] | -0.001 [-0.021, +0.013] |
| all but salinity and currents | 2024 | +0.007 [-0.001, +0.010] | -0.001 [-0.016, +0.010] | +0.021 [-0.004, +0.040] |
| all but salinity and currents | pooled | +0.001 [-0.004, +0.008] | -0.003 [-0.013, +0.004] | +0.010 [-0.006, +0.038] |
| SST + sea level | 2023 | -0.004 [-0.012, +0.005] | -0.007 [-0.013, +0.001] | -0.002 [-0.020, +0.015] |
| SST + sea level | 2024 | -0.002 [-0.007, +0.006] | -0.009 [-0.019, +0.004] | +0.012 [-0.005, +0.030] |
| SST + sea level | pooled | -0.003 [-0.007, +0.004] | -0.008 [-0.015, -0.001] | +0.005 [-0.006, +0.027] |

The decision uses validation numbers only. The test-year scores below are reported for every candidate after the fact and played no part in it.
