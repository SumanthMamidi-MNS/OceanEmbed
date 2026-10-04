# R3 - which surface variable matters where, and does history help (poc)

Test period 2024-01-01 .. 2024-12-15 (350 days), reference GLORYS. Intervals: moving-block bootstrap over test days (overlapping blocks, no wrap), 2000 replicates, blocks of 39 days, 95 % percentile interval of the paired difference. `*` marks an *established* difference (paired interval excludes zero AND the per-seed RMSE ranges of the two methods do not overlap). Delta = experiment minus the full-input model of the same seeds; positive = worse.

## OceanEmbed (no pretraining): retrain without a variable group

Full-input reference seeds: [0, 1, 2, 3, 4] (R1).

| Experiment | Seeds | RMSE 50-200 m | Full-input RMSE | Delta whole domain [95 % CI] | Delta Arabian Sea | Delta Bay of Bengal | Whole domain verdict |
|---|---|---|---|---|---|---|---|
| without SST | 2 | 1.068 ± 0.003 | 1.066 ± 0.011 | +0.002 [-0.011, +0.011] | -0.010 [-0.025, +0.001] | +0.024 [-0.010, +0.060] | not distinguishable from zero |
| without salinity | 2 | 1.066 ± 0.018 | 1.066 ± 0.011 | +0.000 [-0.011, +0.010] | +0.004 [-0.010, +0.018] | -0.008 [-0.020, +0.007] | not distinguishable from zero |
| without sea level | 2 | 1.108 ± 0.003 | 1.066 ± 0.011 | +0.042 [+0.022, +0.059] * | +0.017 [+0.002, +0.035] | +0.087 [+0.043, +0.122] * | established |
| without currents | 2 | 1.068 ± 0.022 | 1.066 ± 0.011 | +0.002 [-0.007, +0.014] | +0.001 [-0.013, +0.016] | +0.002 [-0.012, +0.022] | not distinguishable from zero |
| without winds | 2 | 1.074 ± 0.000 | 1.066 ± 0.011 | +0.008 [-0.006, +0.024] | -0.006 [-0.014, +0.003] | +0.037 [+0.003, +0.077] * | not distinguishable from zero |
| SST only | 2 | 1.344 ± 0.037 | 1.066 ± 0.011 | +0.278 [+0.222, +0.338] * | +0.153 [+0.105, +0.215] * | +0.503 [+0.374, +0.601] * | established |
| SST + sea level | 2 | 1.060 ± 0.000 | 1.066 ± 0.011 | -0.006 [-0.018, -0.002] | -0.024 [-0.044, -0.010] | +0.028 [+0.007, +0.046] | interval excludes 0, seed ranges overlap (not established) |
| all but salinity and currents | 2 | 1.039 ± 0.012 | 1.066 ± 0.011 | -0.027 [-0.038, -0.016] * | -0.043 [-0.063, -0.026] * | +0.004 [-0.012, +0.020] | established |

### OceanEmbed (no pretraining): delta RMSE by depth, whole domain

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.105* | -0.003 | +0.045* | -0.009 | -0.003 | +0.005 | +0.025* | -0.008 |
| 5 m | +0.094* | -0.004 | +0.044* | -0.006 | +0.000 | +0.008 | +0.020 | -0.010 |
| 10 m | +0.095* | -0.006 | +0.048* | -0.003 | +0.001 | +0.013 | +0.022* | -0.005 |
| 20 m | +0.070* | +0.000 | +0.047* | +0.001 | -0.013 | +0.015 | +0.011* | -0.013 |
| 30 m | +0.044* | +0.010 | +0.057* | +0.003 | -0.009 | +0.043* | +0.009 | -0.010 |
| 50 m | +0.007 | +0.011 | +0.055* | -0.001 | -0.004 | +0.138* | +0.000 | -0.012* |
| 75 m | +0.001 | +0.010 | +0.061* | +0.005 | +0.016 | +0.322* | +0.008 | -0.012 |
| 100 m | -0.009 | -0.007 | +0.035* | -0.001 | +0.003 | +0.389* | -0.019* | -0.039* |
| 125 m | +0.003 | -0.001 | +0.043* | +0.004 | +0.012 | +0.359* | -0.017 | -0.036* |
| 150 m | +0.007 | -0.009 | +0.034 | -0.001 | +0.016* | +0.262* | -0.011 | -0.041* |
| 200 m | +0.006 | -0.001 | +0.021 | +0.008* | +0.008 | +0.123* | +0.008 | -0.016* |
| 300 m | -0.003 | +0.000 | +0.001 | +0.013* | -0.002 | +0.022* | -0.005 | -0.003 |
| 500 m | -0.004* | +0.007* | +0.004 | +0.015* | +0.002 | +0.002 | +0.000 | +0.002 |
| 700 m | -0.002 | +0.001 | +0.001 | +0.010* | -0.002 | -0.002 | -0.004* | +0.000 |
| 1000 m | -0.014* | -0.004* | -0.012* | +0.005* | -0.008* | -0.022* | -0.014* | -0.003 |
| 50-200 m pooled | +0.002 | +0.000 | +0.042* | +0.002 | +0.008 | +0.278* | -0.006* | -0.027* |
| 500-1000 m pooled | -0.007* | +0.001 | -0.003 | +0.010* | -0.003 | -0.008* | -0.006* | -0.000 |

### OceanEmbed (no pretraining): delta RMSE by depth, Arabian Sea

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.091* | -0.008* | +0.050* | -0.006 | -0.003 | -0.013 | +0.017 | -0.010 |
| 5 m | +0.083* | -0.007* | +0.053* | -0.002 | +0.003 | -0.004 | +0.013 | -0.011 |
| 10 m | +0.085* | -0.011* | +0.054* | -0.001 | +0.003 | -0.002 | +0.014 | -0.007 |
| 20 m | +0.062* | -0.002 | +0.053* | +0.005 | -0.012 | +0.009 | +0.004 | -0.017 |
| 30 m | +0.040* | +0.008 | +0.063* | +0.008 | -0.009 | +0.040 | +0.005 | -0.016 |
| 50 m | -0.008 | +0.010 | +0.054* | +0.005 | -0.007 | +0.092* | +0.002 | -0.019* |
| 75 m | -0.009* | +0.007 | +0.056* | +0.016 | -0.003 | +0.164* | +0.010 | -0.015* |
| 100 m | -0.026* | -0.015 | +0.000 | +0.002 | -0.032* | +0.203* | -0.054* | -0.062* |
| 125 m | -0.010 | +0.013 | +0.000 | -0.005 | -0.005 | +0.207* | -0.056* | -0.063* |
| 150 m | +0.001 | +0.008 | -0.004 | -0.013 | +0.013* | +0.151* | -0.033 | -0.063* |
| 200 m | -0.005 | +0.005 | -0.003 | +0.007 | +0.003 | +0.067* | -0.001 | -0.027* |
| 300 m | -0.010* | +0.001 | -0.006 | +0.019* | -0.003 | +0.006 | -0.009* | -0.006 |
| 500 m | -0.005* | +0.009* | +0.003 | +0.020* | +0.003 | -0.004* | +0.003 | +0.005* |
| 700 m | -0.004 | -0.000 | +0.001 | +0.012* | -0.003 | -0.006 | -0.004 | +0.001 |
| 1000 m | -0.017* | -0.006 | -0.016* | +0.006* | -0.010 | -0.029* | -0.016* | -0.003 |
| 50-200 m pooled | -0.010 | +0.004 | +0.017* | +0.001 | -0.006 | +0.153* | -0.024* | -0.043* |
| 500-1000 m pooled | -0.009* | +0.001 | -0.004 | +0.013* | -0.003 | -0.013* | -0.006* | +0.001 |

### OceanEmbed (no pretraining): delta RMSE by depth, Bay of Bengal

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.137* | +0.015 | +0.043* | +0.000 | -0.004 | +0.037 | +0.050* | -0.003 |
| 5 m | +0.120* | +0.009 | +0.038* | +0.001 | -0.004 | +0.034 | +0.046* | -0.004 |
| 10 m | +0.119* | +0.010 | +0.038* | +0.003 | -0.005 | +0.041* | +0.045* | -0.002 |
| 20 m | +0.094* | +0.007 | +0.036* | -0.009 | -0.017 | +0.028 | +0.030* | -0.007 |
| 30 m | +0.056* | +0.016 | +0.045* | -0.011 | -0.011* | +0.056* | +0.013 | -0.004 |
| 50 m | +0.035* | +0.012* | +0.061* | -0.014* | -0.001 | +0.242* | -0.010 | -0.003 |
| 75 m | +0.020 | +0.016 | +0.066* | -0.019 | +0.052* | +0.572* | +0.007 | -0.008 |
| 100 m | +0.023 | +0.011 | +0.091* | -0.007 | +0.070* | +0.696* | +0.045* | +0.006 |
| 125 m | +0.027 | -0.030* | +0.121* | +0.020 | +0.045 | +0.636* | +0.060* | +0.018 |
| 150 m | +0.019 | -0.047* | +0.107* | +0.022 | +0.019 | +0.474* | +0.032 | +0.002 |
| 200 m | +0.030* | -0.016 | +0.073* | +0.011 | +0.016 | +0.238* | +0.028 | +0.007 |
| 300 m | +0.020* | -0.001 | +0.022* | -0.002 | +0.003 | +0.071* | +0.007 | +0.008 |
| 500 m | +0.000 | -0.001 | +0.007 | +0.002 | -0.002 | +0.022* | -0.005 | -0.004 |
| 700 m | +0.006 | +0.002 | +0.002 | +0.006 | -0.001 | +0.010 | -0.003 | -0.001 |
| 1000 m | +0.000 | +0.003 | +0.003 | +0.003 | -0.001 | +0.003 | -0.006 | -0.002 |
| 50-200 m pooled | +0.024 | -0.008 | +0.087* | +0.002 | +0.037* | +0.503* | +0.028* | +0.004 |
| 500-1000 m pooled | +0.002 | +0.001 | +0.004 | +0.003 | -0.001 | +0.012* | -0.005 | -0.003 |

### OceanEmbed (no pretraining): temporal context

| History | Region | RMSE 50-200 m | Delta vs k=1 [95 % CI] | Verdict | Skill 500-1000 m | k=1 skill | Skill change |
|---|---|---|---|---|---|---|---|
| history k = 3 | whole domain | 1.061 ± 0.004 | -0.005 [-0.016, +0.010] | not distinguishable from zero | -0.084 ± 0.043 | -0.066 ± 0.006 | -0.018 |
| history k = 3 | Arabian Sea | 1.105 ± 0.010 | -0.002 [-0.019, +0.016] | not distinguishable from zero | -0.111 ± 0.035 | -0.094 ± 0.007 | -0.016 |
| history k = 3 | Bay of Bengal | 0.985 ± 0.007 | -0.010 [-0.030, +0.011] | not distinguishable from zero | 0.073 ± 0.075 | 0.091 ± 0.016 | -0.018 |
| history k = 7 | whole domain | 1.053 ± 0.005 | -0.013 [-0.032, +0.001] | not distinguishable from zero | -0.085 ± 0.080 | -0.066 ± 0.006 | -0.019 |
| history k = 7 | Arabian Sea | 1.096 ± 0.018 | -0.012 [-0.033, +0.001] | not distinguishable from zero | -0.111 ± 0.098 | -0.094 ± 0.007 | -0.016 |
| history k = 7 | Bay of Bengal | 0.974 ± 0.025 | -0.020 [-0.050, +0.010] | not distinguishable from zero | 0.068 ± 0.001 | 0.091 ± 0.016 | -0.023 |

RMSE by depth, whole domain (seed mean; delta vs k = 1):

| Depth | k = 1 | history k = 3 | history k = 7 |
|---|---|---|---|
| 0 m | 0.551 | 0.574 (+0.023) | 0.574 (+0.023) |
| 5 m | 0.555 | 0.574 (+0.019) | 0.583 (+0.027) |
| 10 m | 0.560 | 0.584 (+0.024) | 0.580 (+0.020) |
| 20 m | 0.616 | 0.631 (+0.015) | 0.631 (+0.015) |
| 30 m | 0.716 | 0.726 (+0.009) | 0.725 (+0.009) |
| 50 m | 0.923 | 0.926 (+0.003) | 0.913 (-0.011) |
| 75 m | 1.105 | 1.105 (+0.000) | 1.098 (-0.006) |
| 100 m | 1.219 | 1.205 (-0.015) | 1.199 (-0.020) |
| 125 m | 1.210 | 1.203 (-0.006) | 1.201 (-0.009) |
| 150 m | 1.100 | 1.089 (-0.011) | 1.074 (-0.026*) |
| 200 m | 0.772 | 0.777 (+0.005) | 0.764 (-0.008) |
| 300 m | 0.524 | 0.538 (+0.014*) | 0.530 (+0.007*) |
| 500 m | 0.353 | 0.367 (+0.014*) | 0.361 (+0.008*) |
| 700 m | 0.361 | 0.360 (-0.000) | 0.363 (+0.002) |
| 1000 m | 0.408 | 0.404 (-0.004*) | 0.408 (+0.000) |

## Per-pixel MLP: retrain without a variable group

Full-input reference seeds: [0, 1, 2] (R1).

| Experiment | Seeds | RMSE 50-200 m | Full-input RMSE | Delta whole domain [95 % CI] | Delta Arabian Sea | Delta Bay of Bengal | Whole domain verdict |
|---|---|---|---|---|---|---|---|
| without SST | 3 | 1.102 ± 0.002 | 1.094 ± 0.005 | +0.008 [-0.002, +0.018] | +0.011 [+0.002, +0.021] | +0.002 [-0.011, +0.017] | not distinguishable from zero |
| without salinity | 3 | 1.087 ± 0.004 | 1.094 ± 0.005 | -0.008 [-0.017, +0.001] | -0.004 [-0.014, +0.008] | -0.014 [-0.032, -0.006] | not distinguishable from zero |
| without sea level | 3 | 1.361 ± 0.006 | 1.094 ± 0.005 | +0.267 [+0.227, +0.286] * | +0.178 [+0.156, +0.208] * | +0.417 [+0.329, +0.443] * | established |
| without currents | 3 | 1.104 ± 0.004 | 1.094 ± 0.005 | +0.009 [+0.001, +0.015] * | +0.003 [-0.003, +0.009] | +0.020 [+0.001, +0.034] | established |
| without winds | 3 | 1.099 ± 0.006 | 1.094 ± 0.005 | +0.005 [-0.002, +0.011] | +0.002 [-0.004, +0.013] | +0.010 [-0.005, +0.016] | not distinguishable from zero |
| SST only | 3 | 1.385 ± 0.005 | 1.094 ± 0.005 | +0.290 [+0.239, +0.312] * | +0.194 [+0.174, +0.233] * | +0.454 [+0.332, +0.472] * | established |
| SST + sea level | 3 | 1.088 ± 0.003 | 1.094 ± 0.005 | -0.006 [-0.013, -0.001] | -0.009 [-0.016, +0.001] | -0.001 [-0.020, +0.006] | interval excludes 0, seed ranges overlap (not established) |
| all but salinity and currents | 3 | 1.090 ± 0.005 | 1.094 ± 0.005 | -0.005 [-0.015, +0.004] | -0.005 [-0.015, +0.007] | -0.004 [-0.022, +0.003] | not distinguishable from zero |

### Per-pixel MLP: delta RMSE by depth, whole domain

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.188* | -0.003 | +0.012* | -0.004* | +0.007* | +0.015* | -0.005* | -0.008* |
| 5 m | +0.184* | -0.002 | +0.012* | -0.003* | +0.010* | +0.020* | +0.000 | -0.007* |
| 10 m | +0.185* | -0.002 | +0.014* | -0.004* | +0.010* | +0.025* | +0.003 | -0.008* |
| 20 m | +0.155* | -0.001 | +0.027* | +0.000 | +0.010* | +0.041* | +0.006 | -0.009* |
| 30 m | +0.116* | -0.002 | +0.053* | +0.003 | +0.014* | +0.070* | +0.004 | -0.010* |
| 50 m | +0.052* | -0.013 | +0.137* | +0.012* | +0.012* | +0.156* | -0.010* | -0.012* |
| 75 m | +0.011 | -0.006 | +0.277* | +0.022* | +0.014* | +0.307* | -0.009* | -0.006 |
| 100 m | -0.008 | -0.006 | +0.361* | +0.007 | -0.000 | +0.397* | -0.009 | -0.006 |
| 125 m | -0.008 | -0.007 | +0.358* | +0.001 | -0.001 | +0.384* | -0.006 | -0.001 |
| 150 m | +0.001 | -0.010 | +0.276* | +0.004 | +0.003 | +0.293* | -0.004 | -0.002 |
| 200 m | +0.003 | -0.007* | +0.121* | +0.008* | +0.001 | +0.126* | +0.004 | -0.001 |
| 300 m | +0.000 | -0.003 | +0.027* | +0.002 | -0.000 | +0.024* | -0.001 | -0.002 |
| 500 m | -0.002* | -0.003* | +0.007* | +0.002* | -0.001* | +0.001 | -0.004* | -0.004* |
| 700 m | -0.001 | -0.004* | +0.004 | +0.001 | -0.001 | +0.000 | -0.004* | -0.003* |
| 1000 m | -0.003* | -0.002* | +0.000 | -0.001 | -0.004* | -0.004 | -0.002* | -0.001 |
| 50-200 m pooled | +0.008 | -0.008 | +0.267* | +0.009* | +0.005 | +0.290* | -0.006* | -0.005 |
| 500-1000 m pooled | -0.002* | -0.003* | +0.004 | +0.000 | -0.002* | -0.001 | -0.003* | -0.003* |

### Per-pixel MLP: delta RMSE by depth, Arabian Sea

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.219* | -0.002 | +0.014* | -0.002 | +0.009* | +0.024* | -0.006 | -0.010* |
| 5 m | +0.213* | -0.002 | +0.015* | -0.002 | +0.014* | +0.030* | -0.002 | -0.010* |
| 10 m | +0.210* | -0.003 | +0.019* | -0.002 | +0.013* | +0.034* | +0.002 | -0.012* |
| 20 m | +0.178* | -0.005 | +0.036* | +0.001* | +0.013* | +0.055* | +0.004 | -0.012* |
| 30 m | +0.141* | -0.005 | +0.064* | +0.004 | +0.016* | +0.090* | +0.003 | -0.012* |
| 50 m | +0.079* | -0.010 | +0.103* | +0.010* | +0.016* | +0.136* | -0.004 | -0.011* |
| 75 m | +0.027* | +0.000 | +0.170* | +0.010* | +0.010 | +0.188* | -0.014* | -0.006 |
| 100 m | -0.023 | -0.007 | +0.229* | -0.001 | -0.001 | +0.234* | -0.020* | -0.010 |
| 125 m | -0.019 | -0.005 | +0.237* | -0.003 | -0.004 | +0.247* | -0.011 | -0.004 |
| 150 m | +0.004 | -0.003 | +0.198* | -0.000 | -0.003 | +0.220* | -0.003 | +0.001 |
| 200 m | +0.005* | -0.003 | +0.093* | +0.005 | -0.002 | +0.105* | +0.006 | +0.001 |
| 300 m | +0.001 | -0.002 | +0.019* | +0.000 | -0.001 | +0.019* | -0.000 | -0.002 |
| 500 m | -0.002 | -0.003* | +0.003 | +0.001 | -0.002* | -0.003 | -0.003* | -0.004* |
| 700 m | -0.001 | -0.003* | +0.003 | -0.000 | -0.002* | -0.000 | -0.003* | -0.003* |
| 1000 m | -0.005* | -0.001 | +0.001 | -0.001 | -0.005* | -0.003 | -0.002 | +0.000 |
| 50-200 m pooled | +0.011* | -0.004 | +0.178* | +0.003 | +0.002 | +0.194* | -0.009 | -0.005 |
| 500-1000 m pooled | -0.003* | -0.002* | +0.002 | -0.000 | -0.003* | -0.002 | -0.003* | -0.002* |

### Per-pixel MLP: delta RMSE by depth, Bay of Bengal

`*` = the paired bootstrap interval excludes zero.

| Depth | without SST | without salinity | without sea level | without currents | without winds | SST only | SST + sea level | all but salinity and currents |
|---|---|---|---|---|---|---|---|---|
| 0 m | +0.144* | -0.001 | +0.004 | -0.005* | +0.005* | -0.005 | -0.005 | -0.001 |
| 5 m | +0.141* | +0.001 | +0.005 | -0.004* | +0.008* | -0.001 | +0.004* | +0.002 |
| 10 m | +0.144* | +0.003 | +0.004 | -0.004* | +0.011* | +0.002 | +0.005* | +0.003 |
| 20 m | +0.116* | +0.010* | +0.010* | +0.001 | +0.010* | +0.010 | +0.010* | +0.002 |
| 30 m | +0.065* | +0.005 | +0.036* | +0.004 | +0.012* | +0.030* | +0.007* | -0.002 |
| 50 m | -0.003 | -0.021* | +0.204* | +0.018* | +0.005 | +0.203* | -0.023 | -0.013 |
| 75 m | -0.011 | -0.015* | +0.431* | +0.040* | +0.019* | +0.479* | -0.003 | -0.005 |
| 100 m | +0.019 | -0.005 | +0.566* | +0.021 | +0.002 | +0.645* | +0.010 | -0.001 |
| 125 m | +0.013 | -0.012 | +0.577* | +0.010 | +0.006 | +0.627* | +0.007 | +0.003 |
| 150 m | -0.005 | -0.024* | +0.425* | +0.013 | +0.016* | +0.432* | -0.005 | -0.007 |
| 200 m | -0.003 | -0.016* | +0.175* | +0.015* | +0.009* | +0.171* | +0.000 | -0.005 |
| 300 m | -0.003* | -0.007 | +0.046* | +0.007* | +0.002 | +0.040* | -0.005 | -0.004 |
| 500 m | -0.005 | -0.005* | +0.018* | +0.005* | +0.000 | +0.012 | -0.005* | -0.005* |
| 700 m | -0.001 | -0.005* | +0.009* | +0.004 | +0.003 | +0.001 | -0.005* | -0.002 |
| 1000 m | +0.006* | -0.004 | -0.002 | -0.001 | +0.001 | -0.008* | -0.004* | -0.005* |
| 50-200 m pooled | +0.002 | -0.014* | +0.417* | +0.020* | +0.010 | +0.454* | -0.001 | -0.004 |
| 500-1000 m pooled | -0.000 | -0.005* | +0.008* | +0.003* | +0.001 | +0.002 | -0.005* | -0.004* |

### Per-pixel MLP: temporal context

| History | Region | RMSE 50-200 m | Delta vs k=1 [95 % CI] | Verdict | Skill 500-1000 m | k=1 skill | Skill change |
|---|---|---|---|---|---|---|---|
| history k = 3 | whole domain | 1.106 ± 0.007 | +0.011 [-0.001, +0.022] | not distinguishable from zero | -0.008 ± 0.010 | -0.025 ± 0.010 | +0.017 |
| history k = 3 | Arabian Sea | 1.092 ± 0.010 | +0.009 [+0.001, +0.016] | interval excludes 0, seed ranges overlap (not established) | -0.015 ± 0.007 | -0.029 ± 0.007 | +0.014 |
| history k = 3 | Bay of Bengal | 1.135 ± 0.005 | +0.015 [-0.006, +0.035] | not distinguishable from zero | 0.027 ± 0.021 | -0.003 ± 0.031 | +0.030 |
| history k = 7 | whole domain | 1.086 ± 0.009 | -0.009 [-0.017, +0.000] | not distinguishable from zero | -0.048 ± 0.023 | -0.025 ± 0.010 | -0.023 |
| history k = 7 | Arabian Sea | 1.075 ± 0.008 | -0.008 [-0.022, +0.003] | not distinguishable from zero | -0.055 ± 0.026 | -0.029 ± 0.007 | -0.025 |
| history k = 7 | Bay of Bengal | 1.111 ± 0.010 | -0.010 [-0.016, +0.003] | not distinguishable from zero | -0.014 ± 0.013 | -0.003 ± 0.031 | -0.011 |

RMSE by depth, whole domain (seed mean; delta vs k = 1):

| Depth | k = 1 | history k = 3 | history k = 7 |
|---|---|---|---|
| 0 m | 0.506 | 0.495 (-0.012*) | 0.482 (-0.025*) |
| 5 m | 0.515 | 0.502 (-0.013*) | 0.487 (-0.028*) |
| 10 m | 0.508 | 0.498 (-0.010*) | 0.480 (-0.028*) |
| 20 m | 0.566 | 0.560 (-0.005*) | 0.543 (-0.023*) |
| 30 m | 0.672 | 0.668 (-0.004) | 0.654 (-0.019*) |
| 50 m | 0.922 | 0.918 (-0.004) | 0.908 (-0.014*) |
| 75 m | 1.191 | 1.200 (+0.008) | 1.174 (-0.018) |
| 100 m | 1.274 | 1.283 (+0.009) | 1.263 (-0.011) |
| 125 m | 1.216 | 1.242 (+0.026*) | 1.213 (-0.003) |
| 150 m | 1.101 | 1.123 (+0.022*) | 1.099 (-0.002) |
| 200 m | 0.781 | 0.782 (+0.001) | 0.778 (-0.003) |
| 300 m | 0.526 | 0.519 (-0.007*) | 0.528 (+0.003*) |
| 500 m | 0.354 | 0.350 (-0.004*) | 0.358 (+0.004*) |
| 700 m | 0.358 | 0.355 (-0.003*) | 0.363 (+0.005*) |
| 1000 m | 0.389 | 0.387 (-0.002) | 0.392 (+0.003) |

## Does history change the Bay of Bengal / Arabian Sea contrast?

Advantage of the Transformer over the per-pixel MLP with the same inputs: RMSE(MLP) minus RMSE(Transformer), pooled 50-200 m (positive = the Transformer is better). Seed means of every finished seed of each model.

| History | Whole domain | Arabian Sea | Bay of Bengal | Change of the advantage vs k = 1 |
|---|---|---|---|---|
| k = 1 | +0.042 [-0.003, +0.075] | -0.006 [-0.063, +0.048] | +0.132 [+0.046, +0.199] | reference |
| k = 3 | +0.044 [-0.008, +0.077] | -0.013 [-0.077, +0.042] | +0.151 [+0.054, +0.222] | whole domain +0.002 [-0.014, +0.012]; Arabian Sea -0.006 [-0.029, +0.010]; Bay of Bengal +0.018 [+0.000, +0.032] |
| k = 7 | +0.033 [-0.001, +0.058] | -0.021 [-0.070, +0.024] | +0.136 [+0.069, +0.192] | whole domain -0.009 [-0.023, +0.007]; Arabian Sea -0.014 [-0.033, +0.006]; Bay of Bengal +0.004 [-0.015, +0.030] |

## Permutation importance

RMSE increase when one variable group is permuted across the test days within the same calendar month (within the calendar month (random derangement of each month's days; all channels of a group from the same donor day), averaged over repeats); trained R1 models, no retraining.

### OceanEmbed (no pretraining) (seeds [0, 1], 3 repeats)

| Permuted group | Whole domain 50-200 m | Arabian Sea | Bay of Bengal | Whole domain, seed mean ± SD |
|---|---|---|---|---|
| SST | +0.002 [+0.001, +0.004] | +0.003 [+0.001, +0.006] | +0.001 [-0.003, +0.005] | 0.002 ± 0.001 |
| salinity | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.002] | -0.000 ± 0.000 |
| sea level | +0.106 [+0.083, +0.140] | +0.088 [+0.057, +0.132] | +0.142 [+0.119, +0.180] | 0.106 ± 0.003 |
| currents | +0.026 [+0.020, +0.035] | +0.023 [+0.015, +0.036] | +0.032 [+0.023, +0.043] | 0.026 ± 0.000 |
| winds | +0.001 [-0.000, +0.002] | +0.001 [-0.001, +0.002] | +0.002 [+0.000, +0.003] | 0.001 ± 0.000 |

### Per-pixel MLP (seeds [0, 1, 2], 3 repeats)

| Permuted group | Whole domain 50-200 m | Arabian Sea | Bay of Bengal | Whole domain, seed mean ± SD |
|---|---|---|---|---|
| SST | +0.010 [+0.007, +0.012] | +0.011 [+0.008, +0.014] | +0.007 [+0.003, +0.011] | 0.010 ± 0.000 |
| salinity | +0.001 [-0.002, +0.003] | +0.001 [-0.003, +0.005] | +0.001 [-0.002, +0.002] | 0.001 ± 0.000 |
| sea level | +0.176 [+0.145, +0.220] | +0.151 [+0.108, +0.212] | +0.220 [+0.188, +0.266] | 0.176 ± 0.007 |
| currents | +0.006 [+0.004, +0.008] | +0.007 [+0.004, +0.009] | +0.005 [+0.002, +0.008] | 0.006 ± 0.001 |
| winds | +0.003 [+0.002, +0.006] | +0.003 [+0.001, +0.005] | +0.005 [+0.001, +0.008] | 0.003 ± 0.001 |

## Retrain-without vs permutation ranking (pooled 50-200 m)

Rank 1 = the largest RMSE increase. Retraining lets the network re-learn from the remaining inputs, so a variable that is largely redundant (its information also sits in other inputs) costs little to remove; permutation breaks the learned link without retraining, so a redundant variable the network relies on still costs a lot. Rank differences of two or more are listed.

### OceanEmbed (no pretraining)

| Region | Group | Delta (retrain) | Rank | Delta (permute) | Rank |  |
|---|---|---|---|---|---|---|
| whole domain | SST | +0.002 | 4 | +0.002 | 3 |  |
| whole domain | salinity | +0.000 | 5 | -0.000 | 5 |  |
| whole domain | sea level | +0.042 | 1 | +0.106 | 1 |  |
| whole domain | currents | +0.002 | 3 | +0.026 | 2 |  |
| whole domain | winds | +0.008 | 2 | +0.001 | 4 | differs |
| whole domain | Spearman rank correlation |  |  | 0.70 |  |  |
| Arabian Sea | SST | -0.010 | 5 | +0.003 | 3 | differs |
| Arabian Sea | salinity | +0.004 | 2 | -0.000 | 5 | differs |
| Arabian Sea | sea level | +0.017 | 1 | +0.088 | 1 |  |
| Arabian Sea | currents | +0.001 | 3 | +0.023 | 2 |  |
| Arabian Sea | winds | -0.006 | 4 | +0.001 | 4 |  |
| Arabian Sea | Spearman rank correlation |  |  | 0.30 |  |  |
| Bay of Bengal | SST | +0.024 | 3 | +0.001 | 4 |  |
| Bay of Bengal | salinity | -0.008 | 5 | +0.000 | 5 |  |
| Bay of Bengal | sea level | +0.087 | 1 | +0.142 | 1 |  |
| Bay of Bengal | currents | +0.002 | 4 | +0.032 | 2 | differs |
| Bay of Bengal | winds | +0.037 | 2 | +0.002 | 3 |  |
| Bay of Bengal | Spearman rank correlation |  |  | 0.70 |  |  |

### Per-pixel MLP

| Region | Group | Delta (retrain) | Rank | Delta (permute) | Rank |  |
|---|---|---|---|---|---|---|
| whole domain | SST | +0.008 | 3 | +0.010 | 2 |  |
| whole domain | salinity | -0.008 | 5 | +0.001 | 5 |  |
| whole domain | sea level | +0.267 | 1 | +0.176 | 1 |  |
| whole domain | currents | +0.009 | 2 | +0.006 | 3 |  |
| whole domain | winds | +0.005 | 4 | +0.003 | 4 |  |
| whole domain | Spearman rank correlation |  |  | 0.90 |  |  |
| Arabian Sea | SST | +0.011 | 2 | +0.011 | 2 |  |
| Arabian Sea | salinity | -0.004 | 5 | +0.001 | 5 |  |
| Arabian Sea | sea level | +0.178 | 1 | +0.151 | 1 |  |
| Arabian Sea | currents | +0.003 | 3 | +0.007 | 3 |  |
| Arabian Sea | winds | +0.002 | 4 | +0.003 | 4 |  |
| Arabian Sea | Spearman rank correlation |  |  | 1.00 |  |  |
| Bay of Bengal | SST | +0.002 | 4 | +0.007 | 2 | differs |
| Bay of Bengal | salinity | -0.014 | 5 | +0.001 | 5 |  |
| Bay of Bengal | sea level | +0.417 | 1 | +0.220 | 1 |  |
| Bay of Bengal | currents | +0.020 | 2 | +0.005 | 3 |  |
| Bay of Bengal | winds | +0.010 | 3 | +0.005 | 4 |  |
| Bay of Bengal | Spearman rank correlation |  |  | 0.70 |  |  |

Figures: `figures/ablation_heatmap_<model>.png`, `figures/permutation_by_depth.png`, `figures/history_rmse_by_depth.png`. Numbers: `summary.json`.
