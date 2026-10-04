# OceanEmbed backend reference

Everything that is not the browser UI: data download and harmonisation, models, training, evaluation,
the run folder, the CLI and the read-only data API. Written against the code in `src/oceanembed/`;
where it names a number it says where the number comes from.

Contents: [1 At a glance](#1-at-a-glance) | [2 Diagram](#2-diagram) | [3 Module map](#3-module-map) |
[4 Data layer](#4-data-layer) | [5 Model](#5-model) | [6 Training](#6-training) | [7 Evaluation](#7-evaluation) |
[8 Outputs](#8-outputs) | [9 CLI](#9-cli) | [10 Data API](#10-data-api) | [11 Configuration](#11-configuration) |
[12 Testing and quality](#12-testing-and-quality) | [13 Measured results](#13-measured-results) |
[14 Known limits and gotchas](#14-known-limits-and-gotchas)

## 1. At a glance

| | |
|---|---|
| **What it does** | Turns daily satellite surface fields (seven are harmonised; the released model uses two, SST and sea level anomaly) into temperature at 15 depths (0-1000 m) on a 0.25 deg grid of the North Indian Ocean, then scores itself against GLORYS and Argo floats. A live mode reconstructs the newest day from near-real-time inputs. |
| **Stack** | Python >= 3.12, PyTorch (CUDA wheel installed first, torch unpinned), xarray / Zarr v3 / NetCDF4, Typer CLI, pydantic v2 config, FastAPI + uvicorn. No database: files on disk. |
| **Entry points** | `start.bat` at the project root (one click: environment, dashboard build, server, browser); `oceanembed <command> --config <yaml>` (`[project.scripts]` -> `oceanembed.cli:app`), `oceanembed run-all`, `oceanembed live update`, `oceanembed serve` (data API + built web UI). |
| **Data** | `data/raw/` (real), `data/raw_nrt/` (live), `data/raw_synthetic/`, `data/processed/<run>.zarr` + `<run>_stats.nc`. Git-ignored. Override root with `OCEANEMBED_DATA_ROOT`. |
| **Outputs** | `outputs/<run>/` (checkpoints, logs, NetCDF products, metrics, embeddings, figures, report), git-ignored; override with `OCEANEMBED_OUTPUTS_ROOT`. Tracked copies of what matters: `results/` (metrics, report, research summaries; `export-results`) and `models/final/` (released weights; `export-weights`). |
| **Grid** | 100 lat x 240 lon x 15 depths, 0.25 deg, daily; 5-30 N, 45-105 E ([`grid.py`](../src/oceanembed/grid.py)). |
| **Period (main run `final`)** | 2011-01-01 .. 2024-12-15 = 5 098 days; train 2011-2021 (4 018 d), val 2022 (365 d), test 2023-01-01 .. 2024-12-15 (715 d, scored per year and together). The first real run, `poc`, covers 2018-2024 (2 541 days, test year 2024) and is kept for the research stages R1, R3, R4, R5. |
| **Model** | CNN stem + Transformer encoder, U-Net-style decoder: 3 741 423 parameters (counted by instantiating the default `ModelConfig`); embedding map 128 x 25 x 60. |
| **Cost** | On a 6 GB laptop GPU (RTX 3050): the released model trains in about 19 min on the eleven training years (from scratch, no pretraining); batch 8 with fp16 autocast peaks near 1.3 GB GPU memory. On the five-year `poc` run: pretraining 14 min + supervised training 11 min. |
| **Run it** | `start.bat` (dashboard on the finished runs or the released results), `oceanembed run-all --config configs/synthetic.yaml` (no logins), or the real runs: `configs/poc_long.yaml` for download / harmonise / statistics, then `configs/final.yaml` (free CMEMS + Earthdata accounts; sizes and times in [`reproduce.md`](reproduce.md)). Always from the project root. |

## 2. Diagram

```mermaid
flowchart LR
  subgraph SRC["Sources"]
    CMEMS["CMEMS: SST, SSS, SLA, GLORYS"]
    PODAAC["NASA PO.DAAC: OSCAR, CCMP"]
    ARGOSRC["Argo profiles (ERDDAP via argopy)"]
    SYNW["Synthetic world (analytic truth)"]
  end
  subgraph PROV["providers (data/providers)"]
    PC["cmems.py"]
    PP["podaac.py (OPeNDAP subset, granule fallback)"]
    PA["argo.py"]
    PS["synthetic.py"]
  end
  CMEMS --> PC
  PODAAC --> PP
  ARGOSRC --> PA
  SYNW --> PS
  PC --> RAW
  PP --> RAW
  PS --> RAW
  PA --> ARGOPQ
  RAW["data/raw/product/product_YYYYMM.nc (native grid)"]
  ARGOPQ["data/raw/argo/argo_YYYYMM.parquet"]
  RAW --> HARM["harmonize: regrid, daily mean, vertical interp"]
  HARM --> ZARR[("data/processed/run.zarr")]
  ZARR --> STATS["stats: train-split mean/std, harmonic climatology"]
  STATS --> STATSNC[("run_stats.nc")]
  ZARR --> DS["dataset.py: OceanDataset / SurfaceOnlyDataset"]
  STATSNC --> DS
  DS --> PRE["pretrain: masked surface autoencoder"]
  PRE --> CK1["pretrain.pt"]
  CK1 --> TRN["train: reconstruction (fine-tune encoder)"]
  DS --> TRN
  DS --> TRS["train --no-pretrained --tag scratch"]
  DS --> BASE["baseline: ridge"]
  TRN --> CK2["recon.pt"]
  TRS --> CK3["recon_scratch.pt"]
  BASE --> CK4["ridge.joblib"]
  CK2 --> EMB["embed: embeddings.zarr"]
  CK2 --> PRED["predict: oceanembed_T_YYYYMM.nc"]
  CK3 --> PRED
  CK4 --> PRED
  PRED --> EVAL["evaluate: vs GLORYS"]
  ZARR --> EVAL
  ARGOPQ --> VAL["validate-argo: collocate, score"]
  PRED --> VAL
  EVAL --> REP["report: figures + report.md"]
  VAL --> REP
  EVAL --> RUN
  VAL --> RUN
  REP --> RUN
  EMB --> RUN
  PRED --> RUN
  RUN[("outputs/run/ (run folder)")]
  RUN --> DA["data_access.py (read-only loaders)"]
  ZARR --> DA
  DA --> API["api/: FastAPI (Store, routes, ETag, NaN to null)"]
  API --> BR["Browser: web/dist React app"]
```

The Streamlit proof-of-concept app (`app/`) reads the same run folder through the same `data_access.py`.

## 3. Module map

Paths are relative to `src/oceanembed/`. "-" = none.

| File | Responsibility | Key functions / classes | Inputs | Outputs |
|---|---|---|---|---|
| [`__init__.py`](../src/oceanembed/__init__.py), `data/__init__.py`, `data/providers/__init__.py`, `eval/__init__.py`, `infer/__init__.py`, `models/__init__.py`, `train/__init__.py` | Package markers (only the top one holds `__version__ = "0.1.0"`) | - | - | - |
| [`config.py`](../src/oceanembed/config.py) | Pydantic config models, YAML loader, derived paths, product defaults | `Config`, `load_config`, `default_products`, `STANDARD_DEPTHS`, `SURFACE_VARS` | YAML, env `OCEANEMBED_DATA_ROOT` / `OCEANEMBED_OUTPUTS_ROOT` | `Config` object |
| [`grid.py`](../src/oceanembed/grid.py) | Canonical grid, depths, basin boxes | `Grid`, `build_grid`, `daily_axis`, `BASINS` | `Config` | lat/lon/depth arrays, basin masks |
| [`runmeta.py`](../src/oceanembed/runmeta.py) | Self-description of a run | `build_run_meta`, `write_run_meta`, `read_run_meta`, `data_source` | `Config` | `outputs/<run>/run_meta.json` |
| [`cli.py`](../src/oceanembed/cli.py) | Typer app, `run-all` orchestration, `serve` | `app`, `_step_plan`, `run_all`, `serve` | YAML + flags | calls the modules below |
| [`data_access.py`](../src/oceanembed/data_access.py) | Read-only loaders of a run folder (no UI code), cached | `list_runs`, `load_prediction`, `load_target`, `load_climatology`, `load_profile`, `load_section`, `load_embeddings`, `embedding_pca_rgb`, `prediction_methods`, `available_dates`, `load_metrics_*` | `outputs/<run>`, harmonised Zarr, stats file | xarray / pandas / dict objects |
| [`data/providers/base.py`](../src/oceanembed/data/providers/base.py) | Provider protocol, month iteration, raw paths, dataset-by-date choice, transfer ledger | `Provider`, `make_provider`, `months`, `raw_file`, `source_for_range`, `log_transfer`, `MissingCredentialsError`, `ALL_PRODUCTS` | `Config` | `data/raw/_download_log.jsonl` lines |
| [`data/providers/cmems.py`](../src/oceanembed/data/providers/cmems.py) | CMEMS month-by-month server-side subset | `CmemsProvider.fetch`, `subset_kwargs`, `credentials_available` | CMEMS login | `raw/<product>/<product>_YYYYMM.nc` |
| [`data/providers/podaac.py`](../src/oceanembed/data/providers/podaac.py) | OSCAR / CCMP day by day: OPeNDAP subset, granule fallback, parts, retries | `PodaacProvider`, `parse_dmr`, `index_runs`, `build_constraint` | Earthdata login | raw monthly NC, `raw/_parts/` |
| [`data/providers/argo.py`](../src/oceanembed/data/providers/argo.py) | Argo profiles (argopy), QC filter, TLS and NetCDF-3 workarounds, optional INCOIS grid | `ArgoProvider`, `fetch_argo_box`, `tidy_profiles`, `load_argo`, `load_incois_gridded` | ERDDAP (no login) | `raw/argo/argo_YYYYMM.parquet` |
| [`data/providers/synthetic.py`](../src/oceanembed/data/providers/synthetic.py) | Writes fake raw products in the real layout and native grids | `SyntheticProvider`, `generate_all` | `Config.synthetic` | same raw files as real |
| [`data/providers/synthetic_world.py`](../src/oceanembed/data/providers/synthetic_world.py) | Analytic ocean (land, eddies, thermocline, SSS, winds, geostrophic currents) | `SyntheticWorld`, `is_land`, `bathymetry`, `pressure_to_depth` | (lat, lon, time) | fields on any grid |
| [`data/regrid.py`](../src/oceanembed/data/regrid.py) | Coordinate standardisation, separable block-mean / bilinear regrid, daily mean, vertical interpolation | `standardize`, `regrid_horizontal`, `to_daily`, `interp_to_depths`, `to_celsius`, `crop` | xarray objects | float32 arrays on target grid |
| [`data/harmonize.py`](../src/oceanembed/data/harmonize.py) | Raw -> harmonised Zarr | `harmonize`, `process_variable`, `open_harmonized` | raw monthly files | `data/processed/<run>.zarr` |
| [`data/stats.py`](../src/oceanembed/data/stats.py) | Train-split mean/std, harmonic climatology, anomaly std | `compute_stats`, `Stats`, `harmonic_design`, `n_terms_for_span` | Zarr | `data/processed/<run>_stats.nc` |
| [`data/dataset.py`](../src/oceanembed/data/dataset.py) | Torch datasets, normalisation, de-normalisation | `OceanDataset` (`preload`, `denormalize`), `SurfaceOnlyDataset`, `make_dataset`, `INPUT_CHANNELS` | Zarr + stats | batches `x`, `y`, `mask`, `sv`, `t`, `index` |
| [`models/blocks.py`](../src/oceanembed/models/blocks.py) | Shared conv blocks | `ResBlock`, `Up`, `groups_for` | tensors | tensors |
| [`models/encoder.py`](../src/oceanembed/models/encoder.py) | CNN stem + Transformer -> embedding map | `OceanEncoder`, `sincos_2d`, `EncoderOutput` | `(B,12,H,W)` (+ visible map) | `emb`, `skip1`, `skip2` |
| [`models/mae.py`](../src/oceanembed/models/mae.py) | Masked surface autoencoder | `MaskedAutoencoder`, `PretrainDecoder`, `block_mask` | `x`, `sv` | masked loss, per-channel sums |
| [`models/recon.py`](../src/oceanembed/models/recon.py) | Embedding -> 15-depth anomaly decoder | `ReconModel`, `ReconDecoder`, `load_encoder_state`, `param_groups` | `(B,12,H,W)` | `(B,15,H,W)` standardised anomaly |
| [`models/baselines.py`](../src/oceanembed/models/baselines.py) | Climatology and ridge predictors | `ClimatologyBaseline`, `RidgeBaseline`, `fit_ridge`, `rmse_per_depth` | train dataset | `ridge.joblib` |
| [`models/pixel_mlp.py`](../src/oceanembed/models/pixel_mlp.py) | Per-pixel MLP baseline (research R1) | `PixelMLP`, `load_pixel_mlp` | `(B,12,H,W)` | `(B,15,H,W)` |
| [`train/losses.py`](../src/oceanembed/train/losses.py) | Masked losses | `masked_mse`, `vertical_gradient_loss`, `reconstruction_loss` | pred, target, mask | scalars |
| [`train/utils.py`](../src/oceanembed/train/utils.py) | Seed, device, LR schedule, JSONL logger, loaders, crop | `set_seed`, `cosine_warmup`, `JsonlLogger`, `make_loader`, `random_crop` | - | - |
| [`train/pretrain.py`](../src/oceanembed/train/pretrain.py) | Pretraining loop | `run_pretrain`, `validate` | datasets (train, val) | `checkpoints/pretrain.pt`, `logs/pretrain.jsonl` |
| [`train/train.py`](../src/oceanembed/train/train.py) | Supervised loop, early stopping | `run_train`, `recon_ckpt_name`, `validate` | datasets, `pretrain.pt` | `recon[_tag].pt`, `logs/train[_tag].jsonl` |
| [`train/mlp.py`](../src/oceanembed/train/mlp.py) | MLP training on a random point sample | `run_train_mlp` | datasets | `mlp.pt` |
| [`research/r1.py`](../src/oceanembed/research/r1.py) | R1 runner: train + score every (method, seed), resumable | `run_r1`, `run_job`, `plan_jobs`, `evaluate_to_file` | config, main run's ridge | `research/r1/<method>/seed<k>/` |
| [`research/bootstrap.py`](../src/oceanembed/research/bootstrap.py) | Moving-block bootstrap over days from per-day sums, paired differences, decorrelation time | `block_counts`, `metrics_from_daily`, `paired_difference`, `decorrelation_time` | `(F,T,D)` sums | arrays |
| [`research/r1_report.py`](../src/oceanembed/research/r1_report.py) | R1 summary, tables, figures | `make_r1_report`, `build_summary` | `eval.npz` files | `summary.json`, `summary.md`, `figures/` |
| [`research/armor3d.py`](../src/oceanembed/research/armor3d.py) | ARMOR3D month-by-month CMEMS subset (resumable, ledger entry), regridding with the GLORYS rules | `download_armor`, `subset_kwargs`, `regrid_month` | CMEMS login | `data/raw/armor3d/armor3d_YYYYMM.nc` |
| [`research/r4.py`](../src/oceanembed/research/r4.py), [`r4_report.py`](../src/oceanembed/research/r4_report.py) | R4: regridded ARMOR3D vs Argo matchups and the grid, block-bootstrap summary, figures | `run_r4`, `regrid_all`, `argo_table`, `stream_grid`, `make_r4_report` | `argo_matchups.parquet`, predictions, Zarr | `research/r4/`, `data/processed/<run>_armor3d/` |
| [`research/physical.py`](../src/oceanembed/research/physical.py) | Derived quantities (isotherm depth, layer integral, heat content), terciles, seasons | `isotherm_depth`, `layer_integral`, `heat_content`, `derived_fields` | `(D, ...)` temperature | arrays |
| [`research/r5.py`](../src/oceanembed/research/r5.py), [`r5_report.py`](../src/oceanembed/research/r5_report.py) | R5: streaming per-day sums for derived quantities and strata, report | `run_r5`, `make_r5_report` | predictions, Zarr, stats | `research/r5/` |
| [`research/common.py`](../src/oceanembed/research/common.py) | Bootstrap scoring and number formatting shared by the R4 / R5 reports | `score_series`, `paired`, `interval` | per-day sums | dicts |
| [`research/inputs.py`](../src/oceanembed/research/inputs.py) | R3 input manipulation: variable groups, `InputSpec`, masking (zeroed channels), history channels, `InputView` (training view of a preloaded dataset), test-time batch builder, within-month permutation | `InputSpec`, `EXPERIMENTS`, `InputView`, `make_test_batch_fn`, `sample_points_view`, `month_permutation`, `permute_batch` | preloaded datasets, store | tensors |
| [`research/r3.py`](../src/oceanembed/research/r3.py), [`r3_report.py`](../src/oceanembed/research/r3_report.py) | R3: retrain-without / history / permutation jobs (resumable), multi-pass scoring with one reference read, report | `run_r3`, `plan_jobs`, `score_passes`, `region_sums`, `make_r3_report` | config, Zarr, stats, R1 jobs | `research/r3/` |
| [`research/r2.py`](../src/oceanembed/research/r2.py), [`r2_report.py`](../src/oceanembed/research/r2_report.py) | R2: long-period jobs (Transformer, MLP, ridge, climatology, learning curve), Argo scoring of two test years, report | `run_r2`, `plan_jobs`, `train_days`, `make_r2_report` | long config, Zarr, stats, memmap caches, Argo files | `research/r2/` |
| [`infer/predict.py`](../src/oceanembed/infer/predict.py) | Checkpoint loading, predictors, degC conversion, CF-1.8 NetCDF writer | `load_recon_model`, `model_predictor`, `predict_batch`, `predict_to_netcdf`, `expected_product_files` | checkpoint, surface dataset | `predictions/**/oceanembed_T_YYYYMM.nc` |
| [`infer/embed.py`](../src/oceanembed/infer/embed.py) | Embedding export | `export_embeddings`, `embedding_path` | encoder weights, split | `embeddings/embeddings.zarr` |
| [`live/nrt.py`](../src/oceanembed/live/nrt.py) | Near-real-time catalogue probe and resumable per-day download (`data/raw_nrt/`) | `probe`, `fetch_days`, `runs_of_days`, `required_products`, `describe_dataset`, `subset_to_file` | CMEMS login (download only) | `raw_nrt/<product>/<product>_YYYYMMDD.nc`, ledger lines |
| [`live/harmonise.py`](../src/oceanembed/live/harmonise.py) | Raw NRT days -> canonical grid with the batch rules; small live store | `harmonise_files`, `write_store` | raw day files | `data/processed/live.zarr` |
| [`live/reconstruct.py`](../src/oceanembed/live/reconstruct.py) | Released weights -> reconstruction of chosen days; rolling monthly NetCDF (merge, prune) | `load_weights`, `reconstruct`, `write_days`, `read_predictions` | `models/final`, live store | `outputs/live/predictions/oceanembed_T_YYYYMM.nc` |
| [`live/update.py`](../src/oceanembed/live/update.py) | `live update` / `live status`: availability, window, revision policy, provenance | `run_update`, `status`, `window_for`, `revision_set`, `revision_size` | catalogue, raw days | `live_state.json`, `live_days.parquet`, `first_published/`, `run_meta.json` |
| [`live/verify.py`](../src/oceanembed/live/verify.py) | Running verification against Argo and the operational analysis (rolling record) | `run_verification`, `daily_rows`, `rolling_series`, `argo_matchups` | window, Argo (argopy), CMEMS analysis | `checks/verification/` |
| [`live/shift.py`](../src/oceanembed/live/shift.py) | `live input-shift`: NRT vs reprocessed inputs, reconstruction difference, error vs GLORYS with paired bootstrap | `run_input_shift`, `summarise_shift`, `recon_sums` | reprocessed + NRT inputs, GLORYS | `checks/input_shift/` |
| [`live/state.py`](../src/oceanembed/live/state.py) | Folders, state / provenance files, update lock | `live_paths`, `load_state`, `load_days`, `UpdateLock` | - | - |
| [`research/benchmark.py`](../src/oceanembed/research/benchmark.py), [`benchmark_report.py`](../src/oceanembed/research/benchmark_report.py) | Phase 13: boosted trees, random forest, plain U-Net (and SST + sea level ridge / MLP) under the R2 protocol; one ranked report | `run_benchmark`, `plan_jobs`, `TreePredictor`, `make_benchmark_report` | long config, Zarr, stats, memmap caches, R2 / final-inputs jobs | `research/benchmark/` |
| [`eval/metrics.py`](../src/oceanembed/eval/metrics.py) | Streaming sums-based metrics | `MetricAccumulator`, `metrics_from_sums`, `skill_score`, `point_metrics` | pred / ref arrays | metric dicts |
| [`eval/evaluate.py`](../src/oceanembed/eval/evaluate.py) | All methods vs GLORYS on a split | `evaluate_split`, `load_methods`, `method_label` | checkpoints, Zarr | `metrics/metrics_glorys.json`, `maps_glorys.nc` |
| [`eval/argo_validation.py`](../src/oceanembed/eval/argo_validation.py) | Argo interpolation, collocation, scoring, optional INCOIS | `validate_argo`, `interp_profile`, `collocate`, `max_gap` | Argo parquet, checkpoints | `metrics/metrics_argo.json`, `argo_matchups.parquet` |
| [`eval/report.py`](../src/oceanembed/eval/report.py) | Figures and `report.md` | `make_report`, `write_markdown`, `fig_*` | metrics files | `figures/*.png`, `report.md` |
| [`api/app.py`](../src/oceanembed/api/app.py) | App factory: CORS, GZip, error handlers, SPA serving, warm-up thread | `create_app`, `create_app_from_env`, `default_outputs_root` | outputs root, `web/dist` | FastAPI app |
| [`api/store.py`](../src/oceanembed/api/store.py) | Run registry, validation, byte-bounded LRU, colour ranges, point series | `Store`, `Run`, `ByteLRU`, `ApiError`, `robust_range` | run folders | numpy arrays, dicts |
| [`api/deps.py`](../src/oceanembed/api/deps.py) | `{run}` dependency, ETag / 304, cache headers | `run_dep`, `make_etag`, `reply`, `reply_bytes`, `CACHE_CONTROL` | request | response helpers |
| [`api/encode.py`](../src/oceanembed/api/encode.py) | NaN -> null JSON encoding | `NaNSafeJSONResponse`, `clean`, `array_to_list` | payloads | JSON bytes |
| [`api/schemas.py`](../src/oceanembed/api/schemas.py) | Pydantic response models (drive `/api/docs`) | response classes | - | OpenAPI schema |
| [`api/metrics.py`](../src/oceanembed/api/metrics.py) | Metrics JSON -> metric-major shape, experiments rows | `reshape_metrics`, `experiment_rows`, `method_infos` | metrics JSONs | dicts |
| [`api/routes_runs.py`](../src/oceanembed/api/routes_runs.py) | Run list / detail / dates / mask / training / experiments / compare / report / figures / product | `router` | `Store` | JSON, PNG, NetCDF |
| [`api/routes_fields.py`](../src/oceanembed/api/routes_fields.py) | Fields, ranges, surface, profile, section, timeseries, embeddings | `router` | `Store` | JSON or float32 bytes |
| [`api/routes_metrics.py`](../src/oceanembed/api/routes_metrics.py) | Metrics, error maps, Argo matchups and profiles | `router` | `Store` | JSON |
| `api/__init__.py` | Re-exports `create_app` | - | - | - |

## 4. Data layer

### Sources

All six gridded products are subset to the domain plus a 0.5 deg halo (`download.halo_deg`) and then
harmonised. Dataset ids are those of [`configs/poc.yaml`](../configs/poc.yaml).

| Variable(s) | Product | Dataset id | Provider | Native resolution | Access route | Regrid to 0.25 deg daily |
|---|---|---|---|---|---|---|
| `sst` (`analysed_sst`) | OSTIA reprocessed L4 | `METOFFICE-GLO-SST-L4-REP-OBS-SST` | cmems | 0.05 deg, daily (stamped 12:00) | `copernicusmarine.subset`, one call per month | 5 x 5 block mean; K -> degC |
| `sss` (`sos`) | MULTIOBS SMAP/SMOS | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` | cmems | 0.125 deg, daily | subset per month | 2 x 2 block mean |
| `sla` (`sla`) | DUACS L4 | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | cmems | 0.125 deg (the PRD says 0.25), daily | subset per month | 2 x 2 block mean |
| `uo`, `vo` (`u`, `v`) | OSCAR L4 final v2.0 | `OSCAR_L4_OC_FINAL_V2.0` | podaac | 0.25 deg, daily, nodes on multiples of 0.25 | OPeNDAP per-day subset (below) | bilinear (half-cell offset) |
| `uw`, `vw` (`uwnd`, `vwnd`) | CCMP v3.1 | `CCMP_WINDS_10M6HR_L4_V3.1` | podaac | 0.25 deg, 6-hourly | OPeNDAP per-day subset | daily mean of the 4 samples, then bilinear |
| `temp` (`thetao`) | GLORYS12 reanalysis (target) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | cmems | 1/12 deg, daily; 36 of the 50 levels (<= 1100 m) requested | subset per month, `min_depth 0`, `max_depth 1100` | vertical linear interpolation to the 15 depths, then 3 x 3 block mean |
| Argo temperature | Argo profiles | argopy `erddap` source, `ds="phy"`, `mode="standard"` | argo | profiles | 15 deg boxes per month, QC flags 1 and 2 | not gridded; see section 7 |

Rule for the horizontal method ([`regrid.py`](../src/oceanembed/data/regrid.py)): block mean when the target/source step ratio
is > 1.5 on both axes, else NaN-aware bilinear. Both are `Wy @ field @ Wx.T`. A block-mean cell needs
>= 50 % (`harmonize.min_valid_fraction`) of its nominal native cells valid; a bilinear cell needs >= 50 % of the
interpolation weight on valid neighbours. Vertical interpolation is linear; targets shallower than the
shallowest source level take that level's value, targets below the deepest source level are NaN.

A product may list several datasets with `start` / `end` (`DatasetSource`); each month uses the first dataset whose
window covers it, so switch-over dates must be month boundaries. The built-in default for `sss` switches from the
reprocessed dataset (to 2023-12-31) to the NRT one (from 2024-01-01); `poc.yaml` overrides it with the reprocessed
dataset only, so train and test see one SSS product.

### PO.DAAC route ([`podaac.py`](../src/oceanembed/data/providers/podaac.py))

- `download.podaac_mode: opendap` (default): for each day, one DAP4 request to NASA Hyrax
  (`opendap.earthdata.nasa.gov`, through the `earthaccess` session) asks only for the configured variables over the
  index ranges covering domain + halo. Layout and ranges are read once per product from the granule's DMR and its
  coordinate arrays (handles ascending / descending axes, 0-360 vs -180-180 longitude, a window crossing the seam as
  two runs). About 0.2 MB per OSCAR day and 0.75 MB per CCMP day instead of ~32 MB (runbook, measured on the trial).
- `granule`, or the automatic fallback: download the global granule, crop locally, delete it
  (`download.delete_global_granules`). Fallback triggers: an OPeNDAP refusal (HTTP 4xx or unusable layout) before any
  layout is known switches OPeNDAP off at once; otherwise OPeNDAP is switched off for the rest of the run after 3
  failures (refusals or exhausted retries).
- Resumability: every day is an atomic piece `raw/_parts/<product>_<YYYYMM>/<YYYYMMDD>.nc` (temp file + rename);
  existing pieces are skipped; `download.workers` (1-8, default 4) threads; transient errors (408, 425, 429, 5xx,
  network errors) retry `download.retries` (4) times with back-off starting at `retry_backoff_s` (2 s) and doubling.
  The monthly file is merged only when every available day exists, then the pieces are deleted. CMEMS and Argo
  resume at month level: an existing monthly file is skipped unless `download.overwrite`.
- Failures: if some days fail, the month raises `N of M day(s) failed`, finished days stay in `_parts/`, a re-run
  resumes. A month with no catalogue granules at all raises `no <product> granules found`.

### Transfer ledger

`data/raw/_download_log.jsonl`, one JSON line per completed download (`base.log_transfer`): `ts`, `product`,
`dataset`, `period`, `route` (`cmems_subset`, `opendap`, `granule`, `granule_fallback`, `argopy`), `bytes_written`,
`bytes_transferred`, `transfer_basis` (`wire` = measured compressed bytes, `file_size`, `unknown`), `seconds`. Never
credentials, user names or URLs (tests assert this).

### Credentials (existence checks only; values are never read, printed or stored)

| Service | Where the code looks |
|---|---|
| Copernicus Marine | env `COPERNICUSMARINE_SERVICE_USERNAME` + `..._PASSWORD`, or `~/.copernicusmarine/.copernicusmarine-credentials` (created by `copernicusmarine login`) |
| NASA Earthdata | env `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`, or an entry for `urs.earthdata.nasa.gov` in `~/.netrc` / `~/_netrc` |
| Argo (ERDDAP) | none |

Missing credentials raise `MissingCredentialsError` before any network call; `oceanembed download` prints
`error: ...` and exits with code 2. `.gitignore` excludes `.env`, `.netrc`, `_netrc`, `.copernicusmarine/`.

### Canonical grid and harmonised Zarr

Grid ([`grid.py`](../src/oceanembed/grid.py)): lat 5.125 .. 29.875 (100 centres), lon 45.125 .. 104.875 (240 centres), depths
0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m, time daily 00:00 UTC. `GridConfig` rejects a grid whose
H or W is not divisible by 4. Basin boxes (centre inside): `arabian_sea` 5-25 N, 45-78 E; `bay_of_bengal` 5-25 N, 80-100 E
(cells in neither box, e.g. north of 25 N, are in no basin score).

`data/processed/<run>.zarr` (Zarr v3, Blosc zstd level 3 + bitshuffle, written by `harmonize`, read with `consolidated=False`):

| Variable | Dims | Type | Chunks | Notes |
|---|---|---|---|---|
| `sst sss sla uo vo uw vw` | `(time, lat, lon)` | float32 | `(1, 100, 240)` | NaN over land and gaps; units degC, 1e-3 (PSU), m, m s-1 |
| `temp` | `(time, depth, lat, lon)` | float32 | `(1, 15, 100, 240)` | GLORYS on the 15 depths; NaN below sea floor / land |
| `mask` | `(depth, lat, lon)` | bool | default | valid in `temp` on >= 50 % of all days of the store |

Processing: month by month, in blocks of `harmonize.time_chunk` (8) days, so the 1/12 deg 3-D field is never
fully in memory. Days absent from the raw data stay NaN (a warning counts them). If `raw/argo_gridded/*.nc` exists
it is also regridded to `processed/<run>_argo_gridded.nc`. `harmonize` deletes and rebuilds an existing store.

### Statistics ([`stats.py`](../src/oceanembed/data/stats.py)), train split only

`data/processed/<run>_stats.nc`: `input_mean`, `input_std` per surface channel (over surface ocean points), `clim_coef`
`(5, depth, lat, lon)` (zlib 3), `anom_std` `(depth,)`; attributes `train_start`, `train_end`, `n_train_days`,
`n_harmonic_terms`.

```
T_clim(x, z, date) = c0 + c1 cos(phi) + c2 sin(phi) + c3 cos(2 phi) + c4 sin(2 phi)
phi = 2 pi (date - Jan 1 of its year) / (days in that year)        # leap years handled
```

Fitted per grid point and depth by streaming normal equations (two passes; pass 2 computes the anomaly std per depth
over valid points). Fewer than 180 train days: mean only (1 term); fewer than 548: mean + annual (3); else 5. Points
with fewer than 30 valid days get NaN coefficients. For the real run `n_harmonic_terms` is 5.

## 5. Model

### The 12 input channels (`INPUT_CHANNELS`, in order)

| # | Channel | Content |
|---|---|---|
| 0-6 | `sst, sss, sla, uo, vo, uw, vw` | `(raw - mean) / std` with the train-split surface-ocean mean/std; NaN or land -> 0 |
| 7 | `ocean_mask` | surface level of `mask` (1 = ocean) |
| 8, 9 | `doy_sin`, `doy_cos` | `sin / cos(2 pi (day_of_year - 1) / days_in_year)` |
| 10, 11 | `lat_norm`, `lon_norm` | latitude / longitude scaled to [-1, 1] over the grid |

Item targets: `y = (temp - climatology) / anom_std[depth]` (0 where invalid), `mask (15,H,W)` = static mask and finite
target, `sv (7,H,W)` = surface values really observed. De-normalisation (`OceanDataset.denormalize`):
`temp = climatology(date) + y * anom_std[depth]`; `predict_batch` then sets NaN outside the static mask. The model input is
12 channels; the encoder appends one internal "visible" channel (13 convolution inputs).

### Encoder ([`encoder.py`](../src/oceanembed/models/encoder.py)) with shapes for the 100 x 240 grid

| Stage | Operation | Output shape (B = batch) |
|---|---|---|
| input | 12 channels + visible (1 everywhere downstream) | `(B, 13, 100, 240)` |
| stem0 | conv 3x3 13 -> 32, ResBlock(32) | `skip1 (B, 32, 100, 240)` |
| stem1 | conv 3x3 stride 2 32 -> 64, ResBlock(64) | `skip2 (B, 64, 50, 120)` |
| stem2 | conv 3x3 stride 2 64 -> 128, ResBlock(128) | `(B, 128, 25, 60)` |
| tokens | 1x1 conv 128 -> 192, flatten, add fixed 2-D sin-cos positions | `(B, 1500, 192)` |
| Transformer | 6 pre-norm blocks, 6 heads, MLP ratio 4, SDPA attention | `(B, 1500, 192)` |
| projection | LayerNorm, linear 192 -> 128 | embedding `(B, 128, 25, 60)` |

ResBlock = (GroupNorm - SiLU - conv3x3) x 2 with a 1x1 shortcut when channels change. Parameters (default
`ModelConfig`, counted by instantiating): encoder 3 203 584; reconstruction decoder 537 839; total `ReconModel`
3 741 423. Pretraining model: encoder + 267 783-parameter decoder = 3 471 367.

### Masked pretraining ([`mae.py`](../src/oceanembed/models/mae.py))

- Mask: the image is tiled by `pretrain.block` x `block` (16 x 16 px) blocks; exactly `round(mask_ratio * n_blocks)`
  (50 %) blocks per sample are hidden (ragged border blocks are cropped).
- Only the 7 surface channels are hidden (set to 0). `ocean_mask`, day-of-year and lat/lon are never hidden.
- "Visible" channel: 0 inside hidden blocks, 1 elsewhere; downstream (`ReconModel`, embedding export) it is all ones,
  so one encoder interface serves both stages.
- Leakage guard: the encoder (and therefore its stem skips) only ever sees the masked input; the pretraining decoder
  receives the embedding map only, no skips, so the embedding itself must carry the surface state. Tested by
  `test_hidden_values_cannot_influence_the_prediction` and `test_masking_hides_only_surface_channels`.
- Pretraining decoder: ResBlock(128 -> 64), Up, ResBlock, Up (64 -> 32), ResBlock, GroupNorm, SiLU, 1x1 conv -> 7 channels.
- Loss: MSE (standardised units) over pixels that are hidden AND `sv` (ocean and observed). Validation also reports the
  mean-fill baseline (predict 0 = the train mean) per channel, with a fixed generator (seed 1234) so the hidden blocks
  are identical every epoch.

### Reconstruction decoder ([`recon.py`](../src/oceanembed/models/recon.py))

ResBlock(128 -> 128) on the embedding; Up to 50 x 120, concatenate `skip2`, ResBlock(128 -> 64); Up to 100 x 240,
concatenate `skip1`, ResBlock(64 -> 32); GroupNorm, SiLU, 1x1 conv 32 -> 15. Output: standardised temperature anomaly
`(B, 15, 100, 240)`. `Up` = nearest x2 upsample + 3x3 conv.

### Losses ([`losses.py`](../src/oceanembed/train/losses.py))

```
masked_mse(p, y, m)  = sum_m (p - y)^2 / max(1, count(m))                  # masked points never enter, even as NaN
vgrad(p, y, m)       = masked_mse(p[z+1]-p[z], y[z+1]-y[z], m[z+1] & m[z])  # adjacent depths, both valid
total                = masked_mse + vertical_grad_weight * vgrad            # weight 0.05
```

### Baselines ([`baselines.py`](../src/oceanembed/models/baselines.py))

Both are predictors `x (B,12,H,W) -> standardised anomaly (B,15,H,W)`, like the network.
- **Climatology:** anomaly 0, i.e. the harmonic climatology.
- **Ridge:** 11 features (channels 0-6 and 8-11; the constant mask channel is unused), one `sklearn.Ridge`
  (`alpha = 1.0`) per depth, fitted on a random subsample (`baseline.ridge_max_points` = 1 000 000, seed 0) of train-split
  ocean (day, pixel) samples with all 7 surface values observed, using only samples valid at that depth (depths with
  < 50 samples keep zero weights). Saved as `checkpoints/ridge.joblib`.

## 6. Training

| | Pretraining (`pretrain`) | Reconstruction (`train`) |
|---|---|---|
| Loop | [`pretrain.py`](../src/oceanembed/train/pretrain.py) | [`train.py`](../src/oceanembed/train/train.py) |
| Optimiser | AdamW; weight decay 0.05 on tensors with ndim >= 2, none on the rest | AdamW; weight decay 0.01 likewise; param groups per encoder / decoder |
| Learning rate | 3e-4 | decoder 3e-4; encoder 3e-4 x `encoder_lr_scale` (0.1) when pretrained, x 1.0 for the from-scratch ablation |
| Schedule | linear warm-up for 1 epoch, then cosine decay to 5 % of the base LR (per step; planned over all `epochs`) | same |
| Epochs (default) | 20 | 30 |
| Batch size | 8 (shuffled, last incomplete batch dropped) | 8 |
| Mixed precision | fp16 autocast + `GradScaler`, CUDA only (`amp: true`) | same |
| Gradient clip | 1.0 (global norm) | 1.0 |
| Selection | best epoch by val masked MSE; no early stopping | best epoch by val RMSE in degC; early stopping after `patience` = 8 epochs without improvement |
| Seed | 0 (`set_seed`); `--seed` overrides | 0; `--seed` overrides |
| Data | train and val splits preloaded into RAM (float32; ~2 GB for 2 train years, ~5 GB for 5) | same |
| Augmentation | optional `crop` (h, w multiple of 4), off in every shipped config | same |
| Logged (JSONL, one line per epoch) | `epoch`, `train_loss`, `val_loss`, `val_meanfill`, `val_mse_per_channel`, `val_meanfill_per_channel`, `lr`, `epoch_seconds`, `train_seconds`, `peak_gpu_mb` | `epoch`, `train_loss`, `val_loss`, `val_rmse`, `val_rmse_per_depth`, `lr_decoder`, `epoch_seconds`, `train_seconds`, `peak_gpu_mb` |
| Checkpoint | `checkpoints/pretrain.pt`: `model`, `config`, `epoch`, `metrics` | `recon.pt` (or `recon_<tag>.pt`): the same plus `pretrained`, `tag` |

- `set_seed` seeds python, numpy, torch and every CUDA device and sets cuDNN to deterministic kernels without benchmarking. On CPU
  the same seed gives bit-identical weights (tested). On CUDA it gives close, not identical, results: the backward passes of
  nearest-neighbour upsampling and of scaled-dot-product attention use atomic additions and fp16 autocast reductions are order
  dependent. `--seed` sets the initialisation, the batch order, the masking and the crop RNG together.
- Val RMSE in degC is computed as `(anomaly error) * anom_std[depth]`: the climatology cancels in prediction minus truth.
- `train` loads the encoder from `pretrain.pt` (error if missing) unless `--no-pretrained`; then the tag defaults to
  `scratch` so `recon.pt` is never overwritten.
- Ablation: `ablation: {no_pretrained: true, tag: scratch}` makes `run-all` add `train-ablation` and `predict-ablation`; the
  tag may not be `ridge` (reserved folder) and must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,30}$`.
- Hyper-parameters of the real run are the config defaults (`configs/poc.yaml` has no `model`, `pretrain` or `train` block).
  Epoch counts the run actually used are in `outputs/<run>/logs/*.jsonl` (not re-read for this file).

## 7. Evaluation

### Metrics ([`eval/metrics.py`](../src/oceanembed/eval/metrics.py))

`MetricAccumulator` keeps per grid point `(depth, lat, lon)` the running sums `n, sum x, sum y, sum x^2, sum y^2, sum xy,
sum |e|, sum e^2` (x = prediction, y = reference, e = x - y), skipping pairs where either is non-finite or masked. Sums
are additive, so chunking equals one pass and any grouping (all, one depth, a basin at one depth, one point) is a
reduction afterwards.

```
rmse = sqrt(sum e^2 / n)        bias = mean(x) - mean(y)        mae = sum |e| / n
corr = cov(x, y) / (std x * std y)       NaN if n < 3 or a variance <= 1e-12
skill_vs_clim = 1 - MSE_method / MSE_climatology      (1 perfect, 0 = as good as climatology, < 0 worse)
```

- `corr_raw`: correlation of temperatures (inflated by seasonal, horizontal and, when pooled, vertical gradients).
- `corr_anom`: correlation after removing the climatology from both prediction and reference; the day-to-day skill
  measure. Undefined (null) for the climatology itself.
- Pooled range: depths 50-200 m = 6 levels (50, 75, 100, 125, 150, 200).

### `evaluate` ([`evaluate.py`](../src/oceanembed/eval/evaluate.py))

Streams the split in batches of 8 days. Methods: `model` (`recon.pt`, required), every `recon_<tag>.pt` as
`model_<tag>`, `ridge` (if fitted), `climatology`. A point counts when the static mask and the GLORYS value are
finite. Per method, blocks `overall`, `pooled_50_200m`, `per_depth` (column-oriented), and the same under
`per_basin.{arabian_sea, bay_of_bengal}`. Also per-day, per-depth domain sums giving `daily_rmse`: `rmse`, `bias`,
`corr_anom` as `[day][depth]` (the daily `corr_anom` is the spatial pattern correlation over that day and depth),
and `pooled_rmse`, `pooled_bias`, `pooled_corr_anom` as `[day]`. Also `maps_glorys.nc` (see section 8).

### `validate-argo` ([`argo_validation.py`](../src/oceanembed/eval/argo_validation.py))

1. Load Argo for the split months (cached parquet; missing months are fetched, synthetic ones generated). Profiles
   outside the split's dates are dropped and counted.
2. **Vertical rule** per standard depth z on the profile's good-QC levels: an observation exactly at z is used;
   otherwise z must be bracketed by levels whose spacing is <= `max(10 m, 0.2 z)` (10 m up to 50 m, 20 m at 100 m,
   40 m at 200 m, 100 m at 500 m, 200 m at 1000 m) and is linearly interpolated; for z <= 5 m the shallowest level is used if
   within 10 m; never any extrapolation.
3. **Collocation:** the grid cell that contains the profile position (floor on cell edges) and the same UTC day.
   Dropped with counts: `outside_dates`, `outside_domain`, `land` (surface mask), plus `outside_period`.
4. **Fair-sample rule:** a (profile, depth) row is kept only if the observation, the static mask, GLORYS, the
   climatology and every method are finite there, so all methods are scored on the identical sample.
5. Scored per method and for `glorys` itself: `overall`, `pooled_50_200m`, `per_depth`, `per_basin`; bias is
   method minus Argo. Matchup columns: `profile_id, time, lat, lon, depth, obs, model, ridge, clim, [model_<tag>], glorys,
   basin, grid_lat, grid_lon` (the climatology column is `clim`, its key everywhere else is `climatology`).
6. Optional INCOIS gridded Argo: if `raw/argo_gridded/*.nc` exists, all methods are also scored against it
   (`gridded_argo` block; daily products same day, monthly products against the monthly model mean for fully covered
   months only); silently skipped when absent.

**Caveat (stored in the metrics metadata as `independence_note`):** Argo is independent of the model's inputs, but GLORYS
assimilates Argo and the model is trained on GLORYS, so the comparison is not independent of the training target.
GLORYS-vs-Argo is the floor any GLORYS-trained model faces.

## 8. Outputs

```
outputs/<run>/
  run_meta.json                      config snapshot, data_source, splits, grid, data paths (as the config gave them)
  checkpoints/
    pretrain.pt                      masked-autoencoder weights (best val masked MSE)
    recon.pt                         final model (best val RMSE degC)
    recon_<tag>.pt                   ablation, e.g. recon_scratch.pt
    ridge.joblib                     ridge coefficients per depth
  logs/
    pretrain.jsonl, train.jsonl      one JSON line per epoch
    train_<tag>.jsonl                ablation log
  predictions/
    oceanembed_T_<YYYYMM>.nc         main model, one file per month of the test split
    ridge/oceanembed_T_<YYYYMM>.nc   ridge baseline (reserved folder name)
    <tag>/oceanembed_T_<YYYYMM>.nc   ablation
  metrics/
    metrics_glorys.json              per method: overall, pooled_50_200m, per_depth, per_basin, plus daily_rmse, metadata
    maps_glorys.nc                   (depth, lat, lon): rmse_, bias_, corr_raw_, skill_vs_clim_ per method; corr_anom_ (not climatology); n_valid
    metrics_argo.json                same blocks vs Argo incl. glorys; profile counts, dropped counts, rule, independence note
    argo_matchups.parquet            one row per (profile, depth)
  embeddings/embeddings.zarr         (time, emb, y, x) float32 for the test split; lat(y), lon(x) = 4 x 4 block centres
  figures/                           rmse_profile, bias_profile, anomaly_corr_profile, skill_profile, rmse_profile_basins,
                                     rmse_maps, bias_maps, daily_rmse, example_day_100m, section_lat, argo_validation,
                                     training_curves (.png)
  report.md                          setup, tables, figures, limitations (banner when data_source is synthetic)
```

Outside the run folder: `data/raw/<product>/`, `data/raw/_parts/`, `data/raw/_granules/`, `data/raw/_download_log.jsonl`,
`data/processed/<run>.zarr`, `<run>_stats.nc`, optional `<run>_argo_gridded.nc`.
`run_meta.json` is (re)written by every command except `synth`, `download`, `harmonize`, `stats` (`run-all` writes it first).
The API lists only folders that contain it. `embeddings.zarr` uses the encoder of `recon.pt`, else `pretrain.pt`.

### NetCDF product ([`predict.py`](../src/oceanembed/infer/predict.py))

| | |
|---|---|
| Variable | `temperature(time, depth, lat, lon)` float32, `standard_name` `sea_water_potential_temperature`, `units` `degree_Celsius`, `cell_methods` `time: mean` |
| Fill | NaN stored as `_FillValue` 9.96921e36 outside the static ocean mask |
| Compression | NetCDF4, zlib level 4, shuffle, one chunk per day `(1, 15, 100, 240)` |
| Coordinates | `time` int32 `days since 1970-01-01 00:00:00` (axis T); `depth` (positive down, Z); `lat` (Y); `lon` (X) |
| Global attributes | `Conventions` CF-1.8, `title`, `institution`, `source`, `method` (`model` / `ridge`), `history`, `references`, `comment`, `data_source`, `run_name`, `model_checkpoint`, `model_checkpoint_epoch`, `model_pretrained_encoder`, `training_period`, `grid_resolution_deg`, `config` (JSON) |
| Inputs needed | surface fields from the harmonised store plus the stats file only; the target is never read, so any day in the store works. A partial month holds only the requested days and a later run overwrites the file. |
| Size | about 150 MB per method for a one-year test split (decisions.md) |

## 9. CLI

All commands take `--config / -c <yaml>` (must exist) except `serve`. `--device` is e.g. `cuda` or `cpu` (default auto).

| Command | Options | Reads | Writes |
|---|---|---|---|
| `synth` | `-v/--variable` (repeatable) | config (`provider: synthetic` required) | raw files in `data/raw_synthetic/` |
| `download` | `-v/--variable` from `sst sss sla currents winds temp argo` (default all) | config, credentials | raw files, ledger; exit 2 on missing credentials |
| `harmonize` | - | raw files | `processed/<run>.zarr` (rebuilt) |
| `stats` | - | Zarr | `processed/<run>_stats.nc` |
| `pretrain` | `--seed`, `--device` | Zarr, stats | `checkpoints/pretrain.pt`, `logs/pretrain.jsonl` |
| `train` | `--no-pretrained`, `--tag`, `--seed`, `--device` | Zarr, stats, `pretrain.pt` | `checkpoints/recon[_tag].pt`, `logs/train[_tag].jsonl` |
| `baseline` | - | Zarr, stats | `checkpoints/ridge.joblib`; prints val RMSE of ridge and climatology |
| `embed` | `--split` (test), `--checkpoint`, `--device` | checkpoint, Zarr | `embeddings/embeddings.zarr` |
| `train-mlp` | `--seed`, `--device` | Zarr, stats (cache) | `checkpoints/mlp.pt`, `logs/baselines/mlp.jsonl` |
| `predict` | `--split` or `--start` + `--end`, `--tag`, `--ridge`, `--mlp`, `--weights <folder>`, `--out`, `--device` | checkpoint (ridge, mlp) or a released-weights folder, Zarr, stats | `predictions/[ridge/\|mlp/\|<tag>/]oceanembed_T_<YYYYMM>.nc` (with `--weights`: `predictions_from_weights/` or `--out`) |
| `export-results` | `--out results` | `outputs/` of the run and of the research stages | `results/` (tracked: metrics JSON, report, research summaries) |
| `export-weights` | `--out models/<run>` | checkpoints, stats, Zarr mask | `models/<run>/` (`recon.pt`, `ridge.joblib`, `mlp.pt`, `stats.nc`, `manifest.json`) |
| `evaluate` | `--split` (test), `--device` | checkpoints, Zarr | `metrics/metrics_glorys.json`, `maps_glorys.nc` |
| `validate-argo` | `--split` (test), `--device` | checkpoints, Zarr, Argo (downloads missing months) | `metrics/metrics_argo.json`, `argo_matchups.parquet` |
| `report` | `--date` (middle of test) | metrics files, Zarr, predictions, logs | `figures/*.png`, `report.md` |
| `run-all` | `--skip-existing`, `--device` | config | everything above |
| `research r1` | `--seeds`, `--methods`, `--mlp-seeds`, `--skip-existing/--no-skip-existing`, `--device` | config, Zarr, stats, `ridge.joblib` | `research/r1/<method>/seed<k>/` only |
| `research r1-report` | `--n-boot`, `--block-length`, `--seed` | `research/r1/**/eval.npz` | `research/r1/summary.{json,md}`, `figures/` |
| `research r4` | `--download/--no-download`, `--recompute`, `--n-boot`, `--block-length`, `--seed` | config, predictions, Zarr, stats, `argo_matchups.parquet`, CMEMS login (download only) | `data/raw/armor3d/`, `data/processed/<run>_armor3d/`, `research/r4/` |
| `research r5` | `--recompute`, `--mlp-seed`, `--n-boot`, `--block-length`, `--seed` | config, predictions, Zarr, stats, R1 `mlp.pt` (optional) | `research/r5/` only |
| `research r3` | `--seeds`, `--mlp-seeds`, `--models`, `--experiments`, `--permutation/--no-permutation`, `--perm-repeats`, `--skip-existing/--no-skip-existing`, `--device` | config, Zarr, stats, R1 `scratch` / `mlp` jobs | `research/r3/<model>_<experiment>/seed<k>/`, `research/r3/perm_<model>/seed<k>/` only |
| `research r3-report` | `--n-boot`, `--block-length`, `--seed` | `research/r3/**`, R1 `scratch` / `mlp` / `climatology` | `research/r3/summary.{json,md}`, `figures/` |
| `research r2` | `--seeds`, `--methods`, `--learning-curve/--no-learning-curve`, `--argo/--no-argo`, `--skip-existing/--no-skip-existing`, `--device` | long config, Zarr, stats, stored Argo months | `research/r2/<name>/seed<k>/`, `research/r2/argo/`, `data/processed/cache/<run>/` |
| `research final-inputs` | `--seeds`, `--experiments sst_sla_winds,sst_sla`, `--skip-existing/--no-skip-existing`, `--device` | long config, Zarr, stats, R2 `scratch` jobs | `research/final_inputs/scratch_<set>/seed<k>/` only |
| `research final-inputs-report` | `--n-boot`, `--block-length`, `--seed` | `research/final_inputs/**`, R2 `scratch` / `climatology` | `research/final_inputs/summary.{json,md}` |
| `research r2-report` | `--compare-config`, `--n-boot`, `--block-length`, `--seed` | `research/r2/**`, the compared run's `research/r1/**` (read only) | `research/r2/summary.{json,md}`, `figures/` |
| `research benchmark` | `--seeds`, `--families rf,gbt,unet,mlp,ridge`, `--sets all7,sst_sla`, `--skip-existing/--no-skip-existing`, `--threads`, `--device` | long config, Zarr, stats, memmap caches | `research/benchmark/<family>_<set>/seed<k>/`, `research/benchmark/tuning/` only |
| `research benchmark-report` | `--n-boot`, `--block-length`, `--seed` | `research/benchmark/**`, `research/r2/**`, `research/final_inputs/scratch_sst_sla` | `research/benchmark/summary.{json,md}`, `figures/` |
| `live update` | `--config configs/live.yaml`, `--verify/--no-verify`, `--device` | catalogue, CMEMS login, `models/final`, Argo (verification) | `data/raw_nrt/`, `data/processed/live.zarr`, `outputs/live/` |
| `live status` | `--config`, `--check` (ask the catalogue) | `outputs/live/` (local files) | nothing |
| `live input-shift` | `--config`, `--download/--no-download`, `--recompute`, `--device` | reprocessed + NRT products, GLORYS, `models/final` | `data/raw_nrt/overlap_*`, `data/processed/live_shift*`, `outputs/live/checks/input_shift/` |
| `serve` | `--host` (127.0.0.1), `--port` (8000), `--outputs-root`, `--reload` | `outputs/` (exit 2 if missing) | serves the API; sets `OCEANEMBED_API_OUTPUTS_ROOT` for `--reload` |

`predict` argument rules: `--start` and `--end` together, not with `--split`; `--ridge` and `--tag` exclusive; tag `ridge` rejected.

**`run-all` order** (each step calls its own CLI function in-process; the chain stops at the first failure with a named
step and a non-zero exit code; prints a per-step timing summary; frees GPU memory between steps):

1. `synth` (provider synthetic) or `download` (real; all 7 products incl. Argo for the whole config period)
2. `harmonize`  3. `stats`  4. `pretrain`  5. `train`
6. `train-ablation` (if `ablation.no_pretrained`)
7. `baseline`  8. `embed` (test)  9. `predict` (test)  10. `predict-ridge`
11. `predict-ablation` (if `ablation.no_pretrained`)
12. `evaluate`  13. `validate-argo`  14. `report`

That is 14 steps with the ablation, 12 without. `--skip-existing` skips a step whose output files exist (it does not
detect stale outputs); the `download` check needs every monthly file of every product, Argo included.

### Research commands (stage R1)

`oceanembed research r1` and `research r1-report` (usage: [`usage.md`](usage.md)). Code in `research/`. Nothing outside
`outputs/<run>/research/r1/` is written (they load the config only; `run_meta.json` is not rewritten).

```
outputs/<run>/research/r1/
  <method>/seed<k>/            method in oceanembed | scratch | unet | mlp ; ridge, climatology use seed0
    pretrain.pt, pretrain.jsonl, pretrain.done.json     oceanembed only
    recon.pt (mlp.pt), train.jsonl, train.done.json     trained methods
    eval.npz                   per-day sufficient statistics of the test split vs GLORYS
    done.json                  written last: provenance, timings, parameter counts, best epochs
  summary.json, summary.md, figures/{rmse_by_depth,paired_differences,skill_by_depth}.png
```

- **Methods.** `oceanembed`: pretraining then fine-tuning, both with the seed. `scratch`: the same network with the encoder trained
  from scratch. `unet`: `model.arch = unet`, i.e. the same stem, embedding projection and decoder with the 6 Transformer blocks
  replaced by 4 residual conv blocks of width 192 at H/4 (3 731 055 parameters vs 3 741 423; the decoder is identical at 537 839),
  no pretraining. `mlp`: 11 features (the ridge features) to 15 depths, 3 hidden layers of 256 with SiLU (138 511 parameters), AdamW
  1e-3, batch 4096, up to 30 epochs, early stopping (patience 5) on the RMSE of a fixed 200 000-point validation sample, trained on
  1 000 000 random train (day, pixel) samples chosen by the seed; loss over the depths valid at each sample. `ridge`,
  `climatology`: loaded from the main run, scored once.
- **Evaluation file.** For the whole domain and each basin, raw and climatology-removed, the 8 sums of `eval/metrics.py`
  (`n, sx, sy, sxx, syy, sxy, sae, se2`) per test day and depth: `raw` and `anom` of shape `(region, field, day, depth)`, about 2 MB
  per job. Same valid sample and de-normalisation as `evaluate`; the report asserts all jobs share the same `n`, and tests check that
  ridge and climatology reproduce `metrics_glorys.json`.
- **Resumability.** A job with `done.json` is skipped. A job without it restarts at its first stage lacking a `*.done.json` marker
  (partial files of the unfinished stage are deleted first); scoring always reruns. `--no-skip-existing` also retrains.
- **Memory.** The runner preloads the train and validation splits once (about 7 GB for the real run) and reuses them for every
  training in the process; run nothing else heavy beside it.
- **Statistics** ([`r1_report.py`](../src/oceanembed/research/r1_report.py), [`bootstrap.py`](../src/oceanembed/research/bootstrap.py)).
  Per method: seed mean / SD (n - 1) / min / max of every metric (RMSE, bias, MAE, raw and anomaly correlation, skill) for each
  depth, the pooled 50-200 m range and all depths, for the domain and each basin. Test-day uncertainty: the per-day sums are averaged
  over seeds, then a moving-block bootstrap (overlapping blocks, no wrap-around, `ceil(T/L)` blocks truncated to `T` days, 2000
  replicates) resamples days; metrics are recomputed from the resampled sums and the 95 % interval is the 2.5-97.5 percentile range.
  Replicates are stored as day-count vectors, so one set of replicates is shared by all methods, depths, basins and metrics. Paired
  comparison `A - B`: formed inside each replicate (same days for both), with the percentile interval, whether it excludes 0, a
  two-sided bootstrap p-value, and, across seeds, whether the seed ranges overlap and a Welch t-test.
- **Block length.** Chosen from the data: the median over methods of the e-folding time of the autocorrelation of the daily pooled
  50-200 m MSE, capped at a quarter of the period (`--block-length` overrides). `summary.md` lists the e-folding and integrated
  autocorrelation times and how the interval half-widths change with the block length (1, 7, 15, 30, 45 days).

### Research commands (stages R4 and R5)

`oceanembed research r4` and `research r5` (usage: [`usage.md`](usage.md); results: [`research/r4_armor3d.md`](research/r4_armor3d.md),
[`research/r5_physical.md`](research/r5_physical.md)).

- **R4.** Downloads ARMOR3D (`cmems_obs-mob_glo_phy_my_0.125deg_P1D-m`, variable `to`, daily, 1993-01-01 to 2024-12-31, so the whole
  test year comes from one dataset) with the same subset-per-month / skip-existing / ledger pattern as the other CMEMS products;
  credentials are only checked when a month is missing. Each month is regridded with `harmonize.process_variable` (vertical linear
  to the 15 depths, then 2 x 2 block mean) on the product's own days into `data/processed/<run>_armor3d/`. The Argo scoring adds
  an `armor3d` column to the stored `validate-argo` matchups (same cell, day and depth) and keeps only rows where ARMOR3D is
  defined, so all six products are scored on identical rows; the grid pass streams the days and stores per-day, per-depth sums of
  every product against GLORYS and against ARMOR3D on the common sample (`grid_sums.npz`). The report is a block bootstrap over days.
- **R5.** `run_r5` streams the test split once (GLORYS, the three prediction products, the climatology and, if the R1 checkpoint
  exists, the per-pixel MLP run on the CPU), accumulating per-day sums of the derived quantities (definitions in
  [`physical.py`](../src/oceanembed/research/physical.py)) on a common sample and of the pooled 50-200 m temperature per stratum
  (`sums.npz`); the report turns them into tables and figures. Memory stays at a few fields.
- Neither stage writes under `checkpoints`, `metrics`, `predictions`, `embeddings`, `report.md` or `research/r1`.

### Research commands (stages R3 and R2)

`oceanembed research r3` / `r3-report` and `research r2` / `r2-report` (usage: [`usage.md`](usage.md)). Both runners follow the R1
conventions: one folder per job with `done.json` written last, `eval.npz` sufficient statistics (`raw`, `anom` of shape
`(region, field, day, depth)`), skip-finished / resume-at-the-first-unfinished-stage, block bootstrap with shared replicates,
"established" = paired interval excludes 0 **and** the per-seed RMSE ranges do not overlap. Scoring is `r3.score_passes`: several
passes over the test split with the reference and climatology of a batch read once, and the per-region sums of the eight fields
computed with one matrix product per field (checked equal to `sums_from_arrays`).

- **R3 input groups.** `inputs.InputSpec` = the variable groups kept (`sst`, `sss`, `sla`, `currents` = u + v, `winds` = u + v) and
  the history length `k`. A group that is not kept is zeroed in every channel of that variable (target day and every lag); the
  network layout and initialisation are unchanged, so the R1 model of the same seed is the exact full-input reference (a test
  shows the identity view trains bit-identically on the CPU). History adds, after the 12 channels, the 7 surface fields of day
  `t-1`, `t-2`, ... (`12 + 7 (k - 1)` channels; `model.in_channels` is set per job and stored in the checkpoint). Training reads
  the preloaded arrays through `InputView` (indexing, no copy); the first `k - 1` days of the training and validation splits are
  dropped. Scoring builds the identical input for the test split (`make_test_batch_fn`), taking the days before the split from the
  store (inputs only), so every `k` is scored on the same days. The per-pixel MLP gets the ridge features plus the lagged surface
  values (`PixelMLP(channels=...)`, `run_train_mlp(sample_fn=..., model_kwargs=...)`); its sample of training points is the same
  as the full-input MLP's.
- **R3 permutation.** One pass per (group, repeat) plus an unpermuted pass from the same code; the permutation draws, for every
  calendar month, a random derangement of the days and replaces *all channels of the group* by those of the donor day. Permuting
  within the month keeps the inputs realistic for the season (a global shuffle would put July sea level next to January SST and
  inflate the importance with the seasonal cycle that the day-of-year channels already carry) and measures the information in the
  sub-monthly variations beyond the seasonal mean state, which is also why it can disagree with retrain-without. Repeats are
  averaged at the level of the sums.
- **R2 on-disk cache.** `OceanDataset.use_cache` writes the four standardised arrays (`surf`, `sv`, `y`, `valid`) once to
  `data/processed/cache/<run>/<split>_<start>_<end>_<dtype>/*.npy` and reads them with `np.memmap` by day (`train.cache: memmap`).
  The folder is built as `<name>.tmp` and renamed after `meta.json` (with the dtype's round-off, the statistics fingerprint, the
  split and the store identity) is written, so a partial build is never used and a missing or stale cache is rebuilt.
  `gather(t, p)` serves the ridge / MLP point samples day by day from the maps; `tail(n)` is a no-copy view of the last `n` days
  (learning curve). `ram` (the default) is untouched.
- **R2 jobs.** `scratch`, `mlp` (3 seeds), `ridge` (fitted on the long train split, saved in its job folder), `climatology`, the
  learning-curve jobs `scratch_ty2` / `scratch_ty5` (the headline model on the last 2 / 5 years; normalisation and climatology stay
  those of the full period) and `argo` (headline model seed 0, ridge, climatology and GLORYS at the stored Argo profiles of both
  test years; never downloads). Every job scores the whole test split; the report slices it by year.
- Neither stage writes under `checkpoints`, `metrics`, `predictions`, `embeddings`, `report.md` or any other stage's folder
  (R2 reads the first run's `research/r1` for the 2024 comparison).

### Live nowcast (phase 11: `oceanembed live ...`)

Reconstructs the same day from near-real-time (NRT) inputs with the released `models/final` weights, keeping a rolling
window of the newest days. A **nowcast**, never a forecast: the inputs are daily maps published a day or two late.
Details and the first measured numbers: [`research/live_nowcast.md`](research/live_nowcast.md); operation:
[`runbook.md`](runbook.md) (Live mode). Nothing here touches an evaluated run.

- **Inputs.** The model uses SST and sea level anomaly only, so only those have NRT products in `configs/live.yaml`
  (OSTIA NRT `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2`, DUACS NRT `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D`);
  `required_products` refuses a model that needs an input without an NRT product. Salinity, currents and winds are not fetched.
- **Availability.** A day is available when every input has a non-empty file for it; the window is `live.window_days` days ending
  at the newest such day and never moves backwards. Days one product already has are reported as `pending`.
- **Download.** One `copernicusmarine.subset` per run of consecutive days (at most `live.request_days`), split into per-day files
  `data/raw_nrt/<product>/<product>_YYYYMMDD.nc` (temp file + rename, ledger line per request, retries). Files for days that left
  the window are deleted; nothing outside `data/raw_nrt`, `data/processed/live*` and `outputs/live` is ever pruned.
- **Update** (`live update`, under a lock file): probe the catalogue, fetch missing days, fetch the newest `live.revision_days`
  reconstructed days again (`revision=True`: a file is replaced only if the field changed), harmonise the window with the batch
  rules into `data/processed/live.zarr`, reconstruct new, changed and missing days, merge them into the monthly files, rewrite the
  provenance table, prune, write state and `run_meta.json` (`data_source: "real"`, block `live: {nrt: true, window, last_day, ...}`),
  then run the verification. An interrupted update is continued by the next one (everything is rebuilt from the day files).
- **Provenance** (`live_days.parquet`, one row per day): dataset and version of each input, product version, age of each input
  when first used, first / last update time, number of checks and revisions, size of the revision against the first-published
  version (`rev_sst_rmse`, `rev_sla_rmse`, `rev_recon_rmse_50_200`, `rev_recon_maxabs`), input digest, checkpoint, device.
  `first_published/first_YYYYMMDD.npz` keeps the first-published inputs and reconstruction of each window day for that comparison.
  `live_state.json` holds the window, per-input availability, pending days, the revision log and its statistics by age, the history
  of updates (timings, bytes).
- **Verification** (`checks/verification/`): Argo profiles fetched for the window (newest 15 days re-fetched, older kept), collocated
  and scored like `validate-argo` for model and climatology in three bands (0-30, 50-200, 300-1000 m); optionally the operational
  analysis (`GLOBAL_ANALYSISFORECAST_PHY_001_024`, daily temperature) on the grid for the newest `rolling_days` days when the
  dry-run size estimate is below `analysis_max_mb`, otherwise skipped with the reason recorded. Stored as daily sums, so the
  30-day rolling RMSE / bias is one sum; a verification failure never undoes the update.
- **Input shift** (`live input-shift`): see [`research/live_nowcast.md`](research/live_nowcast.md); sums per day, depth and
  basin in `sums.npz`, then the research block bootstrap for the paired change.

### Comparison study (phase 13: `oceanembed research benchmark`)

[`benchmark.py`](../src/oceanembed/research/benchmark.py) adds the families of the literature under the R2 protocol (same data,
split, GLORYS reference, per-day `eval.npz` sums): per-pixel **boosted trees** (LightGBM, one regressor per depth, rounds by early
stopping on the validation sample) and a **random forest** (scikit-learn, one multi-output forest, depths below the sea floor filled
with the climatological anomaly 0), both with the 11 per-pixel features of the MLP and its training sample
(`sample_points`: up to 1 M random train points, the seed draws them and seeds the model); and the **plain U-Net** on all seven
inputs. Hyper-parameters come from small grids (`GBT_GRID` num_leaves 15 / 63 / 255; `RF_GRID` min_samples_leaf 5 / 20 / 60 x
max_features 0.5 / 1.0) scored on the validation year only, on a smaller training sample, stored in `research/benchmark/tuning/`.
Input sets: all seven inputs (beside the R2 Transformer, MLP, ridge, climatology) and SST + sea level (beside the final model of
`research/final_inputs`; ridge and MLP are trained here for that set). Boosted-tree models are stored (`gbt.joblib`); the forest is
not (large, refits in minutes). [`benchmark_report.py`](../src/oceanembed/research/benchmark_report.py) ranks every family by
pooled 50-200 m RMSE for 2023, 2024 and both, by basin and depth, with the paired block bootstrap, the Transformer against each
family and the best per-pixel family against the Transformer per basin, and a winner or tie (a difference is *established* only
if the paired interval excludes 0 and the per-seed ranges do not overlap). Results: [`research/benchmark.md`](research/benchmark.md).

## 10. Data API

Full payloads: [`api.md`](api.md); interactive docs at `/api/docs` (`/api/openapi.json`, `/api/redoc`). Start:
`oceanembed serve` -> `http://127.0.0.1:8000`. With `web/dist/index.html` built, the same server serves the React app at `/`
with a client-side-route fallback that never shadows `/api`.

**Conventions**
- Base `http://127.0.0.1:8000/api`; every route is `GET`; no authentication; localhost by default.
- NaN / inf are `null` ([`encode.py`](../src/oceanembed/api/encode.py)); float arrays are rounded to 3 decimals in float64 first.
- Dates are `YYYY-MM-DD`. A date outside the predicted range is 400; a gap inside it is 404.
- Depth: `depth` (metres, must equal a grid depth) or `depth_index` (0-based), exactly one; separate parameters because
  `depth=5` would be ambiguous between 5 m and index 5. Points `lat`/`lon` must lie inside the grid edges and snap to the
  nearest cell centre.
- Binary volumes: `format=f32` or `Accept: application/octet-stream` returns raw little-endian float32, C order, NaN preserved
  (`15*100*240*4 = 1 440 000` bytes per volume); metadata in readable `X-Shape`, `X-Dtype`, `X-Byte-Order`, `X-Kind`,
  `X-Method`, `X-Date`, `X-Depth-Index`, `X-Color-Range`, `X-Color-Diverging`, `X-Color-Range-Per-Depth`, `X-Has-Target`.
- Caching: run endpoints send `ETag` (hash of run name, `run_meta.json` mtime, path, query, `Accept`) and
  `Cache-Control: private, max-age=60, must-revalidate`; `If-None-Match` gets 304 before any data is read. `/api/health`,
  `/api/runs`, `/api/compare` are `no-store`. In-process byte-bounded LRU of opened days / series: 768 MB. The API never
  writes into a run folder.
- Errors: always `{"detail": "..."}`. 400 bad parameter (**including FastAPI validation errors, which are mapped from 422 to 400**), 404 unknown
  run / file / method or missing artefact (with a hint naming the missing step), 405 non-GET, 500 generic message (no
  stack trace). Run names must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$` and be a direct child of the outputs root with a
  `run_meta.json`; file / figure names are accepted only if present in the directory listing.
- CORS: only `http://localhost:5173` and `http://127.0.0.1:5173` (`GET`, `HEAD`, `OPTIONS`; the `X-*` headers are exposed). GZip above 1 kB.
- `method=` (default `model`) on `/fields`, `/profile`, `/section`, `/timeseries`, product download: `model`, `ridge`, `model_<tag>`; a
  method without a product is 404 listing the available ones. `climatology` and `glorys` are not methods here (`kind=`).
- A start-up thread warms Zarr handles, statistics, the embedding PCA and the error maps; runs that are still being
  produced never give 500 (truncated product files are skipped).

**Endpoints** (29 routes plus the static / SPA fallback)

| Method | Path | Purpose | Key parameters |
|---|---|---|---|
| GET | `/api/health` | liveness, version, `n_runs`, `ui_built` | - |
| GET | `/api/runs` | summary of every run | - |
| GET | `/api/runs/{run}` | grid, basins, methods, field methods, config summary, products, counts | - |
| GET | `/api/runs/{run}/dates` | days with prediction / target / embedding | - |
| GET | `/api/runs/{run}/mask` | ocean mask per depth, packbits base64 | - |
| GET | `/api/runs/{run}/live` | live run only (404 otherwise), never cached: window, input freshness, pending days, provenance, revisions, input shift, running verification | - |
| GET | `/api/runs/{run}/fields` | 2-D field or 15-level volume | `date`, `kind` (`prediction`, `target`, `climatology`, `difference`, `anomaly_pred`, `anomaly_target`), `depth` / `depth_index`, `format`, `method` |
| GET | `/api/runs/{run}/ranges` | period-wide colour ranges per depth | `samples` (24, 2..60) |
| GET | `/api/runs/{run}/surface` | the 7 input fields of a day, physical units | `date` |
| GET | `/api/runs/{run}/profile` | vertical profile at the nearest cell | `date`, `lat`, `lon`, `method` |
| GET | `/api/runs/{run}/section` | depth x distance section | `date`, one of `lat` / `lon`, `method` |
| GET | `/api/runs/{run}/timeseries` | time x depth at a point + RMSE / bias by day | `lat`, `lon`, `method` |
| GET | `/api/runs/{run}/embeddings/dates` | days with an embedding (test split) | - |
| GET | `/api/runs/{run}/embeddings` | PCA components of the embedding map, 0..1 | `date`, `n_components` (3, 1..8) |
| GET | `/api/runs/{run}/embeddings/similar` | cosine similarity of one cell to all cells | `date`, `lat`, `lon`, `center` |
| GET | `/api/runs/{run}/metrics/glorys` | metrics vs GLORYS, metric-major | - |
| GET | `/api/runs/{run}/metrics/argo` | metrics vs Argo | - |
| GET | `/api/runs/{run}/metrics/maps/index` | which metric / method maps exist | - |
| GET | `/api/runs/{run}/metrics/maps` | one error map at one depth | `metric`, `method`, `depth` / `depth_index` |
| GET | `/api/runs/{run}/argo/matchups` | columnar matchups, deterministic down-sampling | `max_points` (20000, 100..1e6) |
| GET | `/api/runs/{run}/argo/profiles` | profile list with per-method RMSE: filter, sort, page | `basin`, `start`, `end`, `lat_min/max`, `lon_min/max`, `sort`, `order`, `method`, `limit` (5000, max 50000), `offset`, `around` |
| GET | `/api/runs/{run}/argo/profiles/{profile_id}` | one profile vs every method | - |
| GET | `/api/runs/{run}/training` | training / pretraining curves from the logs | - |
| GET | `/api/runs/{run}/experiments` | methods / ablation comparison table | - |
| GET | `/api/compare` | experiment tables of up to 8 runs | `runs` (comma-separated) |
| GET | `/api/runs/{run}/report` | `report.md` text and figure list | - |
| GET | `/api/runs/{run}/figures/{name}` | one PNG | - |
| GET | `/api/runs/{run}/product` | product file list incl. ridge / ablation products | - |
| GET | `/api/runs/{run}/product/{file}` | download one monthly NetCDF | `method` |

`kind` for `difference` is prediction minus target; temperature colour ranges always come from the main model and the
target (1st-99th percentile) so panels of different methods share a scale; diverging ranges are symmetric about 0.

## 11. Configuration

Finalisation additions (all optional; defaults reproduce the earlier behaviour): `model.input_groups` (subset of `sst sss sla currents winds`; dropped groups are zeroed in the dataset's inputs for every consumer), `model.main_init` (`pretrained` | `scratch`: how `recon.pt` is made), `ablation.pretrained` (the pretrained variant of a scratch config as `recon_<tag>.pt`), `baseline.mlp` (fit and publish the per-pixel MLP), `paths.store` (reuse another run's harmonised store, statistics and array cache; `harmonize` / `stats` refuse to rebuild it, `run-all` checks it in a `store` step). `configs/final.yaml` uses all of them. Metrics files gain `metadata.inputs`, `metadata.years` and, for a multi-year split, a `per_year` block.

One YAML per run, validated by [`config.py`](../src/oceanembed/config.py) (`extra="forbid"`: an unknown key is an error; splits must lie
inside `time`, be ordered and not overlap). Relative `paths` resolve against the current directory, so run from the
project root.

| Section | Keys that matter (defaults) |
|---|---|
| top level | `run_name` (also the output / store name), `label`, `description` (display text for the API), `provider` (`synthetic` default, or `real`) |
| `paths` | `data_root` (`data`), `outputs_root` (`outputs`), `raw_dir` (`raw`), `processed_dir` (`processed`); env vars override the two roots |
| `grid` | `lat_min/max` 5 / 30, `lon_min/max` 45 / 105, `resolution` 0.25, `depths` (15 standard); H and W divisible by 4 |
| `time`, `split` | `time.start/end`; `split.train/val/test` as `[start, end]` |
| `products` | per product: `provider` (`cmems` / `podaac`), `datasets` (`id`, optional `start`/`end`), `variables` (canonical -> raw name), `coords`, `min_depth`/`max_depth`; a partial override merges onto `default_products()` |
| `download` | `halo_deg` 0.5, `overwrite` false, `podaac_mode` `opendap`, `workers` 4, `retries` 4, `retry_backoff_s` 2.0, `delete_global_granules` true |
| `argo` | `source` `erddap` (or `gdac`), `max_depth_m` 1100, `qc_flags` [1, 2], `box_deg` 15 |
| `harmonize` | `time_chunk` 8, `min_valid_fraction` 0.5 |
| `model` | `in_channels` 12, `emb_dim` 128, `dim` 192, `depth` 6, `heads` 6, `mlp_ratio` 4.0, `stem_channels` 64, `n_depths` 15, `arch` `transformer` (`unet` = residual convolutions instead of attention, research R1) |
| `pretrain` | `epochs` 20, `batch_size` 8, `lr` 3e-4, `weight_decay` 0.05, `warmup_epochs` 1, `grad_clip` 1, `mask_ratio` 0.5, `block` 16, `amp`, `seed` 0, `crop` null |
| `train` | `epochs` 30, `batch_size` 8, `lr` 3e-4, `encoder_lr_scale` 0.1, `weight_decay` 0.01, `warmup_epochs` 1, `grad_clip` 1, `vertical_grad_weight` 0.05, `patience` 8, `amp`, `seed` 0, `crop` null, `num_workers` 0, `cache` `ram` (`memmap` = on-disk array cache for periods that do not fit in RAM), `cache_dtype` `float16` (memmap cache only) |
| `baseline` | `ridge_alpha` 1.0, `ridge_max_points` 1 000 000, `seed` 0 |
| `mlp` | `hidden` 256, `layers` 3, `max_points` 1 000 000, `val_points` 200 000, `epochs` 30, `batch_size` 4096, `lr` 1e-3, `weight_decay` 1e-4, `patience` 5 (research R1 only) |
| `ablation` | `no_pretrained` false, `tag` `scratch` |
| `synthetic` | `seed` 7, `argo_profiles_per_month` 150, `native_resolution`, `n_eddies_per_100_deg2_year` |

Shipped configs:

| Config | Provider / data | Period and split | Purpose |
|---|---|---|---|
| `final.yaml` | real; reads the store, statistics and array cache of `poc_long` (`paths.store`) | 2011-01-01 .. 2024-12-15; train 2011-21 / val 2022 / test 2023 + 2024 | **the main run**: trained from scratch, inputs SST + sea level, per-pixel MLP and ridge baselines, pretrained variant as the ablation |
| `poc_long.yaml` | real / `raw` | as `final` | builds the long-period data (download, harmonise, statistics); research R2, input selection, benchmark |
| `poc.yaml` | real / `raw` | 2018-01-01 .. 2024-12-15; train 2018-22 / val 2023 / test 2024 | the first real run (pretrained model, all seven inputs); research R1, R3, R4, R5 |
| `poc_trial.yaml` | real / `raw` | 2024-01-01 .. 2024-03-31; 46 d / 14 d / 31 d | plumbing check only (mean-only climatology) |
| `live.yaml` | real, near-real-time / `raw_nrt` | rolling 60-day window ending at the newest available day | the live nowcast with `models/final`; never trains or evaluates |
| `predict_surface.yaml` | real, surface products only | the days you ask for | prediction from the released weights without the training data (`harmonize --surface-only`, `predict --weights models/final`) |
| `synthetic.yaml` | synthetic / `raw_synthetic` | 2021-01-01 .. 2023-12-31; train 2021-22 / val H1 2023 / test H2 2023 | pipeline demo without logins, not skill |
| `test_tiny.yaml` | synthetic, 16 x 24 grid (8-12 N, 72-78 E) | 2022-01-01 .. 2022-03-01; 40 d / 10 d / 10 d | pytest (tiny model: `emb_dim 16`, `dim 32`, `depth 2`) |
| `test_tiny_final.yaml` | synthetic, tiny grid | about 90 days across a New Year | pytest for what `final` uses: scratch main model, reduced inputs, MLP baseline, per-year metrics |

Grid 100 x 240 and the default model (3.7 M parameters) everywhere except the two test configs.

## 12. Testing and quality

`.\.venv\Scripts\python.exe -m pytest` runs **343 tests** in 20 files; all pass (last run 2026-10-04).

| File | Tests | What it covers |
|---|---|---|
| `test_api.py` | 40 | every endpoint, validation and error shape, 304 / ETag, binary format, unfinished / partial runs (no 500), CORS, OpenAPI coverage, SPA fallback |
| `test_providers.py` | 23 | month clipping, dataset-by-date, CMEMS request args and ledger, credentials, Argo tidy / retry / NetCDF-3 fallback, INCOIS loader, CLI errors |
| `test_pipeline.py` | 22 | synthetic raw layout, Zarr structure, mask, regrid against truth, harmonic fit (leap years), dataset contract, dataloader workers |
| `test_research_r3.py` | 22 | input groups and masking, history channels, within-month permutation, resumable jobs, multi-pass scoring |
| `test_final_stage.py` | 21 | scratch main model, input-group selection, shared store, MLP baseline, per-year metrics, results and weights export, prediction from released weights |
| `test_models.py` | 21 | shapes, positional embeddings, block mask, no-leakage test, masked losses, parameter groups, overfit sanity checks |
| `test_research.py` | 19 | seed reproducibility, plain U-Net and per-pixel MLP, block bootstrap against hand-computed cases, paired differences, decorrelation time, R1 runner (resumable) and report |
| `test_podaac.py` | 18 | OPeNDAP index runs, DAP4 constraints, retry back-off, resume, atomic parts, fallback and switch-off, no secrets in ledger |
| `test_products.py` | 18 | one full `run-all` on the tiny config: every contract file, `--skip-existing`, stop-on-failure, NetCDF structure, metrics vs direct numpy, Argo matchups, report |
| `test_live.py` | 18 | window and revision rules, update lock, cold start and incremental update (only new days fetched, only live folders pruned), interrupted update continued, provenance, verification sums, input-shift statistics, the live endpoint and CLI |
| `test_regrid.py` | 17 | block mean, bilinear, NaN rules, daily mean, vertical interpolation |
| `test_config.py` | 17 | shipped configs load, validation rules, env overrides, grid, basins, ablation keys |
| `test_research_r4_r5.py` | 14 | isotherm depth and heat content against analytic profiles, strata, ARMOR3D request and regridding, identical samples for every method |
| `test_research_r2.py` | 13 | long-period jobs, learning-curve subsets, per-year slicing, Argo scoring of two test years |
| `test_dashboard.py` | 13 | the Streamlit app and its data layer (not the React app) |
| `test_argo_validation.py` | 11 | interpolation rule, gap tolerance, no extrapolation, collocation, basins, gridded plan |
| `test_metrics.py` | 10 | streaming equals one-shot, merge, reductions, NaN and constant series, skill |
| `test_benchmark.py` | 9 | boosted-tree and random-forest predictors (round trip, land and sea-floor handling), job plan, ranked report with winner or tie, refusal of runs scored on different samples |
| `test_training.py` | 9 | preload equals lazy, ridge, climatology baseline, `predict_batch`, CLI chain, crop |
| `test_array_cache.py` | 8 | memmap array cache: atomic build, fingerprint and staleness, round-off bound, day-wise gather |

All tests use temporary data / outputs roots (`OCEANEMBED_DATA_ROOT`, `OCEANEMBED_OUTPUTS_ROOT`) and synthetic data; none touch
the network. `conftest.py` provides a session-wide `synth -> harmonize -> stats` fixture and one tiny `run-all` on CPU.
The React app has its own tests (`npm test`, Vitest: 203 tests in 15 files).

- Lint ([`pyproject.toml`](../pyproject.toml)): ruff, line length 100, target py312, rules `E, F, I, UP, B`; `E501` ignored under
  `src/oceanembed/api/*`; bugbear treats `fastapi.Depends` / `Query` as immutable defaults.
- Pytest: `testpaths = tests`, `-q`, `pythonpath = ["."]` (so tests import `app.data_access`), Deprecation / Future / User warnings ignored.

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
```

## 13. Measured results

Main run `final` (trained 2011-2021, from scratch, inputs SST + sea level), test 2023-01-01 .. 2024-12-15 (715 days), pooled over
50-200 m. Source: [`results/final/`](../results/final/) (`metrics_glorys.json`, `metrics_argo.json`) and
[`models/final/MODEL_CARD.md`](../models/final/MODEL_CARD.md). The released model is a single training run; the seed spread of the same
configuration is about 0.003 degC (three seeds, [`research/final_inputs.md`](research/final_inputs.md)).

**Against GLORYS** (RMSE degC)

| Method | 2023 | 2024 | Both years | Anomaly corr. | Skill vs climatology |
|---|---|---|---|---|---|
| OceanEmbed | 0.974 | 0.991 | 0.983 | 0.73 | 0.55 |
| Per-pixel MLP | 0.984 | 1.043 | 1.013 | - | 0.52 |
| Ridge regression | 1.280 | 1.149 | 1.218 | 0.53 | 0.30 |
| Climatology | 1.519 | 1.394 | 1.459 | - | 0 |

**Against Argo** (5 512 profiles, 77 388 matchups, both years; pooled 50-200 m RMSE degC)

| OceanEmbed | Per-pixel MLP | Ridge | Climatology | GLORYS itself (the floor) |
|---|---|---|---|---|
| 1.322 | 1.291 | 1.375 | 1.516 | 1.051 |

- RMSE by depth, model / climatology, both years: 0 m 0.47 / 0.81; 30 m 0.64 / 0.94; 100 m 1.12 / 1.76; 200 m 0.74 / 0.93;
  300 m 0.50 / 0.53; 500 m 0.345 / 0.345; 1000 m 0.37 / 0.36.
- By basin (50-200 m RMSE): Arabian Sea 1.002, Bay of Bengal 0.949 (climatology 1.350 and 1.649); skill 0.45 and 0.67.
- Data: 5 098 gap-free harmonised days; about 29 GB of raw downloads, of which GLORYS is 23 GB ([`reproduce.md`](reproduce.md)).

**Model families under one protocol** (eleven training years, all seven inputs, three seeds, both test years;
[`research/benchmark.md`](research/benchmark.md)): CNN + Transformer 0.989 +/- 0.010, boosted trees 0.990 +/- 0.002, per-pixel MLP
1.004, plain U-Net 1.011, random forest 1.035, ridge 1.207, climatology 1.459 degC. The Transformer and boosted trees tie over the
whole domain; the Transformer is best in the Bay of Bengal (0.938 vs 1.001), boosted trees in the Arabian Sea (0.983 vs 1.018).

**Live nowcast** ([`research/live_nowcast.md`](research/live_nowcast.md)): over 182 overlap days the error against GLORYS is
1.026 degC with near-real-time inputs and 1.036 with reprocessed ones (50-200 m). First 30-day running check, model / climatology:
against Argo 0.96 / 1.26 (0-30 m), 1.43 / 1.43 (50-200 m), 0.33 / 0.32 (300-1000 m); against the operational analysis 0.75 / 1.06,
1.24 / 1.53, 0.36 / 0.37.

**The earlier five-year run `poc`** (test year 2024, pretrained model, all seven inputs): 1.08 degC against GLORYS (from scratch
1.04, ridge 1.15, climatology 1.39) and 1.41 against 2 826 Argo profiles (GLORYS itself 1.08). The other research stages are in
[`research/`](research/).

**Performance numbers that exist in the docs** (not re-measured here)

| Item | Value | Source |
|---|---|---|
| Training, `poc` run | pretraining 862 s (best epoch 18), supervised 683 s (best epoch 6 of 14, 44 s/epoch), no-pretraining ablation 605 s (best epoch 4 of 12) on a 6 GB GPU (RTX 3050) | training logs of the `poc` run |
| Training per seed, eleven years | Transformer 19 min (GPU), plain U-Net 30 min (GPU, machine shared), boosted trees 5 min + about 9 min to score (CPU), random forest 4 min, ridge 3 min, per-pixel MLP 18 s | research/benchmark.md |
| Live update | cold start 22 MB and about 2 min on the GPU (0.03 s per day; 0.52 s per day on a shared CPU); daily update about 13 MB, 1.5-3 min | research/live_nowcast.md |
| GPU memory | ~1.3 GB peak at batch 8, fp16 (synthetic runs) | decisions.md |
| Epoch time, synthetic full grid | 17-18 s (730 train days) | runbook.md |
| Full `run-all` synthetic, from scratch | 2 547 s | phases.md |
| Pre-run estimate for the real run after download | 1.5-2 h in total | runbook.md (an estimate, scaled from synthetic) |
| Harmonise | trial 118 s (3 months); full period estimated 30-55 min | runbook.md |
| OPeNDAP request | about 2.2 s per day, 0.2 / 0.75 MB | runbook.md, decisions.md |
| API: day read | about 10 ms per method | decisions.md |
| API: `/ranges` | cold 2-5 s (<= 8 s for `samples=60`), about 15 ms cached | api.md |
| API: point time series | cold about 1 s per point (netCDF4 directly; xarray lazy indexing took ~4 s for 184 days), cached afterwards | api.md, decisions.md |
| API: product date index, 3 methods | about 50 ms (about 3 s before using netCDF4 directly) | decisions.md |
| API: `/embeddings/similar` | a few ms | api.md |

## 14. Known limits and gotchas

- **No skill below about 300 m**: at 300 m the model is 0.50 vs 0.53 degC for climatology, and indistinguishable deeper, with five
  or eleven training years, any input set and with or without input history.
- **The released model is one training run** (seed spread about 0.003 degC for its configuration); the predictions carry no
  uncertainty estimate; the input is one day (3 or 7 days of history did not help, research R3).
- **No model family wins outright**: per-pixel boosted trees tie the Transformer over the whole domain and beat it in the Arabian
  Sea; the spatial model is better in the Bay of Bengal (research/benchmark.md).
- **Pretraining gave no benefit**: five seeds each on the `poc` run, 1.052 +/- 0.015 degC from scratch against 1.064 +/- 0.016 with
  masked pretraining (research R1). The released model is trained from scratch; the pretrained variant stays as the ablation
  (`ablation.pretrained`) and `pretrain` / `mae.py` stay in the code.
- **The target limits the score**: GLORYS is 1.05 degC from Argo over 50-200 m in 2023-2024 and about 0.4-0.5 degC warmer than the
  floats there; GLORYS assimilates Argo, so the Argo comparison is not independent of the training target.
- **The live run is not an evaluated run**: it has no reanalysis target, so `kind=target` / `difference` return 404 there, while
  `/dates` still lists `target_dates` and the run summary still carries the split of the released model. In its first 30 days the
  nowcast only matched the climatology against Argo in the thermocline (1.43 degC for both). No revised near-real-time map has been
  seen yet, so the change-detection path has run only in tests. No scheduled task is installed; see the runbook.
- **Known data quirks left unfiltered**: two GLORYS cells south of Socotra near 0 degC at 318-454 m; SSS capped at 40 in the Persian Gulf.
- **Version pins**: `zarr>=3` (Zarr v3 stores, `consolidated=False`), `erddapy<3` and `pandas<3` (argopy 1.3 breaks with erddapy 3),
  `streamlit>=1.64`, Python >= 3.12; torch deliberately unpinned so the CUDA wheel installed first is not replaced.
- **Windows TLS**: aiohttp (used by argopy) can fail `unable to get local issuer certificate`; `argo.py` sets `SSL_CERT_FILE` to the
  `certifi` bundle when the variable is unset. If you set it to something else, unset it.
- **Argo reader fallback**: ERDDAP returns classic NetCDF-3; netCDF4's in-memory reader rejects some tiny valid responses, so
  `_netcdf3_fallback` re-opens them with scipy. A 404 for an empty box is treated as "no profiles"; other failures retry 4 times (5 s doubling).
- **`run-all` downloads Argo too**: its `download` step fetches all 7 products for the whole config period (84 months of Argo), although
  `validate-argo` itself only needs the test months; for a manual download, leave out `-v argo` (runbook).
- **Relative paths resolve from the project root**: config paths (`data`, `outputs`) are relative to the current directory, so run
  pipeline commands from the project root. The readers are more forgiving: `data_access` anchors relative paths of `run_meta.json`
  at the project root recovered from the run folder, and the API's default outputs root is `<project root>/outputs` computed from
  the location of `api/app.py` (parents[3]), which assumes the source tree / editable install; elsewhere pass `--outputs-root`.
- **OneDrive**: if the project folder is synced, `data/` (tens of GB with the long period), `outputs/` and `.venv/` (~5 GB) should be excluded or sync paused.
- **`--skip-existing` does not detect stale outputs**; delete a step's output to redo it.
- **`harmonize` and `embed` delete and rebuild** their stores; `predict` overwrites monthly files in place, and a partial month contains
  only the requested days.
- **Mask definition**: a cell is ocean at a depth if GLORYS is valid there on >= 50 % of all days of the store; basin scores ignore cells
  outside the two boxes.
- **Synthetic results are never skill**: every synthetic output carries a note and the report a banner.
- **Not in the API**: day fields of climatology / GLORYS as a `method`, prediction outside the NetCDF product (the API never runs
  the model), ridge profiles at arbitrary points.
