# Architecture — OceanEmbed

Reconstructs depth-wise subsurface ocean temperature (15 standard depths, 0–1000 m) over the
North Indian Ocean (5°N–30°N, 45°E–105°E) at 0.25° / daily resolution, using only surface
satellite observations. This file is the working spec: module contracts here are binding.

## Tech stack

- Language: Python 3.12
- Framework(s): PyTorch (CUDA build, RTX 3050 6 GB — use AMP), Typer CLI, Streamlit + Plotly dashboard (PoC),
  FastAPI + uvicorn data API behind the final React dashboard (`web/`, Phase 9)
- Database (if any): none — Zarr stores + NetCDF files on disk
- Key libraries: xarray, zarr, netCDF4, numpy, scipy, pandas, scikit-learn (baseline),
  pydantic + PyYAML (config), copernicusmarine (CMEMS downloads), earthaccess (PO.DAAC downloads),
  argopy (Argo profiles), matplotlib (report figures), pytest + ruff (verification)
- Everything lives **inside the project folder**: virtual env `.venv/`, data `data/`, run outputs
  `outputs/` (all git-ignored), documents `docs/`. Run every command from the project root.
- Pins that matter: `zarr>=3` (current xarray needs it; stores are Zarr v3, read with
  `consolidated=False`), `erddapy<3` and `pandas<3` (argopy 1.3 breaks with erddapy 3), torch left
  unpinned so the CUDA wheel installed first (`--index-url .../whl/cu126`) is never replaced.
- `OCEANEMBED_DATA_ROOT` / `OCEANEMBED_OUTPUTS_ROOT` can still redirect data and outputs (tests use
  them for temp folders); leave them unset for normal use so the defaults `data/` and `outputs/` apply.

## Canonical grid (never hard-code elsewhere — import from `oceanembed.grid`)

- lat: 100 cell centres, 5.125 … 29.875 (0.25° step)
- lon: 240 cell centres, 45.125 … 104.875 (0.25° step)
- depth (m): 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
- time: daily, `datetime64[ns]` at 00:00 UTC
- Grid size is configurable (tests use a tiny grid); H and W must be divisible by 4.

## Folder structure

```
configs/
  synthetic.yaml        full-grid synthetic demo (no credentials needed)
  poc.yaml              real-data PoC (CMEMS + PO.DAAC + Argo)
  test_tiny.yaml        tiny grid / few days, used by pytest
src/oceanembed/
  config.py             pydantic config models + YAML loader
  grid.py               canonical grid, standard depths, basin masks (Arabian Sea / Bay of Bengal)
  runmeta.py            `run_meta.json` writer/reader (config snapshot, data source, splits, grid, paths)
  cli.py                Typer app: `oceanembed <command> --config <yaml>` (synth, download, harmonize, stats,
                        pretrain, train, baseline, embed, predict, evaluate, validate-argo, report, run-all, serve)
  data/
    providers/
      base.py           Provider protocol: fetch(variable, start, end) -> raw files on native grid
      synthetic.py      synthetic raw products written in the real layout (native grids, int16-packed, ...)
      synthetic_world.py  analytic truth: land/bathymetry, eddy SLA, thermocline, SSS, winds, geostrophy
      cmems.py          OSTIA SST, MULTIOBS SSS, DUACS SLA, GLORYS temperature
      podaac.py         OSCAR currents, CCMP winds: per-granule OPeNDAP subsetting (default), global-granule
                        fallback, resumable daily pieces in raw/_parts, thread pool, retries
      argo.py           Argo profiles (argopy) + optional INCOIS gridded-ARGO NetCDF loader
    regrid.py           horizontal regrid (block-mean coarsen / bilinear), temporal daily mean, vertical interp
    harmonize.py        raw -> harmonised Zarr on the canonical grid
    stats.py            input mean/std, per-depth harmonic climatology + anomaly std (train period only)
    dataset.py          torch Dataset over the harmonised Zarr (optional `preload()` of a split into RAM, or
                        `use_cache()`: on-disk memmap cache for periods that do not fit in RAM, `train.cache`);
                        `SurfaceOnlyDataset`: inputs only (never reads `temp`), batched reads, for predict/eval
  models/
    blocks.py           ResBlock / Up conv blocks shared by encoder and decoders
    encoder.py          OceanEmbed encoder: CNN stem + Transformer -> embedding map (`model.arch: unet`
                        swaps the Transformer for residual convolutions: research baseline)
    mae.py              masked surface-reconstruction pretraining wrapper
    recon.py            embedding -> 15-depth temperature decoder
    baselines.py        climatology, ridge regression
    pixel_mlp.py        per-pixel MLP baseline (research R1)
  train/
    pretrain.py         self-supervised embedding pretraining loop
    train.py            supervised reconstruction training loop
    mlp.py              MLP baseline training (research R1)
    losses.py           masked losses
    utils.py            seeding, cosine+warmup schedule, JSONL logger, loaders
  eval/
    metrics.py          streaming accumulators: RMSE, bias, MAE, correlation, skill score (masked; per depth /
                        basin / grid point)
    evaluate.py         model + baselines + ablations vs GLORYS on a split -> metrics JSON + maps NetCDF
    argo_validation.py  Argo vertical interpolation, collocation, matchups + metrics (+ optional INCOIS gridded)
    report.py           figures + report.md
  infer/
    predict.py          checkpoint loading + predictor / `predict_batch` (degC, NaN off-mask) helpers;
                        CF-1.8 monthly NetCDF writer (`predict_to_netcdf`)
    embed.py            export embedding maps to Zarr
  research/             stage R1 (rigour): `r1.py` resumable multi-seed runner (outputs only under
                        outputs/<run>/research/r1/), `bootstrap.py` moving-block bootstrap, `r1_report.py`
                        summary tables / figures; see docs/backend.md section 9; stage R4 (`armor3d.py`, `r4.py`,
                        `r4_report.py`: ARMOR3D benchmark) and R5 (`physical.py`, `r5.py`, `r5_report.py`: derived
                        physical quantities, stratified skill), shared `common.py`; stage R3 (`inputs.py`, `r3.py`,
                        `r3_report.py`: variable groups / history / permutation importance) and R2 (`r2.py`,
                        `r2_report.py`: long training period, two test years, learning curve, Argo)
  data_access.py        read-only loaders shared by the Streamlit app and the HTTP API (no UI code, no Streamlit
                        import; documented below); `app/data_access.py` is a thin re-export of it
  api/                  read-only FastAPI data API (`oceanembed serve`), documented in docs/api.md
    app.py              `create_app(outputs_root, web_dist)`: CORS (Vite dev origins only), GZip, JSON error
                        handlers, SPA static serving + fallback, startup warm-up thread
    store.py            run registry + validation (run names, dates, depth, points), byte-bounded LRU of opened
                        days / series, colour-range hints, point time series, Argo / maps caches
    deps.py, encode.py  ETag / Cache-Control / 304 helpers; NaN -> null JSON encoding (`NaNSafeJSONResponse`)
    schemas.py          pydantic response models (drive `/api/docs`)
    metrics.py          reshape of the metrics JSONs (metric-major) + experiments table rows
    routes_runs.py      runs, run detail, dates, mask, training, experiments, compare, report, figures, product
    routes_fields.py    fields (JSON / binary float32), surface, profile, section, timeseries, embeddings
    routes_metrics.py   metrics vs GLORYS / Argo, error maps, Argo matchups / profiles
app/
  data_access.py        re-export of `oceanembed.data_access` (so the Streamlit app and its tests import unchanged)
  Home.py               Overview page (Streamlit entry point: `streamlit run app/Home.py`)
  pages/                1_3D_Explorer.py, 2_Profiles_and_Sections.py, 3_Validation.py, 4_Embeddings.py
                        (thin: each composes `app.ui` pieces; Streamlit's default page nav is hidden)
  ui/                   dashboard UI package, imported as `app.ui` from the repo root
    theme.py            design tokens, colormaps, per-method styles, number formatting (no Streamlit)
    figures.py          Plotly figure builders: pure functions of arrays (no Streamlit, no file IO)
    views.py            page-level builders: cards, tables, figures from a run's metrics / fields (no Streamlit)
    data.py             `st.cache_data` wrappers around `app.data_access`, keyed by run dir + run_meta mtime;
                        `outputs_root()` = `OCEANEMBED_OUTPUTS_ROOT` or `<project root>/outputs`
    components.py       page shell: CSS, `st.page_link` navigation, run selector, synthetic-data banner,
                        metric cards, legends, empty states, day / depth controls
.streamlit/config.toml  Streamlit theme (dark ocean tokens), hides the default sidebar navigation and
                        error tracebacks, no usage statistics (read from the current directory: start
                        Streamlit from the project root)
web/                    React single-page app (final dashboard); `web/dist` is served by the API. See "Web dashboard" below
  src/api/              schema.d.ts (generated from /api/openapi.json), types.ts, client.ts, binary.ts (float32 volumes,
                        packbits mask), volumes.ts (client LRU of daily volumes), queries.ts (React Query hooks)
  src/lib/              colormaps, scales, format, dates, geo, stats, methods (one style per method), metricMeta,
                        narrative (metric-to-sentence), embedding, theme (design tokens)
  src/state/            url.ts + router.ts (deep-linkable state), runContext.tsx, linkedView.ts (shared map camera/cursor)
  src/components/       map/ (canvas map, colour bar), charts/ (line chart, depth heat map, scatter, tables),
                        controls/ (timeline, depth rail), shell/, ui/
  src/views/            overview/, explorer/, validation/, representation/, experiments/
  src/styles/           CSS (tokens that are not colours, layout)
tests/                  pytest, runs entirely on synthetic tiny data (`conftest.tiny_run` = one full `run-all`)
data/                   (git-ignored) raw/ (real), raw_synthetic/ (synthetic), processed/ ;
                        root overridable via OCEANEMBED_DATA_ROOT (config `paths.raw_dir` picks the raw folder)
outputs/                (git-ignored) <run_name>/ checkpoints, predictions, metrics, figures, report;
                        experiments/ holds archived tuning runs and their console logs
docs/                   PRD, architecture, phases, decisions, memory (build log), design, runbook
.venv/                  (git-ignored) project virtual env
```

## Data flow

1. **Download** (`oceanembed download`): each provider writes raw NetCDF on its *native* grid to
   `data/raw/<product>/<product>_<YYYYMM>.nc` (monthly files; Argo: `argo_<YYYYMM>.parquet`;
   synthetic runs use `data/raw_synthetic/` so fake and real files never mix). A product may list
   several datasets with validity windows (e.g. SSS: reprocessed until 2023-12, NRT from 2024-01 —
   switch-overs must be month boundaries). Real providers need the user's own logins (`copernicusmarine login`,
   Earthdata `.netrc`). OSCAR / CCMP (`podaac.py`) are fetched day by day (`download.podaac_mode`):
   `opendap` (default) asks NASA's Hyrax OPeNDAP service (`opendap.earthdata.nasa.gov`, through the
   authenticated `earthaccess` session) for only the configured variables over the index ranges covering
   domain + halo (a DAP4 constraint expression, NetCDF-4 reply; ranges are computed from the granule's own
   coordinate arrays, handling ascending/descending axes and 0-360 vs -180-180 longitudes; about 0.2 MB per
   OSCAR day and 0.75 MB per CCMP day instead of the 32 MB global granule); `granule` (or the automatic
   fallback when OPeNDAP is refused or fails 3 times) downloads the global granule, crops it and deletes it.
   Each day is an atomic piece `raw/_parts/<product>_<YYYYMM>/<YYYYMMDD>.nc` (skipped when present, retried with
   exponential back-off, `download.workers` threads); the monthly file is merged only when all days of the month
   exist, then the pieces are removed, so an interrupted download resumes without repeating finished days.
   Every provider appends one JSON line per completed download to the transfer ledger
   `data/raw/_download_log.jsonl` (product, dataset id, period, route, bytes written, bytes transferred and how
   it was measured, seconds; no credentials or URLs). `regrid.standardize` also promotes coordinate variables
   that sit on differently named dimensions (OSCAR `lon` on dimension `longitude`) and rebuilds `datetime64`
   time from cftime (OSCAR declares a julian calendar). The synthetic provider (`oceanembed synth`) writes the same layout with
   the same native resolutions, so every downstream step is identical for real and synthetic data.

   | Variable | Product (PRD) | Native res | Regrid to 0.25° daily |
   |---|---|---|---|
   | SST | OSTIA (moi-00168) | 0.05°, daily | 5×5 block mean |
   | SSS | MULTIOBS SMAP/SMOS (moi-00051) | 0.125°, daily | 2×2 block mean |
   | SLA | DUACS (moi-00145) | live catalogue: 0.125°, daily (PRD says 0.25°) | 2×2 block mean (bilinear if a 0.25° dataset is configured) |
   | Currents U,V | OSCAR L4 final v2.0 | 0.25°, daily, nodes on multiples of 0.25° | bilinear (half-cell offset) |
   | Winds U,V | CCMP v3.1 | 0.25°, 6-hourly | daily mean + bilinear |
   | Temperature (target) | GLORYS12 (moi-00021) | 1/12°, daily, 50 levels | 3×3 block mean + linear vertical interp to standard depths |
   | Validation | Argo profiles / INCOIS gridded ARGO | profiles | vertical interp, nearest cell + day |

2. **Harmonise** (`oceanembed harmonize`): raw → `data/processed/<name>.zarr`
   - `sst, sss, sla, uo, vo, uw, vw` : `(time, lat, lon)` float32, NaN over land / gaps
   - `temp` : `(time, depth, lat, lon)` float32, NaN below sea-floor / land
   - `mask` : `(depth, lat, lon)` bool — ocean & above sea-floor (derived from target)
   - chunked one day per chunk on `time` (random access for training)
   - block-mean regridding ignores NaN and requires ≥50 % valid native cells; the method is chosen
     automatically from the native/target resolution ratio (block when ≥1.5×, else NaN-aware bilinear)
   - source lon is wrapped to 0–360, axes sorted ascending, dims transposed by name, Kelvin → °C by `units`
   - processed month by month in `harmonize.time_chunk`-day blocks (bounded RAM); optional user-supplied
     INCOIS gridded ARGO (`raw/argo_gridded/*.nc`) is regridded to `processed/<name>_argo_gridded.nc`
3. **Stats** (`oceanembed stats`): computed on the **train split only**, saved to `data/processed/<name>_stats.nc`
   - per-channel input mean / std (over ocean points)
   - target climatology: per grid-point, per-depth least-squares fit of mean + annual + semi-annual harmonics
     (phase = 2π·(date − Jan 1)/days-in-year, so leap years work; accumulated as streaming normal
     equations; train periods < 180 d fit the mean only, < 548 d mean + annual)
   - per-depth anomaly std
   - dataset items: `x` (12,H,W), `y` standardised anomaly (15,H,W), `mask` (15,H,W) bool, `t` (days since 1970), `index`
4. **Embedding pretraining** (`oceanembed pretrain`): masked-surface autoencoding. Random 16×16 px
   blocks (default 50 %) of the surface channels are hidden; the encoder + light conv decoder
   reconstruct all 7 channels; loss on hidden ocean pixels only.
5. **Reconstruction training** (`oceanembed train`): pretrained encoder (fine-tuned at reduced LR)
   + decoder predicts the **standardised temperature anomaly** at 15 depths. Loss: masked MSE +
   small vertical-gradient consistency term. Temperature = climatology + anomaly × std.
6. **Predict** (`oceanembed predict [--split test | --start D --end D] [--tag t | --ridge]`): CF-1.8 NetCDF,
   `temperature(time, depth, lat, lon)` float32 degC, daily, 0.25°, one file per month
   (`outputs/<run>/predictions/oceanembed_T_<YYYYMM>.nc`; `--tag t` uses `recon_<tag>.pt` and writes to
   `predictions/<tag>/`; `--ridge` predicts with the fitted ridge baseline and writes `predictions/ridge/`, the
   reserved folder name). The extra products exist so the data API can serve ridge and ablation day fields
   without computing anything per request. Needs only the surface inputs + stats (via `SurfaceOnlyDataset`; the target is never
   read), so it runs for any day present in the harmonised store, including days without GLORYS. NaN is stored as
   `_FillValue` 9.96921e36 outside the static ocean mask; zlib level 4, one chunk per day. Global attrs: Conventions
   CF-1.8, title, institution, source, history, comment, `data_source` (`synthetic` | `real`), run name, checkpoint
   name / epoch / pretrained flag, training period, config JSON; variable attrs `standard_name
   sea_water_potential_temperature`, `units degree_Celsius`; depth `positive: down`, `axis` X/Y/Z/T on coordinates.
   A partial month holds only the requested days (a later run overwrites the file).
7. **Evaluate** (`evaluate`, `validate-argo`, `report`): all under `outputs/<run>/`.
   - `evaluate [--split test]` streams the split in batches of days; for the model, tagged ablations
     (`recon_<tag>.pt` -> method `model_<tag>`), ridge and climatology it accumulates raw-temperature and anomaly
     (climatology-removed) statistics (`eval/metrics.py`: per-grid-point sums, reduced afterwards). It also keeps
     per-day, per-depth domain sums, giving the `daily_rmse` block: `rmse`, `bias`, `corr_anom` (spatial
     anomaly correlation of that day) as `[day][depth]` per method and `pooled_rmse`, `pooled_bias`,
     `pooled_corr_anom` `[day]` over 50-200 m (all derived from the same per-day sums).
     Writes `metrics/metrics_glorys.json` and `metrics/maps_glorys.nc`.
   - `validate-argo [--split test]` loads Argo profiles for the split months (cached parquet; the synthetic provider
     generates them, the real provider downloads missing months), interpolates each to the standard depths
     (rule: exact level, else bracketing levels with spacing <= max(10 m, 0.2 z), no extrapolation, surface
     exception z <= 5 m uses the shallowest level within 10 m), collocates to the containing cell and same UTC day,
     drops out-of-period / out-of-domain / land profiles (counts logged and stored) and keeps (profile, depth) rows
     where observation, mask, GLORYS and every method are finite. Writes `metrics/argo_matchups.parquet` and
     `metrics/metrics_argo.json`. If `raw/argo_gridded/*.nc` exists (INCOIS), the same methods are also scored
     against it (`gridded_argo` block; daily products same day, monthly products against the monthly model mean for
     fully covered months); silently skipped when absent.
   - `report` writes `figures/*.png` and `report.md` (setup, metrics tables, figures, limitations; a prominent banner
     when `data_source == synthetic`).
8. **Dashboard** (`.\.venv\Scripts\streamlit.exe run app/Home.py`, from the project root): reads `outputs/<run>/`
   through `app/data_access.py` (via the cached wrappers in `app/ui/data.py`). Five pages: Overview, 3-D Explorer,
   Profiles & Sections, Validation, Embeddings; design tokens in `docs/design.md`. Pages put the repo root on
   `sys.path` so `app.ui` imports resolve; `run_meta.json` paths that are relative are anchored at the project
   root recovered from the run directory, so the data layer does not depend on the current directory.

`oceanembed run-all --config <yaml> [--skip-existing] [--device d]` chains synth|download -> harmonize -> stats ->
pretrain -> train -> [train-ablation] -> baseline -> embed -> predict -> predict-ridge -> [predict-ablation] ->
evaluate -> validate-argo -> report in-process (each step calls the function of its own CLI command), prints a
banner and timing per step and stops at the first failure with a non-zero exit code. `--skip-existing` skips
steps whose outputs exist (it does not detect stale outputs). The bracketed steps run when the config sets
`ablation: {no_pretrained: true, tag: scratch}` (the from-scratch `train --no-pretrained --tag scratch` and its
`predict --tag scratch`); they run before `evaluate`, so the metrics include the ablation. `predict-ridge`
(`predict --ridge`) always runs. The shipped `poc`, `poc_trial`, `synthetic` and `test_tiny` configs enable the
ablation. Optional `label` / `description` config keys give the run its display name in the API.
Every output-producing command (re)writes `outputs/<run>/run_meta.json`.

## Model

Model input per day: 12 channels on the canonical grid —
7 standardised surface fields (NaN → 0), ocean mask, sin/cos day-of-year, normalised lat, normalised lon.

- **Encoder (the "satellite embedding engine")** — attention-based hybrid:
  CNN stem (residual conv blocks, two stride-2 stages → H/4 × W/4, i.e. 25×60 tokens)
  → Transformer encoder (pre-norm, fixed 2-D sin-cos positions, default 6 layers / 6 heads / dim 192)
  → projection to embedding map `(emb_dim=128, H/4, W/4)`. Stem features at 1/1 and 1/2 are kept as skips.
- The encoder appends one internal "visible" channel to the 12 dataset channels (13 conv inputs). It is
  0 inside hidden blocks during pretraining and 1 everywhere otherwise (all downstream use), so the
  encoder interface is identical in both stages. Stem skips are computed from the same masked input,
  so they cannot leak hidden pixels.
- **Pretraining decoder** — small conv upsampler → 7 channels, fed the embedding map only (no skips).
  Hidden blocks are exactly `round(mask_ratio · n_blocks)` per sample; loss on hidden, ocean, observed pixels.
- **Reconstruction decoder** — U-Net-style upsampling from the embedding map with the stem skips → 15 channels.
- Must train on a 6 GB GPU: mixed precision (fp16 + GradScaler), default batch size 8.
- Training data is preloaded into RAM per split (`OceanDataset.preload()`, float32, ~2 GB for the 2-year
  train split of the full grid); per-day Zarr reads are only used for ad-hoc access.
- Fine-tuning: encoder LR = `train.encoder_lr_scale` × decoder LR; the from-scratch ablation uses the full LR
  for both. Early stopping on val RMSE in °C (patience `train.patience`).
- Baselines are *predictors* (`x (B,12,H,W)` → standardised anomaly `(B,15,H,W)`) exactly like the model, so
  evaluation treats all of them uniformly (`infer/predict.py`). Ridge: 11 features (7 surface, sin/cos DOY,
  lat, lon), one `sklearn.Ridge` per depth fitted on a random subsample of train points valid at that depth.

## Splits and baselines

- Temporal split by date ranges in config (train / val / test) — test period is never seen in training or stats.
- Baselines evaluated alongside the model: harmonic **climatology**, pixel-wise **ridge regression**
  (surface vars + day-of-year + lat/lon → 15 depths). Optional ablation: same network without pretraining.

## Evaluation outputs (contract used by the dashboard)

```
outputs/<run>/
  run_meta.json                   config snapshot, data_source, split dates, grid, data paths (as the config
                                  gave them, i.e. relative to the project root by default)
  checkpoints/pretrain.pt, recon.pt, [recon_<tag>.pt], ridge.joblib
  logs/pretrain.jsonl, train.jsonl, [train_<tag>.jsonl]
  predictions/oceanembed_T_<YYYYMM>.nc      (ridge: predictions/ridge/..., ablation: predictions/<tag>/...)
  metrics/metrics_glorys.json     per method (model, model_<tag>, ridge, climatology): overall, pooled_50_200m,
                                  per_depth, per_basin{arabian_sea, bay_of_bengal}{same blocks}; daily_rmse (dates, depth, rmse,
                                  bias, corr_anom, pooled_rmse, pooled_range_m); metadata
  metrics/maps_glorys.nc          (depth, lat, lon): rmse_<m>, bias_<m>, corr_raw_<m>, skill_vs_clim_<m> for every
                                  method, corr_anom_<m> for all but the climatology, n_valid
  metrics/metrics_argo.json       same blocks for model, ablations, ridge, climatology and glorys vs Argo; metadata
                                  (profile counts, dropped counts, interpolation rule, independence note);
                                  optional `gridded_argo`
  metrics/argo_matchups.parquet   one row per (profile, depth): profile_id, time, lat, lon, depth, obs, model, ridge,
                                  clim, [model_<tag>], grid_lat, grid_lon, glorys, basin
  embeddings/embeddings.zarr      (time, emb, y, x) for the test period
  figures/*.png, report.md
```

Metric blocks (`overall`, `pooled_50_200m`, and the columns of `per_depth`) carry `n, rmse, bias (method - reference),
mae, corr_raw, corr_anom, skill_vs_clim`; `per_depth` is column-oriented (`depth` list + one list per metric; NaN is
`null`). `corr_raw` correlates temperatures (inflated by seasonal / spatial / vertical gradients), `corr_anom`
correlates anomalies with the climatology removed from both sides, `skill_vs_clim = 1 - MSE / MSE_climatology`.
Method keys: `model` (pretrained encoder), `model_<tag>` (ablation, e.g. `model_scratch`), `ridge`, `climatology`,
`glorys` (Argo file only); the matchup parquet calls the climatology column `clim`. `metadata.data_source` is
`synthetic` or `real` in every file, and synthetic outputs carry a note that they only demonstrate the pipeline.

## Dashboard data layer (`oceanembed.data_access`, re-exported as `app.data_access`)

Pure Python (xarray / pandas / numpy; no Streamlit), read-only; every function takes the run directory
(`outputs/<run>`) and hashable arguments so it can be wrapped in `st.cache_data`. Paths come from `run_meta.json`;
relative ones are resolved against the project root (derived from the run directory), never the working directory.
A missing store / statistics file yields `None` / `FileNotFoundError`, which the pages turn into an empty state.
Dates may be str / datetime / np.datetime64 / Timestamp. Missing optional products give `None` (or `[]` / `{}`).

| function | returns |
|---|---|
| `list_runs(outputs_root)` | list of dicts: name, path, data_source, updated, has_predictions / has_metrics / has_argo / has_embeddings / has_report |
| `load_run_meta(run)` | `run_meta.json` as dict |
| `load_metrics_glorys(run)`, `load_metrics_argo(run)` | the metrics JSONs as dicts or `None` |
| `load_argo_matchups(run)` | DataFrame or `None` |
| `load_maps(run)` | `maps_glorys.nc` as in-memory Dataset or `None` |
| `available_dates(run, method='model')` | `DatetimeIndex` of predicted days of that product |
| `prediction_methods(run)` | `{method: folder}` of the products on disk: `model`, `ridge`, `model_<tag>` |
| `load_prediction(run, date, method='model')` | `(depth, lat, lon)` DataArray, degC (`KeyError` if the day has no prediction) |
| `load_target(run, date)` | harmonised GLORYS `(depth, lat, lon)` or `None` |
| `load_climatology(run, date)` | harmonic climatology `(depth, lat, lon)`; NaN off-mask |
| `load_climatology_point(run, dates, i, j)` | climatology `(n_dates, depth)` at one cell (five coefficients per depth) |
| `climatology_info(run)`, `split_day_counts(run)` | stats-file attributes (`n_harmonic_terms`, ...); days per split |
| `load_surface_inputs(run, date)` | Dataset of the 7 surface fields `(lat, lon)` in physical units, or `None` |
| `load_profile(run, date, lat, lon)` | Dataset over `depth`: `predicted`, `target`, `climatology` at the nearest cell |
| `load_section(run, date, lat=… \| lon=…)` | Dataset `(depth, lon)` or `(depth, lat)`: `predicted`, `target`, `climatology`, `difference` |
| `load_embeddings(run, date)` | `(emb, y, x)` DataArray or `None` (test split only) |
| `embedding_pca_rgb(run, date, n_fit_days=24, seed=0, n_components=3)` | `(y, x, rgb)` in [0, 1] with up to 8 components; PCA fitted once per run on a day sample (cached), same axes every day; `attrs['explained_variance_ratio']` |
| `embedding_pca_mean(run)` | mean embedding vector of that PCA sample (for centred similarity) |
| `load_training_logs(run)` | dict stem -> DataFrame (`pretrain`, `train`, `train_<tag>`) |
| `prediction_file_dates(run, method='model')` | dict `YYYYMM` -> `DatetimeIndex` of each monthly product file (cached per file mtime / size) |
| `embedding_dates(run)` | `DatetimeIndex` of days with an embedding (empty if none) |

The harmonised-Zarr / embeddings handles and the per-file date index are cached inside the module (keyed by the
`run_meta.json` mtime resp. file mtime), so repeated calls do not re-open stores.

## Data API (`oceanembed serve`, package `oceanembed.api`)

Read-only FastAPI app on top of the data layer above; full endpoint reference in `docs/api.md`. Chosen for the
final React dashboard because Streamlit's rerun model and widget chrome cap the achievable quality.

- Binds `127.0.0.1:8000`; every route is `GET` under `/api`; CORS only for the Vite dev origins; GZip; errors are
  `{"detail": ...}` (400 bad parameter, 404 unknown run / missing artefact), never a stack trace.
- A run is a folder directly under the outputs root with a `run_meta.json`; names are regex-validated.
  The `run_meta.json` mtime is the run version: it keys the in-process caches and the `ETag` (conditional GET -> 304
  without reading data).
- NaN is `null` everywhere. Large fields come as JSON or as raw little-endian float32 (`format=f32` /
  `Accept: application/octet-stream`, shape and colour range in `X-*` headers). Colour-range hints (1st-99th percentile
  shared by prediction / target, symmetric for differences) are computed server-side.
- Data flow per request: `Store` (validation + LRU of daily volumes: prediction NetCDF, harmonised Zarr target,
  harmonic climatology) -> numpy slice -> JSON / bytes. Point time series read the monthly NetCDF columns with
  `netCDF4` directly (~6x faster than xarray lazy indexing) and are cached per cell.
- `method=model|ridge|model_<tag>` on `/fields`, `/profile`, `/section`, `/timeseries` and the product download
  selects the prediction folder (`prediction_methods`); target and climatology do not depend on it. Temperature
  colour ranges always come from the main model so panels of different methods share a scale.
- `/embeddings/similar` computes the cosine similarity over all `emb_dim` features of one day on request
  (a few ms; land cells are `null`); `/argo/profiles` filters (basin, dates, bounding box) / sorts / pages the
  cached per-profile table and can jump to the page holding one profile (`around`).
- `/ranges` gives per-depth colour ranges over the whole prediction period from a sample of evenly spaced days
  (`Store.period_ranges`: in memory, per run version, never written to the run folder); skill-map limits are the
  95th percentile of |skill| kept within 0.5..3 and each map reports what exceeds its range (`range_info`).
- Runs still being produced never 500: unreadable (truncated) monthly product files are skipped by
  `data_access._prediction_files`, and every missing artefact is a 404 with a hint.
- `web/dist/index.html`, if present, is served at `/` with a client-side-route fallback that never shadows `/api`.

## Web dashboard (`web/`)

Vite + React 19 + TypeScript (strict) single-page app; the only data source is the API above (same-origin `/api`;
in development Vite on :5173 proxies `/api` to `http://127.0.0.1:8000`). Design system in `docs/design.md`.

- Runtime dependencies are kept to four: `react` / `react-dom`, `@tanstack/react-query` (request cache, loading and
  error states per endpoint), `markdown-to-jsx` (the generated report, loaded only when opened) and the bundled
  fonts (`@fontsource`: Newsreader, IBM Plex Sans, IBM Plex Mono; latin subsets). No chart, map or routing library:
  maps are a custom 2-D canvas renderer, charts are small SVG components, routing is a 100-line URL store.
- Typed client: `npm run gen:api` writes `src/api/openapi.json` (snapshot) and `src/api/schema.d.ts`
  (openapi-typescript) from the running API; `src/api/types.ts` narrows the free-form dictionaries as documented in
  `docs/api.md`; `openapi.test.ts` checks every path and query parameter the client uses against the snapshot.
- Data flow: `App` resolves the run (URL or the best default: a run with a reconstruction before one still being
  produced, real before synthetic, trained on at least one year before shorter) and loads run detail, dates and the
  packbits ocean mask; `RunProvider` exposes grid, depth / day / point / estimate selection (snapped to valid values;
  the estimate is one of the run's `field_methods`) and the caveats from the run summary (`n_train_days`,
  `n_harmonic_terms`, `data_source`). Views are lazy chunks.
- Selection bar: `components/controls/SelectionBar.tsx` is the one control surface for day, depth, point and
  estimate. Each view renders it through a portal into a slot of the sticky header (`Shell`), so it sits in the same
  place on every view; it only reads and writes the URL state.
- Fields: the Explorer fetches the full 15-level volumes of a day as float32 (`format=f32`): the prediction of each
  shown method (`method=`), target and climatology, 1.44 MB each. Differences, anomalies, profiles, sections and
  level statistics are computed in the browser from those arrays, so scrubbing depth and stepping days does not touch
  the network; neighbouring days are prefetched and an LRU (200 MB) bounds memory. Colour ranges follow the API hints
  (`X-Color-Range-Per-Depth`); the held range over the period comes from `/ranges`. The time-depth plot comes from `/timeseries?method=` (with its climatology for the
  anomaly mode).
- Argo: `/argo/profiles` is called twice per filter: once for every match (map, drawn on canvas in colour-binned
  batches, `lib/points.ts`) and once per page of ten with the API's sorting and paging (list; `around=` when a
  profile was opened from the map).
- Overview: a fixed result panel. `chooseEvidence` (`lib/narrative.ts`) picks its depth (largest climatology RMSE in
  the pooled range) and day (median pooled daily RMSE) from `/metrics/glorys`; the view ignores the URL's day and depth.
- Maps: `MapCanvas` draws a grid-sized image through a 256-entry lookup table with nearest-neighbour scaling on a
  true-aspect equirectangular canvas, land and sea floor from the mask, coastline from mask cell edges; a `LinkedView`
  object shares camera and cursor between the maps of a group without React re-renders.
- State: `/<view>?run=&date=&depth=&lat=&lon=&est=` plus per-view options; view changes push history, selection
  changes replace it (debounced).
- Text that states a result is generated from the metrics in `src/lib/narrative.ts` (tested), never written by hand.
- Commands (in `web/`): `npm run dev`, `npm run build` (-> `web/dist`: about 180 kB of gzipped JS and CSS in total, about 125 kB for a first view, plus
  230 kB of fonts), `npm run lint`, `npm run typecheck`, `npm test` (Vitest + Testing Library).

## Deployment / how it runs

Local only (Windows 11, RTX 3050 6 GB). `py -3.12 -m venv .venv`, CUDA torch wheel, then
`pip install -e .[dev]` into `.venv`; run `.\.venv\Scripts\oceanembed.exe ...` and
`.\.venv\Scripts\streamlit.exe run app/Home.py` from the project root (Streamlit >= 1.64). The data API for the
React dashboard: `.\.venv\Scripts\oceanembed.exe serve [--host 127.0.0.1] [--port 8000] [--outputs-root ...] [--reload]`
(`/api/docs` is the OpenAPI page; the Vite dev server on :5173 is allowed by CORS). The React dashboard is built once
with `cd web; npm install; npm run build` and is then served by the same command at `http://127.0.0.1:8000/`. Lint:
`ruff check .` and `ruff format --check .`. Tests: `pytest` (they use temp data / outputs roots via the env overrides).
