# Data API — OceanEmbed

Read-only HTTP API over the finished runs in `outputs/`, built for the React dashboard (`web/`).
FastAPI app in `src/oceanembed/api/`; interactive OpenAPI page at **`/api/docs`** (raw spec
`/api/openapi.json`, ReDoc `/api/redoc`) — every endpoint below has a pydantic response model there.

```powershell
.\.venv\Scripts\oceanembed.exe serve                    # http://127.0.0.1:8000
.\.venv\Scripts\oceanembed.exe serve --port 8010 --outputs-root outputs --reload
```

Options: `--host` (default `127.0.0.1`, localhost only), `--port`, `--outputs-root` (default
`OCEANEMBED_OUTPUTS_ROOT` or `<project>/outputs`), `--reload` (dev auto-reload). Programmatic use:
`from oceanembed.api import create_app; app = create_app(outputs_root=None, web_dist=None)`.

## Conventions

- Base URL `http://127.0.0.1:8000`, everything under `/api`. `GET` only; no authentication.
- **NaN is `null`** in every JSON payload (land, below the sea floor, missing target, undefined metric).
  Temperatures are rounded to 0.001 °C, coordinates to 1e-6 degrees.
- A run is identified by its folder name under `outputs/` (`/api/runs` lists them). Names are validated
  (`[A-Za-z0-9][A-Za-z0-9_.-]*`, must be a real run folder) so path traversal is impossible.
- Dates are `YYYY-MM-DD`. Depths are metres and must equal a grid depth (`depth=100`) **or** be given as a
  0-based `depth_index` (exactly one of the two). Points (`lat`, `lon`) must lie inside the grid; they are
  snapped to the nearest cell centre (ties go to the lower coordinate) and the snapped cell is returned.
- **Methods.** `/fields`, `/profile`, `/section`, `/timeseries` and the product download take `method=` (default `model`): `model`, `ridge`, `mlp` (the per-pixel MLP baseline, when the run has one) or an ablation such as `model_scratch` / `model_pretrained` - the same keys as the metrics. `field_methods` of the run detail lists which methods have day fields (the prediction NetCDF products written by `run-all`: `predictions/`, `predictions/ridge/`, `predictions/mlp/`, `predictions/<tag>/`). A method without a product is a **404** whose message lists the available ones; `climatology` and `glorys` are not methods here (use `kind=climatology` / `kind=target`).
- Errors are always `{"detail": "<message>"}`: **400** bad parameter (malformed or out-of-range date, depth
  not on the grid, point outside the grid, unknown `kind`/`metric`, ...), **404** unknown run, unknown
  file / profile / method, or a missing artefact (no embeddings, no Argo, no target for that day, ...),
  **405** non-GET, **500** unexpected (no stack trace is sent).
- Caching: every run endpoint sends `ETag` (hash of run version + URL + `Accept`; the run version is the
  mtime of `run_meta.json`, which every pipeline command rewrites) and
  `Cache-Control: private, max-age=60, must-revalidate`. `If-None-Match` gets a `304` without reading any
  data. `/api/health`, `/api/runs` and `/api/compare` are `no-store`/uncached.
- GZip for responses over 1 kB. CORS only for the Vite dev origins `http://localhost:5173` and
  `http://127.0.0.1:5173` (`GET`/`HEAD`/`OPTIONS`; the `X-*` headers below are exposed).
- Server-side LRU (768 MB budget) of opened days / series, so scrubbing days and depths reads each file once.
- **Runs that are still being produced** list and open at every stage: a folder with only `run_meta.json`, predictions without metrics, metrics without Argo, or a monthly product file that is still being written (an unreadable file is skipped, not an error). Whatever is missing answers **404** with a message naming the missing step, never 500 (checked against every endpoint in `tests/test_api.py`). The API only reads, and keeps derived results (colour ranges, profile tables, ...) in memory per run version, so it never writes into a run folder.

## Running backend and front end together

```powershell
# terminal 1 (project root)
.\.venv\Scripts\oceanembed.exe serve --reload          # API on :8000
# terminal 2
cd web; npm install; npm run dev                       # Vite on :5173
```

Vite calls the API either directly (`http://127.0.0.1:8000/api/...`, CORS is enabled for it) or through a dev
proxy (`server.proxy: {"/api": "http://127.0.0.1:8000"}`). For production run `npm run build` once; with
`web/dist/index.html` present `oceanembed serve` serves the SPA at `/` (hashed files under `/assets/` are
`immutable`, `index.html` is `no-cache`) and falls back to `index.html` for any other non-`/api` path
(client-side routes). A missing asset (`/assets/x.js`) is a `404`, never the HTML shell. Without a build,
`/` returns `{"message": "The OceanEmbed web UI is not built...", "api_docs": "/api/docs"}`. `/api/...` is
never shadowed: an unknown `/api/foo` is a JSON `404`.

## Endpoints

### `GET /api/health`
`{"status": "ok", "version": "0.1.0", "n_runs": 2, "ui_built": false}`

### `GET /api/runs`
`{"runs": [RunSummary]}` — per run: `name` (URL id), `run_name`, `data_source` (`synthetic`|`real`),
`updated`, `period`, `split{train,val,test}{start,end}`, `grid{resolution,n_lat,n_lon,n_depth,lat,lon,depths}`,
`artefacts{predictions, metrics_glorys, metrics_argo, argo_matchups, maps, embeddings, report,
training_logs[stems], n_figures, n_product_files}`, `n_prediction_days`, `note` (set for synthetic runs),
`label` (human display name), `description` (one paragraph: source, period, splits, climatology), `n_train_days`, `n_val_days`, `n_test_days` (days of each split present in the harmonised data) and
`n_harmonic_terms` (climatology fit: 1 = mean only, 3 = + annual cycle, 5 = + semi-annual; `null` without statistics file),
`live` (true only for the rolling near-real-time nowcast run, which also has `live_last_day`; a client that needs a default
run must not choose a `live` run: it has no evaluation against the reanalysis and its days are revised) and `live_last_day`. `label` / `description` come from the config keys of the same name, else they are derived (e.g. `"Synthetic demo, 2021-01 to 2023-12"`).

### `GET /api/runs/{run}`
`{summary, grid, basins, prediction_dates, methods, field_methods, model, training_summary, products, counts,
metrics_metadata}`:
`grid` adds `lat_values[100]`, `lon_values[240]`, `depth_values[15]`, `lat_edges`, `lon_edges`;
`basins` = `[{key, label, box{lat_min,lat_max,lon_min,lon_max}}]` (from `grid.py`);
`methods` = `[{key, label, kind, tag, pretrained, epoch, val_rmse, in_glorys_metrics, in_argo_metrics,
has_day_fields}]` (`model`, `model_scratch`, `ridge`, `climatology`, `glorys`);
`field_methods` = `[{key, label, kind (model|baseline|ablation), tag, n_days, first, last}]`, the methods whose day fields can be requested with `method=`, `model` first, then `ridge`, `mlp`, then ablations; `kind` is `baseline` for ridge and mlp;
`model` = encoder / pretrain / train config summary; `training_summary` = per log: epochs, best val value,
total seconds; `products` = variable → product, `dataset_ids` (from the run config), native resolution,
regridding, role (`input`/`target`/`validation`); `counts` = `n_prediction_days`, `n_train_days`, `n_val_days`, `n_test_days`,
`n_embedding_days`, `n_argo_profiles`, `n_argo_matchups`, `n_argo_profiles_loaded`.

### `GET /api/runs/{run}/live`
Only for the live run (`oceanembed live update`; `summary.live` is true); any other run is `404` with the hint "this is not a
live run". Never cached (`Cache-Control: no-store`): the files change on every update. Payload:
`{run, note, last_update, window{start,end,n_days,window_days}, last_day, inputs[{product (sst|sla), dataset, version, first,
last, latest_data_date, age_days, delay_days_catalogue, available}], window_days[{date, reconstructed, complete, inputs{sst, sla}}], pending[dates], model{checkpoint, weights}, days[...], revision,
input_shift, verification, history[...]}`.
`last_day` is a nowcast of that day, not a forecast. `inputs[].available` says whether that input has data for `last_day`; `window_days` lists every calendar day of the window with a completeness flag. `age_days` is the age of the newest day held for that input at the last update;
`pending` are days one input already has and another does not. `days` is the provenance table, one row per window day: `date`,
`<input>_dataset`, `_version`, `_product_version`, `_age_days` (when the day was first used), `first_update_ts`, `last_update_ts`,
`n_checks`, `n_revisions`, `revised`, `rev_sst_rmse`, `rev_sla_rmse`, `rev_recon_rmse_50_200`, `rev_recon_maxabs` (the size of the
revision against the first-published version), `checkpoint`, `device`. `revision` = `{policy, n_days_checked, n_days_revised,
by_age[{age_days, n_checks, n_changed_since_first, sst_rmse_mean, sla_rmse_mean, recon_rmse_50_200_mean, recon_rmse_50_200_max}]}`.
`input_shift` = `{headline, headline_text, updated}` from `live input-shift` (`null` until it has run); `verification` = the content
of `checks/verification/verification.json` (`null` until the first update with verification): `argo` and `analysis` blocks, each
with `daily[{date, n_profiles, bands{0_30|50_200|300_1000{day{n, model_rmse, model_bias, clim_rmse, clim_bias}, rolling{...,
n_days}}}}]`, `latest`, a `note` (the analysis is a model, not an observation), and for Argo `arrival_by_age`; `headline` = the
rolling numbers of the newest day. The other endpoints work on the live run like on any run (fields, profile, section, time
series, surface, product); kinds that need the target (`target`, `difference`, `anomaly_target`) return the usual `404` hint,
and `metrics/*`, `training`, `experiments` have nothing to show for it.

### `GET /api/runs/{run}/dates`
`{dates, target_dates, embedding_dates, first, last}` — which days have a prediction, a GLORYS target and an
embedding (for date pickers and timelines).

### `GET /api/runs/{run}/mask`
Ocean mask per depth: `{shape:[15,100,240], depths, encoding:"packbits-base64", order:"C", data:[15 strings],
n_ocean:[15]}`. Each string is the base64 of `np.packbits` (MSB first) of the row-major `(lat, lon)` mask of one
depth; bit 1 = ocean (finite in the prediction product). Decode in JS: `atob` → bytes → `(byte >> (7 - k)) & 1`.

### `GET /api/runs/{run}/fields`
Query: `date` (required), `kind` = `prediction` (default) | `target` | `climatology` | `difference`
(prediction − target) | `anomaly_pred` (prediction − climatology) | `anomaly_target`, `depth` or `depth_index`
(optional), `format` = `json` (default) | `f32`, `method` = `model` (default) | `ridge` | `model_<tag>`.

`method` changes the prediction behind `prediction`, `difference` and `anomaly_pred` (the response echoes it in `method` / `X-Method`). `label` follows one pattern for every method, the model included: `"<method label> prediction"`, `"<method label> minus GLORYS"`, `"<method label> anomaly (prediction minus climatology)"`, e.g. `"Ridge regression minus GLORYS"` or `"OceanEmbed (pretrained encoder) prediction"`; the other kinds are labelled `GLORYS reanalysis (target)`, `Harmonic climatology` and `Target anomaly (target minus climatology)`. `target`, `climatology` and `anomaly_target` do not depend on `method`.

- With a depth: one 2-D `(lat, lon)` field; without: the full `(depth, lat, lon)` volume.
- `target`, `difference`, `anomaly_target` → **404** when the day has no GLORYS target.

JSON response:
```json
{"run": "poc_trial", "date": "2024-03-16", "kind": "prediction", "method": "model", "label": "OceanEmbed (pretrained encoder) prediction",
 "units": "degC", "depth": 100.0, "depth_index": 7, "shape": [100, 240],
 "data": [[null, null, 24.113, ...], ...],
 "color_range": {"vmin": 14.2, "vmax": 27.9, "diverging": false},
 "color_range_per_depth": null,
 "stats": {"n_valid": 15342, "min": 12.1, "max": 28.4, "mean": 20.2, "std": 3.1},
 "has_target": true}
```
`stats` (single depth only) adds `rmse`, `bias`, `mae` for `kind=difference`. For a volume `depth` /
`depth_index` / `stats` are `null`, `data` is `[15][100][240]` and `color_range_per_depth` has 15 entries.

**Colour-range hints.** Temperature-like kinds (`prediction`, `target`, `climatology`) share one range = 1st–99th
percentile of prediction ∪ target of the same slice (volume: of the whole volume; per-depth list for each
level), so panels are directly comparable - **also across methods**: the range is always computed from the main model's prediction and the target. Diverging kinds (`difference`, `anomaly_*`) return a range symmetric
about zero (`vmin = -vmax`, 99th percentile of |x|; the two anomaly kinds share one limit) with
`diverging: true`; for a non-default `method` the limit comes from that method's own error, so to compare two methods' difference panels use the larger `vmax` for both. Use them as `zmin`/`zmax`. For ranges that hold over the whole prediction period use `/ranges` below instead of sampling day volumes.

### `GET /api/runs/{run}/ranges[?samples=24]`
Period-wide colour ranges per depth, for a "hold range" that must not change from day to day. One call per run, no day volumes needed:
```json
{"run": "poc_trial", "units": "degC", "depths": [0, 5, ...], "n_days_available": 365, "n_days_sampled": 24,
 "sampled_dates": ["2024-01-01", ...], "n_days_with_target": 24,
 "temperature": [{"vmin": 27.1, "vmax": 30.2, "diverging": false}, ...15 entries...],
 "temperature_volume": {"vmin": 6.9, "vmax": 29.8, "diverging": false},
 "methods": {"model": {"label": "OceanEmbed (pretrained encoder)", "kind": "model",
                       "difference": [{"vmin": -1.2, "vmax": 1.2, "diverging": true}, ...],
                       "anomaly": [{"vmin": -2.1, "vmax": 2.1, "diverging": true}, ...]},
             "ridge": {...}, "model_scratch": {...}},
 "method_note": "24 of 365 predicted days, evenly spaced ..."}
```
Method: `samples` (default 24, 2..60, capped at the number of predicted days) days evenly spaced over the predicted period (first and last included) are read once; per depth the finite values of all of them are pooled (at most 6000 evenly strided ocean values per day and depth, so the pool stays small). `temperature` (use for `prediction`, `target` and `climatology` panels, per depth; `temperature_volume` for a whole-volume view) is the 1st-99th percentile of the main model's prediction united with the GLORYS target (the target only on days that have one) - the same rule as the per-day hint, so a held range and a per-day range are comparable. Per method, `difference` is symmetric about zero with the 99th percentile of |prediction - target| of that method, and `anomaly` the symmetric 99th percentile of |prediction - climatology| united with |target - climatology| - the limit both `anomaly_pred` and `anomaly_target` use (like the per-day hints, whose limit is from one day). `difference` is `null` when the run has no GLORYS target, `anomaly` when it has no statistics file; `methods` has every method with day fields (`model`, `ridge`, ablations). The result is computed in memory the first time (measured cold: 2-5 s for the default 24 days on the three local runs, up to 8 s for `samples=60`; about 15 ms once cached), cached per run version (`run_meta.json` mtime) and sent with the usual ETag; nothing is written into the run folder, so the endpoint also works on a run that is still being produced (404 `this run has no predictions yet` before the first product file exists).

**Binary transport** (`format=f32`, or `Accept: application/octet-stream`; an explicit `format` wins): the body is
the raw array, **little-endian float32, C order, NaN preserved**, `15·100·240·4 = 1 440 000` bytes for a volume
(`100·240·4` for one level). Metadata is in headers (all readable cross-origin):

| header | meaning |
|---|---|
| `X-Shape` | `15,100,240` or `100,240` |
| `X-Dtype`, `X-Byte-Order` | `float32`, `little` |
| `X-Kind`, `X-Method`, `X-Date`, `X-Depth-Index` | echo (`X-Depth-Index` empty for a volume) |
| `X-Color-Range`, `X-Color-Diverging` | `vmin,vmax`; `1`/`0` |
| `X-Color-Range-Per-Depth` | volumes only: JSON `[[vmin,vmax], ...]` |
| `X-Has-Target` | `1`/`0` |

```js
const r = await fetch(`/api/runs/${run}/fields?date=${d}&kind=prediction&format=f32`);
const [nz, ny, nx] = r.headers.get("X-Shape").split(",").map(Number);
const vol = new Float32Array(await r.arrayBuffer());   // NaN = land / below sea floor
```

### `GET /api/runs/{run}/surface?date=`
`{run, date, shape:[100,240], fields:{sst|sss|sla|uo|vo|uw|vw: {long_name, units, data, color_range}}}` — the 7
surface inputs in physical units (degC, PSU, m, m s-1 ×4); `sla`/currents/winds have a diverging range. 404 if
the processed store is missing.

### `GET /api/runs/{run}/profile?date=&lat=&lon=[&method=]`
`{method, method_label, lat, lon, lat_index, lon_index, requested{lat,lon}, is_ocean, depths[15], prediction[15], target[15],
climatology[15], has_target, units}` at the nearest cell. `prediction` is the requested `method`'s (model by default; call twice to overlay model and ridge - both are cached). `target` is all `null` when unavailable, everything
is `null` on land (`is_ocean: false`). The ridge profile of Argo locations is also in `/argo/profiles/{id}`.

### `GET /api/runs/{run}/section?date=&lat=` or `&lon=` `[&method=]`
Exactly one of `lat` (zonal section, `distance` = longitudes) or `lon` (meridional, `distance` = latitudes):
`{method, method_label, along, fixed, fixed_value, requested, distance, depths, prediction, target, climatology, difference
([depth][distance]), color_range, difference_range, has_target}`; `prediction` and `difference` follow `method`.

### `GET /api/runs/{run}/timeseries?lat=&lon=[&method=]`
Hovmöller at one cell over all predicted days: `{method, method_label, lat, lon, is_ocean, dates[T], depths[15], prediction[T][15],
target[T][15], climatology[T][15], rmse_by_day[T], bias_by_day[T], color_range}`. `rmse_by_day` is the RMSE over the depths of
that day at this cell (prediction − target); rows without a target are `null`. `climatology` is the harmonic climatology at the cell (`null` rows off the ocean mask; the whole field is `null` when the run has no statistics file) so the UI can show anomalies (`prediction − climatology`). A method that covers fewer days than the model has `null` rows for the others. Cold cost ≈ 1 s per point
(every daily chunk of the monthly files is decompressed), cached afterwards.

### `GET /api/runs/{run}/metrics/glorys`
Metric-major reshape of `metrics_glorys.json`, nothing dropped:
```json
{"run": "synthetic", "reference": "GLORYS", "metadata": {...}, "metric_names": ["n","rmse","bias","mae","corr_raw","corr_anom","skill_vs_clim"],
 "methods": [{"key": "model", "label": "OceanEmbed (pretrained encoder)", "kind": "model", ...}],
 "overall": {"model": {"n": 30769032, "rmse": 1.47, "bias": -0.17, ...}, "ridge": {...}},
 "pooled": {"model": {...}}, "pooled_range_m": [50, 200], "depths": [0, 5, ...],
 "per_depth": {"rmse": {"model": [..15 values..], "ridge": [...]}, "bias": {...}, "corr_anom": {"climatology": [null, ...]}},
 "per_basin": {"arabian_sea": {"overall": {...}, "pooled": {...}, "per_depth": {...}}, "bay_of_bengal": {...}},
 "daily_rmse": {"dates": [...], "depths": [...], "methods": {"model": [[...15 values...] per day]}},
 "daily": {"dates": [...], "depths": [...], "pooled_range_m": [50, 200], "rmse": {...}, "bias": {...},
           "corr_anom": {...}, "pooled_rmse": {"model": [one value per day]},
           "pooled_bias": {"model": [...]}, "pooled_corr_anom": {"model": [...]}},
 "per_year": null, "extra": {}}
```
Bias is method − reference. `daily` (vs GLORYS only, `null` for Argo) holds, per method, `rmse`, `bias` and `corr_anom` as `[day][depth]` and `pooled_rmse`, `pooled_bias` (method minus reference) and `pooled_corr_anom` as `[day]` over the pooled depth range (`pooled_range_m`, 50-200 m); `daily_rmse` is kept unchanged. A daily `corr_anom` is the *spatial* correlation of the anomalies (climatology removed) over the grid points of that day and depth, unlike `per_depth.corr_anom`, which is the temporal correlation per grid point pooled over the period; `pooled_corr_anom` is the same spatial correlation over the grid points of all pooled depths together of that day (one correlation, not an average of per-depth ones). All three pooled series come from the per-day, per-depth sums `evaluate` already accumulates. `bias`, `corr_anom` and `pooled_rmse` are absent from runs evaluated before they existed, `pooled_bias` / `pooled_corr_anom` from runs evaluated before 9e (re-run `oceanembed evaluate`; the older numbers do not change). The daily numbers equal the served fields: for ridge, the RMSE of `/fields?kind=difference&method=ridge` at a depth and day matches `daily.rmse.ridge` to 5e-6 degC; for the networks to < 1e-4 degC (fp16 autocast is not bit-identical between the batch layouts of `evaluate` and `predict`). `corr_anom` / `skill_vs_clim` of the climatology are `null`.

### `GET /api/runs/{run}/metrics/argo`
Same structure vs Argo (methods include `glorys`; no `daily_rmse`); `metadata` carries profile counts, the
interpolation rule and the **independence note** (GLORYS assimilates Argo and the model is trained on GLORYS),
`extra.gridded_argo` appears when an INCOIS gridded file was scored.

`per_year` (also in `/metrics/argo`) is `null` unless the evaluated period spans several calendar years (the final run: 2023 and 2024). It is `{ "2023": {n_days (or n_profiles, n_matchups for Argo), start, end, overall, pooled, per_depth, per_basin}, "2024": {...} }`, each block shaped like the whole-period fields above (metric-major), so a client can show each test year next to the pooled numbers; the whole-period fields are unchanged. The run detail's `model` block lists `inputs` (the surface variables the model uses), `input_groups`, `dropped_input_groups` and `main_init` (`pretrained` | `scratch`), and every entry of `products` has `used_by_model` (false for an input variable the model does not use).

### `GET /api/runs/{run}/metrics/maps/index` and `/metrics/maps`
`index` → `{metrics: {rmse: ["model","ridge",...], bias: [...], corr_anom: [...], corr_raw: [...],
skill_vs_clim: [...]}, depths, has_n_valid}` (only combinations that exist in `maps_glorys.nc`: `rmse`, `bias`, `corr_raw` and `skill_vs_clim` exist for every method including the climatology, `corr_anom` for every method except the climatology; maps written before this change only have `corr_raw` / `skill_vs_clim` for the model until `evaluate` is re-run).
`maps?metric=rmse&method=model&depth=100` → `{metric, method, label, depth, depth_index, units, shape, data
(lat×lon, per grid point over the evaluated period), color_range, range_info}`; the range is shared by every method at that
depth (rmse `[0, p99]`, bias symmetric about zero with `diverging: true` from the 99th percentile of |bias|, correlations `[p1, ≤1]`).

**Skill maps** (`skill_vs_clim`) are symmetric about zero and centred on it (0 = as good as the climatology). Skill is at most 1 but unbounded below, so the limit is the **95th percentile of |skill| over all methods at that depth, kept within 0.5 .. 3.0** (not the old fixed ±1, which saturated wherever skill was very negative). `range_info` states what the range does to the returned map: `{n_valid, data_min, data_max, n_below, n_above, fraction_outside, exceeds_range, limit_rule, limit_capped}` - `exceeds_range` is true when any value lies outside `color_range` (those points saturate the colour scale; show `data_min` / `fraction_outside` in a legend note), `limit_rule` (skill only; `null` otherwise) names the rule and `limit_capped` says whether the percentile was above the 3.0 cap. `range_info` is returned for every metric.

### `GET /api/runs/{run}/argo/matchups[?max_points=20000]`
Columnar rows, one per (profile, depth): `{n_total, n_returned, downsampled, methods:["model","ridge","clim",
"model_scratch"], columns:{profile_id, time (ISO, Z), lat, lon, depth, basin, obs, glorys, model, ridge, clim,
...}}`. If there are more than `max_points` (100–1 000 000) rows, evenly spaced rows of the stored order are
returned — deterministic: the rows at positions `round(linspace(0, n-1, max_points))` of the stored order (first and last row included), so the same request always returns the same rows and rows of one profile may be split; use `/argo/profiles` + `/argo/profiles/{id}` for whole profiles. Note the matchup column for the climatology is `clim`.

### `GET /api/runs/{run}/argo/profiles` and `/argo/profiles/{profile_id}`
List: `{n_profiles, n_total, n_returned, offset, limit, has_more, around, around_index, profiles:[{profile_id, time, lat, lon, basin, n_levels, rmse:{model, ridge, clim, climatology, glorys,
...}}]}` (per-profile RMSE vs the observation over its matched levels; the climatology is under `climatology`, the key every other endpoint uses, and still under `clim` for older clients - the two are identical, so iterate over one of them, and `sort=rmse&method=climatology` works as well as `method=clim`; the matchup columns and `methods` of `/argo/matchups` keep only `clim`). Query (all optional): `basin` (`arabian_sea` | `bay_of_bengal`), `start` / `end` (`YYYY-MM-DD`, inclusive days), bounding box `lat_min` / `lat_max` / `lon_min` / `lon_max` (degrees, inclusive, any subset; min > max is a 400), `around=<profile_id>` (the page that contains that profile under the current filters and sort: `offset` is ignored, the response's `offset` is the start of that page - `(around_index // limit) * limit` - and `around_index` the profile's 0-based position in the filtered, sorted list; a profile that is unknown or filtered out is a 404 whose message says which), `sort` = `time` (default) | `rmse` (RMSE of `method`, default `model`; profiles without a value come last), `order` = `asc` (default) | `desc`, `limit` (default **5000**, maximum 50000), `offset`. `n_profiles` is the number of matches of the filters (all profiles when none is given, so it equals `len(profiles)` as long as there are at most 5000), `n_total` the unfiltered count, `n_returned` the page length and `has_more` whether more matches follow `offset + limit`. Equal sort keys keep time order, so pages are reproducible; there is no silent sub-sampling - a request without `limit` on a run with more than 5000 profiles returns the first 5000 and `has_more: true`. Detail:
`{profile_id, time, lat, lon, grid_lat, grid_lon, basin, depth[], obs[], series:{model[], ridge[], clim[],
glorys[], ...}}` sorted by depth.

### `GET /api/runs/{run}/embeddings/dates`, `GET /api/runs/{run}/embeddings?date=[&n_components=3]`
`{shape:[n,25,60], data:(n,y,x) in 0..1 (n=3: R,G,B = PC1..PC3), explained_variance_ratio:[n], lat[y], lon[x],
cell_size:[dlat,dlon], ocean:(y,x) 0/1}`. `n_components` (1..8, default 3; at most the embedding size, else 400) returns more components, each scaled 0..1 on its own, with the variance fraction of each in `explained_variance_ratio` (the first three are identical whatever `n_components` is). The PCA is fitted once per run on a day sample, so colours are
comparable between days. `ocean` is the embedding-resolution land mask (a cell is ocean if any surface ocean
cell maps to it). 404 with a hint when the run has no embeddings or the day is not in the test split.

### `GET /api/runs/{run}/embeddings/similar?date=&lat=&lon=[&center=false]`
Cosine similarity between the embedding vector of the cell nearest to (`lat`, `lon`) and **every cell**, over all `emb_dim` features (not the PCA components): `{date, requested{lat,lon}, lat, lon, y_index, x_index, emb_dim, centered, reference_is_ocean, shape:[y,x], data:(y,x) in -1..1, similarity_range{min,max}, lat_values[y], lon_values[x], cell_size, ocean}`. `lat` / `lon` are the snapped embedding-cell centre, `data[y_index][x_index]` is 1 and `similarity_range` covers the other **ocean** cells. **`data` is `null` on land cells** (`ocean` = 0; their embedding carries no information); if the reference cell itself is land, `reference_is_ocean` is `false` and its own value is `null` too. `center=true` subtracts the run's mean embedding first (the mean of the PCA sample). Raw cosine values are not uniformly high: measured over ocean cells for about 40 random reference cells on days across the test split, the median similarity of a reference to the other ocean cells is about 0.4-0.5 (`synthetic` 0.43, `poc_trial` 0.51; it varies by reference from 0.03 to 0.67), the 5th percentile about 0.07, the minimum about -0.5, and the maximum (the nearest neighbour) is up to 0.99. Centring shifts the bulk down and spreads it: `synthetic` median 0.35 (barely changed), `poc_trial` median 0.07 (range -0.12..0.21 by reference), minima -0.56 and -0.81. Same date rules and errors as `/embeddings` (400 for a point outside the grid, 404 for a run without embeddings). A few ms per request.

### `GET /api/runs/{run}/training`
`{logs: {pretrain|train|train_<tag>: {kind, tag, n_epochs, best{epoch, val_*}, columns:{epoch:[...],
train_loss:[...], val_rmse:[...], val_rmse_per_depth:[[...]], ...}}}}` — straight from `logs/*.jsonl`.

### `GET /api/runs/{run}/experiments` and `GET /api/compare?runs=a,b`
Experiments table, one row per method / ablation: `{method, label, kind, pretrained, epoch, val_rmse,
glorys_overall, glorys_pooled, glorys_depths{"0","50","100","200","500"}, argo_overall,
pooled_rmse_gain_vs_climatology_pct, pooled_rmse_gain_vs_ridge_pct}` plus `pooled_range_m`,
`selected_depths`, `notes` (synthetic warning, bias convention, Argo independence). `/api/compare` (≤ 8 runs,
unknown run → 404) returns `{runs:[{name, data_source, test_period, n_days, n_argo_profiles, experiments}]}`.

### `GET /api/runs/{run}/report`, `GET /api/runs/{run}/figures/{name}`
`{markdown, figures:[{name, url, size}]}`; the markdown references `figures/<name>.png`, fetch the image from
`/api/runs/{run}/figures/{name}` (`image/png`; the name must be one of the listed files).

### `GET /api/runs/{run}/product`, `GET /api/runs/{run}/product/{file}[?method=]`
`{files:[{name, size, month, start, end, n_days, url}], total_size, extra_products:[{method, label, kind, files, total_size}]}` and the download of one monthly CF-1.8
NetCDF (`application/x-netcdf`, `Content-Disposition: attachment; filename=...`; the name must be listed).
`files` is the main model's product, unchanged; `extra_products` lists the ridge baseline (`method: "ridge"`, label `Ridge regression`, `kind: "baseline"`) and tagged ablations (`model_scratch`, `OceanEmbed (no pretraining)`, `kind: "ablation"`), each with its own files (`kind` is `baseline` | `ablation`; the same values - plus `model` for the main product - appear in `field_methods[].kind`). The monthly file names repeat across methods, so the `url` of an extra file ends in `?method=<key>`; without it `/product/{file}` serves the main model's file.

## Not in the API (known gaps)

- Day fields of the climatology / GLORYS as a `method` (use `kind=climatology` / `kind=target`).
- Methods exist only where `run-all` (or `predict --ridge` / `predict --tag`) wrote the product; runs made before the ridge / ablation products were introduced list only `model` until those commands are run.
- Prediction of days outside the NetCDF product (the API never runs the model).
- Zonal / meridional means or other aggregations: the client can compute them from `fields` volumes.
