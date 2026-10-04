# Live nowcast — the released model on near-real-time inputs

`oceanembed live update --config configs/live.yaml` keeps a rolling window (60 days) of the newest days reconstructed from
**near-real-time (NRT)** satellite products with the released final model (`models/final`: CNN + Transformer, trained 2011 – 2021 on
SST and sea level anomaly only). It is a **nowcast of the same day as the inputs, not a forecast**, and it is not hourly: the inputs
are daily maps that appear a day or two late and are revised for a few days. Nothing here changes the evaluated runs or their numbers.
Produced on 2026-10-04; numbers below are from that day's runs (`outputs/live/`).

## Products and delays

The model uses SST and sea level anomaly only, so only those two have NRT counterparts in the live config. Salinity, currents and
winds are not used by the model and are not fetched.

| Input | NRT product (CMEMS) | Version | Grid | Catalogue range on 2026-10-04 | Delay |
|---|---|---|---|---|---|
| SST | `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` (OSTIA NRT), `analysed_sst` | 3.5 | 0.05°, daily, kelvin | 2024-01-17 .. 2026-10-02 | 2 days |
| Sea level anomaly | `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D` (DUACS NRT), `sla` | vNov2024 | 0.125°, daily, metres | 2024-07-01 .. 2026-10-04 | 0 days (the catalogue lists today's map; see limits) |

A day is reconstructed only when **both** inputs have data for it, so the newest day is set by SST: on 2026-10-04 it was 2026-10-02
(SST 2 days old, sea level already had 10-03 and 10-04, reported as *pending*); a few hours later, with 10-03 published, 1 day.
The reprocessed counterparts the model was trained on are `METOFFICE-GLO-SST-L4-REP-OBS-SST` (v3.2, to 2026-03-31) and
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` (to 2026-05-15). Same variables, units and grids; they differ only in
the processing version, which is what the input-shift check measures.

## How an update works

New days are downloaded as one file per day under `data/raw_nrt/` (resumable; a ledger line per request), harmonised with the batch
rules onto the 0.25° grid, reconstructed with the released weights and merged into monthly NetCDF files in `outputs/live/`
(`run_meta.json` with `data_source: "real"` and a `live` block). Days that leave the window are pruned from the live folders only.
Per day a provenance table (`live_days.parquet`) keeps the dataset and version of each input, its age when first used, the update
times and the revision size.

**Revision policy.** The newest 7 reconstructed days are fetched again on every update; a day whose inputs changed is reconstructed
again, and the size of the change against the *first-published* version (inputs and reconstruction) is recorded. In the first 13
checks (7 different days, 2 – 8 days old) **no day had changed**: the revision size was 0 for SST, sea level and the reconstruction. That is a
small sample (two updates), so it supports "revisions are small or absent after the first day" but not more.

**Cost.** A cold start (60 days) downloads 22 MB in 4 requests and takes 2 min on the GPU (0.03 s per day to reconstruct); on the CPU
0.52 s per day while other jobs were using the machine (6 min for the cold start). An incremental update downloads about 3 MB
(plus 10 MB for the verification analysis) and takes about 1.5 – 3 min, mostly the catalogue and Argo requests. Disk: about 40 MB in
`outputs/live`, 80 MB raw, 5 MB store.

## Check 1: do the NRT inputs change the reconstruction? (`live input-shift`)

Overlap 2025-10-01 .. 2026-03-31 (182 days; all four products and GLORYS exist). The released model is run on the reprocessed and
on the NRT inputs of the same days. The NRT files are the archive as it is now, so this measures the product difference, not the
first-publication error.

| | whole domain | Arabian Sea | Bay of Bengal |
|---|---|---|---|
| SST, NRT − reprocessed: bias / RMSE (°C) | −0.066 / 0.234 | −0.086 / 0.215 | −0.030 / 0.242 |
| Sea level, NRT − reprocessed: bias / RMSE (cm) | +0.45 / 1.88 | +0.45 / 1.46 | +0.46 / 2.29 |
| Reconstruction difference, 50 – 200 m: bias / RMSE (°C) | +0.032 / 0.171 | +0.029 / 0.142 | +0.036 / 0.213 |
| RMSE vs GLORYS, 50 – 200 m, reprocessed inputs (°C) | 1.036 | 1.064 | 0.982 |
| RMSE vs GLORYS, 50 – 200 m, NRT inputs (°C) | 1.026 | 1.062 | 0.959 |
| Change (NRT − reprocessed), paired block bootstrap, 95 % | −0.009 [−0.018, −0.003] | −0.002 [−0.009, +0.006] | −0.023 [−0.041, −0.006] |

By depth the two reconstructions differ by 0.09 – 0.10 °C RMS in the upper 30 m, 0.12 – 0.22 °C between 50 and 150 m (largest at
100 m, 0.29 °C in the Bay of Bengal) and 0.02 – 0.04 °C below 300 m. **The NRT-driven reconstruction is not measurably worse
against GLORYS; it is slightly better** (0.009 °C, interval excluding zero, in the Bay of Bengal 0.023 °C). The difference is within
what the intervals allow us to call real but is far smaller than the model's own error (about 1 °C), so the shift in the inputs is
not a source of error at this level. Figures: `outputs/live/checks/input_shift/figures/`.

## Check 2: running verification (part of `live update`)

Reconstruction and climatology scored every update against Argo profiles published so far (collocated like `validate-argo`) and,
when the subset is small (here 10 MB per day, 300 MB for the first 30 days), against the Copernicus operational analysis
(`cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m`) on the grid. The analysis is a model, not an observation, and not the GLORYS
reanalysis the model was trained on. Rolling 30-day figures for the window ending 2026-10-03 (RMSE in °C, model / climatology):

| Band | Argo (n pairs) | Operational analysis (n pairs) |
|---|---|---|
| 0 – 30 m | 0.96 / 1.26 (1 358) | 0.75 / 1.06 (1.7 M) |
| 50 – 200 m | 1.43 / 1.43 (1 634) | 1.24 / 1.53 (1.8 M) |
| 300 – 1000 m | 0.33 / 0.32 (1 042) | 0.36 / 0.37 (1.1 M) |

Argo profiles per day grow with the age of the day (4 – 20 per day within about a week; the verification re-fetches the newest 15
days). Reading: near the surface the nowcast is better than climatology against both references; **in the thermocline it is
better than climatology against the analysis but no better than climatology against Argo** in this one month, with a warm bias of
+0.83 °C against the floats (+0.70 °C against the analysis). This matches what is documented for the reanalysis the model was trained
on (GLORYS is 0.5 – 0.8 °C warmer than Argo at 100 – 150 m), so the model inherits it; it also means the live thermocline values are
not yet shown to be more accurate than the seasonal climatology against independent observations. Below 300 m there is no skill,
as in all evaluated runs.

## What the interface can claim

- A temperature profile for the newest day with data from both inputs, labelled **nowcast (same-day reconstruction), not a forecast**,
  with the date of each input and its age.
- That near-real-time inputs do not degrade the reconstruction relative to the reprocessed inputs it was trained on (Check 1).
- Rolling accuracy against Argo and the analysis as measured (Check 2), next to the climatology, with the Argo count; not an accuracy
  guarantee for the thermocline.

## Limits

- The catalogue lists today's DUACS map; whether the newest one or two sea level maps are provisional is not stated in the metadata.
  The revision check is the safeguard, and so far found no change, but only a few days have been checked.
- The input-shift check uses the NRT archive as it is now; the first-publication version of a day may differ more (bounded by the
  revision statistics, which are small so far).
- Verification covers one month (autumn); the Argo count per day is small (4 – 20) and early days have fewer profiles.
- Delays are those of one day's catalogue; they vary.
- The window is 60 days; no accuracy is claimed for older days, and nothing is forecast.
