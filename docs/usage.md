# Using OceanEmbed

Setup, commands, project layout and checks. The overview and results are in the
[README](../README.md); the real-data run is described step by step in [runbook.md](runbook.md).

## Setup (Windows 11, Python 3.12, NVIDIA GPU)

Everything lives inside the project folder: the virtual env `.venv\`, data in `data\`, run outputs in `outputs\`
(both git-ignored), documents in `docs\`. Run every command from the project root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CUDA build of PyTorch first, so that it is never replaced by a CPU wheel (cu126 = CUDA 12.6)
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # expect True
```

Version pins that matter (already in `pyproject.toml`): `zarr>=3`, `erddapy<3`, `pandas<3` (argopy 1.3 breaks with
erddapy 3), `streamlit>=1.64`. The project is OneDrive-synced: `.venv`, `data` and `outputs` can be several GB to
tens of GB, so pause sync or exclude those folders.

## Quick start on synthetic data (no logins)

```powershell
.\.venv\Scripts\oceanembed.exe run-all --config configs/synthetic.yaml     # about 45 min on an RTX 3050
.\.venv\Scripts\streamlit.exe run app/Home.py                              # dashboard at http://localhost:8501
.\.venv\Scripts\oceanembed.exe serve                                          # data API at http://127.0.0.1:8000/api/docs (see docs/api.md)
```

`run-all` chains: synth, harmonize, stats, pretrain, train, train-ablation, baseline, embed, predict, predict-ridge,
predict-ablation, evaluate, validate-argo, report. The two ablation steps (the no-pretraining
`train --no-pretrained --tag scratch` and its predictions) run when the config has
`ablation: {no_pretrained: true}` (all shipped configs do); the ridge product (`predictions/ridge/`) is always written. For a fast smoke test, `pytest` runs the whole chain on a tiny grid in a few minutes.

Outputs go to `outputs\synthetic\` (checkpoints, `predictions\oceanembed_T_<YYYYMM>.nc`, `metrics\`, `embeddings\`,
`figures\`, `report.md`); intermediate data to `data\raw_synthetic\` and `data\processed\`.

## Web dashboard (React)

The final dashboard is a single-page app in `web/` (Vite + React + TypeScript) that reads everything from the data API;
the Streamlit app above stays as a fallback. It needs Node 20+ (developed with Node 24 / npm 11) and no internet at
run time: fonts are bundled and the only requests go to `/api`.

```powershell
# once
cd web; npm install; npm run build; cd ..        # writes web\dist
.\.venv\Scripts\oceanembed.exe serve             # http://127.0.0.1:8000  (API under /api, dashboard at /)

# while developing the UI (two terminals)
.\.venv\Scripts\oceanembed.exe serve             # API on :8000
cd web; npm run dev                              # http://localhost:5173, proxies /api to :8000
```

Five views: Overview (the story with the selected run's own figures), Explorer (day and depth, linked maps, profile,
section, time-depth), Validation (GLORYS and Argo), Representation (the embedding) and Experiments (methods, runs,
training, data, NetCDF product, report). Run, day, depth and point are in the URL, so any state can be linked.
Design system: [`docs/design.md`](design.md). Checks: `npm run lint`, `npm run typecheck`, `npm test`,
`npm run build`; after an API change run `npm run gen:api` (with the API running) to regenerate the typed client.

## Real-data run

Follow [`docs/runbook.md`](runbook.md): create the free accounts, log in, try
`configs/poc_trial.yaml` (3 months) first, then `configs/poc.yaml` (2018-2024).

## CLI reference

All commands take `--config/-c <yaml>`; run `oceanembed <command> --help` for details.

| Command | What it does | Extra options |
|---|---|---|
| `synth` | write synthetic raw products at native resolutions | `--variable/-v` |
| `download` | download real raw products (CMEMS, PO.DAAC, Argo); skips existing monthly files | `--variable/-v sst sss sla currents winds temp argo` (repeatable) |
| `harmonize` | raw files to the harmonised Zarr store on the canonical grid | |
| `stats` | train-split input statistics, harmonic climatology, anomaly std | |
| `pretrain` | masked-surface self-supervised pretraining of the encoder | `--device` |
| `train` | supervised 15-depth reconstruction training | `--no-pretrained`, `--tag`, `--device` |
| `baseline` | fit the ridge baseline, report val RMSE of ridge and climatology | |
| `embed` | export embedding maps to `embeddings.zarr` | `--split`, `--checkpoint`, `--device` |
| `predict` | write the CF-1.8 NetCDF product, one file per month | `--split` or `--start`/`--end`, `--tag`, `--ridge`, `--device` |
| `evaluate` | metrics vs GLORYS (model, ablations, ridge, climatology) | `--split`, `--device` |
| `validate-argo` | collocate Argo profiles and score all methods | `--split`, `--device` |
| `report` | figures and `report.md` | `--date` |
| `run-all` | whole chain, stops at the first failing step | `--skip-existing`, `--device` |

## Project layout

```
configs/      synthetic.yaml (demo), poc.yaml (real, 2018-2024), poc_trial.yaml (real, 3 months), test_tiny.yaml
src/oceanembed/   config, grid, runmeta, cli, data_access; api/ (FastAPI data API); data/ (providers, regrid, harmonize, stats, dataset);
                  models/ (encoder, mae, recon, baselines); train/; eval/; infer/
app/          Streamlit dashboard: Home.py, pages/, ui/ (theme, figures, views, data, components); data_access.py re-exports the package module
web/          React dashboard (final): src/api (typed client), src/components (maps, charts, controls), src/views, src/lib; `npm run build` -> web/dist, served by `oceanembed serve`
tests/        pytest suite (runs on a tiny synthetic grid)
docs/         PRD, architecture, phases, decisions, design, memory (build log), runbook
data/, outputs/, .venv/   generated, git-ignored
```

Details: [`docs/architecture.md`](architecture.md) (module contracts, output contract),
[`docs/decisions.md`](decisions.md), [`docs/design.md`](design.md).

## Lint and tests

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
cd web; npm run lint; npm run typecheck; npm test; npm run build      # the React dashboard
```

## Requirements and where they are met

| PRD requirement / expected solution | Where in the code | Status |
|---|---|---|
| Preprocessing and harmonisation pipeline for multi-source data | `src/oceanembed/data/` (`providers/`, `regrid.py`, `harmonize.py`, `stats.py`, `dataset.py`) | Done; run on the full real period (2018-01-01 to 2024-12-15, 2 541 gap-free days) and on synthetic data |
| Standardise to 0.25 deg, daily | `grid.py`, `data/regrid.py`, `data/harmonize.py` | Done (block-mean / bilinear regridding, daily means) |
| Inputs: SST, SSS, SSH/SLA, currents (U, V), winds (U, V) | `config.py` (`default_products`), `data/providers/cmems.py`, `podaac.py` | Done (see datasets table) |
| Generate satellite embeddings with DL (CNN / ViT / autoencoder / attention hybrids) | `models/encoder.py`, `models/mae.py`, `train/pretrain.py`, `infer/embed.py` | Done: attention-based CNN + Transformer hybrid, masked-autoencoder pretraining, embeddings exported to Zarr |
| Reconstruction model: surface state to temperature profile | `models/recon.py`, `train/train.py` | Done |
| 15 standard depths 0-1000 m | `grid.py`, `config.py` (`STANDARD_DEPTHS`) | Done |
| Evaluate with independent observations and skill metrics (correlation, RMSE, bias, ...) | `eval/metrics.py`, `eval/evaluate.py`, `eval/argo_validation.py`, `eval/report.py` | Done against GLORYS (350 test days of 2024) and 2 826 Argo profiles (argopy); results in the README |
| Regrid products not available at the required resolution | `data/regrid.py` | Done (see datasets table) |
| Target: GLORYS temperature | `data/providers/cmems.py` (`temp`) | Done (GLORYS12, 84 monthly files) |
| In-situ validation: gridded ARGO (INCOIS LAS) | `data/providers/argo.py` | Argo **profiles** via argopy are the primary source; an optional INCOIS gridded NetCDF you download yourself is also scored if present |
| End-to-end pipeline | `cli.py` (`oceanembed run-all`) | Done; verified from scratch on synthetic data and run on the real data |
| Standardised daily 0.25 deg output | `infer/predict.py` | Done: CF-1.8 NetCDF, one file per month |
| Validation framework with independent ARGO | `eval/argo_validation.py` | Done (note: GLORYS assimilates Argo, see limitations) |
| Working PoC over Bay of Bengal / Arabian Sea | `web/` + `src/oceanembed/api/` (dashboard), per-basin metrics in `eval/` | Done on real data: run `poc`, per-basin metrics and dashboard |

## Datasets

All products are subset to the domain plus a 0.5 deg halo at download time and regridded to the canonical grid.

| Variable | Product | Dataset id (live catalogue, verified 2026-10-02) | Native resolution | Regridding to 0.25 deg daily |
|---|---|---|---|---|
| SST | OSTIA reprocessed L4 (doi 10.48670/moi-00168) | CMEMS `METOFFICE-GLO-SST-L4-REP-OBS-SST`, var `analysed_sst` | 0.05 deg, daily | 5 x 5 block mean |
| SSS | MULTIOBS SMAP/SMOS (doi 10.48670/moi-00051) | CMEMS `cmems_obs-mob_glo_phy-sss_my_multi_P1D` (NRT: `..._nrt_multi_P1D`), var `sos` | 0.125 deg, daily | 2 x 2 block mean |
| SLA | DUACS (doi 10.48670/moi-00145) | CMEMS `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`, var `sla` | **0.125 deg**, daily (PRD says 0.25 deg) | 2 x 2 block mean |
| Currents U, V | OSCAR L4 final v2.0 | PO.DAAC `OSCAR_L4_OC_FINAL_V2.0`, vars `u`, `v` | 0.25 deg, daily (OPeNDAP-subset granules) | bilinear (half-cell offset) |
| Winds U, V | CCMP v3.1 (the PRD also lists ASCAT coastal L2, not used: swath data) | PO.DAAC `CCMP_WINDS_10M6HR_L4_V3.1`, vars `uwnd`, `vwnd` | 0.25 deg, 6-hourly (OPeNDAP-subset granules) | daily mean, then bilinear |
| Temperature (target) | GLORYS12 reanalysis (doi 10.48670/moi-00021) | CMEMS `cmems_mod_glo_phy_my_0.083deg_P1D-m`, var `thetao` | 1/12 deg, daily, 36 levels <= 1100 m used | 3 x 3 block mean + linear vertical interpolation to the 15 depths |
| Validation | Argo profiles (argopy, ERDDAP, open access) | - | individual profiles | vertical interpolation, containing cell, same UTC day |
| Validation (optional) | INCOIS gridded ARGO (LAS) | user-supplied NetCDF in `data/raw/argo_gridded/` | gridded | regridded like the other products |

Differences from the PRD table: DUACS is now distributed at 0.125 deg (the pipeline regrids it); the SSS reprocessed
dataset ends 2024-12-15, so the PoC period is 2018-01-01 to 2024-12-15 (train 2018-22, val 2023, test 2024), all with the
same SSS product; the INCOIS LAS has no stable programmatic interface, so individual Argo profiles through argopy are used
as the primary in-situ source.
