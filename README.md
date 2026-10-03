<p align="center">
  <img src="docs/assets/oceanembed-logo.svg" alt="OceanEmbed" height="88">
</p>
<h3 align="center">
  <strong>3D Subsurface Ocean Temperature Reconstruction</strong><br>
  <small>Satellite Surface AI &bull; North Indian Ocean &bull; In-Situ Argo Validation</small>
</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="docs/assets/badges/python.svg" alt="Python 3.12" height="30"></a>
  <a href="https://pytorch.org/"><img src="docs/assets/badges/pytorch.svg" alt="PyTorch 2.x CUDA" height="30"></a>
  <a href="https://fastapi.tiangolo.com"><img src="docs/assets/badges/fastapi.svg" alt="FastAPI Data API" height="30"></a>
  <a href="web/"><img src="docs/assets/badges/dashboard.svg" alt="Dashboard: React 19 + TS" height="30"></a>
  <a href="docs/decisions.md"><img src="docs/assets/badges/offline.svg" alt="Engine: Zero Cloud APIs" height="30"></a>
  <br>
  <a href="docs/architecture.md"><img src="docs/assets/badges/model.svg" alt="Architecture: CNN + Transformer" height="30"></a>
  <a href="docs/backend.md"><img src="docs/assets/badges/resolution.svg" alt="Resolution: 0.25-deg Daily" height="30"></a>
  <a href="docs/backend.md#5-check"><img src="docs/assets/badges/argo.svg" alt="In-Situ Check: 2,826 Argo Floats" height="30"></a>
  <br>
  <a href="tests/"><img src="docs/assets/badges/tests.svg" alt="219 Py · 163 UI Passing" height="30"></a>
  <a href="LICENSE"><img src="docs/assets/badges/license.svg" alt="MIT License" height="30"></a>
</p>

<p align="center">
  Daily 3D ocean temperature (15 depths, 0 &ndash; 1,000&nbsp;m) for the Arabian Sea and Bay of Bengal,<br>
  estimated from satellite surface observations alone and validated against GLORYS reanalysis and in-situ Argo floats.
</p>

<p align="center"><img src="docs/assets/divider.svg" width="100%" height="1" alt=""></p>

<details open>
<summary><strong>Table of Contents</strong></summary>

1. [Why it matters](#why-it-matters)
2. [What it does](#what-it-does)
3. [Results](#results)
4. [Data](#data)
5. [Model](#model)
6. [How it is evaluated](#how-it-is-evaluated)
7. [A look inside](#a-look-inside)
8. [Quick start](#quick-start)
9. [Commands and API](#commands-and-api)
10. [Project structure](#project-structure)
11. [Engineering quality](#engineering-quality)
12. [Limitations and roadmap](#limitations-and-roadmap)
13. [Acknowledgements](#acknowledgements)
14. [License](#license)

</details>

<p align="center"><img src="docs/assets/divider.svg" width="100%" height="1" alt=""></p>

<p align="center">
  <strong>1.08&deg;C pooled RMSE</strong> across the thermocline (50&ndash;200&nbsp;m) &bull;
  <strong>0.64 anomaly correlation</strong> (vs 0.27 climatology) &bull;
  <strong>2,826 Argo float profiles</strong> independent in-situ validation &bull;
  runs offline on a laptop GPU
</p>

<p align="center">
  <a href="docs/images/overview.png"><img src="docs/images/overview.png" alt="OceanEmbed dashboard: reconstruction, reanalysis and difference at 100 m with the skill tables" width="100%"></a>
  <br>
  <sub>A typical day of the test year at 100&nbsp;m: reconstructed from the surface (left), the GLORYS reanalysis it never saw (centre), and the difference (right).</sub>
</p>

<p align="center"><img src="docs/assets/divider.svg" width="100%" height="1" alt=""></p>

## Why it matters

Subsurface temperature drives ocean heat content, stratification, marine heatwaves, cyclone intensification and fisheries.
It is measured by Argo floats, moorings and ships, which leave most of the ocean unsampled on any given day:
in all of 2024, 2 826 float profiles fell inside a domain of about 12 000 ocean grid cells — roughly eight a day.

Satellites do the opposite. They see every cell every day, but only its surface. The surface still carries a signature of
what lies beneath: a raised sea level means a deeper thermocline and warmer water at 100 m; eddies, salinity fronts and
wind mixing all leave marks.

OceanEmbed learns that link. It answers Smart India Hackathon problem statement 26066: *estimate the three-dimensional
ocean temperature using only surface satellite observations, daily, at 0.25°, and validate it against independent data.*

## What it does

<details>
<summary><strong>Architecture flowchart</strong></summary>

<p align="center">
  <a href="docs/images/flowchart.png"><img src="docs/images/flowchart.png" alt="OceanEmbed full pipeline flowchart with measured results" width="100%"></a>
  <br>
  <sub><b>End-to-end architecture flowchart</b>: from raw satellite ingestion and physical regridding to encoder-decoder training, multi-baseline validation against 2,826 Argo floats, and delivery.</sub>
</p>

</details>

<p align="center">
  <a href="docs/images/pipeline.png"><img src="docs/images/pipeline.png" alt="Pipeline: seven surface fields, harmonisation, embedding, decoding, validation, product" width="100%"></a>
  <br>
  <sub><b>The six pipeline stages</b>: 7 surface satellite fields &rarr; 0.25&deg; grid harmonisation &rarr; latent embedding &rarr; 3D anomaly decoding &rarr; Argo validation &rarr; CF-compliant NetCDF delivery.</sub>
</p>

| Step | What happens |
|---|---|
| **1. Input** | Seven daily satellite fields: sea surface temperature, salinity, sea level anomaly, surface currents (U, V), surface winds (U, V). |
| **2. Harmonise** | Six products at different resolutions and time steps are brought to one 0.25° daily grid: 100 × 240 cells, 2 541 gap-free days (2018 – 2024). |
| **3. Encode** | A CNN + Transformer encoder compresses each day's surface into a *satellite embedding*: 128 features on a 25 × 60 map of the basin. |
| **4. Decode** | A U-Net-style decoder predicts the temperature *anomaly* from the seasonal climatology at 15 standard depths. |
| **5. Check** | A whole unseen year is scored against the GLORYS reanalysis and Argo floats, always beside climatology, ridge regression and a no-pretraining ablation. |
| **6. Deliver** | Daily CF-compliant NetCDF files, metrics and a generated report, a data API, and an interactive dashboard. |

Everything runs from one command and one config file.

## Results

Scored on **2024, a year the model never saw** (350 days; trained on 2018 – 2022, checkpoints chosen on 2023).

<p align="center">
  <a href="docs/images/results.png"><img src="docs/images/results.png" alt="Results: error and skill by depth, thermocline comparison against GLORYS and Argo, skill by basin" width="100%"></a>
</p>

**The thermocline, 50 – 200 m** — where temperature varies most and the surface says least:

| Method | RMSE vs GLORYS (°C) | Anomaly correlation | Skill vs climatology | RMSE vs Argo floats (°C) |
|---|---|---|---|---|
| **OceanEmbed** | **1.08** | **0.64** | **0.40** | **1.41** |
| OceanEmbed, no pretraining (ablation) | 1.04 | 0.67 | 0.44 | 1.41 |
| Ridge regression | 1.15 | 0.56 | 0.31 | 1.53 |
| Climatology (no skill) | 1.39 | – | 0 | 1.67 |
| GLORYS reanalysis itself | – | – | – | 1.08 |

What the numbers say:

- **22 % lower error than climatology** over 50 – 200 m, and 26 % at 100 m; 7 % below ridge regression.
- **The same ordering holds against independent observations**: 2 826 Argo profiles, 40 059 matchups.
- **The basins differ**: skill is 0.55 in the Bay of Bengal and 0.29 in the Arabian Sea.
- **Skill ends near 300 m.** Below that the reconstruction is no better than climatology.
- **Pretraining gave no measurable gain.** The same network trained from scratch scored 1.04 °C against 1.08 °C, on one seed each.
- **The reference is not the truth.** GLORYS itself is 1.08 °C from the floats over 50 – 200 m in 2024, which is the floor for any model trained on it.

<details>
<summary><strong>RMSE and skill at every depth</strong></summary>

RMSE in °C against GLORYS over the 350 test days. Skill = 1 − MSE / MSE of climatology.

| Depth (m) | OceanEmbed | No pretraining | Ridge | Climatology | Skill (OceanEmbed) |
|---|---|---|---|---|---|
| 0 | 0.57 | 0.56 | 0.73 | 0.79 | 0.48 |
| 5 | 0.57 | 0.56 | 0.73 | 0.79 | 0.49 |
| 10 | 0.57 | 0.57 | 0.73 | 0.79 | 0.48 |
| 20 | 0.63 | 0.62 | 0.76 | 0.83 | 0.42 |
| 30 | 0.73 | 0.72 | 0.83 | 0.91 | 0.37 |
| 50 | 0.92 | 0.92 | 1.00 | 1.18 | 0.39 |
| 75 | 1.10 | 1.08 | 1.21 | 1.55 | 0.49 |
| 100 | 1.22 | 1.18 | 1.31 | 1.66 | 0.46 |
| 125 | 1.24 | 1.18 | 1.30 | 1.56 | 0.36 |
| 150 | 1.12 | 1.06 | 1.20 | 1.36 | 0.32 |
| 200 | 0.79 | 0.75 | 0.83 | 0.89 | 0.22 |
| 300 | 0.53 | 0.52 | 0.53 | 0.55 | 0.05 |
| 500 | 0.36 | 0.35 | 0.35 | 0.35 | −0.03 |
| 700 | 0.37 | 0.36 | 0.35 | 0.36 | −0.06 |
| 1000 | 0.41 | 0.41 | 0.38 | 0.38 | −0.15 |

</details>

A vertical section through both basins on one test day shows the reconstruction as a water column rather than a map:

<p align="center">
  <a href="docs/images/section.png"><img src="docs/images/section.png" alt="Vertical section along 15°N: reconstruction, GLORYS and difference" width="100%"></a>
  <br>
  <sub>Vertical section along 15&deg;N across both basins: reconstruction (top), GLORYS (middle), and difference (bottom).</sub>
</p>

## Data

All inputs are open products. The target is used only to train and to score; it is never an input.

| Variable | Product | Source | Native grid | Brought to 0.25° daily by |
|---|---|---|---|---|
| Sea surface temperature | OSTIA L4 | Copernicus Marine | 0.05°, daily | 5 × 5 block mean |
| Sea surface salinity | SMOS / SMAP multi-observation | Copernicus Marine | 0.125°, daily | 2 × 2 block mean |
| Sea level anomaly | DUACS altimetry | Copernicus Marine | 0.125°, daily | 2 × 2 block mean |
| Surface currents U, V | OSCAR v2.0 | NASA PO.DAAC | 0.25°, daily | bilinear |
| Surface winds U, V | CCMP v3.1 | NASA PO.DAAC | 0.25°, 6-hourly | daily mean, bilinear |
| Temperature (target) | GLORYS12 reanalysis | Copernicus Marine | 1/12°, 36 levels used | 3 × 3 block mean, interpolation to 15 depths |
| Independent check | Argo float profiles | Argo GDAC (argopy) | profiles | same cell, same day |

- **Domain:** 5 – 30°N, 45 – 105°E. **Depths (m):** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000.
- **Split by time, never at random:** train 2018 – 2022, validate 2023, test 2024. Neighbouring days look alike, so a random split would leak the answer.
- **Efficient downloads:** currents and winds are subset on NASA's servers (OPeNDAP), about 16 GB transferred for seven years instead of about 180 GB of global files. Downloads are resumable and logged.
- **Synthetic twin:** an analytic ocean with eddies, a thermocline and fake floats, written in the same file formats, lets the whole pipeline and test-suite run with no accounts.

## Model

<p align="center">
  <a href="docs/images/architecture.png"><img src="docs/images/architecture.png" alt="Model architecture: encoder with CNN stem and Transformer, embedding, decoder with skip connections" width="100%"></a>
</p>

- **12 input channels:** the seven standardised surface fields, the ocean mask, day of year (sin, cos), latitude and longitude.
- **Encoder (the embedding engine):** a three-stage CNN stem brings the 100 × 240 grid down to 25 × 60; six Transformer blocks let each of the 1 500 patches attend to the whole basin; the result is a 128-feature embedding map.
- **Masked pretraining:** half of the surface is hidden in 16 × 16 blocks and must be rebuilt from the embedding. It needs no subsurface labels. Rebuild error: 0.16 against 1.04 for filling in the mean.
- **Decoder:** upsampling with skip connections to the standardised anomaly at 15 depths; the harmonic climatology (mean + annual + semi-annual cycle per cell and depth) is added back.
- **Loss:** masked mean-squared error that ignores land and cells below the sea floor, plus a small vertical-gradient term.
- **Size and cost:** 3.7 M parameters; 14 min pretraining + 11 min supervised training on a 6 GB laptop GPU.

## How it is evaluated

| Question | How it is answered |
|---|---|
| Is it better than knowing the season? | Every score is shown beside the **climatology** fitted on the training years. |
| Is the deep model needed? | A per-pixel **ridge regression** on the same inputs is scored on the same days. |
| Does the pretraining help? | The same network is trained **without pretraining** and reported in every table. |
| Is the correlation real? | The **anomaly correlation** removes the climatology from both sides; the raw correlation is shown only for reference because seasons and depth inflate it. |
| Does it hold against observations? | **Argo profiles** are interpolated to the standard depths (no extrapolation, no interpolation across large gaps), matched to the same cell and day, and every method is scored on the identical sample. |
| Where does it fail? | Metrics are computed per depth, per basin, per grid cell and per day. |

One caveat is stated wherever Argo appears: GLORYS assimilates Argo, so the floats are independent of the model's *inputs* but not of its training target.

<a id="a-look-inside"></a><a id="the-dashboard"></a>
## A look inside

A React app served by a FastAPI data API. Every number on screen is computed from the selected run; nothing is hard-coded.

<table>
<tr>
<td width="50%" valign="top"><a href="docs/images/explorer.png"><img src="docs/images/explorer.png" alt="Explorer: linked maps at 100 m with depth rail and water-column profile"></a><br><sub><b>Explorer.</b> Any day, depth and point; linked maps at 100&nbsp;m with interactive depth rail and vertical profile of every method, section and time&ndash;depth view.</sub></td>
<td width="50%" valign="top"><a href="docs/images/explorer-compare.png"><img src="docs/images/explorer-compare.png" alt="Compare mode: three estimates beside GLORYS and their differences"></a><br><sub><b>Compare mode.</b> Three estimates beside GLORYS on a unified scale with daily error metrics.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="docs/images/validation.png"><img src="docs/images/validation.png" alt="Validation: pooled tables and metrics by depth"></a><br><sub><b>Validation.</b> Error, bias, anomaly correlation and skill by depth, per basin, as maps and as daily series.</sub></td>
<td width="50%" valign="top"><a href="docs/images/validation-argo.png"><img src="docs/images/validation-argo.png" alt="Argo validation: profile map, sortable list and an opened profile"></a><br><sub><b>Argo in-situ validation.</b> Geographic profile map of 2,826 floats, sortable matchup ledger, and individual water-column sounding.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="docs/images/representation.png"><img src="docs/images/representation.png" alt="Representation: embedding as a colour map, similarity map and correlation table"></a><br><sub><b>Representation.</b> What the 128-feature embedding encodes: principal components, similarity between places, and which surface field each component follows.</sub></td>
<td width="50%" valign="top"><a href="docs/images/experiments.png"><img src="docs/images/experiments.png" alt="Experiments: comparison table and ablation verdict"></a><br><sub><b>Experiments.</b> Side-by-side run comparisons, ablation verdicts, training curves, and downloadable CF-compliant NetCDF products.</sub></td>
</tr>
</table>

## Quick start

Requirements: Python 3.12, an NVIDIA GPU (6 GB is enough), Node 20+.

```powershell
# 1. environment (CUDA build of PyTorch first)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .[dev]

# 2. the whole pipeline on the synthetic ocean, no accounts needed (about 45 min)
.\.venv\Scripts\oceanembed.exe run-all --config configs/synthetic.yaml

# 3. the dashboard
cd web; npm install; npm run build; cd ..
.\.venv\Scripts\oceanembed.exe serve          # http://127.0.0.1:8000
```

**Real data** needs free Copernicus Marine and NASA Earthdata accounts. Try the three-month trial first, then the full run:

```powershell
.\.venv\Scripts\oceanembed.exe run-all --config configs/poc_trial.yaml    # 2024-01 to 2024-03, about 0.6 GB
.\.venv\Scripts\oceanembed.exe run-all --config configs/poc.yaml          # 2018 to 2024, about 16 GB
```

The step-by-step guide, with download sizes and troubleshooting, is in [docs/runbook.md](docs/runbook.md).

## Commands and API

<details>
<summary><strong>Command-line interface</strong></summary>

All commands take `--config <yaml>`.

| Command | What it does |
|---|---|
| `synth` | write the synthetic raw products at native resolutions |
| `download` | download real products (Copernicus Marine, PO.DAAC, Argo); resumable |
| `harmonize` | raw files → one Zarr store on the 0.25° daily grid |
| `stats` | training-period statistics and the harmonic climatology |
| `pretrain` | masked-surface pretraining of the encoder |
| `train` | supervised 15-depth reconstruction (`--no-pretrained` for the ablation) |
| `baseline` | fit the ridge regression baseline |
| `embed` | export the embedding maps |
| `predict` | write the NetCDF product, one file per month |
| `evaluate` | metrics against GLORYS for every method |
| `validate-argo` | collocate Argo profiles and score every method |
| `report` | figures and a markdown report |
| `run-all` | the whole chain; `--skip-existing` resumes |
| `serve` | the data API and the dashboard |

</details>

<details>
<summary><strong>Data API</strong></summary>

`oceanembed serve` exposes a read-only API under `/api` (OpenAPI docs at `/api/docs`): runs and their metadata, daily
fields as JSON or binary float32 volumes, profiles, sections, time series, metrics against GLORYS and Argo, error maps,
Argo profiles with filtering and paging, embeddings and their similarity, training curves, and the NetCDF product.
Reference: [docs/api.md](docs/api.md).

</details>

## Project structure

<details>
<summary><strong>Folders</strong></summary>

```
configs/            synthetic.yaml · poc.yaml (2018-2024) · poc_trial.yaml (3 months) · test_tiny.yaml
src/oceanembed/
  data/             providers (synthetic, Copernicus Marine, PO.DAAC, Argo), regridding, harmonisation, statistics, dataset
  models/           encoder, masked pretraining, reconstruction decoder, baselines
  train/            pretraining and supervised loops, losses
  eval/             streaming metrics, GLORYS evaluation, Argo validation, report
  infer/            NetCDF product, embedding export
  api/              FastAPI data API
  cli.py            the `oceanembed` command
web/                React + TypeScript dashboard (canvas map renderer, typed API client)
app/                earlier Streamlit dashboard, kept as a fallback
tests/              pytest suite, runs on a tiny synthetic grid
docs/               architecture, backend and frontend references, API, runbook, design system, decisions
```

</details>

## Engineering quality

- **Tested:** 219 Python tests (regridding against analytic answers, providers with mocked network, the full chain on a tiny grid, every API endpoint) and 163 UI tests.
- **Reproducible:** one config file per run; every figure and number is produced by a command; outputs follow a documented contract.
- **Honest by construction:** baselines and the ablation are part of the pipeline, not an afterthought; synthetic runs are labelled as demonstrations everywhere.
- **Careful with data:** temporal split, statistics from training years only, mask-aware losses and metrics, an identical sample for every method in the Argo comparison.
- **Runs on a laptop:** mixed precision, about 1.3 GB of GPU memory, streaming metrics that never hold the test year in memory.

## Limitations and roadmap

**Limitations**

- No skill below about 300 m on daily time scales.
- One training seed and one test year, so there are no error bars yet.
- Trained on a reanalysis, and inherits its bias against Argo (GLORYS is about 0.5 – 0.8 °C warmer than the floats at 100 – 150 m in 2024).
- A surface bias of about −0.3 °C in 2024, a year warmer than every training year.
- Single-day inputs: the model sees no history.
- No uncertainty estimate on the predictions.

**Roadmap**

- Several training seeds and confidence intervals on every number.
- Stronger baselines: a plain U-Net and a per-pixel network.
- A comparison with the operational ARMOR3D observation-based product.
- Several days of surface history as input, and attribution of skill to each surface variable by depth.
- Physical metrics: thermocline depth, upper-ocean heat content, skill by monsoon season.
- A correction towards Argo to remove the inherited reanalysis bias.

## Acknowledgements

- **Smart India Hackathon** and the **Ministry of Earth Sciences (MoES) / INCOIS** for problem statement **SIH26066** (*Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations*).
- **[Copernicus Marine Service](https://marine.copernicus.eu)** (OSTIA, multi-observation salinity, DUACS, GLORYS12) for high-resolution satellite products and global ocean reanalysis.
- **[NASA PO.DAAC](https://podaac.jpl.nasa.gov)** (OSCAR surface currents, CCMP winds) for physical surface observation datasets.
- The international **[Argo Programme](https://argo.ucsd.edu)** and **[argopy](https://argopy.readthedocs.io/)** for global in-situ profiling float measurements.

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

<p align="center"><img src="docs/assets/divider.svg" width="100%" height="1" alt=""></p>

<p align="center">
  Designed &amp; Developed by <a href="https://github.com/SumanthMamidi-MNS">Sumanth Mamidi</a><br>
  <sub>For Smart India Hackathon (SIH26066) &bull; Ministry of Earth Sciences (MoES) &bull; INCOIS</sub>
</p>
