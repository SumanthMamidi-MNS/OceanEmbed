# OceanEmbed

**The ocean below the surface, reconstructed from what satellites see above it.**

Satellites map the sea surface every day, but temperature below it is measured only where a float or ship
happens to be. OceanEmbed is a deep-learning pipeline that turns seven daily satellite surface fields into
temperature at 15 depths (0 – 1 000 m) on a 0.25° grid of the Arabian Sea and Bay of Bengal, and then
checks itself against a reanalysis and independent Argo floats.

Built for Smart India Hackathon problem statement 26066.

![Overview of the dashboard](docs/images/overview.png)

## Results

Scored on **2024, a year the model never saw** (350 days; trained on 2018 – 2022). Numbers are for the
thermocline, 50 – 200 m, where the surface says least about the interior.

| Method | RMSE vs GLORYS (°C) | Anomaly correlation | RMSE vs Argo floats (°C) |
|---|---|---|---|
| **OceanEmbed** | **1.08** | **0.64** | **1.41** |
| Ridge regression | 1.15 | 0.56 | 1.53 |
| Climatology (no skill) | 1.39 | – | 1.67 |

- **22 % lower error than climatology** over 50 – 200 m, 26 % at 100 m.
- Better than both baselines against 2 826 independent Argo profiles (40 059 matchups).
- Skill is stronger in the Bay of Bengal (0.55) than in the Arabian Sea (0.29).

What it does **not** do, stated as plainly as the wins:

- Below about 300 m it is no better than climatology.
- Masked pretraining of the encoder gave no measurable gain: the same network trained from scratch scored 1.04 °C.
- The training target (GLORYS) itself differs from Argo by 1.08 °C in the thermocline in 2024, which caps how well any model trained on it can match the floats.
- One training seed; no uncertainty estimate on the predictions.

## How it works

<img src="docs/images/flowchart.png" alt="Pipeline flowchart" width="560">

1. **Seven surface fields** — SST, salinity, sea level, currents (U, V), winds (U, V) from Copernicus Marine and NASA PO.DAAC.
2. **Harmonisation** — everything regridded to one 0.25° daily grid: 2 541 gap-free days, 2018 – 2024.
3. **Embedding engine** — a CNN + Transformer encoder compresses each day's surface into a 128-feature map of the basin.
4. **Reconstruction** — a U-Net-style decoder predicts the temperature anomaly at 15 depths; the seasonal climatology is added back.
5. **Validation** — a held-out year against the GLORYS reanalysis and Argo floats, always next to climatology, ridge regression and a no-pretraining ablation.

The model has 3.7 M parameters and trains in about 25 minutes on a 6 GB laptop GPU (14 pretraining, 11 supervised).

## The dashboard

A React app served by a FastAPI data API: explore any day, depth and point, compare methods, and inspect the validation.

| | |
|---|---|
| ![Explorer](docs/images/explorer.png) | ![Method comparison](docs/images/explorer-compare.png) |
| **Explorer** — linked maps, vertical profile, section, time–depth | **Compare** — reconstruction, baselines and GLORYS on one scale |
| ![Validation](docs/images/validation.png) | ![Argo validation](docs/images/validation-argo.png) |
| **Validation** — error, bias, correlation and skill by depth | **Argo** — every float profile against the reconstruction |
| ![Representation](docs/images/representation.png) | ![Experiments](docs/images/experiments.png) |
| **Representation** — what the embedding encodes | **Experiments** — methods, ablation, training curves, downloads |

## Quick start

Python 3.12, an NVIDIA GPU and Node 20+. Runs end to end on synthetic data with no accounts:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\oceanembed.exe run-all --config configs/synthetic.yaml
cd web; npm install; npm run build; cd ..
.\.venv\Scripts\oceanembed.exe serve          # dashboard at http://127.0.0.1:8000
```

The real-data run needs free Copernicus Marine and NASA Earthdata accounts (about 16 GB of downloads):
see [docs/runbook.md](docs/runbook.md).

## Tech stack

PyTorch · xarray / Zarr / NetCDF · FastAPI · React + TypeScript (Vite) · Typer CLI · pytest, ruff, Vitest.

## More

| | |
|---|---|
| [docs/usage.md](docs/usage.md) | setup, commands, project layout, tests |
| [docs/backend.md](docs/backend.md) | the whole backend at a glance: data, model, training, evaluation, API |
| [docs/frontend.md](docs/frontend.md) | the whole dashboard at a glance: views, state, rendering, design |
| [docs/architecture.md](docs/architecture.md) | design, data products, output contract |
| [docs/runbook.md](docs/runbook.md) | the real-data run, step by step |
| [docs/api.md](docs/api.md) | data API reference |
| [docs/decisions.md](docs/decisions.md) | why each technical choice was made |
| [docs/PRD.md](docs/PRD.md) | the problem statement |
