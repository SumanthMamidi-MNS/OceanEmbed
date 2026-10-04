# Model card - OceanEmbed `final`

## What it is

A convolutional + Transformer encoder-decoder (3.7 M parameters) that reconstructs the **daily subsurface ocean temperature**
(15 levels, 0 - 1000 m) from **sea surface temperature and sea level anomaly** of the same day. It was trained from random
weights (no pretraining; pretraining gave no benefit in the project's research stage R1). Output: the temperature anomaly
relative to a harmonic climatology, decoded to degC on a 0.25 degree grid.

| Item | Value |
|---|---|
| Domain, grid | 5 - 30 N, 45 - 105 E (Arabian Sea, Bay of Bengal, northern Indian Ocean), 0.25 degree, daily |
| Levels (m) | 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 |
| Inputs | SST (OSTIA L4), sea level anomaly (DUACS L4), day of year, latitude / longitude, ocean mask. Salinity, currents and winds exist in the pipeline but this model does not use them (selected on the validation year: `docs/research/final_inputs.md`) |
| Training target | GLORYS12 reanalysis, regridded to 0.25 degree |
| Training period | 2011-01-01 - 2021-12-31 (4 018 days); checkpoint chosen on 2022 (validation RMSE 0.686 degC, epoch 3) |
| Test period | 2023-01-01 - 2024-12-15 (715 days), never used for any choice |
| Files | `recon.pt` (the model), `stats.nc` (input normalisation, harmonic climatology, anomaly scale, ocean mask), `ridge.joblib` and `mlp.pt` (baselines), `manifest.json` (sizes, SHA-256) |
| Licence | MIT, as the repository |

## Intended use

Same-day reconstruction of the temperature profile **inside the domain above, 0 - 300 m**, from the two satellite fields, as a
research prototype and a baseline. Not for navigation, safety or operational forecasting; not a forecast (it needs the day's
own surface fields); not tested outside 5 - 30 N, 45 - 105 E or outside the 2011 - 2024 satellite era.

## Measured skill (test 2023 - 2024, against GLORYS)

Pooled over 50 - 200 m (the thermocline), RMSE in degC; skill = 1 - MSE / MSE of the climatology.

| Method | 2023 | 2024 | Both years | Skill (both) |
|---|---|---|---|---|
| **OceanEmbed (this model)** | 0.974 | 0.991 | 0.983 | 0.55 |
| Per-pixel MLP | 0.984 | 1.043 | 1.013 | 0.52 |
| Ridge regression | 1.280 | 1.149 | 1.218 | 0.30 |
| Climatology | 1.519 | 1.394 | 1.459 | 0 |

Both years, by basin (50 - 200 m): Arabian Sea 1.002, Bay of Bengal 0.949 (climatology 1.350 and 1.649).
Training-seed spread for the same configuration is about 0.003 degC (three seeds, research stage `final_inputs`).

By depth, both years (RMSE of this model / climatology, degC): 0 m 0.47 / 0.81; 30 m 0.64 / 0.94; 100 m 1.12 / 1.76;
200 m 0.74 / 0.93; 300 m 0.50 / 0.53; 500 m 0.345 / 0.345; 1000 m 0.37 / 0.36.

## Limits - read before use

- **No skill below about 300 m.** At 500 - 1000 m the model is as good as the climatology or marginally worse (skill -0.01 to
  -0.06). This is a limit of what the surface fields carry on daily scales, not something more training removed (research
  stages R2, R3).
- **The reference is a reanalysis, not observations.** The model learns GLORYS, errors included. Against Argo floats the
  model's RMSE over 50 - 200 m is 1.32 degC (GLORYS itself: 1.05), and GLORYS is itself about 0.4 - 0.5 degC warmer than
  the floats in the thermocline. GLORYS assimilates Argo, so the Argo comparison is not independent of the training target.
- **A surface cold bias of about -0.2 degC** against GLORYS (the climatology's own is larger): the ocean has warmed relative
  to the training period.
- The result depends on the Bay of Bengal / Arabian Sea contrast: the spatial model is better in the Bay of Bengal, the
  per-pixel MLP is as good or better in the Arabian Sea.
- Land and below-sea-floor cells are NaN; the ocean mask is that of the training data.

## How to load and run

Command line (needs only the harmonised surface inputs of the requested days, see `docs/reproduce.md`):

```powershell
oceanembed predict --config configs/final.yaml --weights models/final --start 2024-06-01 --end 2024-06-30
oceanembed predict --config configs/final.yaml --weights models/final --start 2024-06-01 --end 2024-06-30 --mlp   # baselines: --ridge / --mlp
```

Python:

```python
from oceanembed.infer.predict import load_recon_model, model_predictor

model, cfg = load_recon_model("models/final/recon.pt")  # cfg.model.input_groups == ["sst", "sla"]
predict = model_predictor(model)  # (B, 12, H, W) standardised inputs -> anomalies
```

The statistics file converts to and from standardised units (`oceanembed.data.stats.Stats.load("models/final/stats.nc")`).
Output files are CF-1.8 NetCDF, one per month, variable `temperature` (degC).

Predicting from the released files reproduced the full run's products bit for bit for a whole month (March 2024); a few days
starting mid-month differ by at most 0.006 degC, because GPU half-precision results depend slightly on the batch layout.
