# Phases — OceanEmbed

Living execution plan. Status: `[ ]` todo, `[~]` in progress, `[x]` done and verified.

## Phase 0 — Scaffold
- [x] Git repo on feature branch, `.gitignore`, `pyproject.toml`, venv with CUDA PyTorch
- [x] Config models + YAML configs, canonical grid module, Typer CLI skeleton
- [x] `ruff` + `pytest` wired up and passing

**Success:** `pip install -e .[dev]` works; `ruff check .`, `ruff format --check .`, `pytest` pass; `torch.cuda.is_available()` is True in the venv.

## Phase 1 — Data pipeline
- [x] Synthetic provider writing raw products at native resolutions
- [x] Real providers: CMEMS (OSTIA, SSS, DUACS, GLORYS), PO.DAAC (OSCAR, CCMP), Argo
- [x] Regridding / temporal / vertical harmonisation → Zarr on the canonical grid
- [x] Train-split statistics and harmonic climatology
- [x] Torch dataset

**Success:** `oceanembed synth` + `harmonize` + `stats` run end-to-end on the synthetic config; regrid functions unit-tested (block mean, bilinear, vertical interp); real provider dataset IDs checked against the live catalogues.

## Phase 2 — Satellite embedding engine
- [x] Hybrid CNN + Transformer encoder, masked-surface pretraining, embedding export

**Success:** pretraining loss decreases on synthetic data; masked reconstruction beats a mean-fill baseline; embeddings export to Zarr.

## Phase 3 — Reconstruction model
- [x] Decoder, masked losses, training loop with AMP, checkpointing, baselines (climatology, ridge)

**Success:** on the synthetic test split the model's RMSE is below both climatology and ridge at thermocline depths; fits in 6 GB GPU memory.

**Result (synthetic, single seed):** masked-surface MSE 0.193 vs 0.917 mean-fill; test RMSE pooled 50–200 m: model 2.22 °C, ridge 2.64, climatology 3.39. Pretraining gave no clear gain over training from scratch (2.19) and the supervised stage overfits with only 2 synthetic train years — to be re-judged on the 5-year real-data run.

## Phase 4 — Evaluation & validation
- [x] Metrics vs target on the test split (per depth, per basin, maps)
- [x] Argo collocation + validation
- [x] Figures + markdown report

**Success:** `evaluate`, `validate-argo`, `report` produce every file in the architecture.md output contract.

## Phase 5 — Inference product & orchestration
- [x] CF-compliant NetCDF product, `run-all`, dashboard data-access layer

**Success:** `oceanembed run-all --config configs/synthetic.yaml` completes from scratch; NetCDF opens with xarray and has correct coords/attrs.

## Phase 6 — PoC dashboard
- [x] Streamlit dashboard: overview, 3-D explorer, profiles & sections, validation, embeddings

**Success:** dashboard loads a finished run with no errors; verified in the browser.

## Phase 7 — Documentation & final verification
- [x] README (setup, data access, commands), runbook for the real-data run (`docs/runbook.md`)
- [x] Full lint + tests + clean synthetic run from scratch
- [x] Project made self-contained: documents in `docs/`, `.venv/`, `data/`, `outputs/` inside the project folder

**Result (fresh from-scratch synthetic run, 2547 s):** climatology and ridge reproduce exactly (3.39 / 2.64 °C pooled 50–200 m); the pretrained model came out at 2.28 °C (2.22 in the earlier run — GPU nondeterminism, single seed) and the no-pretraining ablation at 2.18 °C. Pretraining has not shown a gain on synthetic data.

## Phase 8 — Real-data PoC run
- [x] Copernicus Marine and NASA Earthdata logins in place; both verified working (2026-10-02)
- [x] OSCAR / CCMP: per-granule OPeNDAP subsetting is the default (198 KB and 755 KB per day instead of ~33 MB), resumable, with automatic fallback to global granules; Harmony HOSS rejected (fails on OSCAR, ~7× slower on CCMP)
- [x] Trial on 3 months (`configs/poc_trial.yaml`): all seven products downloaded (~575 MB), harmonised data sanity-checked, full chain + Argo validation + dashboard ran. Trial skill numbers are plumbing-only.
- [x] Full run 2018-01-01..2024-12-15 (`configs/poc.yaml`): all six products downloaded (84/84 months each, ~14.7 GB), harmonised to 2 541 gap-free days, pretrained, trained, ablation trained, predicted (model / ridge / ablation) and evaluated against GLORYS on the 350 test days of 2024
- [x] Argo validation + report on the real run: 2 826 profiles, 40 059 matchups; pooled 50–200 m RMSE vs Argo — GLORYS 1.08 (the floor), model 1.41, ridge 1.53, climatology 1.67 °C
- [x] Pretraining decision: keep the pretrained encoder as the documented design, and report plainly that it gives no measurable skill gain at this data scale (see result below)

**Result (real data, test year 2024, vs GLORYS, single seed):**

| Pooled 50–200 m | RMSE °C | anomaly corr. | skill vs climatology |
|---|---|---|---|
| OceanEmbed (pretrained) | 1.08 | 0.64 | 0.40 |
| No pretraining (ablation) | 1.04 | 0.67 | 0.44 |
| Ridge regression | 1.15 | 0.56 | 0.31 |
| Climatology | 1.39 | — | 0 |

- RMSE by depth (model / climatology): 0 m 0.57 / 0.79, 50 m 0.92 / 1.18, 100 m 1.22 / 1.66, 200 m 0.79 / 0.89, 300 m 0.53 / 0.55, 500–1000 m 0.36–0.41 / 0.35–0.38 — no skill below ~300 m.
- Basins (pooled 50–200 m skill): Bay of Bengal 0.55, Arabian Sea 0.29.
- Surface cold bias of −0.3 °C remains (climatology −0.5 °C): 2024 was warmer than every training year.
- Pretraining: masked-surface error 0.16 vs 1.04 mean-fill, but downstream the from-scratch model is ~3 % better (1.04 vs 1.08 °C) — within the run-to-run variation seen earlier, so "no measurable gain", not "worse". A multi-seed study would be needed to say more.

## Phase 9 — Final dashboard
- [x] 9a data API (FastAPI, `oceanembed serve`), 9b React dashboard (light "printed atlas" design), 9c/9e API extensions, 9d selection bar + method comparison + user feedback applied
- [x] Overview redesigned as "result at a glance" (distinct from the Explorer), content pass on the real run, final screenshots on `poc`
- [x] Short README with screenshots; pipeline flowchart (`docs/flowchart/`, `docs/images/flowchart.png`)

## Phase 10 — Research
- [x] R1 rigour: `--seed`, plain U-Net and per-pixel MLP baselines, multi-seed runner, block-bootstrap intervals and paired comparisons (`oceanembed research r1`, `r1-report`); results in `docs/research/r1_rigour.md`
- [x] R4 external benchmark: ARMOR3D against Argo and GLORYS on the same matchups (`oceanembed research r4`); results in `docs/research/r4_armor3d.md`
- [x] R5 physical metrics: thermocline depth, 0–300 m heat content, skill by monsoon season, eddy regime and Bay of Bengal salinity (`oceanembed research r5`); results in `docs/research/r5_physical.md`
- [x] R2 more training years and a second test year (`oceanembed research r2`, run `poc_long`: train 2011–2021, test 2023 and 2024); results in `docs/research/r2_long_period.md` — 0.975 ± 0.007 °C (2023) and 1.004 ± 0.014 °C (2024) over 50–200 m against 1.519 and 1.394 for climatology; 11 years beat 5 by 0.05 °C; the per-pixel MLP is within 0.015 °C overall, better in the Arabian Sea and worse in the Bay of Bengal; still no skill below ~300 m
- [x] R3 input-variable attribution by depth; several days of surface history as input (`oceanembed research r3`); results in `docs/research/r3_attribution.md` — sea level is the one indispensable input for the thermocline, SST + sea level reproduce the full seven-variable result, salinity and currents add nothing (dropping them improves the model by 0.027 °C), and 3 or 7 days of history do not help or reach below 300 m
- [ ] R6 Argo-aware correction; R7 manuscript

**R1 result (5 seeds, pooled 50–200 m RMSE vs GLORYS, 2024):** no pretraining 1.052 ± 0.015, pretrained 1.064 ± 0.016, per-pixel MLP 1.094 ± 0.005, plain U-Net 1.124 ± 0.031, ridge 1.155, climatology 1.390 °C. Pretraining gives no benefit; the Transformer beats the U-Net (established); the advantage over a per-pixel MLP is small, not established for the whole domain, and comes from the Bay of Bengal thermocline.

**R4 result (vs Argo, pooled 50–200 m, identical samples):** ARMOR3D 0.63, GLORYS 1.08, OceanEmbed 1.41, ridge 1.53, climatology 1.67 °C. ARMOR3D ingests Argo, so it is not independent of the floats; the two references (ARMOR3D and GLORYS) themselves differ by 1.20 °C on the grid, as much as OceanEmbed's reconstruction error.

**R5 result:** D20 RMSE 13.8 m (climatology 17.6, ridge 15.0, per-pixel MLP 14.1); 0–300 m heat content 0.72 vs 1.02 ×10⁹ J m⁻². The eddy-regime hypothesis is not supported. The advantage of the spatial model over the per-pixel MLP is confined to the Bay of Bengal, largest in the winter monsoon (0.39 °C) and in the freshest surface water (0.34 °C in the freshest tercile, 0.04 in the saltiest), and absent in the summer monsoon and in the Arabian Sea.

## Phase 11 — Live nowcast (an added section; the evaluated results stay untouched)

Starts after R2 and R3 have finished, so that it uses the final headline model and does not change code under running jobs.

- [ ] L1 Near-real-time inputs: `configs/live.yaml` with the NRT counterparts of the seven surface fields (dataset ids and publication delays verified against the live catalogues), a rolling window of recent days, the same grid and the training statistics of the headline model
- [ ] L2 `oceanembed live update`: fetch the days that are newly available (resumable), harmonise, reconstruct with the headline model, append to a rolling `live` run that follows the normal run contract; record per day which inputs were available and how old they were
- [ ] L3 Honesty checks before anything is shown as "live": (a) input shift — for an overlap period run the model on reprocessed and on NRT inputs and measure the difference by depth; (b) running verification — score each new day against Argo profiles as they are published, and against the operational Mercator analysis where available; rolling 30-day error next to climatology
- [ ] L4 Dashboard: a sixth view, "Live" — latest reconstructed day, freshness of each input, rolling accuracy, clearly labelled as a same-day reconstruction (nowcast), not a forecast; the five existing views and their numbers do not change
- [ ] L5 Operation: one command to update; an optional scheduled daily run that the owner enables; documented in the runbook
- [ ] L6 README: a short "Live mode" section at the end

**Success:** `oceanembed live update` brings the window up to the latest available day from a cold start and from a previous state; the Live view shows the date of each input and the rolling error; the input-shift check is reported with numbers.

## Notes

- Synthetic results only prove the pipeline works; they are never to be presented as scientific skill.
- Known data quirks left unfiltered: two GLORYS cells south of Socotra near 0 °C at 318–454 m; SSS capped at 40 in the Persian Gulf.
- Argo fetch needed two fixes on the full period: retries, and a scipy fallback for small NetCDF-3 responses that netCDF4 refuses.
- GLORYS is warm-biased against Argo at 100–150 m in 2024 (+0.5–0.8 °C; +0.15 °C in 2018, checked on raw files), so every method inherits that bias against the floats.
- Possible follow-ups, not started: multi-seed study for the pretraining question; uncertainty estimates; temporal context (several days of input); INCOIS gridded-ARGO comparison (needs a user-supplied file).
- Project is complete against the PRD. Research follow-up: see the staged plan (multi-seed runs, stronger baselines, ARMOR3D benchmark, physical metrics).
