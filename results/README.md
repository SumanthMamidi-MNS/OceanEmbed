# Results

Small, tracked copies of what the runs produced, so the numbers survive deleting `data/` and `outputs/`. Everything here is generated: `oceanembed export-results --config configs/final.yaml` recreates this folder from `outputs/` (it needs the runs to be present there; see `docs/reproduce.md` for how to rebuild them). Nothing in this folder is edited by hand.

## `final/` - the main run

Final OceanEmbed model for the Indian Ocean (5-30 N, 45-105 E), 2011-2024: trained on 2011-2021, checkpoints chosen on 2022, tested on 2023 and 2024 (scored per year and together); the network is trained from scratch, without pretraining, and uses sea surface temperature and sea level anomaly only (selected on the validation year).

| File | What it is | Produced by |
|---|---|---|
| `final/metrics_glorys.json` | scores of every method against the GLORYS target: whole test period (`methods`), each calendar year (`per_year`), per depth, per basin, daily series | `oceanembed evaluate --config configs/final.yaml` |
| `final/metrics_argo.json` | the same against Argo profiles (`methods`, `per_year`) | `oceanembed validate-argo --config configs/final.yaml` |
| `final/run_meta.json` | the configuration the run used, its grid, splits and input groups | written by every pipeline command |
| `final/report.md`, `final/figures/` | the generated report and its figures | `oceanembed report --config configs/final.yaml` |

JSON files are written without indentation to save space (`python -m json.tool` makes them readable).

## `research/` - the research stages

Each folder holds `summary.md` (the tables), `summary.json` (the same numbers with their confidence intervals) and the figures of the stage.

| Folder | What it is | Produced by |
|---|---|---|
| `research/r1/` | R1: seeds, confidence intervals and stronger baselines, five training years | `oceanembed research r1 / r1-report --config configs/poc.yaml` |
| `research/r2/` | R2: eleven training years, two test years, learning curve, Argo | `oceanembed research r2 / r2-report --config configs/poc_long.yaml` |
| `research/r3/` | R3: variable attribution by depth and the effect of input history | `oceanembed research r3 / r3-report --config configs/poc.yaml` |
| `research/r4/` | R4: comparison with ARMOR3D, an observation-based product | `oceanembed research r4 --config configs/poc.yaml` |
| `research/r5/` | R5: derived physical quantities (D20, D23, heat content) and stratified skill | `oceanembed research r5 --config configs/poc.yaml` |
| `research/final_inputs/` | Input-set selection for the final model (decided on the validation year only) | `oceanembed research final-inputs / final-inputs-report --config configs/poc_long.yaml` |

The written-up interpretation of each stage is in `docs/research/`.
