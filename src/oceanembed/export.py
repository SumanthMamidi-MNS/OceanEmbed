"""Small, version-controllable copies of what a run produced, so results and weights survive
deleting ``data/`` and the large parts of ``outputs/``.

* :func:`export_results` -- ``results/``: the metrics JSON files, ``run_meta.json``, ``report.md``
  and its figures of a run, and the generated ``summary.md`` / ``summary.json`` (plus small
  figures) of the research stages. Re-running it recreates the folder from ``outputs/``.
* :func:`export_weights` -- ``models/<run>/``: the main checkpoint, the statistics file it needs
  (normalisation, climatology and the ocean mask), the ridge and per-pixel-MLP baselines and a
  ``manifest.json``; ``oceanembed predict --weights models/<run>`` runs from exactly these files.

Nothing here reads the large arrays (predictions, embeddings, the harmonised store) except the
store's ocean mask, which is copied into the statistics file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from oceanembed.config import Config
from oceanembed.data.stats import Stats
from oceanembed.infer.predict import WEIGHTS_STATS_FILE
from oceanembed.models.baselines import RIDGE_FILE
from oceanembed.models.pixel_mlp import MLP_FILE
from oceanembed.runmeta import RUN_META_FILE

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

# research stage -> (run whose outputs/<run>/research/<stage>/ holds it, what it is, command)
RESEARCH_STAGES: dict[str, tuple[str, str, str]] = {
    "r1": (
        "poc",
        "R1: seeds, confidence intervals and stronger baselines, five training years",
        "oceanembed research r1 / r1-report --config configs/poc.yaml",
    ),
    "r2": (
        "poc_long",
        "R2: eleven training years, two test years, learning curve, Argo",
        "oceanembed research r2 / r2-report --config configs/poc_long.yaml",
    ),
    "r3": (
        "poc",
        "R3: variable attribution by depth and the effect of input history",
        "oceanembed research r3 / r3-report --config configs/poc.yaml",
    ),
    "r4": (
        "poc",
        "R4: comparison with ARMOR3D, an observation-based product",
        "oceanembed research r4 --config configs/poc.yaml",
    ),
    "r5": (
        "poc",
        "R5: derived physical quantities (D20, D23, heat content) and stratified skill",
        "oceanembed research r5 --config configs/poc.yaml",
    ),
    "final_inputs": (
        "poc_long",
        "Input-set selection for the final model (decided on the validation year only)",
        "oceanembed research final-inputs / final-inputs-report --config configs/poc_long.yaml",
    ),
    "benchmark": (
        "poc_long",
        "Comparison study: boosted trees, random forest, plain U-Net and the other families",
        "oceanembed research benchmark / benchmark-report --config configs/poc_long.yaml",
    ),
}
RUN_FILES = ("metrics/metrics_glorys.json", "metrics/metrics_argo.json", RUN_META_FILE, "report.md")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _copy_json(src: Path, dst: Path) -> None:
    """Copy a JSON file in compact form (no indentation: about a third smaller)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(json.loads(src.read_text(encoding="utf-8")), separators=(",", ":")),
        encoding="utf-8",
    )


def _tree_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def export_results(cfg: Config, out: Path | str = RESULTS_DIR) -> dict:
    """(Re)create the results folder for ``cfg``'s run and the research stages.

    Returns ``{"files": [...], "missing": [...], "bytes": n}`` (paths relative to ``out``)."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = cfg.outputs_root
    files: list[str] = []
    missing: list[str] = []

    def done(path: Path) -> None:
        files.append(path.relative_to(out).as_posix())

    run_dir = cfg.outputs_dir
    for rel in RUN_FILES:
        src = run_dir / rel
        dst = out / cfg.run_name / Path(rel).name
        if not src.is_file():
            missing.append(f"{cfg.run_name}/{rel}")
            continue
        if src.suffix == ".json":
            _copy_json(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        done(dst)
    for fig in sorted((run_dir / "figures").glob("*.png")):
        dst = out / cfg.run_name / "figures" / fig.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig, dst)
        done(dst)

    for stage, (run, _, _) in RESEARCH_STAGES.items():
        src_dir = root / run / "research" / stage
        got = False
        for name in ("summary.md", "summary.json"):
            src = src_dir / name
            dst = out / "research" / stage / name
            if not src.is_file():
                missing.append(f"research/{stage}/{name} (from {src})")
                continue
            if src.suffix == ".json":
                _copy_json(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            done(dst)
            got = True
        if got:
            for fig in sorted((src_dir / "figures").glob("*.png")):
                dst = out / "research" / stage / "figures" / fig.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fig, dst)
                done(dst)

    (out / "README.md").write_text(_results_readme(cfg, set(files)), encoding="utf-8")
    done(out / "README.md")
    return {"files": sorted(files), "missing": missing, "bytes": _tree_size(out)}


def _results_readme(cfg: Config, present: set[str]) -> str:
    run = cfg.run_name
    cfgfile = f"configs/{run}.yaml"
    lines = [
        "# Results",
        "",
        "Small, tracked copies of what the runs produced, so the numbers survive deleting "
        "`data/` and `outputs/`. Everything here is generated: `oceanembed export-results "
        f"--config {cfgfile}` recreates this folder from `outputs/` (it needs the runs to be "
        "present there; see `docs/reproduce.md` for how to rebuild them). Nothing in this folder "
        "is edited by hand.",
        "",
        f"## `{run}/` - the main run",
        "",
        cfg.description or "",
        "",
        "| File | What it is | Produced by |",
        "|---|---|---|",
        f"| `{run}/metrics_glorys.json` | scores of every method against the GLORYS target: whole "
        "test period (`methods`), each calendar year (`per_year`), per depth, per basin, daily "
        "series "
        f"| `oceanembed evaluate --config {cfgfile}` |",
        f"| `{run}/metrics_argo.json` | the same against Argo profiles (`methods`, `per_year`) "
        f"| `oceanembed validate-argo --config {cfgfile}` |",
        f"| `{run}/run_meta.json` | the configuration the run used, its grid, splits and input "
        "groups | written by every pipeline command |",
        f"| `{run}/report.md`, `{run}/figures/` | the generated report and its figures "
        f"| `oceanembed report --config {cfgfile}` |",
        "",
        "JSON files are written without indentation to save space (`python -m json.tool` makes "
        "them readable).",
        "",
        "## `research/` - the research stages",
        "",
        "Each folder holds `summary.md` (the tables), `summary.json` (the same numbers with their "
        "confidence intervals) and the figures of the stage.",
        "",
        "| Folder | What it is | Produced by |",
        "|---|---|---|",
    ]
    for stage, (_, what, cmd) in RESEARCH_STAGES.items():
        have = f"research/{stage}/summary.md" in present
        lines.append(
            f"| `research/{stage}/` | {what}{'' if have else ' (not exported)'} | `{cmd}` |"
        )
    lines += [
        "",
        "The written-up interpretation of each stage is in `docs/research/`.",
        "",
    ]
    return "\n".join(lines)


def export_weights(cfg: Config, out: Path | str | None = None) -> dict:
    """Write ``models/<run>/``: ``recon.pt``, ``stats.nc`` (statistics + ocean mask),
    ``ridge.joblib``, ``mlp.pt`` and ``manifest.json``. ``MODEL_CARD.md`` is hand-written and is
    left alone.

    Returns the manifest."""
    out = Path(out) if out is not None else MODELS_DIR / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    ck = cfg.checkpoints_dir
    for name in ("recon.pt", RIDGE_FILE, MLP_FILE):
        src = ck / name
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found; run the training steps of {cfg.run_name}")
        shutil.copy2(src, out / name)
    stats = Stats.load(cfg.stats_path)
    with xr.open_zarr(cfg.zarr_path, consolidated=False) as store:
        mask = store["mask"].values.astype(bool)
        depth = np.asarray(store["depth"].values)
    ds = stats.to_dataset()
    ds["ocean_mask"] = (
        ("depth", "lat", "lon"),
        mask,
        {"long_name": "ocean and above the sea floor (the grid the model predicts on)"},
    )
    ds = ds.assign_coords(depth=depth)
    enc = {"clim_coef": {"zlib": True, "complevel": 4, "shuffle": True}}
    ds.attrs["description"] = (
        "Train-split statistics (input mean / std, harmonic climatology, anomaly std) and the "
        f"ocean mask of run '{cfg.run_name}'; read by `oceanembed predict --weights`."
    )
    ds.to_netcdf(out / WEIGHTS_STATS_FILE, encoding=enc)
    manifest = {
        "run": cfg.run_name,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_period": [str(cfg.split.train.start), str(cfg.split.train.end)],
        "validation_period": [str(cfg.split.val.start), str(cfg.split.val.end)],
        "test_period": [str(cfg.split.test.start), str(cfg.split.test.end)],
        "domain": {
            "lat": [cfg.grid.lat_min, cfg.grid.lat_max],
            "lon": [cfg.grid.lon_min, cfg.grid.lon_max],
            "resolution_deg": cfg.grid.resolution,
        },
        "main_init": cfg.model.main_init,
        "input_groups": list(cfg.model.input_groups),
        "input_variables": cfg.model.input_variables,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(out.iterdir())
            if p.is_file() and p.name not in ("manifest.json", "MODEL_CARD.md")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
