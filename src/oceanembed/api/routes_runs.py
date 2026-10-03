"""Run catalogue, run description, training logs, experiments, report, figures and product files."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import FileResponse

from oceanembed import data_access as da
from oceanembed.api import schemas as S
from oceanembed.api.deps import file_headers, get_store, reply, run_dep
from oceanembed.api.encode import array_to_list
from oceanembed.api.metrics import experiment_rows, method_infos
from oceanembed.api.store import FILE_NAME_RE, MAIN_METHOD, ApiError, Run, Store, method_kind
from oceanembed.eval.evaluate import method_label
from oceanembed.grid import BASINS

router = APIRouter()

SYNTHETIC_NOTE = "SYNTHETIC DATA: the numbers only demonstrate that the pipeline works; they are not scientific skill."
ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": S.ErrorResponse, "description": "Invalid parameter"},
    404: {"model": S.ErrorResponse, "description": "Unknown run, missing artefact or no data"},
}

# variable -> (long name, units, role, product key in the config, product title, native resolution, regridding)
_PRODUCTS = [
    ("sst", "Sea surface temperature", "degC", "input", "sst", "OSTIA L4 (UK Met Office)",
     "0.05 deg, daily", "5x5 block mean to 0.25 deg"),
    ("sss", "Sea surface salinity", "PSU", "input", "sss", "MULTIOBS SMOS/SMAP L4 (CMEMS)",
     "0.125 deg, daily", "2x2 block mean"),
    ("sla", "Sea level anomaly", "m", "input", "sla", "DUACS L4 (CMEMS)",
     "0.125 deg, daily", "2x2 block mean"),
    ("uo", "Eastward surface current", "m s-1", "input", "currents", "OSCAR L4 final v2.0 (PO.DAAC)",
     "0.25 deg, daily", "bilinear (half-cell offset)"),
    ("vo", "Northward surface current", "m s-1", "input", "currents", "OSCAR L4 final v2.0 (PO.DAAC)",
     "0.25 deg, daily", "bilinear (half-cell offset)"),
    ("uw", "Eastward 10 m wind", "m s-1", "input", "winds", "CCMP v3.1 (PO.DAAC)",
     "0.25 deg, 6-hourly", "daily mean, bilinear"),
    ("vw", "Northward 10 m wind", "m s-1", "input", "winds", "CCMP v3.1 (PO.DAAC)",
     "0.25 deg, 6-hourly", "daily mean, bilinear"),
    ("temp", "Sea water potential temperature (0-1000 m)", "degC", "target", "temp",
     "GLORYS12 reanalysis (CMEMS)", "1/12 deg, daily, 50 levels",
     "3x3 block mean + linear vertical interpolation to the 15 standard depths"),
]  # fmt: skip
BASIN_LABELS = {"arabian_sea": "Arabian Sea", "bay_of_bengal": "Bay of Bengal"}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _span(d: dict | None) -> dict:
    d = d or {}
    return {"start": d.get("start"), "end": d.get("end")}


def _glorys(store: Store, run: Run) -> dict | None:
    return store.cached(run, "m_glorys", lambda: da.load_metrics_glorys(run.path))


def _argo(store: Store, run: Run) -> dict | None:
    return store.cached(run, "m_argo", lambda: da.load_metrics_argo(run.path))


def figure_files(run: Run) -> list:
    d = run.file("figures")
    return sorted(d.glob("*.png")) if d.is_dir() else []


def product_files(run: Run, method: str = MAIN_METHOD) -> dict:
    return {f.name: f for f in da._prediction_files(run.path, method).values()}


TERMS_TEXT = {
    1: "a constant climatology (mean only)",
    3: "a mean-plus-annual-cycle climatology",
    5: "a mean-plus-annual-and-semi-annual-cycle climatology",
}


def _run_facts(store: Store, run: Run) -> dict[str, Any]:
    """Day counts per split, climatology terms, display label and description (cached per run)."""

    def build():
        try:
            counts = da.split_day_counts(run.path)
        except (OSError, KeyError, ValueError):
            counts = {}
        info = da.climatology_info(run.path) or {}
        terms = info.get("n_harmonic_terms")
        if terms is None:
            g = _glorys(store, run) or {}
            terms = (g.get("metadata", {}).get("climatology") or {}).get("n_harmonic_terms")
        terms = None if terms is None else int(terms)
        label, description = _describe(run, counts, terms)
        return {
            "n_train_days": counts.get("train"),
            "n_val_days": counts.get("val"),
            "n_test_days": counts.get("test"),
            "n_harmonic_terms": terms,
            "label": label,
            "description": description,
        }

    return store.cached(run, "run_facts", build)


def _describe(run: Run, counts: dict[str, int], terms: int | None) -> tuple[str, str]:
    """Display label and description: the config's ``label`` / ``description`` when set, else derived
    from the data source, period and splits."""
    cfg = run.meta.get("config") or {}
    synthetic = run.data_source == "synthetic"
    period = run.meta.get("time") or {}
    span = f"{str(period.get('start', '?'))[:7]} to {str(period.get('end', '?'))[:7]}"
    label = cfg.get("label") or f"{'Synthetic demo' if synthetic else 'Real data'}, {span}"
    if cfg.get("description"):
        return str(label), str(cfg["description"])
    split = run.meta.get("split") or {}

    def part(name: str, verb: str) -> str:
        sp = split.get(name) or {}
        n = counts.get(name)
        days = f" ({n} days)" if n is not None else ""
        return f"{verb} {sp.get('start', '?')} to {sp.get('end', '?')}{days}"

    text = (
        "Synthetic data that only demonstrates the pipeline"
        if synthetic
        else "Reanalysis, satellite and in-situ data"
    )
    text += (
        f" for {span}. {part('train', 'Trained on')}, {part('val', 'checkpoints chosen on')}, "
        f"{part('test', 'scored on')}."
    )
    if terms in TERMS_TEXT:
        text += f" Anomalies are relative to {TERMS_TEXT[terms]} fitted on the train split."
    return str(label), text


def summarize(store: Store, run: Run) -> dict[str, Any]:
    meta, grid = run.meta, run.meta.get("grid", {})
    logs = (
        sorted(p.stem for p in run.file("logs").glob("*.jsonl"))
        if run.file("logs").is_dir()
        else []
    )
    files = product_files(run)
    return {
        "name": run.name,
        "run_name": str(meta.get("run_name", run.name)),
        "data_source": run.data_source if run.data_source in ("synthetic", "real") else "unknown",
        "updated": meta.get("updated"),
        "period": _span(meta.get("time")),
        "split": {k: _span(v) for k, v in (meta.get("split") or {}).items()},
        "grid": {
            "resolution": grid.get("resolution"),
            "n_lat": grid.get("n_lat"),
            "n_lon": grid.get("n_lon"),
            "n_depth": len(grid.get("depths", [])),
            "lat": grid.get("lat"),
            "lon": grid.get("lon"),
            "depths": grid.get("depths", []),
        },
        "artefacts": {
            "predictions": bool(files),
            "metrics_glorys": run.file("metrics", "metrics_glorys.json").is_file(),
            "metrics_argo": run.file("metrics", "metrics_argo.json").is_file(),
            "argo_matchups": run.file("metrics", "argo_matchups.parquet").is_file(),
            "maps": run.file("metrics", "maps_glorys.nc").is_file(),
            "embeddings": run.file("embeddings", "embeddings.zarr").exists(),
            "report": run.file("report.md").is_file(),
            "training_logs": logs,
            "n_figures": len(figure_files(run)),
            "n_product_files": len(files),
        },
        "n_prediction_days": len(store.dates(run)),
        "note": SYNTHETIC_NOTE if run.data_source == "synthetic" else None,
        **_run_facts(store, run),
    }


# ----------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------
@router.get("/runs", response_model=S.RunsResponse, summary="List runs with a summary")
def list_runs(request: Request, store: Store = Depends(get_store)):
    runs = []
    for name in store.run_names():
        try:
            runs.append(summarize(store, store.run(name)))
        except (ApiError, OSError, ValueError):
            continue  # a half-written run folder is skipped, not an error
    return reply(request, {"runs": runs})


@router.get(
    "/runs/{run}",
    response_model=S.RunDetail,
    responses=ERRORS,
    summary="Full description of a run: grid, basins, dates, methods, config, data products",
)
def run_detail(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    g = run.grid
    glorys, argo = _glorys(store, run), _argo(store, run)
    summary = summarize(store, run)
    days = store.dates(run)
    cfg = run.meta.get("config", {})
    model = {
        "model": cfg.get("model"),
        "pretrain": {
            k: (cfg.get("pretrain") or {}).get(k) for k in ("epochs", "batch_size", "lr", "mask_ratio", "block")
        },
        "train": {
            k: (cfg.get("train") or {}).get(k)
            for k in ("epochs", "batch_size", "lr", "encoder_lr_scale", "patience", "vertical_grad_weight")
        },
        "inputs": ["sst", "sss", "sla", "uo", "vo", "uw", "vw"],
        "output": "temperature at the standard depths (standardised anomaly decoded to degC)",
    }  # fmt: skip
    products = []
    config_products = cfg.get("products") or {}
    for var, long_name, units, role, pkey, title, native, regrid in _PRODUCTS:
        ids = [
            d.get("id")
            for d in (config_products.get(pkey) or {}).get("datasets", [])
            if d.get("id")
        ]
        products.append(
            {
                "variable": var,
                "role": role,
                "long_name": long_name,
                "units": units,
                "product": title,
                "dataset_ids": ids,
                "native_resolution": native,
                "regridding": regrid,
            }
        )
    products.append(
        {
            "variable": "argo",
            "role": "validation",
            "long_name": "Argo temperature profiles",
            "units": "degC",
            "product": "Argo float profiles (argopy) / optional INCOIS gridded ARGO",
            "dataset_ids": [],
            "native_resolution": "individual profiles",
            "regridding": "vertical interpolation to the standard depths, nearest cell and same day",
        }
    )
    a_meta = (argo or {}).get("metadata", {})
    g_meta = (glorys or {}).get("metadata", {})
    emb_days = da.embedding_dates(run.path)
    split = run.meta.get("split", {})
    test = split.get("test", {})
    counts = {
        "n_prediction_days": len(days),
        "n_train_days": summary["n_train_days"],
        "n_val_days": summary["n_val_days"],
        "n_test_days": g_meta.get("n_days"),
        "n_embedding_days": len(emb_days) if len(emb_days) else None,
        "n_argo_profiles": a_meta.get("n_profiles_used"),
        "n_argo_matchups": a_meta.get("n_matchups"),
        "n_argo_profiles_loaded": a_meta.get("n_profiles_loaded"),
    }
    if counts["n_test_days"] is None and test.get("start") and test.get("end"):
        counts["n_test_days"] = int(
            (pd.Timestamp(test["end"]) - pd.Timestamp(test["start"])).days + 1
        )
    field_methods = store.field_methods(run)
    have_fields = {m["key"] for m in field_methods}
    methods = [{**m, "has_day_fields": m["key"] in have_fields} for m in method_infos(glorys, argo)]
    content = {
        "summary": summary,
        "grid": {
            **summary["grid"],
            "lat_values": array_to_list(g.lat, 6),
            "lon_values": array_to_list(g.lon, 6),
            "depth_values": array_to_list(g.depth, 3),
            "lat_edges": list(g.lat_edges),
            "lon_edges": list(g.lon_edges),
        },
        "basins": [
            {
                "key": k,
                "label": BASIN_LABELS.get(k, k),
                "box": dict(zip(("lat_min", "lat_max", "lon_min", "lon_max"), v, strict=True)),
            }
            for k, v in BASINS.items()
        ],
        "prediction_dates": [d.strftime("%Y-%m-%d") for d in days],
        "methods": methods,
        "field_methods": field_methods,
        "model": model,
        "training_summary": _training_summary(store, run),
        "products": products,
        "counts": counts,
        "metrics_metadata": g_meta or None,
    }
    return reply(request, content)


@router.get(
    "/runs/{run}/dates",
    response_model=S.DatesResponse,
    responses=ERRORS,
    summary="Days with a prediction, a GLORYS target and an embedding",
)
def dates(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    days = store.dates(run)
    zarr = da._open_store(run.path)
    target_days = (
        days.intersection(pd.DatetimeIndex(zarr["time"].values)) if zarr is not None else days[:0]
    )
    fmt = lambda idx: [d.strftime("%Y-%m-%d") for d in idx]  # noqa: E731
    return reply(
        request,
        {
            "dates": fmt(days),
            "target_dates": fmt(target_days),
            "embedding_dates": fmt(da.embedding_dates(run.path)),
            "first": days[0].strftime("%Y-%m-%d") if len(days) else None,
            "last": days[-1].strftime("%Y-%m-%d") if len(days) else None,
        },
    )


@router.get(
    "/runs/{run}/mask",
    response_model=S.MaskResponse,
    responses=ERRORS,
    summary="Ocean mask per depth (bit-packed)",
    description="Cells where the prediction product is finite (the static ocean / above-sea-floor "
    "mask). Decode with `atob` -> bytes -> unpack bits MSB first, row-major (lat, lon), per depth.",
)
def mask(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    def build():
        days = store.dates(run)
        if len(days) == 0:
            raise ApiError(404, "this run has no predictions")
        ocean = np.isfinite(store.prediction(run, days[0]))
        data = [
            base64.b64encode(np.packbits(ocean[k].ravel()).tobytes()).decode()
            for k in range(len(ocean))
        ]
        return {
            "shape": list(ocean.shape),
            "depths": run.depths,
            "encoding": "packbits-base64",
            "order": "C",
            "data": data,
            "n_ocean": [int(ocean[k].sum()) for k in range(len(ocean))],
        }

    return reply(request, store.cached(run, "mask", build))


# ----------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------
def _training_logs(store: Store, run: Run) -> dict[str, dict[str, Any]]:
    def build():
        out: dict[str, dict[str, Any]] = {}
        for stem, df in da.load_training_logs(run.path).items():
            if df.empty:
                continue
            kind = "pretrain" if stem.startswith("pretrain") else "train"
            key = "val_loss" if kind == "pretrain" else "val_rmse"
            best = None
            if key in df and df[key].notna().any():
                i = int(df[key].astype(float).idxmin())
                best = {
                    "epoch": int(df["epoch"].iloc[i]) if "epoch" in df else i + 1,
                    key: float(df[key].iloc[i]),
                }
            out[stem] = {
                "kind": kind,
                "tag": stem.split("_", 1)[1] if "_" in stem else None,
                "n_epochs": int(len(df)),
                "best": best,
                "columns": {c: df[c].tolist() for c in df.columns},
            }
        return out

    return store.cached(run, "training", build, nbytes=200_000)


def _training_summary(store: Store, run: Run) -> dict[str, Any]:
    out = {}
    for stem, log in _training_logs(store, run).items():
        cols = log["columns"]
        out[stem] = {
            "kind": log["kind"],
            "n_epochs": log["n_epochs"],
            "best": log["best"],
            "total_seconds": float(
                np.nansum(np.asarray(cols.get("epoch_seconds", [0.0]), dtype=float))
            ),
        }
    return out


@router.get(
    "/runs/{run}/training",
    response_model=S.TrainingResponse,
    responses=ERRORS,
    summary="Pretraining / training curves from the JSONL logs (including tagged ablations)",
)
def training(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    logs = _training_logs(store, run)
    if not logs:
        raise ApiError(404, "no training logs in this run (logs/*.jsonl missing)")
    return reply(request, {"run": run.name, "logs": logs})


# ----------------------------------------------------------------------------------------------
# experiments / comparison
# ----------------------------------------------------------------------------------------------
def experiments_payload(store: Store, run: Run) -> dict[str, Any]:
    glorys, argo = _glorys(store, run), _argo(store, run)
    rows, chosen = experiment_rows(glorys, argo)
    meta = (glorys or {}).get("metadata", {})
    notes = []
    if run.data_source == "synthetic":
        notes.append(SYNTHETIC_NOTE)
    notes.append("bias = method minus reference; skill = 1 - MSE / MSE of the climatology.")
    if argo:
        notes.append(str(argo.get("metadata", {}).get("independence_note", "")))
    return {
        "run": run.name,
        "data_source": run.data_source,
        "pooled_range_m": [float(x) for x in (meta.get("pooled_depth_range_m") or [50.0, 200.0])],
        "selected_depths": chosen,
        "rows": rows,
        "notes": [n for n in notes if n],
    }


@router.get(
    "/runs/{run}/experiments",
    response_model=S.ExperimentsResponse,
    responses=ERRORS,
    summary="Comparison table across all methods and ablations of a run",
)
def experiments(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    payload = experiments_payload(store, run)
    if not payload["rows"]:
        raise ApiError(404, "this run has no metrics yet (run evaluate / validate-argo)")
    return reply(request, payload)


@router.get(
    "/compare",
    response_model=S.CompareResponse,
    responses=ERRORS,
    summary="Headline experiment tables of several runs side by side",
)
def compare(
    request: Request,
    runs: str = Query(..., description="Comma-separated run names, at most 8."),
    store: Store = Depends(get_store),
):
    names = list(dict.fromkeys(n.strip() for n in runs.split(",") if n.strip()))
    if not names:
        raise ApiError(400, "give at least one run name in 'runs'")
    if len(names) > 8:
        raise ApiError(400, "compare at most 8 runs at a time")
    out = []
    for n in names:
        r = store.run(n)
        g, a = _glorys(store, r), _argo(store, r)
        out.append(
            {
                "name": r.name,
                "data_source": r.data_source,
                "test_period": _span((r.meta.get("split") or {}).get("test")),
                "n_days": (g or {}).get("metadata", {}).get("n_days"),
                "n_argo_profiles": (a or {}).get("metadata", {}).get("n_profiles_used"),
                "experiments": experiments_payload(store, r),
            }
        )
    return reply(request, {"runs": out})


# ----------------------------------------------------------------------------------------------
# report, figures, product
# ----------------------------------------------------------------------------------------------
@router.get(
    "/runs/{run}/report",
    response_model=S.ReportResponse,
    responses=ERRORS,
    summary="Generated report.md and the list of figure files",
    description="Figure links inside the markdown are relative (`figures/<name>.png`); fetch them "
    "from `/api/runs/{run}/figures/{name}`.",
)
def report(request: Request, run: Run = Depends(run_dep)):
    p = run.file("report.md")
    if not p.is_file():
        raise ApiError(404, "no report.md in this run (run the report step)")
    figs = [
        {"name": f.name, "url": f"/api/runs/{run.name}/figures/{f.name}", "size": f.stat().st_size}
        for f in figure_files(run)
    ]
    return reply(
        request, {"run": run.name, "markdown": p.read_text(encoding="utf-8"), "figures": figs}
    )


@router.get(
    "/runs/{run}/figures/{name}",
    responses={**ERRORS, 200: {"content": {"image/png": {}}, "description": "PNG figure"}},
    summary="One report figure (PNG)",
)
def figure(request: Request, name: str, run: Run = Depends(run_dep)):
    files = {f.name: f for f in figure_files(run)}
    if not FILE_NAME_RE.match(name) or name not in files:
        raise ApiError(404, f"figure '{name}' not found in run '{run.name}'")
    return FileResponse(files[name], media_type="image/png", headers=file_headers(request))


def _product_entries(run: Run, method: str) -> list[dict[str, Any]]:
    file_dates = da.prediction_file_dates(run.path, method)
    suffix = "" if method == MAIN_METHOD else f"?method={method}"
    out = []
    for ym, f in da._prediction_files(run.path, method).items():
        days = file_dates[ym]
        out.append(
            {
                "name": f.name,
                "size": f.stat().st_size,
                "month": f"{ym[:4]}-{ym[4:]}",
                "start": days.min().strftime("%Y-%m-%d") if len(days) else None,
                "end": days.max().strftime("%Y-%m-%d") if len(days) else None,
                "n_days": int(len(days)),
                "url": f"/api/runs/{run.name}/product/{f.name}{suffix}",
            }
        )
    return out


@router.get(
    "/runs/{run}/product",
    response_model=S.ProductResponse,
    responses=ERRORS,
    summary="Prediction NetCDF files of the run (the data product), plus ridge / ablation products",
    description="`files` is the main model's product. `extra_products` lists the ridge baseline and "
    "tagged ablations (each with `method`, `label` and its own files); their download URLs carry "
    "`?method=<key>` because the monthly file names repeat.",
)
def product(request: Request, run: Run = Depends(run_dep)):
    files = _product_entries(run, MAIN_METHOD)
    if not files:
        raise ApiError(404, "this run has no prediction files")
    extra = []
    for key in da.prediction_methods(run.path):
        if key == MAIN_METHOD:
            continue
        entries = _product_entries(run, key)
        extra.append(
            {
                "method": key,
                "label": method_label(key),
                "kind": method_kind(key),
                "files": entries,
                "total_size": sum(f["size"] for f in entries),
            }
        )
    return reply(
        request,
        {
            "run": run.name,
            "files": files,
            "total_size": sum(f["size"] for f in files),
            "extra_products": extra,
        },
    )


@router.get(
    "/runs/{run}/product/{file}",
    responses={
        **ERRORS,
        200: {"content": {"application/x-netcdf": {}}, "description": "NetCDF file"},
    },
    summary="Download one prediction NetCDF file",
)
def product_file(
    request: Request,
    file: str,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    method: str = Query(
        MAIN_METHOD, description="model (default), ridge or an ablation model_<tag>"
    ),
):
    method = store.method(run, method)
    files = product_files(run, method)
    if not FILE_NAME_RE.match(file) or file not in files:
        raise ApiError(404, f"product file '{file}' not found in run '{run.name}'")
    return FileResponse(
        files[file], media_type="application/x-netcdf", filename=file, headers=file_headers(request)
    )
