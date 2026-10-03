"""Research stage R5 (physical metrics), streaming pass: per-day sufficient statistics.

Reads, for the test split, the daily GLORYS target (harmonised store), the daily prediction files
of the main model (``predictions/``), the no-pretraining model (``predictions/scratch/``) and
ridge (``predictions/ridge/``), the harmonic climatology (stats file) and, when its R1 checkpoint
exists, the per-pixel MLP of seed 0 (re-run on the surface inputs on the CPU; no training). One
day at a time (the MLP in batches of 8 days) is held in memory, never the year.

Per day it accumulates, as the eight additive fields of :data:`oceanembed.eval.metrics.SUM_FIELDS`
(prediction vs GLORYS, raw and with the climatology removed from both):

* **derived quantities** (:mod:`oceanembed.research.physical`: D20, D23, heat content 0-300 m,
  mean temperature 0-300 m and 0-30 m) for the whole domain and each basin, on the cells where
  *every* product and GLORYS have a defined value (one common sample for all products);
* **pooled 50-200 m temperature** for strata: basin, |SLA| terciles (per cell-day, domain-wide
  edges, also split by basin), and Bay of Bengal surface-salinity terciles (per cell-day, edges
  from the Bay of Bengal cell-days);
* cell maps (mean derived quantity per product, squared error) for the figures.

Everything is written to ``outputs/<run>/research/r5/sums.npz`` and turned into tables, intervals
and figures by :mod:`oceanembed.research.r5_report`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_surface_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.data_access import load_prediction
from oceanembed.eval.evaluate import read_target, split_bounds
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.grid import build_grid
from oceanembed.infer.predict import model_predictor, predict_batch
from oceanembed.models.pixel_mlp import load_pixel_mlp
from oceanembed.research.physical import QUANTITIES, derived_fields, tercile_class, tercile_edges
from oceanembed.train.utils import get_device

POOLED_RANGE = (50.0, 200.0)
REGIONS = ("all", "arabian_sea", "bay_of_bengal")
TERCILES = ("low", "mid", "high")
# product key -> key of the prediction folder in data_access (None = computed here)
PRODUCT_SOURCES = {
    "climatology": None,
    "ridge": "ridge",
    "mlp": None,
    "scratch": "model_scratch",
    "model": "model",
}
PRODUCT_LABELS = {
    "climatology": "Climatology",
    "ridge": "Ridge regression",
    "mlp": "Per-pixel MLP (R1, seed 0)",
    "scratch": "OceanEmbed (no pretraining)",
    "model": "OceanEmbed (pretrained)",
}
SUMS_FILE = "sums.npz"


def r5_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "r5"


def class_names() -> list[str]:
    names = [f"region:{r}" for r in REGIONS]
    names += [f"sla:{r}:{t}" for r in REGIONS for t in TERCILES]
    names += [f"sss:bay_of_bengal:{t}" for t in TERCILES]
    return names


def _say(msg: str) -> None:
    print(msg, flush=True)


def mlp_checkpoint(cfg: Config, seed: int = 0) -> Path:
    return cfg.outputs_dir / "research" / "r1" / "mlp" / f"seed{seed}" / "mlp.pt"


def strata_edges(sla: np.ndarray, sss: np.ndarray, ocean: np.ndarray, bob: np.ndarray) -> dict:
    """|SLA| terciles over all ocean cell-days (domain-wide) and surface-salinity terciles over the
    Bay of Bengal cell-days; ``sla`` / ``sss`` are ``(T, H, W)``, ``ocean`` / ``bob`` ``(H, W)``."""
    return {
        "sla_abs": tercile_edges(np.abs(sla)[:, ocean]),
        "sss_bob": tercile_edges(sss[:, ocean & bob]),
    }


def day_class_masks(
    sla_day: np.ndarray, sss_day: np.ndarray, region_masks: list[np.ndarray], edges: dict
) -> list[np.ndarray]:
    """Boolean ``(H, W)`` mask per entry of :func:`class_names` for one day."""
    sla_cls = tercile_class(np.abs(sla_day), edges["sla_abs"])
    sss_cls = tercile_class(sss_day, edges["sss_bob"])
    masks = [m for m in region_masks]
    for m in region_masks:
        masks += [m & (sla_cls == t) for t in range(3)]
    bob = region_masks[REGIONS.index("bay_of_bengal")]
    masks += [bob & (sss_cls == t) for t in range(3)]
    return masks


def _field_sums(x, y, valid, axis=None) -> np.ndarray:
    """``(F,)`` or ``(F, ...)`` stack of the eight sums."""
    part = sums_from_arrays(x, y, valid, axis=axis)
    return np.stack([part[f] for f in SUM_FIELDS])


def accumulate_day(
    fields: dict[str, np.ndarray],
    products: list[str],
    depths: np.ndarray,
    valid_t: np.ndarray,
    class_masks: list[np.ndarray],
    region_masks: list[np.ndarray],
    temp_sums: np.ndarray,
    dq_sums: np.ndarray,
    maps: dict,
    t: int,
) -> None:
    """Add one day. ``fields`` maps ``glorys`` and every product to ``(D, H, W)`` temperature (NaN
    outside the static mask); ``valid_t`` is the ``(D, H, W)`` sample mask of the temperature
    scores. ``temp_sums`` is ``(C, P, 2, F, T)`` and ``dq_sums`` ``(Q, R, P, 2, F, T)``."""
    ref, clim = fields["glorys"], fields["climatology"]
    pooled = (depths >= POOLED_RANGE[0]) & (depths <= POOLED_RANGE[1])
    for p, name in enumerate(products):
        x = fields[name]
        for k, (xx, yy) in enumerate(((x, ref), (x - clim, ref - clim))):
            part = _field_sums(xx[pooled], yy[pooled], valid_t[pooled], axis=0)  # (F, H, W)
            for c, cm in enumerate(class_masks):
                temp_sums[c, p, k, :, t] += part[:, cm].sum(axis=1)

    dq = {name: derived_fields(f, depths) for name, f in fields.items()}
    for qi, q in enumerate(QUANTITIES):
        common = np.isfinite(dq["glorys"][q])
        for name in products:
            common &= np.isfinite(dq[name][q])
        g, c0 = dq["glorys"][q], dq["climatology"][q]
        maps["n"][qi] += common
        maps["glorys"][qi] += np.where(common, g, 0.0)
        for p, name in enumerate(products):
            x = dq[name][q]
            maps["sum"][qi, p] += np.where(common, x, 0.0)
            maps["se2"][qi, p] += np.where(common, (x - g) ** 2, 0.0)
            maps["finite"][qi, p] += np.isfinite(x)
            for r, rm in enumerate(region_masks):
                v = common & rm
                dq_sums[qi, r, p, 0, :, t] += _field_sums(x, g, v)
                dq_sums[qi, r, p, 1, :, t] += _field_sums(x - c0, g - c0, v)
        maps["finite_glorys"][qi] += np.isfinite(g)


def run_r5(
    cfg: Config,
    mlp_seed: int = 0,
    progress: bool = True,
    out_dir: Path | None = None,
) -> Path:
    """Stream the test split and write ``sums.npz``; returns its path."""
    out = Path(out_dir) if out_dir else r5_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device("cpu")  # the MLP is tiny; the stage is documented as CPU only
    start, end = split_bounds(cfg, "test")
    ds = make_surface_dataset(cfg, start, end)
    zds = open_harmonized(cfg)
    grid = build_grid(cfg)
    depths = np.asarray(ds.depth, dtype=np.float64)
    dates = ds.dates()
    n_days = len(ds)
    run_dir = cfg.outputs_dir

    products = []
    mlp_pred = None
    for name in PRODUCT_SOURCES:
        src = PRODUCT_SOURCES[name]
        if name == "mlp":
            ck = mlp_checkpoint(cfg, mlp_seed)
            if ck.exists():
                model, _ = load_pixel_mlp(ck, dev)
                mlp_pred = model_predictor(model, amp=False)
                products.append(name)
            continue
        if src is not None:
            try:
                load_prediction(run_dir, dates[0], src)
            except KeyError as e:
                raise FileNotFoundError(
                    f"no daily prediction product '{src}' under {run_dir / 'predictions'}: {e}"
                ) from e
        products.append(name)

    basins = grid.basin_masks()
    region_masks = [np.ones(grid.shape, dtype=bool), basins["arabian_sea"], basins["bay_of_bengal"]]
    ocean = np.asarray(ds.mask[0], dtype=bool)
    lo, hi = int(ds.indices[0]), int(ds.indices[-1]) + 1
    sla = zds["sla"].isel(time=slice(lo, hi)).values.astype(np.float32)
    sss = zds["sss"].isel(time=slice(lo, hi)).values.astype(np.float32)
    edges = strata_edges(sla, sss, ocean, basins["bay_of_bengal"])
    _say(
        f"|SLA| tercile edges {edges['sla_abs']}; "
        f"Bay of Bengal SSS tercile edges {edges['sss_bob']}"
    )

    names = class_names()
    n_f, n_q, n_p, n_r = len(SUM_FIELDS), len(QUANTITIES), len(products), len(REGIONS)
    temp_sums = np.zeros((len(names), n_p, 2, n_f, n_days))
    dq_sums = np.zeros((n_q, n_r, n_p, 2, n_f, n_days))
    h, w = grid.shape
    maps = {
        "n": np.zeros((n_q, h, w)),
        "glorys": np.zeros((n_q, h, w)),
        "sum": np.zeros((n_q, n_p, h, w)),
        "se2": np.zeros((n_q, n_p, h, w)),
        "finite": np.zeros((n_q, n_p, h, w)),
        "finite_glorys": np.zeros((n_q, h, w)),
    }
    bar = progress and sys.stderr.isatty()
    t0 = time.time()
    batch = 8
    for a in tqdm(range(0, n_days, batch), desc="r5", disable=not bar):
        b = min(a + batch, n_days)
        ref = read_target(zds, ds, a, b)
        clim = ds.stats.climatology(dates[a:b].values)
        mlp = predict_batch(mlp_pred, ds.batch(a, b), ds, dev) if mlp_pred is not None else None
        for i in range(b - a):
            day = a + i
            fields = {"glorys": ref[i], "climatology": clim[i]}
            for name in products:
                src = PRODUCT_SOURCES[name]
                if name == "mlp":
                    fields[name] = mlp[i]
                elif src is not None:
                    fields[name] = load_prediction(run_dir, dates[day], src).values
            fields = {k: np.where(ds.mask, v, np.nan).astype(np.float32) for k, v in fields.items()}
            valid_t = ds.mask & np.isfinite(fields["glorys"])
            masks = day_class_masks(sla[day], sss[day], region_masks, edges)
            accumulate_day(
                fields, products, depths, valid_t, masks, region_masks,
                temp_sums, dq_sums, maps, day,
            )  # fmt: skip
    _say(f"streamed {n_days} days in {time.time() - t0:.0f}s")

    tmp = out / "sums.tmp.npz"
    np.savez_compressed(
        tmp,
        temp_sums=temp_sums,
        dq_sums=dq_sums,
        map_n=maps["n"],
        map_glorys=maps["glorys"],
        map_sum=maps["sum"],
        map_se2=maps["se2"],
        map_finite=maps["finite"],
        map_finite_glorys=maps["finite_glorys"],
        dates=np.array([str(t.date()) for t in dates]),
        depths=depths,
        products=np.array(products),
        classes=np.array(names),
        quantities=np.array(list(QUANTITIES)),
        regions=np.array(REGIONS),
        fields=np.array(SUM_FIELDS),
        sla_edges=np.array(edges["sla_abs"]),
        sss_edges=np.array(edges["sss_bob"]),
        lat=grid.lat,
        lon=grid.lon,
        ocean=ocean,
    )
    os.replace(tmp, out / SUMS_FILE)
    return out / SUMS_FILE


def load_sums(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
