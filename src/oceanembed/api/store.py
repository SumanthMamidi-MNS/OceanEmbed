"""Run registry, input validation and cached loaders behind the HTTP API.

Everything here is read-only. Files are only ever read through :mod:`oceanembed.data_access` (the
same layer the Streamlit dashboard uses); this module adds

* run-name validation (a run is a direct sub-folder of the outputs root that holds a ``run_meta.json``),
* date / depth / point validation with clean ``ApiError`` messages,
* a byte-bounded in-process LRU so scrubbing through days and depths does not touch the disk twice,
* the few derived products the API needs (volumes of every ``kind``, colour-range hints, point time
  series, argo profile tables).
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd

from oceanembed import data_access as da
from oceanembed.config import Config
from oceanembed.eval.evaluate import method_label
from oceanembed.grid import Grid, build_grid
from oceanembed.runmeta import RUN_META_FILE

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,150}$")

FIELD_KINDS = (
    "prediction",
    "target",
    "climatology",
    "difference",
    "anomaly_pred",
    "anomaly_target",
)
DIVERGING_KINDS = ("difference", "anomaly_pred", "anomaly_target")
KIND_LABELS = {
    "prediction": "OceanEmbed prediction",
    "target": "GLORYS reanalysis (target)",
    "climatology": "Harmonic climatology",
    "difference": "Prediction minus GLORYS",
    "anomaly_pred": "Predicted anomaly (prediction minus climatology)",
    "anomaly_target": "Target anomaly (target minus climatology)",
}
MAIN_METHOD = "model"
RANGE_SAMPLE_DAYS = 24
METHOD_KINDS = ("prediction", "difference", "anomaly_pred")


def kind_label(kind: str, method: str = MAIN_METHOD) -> str:
    """Label of a field kind. The kinds that depend on the prediction method all follow one pattern,
    ``<method label> prediction`` / ``<method label> minus GLORYS`` / ``<method label> anomaly
    (prediction minus climatology)``, for the model and every other method alike."""
    if kind not in METHOD_KINDS:
        return KIND_LABELS[kind]
    name = method_label(method)
    return {
        "prediction": f"{name} prediction",
        "difference": f"{name} minus GLORYS",
        "anomaly_pred": f"{name} anomaly (prediction minus climatology)",
    }[kind]


def method_kind(key: str) -> str:
    """``model`` | ``baseline`` (ridge) | ``ablation`` (a tagged ``model_<tag>`` product)."""
    if key == MAIN_METHOD:
        return "model"
    return "baseline" if key == "ridge" else "ablation"


class ApiError(Exception):
    """An error that is turned into ``{"detail": ...}`` with ``status``."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ByteLRU:
    """Thread-safe LRU bounded by an approximate byte budget."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._d: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                return None
            self._d.move_to_end(key)
            return hit[0]

    def put(self, key, value, nbytes: int) -> None:
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._d[key] = (value, nbytes)
            self._bytes += nbytes
            while self._bytes > self.max_bytes and len(self._d) > 1:
                _, (_, n) = self._d.popitem(last=False)
                self._bytes -= n

    def get_or_set(self, key, factory: Callable[[], Any], nbytes: Callable[[Any], int]):
        hit = self.get(key)
        if hit is not None:
            return hit
        value = factory()
        self.put(key, value, nbytes(value))
        return value

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


_MISSING = object()


@dataclass(frozen=True)
class Run:
    """A validated run: its folder, ``run_meta.json`` content and cache stamp."""

    name: str
    path: Path
    stamp: int  # run_meta.json mtime (ns); every pipeline command rewrites that file
    meta: dict

    @property
    def key(self) -> tuple[str, int]:
        return (str(self.path), self.stamp)

    @property
    def data_source(self) -> str:
        return str(self.meta.get("data_source", "unknown"))

    @cached_property
    def config(self) -> Config:
        return Config.model_validate(self.meta["config"])

    @cached_property
    def grid(self) -> Grid:
        return build_grid(self.config)

    @property
    def depths(self) -> list[float]:
        return [float(d) for d in self.grid.depth]

    def file(self, *parts: str) -> Path:
        return self.path.joinpath(*parts)


class Store:
    """Registry of runs under ``outputs_root`` plus shared caches."""

    def __init__(self, outputs_root: Path, cache_mb: int = 768):
        self.root = Path(outputs_root)
        self.cache = ByteLRU(cache_mb * 1024 * 1024)
        self._runs: dict[tuple[str, int], Run] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ runs
    def run_names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and RUN_NAME_RE.match(p.name) and (p / RUN_META_FILE).is_file()
        )

    def run(self, name: str) -> Run:
        """The run called ``name`` (404 unless it is a real run folder directly under the root)."""
        if not isinstance(name, str) or not RUN_NAME_RE.match(name):
            raise ApiError(404, f"run '{name}' not found")
        path = self.root / name
        meta_file = path / RUN_META_FILE
        try:
            if path.resolve().parent != self.root.resolve() or not meta_file.is_file():
                raise ApiError(404, f"run '{name}' not found")
            stamp = meta_file.stat().st_mtime_ns
        except OSError as e:
            raise ApiError(404, f"run '{name}' not found") from e
        key = (str(path), stamp)
        with self._lock:
            run = self._runs.get(key)
        if run is None:
            try:
                meta = da.load_run_meta(path)
            except (OSError, ValueError) as e:
                raise ApiError(404, f"run '{name}' has no readable {RUN_META_FILE}") from e
            run = Run(name=name, path=path, stamp=stamp, meta=meta)
            with self._lock:
                self._runs = {k: v for k, v in self._runs.items() if k[0] != key[0]}
                self._runs[key] = run
        return run

    def cached(self, run: Run, tag: str, factory: Callable[[], Any], nbytes: int = 50_000):
        """Memoise a small derived object per run version (``run_meta.json`` mtime)."""
        return self.cache.get_or_set((*run.key, "obj", tag), factory, lambda _: nbytes)

    # ------------------------------------------------------------------ validation
    def dates(self, run: Run) -> pd.DatetimeIndex:
        return da.available_dates(run.path)

    def parse_date(self, run: Run, text: str | None, *, what: str = "date") -> pd.Timestamp:
        """Strict ``YYYY-MM-DD``; 400 if malformed or outside the predicted range, 404 for a gap."""
        if text is None or not DATE_RE.match(text):
            raise ApiError(400, f"'{what}' must be a date formatted YYYY-MM-DD, got {text!r}")
        try:
            d = pd.Timestamp(text)
        except ValueError as e:
            raise ApiError(400, f"'{what}' is not a valid calendar date: {text!r}") from e
        days = self.dates(run)
        if len(days) == 0:
            raise ApiError(404, "this run has no predictions")
        if d < days[0] or d > days[-1]:
            raise ApiError(
                400,
                f"{what} {text} is outside the available range {days[0].date()} .. {days[-1].date()}",
            )
        if d not in days:
            raise ApiError(404, f"no prediction for {text} (a gap inside the available range)")
        return d

    def depth_index(self, run: Run, depth: float | None, depth_index: int | None) -> int:
        """Index of the requested level: ``depth`` (metres, must equal a grid depth) or
        ``depth_index`` (0-based). Exactly one of them."""
        depths = np.asarray(run.grid.depth)
        if (depth is None) == (depth_index is None):
            raise ApiError(400, "give exactly one of 'depth' (metres) or 'depth_index'")
        if depth_index is not None:
            if not 0 <= depth_index < len(depths):
                raise ApiError(400, f"depth_index must be 0..{len(depths) - 1}, got {depth_index}")
            return int(depth_index)
        assert depth is not None
        hit = np.flatnonzero(np.isclose(depths, depth, atol=1e-6))
        if hit.size == 0:
            levels = ", ".join(f"{d:g}" for d in depths)
            raise ApiError(400, f"depth {depth:g} m is not a grid depth; choose one of {levels}")
        return int(hit[0])

    def cell(self, run: Run, lat: float, lon: float) -> tuple[int, int]:
        """Nearest cell of a point (400 if the point lies outside the grid)."""
        g = run.grid
        la0, la1 = g.lat_edges
        lo0, lo1 = g.lon_edges
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ApiError(400, "lat and lon must be finite numbers")
        if not la0 <= lat <= la1:
            raise ApiError(400, f"lat {lat:g} is outside the grid ({la0:g} .. {la1:g})")
        if not lo0 <= lon <= lo1:
            raise ApiError(400, f"lon {lon:g} is outside the grid ({lo0:g} .. {lo1:g})")
        return int(np.abs(g.lat - lat).argmin()), int(np.abs(g.lon - lon).argmin())

    def method(self, run: Run, key: str | None) -> str:
        """Validated prediction method (``model`` | ``ridge`` | ``model_<tag>``); 404 when that
        method has no day-field product in this run. ``model`` is never checked here (a run
        without predictions fails later with its own message)."""
        key = key or MAIN_METHOD
        if key == MAIN_METHOD:
            return key
        avail = da.prediction_methods(run.path)
        if key not in avail:
            have = ", ".join(avail) or "none"
            raise ApiError(404, f"no day fields for method '{key}' in this run; available: {have}")
        return key

    def field_methods(self, run: Run) -> list[dict[str, Any]]:
        """Methods with a prediction product: key, label, kind, tag and the days they cover."""

        def build():
            out = []
            for key in da.prediction_methods(run.path):
                days = da.available_dates(run.path, key)
                out.append(
                    {
                        "key": key,
                        "label": method_label(key),
                        "kind": method_kind(key),
                        "tag": key.removeprefix("model_") if key.startswith("model_") else None,
                        "n_days": int(len(days)),
                        "first": days[0].strftime("%Y-%m-%d") if len(days) else None,
                        "last": days[-1].strftime("%Y-%m-%d") if len(days) else None,
                    }
                )
            return out

        return self.cached(run, "field_methods", build)

    # ------------------------------------------------------------------ cached volumes
    def prediction(self, run: Run, date: pd.Timestamp, method: str = MAIN_METHOD) -> np.ndarray:
        """Predicted temperature ``(depth, lat, lon)`` float32 (NaN off-mask), cached."""

        def load():
            try:
                return da.load_prediction(run.path, date, method).values.astype(np.float32)
            except KeyError as e:
                raise ApiError(404, f"method '{method}' has no prediction for {date.date()}") from e
            except (OSError, RuntimeError) as e:  # a product file that is still being written
                raise ApiError(
                    404, f"the '{method}' prediction for {date.date()} cannot be read yet"
                ) from e

        key = (*run.key, "pred", date.value) + (() if method == MAIN_METHOD else (method,))
        return self.cache.get_or_set(key, load, lambda a: a.nbytes)

    def target(self, run: Run, date: pd.Timestamp) -> np.ndarray | None:
        """Harmonised GLORYS ``(depth, lat, lon)`` float32, or ``None`` when unavailable."""
        got = self.cache.get_or_set(
            (*run.key, "tgt", date.value),
            lambda: _or_missing(da.load_target(run.path, date)),
            lambda a: 0 if a is _MISSING else a.nbytes,
        )
        return None if got is _MISSING else got

    def climatology(self, run: Run, date: pd.Timestamp) -> np.ndarray | None:
        def load():
            try:
                return da.load_climatology(run.path, date).values.astype(np.float32)
            except (OSError, KeyError):
                return _MISSING

        got = self.cache.get_or_set(
            (*run.key, "clim", date.value), load, lambda a: 0 if a is _MISSING else a.nbytes
        )
        return None if got is _MISSING else got

    def surface(self, run: Run, date: pd.Timestamp) -> dict[str, np.ndarray] | None:
        def load():
            ds = da.load_surface_inputs(run.path, date)
            if ds is None:
                return _MISSING
            return {k: ds[k].values.astype(np.float32) for k in ds.data_vars}

        got = self.cache.get_or_set(
            (*run.key, "surface", date.value),
            load,
            lambda a: 0 if a is _MISSING else sum(v.nbytes for v in a.values()),
        )
        return None if got is _MISSING else got

    def volume(
        self, run: Run, kind: str, date: pd.Timestamp, method: str = MAIN_METHOD
    ) -> np.ndarray:
        """The 15-level volume of ``kind`` for one day (404 when a needed input is missing).
        ``method`` picks the prediction used by the prediction-based kinds."""
        pred = self.prediction(run, date, method)
        if kind == "prediction":
            return pred
        if kind == "climatology":
            clim = self.climatology(run, date)
            if clim is None:
                raise ApiError(404, "climatology is unavailable (statistics file missing)")
            return clim
        if kind == "target":
            return self._need_target(run, date)
        if kind == "difference":
            return pred - self._need_target(run, date)
        clim = self.climatology(run, date)
        if clim is None:
            raise ApiError(404, "climatology is unavailable (statistics file missing)")
        if kind == "anomaly_pred":
            return pred - clim
        if kind == "anomaly_target":
            return self._need_target(run, date) - clim
        raise ApiError(400, f"kind must be one of {', '.join(FIELD_KINDS)}")

    def _need_target(self, run: Run, date: pd.Timestamp) -> np.ndarray:
        tgt = self.target(run, date)
        if tgt is None:
            raise ApiError(
                404, f"no GLORYS target for {date.date()} in this run (prediction-only day)"
            )
        return tgt

    # ------------------------------------------------------------------ colour hints
    def colour_hint(
        self, run: Run, kind: str, date: pd.Timestamp, level: int | None, method: str = MAIN_METHOD
    ) -> dict[str, Any]:
        """Robust colour limits so the UI never scans the data.

        Temperature-like kinds (prediction / target / climatology) share one range computed from
        the main model's prediction + target (1st-99th percentile) whatever ``method`` is asked
        for, so model / ridge / ablation panels are directly comparable. Diverging kinds are
        symmetric about zero with the 99th percentile of the absolute value of the requested
        method's own error; the two anomaly kinds share their limit.
        """

        def compute():
            sel = (lambda a: a) if level is None else (lambda a: a[level])
            pred = self.prediction(run, date, method if kind in DIVERGING_KINDS else MAIN_METHOD)
            tgt = self.target(run, date)
            clim = self.climatology(run, date)
            if kind in DIVERGING_KINDS:
                parts = []
                if kind == "difference" and tgt is not None:
                    parts.append(pred - tgt)
                elif kind.startswith("anomaly") and clim is not None:
                    parts.append(pred - clim)
                    if tgt is not None:
                        parts.append(tgt - clim)
                limit = _sym_limit([sel(p) for p in parts])
                return {"vmin": -limit, "vmax": limit, "diverging": True}
            arrays = [sel(pred)] + ([sel(tgt)] if tgt is not None else [])
            lo, hi = robust_range(arrays)
            return {"vmin": lo, "vmax": hi, "diverging": False}

        mkey = method if kind in DIVERGING_KINDS else MAIN_METHOD
        return self.cache.get_or_set(
            (*run.key, "hint", kind, date.value, level, mkey), compute, lambda _: 256
        )

    # ------------------------------------------------------------------ period-wide colour ranges
    def period_ranges(self, run: Run, n_days: int = RANGE_SAMPLE_DAYS) -> dict[str, Any]:
        """Per-depth colour ranges over the whole prediction period, from a sample of days.

        ``n_days`` days evenly spaced over the predicted period (first and last included) are read
        once; for every depth the finite values of all sampled days are pooled and

        * ``temperature``: 1st-99th percentile of the main model's prediction united with the
          GLORYS target (the target only on days that have one) - the same rule as the per-day
          hint, so a held range and a per-day range are directly comparable;
        * ``methods[key].difference``: symmetric, ``99th percentile of |prediction - target|``;
        * ``methods[key].anomaly``: symmetric, 99th percentile of ``|prediction - climatology|``
          united with ``|target - climatology|`` (shared by both anomaly kinds, as per day).

        Computed in memory once per run version (``run_meta.json`` mtime) and cached; nothing is
        written to the run folder. 404 when the run has no predictions yet.
        """
        days = self.dates(run)
        if len(days) == 0:
            raise ApiError(404, "this run has no predictions yet")
        n = int(min(max(n_days, 1), len(days)))
        return self.cache.get_or_set(
            (*run.key, "ranges", n), lambda: self._compute_ranges(run, days, n), lambda _: 20_000
        )

    def _compute_ranges(self, run: Run, days: pd.DatetimeIndex, n: int) -> dict[str, Any]:
        idx = np.unique(np.linspace(0, len(days) - 1, n).round().astype(int))
        sample = days[idx]
        methods = list(da.prediction_methods(run.path))
        n_depth = len(run.grid.depth)
        temp: list[list[np.ndarray]] = [[] for _ in range(n_depth)]
        diff = {m: [[] for _ in range(n_depth)] for m in methods}
        anom = {m: [[] for _ in range(n_depth)] for m in methods}
        have_target = have_clim = 0
        used = []
        for d in sample:
            preds = {}
            for m in methods:
                try:
                    preds[m] = (
                        self.prediction(run, d, m)
                        if m == MAIN_METHOD
                        else da.load_prediction(run.path, d, m).values.astype(np.float32)
                    )
                except (KeyError, ApiError):
                    continue  # a method that does not cover this day
            if MAIN_METHOD not in preds:
                continue
            used.append(d)
            tgt = self.target(run, d)
            clim = self.climatology(run, d)
            have_target += tgt is not None
            have_clim += clim is not None
            for k in range(n_depth):
                temp[k].append(_thin(preds[MAIN_METHOD][k]))
                if tgt is not None:
                    temp[k].append(_thin(tgt[k]))
                for m, p in preds.items():
                    if tgt is not None:
                        diff[m][k].append(_thin(p[k] - tgt[k]))
                    if clim is not None:
                        anom[m][k].append(_thin(p[k] - clim[k]))
                        if tgt is not None:
                            anom[m][k].append(_thin(tgt[k] - clim[k]))

        def span(arrs: list[np.ndarray]) -> dict[str, Any]:
            lo, hi = robust_range(arrs)
            return {"vmin": lo, "vmax": hi, "diverging": False}

        def sym(arrs: list[np.ndarray]) -> dict[str, Any] | None:
            if not arrs:
                return None
            lim = _sym_limit(arrs)
            return {"vmin": -lim, "vmax": lim, "diverging": True}

        out_methods = {}
        for m in methods:
            out_methods[m] = {
                "label": method_label(m),
                "kind": method_kind(m),
                "difference": [sym(diff[m][k]) for k in range(n_depth)] if have_target else None,
                "anomaly": [sym(anom[m][k]) for k in range(n_depth)] if have_clim else None,
            }
        flat = [a for k in range(n_depth) for a in temp[k]]
        return {
            "run": run.name,
            "units": "degC",
            "depths": run.depths,
            "n_days_available": int(len(days)),
            "n_days_sampled": len(used),
            "sampled_dates": [d.strftime("%Y-%m-%d") for d in used],
            "n_days_with_target": int(have_target),
            "temperature": [span(temp[k]) for k in range(n_depth)],
            "temperature_volume": span(flat),
            "methods": out_methods,
            "method_note": (
                f"{len(used)} of {len(days)} predicted days, evenly spaced over the period (first and "
                "last included), pooled per depth (at most 6000 evenly strided ocean values per day "
                "and depth). temperature: 1st-99th percentile of the main "
                "model's prediction united with the GLORYS target. difference: symmetric about "
                "zero, 99th percentile of |prediction - target| of that method. anomaly: symmetric, "
                "99th percentile of |prediction - climatology| united with |target - climatology| "
                "(the limit both anomaly kinds share). Computed in memory once per run version."
            ),
        }

    # ------------------------------------------------------------------ point series
    def point_series(self, run: Run, i: int, j: int, method: str = MAIN_METHOD) -> dict[str, Any]:
        """Prediction and target at one cell for every predicted day: arrays ``(time, depth)``.

        ``method`` picks the prediction product (days it does not cover stay NaN); the target is
        shared between methods."""

        def compute():
            days = self.dates(run)
            pred = np.full((len(days), len(run.grid.depth)), np.nan, dtype=np.float32)
            by_month = da.prediction_file_dates(run.path, method)
            for ym, f in da._prediction_files(run.path, method).items():
                # netCDF4 directly: reading one column through xarray's lazy indexing is ~6x slower
                with da.NETCDF4_LOCK, netCDF4.Dataset(f) as nc:
                    block = np.ma.filled(nc.variables["temperature"][:, :, i, j], np.nan)
                pos = days.get_indexer(by_month[ym])
                keep = pos >= 0
                pred[pos[keep]] = block[keep]
            return {"dates": days, "prediction": pred}

        got = self.cache.get_or_set(
            (*run.key, "series", i, j) + (() if method == MAIN_METHOD else (method,)),
            compute,
            lambda d: d["prediction"].nbytes,
        )
        return {**got, "target": self._point_target(run, i, j, got["dates"])}

    def _point_target(self, run: Run, i: int, j: int, days: pd.DatetimeIndex) -> np.ndarray:
        def compute():
            tgt = np.full((len(days), len(run.grid.depth)), np.nan, dtype=np.float32)
            store = da._open_store(run.path)
            if store is not None:
                pos = pd.DatetimeIndex(store["time"].values).get_indexer(days)
                have = np.flatnonzero(pos >= 0)
                if have.size:
                    tgt[have] = store["temp"].isel(lat=i, lon=j, time=pos[have]).values
            return tgt

        return self.cache.get_or_set((*run.key, "series_tgt", i, j), compute, lambda a: a.nbytes)

    def point_climatology(
        self, run: Run, i: int, j: int, days: pd.DatetimeIndex
    ) -> np.ndarray | None:
        """Harmonic climatology ``(time, depth)`` at one cell (NaN off the ocean mask of the first
        predicted day); ``None`` when the statistics file is missing."""

        def compute():
            try:
                clim = da.load_climatology_point(run.path, days, i, j)
            except (OSError, KeyError):
                return _MISSING
            if len(days):
                clim = np.where(np.isfinite(self.prediction(run, days[0])[:, i, j]), clim, np.nan)
            return clim.astype(np.float32)

        got = self.cache.get_or_set(
            (*run.key, "series_clim", i, j), compute, lambda a: 0 if a is _MISSING else a.nbytes
        )
        return None if got is _MISSING else got

    # ------------------------------------------------------------------ tables from files
    def argo_matchups(self, run: Run) -> pd.DataFrame | None:
        p = run.file("metrics", "argo_matchups.parquet")
        if not p.is_file():
            return None
        key = (*run.key, "matchups", p.stat().st_mtime_ns)
        return self.cache.get_or_set(
            key,
            lambda: da.load_argo_matchups(run.path),
            lambda df: int(df.memory_usage(deep=True).sum()),
        )

    def map_arrays(self, run: Run) -> dict[str, np.ndarray] | None:
        """Every variable of ``maps_glorys.nc`` as a float32 array (cached; ~20 MB full grid)."""
        p = run.file("metrics", "maps_glorys.nc")
        if not p.is_file():
            return None
        key = (*run.key, "maps", p.stat().st_mtime_ns)

        def load():
            ds = da.load_maps(run.path)
            return {k: ds[k].values.astype(np.float32) for k in ds.data_vars}

        return self.cache.get_or_set(key, load, lambda d: sum(v.nbytes for v in d.values()))


def _thin(a: np.ndarray, cap: int = 6000) -> np.ndarray:
    """The finite values of a slice, at most ``cap`` of them at an even stride (deterministic):
    enough for stable 1st / 99th percentiles while keeping a period-wide pool small."""
    v = a[np.isfinite(a)]
    if v.size > cap:
        v = v[np.linspace(0, v.size - 1, cap).astype(np.int64)]
    return v


def _or_missing(arr):
    return _MISSING if arr is None else arr.values.astype(np.float32)


def robust_range(
    arrays: list[np.ndarray], lo: float = 1.0, hi: float = 99.0
) -> tuple[float, float]:
    """Percentile range over the finite values of several arrays (``(0, 1)`` when all are NaN)."""
    v = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays]) if arrays else []
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    if b - a < 1e-9:
        b = a + 1e-3
    return round(float(a), 4), round(float(b), 4)


def _sym_limit(arrays: list[np.ndarray], pct: float = 99.0) -> float:
    v = (
        np.concatenate([np.abs(np.asarray(a, dtype=np.float32)).ravel() for a in arrays])
        if arrays
        else []
    )
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return max(round(float(np.percentile(v, pct)), 4), 1e-3)
