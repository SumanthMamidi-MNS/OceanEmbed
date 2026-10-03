"""Cached access to a run for the Streamlit pages.

Thin ``st.cache_data`` wrappers around ``app.data_access`` (which stays Streamlit-free). Every
loader is keyed by the run directory and a ``stamp`` (modification time of ``run_meta.json``, which
every pipeline command rewrites), so re-running the pipeline invalidates the cache. Per-day loaders
read exactly one day; nothing here loads a month.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from app import data_access as da

from oceanembed.config import OUTPUTS_ROOT_ENV
from oceanembed.runmeta import RUN_META_FILE

_DAY_ENTRIES = 96  # cached days per loader (a day of 15 x 100 x 240 float32 is ~1.4 MB)
_MISSING = (KeyError, FileNotFoundError, OSError, ValueError)


@dataclass(frozen=True)
class Run:
    """The selected run: where it lives, what it contains and its cache stamp."""

    name: str
    path: str
    stamp: float
    info: dict
    meta: dict

    @property
    def data_source(self) -> str:
        return str(self.meta.get("data_source") or self.info.get("data_source") or "unknown")

    @property
    def is_synthetic(self) -> bool:
        return self.data_source == "synthetic"

    @property
    def depths(self) -> list[float]:
        return [float(d) for d in self.meta.get("grid", {}).get("depths", [])]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def outputs_root() -> Path:
    """Folder holding the runs: ``OCEANEMBED_OUTPUTS_ROOT`` or ``<project root>/outputs`` (the CLI's
    default when run from the project root; anchored here so the dashboard also works when
    Streamlit is started from another directory)."""
    env = os.environ.get(OUTPUTS_ROOT_ENV)
    return Path(env).resolve() if env else PROJECT_ROOT / "outputs"


def run_stamp(path: str) -> float:
    try:
        return (Path(path) / RUN_META_FILE).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(ttl=15, show_spinner=False)
def list_runs(root: str) -> list[dict]:
    try:
        return da.list_runs(root)
    except (OSError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def run_meta(path: str, stamp: float) -> dict:
    try:
        return da.load_run_meta(path)
    except _MISSING:
        return {}


@st.cache_data(show_spinner=False)
def metrics_glorys(path: str, stamp: float) -> dict | None:
    return da.load_metrics_glorys(path)


@st.cache_data(show_spinner=False)
def metrics_argo(path: str, stamp: float) -> dict | None:
    return da.load_metrics_argo(path)


@st.cache_data(show_spinner=False)
def argo_matchups(path: str, stamp: float) -> pd.DataFrame | None:
    return da.load_argo_matchups(path)


@st.cache_resource(show_spinner=False, max_entries=4)
def error_maps(path: str, stamp: float) -> xr.Dataset | None:
    """Verification maps (about 20 MB for the full grid): shared read-only, never copied."""
    return da.load_maps(path)


@st.cache_data(show_spinner=False)
def dates(path: str, stamp: float) -> list[pd.Timestamp]:
    try:
        return list(da.available_dates(path))
    except _MISSING:
        return []


@st.cache_data(show_spinner=False, max_entries=_DAY_ENTRIES)
def prediction(path: str, stamp: float, day: str) -> xr.DataArray | None:
    try:
        return da.load_prediction(path, day).drop_vars("time", errors="ignore")
    except _MISSING:
        return None


@st.cache_data(show_spinner=False, max_entries=_DAY_ENTRIES)
def target(path: str, stamp: float, day: str) -> xr.DataArray | None:
    try:
        tgt = da.load_target(path, day)
    except _MISSING:
        return None
    return None if tgt is None else tgt.drop_vars("time", errors="ignore")


@st.cache_data(show_spinner=False, max_entries=_DAY_ENTRIES)
def climatology(path: str, stamp: float, day: str) -> xr.DataArray | None:
    try:
        return da.load_climatology(path, day)
    except _MISSING:
        return None


@st.cache_data(show_spinner=False, max_entries=_DAY_ENTRIES)
def surface_inputs(path: str, stamp: float, day: str) -> xr.Dataset | None:
    try:
        ds = da.load_surface_inputs(path, day)
    except _MISSING:
        return None
    return None if ds is None else ds.drop_vars("time", errors="ignore")


@st.cache_data(show_spinner=False, max_entries=256)
def profile(path: str, stamp: float, day: str, lat: float, lon: float) -> xr.Dataset | None:
    try:
        return da.load_profile(path, day, lat, lon)
    except _MISSING:
        return None


@st.cache_data(show_spinner=False, max_entries=128)
def section(
    path: str, stamp: float, day: str, lat: float | None = None, lon: float | None = None
) -> xr.Dataset | None:
    try:
        return da.load_section(path, day, lat=lat, lon=lon)
    except _MISSING:
        return None


@st.cache_data(show_spinner=False, max_entries=_DAY_ENTRIES)
def embedding_rgb(path: str, stamp: float, day: str) -> xr.DataArray | None:
    try:
        return da.embedding_pca_rgb(path, day)
    except _MISSING:
        return None


@st.cache_data(show_spinner=False)
def ocean_mask(path: str, stamp: float) -> xr.DataArray | None:
    """Surface ocean mask ``(lat, lon)``: where the first predicted day has a surface value."""
    days = dates(path, stamp)
    if not days:
        return None
    pred = prediction(path, stamp, day_key(days[0]))
    if pred is None:
        return None
    return xr.DataArray(
        np.isfinite(pred.isel(depth=0).values),
        dims=("lat", "lon"),
        coords={"lat": pred["lat"].values, "lon": pred["lon"].values},
    )


def day_key(day) -> str:
    """Cache key of a day (ISO date string)."""
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def load_run(info: dict) -> Run:
    stamp = run_stamp(info["path"])
    return Run(info["name"], info["path"], stamp, info, run_meta(info["path"], stamp))
