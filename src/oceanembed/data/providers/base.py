"""Provider interface and helpers shared by the synthetic and real download providers.

A provider writes **raw monthly files on the product's native grid** to
``<data_root>/<raw_dir>/<product>/<product>_<YYYYMM>.nc`` (Argo: ``argo_<YYYYMM>.parquet``).
Harmonisation reads only those files, so real and synthetic data take the same path.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from oceanembed.config import Config, DatasetSource, ProductConfig

GRIDDED_PRODUCTS = ["sst", "sss", "sla", "currents", "winds", "temp"]
ALL_PRODUCTS = [*GRIDDED_PRODUCTS, "argo"]

# tidy Argo table schema shared by the synthetic and the real Argo provider
ARGO_COLUMNS = [
    "profile_id", "platform", "cycle", "time", "latitude", "longitude",
    "pres", "depth", "temp", "temp_qc",
]  # fmt: skip


class MissingCredentialsError(RuntimeError):
    """Raised before any network call when the required login is not configured."""


class Provider(Protocol):
    def fetch(self, variable: str, start: date, end: date) -> list[Path]:
        """Write raw files for ``variable`` (a product name, or ``argo``) covering
        ``[start, end]`` and return the paths written (existing files are skipped)."""
        ...


LEDGER_NAME = "_download_log.jsonl"
_LEDGER_LOCK = threading.Lock()


def ledger_path(cfg: Config) -> Path:
    return cfg.raw_root / LEDGER_NAME


def log_transfer(
    cfg: Config,
    *,
    product: str,
    period: str,
    route: str,
    bytes_written: int,
    seconds: float,
    bytes_transferred: int | None = None,
    transfer_basis: str = "unknown",
    dataset: str | None = None,
) -> None:
    """Append one JSON line to the transfer ledger ``<raw>/_download_log.jsonl``.

    ``bytes_written`` is what ended up on disk; ``bytes_transferred`` is what crossed the network
    (``transfer_basis`` says how it was obtained: ``wire`` = measured compressed bytes on the
    connection, ``file_size`` = taken from the downloaded file's size, ``unknown`` = not
    measurable). Only product / dataset id / period / route / sizes / timing are recorded: never
    credentials, user names or (signed) URLs."""
    rec = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product": product,
        "dataset": dataset,
        "period": period,
        "route": route,
        "bytes_written": int(bytes_written),
        "bytes_transferred": None if bytes_transferred is None else int(bytes_transferred),
        "transfer_basis": transfer_basis,
        "seconds": round(float(seconds), 3),
    }
    path = ledger_path(cfg)
    with _LEDGER_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def months(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield ``(first_day, last_day)`` of each calendar month overlapping ``[start, end]``,
    clipped to the requested range."""
    cur = pd.Timestamp(start).replace(day=1)
    stop = pd.Timestamp(end)
    while cur <= stop:
        nxt = cur + pd.offsets.MonthBegin(1)
        first = max(cur, pd.Timestamp(start))
        last = min(nxt - pd.Timedelta(days=1), stop)
        yield first.date(), last.date()
        cur = nxt


def raw_dir(cfg: Config, product: str) -> Path:
    return cfg.raw_root / product


def raw_file(cfg: Config, product: str, month_start: date) -> Path:
    ext = "parquet" if product == "argo" else "nc"
    return raw_dir(cfg, product) / f"{product}_{month_start:%Y%m}.{ext}"


def source_for_range(product: ProductConfig, first: date, last: date) -> DatasetSource:
    """The configured dataset whose validity window contains the whole ``[first, last]``.

    Switch-over dates between datasets (e.g. reprocessed -> near-real-time) must therefore fall
    on month boundaries."""
    for ds in product.datasets:
        if (ds.start is None or ds.start <= first) and (ds.end is None or ds.end >= last):
            return ds
    raise ValueError(
        f"no configured dataset covers {first} .. {last} entirely; check the dataset "
        "start/end dates (switch-over dates must be month boundaries)"
    )


def region_with_halo(cfg: Config) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of the domain plus the download halo."""
    h = cfg.download.halo_deg
    g = cfg.grid
    return g.lon_min - h, g.lon_max + h, g.lat_min - h, g.lat_max + h


def make_provider(cfg: Config, product: str) -> Provider:
    """Provider responsible for ``product`` under the configured provider mode."""
    if cfg.provider == "synthetic":
        from oceanembed.data.providers.synthetic import SyntheticProvider

        return SyntheticProvider(cfg)
    if product == "argo":
        from oceanembed.data.providers.argo import ArgoProvider

        return ArgoProvider(cfg)
    p = cfg.products[product]
    if p.provider == "cmems":
        from oceanembed.data.providers.cmems import CmemsProvider

        return CmemsProvider(cfg)
    from oceanembed.data.providers.podaac import PodaacProvider

    return PodaacProvider(cfg)
