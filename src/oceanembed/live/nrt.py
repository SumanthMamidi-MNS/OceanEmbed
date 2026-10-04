"""Near-real-time products: what the server publishes, and a resumable per-day download.

Raw files are written one per day and product, ``<raw_nrt>/<product>/<product>_<YYYYMMDD>.nc``
(atomic: temp file + rename), so an interrupted update resumes where it stopped and a day that is
re-fetched because the product is revised replaces exactly its own file. One request to the server
covers a run of consecutive days (at most ``live.request_days``); it is split into day files here.

The two network calls (:func:`describe_dataset`, :func:`subset_to_file`) are module-level functions
so tests can replace them. Nothing here reads, prints or stores credentials.
"""

from __future__ import annotations

import logging
import tempfile
import time as _time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanembed.config import INPUT_GROUPS, Config
from oceanembed.data.providers.base import MissingCredentialsError, log_transfer
from oceanembed.data.providers.cmems import CmemsProvider

log = logging.getLogger(__name__)

# model input group -> product key of the config; the near-real-time products exist for these two
LIVE_PRODUCTS = {"sst": "sst", "sla": "sla"}


class LiveError(RuntimeError):
    """A live-mode precondition is not met (e.g. an input has no NRT product)."""


@dataclass
class InputAvailability:
    """What the catalogue says about one near-real-time input."""

    product: str
    dataset: str
    version: str | None
    first: date
    last: date
    checked_at: str

    def to_json(self) -> dict:
        return {
            "product": self.product,
            "dataset": self.dataset,
            "version": self.version,
            "first": str(self.first),
            "last": str(self.last),
            "checked_at": self.checked_at,
        }


@dataclass
class FetchResult:
    """Outcome of fetching a set of days of one product."""

    got: set[date] = field(default_factory=set)  # days the server delivered (new, changed or same)
    changed: set[date] = field(default_factory=set)  # delivered days whose existing file changed
    missing: set[date] = field(default_factory=set)  # requested but not delivered / all NaN
    bytes_transferred: int = 0
    seconds: float = 0.0
    requests: int = 0


def required_products(cfg: Config) -> list[str]:
    """Products behind the model's input groups. Live mode needs a near-real-time counterpart of
    each, and only SST and sea level anomaly have one in the config."""
    products = []
    for group in cfg.model.input_groups:
        if group not in LIVE_PRODUCTS:
            raise LiveError(
                f"the model uses the input group '{group}', which has no near-real-time product "
                f"in live mode (supported: {', '.join(LIVE_PRODUCTS)}); the released final model "
                "uses sst and sla only"
            )
        products.append(LIVE_PRODUCTS[group])
    assert all(g in INPUT_GROUPS for g in cfg.model.input_groups)
    return products


def product_dataset(cfg: Config, product: str) -> str:
    return cfg.products[product].datasets[0].id


def check_credentials() -> None:
    CmemsProvider.check_credentials()


# ----------------------------------------------------------------------------------------
# the two network calls (replaced in tests)
# ----------------------------------------------------------------------------------------
def describe_dataset(dataset_id: str) -> dict:
    """``{"version", "first", "last"}`` of a CMEMS dataset's time axis from the catalogue (read
    only, no credentials needed). Dates are the day of the first / last time stamp."""
    import copernicusmarine

    cat = copernicusmarine.describe(
        dataset_id=dataset_id, disable_progress_bar=True, show_all_versions=False
    )
    for prod in cat.products:
        for ds in prod.datasets:
            if ds.dataset_id != dataset_id:
                continue
            for ver in ds.versions:
                for part in ver.parts:
                    for service in part.services:
                        for var in service.variables:
                            for coord in var.coordinates:
                                if coord.coordinate_id == "time":
                                    lo = pd.Timestamp(float(coord.minimum_value), unit="ms")
                                    hi = pd.Timestamp(float(coord.maximum_value), unit="ms")
                                    return {
                                        "version": str(ver.label),
                                        "first": lo.date(),
                                        "last": hi.date(),
                                    }
    raise LiveError(f"dataset {dataset_id} not found in the Copernicus Marine catalogue")


def subset_to_file(**kwargs) -> None:
    """One ``copernicusmarine.subset`` call (``output_directory`` / ``output_filename`` given)."""
    import copernicusmarine

    copernicusmarine.subset(**kwargs)


def probe(cfg: Config, product: str, dataset_id: str | None = None) -> InputAvailability:
    """First and last published day of ``product`` (its configured dataset unless given)."""
    ds_id = dataset_id or product_dataset(cfg, product)
    info = describe_dataset(ds_id)
    return InputAvailability(
        product=product,
        dataset=ds_id,
        version=info.get("version"),
        first=info["first"],
        last=info["last"],
        checked_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ----------------------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------------------
def day_file(root: Path, product: str, day: date | pd.Timestamp) -> Path:
    return root / product / f"{product}_{pd.Timestamp(day):%Y%m%d}.nc"


def day_files(root: Path, product: str) -> dict[date, Path]:
    """Existing per-day raw files of a product, ``{day: path}`` (in date order)."""
    out: dict[date, Path] = {}
    folder = root / product
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob(f"{product}_????????.nc")):
        try:
            out[datetime.strptime(f.stem.split("_")[-1], "%Y%m%d").date()] = f
        except ValueError:
            continue
    return out


def runs_of_days(days: list[date], max_len: int) -> list[tuple[date, date]]:
    """Consecutive runs ``(first, last)`` of ``days`` (sorted, unique), each at most ``max_len``."""
    runs: list[tuple[date, date]] = []
    start = prev = None
    for d in sorted(set(days)):
        if start is None:
            start = prev = d
        elif d == prev + timedelta(days=1) and (d - start).days + 1 <= max_len:
            prev = d
        else:
            runs.append((start, prev))
            start = prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


def _same_field(path: Path, new: xr.DataArray) -> bool:
    try:
        with xr.open_dataset(path) as old:
            var = [v for v in old.data_vars if old[v].ndim >= 2][0]
            a = old[var].values
        b = new.values
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True))
    except (OSError, ValueError, IndexError):
        return False


def _write_day(path: Path, ds: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part.nc")
    enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
    ds.to_netcdf(part, encoding=enc)
    part.replace(path)


def fetch_days(
    cfg: Config,
    product: str,
    days: list[date],
    *,
    dataset_id: str | None = None,
    root: Path | None = None,
    revision: bool = False,
) -> FetchResult:
    """Download ``days`` of ``product`` into per-day files under ``root`` (default raw root).

    ``revision=True`` re-fetches days that already have a file and replaces the file only when the
    field really changed (``FetchResult.changed``). Days the server does not deliver, or delivers
    as all-NaN, are reported in ``missing`` and get no file."""
    root = cfg.raw_root if root is None else Path(root)
    res = FetchResult()
    if not days:
        return res
    check_credentials()
    prov = CmemsProvider(cfg)
    ds_id = dataset_id or product_dataset(cfg, product)
    var = next(iter(cfg.products[product].variables.values()))
    for first, last in runs_of_days(days, cfg.live.request_days if cfg.live else 31):
        kw = prov.subset_kwargs(product, first, last) | {"dataset_id": ds_id}
        t0 = _time.time()
        with tempfile.TemporaryDirectory(dir=_tmp_parent(root), prefix=f"{product}_") as tmp:
            part = Path(tmp) / "range.nc"
            _subset_with_retries(cfg, kw, part)
            res.requests += 1
            size = part.stat().st_size
            with xr.open_dataset(part) as raw:
                got = _split_days(root, product, var, raw, first, last, revision, res)
            res.bytes_transferred += size
        secs = _time.time() - t0
        res.seconds += secs
        log_transfer(
            cfg, product=product, dataset=ds_id, period=f"{first}..{last}",
            route="cmems_subset_nrt", bytes_written=size, bytes_transferred=size,
            transfer_basis="file_size", seconds=secs,
        )  # fmt: skip
        log.info("%s %s..%s: %d day(s) from %s", product, first, last, len(got), ds_id)
    res.missing = set(days) - res.got
    return res


def _tmp_parent(root: Path) -> Path:
    tmp = root / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _split_days(root, product, var, raw: xr.Dataset, first, last, revision, res) -> set[date]:
    time_name = next((n for n in ("time", "valid_time", "t") if n in raw.coords), None)
    if time_name is None or var not in raw:
        return set()
    stamps = pd.DatetimeIndex(raw[time_name].values)
    got: set[date] = set()
    for i, ts in enumerate(stamps):
        day = ts.date()
        if not (first <= day <= last):
            continue
        sub = raw.isel({time_name: [i]}).load()
        if not np.isfinite(sub[var].values).any():
            continue  # a day the server lists but delivers empty: not available
        path = day_file(root, product, day)
        if revision and path.exists():
            res.got.add(day)
            got.add(day)
            if _same_field(path, sub[var]):
                continue
            res.changed.add(day)
        _write_day(path, sub)
        res.got.add(day)
        got.add(day)
    return got


def _subset_with_retries(cfg: Config, kw: dict, target: Path) -> None:
    attempts = cfg.download.retries + 1
    for attempt in range(attempts):
        try:
            subset_to_file(
                **kw, output_directory=str(target.parent), output_filename=target.name,
                overwrite=True,
            )  # fmt: skip
            if not target.exists():
                raise LiveError("the server returned no file")
            return
        except MissingCredentialsError:
            raise
        except Exception as e:  # noqa: BLE001 - the client raises many types on network trouble
            if attempt == attempts - 1:
                raise
            wait = cfg.download.retry_backoff_s * 2**attempt
            log.warning("request failed (%s); retry %d/%d in %.0f s", type(e).__name__, attempt + 1,
                        cfg.download.retries, wait)  # fmt: skip
            _time.sleep(wait)
