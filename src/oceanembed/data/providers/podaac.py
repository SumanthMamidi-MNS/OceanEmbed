"""PO.DAAC provider (via ``earthaccess``): OSCAR surface currents and CCMP winds.

Both products are distributed as one *global* granule per day (~32 MB). Two routes:

``opendap`` (default)
    Per-granule **server-side subsetting** through NASA's Hyrax OPeNDAP service
    (``opendap.earthdata.nasa.gov``). Only the configured variables and the index ranges that
    cover the domain plus halo are requested (a DAP4 constraint expression, returned as a small
    NetCDF-4 file): about 0.2 MB (OSCAR) / 0.75 MB (CCMP) per day instead of 32 MB. Index ranges
    are derived from the coordinate arrays of the first granule, not hard-coded.
``granule``
    Download the whole global granule, crop it locally, delete it. Used on request
    (``download.podaac_mode: granule``) and as the **automatic fallback** when OPeNDAP refuses
    (HTTP 4xx, e.g. the Earthdata application is not authorised) or keeps failing.

Every day is a separate *piece* written atomically (temp file + rename) to
``<raw>/_parts/<product>_<YYYYMM>/<YYYYMMDD>.nc``. Finished pieces are skipped on a re-run, so an
interrupted download resumes where it stopped. Pieces are fetched by a small thread pool, retried
with exponential back-off on transient errors, and merged into the monthly file
``<raw>/<product>/<product>_<YYYYMM>.nc`` only once the whole month is present; the pieces are
then removed. Every piece is recorded in the transfer ledger (``base.log_transfer``).

Authentication is the user's own NASA Earthdata login (``~/.netrc`` entry for
``urs.earthdata.nasa.gov`` or ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), used through the
authenticated session that ``earthaccess`` provides; this module never reads or logs the values.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import threading
import time
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.providers.base import (
    MissingCredentialsError,
    log_transfer,
    months,
    raw_dir,
    raw_file,
    region_with_halo,
    source_for_range,
)
from oceanembed.data.regrid import crop, standardize

log = logging.getLogger(__name__)

_DAP_NS = {"d": "http://xml.opendap.org/ns/DAP/4.0#"}
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_FAILURES_BEFORE_FALLBACK = 3  # OPeNDAP refusals / exhausted retries before it is switched off


class SubsettingRefused(RuntimeError):
    """The OPeNDAP service refused (HTTP 4xx, an unexpected reply, or an unusable layout)."""


class TransientError(RuntimeError):
    """A retryable failure (network error, HTTP 429 / 5xx); raised for good when retries run out."""


@dataclass
class Reply:
    status: int
    content_type: str
    body: bytes
    wire_bytes: int  # bytes received on the connection (compressed, as transferred)


# ----------------------------------------------------------------------------------------
# credentials
# ----------------------------------------------------------------------------------------
def _netrc_has_earthdata() -> bool:
    for name in (".netrc", "_netrc"):
        f = Path.home() / name
        try:
            if f.exists() and "urs.earthdata.nasa.gov" in f.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def credentials_strategy() -> str | None:
    """``'environment'`` / ``'netrc'`` when Earthdata credentials are configured, else None."""
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return "environment"
    if _netrc_has_earthdata():
        return "netrc"
    return None


# ----------------------------------------------------------------------------------------
# OPeNDAP helpers (pure functions, unit-tested without a network)
# ----------------------------------------------------------------------------------------
@dataclass
class Layout:
    """What the DMR of one granule says: dimension sizes, the dims of each data variable, and
    which coordinate variable belongs to each dimension (OSCAR: dim ``longitude`` / variable
    ``lon``) and to which axis (``lon`` / ``lat`` / ``time``) it belongs."""

    dims: dict[str, int]
    var_dims: dict[str, list[str]]
    coord_var: dict[str, str]
    axis_dim: dict[str, str]  # 'lon' / 'lat' / 'time' -> dimension name
    lon_runs: list[tuple[int, int]] = field(default_factory=list)  # index runs (inclusive)
    lat_run: tuple[int, int] = (0, 0)


def parse_dmr(xml: bytes | str) -> Layout:
    """Parse a DAP4 DMR document into a :class:`Layout` (axes are recognised by ``units``)."""
    root = ET.fromstring(xml)
    dims = {d.get("name"): int(d.get("size")) for d in root.findall("d:Dimension", _DAP_NS)}
    var_dims: dict[str, list[str]] = {}
    units: dict[str, str] = {}
    for el in root:
        name = el.get("name")
        dl = [x.get("name").lstrip("/") for x in el.findall("d:Dim", _DAP_NS)]
        if not name or not dl:
            continue
        var_dims[name] = dl
        for a in el.findall("d:Attribute", _DAP_NS):
            if a.get("name") == "units":
                v = a.find("d:Value", _DAP_NS)
                units[name] = (v.text or "").strip() if v is not None else ""
    coord_var = {}
    axis_dim: dict[str, str] = {}
    for name, dl in var_dims.items():
        if len(dl) != 1:
            continue
        d, u = dl[0], units.get(name, "").lower()
        coord_var.setdefault(d, name)
        if u.startswith("degrees_east") or u == "degree_east":
            axis_dim["lon"] = d
        elif u.startswith("degrees_north") or u == "degree_north":
            axis_dim["lat"] = d
        elif "since" in u or d == "time":
            axis_dim.setdefault("time", d)
    return Layout(dims, var_dims, coord_var, axis_dim)


def index_runs(
    coord: np.ndarray, lo: float, hi: float, periodic: bool = False
) -> list[tuple[int, int]]:
    """Inclusive index runs ``(i0, i1)`` of ``coord`` values inside ``[lo, hi]``.

    ``coord`` may be ascending or descending (an index run is contiguous either way). With
    ``periodic`` (longitude) both the coordinate and the window are compared modulo 360, so a
    0-360 or -180-180 source and a window given in either convention work; a window that crosses
    the 0/360 seam yields two runs."""
    c = np.asarray(coord, dtype=float)
    eps = 1e-6
    if periodic:
        c = np.mod(c, 360.0)
        span = hi - lo
        if span >= 360.0:
            mask = np.ones(c.shape, dtype=bool)
        else:
            lo_n = np.mod(lo, 360.0)
            hi_n = lo_n + span
            if hi_n < 360.0:
                mask = (c >= lo_n - eps) & (c <= hi_n + eps)
            else:
                mask = (c >= lo_n - eps) | (c <= hi_n - 360.0 + eps)
    else:
        mask = (c >= lo - eps) & (c <= hi + eps)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"no coordinate values inside [{lo}, {hi}]")
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def build_constraint(layout: Layout, variables: list[str], windows: dict[str, tuple[int, int]]):
    """DAP4 constraint expression: the coordinate variables of the dimensions in use (sliced where
    ``windows`` gives an inclusive index range for that dimension) plus every requested variable,
    sliced per dimension (dimensions without a window are taken whole)."""
    missing = [v for v in variables if v not in layout.var_dims]
    if missing:
        raise SubsettingRefused(f"variables {missing} not in the granule ({list(layout.var_dims)})")

    def sl(dim: str) -> str:
        i0, i1 = windows.get(dim, (0, layout.dims[dim] - 1))
        return f"[{i0}:{i1}]"

    used: list[str] = []
    for v in variables:
        for d in layout.var_dims[v]:
            if d not in used:
                used.append(d)
    parts = []
    for d in used:
        cv = layout.coord_var.get(d)
        if cv is None:
            raise SubsettingRefused(f"dimension {d!r} has no coordinate variable")
        parts.append(f"/{cv}" + (sl(d) if d in windows else ""))
    for v in variables:
        parts.append(f"/{v}" + "".join(sl(d) for d in layout.var_dims[v]))
    return ";".join(parts)


def granule_day(g) -> date:
    """UTC calendar day of a CMR granule record (begin of its temporal extent)."""
    try:
        s = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        return date.fromisoformat(str(s)[:10])
    except (KeyError, TypeError, ValueError):
        m = re.search(r"(\d{4})(\d{2})(\d{2})", str(g["umm"].get("GranuleUR", "")))
        if not m:
            raise ValueError("cannot determine the day of a granule") from None
        return date(*(int(x) for x in m.groups()))


def opendap_url(g) -> str:
    """OPeNDAP base URL (no query) of a granule record; built from the concept ids if the CMR
    record does not list it."""
    for link in g["umm"].get("RelatedUrls", []):
        u = str(link.get("URL", ""))
        if "opendap.earthdata.nasa.gov" in u:
            return u.split("?")[0]
    coll = g["meta"]["collection-concept-id"]
    return f"https://opendap.earthdata.nasa.gov/collections/{coll}/granules/{g['umm']['GranuleUR']}"


def _write_nc(ds: xr.Dataset, path: Path) -> None:
    """Atomic write: temp file in the same folder, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    enc = {v: {"zlib": True, "complevel": 1, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------
# provider
# ----------------------------------------------------------------------------------------
class PodaacProvider:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._sess = None
        self._lock = threading.Lock()
        self._layouts: dict[str, Layout] = {}
        self._failures = 0
        self._opendap_off = False

    # ---- login -----------------------------------------------------------------------
    @staticmethod
    def login():
        import earthaccess

        strategy = credentials_strategy()
        if strategy is None:
            raise MissingCredentialsError(
                "No NASA Earthdata credentials found. Register (free) at "
                "https://urs.earthdata.nasa.gov, then either add a `machine "
                "urs.earthdata.nasa.gov login <user> password <pass>` line to your ~/.netrc "
                "(_netrc on Windows) or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD, and "
                "repeat the download."
            )
        auth = earthaccess.login(strategy=strategy)
        if not getattr(auth, "authenticated", True):
            raise MissingCredentialsError("Earthdata login failed; check your credentials.")
        return auth

    def search_kwargs(self, product: str, first: date, last: date) -> dict:
        """Arguments passed to ``earthaccess.search_data`` for one month."""
        src = source_for_range(self.cfg.products[product], first, last)
        lon0, lon1, lat0, lat1 = region_with_halo(self.cfg)
        return {
            "short_name": src.id,
            "temporal": (first.isoformat(), last.isoformat()),
            "bounding_box": (lon0, lat0, lon1, lat1),
        }

    # ---- low-level HTTP (mocked in tests) ---------------------------------------------
    def _session(self):
        if self._sess is None:
            import earthaccess

            self._sess = earthaccess.get_requests_https_session()
        return self._sess

    def _get(self, url: str, params: dict | None = None) -> Reply:
        """One GET with the authenticated Earthdata session; never retries, never logs the URL."""
        import requests
        import urllib3

        try:
            r = self._session().get(url, params=params, stream=True, timeout=(15, 180))
            # read the bytes as transferred (still compressed) so the ledger counts the wire size
            raw = r.raw.read(decode_content=False)
            enc = r.headers.get("content-encoding", "").lower()
            if enc == "gzip":
                body = zlib.decompress(raw, 16 + zlib.MAX_WBITS)
            elif enc == "deflate":
                body = zlib.decompress(raw)
            else:
                body = raw
            return Reply(r.status_code, r.headers.get("content-type", ""), body, len(raw))
        except (requests.RequestException, urllib3.exceptions.HTTPError, zlib.error, OSError) as e:
            raise TransientError(type(e).__name__) from e

    @staticmethod
    def _check(reply: Reply, expect: str) -> Reply:
        s = reply.status
        if s in _TRANSIENT_STATUS:
            raise TransientError(f"HTTP {s}")
        if s != 200:
            raise SubsettingRefused(f"HTTP {s}")
        if expect not in reply.content_type.lower():
            raise SubsettingRefused(f"unexpected reply type {reply.content_type!r}")
        return reply

    def _with_retries(self, fn, what: str):
        """Call ``fn`` and retry on :class:`TransientError` with exponential back-off."""
        n, wait = self.cfg.download.retries, self.cfg.download.retry_backoff_s
        for attempt in range(n + 1):
            try:
                return fn()
            except TransientError as e:
                if attempt == n:
                    raise TransientError(f"{what}: {e} (after {n + 1} attempts)") from e
                delay = wait * 2**attempt
                log.warning("%s: %s; retry %d/%d in %.0fs", what, e, attempt + 1, n, delay)
                time.sleep(delay)

    # ---- OPeNDAP route ------------------------------------------------------------------
    def _layout(self, product: str, g) -> Layout:
        """Layout + index windows of the product, fetched once from the first granule."""
        with self._lock:
            if product in self._layouts:
                return self._layouts[product]
            base = opendap_url(g)
            pcfg = self.cfg.products[product]
            t0 = time.time()
            rep = self._with_retries(lambda: self._check(self._get(base + ".dmr"), "xml"), "DMR")
            layout = parse_dmr(rep.body)
            if "lon" not in layout.axis_dim or "lat" not in layout.axis_dim:
                raise SubsettingRefused("could not identify the lon / lat dimensions")
            lon_d, lat_d = layout.axis_dim["lon"], layout.axis_dim["lat"]
            ce = ";".join(f"/{layout.coord_var[d]}" for d in (lon_d, lat_d))
            rep2 = self._with_retries(
                lambda: self._check(self._get(base + ".dap.nc4", {"dap4.ce": ce}), "netcdf"),
                "coordinates",
            )
            with xr.open_dataset(io.BytesIO(rep2.body)) as cds:
                lon = cds[layout.coord_var[lon_d]].values
                lat = cds[layout.coord_var[lat_d]].values
            lon0, lon1, lat0, lat1 = region_with_halo(self.cfg)
            layout.lon_runs = index_runs(lon, lon0, lon1, periodic=True)
            layout.lat_run = index_runs(lat, lat0, lat1)[0]
            missing = [v for v in pcfg.variables.values() if v not in layout.var_dims]
            if missing:
                raise SubsettingRefused(f"variables {missing} not found ({list(layout.var_dims)})")
            self._layouts[product] = layout
            src = source_for_range(pcfg, granule_day(g), granule_day(g))
            log_transfer(
                self.cfg, product=product, dataset=src.id, period="layout", route="opendap",
                bytes_written=0, bytes_transferred=rep.wire_bytes + rep2.wire_bytes,
                transfer_basis="wire", seconds=time.time() - t0,
            )  # fmt: skip
            log.info(
                "%s OPeNDAP layout: lon index runs %s, lat index run %s", product,
                layout.lon_runs, layout.lat_run,
            )  # fmt: skip
            return layout

    def _opendap_day(self, product: str, g) -> tuple[xr.Dataset, int]:
        """Subset of one daily granule via OPeNDAP -> (standardised dataset, wire bytes)."""
        layout = self._layout(product, g)
        pcfg = self.cfg.products[product]
        names = list(pcfg.variables.values())
        base = opendap_url(g)
        wire = 0
        pieces = []
        for run in layout.lon_runs:
            ce = build_constraint(
                layout, names,
                {layout.axis_dim["lon"]: run, layout.axis_dim["lat"]: layout.lat_run},
            )  # fmt: skip
            rep = self._with_retries(
                lambda ce=ce: self._check(self._get(base + ".dap.nc4", {"dap4.ce": ce}), "netcdf"),
                f"{product} {granule_day(g)}",
            )
            wire += rep.wire_bytes
            with xr.open_dataset(io.BytesIO(rep.body)) as ds:
                ds = ds.load()
            pieces.append(self._tidy(ds, product)[names])
        ds = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="lon").sortby("lon")
        return ds, wire

    # ---- granule route --------------------------------------------------------------------
    def _tidy(self, ds: xr.Dataset, product: str) -> xr.Dataset:
        ds = standardize(ds, self.cfg.products[product].coords)
        ds.attrs = {}
        return ds.transpose("time", "lat", "lon", ...)

    def _subset_granule(self, path: Path, product: str) -> xr.Dataset:
        pcfg = self.cfg.products[product]
        lon0, lon1, lat0, lat1 = region_with_halo(self.cfg)
        with xr.open_dataset(path) as ds:
            ds = standardize(ds, pcfg.coords)
            missing = [v for v in pcfg.variables.values() if v not in ds]
            if missing:
                raise KeyError(f"{path.name}: variables {missing} not found ({list(ds.data_vars)})")
            ds = crop(ds[list(pcfg.variables.values())], (lat0, lat1), (lon0, lon1))
            ds = ds.load()
        return self._tidy(ds, product)

    def _granule_day(self, product: str, g, day: date) -> tuple[xr.Dataset, int]:
        """Global granule download + local crop -> (dataset, bytes downloaded)."""
        import earthaccess

        scratch = raw_dir(self.cfg, "_granules") / f"{product}_{day:%Y%m}"
        scratch.mkdir(parents=True, exist_ok=True)

        def download() -> Path:
            try:
                files = earthaccess.download([g], local_path=scratch)
            except Exception as e:  # earthaccess raises assorted network errors
                raise TransientError(type(e).__name__) from e
            if not files or isinstance(files[0], Exception):
                raise TransientError("granule download failed")
            return Path(str(files[0]))

        f = self._with_retries(download, f"{product} {day} granule")
        try:
            size = f.stat().st_size
            return self._subset_granule(f, product), size
        finally:
            if self.cfg.download.delete_global_granules:
                f.unlink(missing_ok=True)

    # ---- one day -------------------------------------------------------------------------
    def _note_failure(self, why: str) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= _MAX_FAILURES_BEFORE_FALLBACK and not self._opendap_off:
                self._opendap_off = True
                log.warning(
                    "OPeNDAP subsetting failed %d times (last: %s): using global granules for the "
                    "rest of this run", self._failures, why,
                )  # fmt: skip

    def _fetch_day(self, product: str, g, parts_dir: Path) -> str:
        day = granule_day(g)
        part = parts_dir / f"{day:%Y%m%d}.nc"
        if part.exists():
            return "cached"
        src = source_for_range(self.cfg.products[product], day, day)
        t0 = time.time()
        route = "granule"
        ds = None
        if self.cfg.download.podaac_mode == "opendap" and not self._opendap_off:
            try:
                ds, transferred = self._opendap_day(product, g)
                route = "opendap"
            except (SubsettingRefused, TransientError) as e:
                if isinstance(e, SubsettingRefused) and not self._layouts.get(product):
                    with self._lock:  # refused before any layout: do not keep trying
                        self._failures = _MAX_FAILURES_BEFORE_FALLBACK
                self._note_failure(str(e))
                log.warning(
                    "%s %s: OPeNDAP subsetting unavailable (%s); falling back to the global "
                    "granule download", product, day, e,
                )  # fmt: skip
                route = "granule_fallback"
        elif self.cfg.download.podaac_mode == "opendap":
            route = "granule_fallback"
        if ds is None:
            ds, transferred = self._granule_day(product, g, day)
        _write_nc(ds, part)
        log_transfer(
            self.cfg, product=product, dataset=src.id, period=day.isoformat(), route=route,
            bytes_written=part.stat().st_size, bytes_transferred=transferred,
            transfer_basis="wire" if route == "opendap" else "file_size",
            seconds=time.time() - t0,
        )  # fmt: skip
        return route

    # ---- one month -------------------------------------------------------------------------
    def _fetch_month(self, product: str, first: date, last: date) -> Path:
        import earthaccess

        path = raw_file(self.cfg, product, first)
        results = earthaccess.search_data(**self.search_kwargs(product, first, last))
        by_day: dict[date, object] = {}
        for g in results:
            d = granule_day(g)
            if first <= d <= last:
                by_day.setdefault(d, g)
        if not by_day:
            raise RuntimeError(f"no {product} granules found for {first:%Y-%m}")
        n_expected = (last - first).days + 1
        if len(by_day) < n_expected:
            log.warning(
                "%s %s: the catalogue has %d of %d days; the rest will be NaN",
                product, f"{first:%Y-%m}", len(by_day), n_expected,
            )  # fmt: skip
        parts_dir = raw_dir(self.cfg, "_parts") / f"{product}_{first:%Y%m}"
        if self.cfg.download.overwrite:
            shutil.rmtree(parts_dir, ignore_errors=True)
        parts_dir.mkdir(parents=True, exist_ok=True)
        for stale in parts_dir.glob("*.tmp"):  # leftovers of an interrupted write
            stale.unlink(missing_ok=True)
        todo = [g for d, g in sorted(by_day.items()) if not (parts_dir / f"{d:%Y%m%d}.nc").exists()]
        if todo:
            # the layout lookup (first request) is serialised by a lock; days then run in parallel
            errors: list[str] = []
            with ThreadPoolExecutor(max_workers=self.cfg.download.workers) as ex:
                futures = [ex.submit(self._fetch_day, product, g, parts_dir) for g in todo]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as e:  # collect, finish the others, report below
                        errors.append(f"{type(e).__name__}: {e}")
            if errors:
                raise RuntimeError(
                    f"{product} {first:%Y-%m}: {len(errors)} of {len(todo)} day(s) failed "
                    f"(first error: {errors[0]}); finished days are kept in {parts_dir}, "
                    "re-run the download to resume"
                )
        files = [parts_dir / f"{d:%Y%m%d}.nc" for d in sorted(by_day)]
        parts = []
        for f in files:
            with xr.open_dataset(f) as ds:
                parts.append(ds.load())
        merged = xr.concat(parts, dim="time").sortby("time")
        merged.attrs = {"source": "NASA PO.DAAC via OPeNDAP / granules; see oceanembed docs"}
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_nc(merged, path)
        shutil.rmtree(parts_dir, ignore_errors=True)
        try:
            parts_dir.parent.rmdir()  # drop the empty _parts folder
        except OSError:
            pass
        return path

    def fetch(self, variable: str, start: date, end: date) -> list[Path]:
        if variable not in self.cfg.products or self.cfg.products[variable].provider != "podaac":
            raise ValueError(f"{variable!r} is not a PO.DAAC product in this config")
        self.login()
        written: list[Path] = []
        for first, last in tqdm(list(months(start, end)), desc=f"podaac {variable}", leave=False):
            path = raw_file(self.cfg, variable, first)
            if path.exists() and not self.cfg.download.overwrite:
                continue
            written.append(self._fetch_month(variable, first, last))
        return written
