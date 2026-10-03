"""Copernicus Marine (CMEMS) provider: OSTIA SST, MULTIOBS SSS, DUACS sea level, GLORYS12.

Uses ``copernicusmarine.subset`` month by month on the domain plus a halo; files that already
exist are skipped. Credentials are the user's own (``copernicusmarine login`` or the
``COPERNICUSMARINE_SERVICE_USERNAME`` / ``..._PASSWORD`` environment variables); nothing is
ever read, stored or logged by this module.
"""

from __future__ import annotations

import logging
import os
import time as _time
from datetime import date, datetime, time
from pathlib import Path

from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.providers.base import (
    MissingCredentialsError,
    log_transfer,
    months,
    raw_file,
    region_with_halo,
    source_for_range,
)

log = logging.getLogger(__name__)

CREDENTIALS_FILE = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"


def credentials_available() -> bool:
    """True if CMEMS credentials are configured (env vars or the login credentials file)."""
    env_ok = bool(
        os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    )
    return env_ok or CREDENTIALS_FILE.exists()


class CmemsProvider:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def check_credentials() -> None:
        if not credentials_available():
            raise MissingCredentialsError(
                "No Copernicus Marine credentials found. Create a free account at "
                "https://marine.copernicus.eu and run `copernicusmarine login` once "
                "(or set COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD), "
                "then repeat the download."
            )

    def subset_kwargs(self, product: str, first: date, last: date) -> dict:
        """The exact arguments passed to ``copernicusmarine.subset`` for one month."""
        pcfg = self.cfg.products[product]
        src = source_for_range(pcfg, first, last)
        lon0, lon1, lat0, lat1 = region_with_halo(self.cfg)
        kw: dict = {
            "dataset_id": src.id,
            "variables": list(pcfg.variables.values()),
            "minimum_longitude": lon0,
            "maximum_longitude": lon1,
            "minimum_latitude": lat0,
            "maximum_latitude": lat1,
            "start_datetime": datetime.combine(first, time(0, 0, 0)).isoformat(),
            "end_datetime": datetime.combine(last, time(23, 59, 59)).isoformat(),
            "coordinates_selection_method": "inside",
            "netcdf_compression_level": 1,
            "disable_progress_bar": True,
        }
        if pcfg.min_depth is not None:
            kw["minimum_depth"] = pcfg.min_depth
        if pcfg.max_depth is not None:
            kw["maximum_depth"] = pcfg.max_depth
        return kw

    def fetch(self, variable: str, start: date, end: date) -> list[Path]:
        import copernicusmarine  # imported late so tests can patch ``subset``

        if variable not in self.cfg.products or self.cfg.products[variable].provider != "cmems":
            raise ValueError(f"{variable!r} is not a CMEMS product in this config")
        self.check_credentials()
        written: list[Path] = []
        for first, last in tqdm(list(months(start, end)), desc=f"cmems {variable}", leave=False):
            path = raw_file(self.cfg, variable, first)
            if path.exists() and not self.cfg.download.overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part.nc")
            kw = self.subset_kwargs(variable, first, last)
            log.info("cmems %s %s -> %s", variable, kw["dataset_id"], path.name)
            t0 = _time.time()
            copernicusmarine.subset(
                output_directory=str(path.parent),
                output_filename=part.name,
                overwrite=True,
                **kw,
            )
            part.replace(path)
            size = path.stat().st_size
            log_transfer(
                self.cfg, product=variable, dataset=kw["dataset_id"], period=f"{first:%Y-%m}",
                route="cmems_subset", bytes_written=size, bytes_transferred=size,
                transfer_basis="file_size", seconds=_time.time() - t0,
            )  # fmt: skip
            written.append(path)
        return written
