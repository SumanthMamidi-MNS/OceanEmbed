"""Canonical grid, standard depths and basin masks.

Every other module obtains grid geometry from here; nothing else hard-codes lat/lon/depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oceanembed.config import STANDARD_DEPTHS, Config, GridConfig

# Basin boxes (lat_min, lat_max, lon_min, lon_max) used for the PoC regional scores.
# Arabian Sea: west of the Indian peninsula tip (~78E), east of the Horn of Africa coast.
# Bay of Bengal: east of the Indian/Sri Lankan east coast (~80E), west of the Malay peninsula.
BASINS: dict[str, tuple[float, float, float, float]] = {
    "arabian_sea": (5.0, 25.0, 45.0, 78.0),
    "bay_of_bengal": (5.0, 25.0, 80.0, 100.0),
}

__all__ = ["BASINS", "STANDARD_DEPTHS", "Grid", "build_grid", "daily_axis"]


@dataclass(frozen=True)
class Grid:
    lat: np.ndarray  # (H,) cell centres, ascending
    lon: np.ndarray  # (W,) cell centres, ascending
    depth: np.ndarray  # (D,) metres, positive down
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lat), len(self.lon)

    @property
    def n_depth(self) -> int:
        return len(self.depth)

    @property
    def lat_edges(self) -> tuple[float, float]:
        h = self.resolution / 2
        return float(self.lat[0] - h), float(self.lat[-1] + h)

    @property
    def lon_edges(self) -> tuple[float, float]:
        h = self.resolution / 2
        return float(self.lon[0] - h), float(self.lon[-1] + h)

    def basin_mask(self, name: str) -> np.ndarray:
        """Boolean (H, W) mask of the cells whose centre lies in the named basin box."""
        la0, la1, lo0, lo1 = BASINS[name]
        in_lat = (self.lat >= la0) & (self.lat <= la1)
        in_lon = (self.lon >= lo0) & (self.lon <= lo1)
        return in_lat[:, None] & in_lon[None, :]

    def basin_masks(self) -> dict[str, np.ndarray]:
        return {name: self.basin_mask(name) for name in BASINS}

    def coords(self) -> dict[str, np.ndarray]:
        return {"depth": self.depth, "lat": self.lat, "lon": self.lon}


def build_grid(cfg: Config | GridConfig) -> Grid:
    g = cfg.grid if isinstance(cfg, Config) else cfg
    res = g.resolution
    ny = round((g.lat_max - g.lat_min) / res)
    nx = round((g.lon_max - g.lon_min) / res)
    lat = g.lat_min + res * (np.arange(ny) + 0.5)
    lon = g.lon_min + res * (np.arange(nx) + 0.5)
    return Grid(
        lat=np.round(lat, 6),
        lon=np.round(lon, 6),
        depth=np.asarray(g.depths, dtype=np.float64),
        resolution=res,
    )


def daily_axis(start, end) -> pd.DatetimeIndex:
    """Daily 00:00 UTC axis, inclusive of both ends, ``datetime64[ns]``."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D").astype("datetime64[ns]")
