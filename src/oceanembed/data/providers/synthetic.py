"""Synthetic provider: writes raw products in the real products' layout and native grids.

Mirrors what the real providers deliver so harmonisation is exercised identically:

=========  ======================  ==============================================================
product    native grid             conventions imitated
=========  ======================  ==============================================================
sst        0.05 deg daily          OSTIA: kelvin, int16-packed, time stamped 12:00, lat/lon
sss        0.125 deg daily         MULTIOBS ``sos``: int16-packed, NaN in a coastal band
sla        0.25 deg daily          DUACS ``sla`` (m), int16-packed
currents   0.25 deg daily          OSCAR: dims (time, longitude, latitude), latitude DESCENDING,
                                   grid nodes on multiples of 0.25 (half-cell offset from the
                                   canonical centres -> true bilinear interpolation)
winds      0.25 deg 6-hourly       CCMP: dims (time, latitude, longitude), 4 samples per day
temp       1/12 deg daily, 36      GLORYS ``thetao``: GLORYS depth levels 0.49 .. 1062 m,
           non-standard levels     int16-packed, NaN below the sea floor / on land
=========  ======================  ==============================================================

Months are generated one at a time, so the full 1/12 deg 3-D field is never held in memory.
Output is deterministic given ``cfg.synthetic.seed``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.providers.base import (
    ARGO_COLUMNS,
    GRIDDED_PRODUCTS,
    months,
    raw_dir,
    raw_file,
    region_with_halo,
)
from oceanembed.data.providers.synthetic_world import (
    SyntheticWorld,
    bathymetry,
    coast_distance,
    days_since_2000,
    is_land,
    pressure_to_depth,
)

# GLORYS12 vertical levels (m) down to the ~1100 m limit used for this project
GLORYS_LEVELS = np.array(
    [0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.930, 9.573, 11.405, 13.467, 15.810, 18.496,
     21.599, 25.211, 29.445, 34.434, 40.344, 47.373, 55.764, 65.807, 77.854, 92.326, 109.729,
     130.666, 155.851, 186.126, 222.475, 266.040, 318.127, 380.213, 453.938, 541.089, 643.567,
     763.333, 902.339, 1062.440]
)  # fmt: skip

# half = cell-centred grid (nodes at (k + 1/2) * res); otherwise nodes at k * res
_LAYOUT = {
    "sst": {"half": True},
    "sss": {"half": True},
    "sla": {"half": True},
    "currents": {"half": False},
    "winds": {"half": True},
    "temp": {"half": False},
}


def native_axis(lo: float, hi: float, res: float, half: bool) -> np.ndarray:
    off = res / 2 if half else 0.0
    k0 = int(np.ceil((lo - off) / res - 1e-9))
    k1 = int(np.floor((hi - off) / res + 1e-9))
    return np.round(off + res * np.arange(k0, k1 + 1), 6)


def _pack(scale: float, offset: float = 0.0) -> dict:
    return {
        "dtype": "int16",
        "scale_factor": scale,
        "add_offset": offset,
        "_FillValue": np.int16(-32768),
        "zlib": True,
        "complevel": 1,
        "shuffle": True,
    }


_FLOAT_ENC = {
    "dtype": "float32",
    "zlib": True,
    "complevel": 1,
    "shuffle": True,
    "_FillValue": np.nan,
}


class SyntheticProvider:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.halo = cfg.download.halo_deg
        lon0, lon1, lat0, lat1 = region_with_halo(cfg)
        self.lat_rng, self.lon_rng = (lat0, lat1), (lon0, lon1)
        t0 = float(days_since_2000(np.datetime64(cfg.time.start)))
        t1 = float(days_since_2000(np.datetime64(cfg.time.end)))
        self.world = SyntheticWorld(
            cfg.synthetic.seed,
            lat_range=(cfg.grid.lat_min, cfg.grid.lat_max),
            lon_range=(cfg.grid.lon_min, cfg.grid.lon_max),
            t_range=(t0, t1),
            eddies_per_100deg2_year=cfg.synthetic.n_eddies_per_100_deg2_year,
        )

    # ------------------------------------------------------------------ helpers
    def _axes(self, product: str):
        res = self.cfg.synthetic.native_resolution[product]
        half = _LAYOUT[product]["half"]
        lat = native_axis(*self.lat_rng, res, half)
        lon = native_axis(*self.lon_rng, res, half)
        return lat, lon

    def _days(self, first: date, last: date) -> np.ndarray:
        return pd.date_range(first, last, freq="D").values.astype("datetime64[ns]")

    def _noise(self, code: int, day_t: float, shape, sigma: float) -> np.ndarray:
        rng = np.random.default_rng([self.cfg.synthetic.seed, code, int(day_t) + 100_000])
        return (sigma * rng.standard_normal(shape)).astype(np.float32)

    def _write(self, ds: xr.Dataset, path: Path, enc: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp, encoding=enc)
        tmp.replace(path)

    # ------------------------------------------------------------------ products
    def _sst(self, days) -> xr.Dataset:
        lat, lon = self._axes("sst")
        la, lo = lat[:, None], lon[None, :]
        land = is_land(la, lo)
        out = np.empty((len(days), len(lat), len(lon)), np.float32)
        for i, d in enumerate(days):
            t = float(days_since_2000(d)) + 0.5
            sst = self.world.sst_surface(la, lo, t) + self.world.fine_sst(la, lo, t)
            sst = sst + self._noise(11, t, sst.shape, 0.08)
            out[i] = np.where(land, np.nan, sst + 273.15)
        var = self.cfg.products["sst"].variables["sst"]
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), out)},
            coords={"time": days + np.timedelta64(12, "h"), "latitude": lat, "longitude": lon},
        )
        ds[var].attrs.update(units="kelvin", standard_name="sea_surface_foundation_temperature")
        return ds, {var: _pack(0.001, 273.15)}

    def _sss(self, days):
        lat, lon = self._axes("sss")
        la, lo = lat[:, None], lon[None, :]
        bad = is_land(la, lo) | (coast_distance(la, lo) < 0.3)
        out = np.empty((len(days), len(lat), len(lon)), np.float32)
        for i, d in enumerate(days):
            t = float(days_since_2000(d)) + 0.5
            s = self.world.sss(la, lo, t) + self._noise(12, t, (len(lat), len(lon)), 0.05)
            out[i] = np.where(bad, np.nan, s)
        var = self.cfg.products["sss"].variables["sss"]
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), out)},
            coords={"time": days, "latitude": lat, "longitude": lon},
        )
        ds[var].attrs.update(units="1e-3", standard_name="sea_surface_salinity")
        return ds, {var: _pack(0.001, 35.0)}

    def _sla(self, days):
        lat, lon = self._axes("sla")
        la, lo = lat[:, None], lon[None, :]
        land = is_land(la, lo)
        out = np.empty((len(days), len(lat), len(lon)), np.float32)
        for i, d in enumerate(days):
            t = float(days_since_2000(d)) + 0.5
            s = self.world.sla(la, lo, t) + self._noise(13, t, (len(lat), len(lon)), 0.01)
            out[i] = np.where(land, np.nan, s)
        var = self.cfg.products["sla"].variables["sla"]
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), out)},
            coords={"time": days, "latitude": lat, "longitude": lon},
        )
        ds[var].attrs.update(units="m", standard_name="sea_surface_height_above_sea_level")
        return ds, {var: _pack(0.0001)}

    def _currents(self, days):
        lat, lon = self._axes("currents")
        lat = lat[::-1].copy()  # OSCAR-like: descending latitude
        la, lo = lat[:, None], lon[None, :]
        land = is_land(la, lo)
        n = len(days)
        u = np.empty((n, len(lat), len(lon)), np.float32)
        v = np.empty_like(u)
        for i, d in enumerate(days):
            t = float(days_since_2000(d)) + 0.5
            uu, vv = self.world.geostrophic(la, lo, t)
            uu = uu + self._noise(14, t, uu.shape, 0.02)
            vv = vv + self._noise(15, t, vv.shape, 0.02)
            u[i], v[i] = np.where(land, np.nan, uu), np.where(land, np.nan, vv)
        names = self.cfg.products["currents"].variables
        dims = ("time", "longitude", "latitude")  # OSCAR order
        ds = xr.Dataset(
            {
                names["uo"]: (dims, np.ascontiguousarray(u.transpose(0, 2, 1))),
                names["vo"]: (dims, np.ascontiguousarray(v.transpose(0, 2, 1))),
            },
            coords={"time": days, "latitude": lat, "longitude": lon},
        )
        for k in names.values():
            ds[k].attrs.update(units="m s-1")
        return ds, {k: dict(_FLOAT_ENC) for k in names.values()}

    def _winds(self, days):
        lat, lon = self._axes("winds")
        la, lo = lat[:, None], lon[None, :]
        land = is_land(la, lo)
        times, us, vs = [], [], []
        for d in days:
            for h in (0, 6, 12, 18):
                t = float(days_since_2000(d)) + h / 24.0
                u, v = self.world.wind(la, lo, t)
                u = u + self._noise(16, t * 4, u.shape, 0.4)
                v = v + self._noise(17, t * 4, v.shape, 0.4)
                us.append(np.where(land, np.nan, u))
                vs.append(np.where(land, np.nan, v))
                times.append(d + np.timedelta64(h, "h"))
        names = self.cfg.products["winds"].variables
        dims = ("time", "latitude", "longitude")
        ds = xr.Dataset(
            {names["uw"]: (dims, np.stack(us)), names["vw"]: (dims, np.stack(vs))},
            coords={"time": np.array(times), "latitude": lat, "longitude": lon},
        )
        for k in names.values():
            ds[k].attrs.update(units="m s-1")
        return ds, {k: dict(_FLOAT_ENC) for k in names.values()}

    def _temp(self, days):
        lat, lon = self._axes("temp")
        la, lo = lat[:, None], lon[None, :]
        out = np.empty((len(days), len(GLORYS_LEVELS), len(lat), len(lon)), np.float32)
        for i, d in enumerate(days):
            t = float(days_since_2000(d)) + 0.5
            out[i] = self.world.temperature(la, lo, t, GLORYS_LEVELS)
        var = self.cfg.products["temp"].variables["temp"]
        ds = xr.Dataset(
            {var: (("time", "depth", "latitude", "longitude"), out)},
            coords={"time": days, "depth": GLORYS_LEVELS, "latitude": lat, "longitude": lon},
        )
        ds[var].attrs.update(units="degrees_C", standard_name="sea_water_potential_temperature")
        ds["depth"].attrs.update(units="m", positive="down")
        enc = _pack(0.001, 17.5)
        enc["chunksizes"] = (1, len(GLORYS_LEVELS), len(lat), len(lon))
        return ds, {var: enc}

    # ------------------------------------------------------------------ argo
    def _argo_month(self, first: date, last: date) -> pd.DataFrame:
        cfg = self.cfg
        rng = np.random.default_rng([cfg.synthetic.seed, 99, first.year * 12 + first.month])
        n_target = cfg.synthetic.argo_profiles_per_month
        g = cfg.grid
        t_lo = float(days_since_2000(np.datetime64(first)))
        t_hi = float(days_since_2000(np.datetime64(last))) + 1.0
        rows = []
        made = 0
        tries = 0
        while made < n_target and tries < n_target * 50:
            tries += 1
            lat = rng.uniform(g.lat_min, g.lat_max)
            lon = rng.uniform(g.lon_min, g.lon_max)
            if is_land(lat, lon) or bathymetry(lat, lon) < 300.0:
                continue
            t = rng.uniform(t_lo, t_hi - 1e-6)
            # irregular pressure levels, dense near the surface, ~10 % random drop-outs
            pres = np.concatenate(
                [np.arange(5, 100, 5.0), np.arange(100, 400, 10.0), np.arange(400, 1100, 25.0)]
            )
            pres = pres + rng.uniform(-1.5, 1.5, pres.shape)
            pres = pres[rng.random(pres.shape) > 0.1]
            depth = pressure_to_depth(pres, lat)
            floor = bathymetry(lat, lon)
            keep = depth <= floor
            pres, depth = pres[keep], depth[keep]
            st = self.world.state(np.float32(lat), np.float32(lon), t)
            truth = SyntheticWorld.profile(st, depth.astype(np.float32))
            obs = truth + rng.normal(0.0, 0.03, truth.shape)
            ts = T0_NS + np.timedelta64(int(round(t * 86400e9)), "ns")
            cycle = (first.year - 2000) * 12 + first.month
            pid = f"{2900000 + made}_{cycle:03d}"
            rows.append(
                pd.DataFrame(
                    {
                        "profile_id": pid,
                        "platform": pid.split("_")[0],
                        "cycle": int(pid.split("_")[1]),
                        "time": ts,
                        "latitude": lat,
                        "longitude": lon,
                        "pres": pres.astype(np.float32),
                        "depth": depth.astype(np.float32),
                        "temp": obs.astype(np.float32),
                        "temp_qc": np.int8(1),
                    }
                )
            )
            made += 1
        if not rows:
            return pd.DataFrame(columns=ARGO_COLUMNS)
        df = pd.concat(rows, ignore_index=True)
        df["time"] = df["time"].astype("datetime64[ns]")
        return df[ARGO_COLUMNS]

    # ------------------------------------------------------------------ public API
    def fetch(self, variable: str, start: date, end: date) -> list[Path]:
        written: list[Path] = []
        builders = {
            "sst": self._sst,
            "sss": self._sss,
            "sla": self._sla,
            "currents": self._currents,
            "winds": self._winds,
            "temp": self._temp,
        }
        for first, last in tqdm(list(months(start, end)), desc=f"synth {variable}", leave=False):
            path = raw_file(self.cfg, variable, first)
            if path.exists() and not self.cfg.download.overwrite:
                continue
            if variable == "argo":
                path.parent.mkdir(parents=True, exist_ok=True)
                self._argo_month(first, last).to_parquet(path, index=False)
            else:
                ds, enc = builders[variable](self._days(first, last))
                ds.attrs.update(title=f"Synthetic {variable} (OceanEmbed test product)")
                self._write(ds, path, enc)
            written.append(path)
        return written


T0_NS = np.datetime64("2000-01-01", "ns")


def generate_all(cfg: Config, products: list[str] | None = None) -> dict[str, list[Path]]:
    """Write every synthetic raw product (and Argo-like profiles) for the configured period."""
    prov = SyntheticProvider(cfg)
    out = {}
    for p in products or [*GRIDDED_PRODUCTS, "argo"]:
        raw_dir(cfg, p).mkdir(parents=True, exist_ok=True)
        out[p] = prov.fetch(p, cfg.time.start, cfg.time.end)
    return out
