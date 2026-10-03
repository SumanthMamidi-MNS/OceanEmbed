"""A small, deterministic, physically-structured "ocean" used to fabricate raw products.

Everything is an analytic function of ``(lat, lon, time)`` so every product can be sampled on
its own native grid (and Argo-like profiles at arbitrary points) from one consistent truth.

Physics (deliberately simple but *learnable*):

* land: an India-like peninsula + Sri Lanka, Arabian peninsula, northern and eastern land;
  bathymetry grows with distance from the coast, so deep levels are masked near shore;
* SLA: propagating Gaussian mesoscale eddies + basin-scale Rossby-like waves + seasonal term;
* thermocline depth ~ base + mixed-layer depth + 250 m per metre of SLA (positive SLA ->
  deeper thermocline -> warmer subsurface); a small unpredictable "hidden" displacement and
  deep-temperature perturbation keep skill below 100 %;
* SSS: saltier Arabian Sea, fresher Bay of Bengal (monsoon river plume); freshness thins the
  mixed layer and sharpens the thermocline (barrier layer);
* winds: monsoon seasonal cycle + synoptic modes; wind speed deepens the mixed layer;
* currents: geostrophic from SLA (plus a little wind drift).

Time ``t`` is expressed in days since 2000-01-01.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

T0 = np.datetime64("2000-01-01", "ns")
YEAR = 365.2425

# fixed reference window for coastline distance (independent of the run's grid)
_REF_LAT = (0.0, 36.0)
_REF_LON = (38.0, 112.0)
_REF_RES = 0.05


def days_since_2000(t) -> np.ndarray:
    t = np.asarray(t).astype("datetime64[ns]")
    return (t - T0) / np.timedelta64(1, "D")


def pressure_to_depth(p, lat):
    """UNESCO/Saunders pressure (dbar) -> depth (m) at a given latitude."""
    p = np.asarray(p, dtype=np.float64)
    x = np.sin(np.deg2rad(np.asarray(lat, dtype=np.float64))) ** 2
    g = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    num = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return num / g


def is_land(lat, lon):
    """Analytic coastline. ``lat``/``lon`` broadcast against each other."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    # northern land: a dent at the Bay of Bengal head, Gulf of Khambhat/Kutch kept as sea
    north = 25.0 - 3.6 * np.exp(-(((lon - 89.0) / 5.0) ** 2))
    land = (lat > north) & (lon > 57.0)
    # Arabian peninsula (west) running NE to Oman
    land |= (lon < 59.5) & (lat > 12.5 + 0.55 * (lon - 45.0))
    land |= (lon < 57.0) & (lat > 24.0)
    # Indian peninsula
    west = np.interp(lat, [7.5, 10.0, 15.0, 20.0, 22.5, 25.0], [77.3, 76.3, 74.0, 72.8, 70.5, 67.0])
    east = np.interp(lat, [7.5, 10.0, 15.0, 18.0, 20.0, 22.0], [77.8, 79.8, 80.3, 83.0, 86.5, 89.0])
    land |= (lat >= 7.5) & (lat <= 23.0) & (lon > west) & (lon < east)
    # Sri Lanka
    land |= (((lon - 80.7) / 0.9) ** 2 + ((lat - 7.6) / 1.4) ** 2) < 1.0
    # eastern land: Myanmar / Thailand / Malay peninsula
    east_coast = np.interp(
        lat, [5.0, 10.0, 15.0, 20.0, 22.0, 30.0], [98.5, 98.2, 94.5, 93.5, 92.5, 92.0]
    )
    land |= lon > east_coast
    return land


class _RefField:
    """Distance-to-coast (degrees) and bathymetry on a fixed 0.05 deg reference grid."""

    def __init__(self):
        lat = _REF_LAT[0] + _REF_RES * (
            np.arange(round((_REF_LAT[1] - _REF_LAT[0]) / _REF_RES)) + 0.5
        )
        lon = _REF_LON[0] + _REF_RES * (
            np.arange(round((_REF_LON[1] - _REF_LON[0]) / _REF_RES)) + 0.5
        )
        land = is_land(lat[:, None], lon[None, :])
        dist = ndimage.distance_transform_edt(~land, sampling=_REF_RES)
        dist = np.where(land, 0.0, dist)
        self.dist = dist.astype(np.float32)
        ridge = 1.0 + 0.25 * np.sin(0.7 * lon[None, :] + 1.3) * np.cos(0.9 * lat[:, None])
        bathy = (10.0 + 4600.0 * (1.0 - np.exp(-((dist / 2.5) ** 1.5)))) * ridge
        self.bathy = np.where(land, 0.0, bathy).astype(np.float32)

    def _sample(self, field, lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        iy = (lat - _REF_LAT[0]) / _REF_RES - 0.5
        ix = (lon - _REF_LON[0]) / _REF_RES - 0.5
        out = ndimage.map_coordinates(field, [iy.ravel(), ix.ravel()], order=1, mode="nearest")
        return out.reshape(lat.shape)

    def distance(self, lat, lon):
        return self._sample(self.dist, lat, lon)

    def depth(self, lat, lon):
        return self._sample(self.bathy, lat, lon)


_REF: _RefField | None = None


def _ref() -> _RefField:
    global _REF
    if _REF is None:
        _REF = _RefField()
    return _REF


def coast_distance(lat, lon):
    return _ref().distance(lat, lon)


def bathymetry(lat, lon):
    """Sea-floor depth (m); 0 on land."""
    return np.where(is_land(lat, lon), 0.0, _ref().depth(lat, lon))


def _gauss(x):
    return np.exp(-(x**2))


class SyntheticWorld:
    def __init__(
        self,
        seed: int,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        t_range: tuple[float, float],
        eddies_per_100deg2_year: float = 25.0,
    ):
        self.seed = seed
        rng = np.random.default_rng([seed, 1])
        # eddies are seeded a few degrees beyond the domain so they can drift in
        m = 4.0
        la0, la1 = lat_range[0] - m, lat_range[1] + m
        lo0, lo1 = lon_range[0] - m, lon_range[1] + m + 6.0
        span = (t_range[1] - t_range[0]) + 160.0
        n = max(8, int(eddies_per_100deg2_year * (la1 - la0) * (lo1 - lo0) / 100.0 * span / YEAR))
        self.ed_tc = rng.uniform(t_range[0] - 80.0, t_range[1] + 80.0, n)
        self.ed_x = rng.uniform(lo0, lo1, n)
        self.ed_y = rng.uniform(la0, la1, n)
        self.ed_a = rng.choice([-1.0, 1.0], n) * rng.uniform(0.06, 0.24, n)
        self.ed_s = rng.uniform(0.8, 1.8, n)
        self.ed_tau = rng.uniform(25.0, 55.0, n)
        self.ed_cx = rng.uniform(-0.06, -0.015, n)
        self.ed_cy = rng.uniform(-0.01, 0.01, n)
        self.modes = {
            "sla": self._modes(
                rng,
                6,
                amp=0.035,
                kmin=2 * np.pi / 28,
                kmax=2 * np.pi / 8,
                pmin=120,
                pmax=365,
                westward=True,
            ),
            "hid": self._modes(
                rng,
                7,
                amp=1.0 / np.sqrt(7 / 2),
                kmin=2 * np.pi / 14,
                kmax=2 * np.pi / 4,
                pmin=40,
                pmax=200,
                westward=True,
            ),
            "hid2": self._modes(
                rng,
                5,
                amp=1.0 / np.sqrt(5 / 2),
                kmin=2 * np.pi / 20,
                kmax=2 * np.pi / 6,
                pmin=60,
                pmax=300,
                westward=True,
            ),
            "syn": self._modes(
                rng,
                6,
                amp=1.6,
                kmin=2 * np.pi / 22,
                kmax=2 * np.pi / 8,
                pmin=4,
                pmax=14,
                westward=False,
            ),
            "sssn": self._modes(
                rng,
                5,
                amp=0.13,
                kmin=2 * np.pi / 25,
                kmax=2 * np.pi / 6,
                pmin=50,
                pmax=250,
                westward=True,
            ),
            "fine": self._modes(
                rng,
                5,
                amp=0.14,
                kmin=2 * np.pi / 1.6,
                kmax=2 * np.pi / 0.5,
                pmin=3,
                pmax=20,
                westward=False,
            ),
        }

    # ---------------------------------------------------------------- building blocks
    @staticmethod
    def _modes(rng, n, amp, kmin, kmax, pmin, pmax, westward):
        k = rng.uniform(kmin, kmax, n)
        ang = rng.uniform(0, 2 * np.pi, n)
        kx, ky = k * np.cos(ang), k * np.sin(ang)
        if westward:  # phase travels west: omega such that cos(kx x + ky y + w t)
            w = 2 * np.pi / rng.uniform(pmin, pmax, n) * np.sign(kx + 1e-9)
        else:
            w = 2 * np.pi / rng.uniform(pmin, pmax, n) * rng.choice([-1, 1], n)
        return kx, ky, w, rng.uniform(0, 2 * np.pi, n), amp

    def _sum_modes(self, name, lat, lon, t):
        kx, ky, w, ph, amp = self.modes[name]
        out = 0.0
        for i in range(len(kx)):
            out = out + np.cos(kx[i] * lon + ky[i] * lat + w[i] * t + ph[i])
        return amp * out / np.sqrt(len(kx) / 2.0) if name not in ("hid", "hid2") else amp * out

    @staticmethod
    def _season(t, peak_doy):
        return np.cos(2 * np.pi * (t - peak_doy) / YEAR)

    # ---------------------------------------------------------------- surface fields
    def sla(self, lat, lon, t):
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        out = 0.0
        dt = t - self.ed_tc
        active = np.nonzero(np.abs(dt) < 3.3 * self.ed_tau)[0]
        for i in active:
            env = self.ed_a[i] * np.exp(-0.5 * (dt[i] / self.ed_tau[i]) ** 2)
            xc = self.ed_x[i] + self.ed_cx[i] * dt[i]
            yc = self.ed_y[i] + self.ed_cy[i] * dt[i]
            s = self.ed_s[i]
            out = out + np.float32(env) * (
                np.exp(-0.5 * ((lat - yc) / s) ** 2) * np.exp(-0.5 * ((lon - xc) / s) ** 2)
            ).astype(np.float32)
        seas = 0.06 * self._season(t, 250.0) * np.sin(np.deg2rad(lat * 4.0))
        return (out + seas + self._sum_modes("sla", lat, lon, t)).astype(np.float32)

    def wind(self, lat, lon, t):
        """Wind components (m/s) at continuous time ``t`` (days)."""
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        cs = self._season(t, 195.0)
        gl = 0.4 + 0.6 * _gauss((lat - 12.0) / 9.0)
        u = 2.0 + 6.5 * cs * gl + self._sum_modes("syn", lat, lon, t) * 0.8
        v = (
            1.0
            + 4.0 * cs * (0.3 + 0.7 * _gauss((lat - 15.0) / 10.0))
            + self._sum_modes("syn", lon, lat, t + 3.0) * 0.8
        )
        hour = (t % 1.0) * 24.0
        diurnal = 0.6 * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
        return (u + diurnal).astype(np.float32), (v + 0.5 * diurnal).astype(np.float32)

    def sss(self, lat, lon, t, sla=None):
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        if sla is None:
            sla = self.sla(lat, lon, t)
        salty = 1.3 * _gauss((lat - 22.0) / 6.0) * _gauss((lon - 62.0) / 10.0)
        bob = _gauss((lon - 90.0) / 9.0) * _gauss((lat - 17.0) / 7.0)
        monsoon = 0.65 + 0.35 * self._season(t, 270.0)
        plume = 1.8 * _gauss((lat - 21.0) / 2.5) * _gauss((lon - 89.0) / 4.0)
        s = 35.0 + salty - (3.0 * bob * monsoon + plume * monsoon) + 2.0 * sla
        s = s + self._sum_modes("sssn", lat, lon, t)
        return np.clip(s, 31.0, 37.5).astype(np.float32)

    # ---------------------------------------------------------------- ocean interior
    def state(self, lat, lon, t):
        """Everything the profile model needs at ``(lat, lon, t)`` (daily-mean wind used)."""
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        sla = self.sla(lat, lon, t)
        sss = self.sss(lat, lon, t, sla)
        u, v = self.wind(lat, lon, np.floor(t) + 0.375)
        ws = np.sqrt(u**2 + v**2)
        latn = np.clip((lat - 5.0) / 25.0, 0.0, 1.0)
        base = 29.3 - 0.22 * np.maximum(lat - 5.0, 0.0)
        amp = 0.8 + 0.12 * np.maximum(lat - 5.0, 0.0)
        semi = 0.5 * np.cos(4 * np.pi * (t - 110.0) / YEAR) * _gauss((lat - 12.0) / 8.0)
        upw = (
            np.maximum(0.0, self._season(t, 200.0))
            * _gauss((lon - 58.0) / 6.0)
            * _gauss((lat - 14.0) / 8.0)
        )
        tm = (
            base
            + amp * self._season(t, 140.0)
            + semi
            - 2.8 * upw
            - 0.1 * np.maximum(ws - 6.0, 0.0)
            + 4.0 * sla
        )
        fresh = np.clip((35.2 - sss) / 3.0, 0.0, 1.0)
        mld = (30.0 + 25.0 * self._season(t, 20.0) * latn + 4.0 * np.maximum(ws - 5.0, 0.0)) * (
            1.0 - 0.35 * fresh
        )
        mld = np.clip(mld, 8.0, 90.0)
        w = 28.0 - 10.0 * fresh
        hid = self._sum_modes("hid", lat, lon, t)
        hid2 = self._sum_modes("hid2", lat, lon, t)
        dtc = mld + 2.4 * w + (25.0 - np.maximum(lat - 5.0, 0.0)) + 250.0 * sla + 12.0 * hid
        dtc = np.clip(dtc, 30.0, 420.0)
        return {
            "sla": sla, "sss": sss, "u10": u, "v10": v, "ws": ws, "tm": tm.astype(np.float32),
            "mld": mld.astype(np.float32), "w": w.astype(np.float32),
            "dtc": dtc.astype(np.float32), "hid2": hid2.astype(np.float32),
            "heave": (250.0 * sla + 12.0 * hid).astype(np.float32),
        }  # fmt: skip

    def state_grid(self, lat1d, lon1d, t, coarse=0.25):
        """``state`` on a regular grid, evaluated on a coarse (0.25 deg) grid and bilinearly
        interpolated: the interior fields are smooth at that scale and this is ~10x cheaper
        than evaluating every 1/12 or 1/20 deg point. Returns arrays of shape (ny, nx)."""
        from oceanembed.data.regrid import bilinear_weights

        def axis(v):
            lo, hi = float(v.min()) - coarse, float(v.max()) + coarse
            k0, k1 = int(np.floor(lo / coarse)), int(np.ceil(hi / coarse))
            return coarse * np.arange(k0, k1 + 1)

        ca, co = axis(np.asarray(lat1d)), axis(np.asarray(lon1d))
        st = self.state(ca[:, None], co[None, :], t)
        wy = bilinear_weights(ca, np.asarray(lat1d, float))
        wx = bilinear_weights(co, np.asarray(lon1d, float))
        return {
            k: (wy @ np.asarray(v, np.float64) @ wx.T).astype(np.float32) for k, v in st.items()
        }

    @staticmethod
    def profile(st, z):
        """Temperature (degC) at depth ``z`` (m); ``z`` broadcasts against the state arrays."""
        z = np.asarray(z, np.float32)
        # deep water is heaved up/down with the thermocline (decaying with depth)
        zz = np.maximum(z - st["heave"] * np.exp(-z / 700.0), 0.0)
        td = 4.3 + 8.5 * np.exp(-zz / 420.0) + 3.0 * np.exp(-zz / 1500.0)
        s = 0.5 * (1.0 - np.tanh((z - st["dtc"]) / st["w"]))
        t = td + (st["tm"] - td) * s
        # unpredictable (not visible at the surface) subsurface perturbations
        pert = 0.3 * np.exp(-(((z - 250.0) / 200.0) ** 2)) + 0.1 * (
            1 - np.exp(-z / 150.0)
        ) * np.exp(-z / 1500.0)
        t = t + pert * st["hid2"]
        return t.astype(np.float32)

    def temperature(self, lat, lon, t, depths):
        """``lat`` (ny,1), ``lon`` (1,nx) regular axes, scalar ``t`` -> (nz, ny, nx), NaN below
        the sea floor and on land."""
        st = self.state_grid(np.ravel(lat), np.ravel(lon), t)
        z = np.asarray(depths, np.float32).reshape(-1, 1, 1)
        out = self.profile({k: np.asarray(v)[None] for k, v in st.items()}, z)
        floor = bathymetry(lat, lon)[None]
        out = np.where(z <= floor, out, np.nan)
        return out.astype(np.float32)

    def sst_surface(self, lat, lon, t):
        """True mixed-layer / surface temperature (profile at z = 0)."""
        st = self.state_grid(np.ravel(lat), np.ravel(lon), t)
        return self.profile(st, 0.0)

    def fine_sst(self, lat, lon, t):
        """Sub-0.25 deg structure only the high-resolution SST product sees."""
        return self._sum_modes("fine", np.asarray(lat, np.float32), np.asarray(lon, np.float32), t)

    # ---------------------------------------------------------------- currents
    def geostrophic(self, lat, lon, t, step=0.125):
        """Geostrophic (u, v) in m/s from SLA central differences, plus a little wind drift."""
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        g, omega = 9.81, 7.292e-5
        dy = 2 * step * 111_200.0
        dx = dy * np.cos(np.deg2rad(lat))
        deta_dy = (self.sla(lat + step, lon, t) - self.sla(lat - step, lon, t)) / dy
        deta_dx = (self.sla(lat, lon + step, t) - self.sla(lat, lon - step, t)) / dx
        f = 2 * omega * np.sin(np.deg2rad(lat))
        f_min = 2 * omega * np.sin(np.deg2rad(6.0))
        f = np.where(np.abs(f) < f_min, f_min, f)
        u = -g / f * deta_dy
        v = g / f * deta_dx
        uw, vw = self.wind(lat, lon, np.floor(t) + 0.375)
        u = np.clip(u + 0.015 * uw, -1.5, 1.5)
        v = np.clip(v + 0.015 * vw, -1.5, 1.5)
        return u.astype(np.float32), v.astype(np.float32)
