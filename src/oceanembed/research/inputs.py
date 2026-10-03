"""Input manipulation for research stage R3: variable groups, masking, history and permutation.

Channel layout of a model input ``x`` (``C = 12 + 7 (k - 1)`` channels, history of ``k`` days)::

    0..6    the 7 standardised surface fields of the target day (sst sss sla uo vo uw vw)
    7       ocean mask       8, 9   sin / cos day of year       10, 11   normalised lat / lon
    12..18  surface fields of day t-1    19..25  of day t-2   ...   (history; none when k = 1)

An :class:`InputSpec` says which variable *groups* the model may use and how many days of surface
history it gets. A group that is not kept is **zeroed** in every channel of that variable (the
target day and every lag): in standardised units zero is the training mean, the value the dataset
already uses for missing data, so the network sees a constant it can learn to ignore, the layout
and initialisation of the network stay identical to the full-input model, and nothing is left for it
to use. The mask, day-of-year and lat / lon channels are never touched.

Training reads the preloaded train / validation arrays through :class:`InputView` (no copy of the
data: the history of a day is indexed from the same arrays); scoring builds the same inputs for the
test split with :func:`make_test_batch_fn`, taking the days before the split from the store (inputs
only, never a target). The first ``k - 1`` days of a training / validation split have no full
history and are dropped from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.dataset import (
    N_INPUT_CHANNELS,
    N_SURFACE,
    OceanDataset,
    SurfaceOnlyDataset,
    make_surface_dataset,
)
from oceanembed.models.baselines import FEATURE_CHANNELS

GROUPS: dict[str, tuple[str, ...]] = {
    "sst": ("sst",),
    "sss": ("sss",),
    "sla": ("sla",),
    "currents": ("uo", "vo"),
    "winds": ("uw", "vw"),
}
GROUP_LABELS = {
    "sst": "SST",
    "sss": "salinity",
    "sla": "sea level",
    "currents": "currents",
    "winds": "winds",
}
GROUP_SURFACE_INDEX = {g: tuple(SURFACE_VARS.index(v) for v in vs) for g, vs in GROUPS.items()}


@dataclass(frozen=True)
class InputSpec:
    """Variable groups the model may use (``keep``) and the history length in days (``history``)."""

    keep: tuple[str, ...] = tuple(GROUPS)
    history: int = 1

    def __post_init__(self):
        bad = [g for g in self.keep if g not in GROUPS]
        if bad:
            raise ValueError(f"unknown variable group(s) {bad}; choose from {list(GROUPS)}")
        if self.history < 1:
            raise ValueError("history must be >= 1 day")

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(g for g in GROUPS if g not in self.keep)

    @property
    def offset(self) -> int:
        """Days at the start of a split that have no full history."""
        return self.history - 1

    @property
    def n_channels(self) -> int:
        return N_INPUT_CHANNELS + N_SURFACE * self.offset

    @property
    def zero_channels(self) -> list[int]:
        """Channels (current day and every lag) of the dropped groups."""
        out = []
        for lag in range(self.history):
            base = 0 if lag == 0 else N_INPUT_CHANNELS + N_SURFACE * (lag - 1)
            for g in self.dropped:
                out += [base + i for i in GROUP_SURFACE_INDEX[g]]
        return sorted(out)

    @property
    def feature_channels(self) -> list[int]:
        """Per-pixel MLP features: the ridge features, then the surface fields of the lags."""
        return [*FEATURE_CHANNELS, *range(N_INPUT_CHANNELS, self.n_channels)]

    @property
    def is_identity(self) -> bool:
        return not self.dropped and self.history == 1


def _without(*groups: str) -> tuple[str, ...]:
    return tuple(g for g in GROUPS if g not in groups)


EXPERIMENTS: dict[str, InputSpec] = {
    "full": InputSpec(),
    "no_sst": InputSpec(keep=_without("sst")),
    "no_sss": InputSpec(keep=_without("sss")),
    "no_sla": InputSpec(keep=_without("sla")),
    "no_currents": InputSpec(keep=_without("currents")),
    "no_winds": InputSpec(keep=_without("winds")),
    "hist3": InputSpec(history=3),
    "hist7": InputSpec(history=7),
    "sst_only": InputSpec(keep=("sst",)),
    "sst_sla": InputSpec(keep=("sst", "sla")),
    "sst_sla_winds": InputSpec(keep=("sst", "sla", "winds")),
}
EXPERIMENT_LABELS = {
    "full": "all inputs (k = 1)",
    "no_sst": "without SST",
    "no_sss": "without salinity",
    "no_sla": "without sea level",
    "no_currents": "without currents",
    "no_winds": "without winds",
    "hist3": "history k = 3",
    "hist7": "history k = 7",
    "sst_only": "SST only",
    "sst_sla": "SST + sea level",
    "sst_sla_winds": "all but salinity and currents",
}
# most informative first: single removals, then history, then the reduced sets
EXPERIMENT_ORDER = (
    "no_sst",
    "no_sss",
    "no_sla",
    "no_currents",
    "no_winds",
    "hist3",
    "hist7",
    "sst_only",
    "sst_sla",
    "sst_sla_winds",
)
REMOVAL_EXPERIMENTS = ("no_sst", "no_sss", "no_sla", "no_currents", "no_winds")
REDUCED_EXPERIMENTS = ("sst_only", "sst_sla", "sst_sla_winds")
HISTORY_EXPERIMENTS = {"hist3": 3, "hist7": 7}


def zero_channels(x, channels: list[int]):
    """``x`` with the given channels set to 0 (a copy only when there is something to zero)."""
    if not channels:
        return x
    x = x.clone() if isinstance(x, torch.Tensor) else np.array(x, copy=True)
    x[..., channels, :, :] = 0
    return x


# ----------------------------------------------------------------------------------------
# training: a view of a preloaded dataset
# ----------------------------------------------------------------------------------------
class InputView(Dataset):
    """Read-only view of a preloaded :class:`OceanDataset` with an :class:`InputSpec` applied.

    Item ``i`` of the view is day ``i + (history - 1)`` of the base dataset (the first days have
    no full history), with the dropped groups zeroed and the surface fields of the previous days
    appended as extra channels. The base arrays are shared, never copied or modified."""

    def __init__(self, base: OceanDataset, spec: InputSpec):
        if spec.history > 1 and not base.preloaded:
            raise ValueError("a history view needs a preloaded dataset")
        if len(base) <= spec.offset:
            raise ValueError("the split is not longer than the history")
        self.base, self.spec = base, spec
        self.offset = spec.offset
        self.stats = base.stats
        self.shape, self.mask, self.depth = base.shape, base.mask, base.depth
        self._zero = spec.zero_channels

    def __len__(self) -> int:
        return len(self.base) - self.offset

    def dates(self) -> pd.DatetimeIndex:
        return self.base.dates()[self.offset :]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0:
            i += len(self)
        item = self.base[i + self.offset]
        if self.spec.is_identity:
            return item
        x = item["x"]
        if self.offset:
            surf = self.base.arrays()["surf"]
            lags = [
                torch.from_numpy(surf[i + self.offset - j]) for j in range(1, self.spec.history)
            ]
            x = torch.cat([x, *lags])
        item["x"] = zero_channels(x, self._zero)
        item["index"] = torch.tensor(i)
        return item


def make_views(train: OceanDataset, val: OceanDataset, spec: InputSpec) -> tuple:
    return InputView(train, spec), InputView(val, spec)


# ----------------------------------------------------------------------------------------
# scoring: the same inputs for the test split
# ----------------------------------------------------------------------------------------
BatchFn = Callable[[int, int], dict]


def make_test_batch_fn(cfg: Config, ds: SurfaceOnlyDataset, spec: InputSpec) -> BatchFn:
    """``fn(a, b)`` -> batch of test days ``a .. b-1`` whose ``x`` follows ``spec``.

    Every test day is scored for every history length: the days before the split come from the
    store (their *inputs* only), so all specs are scored on the identical days."""
    zero = spec.zero_channels
    k = spec.history
    if k == 1:

        def plain(a: int, b: int) -> dict:
            batch = ds.batch(a, b)
            batch["x"] = zero_channels(batch["x"], zero)
            return batch

        return plain
    dates = ds.dates()
    ext = make_surface_dataset(cfg, dates[0] - pd.Timedelta(days=k - 1), dates[-1])
    if len(ext) != len(ds) + k - 1 or ext.dates()[k - 1] != dates[0]:
        raise ValueError(
            f"the store has no {k - 1} day(s) before {dates[0].date()}: the history of the first "
            "test days cannot be built"
        )

    def with_history(a: int, b: int) -> dict:
        eb = ext.batch(a, b + k - 1)  # extended items a .. b + k - 2 = days a - (k-1) .. b - 1
        n = b - a
        x = eb["x"]
        cur = x[k - 1 :]
        lags = [x[k - 1 - j : k - 1 - j + n, :N_SURFACE] for j in range(1, k)]
        batch = {key: v[k - 1 :] for key, v in eb.items()}
        batch["x"] = zero_channels(torch.cat([cur, *lags], dim=1), zero)
        return batch

    return with_history


def sample_points_view(view: InputView, max_points: int, seed: int):
    """:func:`oceanembed.models.baselines.sample_points` for a view: random (day, ocean pixel)
    samples with the MLP features of ``spec.feature_channels`` (dropped groups zeroed, the surface
    fields of the previous days appended). For the identity spec the draws and the result are
    exactly those of ``sample_points``. A sample counts when the 7 surface values of its own day
    were all observed, as for the full-input model, so every ablation trains on the same points."""
    base, spec, off = view.base, view.spec, view.offset
    n_days = len(view)
    ocean = np.flatnonzero(base.mask[0].reshape(-1))
    rng = np.random.default_rng(seed)
    n = min(max_points, n_days * len(ocean))
    t = rng.integers(0, n_days, n)
    p = ocean[rng.integers(0, len(ocean), n)]
    tb = t + off
    surf, sv, y, valid = base.gather(tb, p)  # (n, 7), (n, 7), (n, D), (n, D)
    ok = sv.all(axis=1)
    lat = base._lat_plane.reshape(-1)[p]
    lon = base._lon_plane.reshape(-1)[p]
    cols = [surf, base._sin[tb][:, None], base._cos[tb][:, None], lat[:, None], lon[:, None]]
    cols += [base.gather(tb - j, p, ("surf",))[0] for j in range(1, spec.history)]
    feats = np.concatenate(cols, axis=1).astype(np.float32)
    # feature columns in the order of spec.feature_channels; zero the dropped groups
    channel_of_col = spec.feature_channels
    drop = set(spec.zero_channels)
    for col, ch in enumerate(channel_of_col):
        if ch in drop:
            feats[:, col] = 0.0
    return feats[ok], y[ok], valid[ok]


# ----------------------------------------------------------------------------------------
# permutation within the calendar month
# ----------------------------------------------------------------------------------------
def month_permutation(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """``donor[i]``: a day of the same calendar month (same year) other than ``i`` -- a random
    derangement of each month's days (a month with a single day maps to itself).

    Permuting within a month keeps every permuted input realistic for its season (the seasonal
    cycle, which the day-of-year channels carry anyway, is not destroyed) and breaks only the
    day-to-day co-variability with the other inputs and with the subsurface, so the importance
    measures information beyond the seasonal mean state."""
    ym = np.asarray(dates.year * 100 + dates.month)
    donor = np.arange(len(dates))
    for key in np.unique(ym):
        idx = np.flatnonzero(ym == key)
        m = len(idx)
        if m < 2:
            continue
        for _ in range(1000):
            perm = rng.permutation(m)
            if not (perm == np.arange(m)).any():
                break
        else:  # pragma: no cover - probability ~ (1 - 1/e)^1000
            perm = np.roll(np.arange(m), 1)
        donor[idx] = idx[perm]
    return donor


def permute_batch(
    x_all: torch.Tensor, donor: np.ndarray, channels: list[int], a: int, b: int
) -> torch.Tensor:
    """Days ``a .. b-1`` of ``x_all`` with ``channels`` taken from the donor days (all channels of
    a group from the *same* donor day, so e.g. u and v stay a consistent vector)."""
    x = x_all[a:b].clone()
    x[:, channels] = x_all[torch.from_numpy(donor[a:b])][:, channels]
    return x


def load_test_inputs(ds: SurfaceOnlyDataset, batch_size: int = 32) -> torch.Tensor:
    """All inputs ``(n, 12, H, W)`` of the test split in RAM (about 0.4 GB for the real run)."""
    h, w = ds.shape
    out = torch.empty(len(ds), N_INPUT_CHANNELS, h, w)
    for a in range(0, len(ds), batch_size):
        b = min(a + batch_size, len(ds))
        out[a:b] = ds.batch(a, b)["x"]
    return out
