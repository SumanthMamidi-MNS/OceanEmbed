"""Pydantic configuration models and the YAML loader.

Everything the pipeline needs to know about a run lives in one YAML file. Later phases
(model, pretraining, training) only *read* the ``model`` / ``pretrain`` / ``train`` sections.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATA_ROOT_ENV = "OCEANEMBED_DATA_ROOT"
OUTPUTS_ROOT_ENV = "OCEANEMBED_OUTPUTS_ROOT"

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Canonical surface channels, in model-input order.
SURFACE_VARS = ["sst", "sss", "sla", "uo", "vo", "uw", "vw"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    data_root: Path = Path("data")
    outputs_root: Path = Path("outputs")
    # Sub-folder of data_root that holds raw downloads. Synthetic configs use a different
    # folder so that fake and real raw files can never be mixed up.
    raw_dir: str = "raw"
    processed_dir: str = "processed"


class GridConfig(_Base):
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25
    depths: list[float] = Field(default_factory=lambda: list(STANDARD_DEPTHS))

    @model_validator(mode="after")
    def _check(self) -> GridConfig:
        if self.lat_max <= self.lat_min or self.lon_max <= self.lon_min:
            raise ValueError("grid bounds must satisfy min < max")
        ny = round((self.lat_max - self.lat_min) / self.resolution)
        nx = round((self.lon_max - self.lon_min) / self.resolution)
        if ny % 4 or nx % 4:
            raise ValueError(f"grid is {ny}x{nx}; H and W must be divisible by 4")
        if sorted(self.depths) != list(self.depths):
            raise ValueError("depths must be ascending")
        return self


def _as_date(v: date | str) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


class DateRange(_Base):
    start: date
    end: date

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, v):
        if isinstance(v, list | tuple):
            if len(v) != 2:
                raise ValueError("a date range is [start, end]")
            return {"start": v[0], "end": v[1]}
        return v

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("range end is before start")
        return self


class SplitConfig(_Base):
    train: DateRange
    val: DateRange
    test: DateRange

    def get(self, name: str) -> DateRange:
        return getattr(self, name)


class DatasetSource(_Base):
    """One remote dataset (CMEMS dataset id or Earthdata short name) and the dates it serves.

    Several sources per product are allowed (e.g. a reprocessed and a near-real-time
    dataset); each month is fetched from the first source whose range covers it.
    """

    id: str
    start: date | None = None
    end: date | None = None


class ProductConfig(_Base):
    provider: Literal["cmems", "podaac"]
    datasets: list[DatasetSource]
    # canonical variable name -> variable name inside the raw files
    variables: dict[str, str]
    # optional overrides of the coordinate names used in the raw files
    coords: dict[str, str] = Field(default_factory=dict)
    # CMEMS vertical subset (m); only used by the temperature product
    min_depth: float | None = None
    max_depth: float | None = None


def default_products() -> dict[str, ProductConfig]:
    """Product definitions. Dataset ids were verified against the live catalogues (see
    configs/poc.yaml for the verification notes)."""
    return {
        "sst": ProductConfig(
            provider="cmems",
            datasets=[DatasetSource(id="METOFFICE-GLO-SST-L4-REP-OBS-SST")],
            variables={"sst": "analysed_sst"},
        ),
        "sss": ProductConfig(
            provider="cmems",
            datasets=[
                DatasetSource(id="cmems_obs-mob_glo_phy-sss_my_multi_P1D", end=date(2023, 12, 31)),
                DatasetSource(id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D", start=date(2024, 1, 1)),
            ],
            variables={"sss": "sos"},
        ),
        "sla": ProductConfig(
            provider="cmems",
            datasets=[DatasetSource(id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D")],
            variables={"sla": "sla"},
        ),
        "currents": ProductConfig(
            provider="podaac",
            datasets=[DatasetSource(id="OSCAR_L4_OC_FINAL_V2.0")],
            variables={"uo": "u", "vo": "v"},
        ),
        "winds": ProductConfig(
            provider="podaac",
            datasets=[DatasetSource(id="CCMP_WINDS_10M6HR_L4_V3.1")],
            variables={"uw": "uwnd", "vw": "vwnd"},
        ),
        "temp": ProductConfig(
            provider="cmems",
            datasets=[DatasetSource(id="cmems_mod_glo_phy_my_0.083deg_P1D-m")],
            variables={"temp": "thetao"},
            min_depth=0.0,
            max_depth=1100.0,
        ),
    }


class SyntheticConfig(_Base):
    seed: int = 7
    argo_profiles_per_month: int = 150
    # native grid spacing (deg) of each synthetic raw product
    native_resolution: dict[str, float] = Field(
        default_factory=lambda: {
            "sst": 0.05,
            "sss": 0.125,
            "sla": 0.25,
            "currents": 0.25,
            "winds": 0.25,
            "temp": 1.0 / 12.0,
        }
    )
    n_eddies_per_100_deg2_year: float = 25.0


class DownloadConfig(_Base):
    halo_deg: float = 0.5
    overwrite: bool = False
    delete_global_granules: bool = True
    # PO.DAAC route: "opendap" = per-granule server-side subsetting (default, ~0.2-0.8 MB per day);
    # "granule" = download whole global daily granules (~32 MB) and crop locally.
    # OPeNDAP falls back to the granule route automatically when the server refuses / keeps failing.
    podaac_mode: Literal["opendap", "granule"] = "opendap"
    workers: int = Field(default=4, ge=1, le=8)  # concurrent PO.DAAC requests (polite default)
    retries: int = Field(default=4, ge=0)  # extra attempts per piece on transient errors
    retry_backoff_s: float = Field(default=2.0, ge=0.0)  # first back-off; doubles each retry


class ArgoConfig(_Base):
    source: Literal["erddap", "gdac"] = "erddap"
    max_depth_m: float = 1100.0
    qc_flags: list[int] = Field(default_factory=lambda: [1, 2])
    # fetch this many degrees of longitude / latitude per request (argopy region box)
    box_deg: float = 15.0


class HarmonizeConfig(_Base):
    time_chunk: int = 8  # days processed together
    min_valid_fraction: float = 0.5  # block-mean validity rule


# ----- model / training sections (defaults from docs/architecture.md) -----


class ModelConfig(_Base):
    in_channels: int = 12  # dataset channels; the encoder adds one internal "visible" channel
    emb_dim: int = 128
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    stem_channels: int = 64
    n_depths: int = 15


class PretrainConfig(_Base):
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    mask_ratio: float = 0.5
    block: int = 16
    amp: bool = True
    seed: int = 0
    # random spatial crop (height, width in px, multiples of 4) of every training batch; None = off
    crop: tuple[int, int] | None = None


class TrainConfig(_Base):
    epochs: int = 30
    batch_size: int = 8
    lr: float = 3e-4
    encoder_lr_scale: float = 0.1  # encoder LR = lr * scale when fine-tuning a pretrained encoder
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    vertical_grad_weight: float = 0.05
    patience: int = 8  # early stopping on val RMSE (degC), in epochs
    amp: bool = True
    seed: int = 0
    crop: tuple[int, int] | None = None  # as pretrain.crop
    num_workers: int = 0


class BaselineConfig(_Base):
    ridge_alpha: float = 1.0
    ridge_max_points: int = 1_000_000  # random subsample of valid train-split ocean points
    seed: int = 0


class AblationConfig(_Base):
    """Optional extra training run(s) that ``run-all`` performs and publishes next to the model."""

    # Train the same network without the pretrained encoder, then predict it (predictions/<tag>/)
    no_pretrained: bool = False
    tag: str = Field(default="scratch", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,30}$")

    @field_validator("tag")
    @classmethod
    def _tag_not_reserved(cls, v: str) -> str:
        if v == "ridge":
            raise ValueError("'ridge' is reserved for the ridge baseline product folder")
        return v


class Config(_Base):
    run_name: str
    # Human display name / one-paragraph description for the dashboard; derived when left out.
    label: str | None = None
    description: str | None = None
    paths: PathsConfig = Field(default_factory=PathsConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    time: DateRange
    split: SplitConfig
    provider: Literal["synthetic", "real"] = "synthetic"
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    products: dict[str, ProductConfig] = Field(default_factory=default_products)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    argo: ArgoConfig = Field(default_factory=ArgoConfig)
    harmonize: HarmonizeConfig = Field(default_factory=HarmonizeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)

    @field_validator("products")
    @classmethod
    def _products_complete(cls, v: dict[str, ProductConfig]) -> dict[str, ProductConfig]:
        # a partial override in YAML is merged on top of the defaults
        merged = default_products()
        merged.update(v)
        return merged

    @model_validator(mode="after")
    def _splits_inside_time(self) -> Config:
        for name in ("train", "val", "test"):
            r = self.split.get(name)
            if r.start < self.time.start or r.end > self.time.end:
                raise ValueError(f"split.{name} lies outside the configured time range")
        if not (self.split.train.end < self.split.val.start <= self.split.val.end):
            raise ValueError("train must end before val starts")
        if not (self.split.val.end < self.split.test.start):
            raise ValueError("val must end before test starts")
        return self

    # ----- derived paths -----
    @property
    def data_root(self) -> Path:
        env = os.environ.get(DATA_ROOT_ENV)
        return Path(env) if env else self.paths.data_root

    @property
    def raw_root(self) -> Path:
        return self.data_root / self.paths.raw_dir

    @property
    def processed_root(self) -> Path:
        return self.data_root / self.paths.processed_dir

    @property
    def zarr_path(self) -> Path:
        return self.processed_root / f"{self.run_name}.zarr"

    @property
    def stats_path(self) -> Path:
        return self.processed_root / f"{self.run_name}_stats.nc"

    @property
    def outputs_root(self) -> Path:
        env = os.environ.get(OUTPUTS_ROOT_ENV)
        return Path(env) if env else self.paths.outputs_root

    @property
    def outputs_dir(self) -> Path:
        return self.outputs_root / self.run_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return Config.model_validate(raw)
