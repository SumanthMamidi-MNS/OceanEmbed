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

# Input groups: the unit in which inputs are kept or dropped (a group is one product; currents and
# winds are vectors, so both components go together).
INPUT_GROUPS: dict[str, tuple[str, ...]] = {
    "sst": ("sst",),
    "sss": ("sss",),
    "sla": ("sla",),
    "currents": ("uo", "vo"),
    "winds": ("uw", "vw"),
}


RESERVED_TAGS = ("ridge", "mlp")  # prediction product folders of the baselines


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    data_root: Path = Path("data")
    outputs_root: Path = Path("outputs")
    # Sub-folder of data_root that holds raw downloads. Synthetic configs use a different
    # folder so that fake and real raw files can never be mixed up.
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    # Name of an existing harmonised store (and statistics file, and array cache) to reuse
    # instead of ``<run_name>``: ``processed/<store>.zarr``, ``processed/<store>_stats.nc``. Lets a
    # run share the data of another run without copying it. ``harmonize`` / ``stats`` refuse to
    # rebuild a shared store.
    store: str | None = None


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
    # "transformer": the OceanEmbed encoder. "unet": the Transformer blocks are replaced by residual
    # convolution blocks of the same width (research baseline; see models/encoder.py).
    arch: Literal["transformer", "unet"] = "transformer"
    # Input groups the model (and the ridge / per-pixel MLP baselines) may use. A group that is not
    # listed is zeroed (the training mean, in standardised units) in every input channel of every
    # consumer: training, prediction, evaluation, embedding export. Default: all seven inputs.
    input_groups: list[str] = Field(default_factory=lambda: list(INPUT_GROUPS))
    # How the *main* model (checkpoints/recon.pt) is initialised: "pretrained" = fine-tuned from
    # the masked-surface pretraining (needs the `pretrain` step), "scratch" = random weights.
    main_init: Literal["pretrained", "scratch"] = "pretrained"

    @field_validator("input_groups")
    @classmethod
    def _groups_valid(cls, v: list[str]) -> list[str]:
        bad = [g for g in v if g not in INPUT_GROUPS]
        if bad:
            raise ValueError(f"unknown input group(s) {bad}; choose from {list(INPUT_GROUPS)}")
        if not v:
            raise ValueError("input_groups must keep at least one group")
        return [g for g in INPUT_GROUPS if g in v]  # canonical order, no duplicates

    @property
    def dropped_groups(self) -> list[str]:
        return [g for g in INPUT_GROUPS if g not in self.input_groups]

    @property
    def input_variables(self) -> list[str]:
        """Surface variables the model may use, in canonical order."""
        return [v for g in self.input_groups for v in INPUT_GROUPS[g]]


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
    # Where the train / validation arrays live: "ram" (preloaded, the default) or "memmap" (a
    # one-off on-disk cache read by day, for periods too long for RAM; see data/dataset.py).
    cache: Literal["ram", "memmap"] = "ram"
    cache_dtype: Literal["float16", "float32"] = "float16"  # the memmap cache only


class BaselineConfig(_Base):
    ridge_alpha: float = 1.0
    ridge_max_points: int = 1_000_000  # random subsample of valid train-split ocean points
    seed: int = 0
    # also fit the per-pixel MLP (the ``mlp`` section) in run-all and publish it next to ridge:
    # checkpoints/mlp.pt, predictions/mlp/, scored by evaluate and validate-argo
    mlp: bool = False


class MlpConfig(_Base):
    """Per-pixel MLP baseline (research stage R1): the ridge features, a non-linear model."""

    hidden: int = 256
    layers: int = 3  # hidden layers
    max_points: int = 1_000_000  # random train (day, pixel) samples
    val_points: int = 200_000  # fixed random val sample for early stopping
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5


class AblationConfig(_Base):
    """Optional extra training run(s) that ``run-all`` performs and publishes next to the model."""

    # Train the same network without the pretrained encoder, then predict it (predictions/<tag>/).
    # For a config whose main model is pretrained (model.main_init: pretrained).
    no_pretrained: bool = False
    # The reverse, for a config whose main model is trained from scratch: also pretrain, fine-tune
    # and predict the pretrained variant (checkpoint recon_<tag>.pt, predictions/<tag>/).
    pretrained: bool = False
    tag: str = Field(default="scratch", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,30}$")

    @field_validator("tag")
    @classmethod
    def _tag_not_reserved(cls, v: str) -> str:
        if v in RESERVED_TAGS:
            raise ValueError(f"'{v}' is reserved for the {v} baseline product folder")
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
    mlp: MlpConfig = Field(default_factory=MlpConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)

    @field_validator("products")
    @classmethod
    def _products_complete(cls, v: dict[str, ProductConfig]) -> dict[str, ProductConfig]:
        # a partial override in YAML is merged on top of the defaults
        merged = default_products()
        merged.update(v)
        return merged

    @model_validator(mode="after")
    def _ablation_matches_main(self) -> Config:
        scratch_main = self.model.main_init == "scratch"
        if scratch_main and self.ablation.no_pretrained:
            raise ValueError(
                "ablation.no_pretrained needs a pretrained main model; the main model of this "
                "config is already trained from scratch (use ablation.pretrained instead)"
            )
        if not scratch_main and self.ablation.pretrained:
            raise ValueError(
                "ablation.pretrained needs model.main_init: scratch (the main model is pretrained)"
            )
        return self

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
    def store_name(self) -> str:
        """Name of the harmonised store this run reads: its own run name unless ``paths.store``."""
        return self.paths.store or self.run_name

    @property
    def shares_store(self) -> bool:
        """True when the run reads the harmonised store / statistics of another run."""
        return self.store_name != self.run_name

    @property
    def zarr_path(self) -> Path:
        return self.processed_root / f"{self.store_name}.zarr"

    @property
    def stats_path(self) -> Path:
        return self.processed_root / f"{self.store_name}_stats.nc"

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
