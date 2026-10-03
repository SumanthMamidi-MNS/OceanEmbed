"""Pydantic response models (they drive the OpenAPI page at ``/api/docs``).

Large numeric payloads are built straight from numpy and returned as ready-made responses (FastAPI
does not re-validate those), so the models document the shape exactly while staying fast. NaN is always
``null`` on the wire, hence ``float | None`` everywhere.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Num = float | None
Grid2D = list[list[Num]]
Grid3D = list[list[list[Num]]]


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable reason; never a stack trace.")


class Span(BaseModel):
    start: str | None = None
    end: str | None = None


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    n_runs: int
    ui_built: bool = Field(description="True when web/dist/index.html exists and is served at /.")


# ---------------------------------------------------------------------------------------- runs
class Artefacts(BaseModel):
    predictions: bool
    metrics_glorys: bool
    metrics_argo: bool
    argo_matchups: bool
    maps: bool
    embeddings: bool
    report: bool
    training_logs: list[str] = Field(description="Log stems, e.g. pretrain, train, train_scratch.")
    n_figures: int
    n_product_files: int


class GridSummary(BaseModel):
    resolution: float
    n_lat: int
    n_lon: int
    n_depth: int
    lat: list[float] = Field(description="[first, last] cell-centre latitude")
    lon: list[float] = Field(description="[first, last] cell-centre longitude")
    depths: list[float]


class RunSummary(BaseModel):
    name: str = Field(description="Folder name; the identifier used in every URL.")
    run_name: str
    data_source: Literal["synthetic", "real", "unknown"]
    updated: str | None
    period: Span
    split: dict[str, Span]
    grid: GridSummary
    artefacts: Artefacts
    n_prediction_days: int
    note: str | None = Field(
        None, description="Set for synthetic runs: they only demo the pipeline."
    )
    label: str = Field(
        "", description="Human display name (config `label`, else derived from source and period)."
    )
    description: str = Field(
        "", description="One-paragraph description (config `description`, else derived)."
    )
    n_train_days: int | None = Field(None, description="Days of the train split in the data.")
    n_val_days: int | None = None
    n_test_days: int | None = None
    n_harmonic_terms: int | None = Field(
        None,
        description="Terms of the climatology fit: 1 = mean only, 3 = + annual, 5 = + semi-annual.",
    )


class RunsResponse(BaseModel):
    runs: list[RunSummary]


class GridDetail(GridSummary):
    lat_values: list[float]
    lon_values: list[float]
    depth_values: list[float]
    lat_edges: list[float]
    lon_edges: list[float]


class Basin(BaseModel):
    key: str
    label: str
    box: dict[str, float] = Field(description="lat_min, lat_max, lon_min, lon_max in degrees")


class MethodInfo(BaseModel):
    key: str
    label: str
    kind: str = Field(description="model | baseline | reanalysis")
    tag: str | None = None
    pretrained: bool | None = None
    epoch: int | None = None
    val_rmse: Num = None
    description: str | None = None
    in_glorys_metrics: bool
    in_argo_metrics: bool
    has_day_fields: bool = Field(
        False, description="True when /fields, /profile, ... can serve this method (`method=`)."
    )


class DataProduct(BaseModel):
    variable: str
    used_by_model: bool = Field(
        True, description="False for an input variable the run's model does not use (input_groups)."
    )
    role: Literal["input", "target", "validation"]
    long_name: str
    units: str
    product: str
    dataset_ids: list[str]
    native_resolution: str
    regridding: str


class FieldMethod(BaseModel):
    key: str = Field(description="Value of the `method` query parameter.")
    label: str
    kind: str = Field(description="model | baseline (ridge, mlp) | ablation (model_<tag>)")
    tag: str | None = Field(None, description="Ablation tag (model_<tag>), else null.")
    n_days: int
    first: str | None
    last: str | None


class RunDetail(BaseModel):
    summary: RunSummary
    grid: GridDetail
    basins: list[Basin]
    prediction_dates: list[str]
    methods: list[MethodInfo]
    model: dict[str, Any] = Field(description="Model / pretraining / training config summary.")
    training_summary: dict[str, Any]
    products: list[DataProduct]
    counts: dict[str, int | None]
    metrics_metadata: dict[str, Any] | None = Field(
        None, description="metadata block of metrics_glorys.json (pooled range, conventions, ...)"
    )
    field_methods: list[FieldMethod] = Field(
        default_factory=list,
        description="Methods with day fields (prediction product), `model` first.",
    )


class DatesResponse(BaseModel):
    dates: list[str] = Field(description="Days with a prediction")
    target_dates: list[str] = Field(description="Subset that also has a GLORYS target")
    embedding_dates: list[str]
    first: str | None
    last: str | None


class MaskResponse(BaseModel):
    shape: list[int] = Field(description="[n_depth, n_lat, n_lon]")
    depths: list[float]
    encoding: Literal["packbits-base64"] = "packbits-base64"
    order: str = Field(
        "C", description="Bits are the C-order (lat, lon) mask of one depth, MSB first."
    )
    data: list[str] = Field(description="One base64 string per depth; bit 1 = ocean (finite).")
    n_ocean: list[int]


# ---------------------------------------------------------------------------------------- fields
class ColourRange(BaseModel):
    vmin: float
    vmax: float
    diverging: bool = Field(description="True: symmetric about zero (use a diverging colormap).")


class FieldStats(BaseModel):
    n_valid: int
    min: Num
    max: Num
    mean: Num
    std: Num
    rmse: Num = None
    bias: Num = None
    mae: Num = None


class FieldResponse(BaseModel):
    run: str
    date: str
    kind: str
    method: str = Field("model", description="Prediction method used by the prediction kinds.")
    label: str
    units: str
    depth: float | None = Field(description="Metres; null for a full volume.")
    depth_index: int | None
    shape: list[int] = Field(description="[lat, lon] or [depth, lat, lon]")
    data: Grid2D | Grid3D
    color_range: ColourRange = Field(description="Hint for the requested slice / the whole volume.")
    color_range_per_depth: list[ColourRange] | None = Field(
        None, description="Volumes only: one hint per depth level."
    )
    stats: FieldStats | None = Field(None, description="Single-depth responses only.")
    has_target: bool


class MethodRanges(BaseModel):
    label: str
    kind: str = Field(description="model | baseline | ablation")
    difference: list[ColourRange | None] | None = Field(
        None, description="[depth] symmetric limits of prediction minus target (null: no target)"
    )
    anomaly: list[ColourRange | None] | None = Field(
        None,
        description="[depth] symmetric limit shared by anomaly_pred and anomaly_target "
        "(null: no climatology)",
    )


class RangesResponse(BaseModel):
    run: str
    units: str = "degC"
    depths: list[float]
    n_days_available: int
    n_days_sampled: int
    sampled_dates: list[str]
    n_days_with_target: int
    temperature: list[ColourRange] = Field(
        description="[depth] range shared by prediction, target and climatology panels"
    )
    temperature_volume: ColourRange = Field(description="One range over all depths")
    methods: dict[str, MethodRanges] = Field(description="method key -> diverging ranges")
    method_note: str = Field(description="How the ranges were computed (days, percentiles)")


class SurfaceField(BaseModel):
    long_name: str
    units: str
    data: Grid2D
    color_range: ColourRange


class SurfaceResponse(BaseModel):
    run: str
    date: str
    shape: list[int]
    fields: dict[str, SurfaceField]


class ProfileResponse(BaseModel):
    run: str
    date: str
    method: str = Field("model", description="Method of the `prediction` array.")
    method_label: str = ""
    requested: dict[str, float]
    lat: float = Field(description="Snapped cell-centre latitude")
    lon: float
    lat_index: int
    lon_index: int
    is_ocean: bool
    depths: list[float]
    prediction: list[Num]
    target: list[Num]
    climatology: list[Num]
    has_target: bool
    units: str = "degC"


class SectionResponse(BaseModel):
    run: str
    date: str
    method: str = Field("model", description="Method of `prediction` / `difference`.")
    method_label: str = ""
    along: Literal["lon", "lat"] = Field(description="Axis of the section: lon = a zonal section.")
    fixed: Literal["lat", "lon"]
    fixed_value: float = Field(description="Snapped cell-centre value of the fixed coordinate.")
    requested: float
    distance: list[float] = Field(description="Cell centres along the section (degrees).")
    depths: list[float]
    prediction: Grid2D = Field(description="[depth][distance]")
    target: Grid2D
    climatology: Grid2D
    difference: Grid2D
    color_range: ColourRange
    difference_range: ColourRange
    has_target: bool
    units: str = "degC"


class TimeseriesResponse(BaseModel):
    run: str
    method: str = Field("model", description="Method of `prediction` / rmse / bias.")
    method_label: str = ""
    requested: dict[str, float]
    lat: float
    lon: float
    is_ocean: bool
    dates: list[str]
    depths: list[float]
    prediction: Grid2D = Field(description="[time][depth]")
    target: Grid2D = Field(description="[time][depth]; null rows where GLORYS is unavailable")
    rmse_by_day: list[Num] = Field(
        description="sqrt(mean over depths of (pred - target)^2) per day"
    )
    bias_by_day: list[Num]
    climatology: Grid2D | None = Field(
        None, description="[time][depth] harmonic climatology at the cell (null: no statistics)"
    )
    color_range: ColourRange
    units: str = "degC"


# ---------------------------------------------------------------------------------------- metrics
class MetricsResponse(BaseModel):
    run: str
    reference: str
    metadata: dict[str, Any]
    methods: list[MethodInfo]
    metric_names: list[str]
    overall: dict[str, dict[str, Num]] = Field(description="method -> metric block")
    pooled: dict[str, dict[str, Num]] = Field(description="method -> pooled 50-200 m block")
    pooled_range_m: list[float]
    depths: list[float]
    per_depth: dict[str, dict[str, list[Num]]] = Field(description="metric -> method -> [depth]")
    per_basin: dict[str, dict[str, Any]] = Field(
        description="basin -> {overall, pooled, per_depth} with the same structure as above"
    )
    daily_rmse: dict[str, Any] | None = Field(
        None, description="{dates, depths, methods: {method: [time][depth]}} (vs GLORYS only)"
    )
    daily: dict[str, Any] | None = Field(
        None,
        description="Daily series vs GLORYS: {dates, depths, pooled_range_m, rmse, bias, corr_anom: "
        "{method: [time][depth]}, pooled_rmse / pooled_bias / pooled_corr_anom: {method: [time]} over "
        "pooled_range_m}; corr_anom is the spatial anomaly correlation of that day. Keys missing "
        "from older runs are absent (re-run `evaluate`).",
    )
    per_year: dict[str, Any] | None = Field(
        None,
        description="Present when the evaluated period spans several calendar years: "
        "{year: {n_days | n_profiles, ..., overall, pooled, per_depth, per_basin}}, each block "
        "shaped like the whole-period fields above. The whole-period fields stay as they were.",
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="e.g. gridded_argo (vs Argo)")


class MapsIndex(BaseModel):
    run: str
    metrics: dict[str, list[str]] = Field(description="metric -> methods that have a map")
    depths: list[float]
    has_n_valid: bool


class RangeInfo(BaseModel):
    n_valid: int
    data_min: Num
    data_max: Num
    n_below: int = Field(description="Valid points below color_range.vmin (they saturate)")
    n_above: int = Field(description="Valid points above color_range.vmax")
    fraction_outside: float
    exceeds_range: bool = Field(description="True when any value lies outside the colour range")
    limit_rule: str | None = Field(None, description="skill_vs_clim: how the limit was chosen")
    limit_capped: bool | None = Field(
        None, description="skill_vs_clim: True when the percentile limit was cut at the cap"
    )


class MapResponse(BaseModel):
    run: str
    metric: str
    method: str
    label: str
    depth: float
    depth_index: int
    units: str
    shape: list[int]
    data: Grid2D
    color_range: ColourRange = Field(description="Shared by every method at this depth.")
    range_info: RangeInfo | None = Field(
        None, description="Data extremes and how many points fall outside color_range."
    )


# ---------------------------------------------------------------------------------------- argo
class MatchupsResponse(BaseModel):
    run: str
    n_total: int
    n_returned: int
    downsampled: bool
    methods: list[str] = Field(description="Prediction columns present (model, ridge, clim, ...).")
    columns: dict[str, list[Any]] = Field(
        description="Columnar rows: profile_id, time, lat, lon, depth, basin, obs, glorys, + methods"
    )


class ArgoProfileSummary(BaseModel):
    profile_id: str
    time: str
    lat: float
    lon: float
    basin: str | None
    n_levels: int
    rmse: dict[str, Num] = Field(description="Per-profile RMSE vs Argo for every method column.")


class ArgoProfilesResponse(BaseModel):
    run: str
    n_profiles: int = Field(description="Profiles matching the filters (all when none is given).")
    n_total: int = Field(0, description="Profiles of the run, before filtering.")
    n_returned: int = Field(0, description="Length of `profiles` (the requested page).")
    offset: int = 0
    limit: int = 0
    has_more: bool = Field(False, description="True when more matching profiles follow the page.")
    around: str | None = Field(None, description="Echo of the `around` parameter.")
    around_index: int | None = Field(
        None, description="Position of `around` in the filtered, sorted list (0-based)."
    )
    profiles: list[ArgoProfileSummary]


class ArgoProfileDetail(BaseModel):
    run: str
    profile_id: str
    time: str
    lat: float
    lon: float
    grid_lat: Num
    grid_lon: Num
    basin: str | None
    depth: list[float]
    obs: list[Num]
    series: dict[str, list[Num]] = Field(description="method column -> values at the same depths")
    units: str = "degC"


# ---------------------------------------------------------------------------------------- embeddings
class EmbeddingResponse(BaseModel):
    run: str
    date: str
    shape: list[int] = Field(description="[n_components, y, x]")
    data: Grid3D = Field(description="PCA components scaled to 0..1 (1-3 are R, G, B)")
    explained_variance_ratio: list[float] = Field(description="One value per component.")
    lat: list[float] = Field(description="Cell-centre latitude of each embedding row (y)")
    lon: list[float]
    cell_size: list[float] = Field(description="[dlat, dlon] of one embedding cell in degrees")
    ocean: list[list[int]] = Field(
        description="1 where any 0.25 degree surface ocean cell maps here"
    )


class EmbeddingSimilarityResponse(BaseModel):
    run: str
    date: str
    requested: dict[str, float]
    lat: float = Field(description="Snapped embedding-cell centre latitude")
    lon: float
    y_index: int
    x_index: int
    emb_dim: int = Field(description="Features used (all of them, not the PCA components).")
    centered: bool = Field(description="True when the run's mean embedding was subtracted first.")
    reference_is_ocean: bool | None = Field(
        None,
        description="False when the reference cell is land: its own value is then null as well "
        "(always present in responses; optional in the schema so older fixtures stay valid).",
    )
    shape: list[int] = Field(description="[y, x]")
    data: Grid2D = Field(
        description="Cosine similarity to the reference cell, -1..1 (1 = itself); null on land "
        "cells (`ocean` = 0)"
    )
    similarity_range: dict[str, Num] = Field(
        description="min / max over the other ocean cells (land is excluded)"
    )
    lat_values: list[float] = Field(description="Cell-centre latitudes (y)")
    lon_values: list[float]
    cell_size: list[float]
    ocean: list[list[int]]


class EmbeddingDatesResponse(BaseModel):
    run: str
    dates: list[str]


# ---------------------------------------------------------------------------------------- misc
class TrainingResponse(BaseModel):
    run: str
    logs: dict[str, dict[str, Any]] = Field(
        description="stem -> {kind, n_epochs, columns: {name: [values per epoch]}, best}"
    )


class ExperimentRow(BaseModel):
    method: str
    label: str
    kind: str
    pretrained: bool | None = None
    epoch: int | None = None
    val_rmse: Num = None
    glorys_overall: dict[str, Num] | None = None
    glorys_pooled: dict[str, Num] | None = None
    glorys_depths: dict[str, dict[str, Num]] | None = Field(
        None, description="'<depth m>' -> metric block at that depth"
    )
    argo_overall: dict[str, Num] | None = None
    pooled_rmse_gain_vs_climatology_pct: Num = None
    pooled_rmse_gain_vs_ridge_pct: Num = None


class ExperimentsResponse(BaseModel):
    run: str
    data_source: str
    pooled_range_m: list[float]
    selected_depths: list[float]
    rows: list[ExperimentRow]
    notes: list[str]


class CompareRun(BaseModel):
    name: str
    data_source: str
    test_period: Span
    n_days: int | None
    n_argo_profiles: int | None
    experiments: ExperimentsResponse


class CompareResponse(BaseModel):
    runs: list[CompareRun]


class FigureInfo(BaseModel):
    name: str
    url: str
    size: int


class ReportResponse(BaseModel):
    run: str
    markdown: str
    figures: list[FigureInfo]


class ProductFile(BaseModel):
    name: str
    size: int
    month: str
    start: str | None
    end: str | None
    n_days: int
    url: str = Field(description="Download URL; extra products carry `?method=<key>`.")


class ExtraProduct(BaseModel):
    method: str = Field(description="`ridge` or an ablation `model_<tag>`")
    label: str
    kind: str = Field(description="baseline (ridge) | ablation (tagged model_<tag>)")
    files: list[ProductFile]
    total_size: int


class ProductResponse(BaseModel):
    run: str
    files: list[ProductFile] = Field(description="The main model's monthly files (method=model).")
    total_size: int
    extra_products: list[ExtraProduct] = Field(
        default_factory=list, description="Ridge and ablation products, clearly labelled."
    )
