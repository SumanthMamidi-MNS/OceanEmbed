/**
 * API types. The response shapes come from `schema.d.ts`, generated from /api/openapi.json
 * (`npm run gen:api`). Where the OpenAPI document only says "object" (free-form dictionaries),
 * the structure documented in docs/api.md is narrowed here.
 */
import type { components } from "./schema";

type S = components["schemas"];

export type Health = S["Health"];
export type RunSummary = S["RunSummary"];
export type GridDetail = S["GridDetail"];
export type Basin = Omit<S["Basin"], "box"> & {
  box: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
};
export type MethodInfo = S["MethodInfo"];
export type DataProduct = S["DataProduct"];
export type ColourRange = S["ColourRange"];
export type DatesResponse = S["DatesResponse"];
export type MaskResponse = S["MaskResponse"];
export type SurfaceResponse = S["SurfaceResponse"];
export type ProfileResponse = S["ProfileResponse"];
export type SectionResponse = S["SectionResponse"];
export type TimeseriesResponse = S["TimeseriesResponse"];
export type MapsIndex = S["MapsIndex"];
export type MapResponse = S["MapResponse"];
export type ArgoProfileSummary = S["ArgoProfileSummary"];
export type ArgoProfilesResponse = S["ArgoProfilesResponse"];
export type ArgoProfileDetail = S["ArgoProfileDetail"];
export type EmbeddingResponse = S["EmbeddingResponse"];
export type EmbeddingSimilarityResponse = S["EmbeddingSimilarityResponse"];
export type FieldMethod = S["FieldMethod"];
export type ExtraProduct = S["ExtraProduct"];
export type RangesResponse = S["RangesResponse"];
export type RangeInfo = S["RangeInfo"];
export type ExperimentRow = S["ExperimentRow"];
export type ExperimentsResponse = S["ExperimentsResponse"];
export type CompareResponse = S["CompareResponse"];
export type CompareRun = S["CompareRun"];
export type ReportResponse = S["ReportResponse"];
export type ProductResponse = S["ProductResponse"];
export type ProductFile = S["ProductFile"];
export type Span = S["Span"];

/** n, rmse, bias, mae, corr_raw, corr_anom, skill_vs_clim */
export type MetricBlock = Record<string, number | null>;
export type MetricName = "rmse" | "bias" | "mae" | "corr_raw" | "corr_anom" | "skill_vs_clim" | "n";

/** metric -> method -> value per depth */
export type PerDepth = Record<string, Record<string, (number | null)[]>>;

export interface BasinMetrics {
  overall: Record<string, MetricBlock>;
  pooled: Record<string, MetricBlock>;
  per_depth: PerDepth;
}

export interface DailyRmse {
  dates: string[];
  depths: number[];
  /** method -> [day][depth] */
  methods: Record<string, (number | null)[][]>;
}

/** method -> [day][depth] */
export type DailyByDepth = Record<string, (number | null)[][]>;

/**
 * Daily series against GLORYS. `corr_anom` is the spatial correlation of the anomalies of one day
 * at one depth (not the temporal correlation of the per-depth table). Blocks other than `rmse`
 * are absent from runs evaluated before they existed.
 */
export interface DailyMetrics {
  dates: string[];
  depths: number[];
  pooled_range_m?: number[];
  rmse?: DailyByDepth;
  bias?: DailyByDepth;
  corr_anom?: DailyByDepth;
  /** method -> one value per day over the pooled depth range */
  pooled_rmse?: Record<string, (number | null)[]>;
  pooled_bias?: Record<string, (number | null)[]>;
  /** spatial anomaly correlation over the grid points of all pooled depths of that day */
  pooled_corr_anom?: Record<string, (number | null)[]>;
}

export interface MetricsMetadata {
  data_source?: string;
  split?: string;
  start?: string;
  end?: string;
  n_days?: number;
  units?: string;
  reference?: string;
  conventions?: Record<string, string>;
  pooled_depth_range_m?: number[];
  climatology?: { train_start?: string; train_end?: string; n_harmonic_terms?: number };
  note?: string | null;
  // Argo file only
  bias_convention?: string;
  n_profiles_loaded?: number;
  n_profiles_in_period?: number;
  n_profiles_collocated?: number;
  n_profiles_used?: number;
  n_matchups?: number;
  dropped_profiles?: Record<string, number>;
  interpolation_rule?: string;
  collocation?: string;
  independence_note?: string;
  [key: string]: unknown;
}

/** One calendar year of a test period that spans several: shaped like the whole-period fields. */
export interface YearMetrics extends BasinMetrics {
  start?: string;
  end?: string;
  /** against GLORYS */
  n_days?: number;
  /** against Argo */
  n_profiles?: number;
  n_matchups?: number;
  per_basin?: Record<string, BasinMetrics>;
}

export type MetricsResponse = Omit<S["MetricsResponse"], "per_basin" | "daily_rmse" | "daily" | "metadata" | "per_depth" | "per_year"> & {
  metadata: MetricsMetadata;
  per_depth: PerDepth;
  per_basin: Record<string, BasinMetrics>;
  daily_rmse?: DailyRmse | null;
  daily?: DailyMetrics | null;
  /** null unless the evaluated period spans several calendar years */
  per_year?: Record<string, YearMetrics> | null;
};

export interface ModelConfig {
  model?: {
    in_channels?: number;
    emb_dim?: number;
    dim?: number;
    depth?: number;
    heads?: number;
    stem_channels?: number;
    n_depths?: number;
    arch?: string;
  };
  pretrain?: { epochs?: number; batch_size?: number; lr?: number; mask_ratio?: number; block?: number };
  train?: {
    epochs?: number;
    batch_size?: number;
    lr?: number;
    encoder_lr_scale?: number;
    patience?: number;
    vertical_grad_weight?: number;
  };
  /** surface variables the model uses (a subset of the input products) */
  inputs?: string[];
  input_groups?: string[];
  dropped_input_groups?: string[];
  /** how the main model's encoder starts: "pretrained" or "scratch" */
  main_init?: string;
  output?: string;
}

export interface TrainingSummaryEntry {
  kind?: string;
  n_epochs?: number;
  best?: Record<string, number>;
  total_seconds?: number;
}

export type RunDetail = Omit<S["RunDetail"], "basins" | "model" | "training_summary" | "metrics_metadata"> & {
  basins: Basin[];
  model: ModelConfig;
  training_summary: Record<string, TrainingSummaryEntry>;
  metrics_metadata?: MetricsMetadata | null;
};

export interface TrainingLog {
  kind: string;
  tag: string | null;
  n_epochs: number;
  best: Record<string, number> | null;
  /** column name -> one entry per epoch (numbers, per-depth lists or per-channel dicts) */
  columns: Record<string, unknown[]>;
}

export interface TrainingResponse {
  run: string;
  logs: Record<string, TrainingLog>;
}

export interface MatchupColumns {
  profile_id: string[];
  time: string[];
  lat: number[];
  lon: number[];
  depth: number[];
  basin: (string | null)[];
  obs: number[];
  glorys?: (number | null)[];
  [method: string]: unknown[] | undefined;
}

export type MatchupsResponse = Omit<S["MatchupsResponse"], "columns"> & { columns: MatchupColumns };

export type FieldKind = "prediction" | "target" | "climatology" | "difference" | "anomaly_pred" | "anomaly_target";

// ---- live nowcast (GET /runs/{run}/live); the free-form blocks of the schema, narrowed by hand -----

export type LiveInput = S["LiveInput"];

/** Error of the reconstruction and of climatology over one depth band (one day, or the rolling window). */
export interface LiveBandScore {
  n: number | null;
  model_rmse: number | null;
  model_bias: number | null;
  clim_rmse: number | null;
  clim_bias: number | null;
  /** days in the rolling window (rolling blocks only) */
  n_days?: number | null;
}

export interface LiveVerificationDay {
  date: string;
  /** Argo profiles of that day (null for the analysis) */
  n_profiles: number | null;
  bands: Record<string, { day?: LiveBandScore | null; rolling?: LiveBandScore | null }>;
}

export interface LiveReference {
  reference?: string;
  note?: string;
  available?: boolean;
  n_profiles?: number | null;
  n_matchups?: number | null;
  daily?: LiveVerificationDay[];
  latest?: LiveVerificationDay | null;
}

export interface LiveVerification {
  updated?: string | null;
  rolling_days?: number | null;
  bands?: Record<string, { label?: string; depth_range_m?: number[] }>;
  argo?: LiveReference | null;
  analysis?: LiveReference | null;
}

export interface LiveShiftChange {
  point: number | null;
  ci_lo?: number | null;
  ci_hi?: number | null;
  excludes_zero?: boolean;
}

export interface LiveInputShift {
  updated?: string | null;
  headline?: {
    period?: string[];
    n_days?: number | null;
    sst_bias?: number | null;
    sst_rmse?: number | null;
    sla_bias?: number | null;
    sla_rmse?: number | null;
    recon_difference_pooled_50_200m?: { bias: number | null; rmse: number | null } | null;
    vs_glorys_all?: { rmse_reprocessed: number | null; rmse_nrt: number | null; change?: LiveShiftChange | null } | null;
    measurably_worse?: boolean | null;
  } | null;
}

export interface LiveWindowDay {
  date: string;
  reconstructed: boolean;
  complete: boolean;
  inputs: Record<string, boolean>;
}

export interface LiveRevision {
  policy?: { revision_days?: number | null; rule?: string | null } | null;
  n_days_checked?: number | null;
  n_days_revised?: number | null;
  by_age?: { age_days: number; n_checks: number; n_changed_since_first: number; recon_rmse_50_200_mean?: number | null; recon_rmse_50_200_max?: number | null }[];
}

export type LiveResponse = Omit<S["LiveResponse"], "window" | "window_days" | "revision" | "input_shift" | "verification"> & {
  window: { start?: string; end?: string; n_days?: number; window_days?: number } | null;
  window_days: LiveWindowDay[];
  revision: LiveRevision;
  input_shift: LiveInputShift | null;
  verification: LiveVerification | null;
};
