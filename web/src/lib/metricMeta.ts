/** How each skill metric is named, explained and scaled, in one place. */
import type { ColormapName } from "./colormaps";
import { padRange } from "./scales";
import { extent } from "./stats";

export interface MetricMeta {
  key: string;
  label: string;
  short: string;
  /** typographic units ("" for dimensionless) */
  unit: string;
  digits: number;
  /** how the axis domain is built from the data */
  domain: "fromZero" | "symmetric" | "upToOne" | "auto";
  /** draw an emphasised zero line */
  zero: boolean;
  /** lower is better (RMSE, MAE) or higher is better (correlation, skill); null for bias */
  better: "lower" | "higher" | null;
  explain: string;
  cmap: ColormapName;
  cmapReversed?: boolean;
}

export const METRIC_META: Record<string, MetricMeta> = {
  rmse: {
    key: "rmse",
    label: "RMSE",
    short: "RMSE",
    unit: "°C",
    digits: 2,
    domain: "fromZero",
    zero: false,
    better: "lower",
    explain: "Root-mean-square error against the reference.",
    cmap: "amp",
  },
  bias: {
    key: "bias",
    label: "Bias",
    short: "Bias",
    unit: "°C",
    digits: 2,
    domain: "symmetric",
    zero: true,
    better: null,
    explain: "Mean of method minus reference: positive is too warm.",
    cmap: "balance",
  },
  mae: {
    key: "mae",
    label: "Mean absolute error",
    short: "MAE",
    unit: "°C",
    digits: 2,
    domain: "fromZero",
    zero: false,
    better: "lower",
    explain: "Mean absolute error against the reference.",
    cmap: "amp",
  },
  corr_anom: {
    key: "corr_anom",
    label: "Anomaly correlation",
    short: "Anom. corr.",
    unit: "",
    digits: 2,
    domain: "upToOne",
    zero: true,
    better: "higher",
    explain: "Correlation after removing the climatology from both sides: the day-to-day skill.",
    cmap: "viridis",
  },
  corr_raw: {
    key: "corr_raw",
    label: "Raw correlation",
    short: "Raw corr.",
    unit: "",
    digits: 3,
    domain: "upToOne",
    zero: false,
    better: "higher",
    explain: "Correlation of the temperatures themselves; inflated by seasonal, spatial and vertical gradients.",
    cmap: "viridis",
  },
  skill_vs_clim: {
    key: "skill_vs_clim",
    label: "Skill vs climatology",
    short: "Skill",
    unit: "",
    digits: 2,
    domain: "upToOne",
    zero: true,
    better: "higher",
    explain: "1 − MSE / MSE of the climatology: above 0 beats climatology, 1 is perfect.",
    cmap: "curl",
    cmapReversed: true,
  },
};

export function metricMeta(key: string): MetricMeta {
  return (
    METRIC_META[key] ?? {
      key,
      label: key,
      short: key,
      unit: "",
      digits: 2,
      domain: "auto",
      zero: false,
      better: null,
      explain: "",
      cmap: "viridis",
    }
  );
}

export function metricAxisLabel(meta: MetricMeta): string {
  return meta.unit ? `${meta.label} (${meta.unit})` : meta.label;
}

/** Axis domain of a metric over a set of values, following the metric's convention. */
export function metricDomain(meta: MetricMeta, values: ArrayLike<number | null>): [number, number] {
  const [lo, hi] = extent(values);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  switch (meta.domain) {
    case "fromZero":
      return [0, hi > 0 ? hi * 1.06 : 1];
    case "symmetric": {
      const lim = Math.max(Math.abs(lo), Math.abs(hi)) * 1.1 || 1;
      return [-lim, lim];
    }
    case "upToOne": {
      const top = Math.min(1, hi + (hi - lo) * 0.06);
      const bottom = lo < 0 ? lo - (hi - lo) * 0.06 : Math.max(0, lo - (hi - lo) * 0.1);
      return [Math.min(bottom, 0.999 * top), top <= bottom ? bottom + 1 : top];
    }
    default:
      return padRange(lo, hi, 0.06);
  }
}

/** Order in which metrics are shown: the honest one (anomaly correlation) before the raw one. */
export const METRIC_ORDER = ["rmse", "bias", "corr_anom", "corr_raw", "skill_vs_clim"] as const;
