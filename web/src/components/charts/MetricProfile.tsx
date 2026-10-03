/** One skill metric against depth for several methods (depth on the y axis, increasing downward). */
import { useMemo } from "react";
import type { PerDepth } from "@/api/types";
import { fmt, fmtDepth } from "@/lib/format";
import type { MethodStyle } from "@/lib/methods";
import { metricAxisLabel, metricDomain, metricMeta } from "@/lib/metricMeta";
import { XYChart, type ChartSeries } from "./XYChart";

export interface MetricProfileProps {
  perDepth: PerDepth;
  depths: readonly number[];
  metric: string;
  methods: readonly string[];
  styleOf: (key: string) => MethodStyle;
  labelOf: (key: string) => string;
  height?: number;
  /** shaded pooled depth range */
  pooledRange?: readonly number[] | null;
  /** depth selected elsewhere */
  depth?: number | null;
  onPickDepth?: (depth: number) => void;
  reference: string;
  compact?: boolean;
  /** share a domain between charts of the same metric (e.g. two basins) */
  domain?: [number, number];
}

export function metricSeries(
  perDepth: PerDepth,
  depths: readonly number[],
  metric: string,
  methods: readonly string[],
  styleOf: (key: string) => MethodStyle,
  labelOf: (key: string) => string,
): ChartSeries[] {
  const block = perDepth[metric] ?? {};
  return methods
    // climatology has no anomaly correlation and a skill of exactly 0 by definition: the zero line says it
    .filter((m) => !(m === "climatology" && (metric === "skill_vs_clim" || metric === "corr_anom")))
    .filter((m) => block[m]?.some((v) => v != null))
    .map((m) => ({
      key: m,
      label: labelOf(m),
      style: styleOf(m),
      points: depths.map((d, k) => ({ x: block[m][k] ?? null, y: d })),
    }));
}

export function MetricProfile(props: MetricProfileProps) {
  const { perDepth, depths, metric, methods, styleOf, labelOf, height = 340, pooledRange, depth, onPickDepth, reference, compact } = props;
  const meta = metricMeta(metric);
  const series = useMemo(() => metricSeries(perDepth, depths, metric, methods, styleOf, labelOf), [perDepth, depths, metric, methods, styleOf, labelOf]);
  const domain = useMemo(
    () => props.domain ?? metricDomain(meta, series.flatMap((s) => s.points.map((p) => p.x))),
    [props.domain, meta, series],
  );
  const maxDepth = depths[depths.length - 1] ?? 1;
  return (
    <XYChart
      series={series}
      x={{ label: metricAxisLabel(meta), domain, format: (v) => fmt(v, meta.digits) }}
      y={{ label: "Depth (m)", domain: [0, maxDepth], scale: "depth", ticks: [...depths] }}
      hover="y"
      height={height}
      compact={compact}
      zero={meta.zero ? "x" : null}
      band={pooledRange && pooledRange.length >= 2 ? { axis: "y", from: pooledRange[0], to: pooledRange[1], label: "pooled range" } : null}
      reference={depth != null ? { axis: "y", value: depth } : null}
      onPick={onPickDepth}
      hoverTitle={(d) => `${meta.label} at ${fmtDepth(d)}`}
      ariaLabel={`${meta.label} against ${reference} by depth for ${series.map((s) => s.label).join(", ")}`}
    />
  );
}
