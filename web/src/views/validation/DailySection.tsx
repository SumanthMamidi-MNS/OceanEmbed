/**
 * When the error is: the scores of each test day. Lines for every method (RMSE, bias and the
 * spatial anomaly correlation at the selected depth, RMSE pooled over the thermocline range) and a
 * day-by-depth field for the selected estimate.
 *
 * The daily anomaly correlation is spatial: the correlation, across the grid cells of one day at
 * one depth, of the anomalies of the estimate and of GLORYS. It is not the temporal correlation
 * of the per-depth table, and the labels say so.
 */
import { useMemo } from "react";
import type { DailyByDepth, MetricsResponse } from "@/api/types";
import { DepthHeatmap } from "@/components/charts/DepthHeatmap";
import { Legend } from "@/components/charts/marks";
import { XYChart, type ChartSeries } from "@/components/charts/XYChart";
import { Colorbar } from "@/components/map/Colorbar";
import { Empty, Panel, Segmented } from "@/components/ui/primitives";
import type { ColormapName } from "@/lib/colormaps";
import { fmtDate, fromUtcMs, toUtcMs } from "@/lib/dates";
import { fmt, fmtDepth, fmtSigned } from "@/lib/format";
import { sortMethods } from "@/lib/methods";
import { padRange } from "@/lib/scales";
import { extent, quantile } from "@/lib/stats";
import { color } from "@/lib/theme";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { byDepth } from "../explorer/fields";

type HeatKey = "rmse" | "gain" | "bias" | "corr";

interface HeatSpec {
  values: (number | null)[][];
  cmap: ColormapName;
  vmin: number;
  vmax: number;
  label: string;
  units: string;
  signed: boolean;
  extend: "both" | "max" | "none";
  format: (v: number) => string;
}

function flat(m: (number | null)[][]): number[] {
  const out: number[] = [];
  for (const row of m) for (const v of row) if (v != null && Number.isFinite(v)) out.push(v);
  return out;
}

export function DailySection({ metrics }: { metrics: MetricsResponse }) {
  const { depths, depthIndex, setDepthIndex, styleOf, labelOf, dates, date, setDateIndex, estimate } = useRunContext();
  const [url, setUrl] = useUrlState();
  const daily = metrics.daily;
  // runs evaluated before the daily block existed only have the RMSE
  const rmse: DailyByDepth | undefined = daily?.rmse ?? metrics.daily_rmse?.methods;
  const dayList = useMemo(() => daily?.dates ?? metrics.daily_rmse?.dates ?? [], [daily, metrics.daily_rmse]);
  const levels = useMemo(() => daily?.depths ?? metrics.daily_rmse?.depths ?? [], [daily, metrics.daily_rmse]);
  const ms = useMemo(() => dayList.map(toUtcMs), [dayList]);
  const k = levels.indexOf(depths[depthIndex]);
  const depthText = fmtDepth(depths[depthIndex]);
  const range = daily?.pooled_range_m ?? metrics.pooled_range_m;
  const rangeText = range?.length >= 2 ? `${fmt(range[0], 0)}–${fmt(range[1], 0)} m` : "the pooled range";

  // scope of the lines: the selected depth, or the pooled thermocline range (one value per day)
  const hasPooled = !!daily?.pooled_rmse;
  const pooledScope = hasPooled && url.opts.ds === "pooled";
  const series = (value: (m: string, t: number) => number | null | undefined, methods: readonly string[]): ChartSeries[] =>
    sortMethods([...methods])
      .map((m) => ({ key: m, label: labelOf(m), style: styleOf(m), markers: false, points: dayList.map((_, t) => ({ x: ms[t], y: value(m, t) ?? null })) }))
      .filter((s2) => s2.points.some((p) => p.y != null));
  const atDepth = (block: DailyByDepth | undefined, skip: readonly string[] = []) =>
    block && k >= 0 ? series((m, t) => block[m][t]?.[k], Object.keys(block).filter((m) => !skip.includes(m))) : [];
  const pooledOf = (block: Record<string, (number | null)[]> | undefined, skip: readonly string[] = []) =>
    block ? series((m, t) => block[m]?.[t], Object.keys(block).filter((m) => !skip.includes(m))) : [];
  // the climatology has no anomaly, so no anomaly correlation
  const rmseLines = pooledScope ? pooledOf(daily?.pooled_rmse) : atDepth(rmse);
  const biasLines = pooledScope ? pooledOf(daily?.pooled_bias) : atDepth(daily?.bias);
  const corrLines = pooledScope ? pooledOf(daily?.pooled_corr_anom, ["climatology"]) : atDepth(daily?.corr_anom, ["climatology"]);

  // ---- day x depth field of the selected estimate ------------------------------------------
  const heatMethod = rmse?.[estimate] ? estimate : "model";
  const heat = useMemo(() => {
    const nz = levels.length;
    const out: Partial<Record<HeatKey, HeatSpec>> = {};
    const r = rmse?.[heatMethod];
    if (r) {
      const values = byDepth(r, nz);
      out.rmse = { values, cmap: "amp", vmin: 0, vmax: quantile(flat(values), 0.99) || 1, label: "RMSE", units: "°C", signed: false, extend: "max", format: (v) => `RMSE ${fmt(v)} °C` };
      const c = rmse?.climatology;
      if (c) {
        const clim = byDepth(c, nz);
        const gain = values.map((row, d) => row.map((v, t) => (v == null || clim[d][t] == null ? null : v - (clim[d][t] as number))));
        const lim = quantile(flat(gain).map(Math.abs), 0.99) || 1;
        out.gain = { values: gain, cmap: "balance", vmin: -lim, vmax: lim, label: "RMSE − climatology's", units: "°C", signed: true, extend: "both", format: (v) => `${fmtSigned(v)} °C vs climatology` };
      }
    }
    const b = daily?.bias?.[heatMethod];
    if (b) {
      const values = byDepth(b, nz);
      const lim = quantile(flat(values).map(Math.abs), 0.99) || 1;
      out.bias = { values, cmap: "balance", vmin: -lim, vmax: lim, label: "Bias", units: "°C", signed: true, extend: "both", format: (v) => `bias ${fmtSigned(v)} °C` };
    }
    const ca = daily?.corr_anom?.[heatMethod];
    if (ca) {
      const values = byDepth(ca, nz);
      const lim = Math.min(1, Math.max(0.1, quantile(flat(values).map(Math.abs), 0.99) || 1));
      out.corr = { values, cmap: "balance", vmin: -lim, vmax: lim, label: "Spatial anomaly correlation", units: "", signed: true, extend: "none", format: (v) => `spatial anomaly correlation ${fmtSigned(v)}` };
    }
    return out;
  }, [rmse, daily, heatMethod, levels.length]);

  if (!rmse || dayList.length === 0) {
    return <Empty title="No daily error series in this run's metrics">Re-run “oceanembed evaluate” to produce the daily series.</Empty>;
  }

  const xAxis = { label: "", domain: [ms[0], ms[ms.length - 1]] as [number, number], scale: "time" as const };
  const markX = date ? toUtcMs(date) : null;
  const reference = markX != null ? { axis: "x" as const, value: markX } : null;
  const pickDay = (msValue: number) => {
    const i = dates.indexOf(fromUtcMs(msValue));
    if (i >= 0) setDateIndex(i);
  };
  const hoverTitle = (v: number) => fmtDate(fromUtcMs(v));
  const top = (series: ChartSeries[]) => Math.max(0, ...series.flatMap((s) => s.points.map((p) => p.y ?? 0)));
  const fromZero = (series: ChartSeries[]): [number, number] => [0, top(series) > 0 ? top(series) * 1.08 : 1];
  // the data and the zero line, whichever side of it the bias sits on
  const aroundZero = (series: ChartSeries[]): [number, number] => {
    const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p.y)));
    if (!Number.isFinite(lo)) return [-1, 1];
    return padRange(Math.min(lo, 0), Math.max(hi, 0), 0.1);
  };
  const corrDomain = (series: ChartSeries[]): [number, number] => {
    const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p.y)));
    if (!Number.isFinite(lo)) return [0, 1];
    const [a, b] = padRange(Math.min(lo, 0), hi, 0.08);
    return [Math.max(-1, a), Math.min(1, b)];
  };

  const where = pooledScope ? `pooled over ${rangeText}` : `at ${depthText}`;
  const cells = pooledScope ? "every cell and level of the range" : "every ocean cell of the level";
  const charts: { key: string; title: string; sub: string; series: ChartSeries[]; y: string; domain: [number, number]; zero: boolean }[] = [
    { key: "rmse", title: `RMSE ${where}`, sub: `over ${cells}, one value per day`, series: rmseLines, y: "RMSE (°C)", domain: fromZero(rmseLines), zero: false },
  ];
  if (biasLines.length > 0) {
    charts.push({ key: "bias", title: `Bias ${where}`, sub: `mean of method minus GLORYS over ${cells}: positive is too warm`, series: biasLines, y: "Bias (°C)", domain: aroundZero(biasLines), zero: true });
  }
  if (corrLines.length > 0) {
    charts.push({
      key: "corr",
      title: `Spatial anomaly correlation ${where}`,
      sub: `across ${pooledScope ? "the cells and levels of the range" : "the cells of the level"} on that day, climatology removed from both sides: whether the day's pattern is right. Not the temporal correlation of the per-depth table.`,
      series: corrLines,
      y: "Correlation",
      domain: corrDomain(corrLines),
      zero: true,
    });
  }

  const heatKeys = (["rmse", "gain", "bias", "corr"] as const).filter((h) => heat[h]);
  const heatKey: HeatKey = heatKeys.includes(url.opts.dh as HeatKey) ? (url.opts.dh as HeatKey) : (heatKeys[0] ?? "rmse");
  const spec = heat[heatKey];
  const HEAT_LABEL: Record<HeatKey, string> = { rmse: "RMSE", gain: "vs climatology", bias: "Bias", corr: "Spatial anom. corr." };

  return (
    <>
      <Panel
        title="Day by day"
        subtitle={`against GLORYS, whole domain, ${dayList.length} test days · click a day to select it${pooledScope ? "" : " · the depth is the one of the selection bar"}`}
        actions={
          hasPooled && (
            <Segmented
              label="Depths of the daily series"
              size="sm"
              value={pooledScope ? "pooled" : "depth"}
              onChange={(v) => setUrl({ opts: { ds: v === "depth" ? null : v } })}
              options={[
                { value: "depth", label: `At ${depthText}` },
                { value: "pooled", label: `Pooled ${rangeText}`, title: "Every level of the thermocline range together, one value per day" },
              ]}
            />
          )
        }
      >
        <Legend items={rmseLines.map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
        <div className="dailystack">
          {charts.map((c) => (
            <div key={c.key} className="multiples__cell">
              <p className="multiples__title">{c.title}</p>
              <p className="multiples__sub">{c.sub}</p>
              <XYChart
                series={c.series}
                x={xAxis}
                y={{ label: c.y, domain: c.domain }}
                hover="x"
                height={200}
                zero={c.zero ? "y" : null}
                reference={reference}
                onPick={pickDay}
                hoverTitle={hoverTitle}
                ariaLabel={`Daily ${c.title} against GLORYS for ${c.series.map((s) => s.label).join(", ")}`}
              />
            </div>
          ))}
        </div>
      </Panel>

      {spec && (
        <Panel
          className="gap-top"
          title={`Every day, every depth: ${labelOf(heatMethod)}`}
          subtitle="whole domain · click a cell to select that day and depth · the estimate is the one of the selection bar"
          actions={
            heatKeys.length > 1 && (
              <Segmented
                label="Daily score shown"
                size="sm"
                value={heatKey}
                onChange={(v) => setUrl({ opts: { dh: v === heatKeys[0] ? null : v } })}
                options={heatKeys.map((h) => ({ value: h, label: HEAT_LABEL[h], title: heat[h]?.label }))}
              />
            )
          }
        >
          <div className="stack">
            <DepthHeatmap
              title={spec.label}
              values={spec.values}
              depths={levels}
              xValues={ms}
              xKind="time"
              cmap={spec.cmap}
              vmin={spec.vmin}
              vmax={spec.vmax}
              height={250}
              markX={markX}
              markDepth={depths[depthIndex]}
              onPick={(xi, di) => {
                const i = dates.indexOf(dayList[xi]);
                if (i >= 0) setDateIndex(i);
                const d = depths.indexOf(levels[di]);
                if (d >= 0) setDepthIndex(d);
              }}
              formatValue={spec.format}
              nullFill={color.sunken}
              ariaLabel={`${spec.label} of ${labelOf(heatMethod)} by day and depth`}
            />
            <div className="stack__bars">
              <Colorbar spec={{ cmap: spec.cmap, vmin: spec.vmin, vmax: spec.vmax, units: spec.units, label: spec.label, extend: spec.extend, signed: spec.signed }} />
            </div>
          </div>
          {heatKey === "gain" && <p className="caption gap-top-sm">Blue: the estimate's daily RMSE is below the climatology's.</p>}
          {heatKey === "corr" && (
            <p className="caption gap-top-sm">
              Spatial correlation of the anomalies of that day at that depth (climatology removed from the estimate and from GLORYS). Zero is white.
            </p>
          )}
        </Panel>
      )}
    </>
  );
}
