/**
 * Skill against the GLORYS target: tables, per-depth profiles of every metric, basin breakdown.
 * The methods are those of the view (run context `scoped`): the product and climatology on the
 * Accuracy page, every method in the Research area.
 */
import { useMemo, useState } from "react";
import type { MetricsResponse } from "@/api/types";
import { Legend } from "@/components/charts/marks";
import { MetricProfile, metricSeries } from "@/components/charts/MetricProfile";
import { MetricsTable } from "@/components/charts/MetricsTable";
import { DataTable, Note, Panel, Segmented, Select, TableTwin, type Column } from "@/components/ui/primitives";
import { fmt, fmtDepth, fmtLat, fmtLon, fmtSigned } from "@/lib/format";
import { METRIC_ORDER, metricDomain, metricMeta } from "@/lib/metricMeta";
import { YearSwitch, useYear } from "@/components/controls/YearSwitch";
import { climatologyName, methodList, scopeToYear, yearStability, yearsOf } from "@/lib/narrative";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";

export function GlorysSection({ metrics: whole }: { metrics: MetricsResponse }) {
  const { run, detail, depths, depth, setDepthIndex, styleOf, labelOf, scoped, scope } = useRunContext();
  const [url, setUrl] = useUrlState();
  // one test year at a time, or the whole period
  const years = yearsOf(whole);
  const year = useYear(years);
  const metrics = useMemo(() => scopeToYear(whole, year), [whole, year]);
  const stability = useMemo(() => yearStability(whole, scope === "research" ? "all" : "climatology", climatologyName(run.n_harmonic_terms)), [whole, scope, run.n_harmonic_terms]);
  const period = year ? `test days of ${year}` : "test period";
  const basins = detail.basins.filter((b) => metrics.per_basin[b.key]);
  const basin = basins.some((b) => b.key === url.opts.basin) ? url.opts.basin : "all";
  const source = basin === "all" ? metrics : metrics.per_basin[basin];
  const methods = useMemo(() => scoped(methodList(metrics).map((m) => m.key)), [metrics, scoped]);
  const legend = methods.map((k) => ({ key: k, label: labelOf(k), style: styleOf(k) }));
  const [twinMetric, setTwinMetric] = useState<string>("rmse");
  const range = metrics.pooled_range_m;
  const rangeText = range?.length >= 2 ? `${fmt(range[0], 0)}–${fmt(range[1], 0)} m` : "pooled range";
  const where = basin === "all" ? "whole domain" : (basins.find((b) => b.key === basin)?.label ?? basin);
  const pick = (d: number) => setDepthIndex(depths.indexOf(d));

  const metricKeys = METRIC_ORDER.filter((k) => source.per_depth[k]);

  const twinColumns: Column<number>[] = [
    { key: "depth", label: "Depth", render: (k) => fmtDepth(depths[k]) },
    ...methods
      .filter((m) => source.per_depth[twinMetric]?.[m])
      .map<Column<number>>((m) => ({
        key: m,
        label: labelOf(m),
        align: "right",
        render: (k) => {
          const v = source.per_depth[twinMetric]?.[m]?.[k];
          return twinMetric === "bias" ? fmtSigned(v, metricMeta(twinMetric).digits) : fmt(v, metricMeta(twinMetric).digits);
        },
      })),
  ];

  // basin comparison: one RMSE profile per basin on a shared axis
  const basinDomain = useMemo(() => {
    const all: (number | null)[] = [];
    for (const b of basins) {
      for (const s of metricSeries(metrics.per_basin[b.key].per_depth, depths, "rmse", methods, styleOf, labelOf)) {
        for (const p of s.points) all.push(p.x);
      }
    }
    return metricDomain(metricMeta("rmse"), all);
  }, [basins, metrics, depths, methods, styleOf, labelOf]);

  return (
    <>
      <div className="toolbar">
        {basins.length > 0 && (
          <Segmented
            label="Region"
            showLabel
            value={basin}
            onChange={(v) => setUrl({ opts: { basin: v === "all" ? null : v } })}
            options={[{ value: "all", label: "Whole domain" }, ...basins.map((b) => ({ value: b.key, label: b.label }))]}
          />
        )}
        <YearSwitch years={years} />
        {years.length >= 2 && !year && stability && <span className="caption">{stability}</span>}
      </div>

      <div className="twocol">
        <Panel title={`Pooled over ${rangeText}`} subtitle={`${where} · ${period} · the depth range where the temperature varies most`}>
          <MetricsTable blocks={source.pooled} methods={methods} styleOf={styleOf} labelOf={labelOf} caption={`Pooled metrics over ${rangeText}, ${where}`} />
        </Panel>
        <Panel title="All depths" subtitle={`${where} · ${period} · dominated by the many quiet deep levels`}>
          <MetricsTable blocks={source.overall} methods={methods} styleOf={styleOf} labelOf={labelOf} caption={`Metrics over all depths, ${where}`} />
        </Panel>
      </div>

      <Note kind="honesty" title="Reading the correlations" className="gap-top">
        The <strong>anomaly correlation</strong> is the one to trust: the climatology is removed from both the estimate and GLORYS
        before correlating, so it measures day-to-day skill. The <strong>raw correlation</strong> is shown beside it only for
        reference; it is close to 1 for every method, climatology included, because warm-over-cold stratification and the seasons
        dominate the variance. Bias is method minus reference.
      </Note>

      <Panel
        className="gap-top"
        title="Every metric by depth"
        subtitle={`${where} · ${period} · the shaded band is the pooled range · click a depth to select it for the maps and the time series below`}
      >
        <Legend items={legend} band={`pooled range ${rangeText}`} />
        <div className="multiples">
          {metricKeys.map((k) => (
            <div key={k} className="multiples__cell">
              <p className="multiples__title">{metricMeta(k).label}</p>
              <p className="multiples__sub">{metricMeta(k).explain}</p>
              <MetricProfile
                perDepth={source.per_depth}
                depths={depths}
                metric={k}
                methods={methods}
                styleOf={styleOf}
                labelOf={labelOf}
                pooledRange={range}
                depth={depth}
                onPickDepth={pick}
                reference="GLORYS"
                height={340}
                compact
              />
            </div>
          ))}
        </div>
        <TableTwin>
          <div className="toolbar">
            <Select
              label="Metric"
              value={twinMetric}
              onChange={setTwinMetric}
              options={metricKeys.map((k) => ({ value: k as string, label: metricMeta(k).label }))}
            />
          </div>
          <DataTable columns={twinColumns} rows={depths.map((_, k) => k)} rowKey={(k) => String(k)} dense caption={`${metricMeta(twinMetric).label} by depth, ${where}`} />
        </TableTwin>
      </Panel>

      {basins.length > 1 && (
        <Panel className="gap-top" title="Basin by basin" subtitle="RMSE against GLORYS by depth, same axis for both basins">
          <Legend items={legend} />
          <div className="multiples multiples--wide">
            {basins.map((b) => (
              <div key={b.key} className="multiples__cell">
                <p className="multiples__title">{b.label}</p>
                <p className="multiples__sub num">
                  {fmtLat(b.box.lat_min, 0)}–{fmtLat(b.box.lat_max, 0)}, {fmtLon(b.box.lon_min, 0)}–{fmtLon(b.box.lon_max, 0)} · pooled RMSE{" "}
                  {methods
                    .map((m) => `${labelOf(m)} ${fmt(metrics.per_basin[b.key].pooled[m]?.rmse)}`)
                    .join(" · ")}{" "}
                  °C
                </p>
                <MetricProfile
                  perDepth={metrics.per_basin[b.key].per_depth}
                  depths={depths}
                  metric="rmse"
                  methods={methods}
                  styleOf={styleOf}
                  labelOf={labelOf}
                  pooledRange={range}
                  depth={depth}
                  onPickDepth={pick}
                  reference={`GLORYS in the ${b.label}`}
                  height={330}
                  domain={basinDomain}
                />
              </div>
            ))}
          </div>
        </Panel>
      )}
    </>
  );
}
