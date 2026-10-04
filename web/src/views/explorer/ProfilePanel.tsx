/**
 * Vertical temperature profile under the selected point: the estimates the view offers (the
 * reconstruction; in the Research area also ridge regression and the ablation), GLORYS and the climatology, each with its fixed
 * colour, dash and marker, and the values at the selected depth as a table.
 */
import { useMemo } from "react";
import type { DayVolumes } from "@/api/volumes";
import { Legend, Swatch } from "@/components/charts/marks";
import { XYChart, type ChartSeries } from "@/components/charts/XYChart";
import { Empty, Panel } from "@/components/ui/primitives";
import { fmt, fmtDepth, fmtLatLon, fmtSigned } from "@/lib/format";
import { cellAt } from "@/lib/geo";
import { padRange } from "@/lib/scales";
import { extent } from "@/lib/stats";
import { useMediaQuery } from "@/lib/useSize";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { columnOf } from "./fields";

export function ProfilePanel({ day, compact = false }: { day: DayVolumes; compact?: boolean }) {
  const ctx = useRunContext();
  const { point, geom, depths, depthIndex, setDepthIndex, styleOf, labelOf, pointIsDefault, fieldMethods } = ctx;
  const [url, setUrl] = useUrlState();
  const wide = useMediaQuery("(min-width: 1181px)");
  const cell = point ? cellAt(geom, point.lat, point.lon) : null;
  const j = cell?.j ?? -1;
  const i = cell?.i ?? -1;
  const showAll = url.opts.pall !== "0";

  // the loaded estimates in the run's order; the day's primary estimate is drawn on top
  const methods = useMemo(
    () => (showAll ? fieldMethods.map((m) => m.key).filter((k) => day.estimates[k]) : [day.primary]),
    [fieldMethods, day, showAll],
  );

  const series = useMemo<ChartSeries[]>(() => {
    if (j < 0 || i < 0) return [];
    const mk = (key: string, label: string, column: (number | null)[], top = false): ChartSeries => {
      const style = styleOf(key);
      return { key, label, style: top ? { ...style, z: 20 } : style, points: depths.map((d, k) => ({ x: column[k] ?? null, y: d })) };
    };
    const out: ChartSeries[] = [mk("climatology", labelOf("climatology"), columnOf(day.climatology, j, i))];
    if (day.target) out.push(mk("glorys", labelOf("glorys"), columnOf(day.target, j, i)));
    for (const m of methods) {
      if (m !== day.primary) out.push(mk(m, labelOf(m), columnOf(day.estimates[m], j, i)));
    }
    out.push(mk(day.primary, labelOf(day.primary), columnOf(day.prediction, j, i), true));
    return out;
  }, [day, methods, j, i, depths, styleOf, labelOf]);

  const isOcean = series.some((s) => s.key === day.primary && s.points.some((p) => p.x != null));
  const domain = useMemo(() => {
    const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p.x)));
    return padRange(lo, hi, 0.06);
  }, [series]);

  // legend and table: estimates first (selected one leading), then the references
  const ordered = useMemo(() => [...series].reverse(), [series]);
  // differences are read against GLORYS; a day without a reanalysis (the live days) reads against the climatology
  const refKey = day.target ? "glorys" : "climatology";
  const glorys = series.find((s) => s.key === refKey)?.points[depthIndex]?.x ?? null;
  const canToggle = fieldMethods.length > 1;

  return (
    <Panel
      className="profile"
      title="Water column"
      subtitle={point ? `${fmtLatLon(point.lat, point.lon)}${pointIsDefault ? " · default point, click a map to move it" : ""}` : "Click a map to choose a point"}
      actions={
        canToggle && (
          <button
            type="button"
            className="toggle"
            aria-pressed={showAll}
            title="Draw every estimate of this run, or only the selected one, beside GLORYS and the climatology"
            onClick={() => setUrl({ opts: { pall: showAll ? "0" : null } })}
          >
            All estimates
          </button>
        )
      }
    >
      {!point ? (
        <Empty title="No point selected" height={280}>
          Click anywhere on a map to read the water column beneath it.
        </Empty>
      ) : !isOcean ? (
        <Empty title="This point is on land" height={280}>
          Choose an ocean cell to see its temperature profile.
        </Empty>
      ) : (
        <>
          <Legend items={ordered.map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
          <XYChart
            series={series}
            x={{ label: "Temperature (°C)", domain }}
            y={{ label: "Depth (m)", domain: [0, depths[depths.length - 1]], scale: "depth", ticks: depths }}
            hover="y"
            height={compact ? 380 : wide ? 470 : 360}
            reference={{ axis: "y", value: depths[depthIndex], label: fmtDepth(depths[depthIndex]) }}
            onPick={(d) => setDepthIndex(depths.indexOf(d))}
            hoverTitle={(d) => fmtDepth(d)}
            ariaLabel={`Temperature profile at ${fmtLatLon(point.lat, point.lon)}: ${ordered.map((s) => s.label).join(", ")} against depth`}
          />
          <table className="table table--dense readtable">
            <caption>
              At {fmtDepth(depths[depthIndex])}, °C
            </caption>
            <thead>
              <tr>
                <th scope="col">
                  <span className="visually-hidden">Series</span>
                </th>
                <th scope="col" className="num right">
                  Temperature
                </th>
                <th scope="col" className="num right">
                  − {labelOf(refKey)}
                </th>
              </tr>
            </thead>
            <tbody>
              {ordered.map((s) => {
                const v = s.points[depthIndex]?.x ?? null;
                return (
                  <tr key={s.key} className={s.key === day.primary ? "is-lead" : undefined}>
                    <th scope="row">
                      <span className="methodcell">
                        <Swatch style={s.style} width={22} />
                        {s.label}
                      </span>
                    </th>
                    <td className="num right">{fmt(v)}</td>
                    <td className="num right">{s.key === refKey ? "" : v != null && glorys != null ? fmtSigned(v - glorys) : fmt(null)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}
