/**
 * Through the selected point: a vertical section (zonal or meridional) of the selected day, and
 * the time-depth history of the column over every reconstructed day. Both show the selected
 * estimate, as temperature or as the anomaly from the climatology.
 */
import { useMemo, type ReactNode } from "react";
import { useTimeseries } from "@/api/queries";
import type { DayVolumes } from "@/api/volumes";
import { DepthHeatmap } from "@/components/charts/DepthHeatmap";
import { Colorbar } from "@/components/map/Colorbar";
import { Empty, Panel, QueryState, Segmented } from "@/components/ui/primitives";
import { toUtcMs } from "@/lib/dates";
import { fmt, fmtLat, fmtLon, fmtSigned } from "@/lib/format";
import { cellAt } from "@/lib/geo";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { divergingBar, matrixDiff, matrixLimit, sectionOf, sharedMatrixLimit, tempBar, timeDepth, type Quantity } from "./fields";

const fTemp = (v: number) => `${fmt(v)}\u00A0°C`;
const fDiff = (v: number) => `${fmtSigned(v)}\u00A0°C`;

export function PointPanels({ day, quantity, lead }: { day: DayVolumes; quantity: Quantity; /** an extra first panel (the water column, when the maps take the full width) */ lead?: ReactNode }) {
  const ctx = useRunContext();
  const { run, point, geom, depths, depthIndex, setDepthIndex, setPoint, setDateIndex, detail, dates, date, labelOf } = ctx;
  const [url, setUrl] = useUrlState();
  const along = url.opts.sec === "merid" ? "meridional" : "zonal";
  const cell = point ? cellAt(geom, point.lat, point.lon) : null;
  const j = cell?.j ?? -1;
  const i = cell?.i ?? -1;
  const method = day.primary;
  const name = method === "model" ? "Reconstruction" : labelOf(method);
  const anom = quantity === "anom";
  // the time-depth panel follows the quantity of the maps unless it was switched by itself
  const tdAnom = url.opts.td ? url.opts.td === "anom" : anom;

  const section = useMemo(() => {
    if (j < 0 || i < 0) return null;
    const pred = sectionOf(day.prediction, along, j, i);
    const targ = day.target ? sectionOf(day.target, along, j, i) : null;
    const diff = targ ? matrixDiff(pred, targ) : null;
    const limit = diff ? matrixLimit(diff) : 1;
    if (!anom) return { pred, targ, diff, limit, anomLimit: 1 };
    const clim = sectionOf(day.climatology, along, j, i);
    const predA = matrixDiff(pred, clim);
    const targA = targ ? matrixDiff(targ, clim) : null;
    return { pred: predA, targ: targA, diff, limit, anomLimit: sharedMatrixLimit(targA ? [predA, targA] : [predA]) };
  }, [day, along, j, i, anom]);

  const xValues = along === "zonal" ? detail.grid.lon_values : detail.grid.lat_values;
  const xKind = along === "zonal" ? "lon" : "lat";
  const tr = day.prediction.range;
  const markX = point ? (along === "zonal" ? point.lon : point.lat) : null;

  const pickSection = (xi: number, di: number) => {
    if (!point) return;
    if (along === "zonal") setPoint(point.lat, detail.grid.lon_values[xi]);
    else setPoint(detail.grid.lat_values[xi], point.lon);
    setDepthIndex(di);
  };

  const ts = useTimeseries(run.name, point?.lat ?? null, point?.lon ?? null, method, !!point);
  const hov = useMemo(() => {
    const d = ts.data;
    if (!d || !d.is_ocean) return null;
    const td = timeDepth(d.prediction, d.target, d.climatology, d.depths.length, tdAnom);
    return { ...td, ms: d.dates.map(toUtcMs), range: d.color_range, depths: d.depths, dates: d.dates };
  }, [ts.data, tdAnom]);

  if (!point) return lead ? <div className="pointpanels">{lead}</div> : null;
  const where = along === "zonal" ? `along ${fmtLat(point.lat)}` : `along ${fmtLon(point.lon)}`;
  const secMain = anom ? { cmap: "balance" as const, vmin: -(section?.anomLimit ?? 1), vmax: section?.anomLimit ?? 1, fmt: fDiff } : { cmap: "thermal" as const, vmin: tr.vmin, vmax: tr.vmax, fmt: fTemp };
  // anomalies were asked for but the run has no climatology at the point: fall back to temperature, and say so
  const tdShowsAnom = tdAnom && !!hov?.hasClimatology;
  const tdMain = tdShowsAnom
    ? { cmap: "balance" as const, vmin: -(hov?.anomalyLimit ?? 1), vmax: hov?.anomalyLimit ?? 1, fmt: fDiff }
    : { cmap: "thermal" as const, vmin: hov?.range.vmin ?? 0, vmax: hov?.range.vmax ?? 1, fmt: fTemp };
  const suffix = (on: boolean) => (on ? " anomaly" : "");

  return (
    <div className={`pointpanels ${lead ? "pointpanels--3" : ""}`}>
      {lead}
      <Panel
        title="Vertical section"
        subtitle={`${where}, through the selected point · click to move the point and the depth`}
        actions={
          <Segmented
            label="Section direction"
            size="sm"
            value={along}
            onChange={(v) => setUrl({ opts: { sec: v === "zonal" ? null : "merid" } })}
            options={[
              { value: "zonal", label: "Zonal (W–E)" },
              { value: "meridional", label: "Meridional (S–N)" },
            ]}
          />
        }
      >
        {!section ? (
          <Empty title="No section" height={200} />
        ) : (
          <div className="stack">
            <DepthHeatmap
              title={`${name}${suffix(anom)}`}
              values={section.pred}
              depths={depths}
              xValues={xValues}
              xKind={xKind}
              cmap={secMain.cmap}
              vmin={secMain.vmin}
              vmax={secMain.vmax}
              height={150}
              showXAxis={false}
              markX={markX}
              markDepth={depths[depthIndex]}
              onPick={pickSection}
              formatValue={secMain.fmt}
              ariaLabel={`${name}: temperature${suffix(anom)} section ${where}`}
            />
            {section.targ && (
              <DepthHeatmap
                title={`GLORYS${suffix(anom)}`}
                values={section.targ}
                depths={depths}
                xValues={xValues}
                xKind={xKind}
                cmap={secMain.cmap}
                vmin={secMain.vmin}
                vmax={secMain.vmax}
                height={150}
                showXAxis={false}
                markX={markX}
                markDepth={depths[depthIndex]}
                onPick={pickSection}
                formatValue={secMain.fmt}
                ariaLabel={`GLORYS temperature${suffix(anom)} section ${where}`}
              />
            )}
            {section.diff && (
              <DepthHeatmap
                title={`Error (${name.toLowerCase()} − GLORYS)`}
                values={section.diff}
                depths={depths}
                xValues={xValues}
                xKind={xKind}
                cmap="balance"
                vmin={-section.limit}
                vmax={section.limit}
                height={170}
                markX={markX}
                markDepth={depths[depthIndex]}
                onPick={pickSection}
                formatValue={fDiff}
                ariaLabel={`Temperature difference section ${where}`}
              />
            )}
            <div className="stack__bars">
              <Colorbar spec={anom ? { ...divergingBar(section.anomLimit), label: "Anomaly" } : tempBar(tr.vmin, tr.vmax)} />
              {section.diff && <Colorbar spec={{ ...divergingBar(section.limit), label: "Error" }} />}
            </div>
          </div>
        )}
      </Panel>

      <Panel
        title="Time–depth at the point"
        subtitle={`every reconstructed day at ${fmtLat(point.lat)}, ${fmtLon(point.lon)} · click to jump to a day and depth`}
        actions={
          <Segmented
            label="Quantity of the time–depth plot"
            size="sm"
            value={tdAnom ? "anom" : "temp"}
            onChange={(v) => setUrl({ opts: { td: v === (anom ? "anom" : "temp") ? null : v } })}
            options={[
              { value: "temp", label: "Temperature" },
              { value: "anom", label: "Anomaly", title: "Estimate and GLORYS minus the harmonic climatology at this point: the seasonal cycle is removed, events remain" },
            ]}
          />
        }
      >
        <QueryState query={ts} what="The time series at this point" height={420}>
          {() =>
            !hov ? (
              <Empty title="This point is on land" height={200} />
            ) : (
              <div className={`stack ${ts.isPlaceholderData ? "is-pending" : ""}`}>
                {(
                  [
                    [`${name}${suffix(tdShowsAnom)}`, hov.estimate, false],
                    ...(hov.hasTarget
                      ? ([
                          [`GLORYS${suffix(tdShowsAnom)}`, hov.target, false],
                          [`Error (${name.toLowerCase()} − GLORYS)`, hov.error, true],
                        ] as const)
                      : []),
                  ] as const
                ).map(([label, values, isDiff], idx, arr) => (
                  <DepthHeatmap
                    key={label}
                    title={label}
                    values={values}
                    depths={hov.depths}
                    xValues={hov.ms}
                    xKind="time"
                    cmap={isDiff ? "balance" : tdMain.cmap}
                    vmin={isDiff ? -hov.errorLimit : tdMain.vmin}
                    vmax={isDiff ? hov.errorLimit : tdMain.vmax}
                    height={idx === arr.length - 1 ? 170 : 150}
                    showXAxis={idx === arr.length - 1}
                    markX={date ? toUtcMs(date) : null}
                    markDepth={depths[depthIndex]}
                    onPick={(xi, di) => {
                      const target = dates.indexOf(hov.dates[xi]);
                      if (target >= 0) setDateIndex(target);
                      setDepthIndex(di);
                    }}
                    formatValue={isDiff ? fDiff : tdMain.fmt}
                    ariaLabel={`${label} over time and depth at the selected point`}
                  />
                ))}
                <div className="stack__bars">
                  <Colorbar spec={tdShowsAnom ? { ...divergingBar(hov.anomalyLimit), label: "Anomaly" } : tempBar(hov.range.vmin, hov.range.vmax)} />
                  {hov.hasTarget && <Colorbar spec={{ ...divergingBar(hov.errorLimit), label: "Error" }} />}
                </div>
                {tdAnom && !hov.hasClimatology && (
                  <p className="caption gap-top-sm">This run has no climatology at this point, so temperature is shown instead of the anomaly.</p>
                )}
              </div>
            )
          }
        </QueryState>
      </Panel>
    </div>
  );
}
