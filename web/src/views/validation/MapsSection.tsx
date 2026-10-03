/** Where the error is: per-grid-point metric maps over the test period, one linked map per method. */
import { useMemo } from "react";
import { useErrorMap, useMapsIndex } from "@/api/queries";
import type { RasterLayer } from "@/components/map/MapCanvas";
import { MapFigure, MaskKey } from "@/components/map/MapFigure";
import { ZoomControls } from "@/components/map/ZoomControls";
import { ErrorState, Loading, Panel, QueryState, Select } from "@/components/ui/primitives";
import type { RangeInfo } from "@/api/types";
import { fmt, fmtDepth, prettyUnits } from "@/lib/format";
import { METRIC_ORDER, metricMeta } from "@/lib/metricMeta";
import { sortMethods } from "@/lib/methods";
import type { LinkedView } from "@/state/linkedView";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";

const METRIC_UNITLESS: Record<string, true> = { corr_anom: true, corr_raw: true, skill_vs_clim: true };

/** "9 % of the cells lie outside the colour range (lowest −3.04)": what the scale clips, from the API. */
export function outsideNote(info: RangeInfo, digits: number): string {
  const pct = info.fraction_outside * 100;
  const share = pct >= 0.5 ? `${fmt(pct, 0)}\u00A0%` : "under 1\u00A0%";
  const ends: string[] = [];
  if (info.n_below > 0 && info.data_min != null) ends.push(`lowest ${fmt(info.data_min, digits)}`);
  if (info.n_above > 0 && info.data_max != null) ends.push(`highest ${fmt(info.data_max, digits)}`);
  return `${share} of the cells lie beyond the colour scale${ends.length ? ` (${ends.join(", ")})` : ""}`;
}

function MethodMap({ metric, method, link }: { metric: string; method: string; link: LinkedView }) {
  const { run, geom, mask, depths, depthIndex, labelOf, detail, point, setPoint } = useRunContext();
  const q = useErrorMap(run.name, metric, method, depths[depthIndex]);
  const meta = metricMeta(metric);
  const raster = useMemo<RasterLayer | null>(() => {
    if (!q.data) return null;
    // the API's range is shared by every method at this depth; bias and skill come symmetric about zero
    return { data: q.data.data, nx: geom.nLon, ny: geom.nLat, vmin: q.data.range.vmin, vmax: q.data.range.vmax, cmap: meta.cmap, reversed: meta.cmapReversed };
  }, [q.data, geom, meta]);
  if (q.isError) return <ErrorState error={q.error} what={`The ${meta.label} map of ${labelOf(method)}`} height={180} />;
  if (!raster || !q.data) return <Loading height={220} />;
  // correlations and skill are dimensionless whatever the file's units attribute says
  const units = meta.key in METRIC_UNITLESS ? "" : meta.unit || prettyUnits(q.data.units);
  const info = q.data.rangeInfo;
  const outside = info?.exceeds_range ? outsideNote(info, meta.digits) : null;
  // pointed ends only where values really lie beyond the scale
  const extend = info ? (info.n_below > 0 && info.n_above > 0 ? "both" : info.n_above > 0 ? "max" : info.n_below > 0 ? "both" : "none") : meta.domain === "fromZero" ? "max" : "both";
  return (
    <MapFigure
      geom={geom}
      link={link}
      title={labelOf(method)}
      subtitle={`${meta.label} · ${fmtDepth(depths[depthIndex])}`}
      raster={raster}
      surfaceMask={mask?.levels[0] ?? null}
      levelMask={mask?.levels[depthIndex] ?? null}
      coast={mask?.coast ?? null}
      basins={detail.basins}
      marker={point}
      onPick={setPoint}
      pending={q.isPlaceholderData}
      digits={meta.digits}
      colorbar={{
        cmap: meta.cmap,
        reversed: meta.cmapReversed,
        vmin: raster.vmin,
        vmax: raster.vmax,
        units,
        label: meta.short,
        extend,
        signed: q.data.range.diverging,
      }}
      stats={outside}
      ariaLabel={`${meta.label} of ${labelOf(method)} at ${fmtDepth(depths[depthIndex])} over the test period`}
    />
  );
}

export function MapsSection() {
  const { run, geom, depths, depthIndex } = useRunContext();
  const [url, setUrl] = useUrlState();
  const link = useLinkedView(geom);
  const index = useMapsIndex(run.name, run.artefacts.maps);

  return (
    <QueryState query={index} what="The error maps" height={260}>
      {(idx) => {
        const available = [...METRIC_ORDER.filter((k) => idx.metrics[k]), ...Object.keys(idx.metrics).filter((k) => !(METRIC_ORDER as readonly string[]).includes(k))];
        const metric = available.includes(url.opts.metric) ? url.opts.metric : available[0];
        if (!metric) return <ErrorState error={new Error("maps_glorys.nc has no metric variables")} what="The error maps" />;
        // the climatology's skill against itself is zero everywhere: a blank map says nothing
        const stored = sortMethods(idx.metrics[metric]);
        const methods = metric === "skill_vs_clim" ? stored.filter((m) => m !== "climatology") : stored;
        const meta = metricMeta(metric);
        const nAll = idx.metrics.rmse?.length ?? 0;
        return (
          <Panel
            title={`${meta.label} at ${fmtDepth(depths[depthIndex])}, at every grid point`}
            subtitle={`over the whole test period at each cell · ${meta.explain} One colour scale for all methods.`}
            actions={
              <>
                <Select
                  label="Metric"
                  value={metric}
                  onChange={(v) => setUrl({ opts: { metric: v } })}
                  options={available.map((k) => ({ value: k, label: metricMeta(k).label }))}
                />
                <ZoomControls link={link} />
              </>
            }
          >
            <div className={`mapgrid mapgrid--${methods.length === 3 ? 3 : methods.length === 1 ? 1 : 2}`}>
              {methods.map((m) => (
                <MethodMap key={m} metric={metric} method={m} link={link} />
              ))}
            </div>
            <div className="gap-top-sm">
              <MaskKey />
            </div>
            {methods.length < nAll && (
              <p className="caption gap-top-sm">
                {metric === "skill_vs_clim"
                  ? "Climatology is the reference of this score: its own skill is zero everywhere, so it has no map. Skill is at most 1 and unbounded below; the scale is symmetric about zero and clips the most negative cells, as noted under each map."
                  : metric === "corr_anom"
                    ? "Climatology has no anomaly correlation: its anomaly is zero by definition."
                    : `${meta.label} maps are stored for ${methods.length === 1 ? "the main model only" : "these methods only"}; the per-depth profiles above cover every method.`}
              </p>
            )}
          </Panel>
        );
      }}
    </QueryState>
  );
}
