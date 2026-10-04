/**
 * One day at one depth. Single estimate: the estimate, the GLORYS target and their difference on
 * three linked maps. Side by side: every estimate the run has, next to GLORYS, on one temperature
 * scale, and every difference on one symmetric scale.
 */
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import type { RangesResponse } from "@/api/types";
import type { DayVolumes } from "@/api/volumes";
import { Swatch } from "@/components/charts/marks";
import { Colorbar } from "@/components/map/Colorbar";
import { MapFigure, MaskKey } from "@/components/map/MapFigure";
import { Empty } from "@/components/ui/primitives";
import { fmt, fmtDepth, fmtSigned } from "@/lib/format";
import { estimateRole } from "@/lib/methods";
import type { LinkedView } from "@/state/linkedView";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { divergingBar, heldRanges, levelFields, levelRanges, periodRanges, raster, sharedRanges, tempBar, type Quantity, type Ranges } from "./fields";

type PanelKey = "recon" | "target" | "diff";

export interface SubsurfaceProps {
  day: DayVolumes;
  link: LinkedView;
  quantity: Quantity;
  holdRange: boolean;
  /** period-wide colour ranges of the run (hold range), once loaded */
  ranges: RangesResponse | undefined;
  showBasins: boolean;
  pending: boolean;
}

/** Ranges of the shown estimates at this depth, with "hold range" applied (limits only widen). */
function useRanges(props: SubsurfaceProps, methods: readonly string[], depthIndex: number) {
  const { day, quantity, holdRange, ranges: runRanges } = props;
  const fields = useMemo(
    () => methods.map((m) => ({ method: m, f: levelFields(day, depthIndex, quantity, day.estimates[m]) })),
    [day, depthIndex, quantity, methods],
  );
  const held = useRef(new Map<string, Ranges>());
  useEffect(() => {
    if (!holdRange) held.current.clear();
  }, [holdRange]);
  const period = useMemo(() => periodRanges(runRanges, depthIndex, methods), [runRanges, depthIndex, methods]);
  const ranges = useMemo(() => {
    const current = sharedRanges(fields.map(({ f }) => levelRanges(day, depthIndex, quantity, f)));
    if (!holdRange) return current;
    const key = `${quantity}:${depthIndex}:${methods.join(",")}`;
    const next = heldRanges(current, held.current.get(key), period, quantity);
    held.current.set(key, next);
    return next;
  }, [fields, day, depthIndex, quantity, holdRange, period, methods]);
  return { fields, ranges };
}

export function SubsurfaceMaps(props: SubsurfaceProps & { /** the depth rail, beside the main map so its scale reads against it */ rail: ReactNode }) {
  const { day, link, quantity, showBasins, pending, rail } = props;
  const ctx = useRunContext();
  const { depthIndex, depths, geom, mask, point, setPoint, detail, labelOf } = ctx;
  const [url, setUrl] = useUrlState();
  const focus: PanelKey = url.opts.focus === "target" || url.opts.focus === "diff" ? url.opts.focus : "recon";
  const along: "zonal" | "meridional" = url.opts.sec === "merid" ? "meridional" : "zonal";
  const sectionLine = useMemo(() => (point ? { along, value: along === "zonal" ? point.lat : point.lon } : null), [point, along]);

  // what is on screen is the loaded day's estimate (while another one loads, the title must not lie)
  const method = day.primary;
  const methods = useMemo(() => [method], [method]);
  const { fields: all, ranges } = useRanges(props, methods, depthIndex);
  const fields = all[0].f;

  const mainCmap = quantity === "temp" ? "thermal" : "balance";
  const reconRaster = useMemo(() => raster(fields.recon, day.prediction, ranges.main[0], ranges.main[1], mainCmap), [fields.recon, day.prediction, ranges, mainCmap]);
  const targetRaster = useMemo(
    () => (fields.target ? raster(fields.target, day.prediction, ranges.main[0], ranges.main[1], mainCmap) : null),
    [fields.target, day.prediction, ranges, mainCmap],
  );
  const diffRaster = useMemo(
    () => (fields.diff ? raster(fields.diff, day.prediction, -ranges.diff, ranges.diff, "balance") : null),
    [fields.diff, day.prediction, ranges],
  );

  const mainBar = quantity === "temp" ? tempBar(ranges.main[0], ranges.main[1]) : divergingBar(ranges.main[1]);
  const diffBar = divergingBar(ranges.diff);
  const depthText = fmtDepth(depths[depthIndex]);
  const res = detail.grid.resolution;

  const common = {
    geom,
    link,
    surfaceMask: mask?.levels[0] ?? null,
    levelMask: mask?.levels[depthIndex] ?? null,
    coast: mask?.coast ?? null,
    basins: showBasins ? detail.basins : null,
    marker: point,
    sectionLine,
    onPick: setPoint,
    wheel: "plain" as const,
    pending,
  } as const;

  const enlarge = (key: PanelKey) =>
    focus === key ? undefined : (
      <button type="button" className="linkbtn" onClick={() => setUrl({ opts: { focus: key === "recon" ? null : key } })}>
        Enlarge
      </button>
    );

  const others = (["recon", "target", "diff"] as const).filter((k) => k !== focus);
  const cls = (key: PanelKey) => `submaps__cell submaps__cell--${focus === key ? "main" : others[0] === key ? "a" : "b"}`;
  const size = (key: PanelKey) => (focus === key ? "lg" : "md");
  const readout = (key: PanelKey) => (focus === key ? "full" : "value");
  const anom = quantity === "anom";
  const rs = fields.reconStats;
  const isMain = method === "model";
  const name = labelOf(method);
  const title = isMain ? "Reconstruction" : name;

  return (
    <div className="submaps">
      <div className="submaps__rail">{rail}</div>
      <div className={cls("recon")}>
        <MapFigure
          {...common}
          size={size("recon")}
          readout={readout("recon")}
          title={anom ? `${title} anomaly` : title}
          subtitle={`${isMain ? "OceanEmbed, " : ""}${estimateRole(method)} · ${depthText}`}
          raster={reconRaster}
          colorbar={mainBar}
          actions={enlarge("recon")}
          stats={anom ? `mean ${fmtSigned(rs.mean)} · RMS ${fmt(rs.rms)} °C` : `mean ${fmt(rs.mean)} · min ${fmt(rs.min, 1)} · max ${fmt(rs.max, 1)} °C`}
          ariaLabel={`${name}: ${anom ? "temperature anomaly" : "temperature"} at ${depthText}`}
        />
      </div>
      <div className={cls("target")}>
        {targetRaster ? (
          <MapFigure
            {...common}
            size={size("target")}
            readout={readout("target")}
            title={anom ? "GLORYS anomaly" : "GLORYS target"}
            subtitle={`reanalysis on the ${res}° grid · ${depthText}`}
            raster={targetRaster}
            colorbar={mainBar}
            actions={enlarge("target")}
            ariaLabel={`GLORYS ${anom ? "temperature anomaly" : "temperature"} at ${depthText}`}
          />
        ) : (
          <Empty title="No GLORYS target for this day" height={160}>
            The reconstruction needs only the surface fields, so it exists for days the reanalysis does not cover.
          </Empty>
        )}
      </div>
      <div className={cls("diff")}>
        {diffRaster && fields.diffStats ? (
          <MapFigure
            {...common}
            size={size("diff")}
            readout={readout("diff")}
            title="Difference"
            subtitle={`${isMain ? "reconstruction" : name} − GLORYS · ${depthText}`}
            raster={diffRaster}
            colorbar={diffBar}
            actions={enlarge("diff")}
            stats={`RMSE ${fmt(fields.diffStats.rms)} · bias ${fmtSigned(fields.diffStats.mean)} · MAE ${fmt(fields.diffStats.meanAbs)} °C`}
            ariaLabel={`${name} minus GLORYS at ${depthText}`}
          />
        ) : (
          <Empty title="No difference without a target" height={160} />
        )}
      </div>
    </div>
  );
}

/** Every estimate of the run beside GLORYS: shared temperature scale, shared difference scale. */
export function CompareMaps(props: SubsurfaceProps & { methods: readonly string[] }) {
  const { day, link, quantity, showBasins, pending } = props;
  const ctx = useRunContext();
  const { depthIndex, depths, geom, mask, point, setPoint, detail, labelOf, styleOf } = ctx;
  // only the estimates the loaded day really has
  const methods = useMemo(() => props.methods.filter((m) => day.estimates[m]), [props.methods, day]);
  const { fields, ranges } = useRanges(props, methods, depthIndex);
  const anom = quantity === "anom";
  const mainCmap = anom ? "balance" : "thermal";
  const mainBar = anom ? divergingBar(ranges.main[1]) : tempBar(ranges.main[0], ranges.main[1]);
  const diffBar = divergingBar(ranges.diff);
  const depthText = fmtDepth(depths[depthIndex]);
  const target = fields[0]?.f.target ?? null;
  const hasDiff = fields.some(({ f }) => f.diff);

  const common = {
    geom,
    link,
    size: "sm" as const,
    surfaceMask: mask?.levels[0] ?? null,
    levelMask: mask?.levels[depthIndex] ?? null,
    coast: mask?.coast ?? null,
    basins: showBasins ? detail.basins : null,
    marker: point,
    onPick: setPoint,
    wheel: "plain" as const,
    pending,
    showColorbar: false,
  } as const;

  // the estimate with the lowest error of this day at this depth
  const best = fields.reduce<string | null>((acc, { method, f }) => {
    if (!f.diffStats) return acc;
    const cur = acc ? fields.find((x) => x.method === acc)?.f.diffStats?.rms : undefined;
    return cur == null || f.diffStats.rms < cur ? method : acc;
  }, null);
  // at most four maps in a row: five or more wrap into rows of three, so each stays readable
  const cols = { ["--n" as string]: methods.length + 1 > 4 ? 3 : methods.length + 1 };

  return (
    <div className="cmp">
      <div className="cmp__row">
        <div className="cmp__head">
          <p className="cmp__label">{anom ? "Anomaly from climatology" : "Temperature"} · {depthText}</p>
          <Colorbar spec={mainBar} />
        </div>
        <div className="cmp__grid" style={cols}>
          {fields.map(({ method, f }) => (
            <MapFigure
              key={method}
              {...common}
              title={labelOf(method)}
              subtitle={estimateRole(method)}
              raster={raster(f.recon, day.prediction, ranges.main[0], ranges.main[1], mainCmap)}
              colorbar={mainBar}
              ariaLabel={`${labelOf(method)}: ${anom ? "temperature anomaly" : "temperature"} at ${depthText}`}
            />
          ))}
          {target ? (
            <MapFigure
              {...common}
              title="GLORYS"
              subtitle="the target (reanalysis)"
              raster={raster(target, day.prediction, ranges.main[0], ranges.main[1], mainCmap)}
              colorbar={mainBar}
              ariaLabel={`GLORYS ${anom ? "temperature anomaly" : "temperature"} at ${depthText}`}
            />
          ) : (
            <Empty title="No GLORYS target for this day" height={120} />
          )}
        </div>
      </div>

      {hasDiff && (
        <div className="cmp__row">
          <div className="cmp__head">
            <p className="cmp__label">Estimate − GLORYS · {depthText}</p>
            <Colorbar spec={diffBar} />
          </div>
          <div className="cmp__grid" style={cols}>
            {fields.map(({ method, f }) =>
              f.diff && f.diffStats ? (
                <MapFigure
                  key={method}
                  {...common}
                  title={`${labelOf(method)} − GLORYS`}
                  raster={raster(f.diff, day.prediction, -ranges.diff, ranges.diff, "balance")}
                  colorbar={diffBar}
                  ariaLabel={`${labelOf(method)} minus GLORYS at ${depthText}`}
                />
              ) : null,
            )}
            <div className="cmp__table">
              <table className="table table--dense">
                <caption>Error of this day at {depthText}, °C</caption>
                <thead>
                  <tr>
                    <th scope="col">Estimate</th>
                    <th scope="col" className="num right">
                      RMSE
                    </th>
                    <th scope="col" className="num right">
                      Bias
                    </th>
                    <th scope="col" className="num right">
                      MAE
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {fields.map(({ method, f }) => (
                    <tr key={method}>
                      <th scope="row">
                        <span className="methodcell">
                          <Swatch style={styleOf(method)} width={22} />
                          {labelOf(method)}
                        </span>
                      </th>
                      <td className="num right">{method === best ? <span className="best">{fmt(f.diffStats?.rms)}</span> : fmt(f.diffStats?.rms)}</td>
                      <td className="num right">{fmtSigned(f.diffStats?.mean)}</td>
                      <td className="num right">{fmt(f.diffStats?.meanAbs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="caption">Over the ocean cells of this level. One colour scale per row, set by the largest limit among the estimates.</p>
            </div>
          </div>
        </div>
      )}
      <MaskKey />
    </div>
  );
}
