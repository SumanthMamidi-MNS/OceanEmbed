/**
 * Everything a view needs about the selected run, resolved once: grid, depth and day selection
 * (snapped to values that exist), ocean mask, default point and the caveats to disclose.
 */
import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";
import type { OceanMask } from "@/api/queries";
import type { DatesResponse, FieldMethod, MethodInfo, RunDetail, RunSummary } from "@/api/types";
import { MAIN_METHOD } from "@/api/volumes";
import { defaultDateIndex, nearestDateIndex } from "@/lib/dates";
import { makeGeom, nearestOceanCell, cellCentre, type GridGeom } from "@/lib/geo";
import { ablationKeysOf, canonicalMethod, methodStyle, shortLabel, type MethodStyle } from "@/lib/methods";
import { runCaveats, type RunCaveats } from "@/lib/narrative";
import { nearestDepthIndex } from "@/lib/scales";
import { useUrlState } from "./router";
import { isResearchView } from "./url";

/**
 * Which methods a view shows. The product views show the product, the seasonal climatology (the
 * user's no-skill reference) and the references it is scored against; the Research views show
 * every method of the run (baselines, ablations).
 */
export type MethodScope = "product" | "research";

const PRODUCT_METHODS: ReadonlySet<string> = new Set(["model", "climatology", "glorys", "obs"]);

/** The methods of `keys` that belong on a view of this scope (order kept). */
export function scopeMethods<T extends string>(keys: readonly T[], scope: MethodScope): T[] {
  return scope === "research" ? [...keys] : keys.filter((k) => PRODUCT_METHODS.has(canonicalMethod(k)));
}

export interface RunContextValue {
  runs: RunSummary[];
  run: RunSummary;
  detail: RunDetail;
  geom: GridGeom;
  depths: number[];
  depthIndex: number;
  depth: number;
  /** days with a reconstruction */
  dates: string[];
  dateIndex: number;
  date: string | null;
  targetDates: ReadonlySet<string>;
  embeddingDates: string[];
  mask: OceanMask | undefined;
  /** selected water column (from the URL, else a default ocean point once the mask is known) */
  point: { lat: number; lon: number } | null;
  pointIsDefault: boolean;
  caveats: RunCaveats;
  /** product views show the product against its references; Research views show every method */
  scope: MethodScope;
  /** the methods of a list that this view shows (see {@link scopeMethods}) */
  scoped: <T extends string>(keys: readonly T[]) => T[];
  /** methods whose day fields this view offers (`method=`), the main model first; only the product outside Research */
  fieldMethods: FieldMethod[];
  /** the estimate shown wherever a single method is shown (from the URL, else the main model) */
  estimate: string;
  ablations: string[];
  /** the pooled (thermocline) depth range of the metrics, metres */
  pooledRange: number[] | null;
  styleOf: (methodKey: string) => MethodStyle;
  labelOf: (methodKey: string, short?: boolean) => string;
  setDateIndex: (index: number) => void;
  setDepthIndex: (index: number) => void;
  setPoint: (lat: number, lon: number) => void;
  setEstimate: (methodKey: string) => void;
}

const Ctx = createContext<RunContextValue | null>(null);

export function useRunContext(): RunContextValue {
  const value = useContext(Ctx);
  if (!value) throw new Error("useRunContext must be used inside <RunProvider>");
  return value;
}

/** Depth shown first: the geometric centre of the pooled metrics range (the thermocline). */
export function defaultDepthIndex(depths: readonly number[], pooled: readonly number[] | null | undefined): number {
  if (depths.length === 0) return 0;
  if (pooled && pooled.length >= 2 && pooled[0] > 0 && pooled[1] > 0) {
    return nearestDepthIndex(depths, Math.sqrt(pooled[0] * pooled[1]));
  }
  return Math.floor(depths.length / 2);
}

export function RunProvider(props: {
  runs: RunSummary[];
  run: RunSummary;
  detail: RunDetail;
  dates: DatesResponse | undefined;
  mask: OceanMask | undefined;
  children: ReactNode;
}) {
  const { runs, run, detail, dates: datesRes, mask, children } = props;
  const [url, setUrl] = useUrlState();

  const geom = useMemo(
    () => makeGeom(detail.grid.n_lat, detail.grid.n_lon, detail.grid.lat_edges, detail.grid.lon_edges),
    [detail.grid],
  );
  const depths = detail.grid.depth_values;
  const pooledRange = detail.metrics_metadata?.pooled_depth_range_m ?? null;
  const depthIndex = url.depth != null ? nearestDepthIndex(depths, url.depth) : defaultDepthIndex(depths, pooledRange);

  const dates = datesRes?.dates ?? detail.prediction_dates;
  const targetDates = useMemo(() => new Set(datesRes?.target_dates ?? []), [datesRes]);
  const embeddingDates = useMemo(() => datesRes?.embedding_dates ?? [], [datesRes]);
  // first view of a run: the day nearest the middle of the period that has a target and an embedding
  const fallbackIndex = useMemo(() => defaultDateIndex(dates, [targetDates, new Set(embeddingDates)]), [dates, targetDates, embeddingDates]);
  // the live run opens on its latest day, and a day it does not have is not snapped to a neighbour
  const liveIndex = run.live ? (url.date && dates.includes(url.date) ? dates.indexOf(url.date) : dates.length - 1) : null;
  const dateIndex = dates.length === 0 ? -1 : liveIndex != null ? liveIndex : url.date ? nearestDateIndex(dates, url.date) : fallbackIndex;
  const date = dateIndex >= 0 ? dates[dateIndex] : null;

  const defaultPoint = useMemo(() => {
    if (!mask) return null;
    const box = detail.basins[0]?.box;
    const lat = box ? (box.lat_min + box.lat_max) / 2 : (geom.lat0 + geom.lat1) / 2;
    const lon = box ? (box.lon_min + box.lon_max) / 2 : (geom.lon0 + geom.lon1) / 2;
    // a column that reaches the deepest level, so the default profile is complete; else any ocean cell
    const cell = nearestOceanCell(mask.levels[mask.levels.length - 1], geom, lat, lon) ?? nearestOceanCell(mask.levels[0], geom, lat, lon);
    return cell ? cellCentre(geom, cell) : null;
  }, [mask, detail.basins, geom]);

  const urlPoint = useMemo(() => {
    if (url.lat == null || url.lon == null) return null;
    const inside = url.lat >= geom.lat0 && url.lat <= geom.lat1 && url.lon >= geom.lon0 && url.lon <= geom.lon1;
    return inside ? { lat: url.lat, lon: url.lon } : null;
  }, [url.lat, url.lon, geom]);

  const methods: MethodInfo[] = detail.methods;
  const ablations = useMemo(() => ablationKeysOf(methods.map((m) => m.key)), [methods]);
  const caveats = useMemo(() => runCaveats(run), [run]);
  const scope: MethodScope = isResearchView(url.view) ? "research" : "product";
  const scoped = useCallback(<T extends string>(keys: readonly T[]) => scopeMethods(keys, scope), [scope]);
  const fieldMethods = useMemo<FieldMethod[]>(() => {
    const listed = detail.field_methods ?? [];
    // outside the Research area only the product itself is offered as an estimate
    if (listed.length > 0) return scope === "research" ? listed : listed.filter((m) => m.key === MAIN_METHOD || m.kind === "model").slice(0, 1);
    // a run described by an older API: only the main product is known to exist
    return [{ key: MAIN_METHOD, label: methods.find((m) => m.key === MAIN_METHOD)?.label ?? "OceanEmbed", kind: "model", n_days: dates.length, first: dates[0] ?? null, last: dates[dates.length - 1] ?? null }];
  }, [detail.field_methods, methods, dates, scope]);

  const styleOf = useCallback((key: string) => methodStyle(key, ablations), [ablations]);
  const labelOf = useCallback(
    (key: string, short = true) => {
      const k = canonicalMethod(key);
      const info = methods.find((m) => m.key === k);
      return short ? shortLabel(k, info?.label) : (info?.label ?? shortLabel(k));
    },
    [methods],
  );

  const setDateIndex = useCallback(
    (index: number) => {
      const i = Math.min(dates.length - 1, Math.max(0, index));
      if (dates[i]) setUrl({ date: dates[i] });
    },
    [dates, setUrl],
  );
  const setDepthIndex = useCallback(
    (index: number) => {
      const k = Math.min(depths.length - 1, Math.max(0, index));
      setUrl({ depth: depths[k] });
    },
    [depths, setUrl],
  );
  const setPoint = useCallback((lat: number, lon: number) => setUrl({ lat, lon }), [setUrl]);
  const estimate = fieldMethods.some((m) => m.key === url.est) ? (url.est as string) : (fieldMethods[0]?.key ?? MAIN_METHOD);
  const setEstimate = useCallback((key: string) => setUrl({ est: key === MAIN_METHOD ? null : key }), [setUrl]);

  const value: RunContextValue = {
    runs,
    run,
    detail,
    geom,
    depths,
    depthIndex,
    depth: depths[depthIndex],
    dates,
    dateIndex,
    date,
    targetDates,
    embeddingDates,
    mask,
    point: urlPoint ?? defaultPoint,
    pointIsDefault: !urlPoint,
    caveats,
    scope,
    scoped,
    fieldMethods,
    estimate,
    ablations,
    pooledRange,
    styleOf,
    labelOf,
    setDateIndex,
    setDepthIndex,
    setPoint,
    setEstimate,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
