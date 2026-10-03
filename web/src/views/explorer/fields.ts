/** Field derivations for the explorer: slices, anomalies, differences and their colour ranges. */
import { levelOf, type Volume } from "@/api/binary";
import type { ColourRange, RangesResponse } from "@/api/types";
import type { DayVolumes } from "@/api/volumes";
import type { ColorbarSpec } from "@/components/map/Colorbar";
import type { RasterLayer } from "@/components/map/MapCanvas";
import type { ColormapName } from "@/lib/colormaps";
import { fieldStats, quantile, subtract, symmetricLimit, type FieldStats } from "@/lib/stats";

export type Quantity = "temp" | "anom";

export interface LevelFields {
  /** reconstruction, target (or null), difference (or null) for the chosen quantity */
  recon: Float32Array;
  target: Float32Array | null;
  diff: Float32Array | null;
  diffStats: FieldStats | null;
  reconStats: FieldStats;
}

/** Fields of one level for one estimate of the day (default: the day's primary estimate). */
export function levelFields(day: DayVolumes, depthIndex: number, quantity: Quantity, estimate: Volume = day.prediction): LevelFields {
  const pred = levelOf(estimate, depthIndex);
  const targ = day.target ? levelOf(day.target, depthIndex) : null;
  const diff = targ ? subtract(pred, targ) : null;
  if (quantity === "temp") {
    return { recon: pred, target: targ, diff, diffStats: diff ? fieldStats(diff) : null, reconStats: fieldStats(pred) };
  }
  const clim = levelOf(day.climatology, depthIndex);
  const recon = subtract(pred, clim);
  return {
    recon,
    target: targ ? subtract(targ, clim) : null,
    diff,
    diffStats: diff ? fieldStats(diff) : null,
    reconStats: fieldStats(recon),
  };
}

export interface Ranges {
  /** shared by reconstruction and target */
  main: [number, number];
  /** symmetric limit of the difference panel */
  diff: number;
}

/**
 * Colour ranges of one level, following the API's conventions (percentile-clipped, shared). The
 * temperature range is the API's hint, which is computed from the main reconstruction and GLORYS
 * and is therefore the same for every method.
 */
export function levelRanges(day: DayVolumes, depthIndex: number, quantity: Quantity, f: LevelFields): Ranges {
  const diff = f.diff ? symmetricLimit(f.diff, 0.99) : 1;
  if (quantity === "temp") {
    const hint = day.prediction.rangePerDepth?.[depthIndex] ?? day.prediction.range;
    if (hint && hint.vmax > hint.vmin) return { main: [hint.vmin, hint.vmax], diff };
    return { main: [quantile(f.recon, 0.01), quantile(f.recon, 0.99)], diff };
  }
  const a = symmetricLimit(f.recon, 0.99);
  const b = f.target ? symmetricLimit(f.target, 0.99) : 0;
  const lim = Math.max(a, b);
  return { main: [-lim, lim], diff };
}

/**
 * One set of ranges for several estimates shown side by side: the larger limit wins, so the same
 * colour means the same value in every panel (the API's rule for comparing methods).
 */
export function sharedRanges(list: readonly Ranges[]): Ranges {
  if (list.length === 0) return { main: [0, 1], diff: 1 };
  let lo = Infinity;
  let hi = -Infinity;
  let diff = 0;
  for (const r of list) {
    if (r.main[0] < lo) lo = r.main[0];
    if (r.main[1] > hi) hi = r.main[1];
    if (r.diff > diff) diff = r.diff;
  }
  return { main: [lo, hi], diff: diff > 0 ? diff : 1 };
}

/** Period-wide limits of one depth for the shown methods (any part may be unknown). */
export interface PeriodRanges {
  temperature: [number, number] | null;
  /** symmetric limit shared by the anomaly of the estimate and of GLORYS */
  anomaly: number | null;
  difference: number | null;
}

/**
 * Period-wide limits at one depth from the API's `/ranges`, for the methods on screen: the larger
 * symmetric limit wins, so side-by-side panels stay comparable.
 */
export function periodRanges(ranges: RangesResponse | undefined, depthIndex: number, methods: readonly string[]): PeriodRanges | null {
  if (!ranges) return null;
  const t = ranges.temperature?.[depthIndex];
  const lim = (pick: (m: RangesResponse["methods"][string]) => ColourRange | null | undefined): number | null => {
    let out: number | null = null;
    for (const m of methods) {
      const r = ranges.methods?.[m];
      const v = r ? pick(r)?.vmax : null;
      if (v != null && Number.isFinite(v) && v > 0 && (out == null || v > out)) out = v;
    }
    return out;
  };
  return {
    temperature: t && t.vmax > t.vmin ? [t.vmin, t.vmax] : null,
    anomaly: lim((m) => m.anomaly?.[depthIndex]),
    difference: lim((m) => m.difference?.[depthIndex]),
  };
}

/**
 * "Hold range": one scale for every day. The period-wide limits are used where the API knows
 * them; where it does not, the limits keep the widest value met so far (they only widen).
 */
export function heldRanges(current: Ranges, kept: Ranges | undefined, period: PeriodRanges | null | undefined, quantity: Quantity): Ranges {
  let main: [number, number];
  if (quantity === "temp" && period?.temperature) main = period.temperature;
  else if (quantity === "anom" && period?.anomaly) main = [-period.anomaly, period.anomaly];
  else main = kept ? [Math.min(current.main[0], kept.main[0]), Math.max(current.main[1], kept.main[1])] : current.main;
  const diff = period?.difference ?? (kept ? Math.max(kept.diff, current.diff) : current.diff);
  return { main, diff };
}

export function raster(
  data: Float32Array,
  vol: Pick<Volume, "nx" | "ny">,
  vmin: number,
  vmax: number,
  cmap: ColormapName,
  reversed = false,
): RasterLayer {
  return { data, nx: vol.nx, ny: vol.ny, vmin, vmax, cmap, reversed };
}

export function tempBar(vmin: number, vmax: number): ColorbarSpec {
  return { cmap: "thermal", vmin, vmax, units: "°C", extend: "both" };
}

export function divergingBar(limit: number, units = "°C"): ColorbarSpec {
  return { cmap: "balance", vmin: -limit, vmax: limit, units, extend: "both", signed: true };
}

/** Mean of each depth level over the finite cells (for the depth rail's context profile). */
export function levelMeans(volume: Volume): number[] {
  const size = volume.nx * volume.ny;
  const out: number[] = [];
  for (let k = 0; k < volume.nz; k++) {
    let sum = 0;
    let n = 0;
    const base = k * size;
    for (let c = 0; c < size; c++) {
      const v = volume.data[base + c];
      if (v === v) {
        sum += v;
        n++;
      }
    }
    out.push(n > 0 ? sum / n : NaN);
  }
  return out;
}

/** Column of a volume at one cell: one value per depth (NaN below the sea floor). */
export function columnOf(volume: Volume | null, j: number, i: number): (number | null)[] {
  if (!volume) return [];
  const size = volume.nx * volume.ny;
  const out: (number | null)[] = [];
  for (let k = 0; k < volume.nz; k++) {
    const v = volume.data[k * size + j * volume.nx + i];
    out.push(v === v ? v : null);
  }
  return out;
}

/** Vertical section [depth][distance] through a volume along a row (zonal) or a column (meridional). */
export function sectionOf(volume: Volume | null, along: "zonal" | "meridional", j: number, i: number): (number | null)[][] {
  if (!volume) return [];
  const { nx, ny, nz, data } = volume;
  const size = nx * ny;
  const rows: (number | null)[][] = [];
  for (let k = 0; k < nz; k++) {
    const row: (number | null)[] = [];
    if (along === "zonal") {
      for (let x = 0; x < nx; x++) {
        const v = data[k * size + j * nx + x];
        row.push(v === v ? v : null);
      }
    } else {
      for (let y = 0; y < ny; y++) {
        const v = data[k * size + y * nx + i];
        row.push(v === v ? v : null);
      }
    }
    rows.push(row);
  }
  return rows;
}

/** a - b for [depth][x] matrices (null where either is null). */
export function matrixDiff(a: (number | null)[][], b: (number | null)[][]): (number | null)[][] {
  return a.map((row, k) =>
    row.map((v, x) => {
      const w = b[k]?.[x];
      return v == null || w == null ? null : v - w;
    }),
  );
}

export function matrixLimit(m: (number | null)[][], q = 0.99): number {
  const flat: number[] = [];
  for (const row of m) for (const v of row) if (v != null) flat.push(Math.abs(v));
  const lim = quantile(flat, q);
  return Number.isFinite(lim) && lim > 0 ? lim : 1;
}

/** Transpose [time][depth] rows into [depth][time] (null where a row is missing). */
export function byDepth(rows: ReadonlyArray<ReadonlyArray<number | null> | null | undefined> | null | undefined, nz: number): (number | null)[][] {
  const n = rows?.length ?? 0;
  const out: (number | null)[][] = Array.from({ length: nz }, () => new Array<number | null>(n).fill(null));
  rows?.forEach((row, t) => {
    for (let k = 0; k < nz; k++) out[k][t] = row?.[k] ?? null;
  });
  return out;
}

/** Symmetric limit shared by several matrices (99th percentile of |x| over all of them). */
export function sharedMatrixLimit(matrices: readonly (number | null)[][][], q = 0.99): number {
  const flat: number[] = [];
  for (const m of matrices) for (const row of m) for (const v of row) if (v != null) flat.push(Math.abs(v));
  const lim = quantile(flat, q);
  return Number.isFinite(lim) && lim > 0 ? lim : 1;
}

export interface TimeDepth {
  /** [depth][time] */
  estimate: (number | null)[][];
  target: (number | null)[][];
  error: (number | null)[][];
  hasTarget: boolean;
  /** symmetric limit of the error panel */
  errorLimit: number;
  /** symmetric limit shared by the two anomaly panels (anomaly mode only) */
  anomalyLimit: number;
  /** false when anomalies were asked for but the run has no climatology at the point */
  hasClimatology: boolean;
}

/**
 * The time-depth panels at a point from the API's [time][depth] rows. In anomaly mode the
 * climatology is removed from the estimate and from GLORYS; the error (estimate minus GLORYS) is
 * the same in both modes, because the climatology cancels.
 */
export function timeDepth(
  prediction: ReadonlyArray<ReadonlyArray<number | null>>,
  target: ReadonlyArray<ReadonlyArray<number | null>>,
  climatology: ReadonlyArray<ReadonlyArray<number | null>> | null | undefined,
  nz: number,
  anomaly: boolean,
): TimeDepth {
  const pred = byDepth(prediction, nz);
  const targ = byDepth(target, nz);
  const error = matrixDiff(pred, targ);
  const hasTarget = targ.some((row) => row.some((v) => v != null));
  const clim = climatology ? byDepth(climatology, nz) : null;
  const hasClimatology = !!clim && clim.some((row) => row.some((v) => v != null));
  const errorLimit = matrixLimit(error);
  if (!anomaly || !clim || !hasClimatology) {
    return { estimate: pred, target: targ, error, hasTarget, errorLimit, anomalyLimit: 1, hasClimatology };
  }
  const estimate = matrixDiff(pred, clim);
  const targetAnom = matrixDiff(targ, clim);
  return { estimate, target: targetAnom, error, hasTarget, errorLimit, anomalyLimit: sharedMatrixLimit(hasTarget ? [estimate, targetAnom] : [estimate]), hasClimatology };
}
