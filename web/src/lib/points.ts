/**
 * Observation points for the canvas maps (Argo profiles). Thousands of dots are drawn in a few
 * batches: the points are sorted once into colour bins, and each bin is one path with one fill,
 * so a redraw costs a handful of canvas state changes whatever the number of profiles. Bins are
 * drawn from the lowest value to the highest, so the largest errors end up on top.
 */
import { getLut, LUT_SIZE, type ColormapName } from "./colormaps";
import type { GridGeom } from "./geo";

export const POINT_BINS = 24;

export interface PointLayer {
  n: number;
  lat: Float32Array;
  lon: Float32Array;
  /** value that colours each point (NaN: no value, drawn in the neutral ink) */
  value: Float32Array | null;
  /** point indices sorted by colour bin; bin b is order[binStart[b] .. binStart[b + 1]) */
  order: Uint32Array;
  /** POINT_BINS + 2 offsets: bin 0 holds the points without a value */
  binStart: Uint32Array;
  /** CSS colour of each bin (index 0: the neutral) */
  colors: string[];
}

export interface PointColour {
  vmin: number;
  vmax: number;
  cmap: ColormapName;
  /** fraction of the colormap left out at its pale end, so the lowest values still read on the map */
  floor?: number;
}

/** Bin of a value: 0 for "no value", 1..POINT_BINS from vmin to vmax (clamped). */
export function pointBin(value: number, vmin: number, vmax: number): number {
  if (!Number.isFinite(value)) return 0;
  const span = vmax - vmin;
  const t = span > 0 ? (value - vmin) / span : 0.5;
  return 1 + Math.min(POINT_BINS - 1, Math.max(0, Math.floor(t * POINT_BINS)));
}

export function buildPointLayer(
  lat: ArrayLike<number>,
  lon: ArrayLike<number>,
  value: ArrayLike<number | null | undefined> | null,
  colour: PointColour | null,
  neutral: string,
): PointLayer {
  const n = Math.min(lat.length, lon.length);
  const lats = new Float32Array(n);
  const lons = new Float32Array(n);
  const vals = value && colour ? new Float32Array(n) : null;
  const bins = new Uint8Array(n);
  const counts = new Uint32Array(POINT_BINS + 2);
  for (let i = 0; i < n; i++) {
    lats[i] = lat[i];
    lons[i] = lon[i];
    let b = 0;
    if (vals && colour && value) {
      const v = value[i];
      vals[i] = v == null ? NaN : v;
      b = pointBin(vals[i], colour.vmin, colour.vmax);
    }
    bins[i] = b;
    counts[b + 1]++;
  }
  const binStart = new Uint32Array(POINT_BINS + 2);
  for (let b = 1; b < binStart.length; b++) binStart[b] = binStart[b - 1] + counts[b];
  const cursor = binStart.slice(0, POINT_BINS + 1);
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[cursor[bins[i]]++] = i;

  const colors = [neutral];
  if (colour) {
    const lut = getLut(colour.cmap);
    const floor = colour.floor ?? 0;
    for (let b = 0; b < POINT_BINS; b++) {
      const t = floor + (1 - floor) * ((b + 0.5) / POINT_BINS);
      const k = Math.min(LUT_SIZE - 1, Math.round(t * (LUT_SIZE - 1))) * 4;
      colors.push(`rgb(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]})`);
    }
  } else {
    for (let b = 0; b < POINT_BINS; b++) colors.push(neutral);
  }
  return { n, lat: lats, lon: lons, value: vals, order, binStart, colors };
}

/** Index of the point nearest to a screen position within `radius` pixels, or -1. */
export function nearestPoint(layer: PointLayer, X: (lon: number) => number, Y: (lat: number) => number, px: number, py: number, radius = 10): number {
  let best = -1;
  let bestD = radius * radius;
  for (let i = 0; i < layer.n; i++) {
    const dx = X(layer.lon[i]) - px;
    if (dx > radius || dx < -radius) continue;
    const dy = Y(layer.lat[i]) - py;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

export interface CellCount {
  n: number;
  /** mean of the finite values of the points in the cell (NaN without any) */
  mean: number;
}

/** Points per grid cell, for the hover read-out of a dense map: cell index -> count and mean value. */
export function countByCell(layer: PointLayer, g: GridGeom): Map<number, CellCount> {
  const sums = new Map<number, { n: number; sum: number; nv: number }>();
  for (let i = 0; i < layer.n; i++) {
    const lat = layer.lat[i];
    const lon = layer.lon[i];
    if (!(lat >= g.lat0 && lat <= g.lat1 && lon >= g.lon0 && lon <= g.lon1)) continue;
    const j = Math.min(g.nLat - 1, Math.floor((lat - g.lat0) / g.dLat));
    const c = Math.min(g.nLon - 1, Math.floor((lon - g.lon0) / g.dLon));
    const key = j * g.nLon + c;
    let e = sums.get(key);
    if (!e) {
      e = { n: 0, sum: 0, nv: 0 };
      sums.set(key, e);
    }
    e.n++;
    const v = layer.value ? layer.value[i] : NaN;
    if (v === v) {
      e.sum += v;
      e.nv++;
    }
  }
  const out = new Map<number, CellCount>();
  for (const [key, e] of sums) out.set(key, { n: e.n, mean: e.nv > 0 ? e.sum / e.nv : NaN });
  return out;
}
