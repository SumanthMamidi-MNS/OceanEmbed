/**
 * Grid geometry. Fields are row-major (lat, lon) with row 0 at the southern edge. Maps use an
 * equirectangular (plate carrée) projection: one degree of longitude and of latitude are the same
 * length on screen, so the true aspect of the domain is (lon span) : (lat span).
 */

export interface GridGeom {
  nLat: number;
  nLon: number;
  /** southern / western edge of the domain, degrees */
  lat0: number;
  lon0: number;
  /** northern / eastern edge */
  lat1: number;
  lon1: number;
  dLat: number;
  dLon: number;
}

export function makeGeom(nLat: number, nLon: number, latEdges: readonly number[], lonEdges: readonly number[]): GridGeom {
  const [lat0, lat1] = [latEdges[0], latEdges[latEdges.length - 1]];
  const [lon0, lon1] = [lonEdges[0], lonEdges[lonEdges.length - 1]];
  return { nLat, nLon, lat0, lat1, lon0, lon1, dLat: (lat1 - lat0) / nLat, dLon: (lon1 - lon0) / nLon };
}

/** Geometry of a regular grid given its cell centres (used for the coarse embedding grid). */
export function geomFromCentres(lat: readonly number[], lon: readonly number[]): GridGeom {
  const dLat = lat.length > 1 ? lat[1] - lat[0] : 1;
  const dLon = lon.length > 1 ? lon[1] - lon[0] : 1;
  return {
    nLat: lat.length,
    nLon: lon.length,
    lat0: lat[0] - dLat / 2,
    lat1: lat[lat.length - 1] + dLat / 2,
    lon0: lon[0] - dLon / 2,
    lon1: lon[lon.length - 1] + dLon / 2,
    dLat,
    dLon,
  };
}

/** Width / height of the domain with square degrees. */
export function aspectRatio(g: GridGeom): number {
  return (g.lon1 - g.lon0) / (g.lat1 - g.lat0);
}

export interface Cell {
  /** latitude index (0 = south) */
  j: number;
  /** longitude index (0 = west) */
  i: number;
}

/** Cell containing a point, or null outside the domain. */
export function cellAt(g: GridGeom, lat: number, lon: number): Cell | null {
  if (!(lat >= g.lat0 && lat <= g.lat1 && lon >= g.lon0 && lon <= g.lon1)) return null;
  const j = Math.min(g.nLat - 1, Math.floor((lat - g.lat0) / g.dLat));
  const i = Math.min(g.nLon - 1, Math.floor((lon - g.lon0) / g.dLon));
  return { j, i };
}

export function cellCentre(g: GridGeom, cell: Cell): { lat: number; lon: number } {
  return { lat: g.lat0 + (cell.j + 0.5) * g.dLat, lon: g.lon0 + (cell.i + 0.5) * g.dLon };
}

/**
 * Coastline derived from an ocean mask: every cell edge that separates an ocean cell from a land
 * cell (the domain border is not a coast). Returned as [x0, y0, x1, y1] quadruples in cell-corner
 * units (x: 0..nLon west->east, y: 0..nLat south->north).
 */
export function coastlineSegments(mask: Uint8Array, nLon: number, nLat: number): Float32Array {
  const seg: number[] = [];
  for (let j = 0; j < nLat; j++) {
    for (let i = 0; i < nLon; i++) {
      const o = mask[j * nLon + i];
      if (i + 1 < nLon && mask[j * nLon + i + 1] !== o) seg.push(i + 1, j, i + 1, j + 1);
      if (j + 1 < nLat && mask[(j + 1) * nLon + i] !== o) seg.push(i, j + 1, i + 1, j + 1);
    }
  }
  return new Float32Array(seg);
}

/** Nearest ocean cell to a point (searching outward), or null if the mask has no ocean. */
export function nearestOceanCell(mask: Uint8Array, g: GridGeom, lat: number, lon: number): Cell | null {
  const start = cellAt(g, clampTo(lat, g.lat0, g.lat1), clampTo(lon, g.lon0, g.lon1));
  if (!start) return null;
  if (mask[start.j * g.nLon + start.i]) return start;
  let best: Cell | null = null;
  let bestD = Infinity;
  for (let j = 0; j < g.nLat; j++) {
    for (let i = 0; i < g.nLon; i++) {
      if (!mask[j * g.nLon + i]) continue;
      const d = (j - start.j) ** 2 + (i - start.i) ** 2;
      if (d < bestD) {
        bestD = d;
        best = { j, i };
      }
    }
  }
  return best;
}

function clampTo(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Graticule line positions: multiples of `step` strictly inside (lo, hi). */
export function graticule(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  const first = Math.ceil(lo / step - 1e-9) * step;
  for (let v = first; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** Graticule step (degrees) that gives roughly `target` lines over a span. */
export function graticuleStep(spanDeg: number, target = 6): number {
  const steps = [0.5, 1, 2, 5, 10, 20, 30];
  for (const s of steps) if (spanDeg / s <= target) return s;
  return 30;
}
