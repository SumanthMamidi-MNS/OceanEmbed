/** Helpers for the views of the satellite embedding: its principal components and its similarity maps. */
import type { EmbeddingResponse, EmbeddingSimilarityResponse } from "@/api/types";
import { geomFromCentres, type GridGeom } from "./geo";
import { blockMean, pearson } from "./stats";

export interface EmbeddingMap {
  nx: number;
  ny: number;
  geom: GridGeom;
  /** every served principal component as a row-major (ny, nx) array in 0..1, NaN over land; row 0 = south */
  pcs: Float32Array[];
  /** north-up RGBA, land transparent: R, G, B = PC1, PC2, PC3 */
  rgba: Uint8ClampedArray;
  ocean: Uint8Array;
  /** fraction of the embedding's variance carried by each component */
  explained: number[];
}

export function decodeEmbedding(res: EmbeddingResponse): EmbeddingMap {
  const [n, ny, nx] = res.shape;
  const pcs = Array.from({ length: n }, () => new Float32Array(nx * ny));
  const ocean = new Uint8Array(nx * ny);
  const rgba = new Uint8ClampedArray(nx * ny * 4);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      const isOcean = res.ocean[j]?.[i] === 1;
      ocean[idx] = isOcean ? 1 : 0;
      const dst = ((ny - 1 - j) * nx + i) * 4;
      let any = false;
      for (let c = 0; c < n; c++) {
        const v = res.data[c]?.[j]?.[i];
        const ok = isOcean && v != null && Number.isFinite(v);
        pcs[c][idx] = ok ? (v as number) : NaN;
        if (ok && c < 3) {
          rgba[dst + c] = Math.round(Math.min(1, Math.max(0, v as number)) * 255);
          any = true;
        }
      }
      rgba[dst + 3] = any ? 255 : 0;
    }
  }
  return { nx, ny, geom: geomFromCentres(res.lat, res.lon), pcs, rgba, ocean, explained: res.explained_variance_ratio };
}

/** Running total of the explained-variance fractions: cumulative[k] = share carried by components 1..k+1. */
export function cumulative(fractions: readonly number[]): number[] {
  let sum = 0;
  return fractions.map((f) => (sum += f));
}

export interface SimilarityMap {
  nx: number;
  ny: number;
  /** cosine similarity to the reference cell over all embedding features (-1..1), NaN over land */
  data: Float32Array;
  /** the reference cell (row from the south, column from the west) and its centre */
  j: number;
  i: number;
  lat: number;
  lon: number;
  embDim: number;
  centered: boolean;
  /** false when the reference cell is land (the map then says nothing) */
  referenceIsOcean: boolean;
  /** range over the other ocean cells */
  min: number;
  max: number;
  /** share of the other ocean cells whose similarity is above 0.5 */
  alike: number;
}

/** Decode `/embeddings/similar`: land cells are null in the payload; the range over the other ocean cells is the API's. */
export function decodeSimilarity(res: EmbeddingSimilarityResponse): SimilarityMap {
  const [ny, nx] = res.shape;
  const data = new Float32Array(nx * ny).fill(NaN);
  let n = 0;
  let alike = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = res.data[j]?.[i];
      if (v == null || !Number.isFinite(v)) continue;
      data[j * nx + i] = v;
      if (j === res.y_index && i === res.x_index) continue;
      n++;
      if (v > 0.5) alike++;
    }
  }
  const min = res.similarity_range?.min ?? NaN;
  const max = res.similarity_range?.max ?? NaN;
  return {
    nx,
    ny,
    data,
    j: res.y_index,
    i: res.x_index,
    lat: res.lat,
    lon: res.lon,
    embDim: res.emb_dim,
    centered: res.centered,
    referenceIsOcean: res.reference_is_ocean !== false,
    min,
    max,
    alike: n > 0 ? alike / n : NaN,
  };
}

/**
 * Pearson correlation between each principal component and each surface field, after averaging
 * the field onto the embedding grid. Rows: components, columns: fields (same order as `fields`).
 */
export function componentFieldCorrelation(
  map: EmbeddingMap,
  fields: readonly { data: Float32Array; nx: number; ny: number }[],
): number[][] {
  const coarse = fields.map((f) => {
    const fx = Math.max(1, Math.round(f.nx / map.nx));
    const fy = Math.max(1, Math.round(f.ny / map.ny));
    const m = blockMean(f.data, f.nx, f.ny, fx, fy);
    return m.length === map.nx * map.ny ? m : null;
  });
  return map.pcs.map((pc) => coarse.map((f) => (f ? pearson(pc, f) : NaN)));
}
