/**
 * Binary transport of fields (docs/api.md): the body is a raw little-endian float32 array in C
 * order with NaN preserved; shape and colour-range hints travel in `X-*` headers.
 */
import type { ColourRange } from "./types";

export interface Volume {
  /** [depth, lat, lon] (or [1, lat, lon] for a single level) */
  nz: number;
  ny: number;
  nx: number;
  data: Float32Array;
  kind: string;
  date: string;
  hasTarget: boolean;
  /** hint for the whole array */
  range: ColourRange;
  /** one hint per depth level (volumes only) */
  rangePerDepth: ColourRange[] | null;
}

const HOST_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** Decode little-endian float32 bytes into a Float32Array on any host. */
export function decodeFloat32LE(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`float32 payload has ${buffer.byteLength} bytes, not a multiple of 4`);
  }
  if (HOST_LITTLE_ENDIAN) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const out = new Float32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export function parseShape(header: string | null): number[] {
  if (!header) throw new Error("missing X-Shape header");
  const shape = header.split(",").map((s) => Number(s.trim()));
  if (shape.length < 2 || shape.length > 3 || shape.some((n) => !Number.isInteger(n) || n <= 0)) {
    throw new Error(`bad X-Shape header: ${header}`);
  }
  return shape;
}

export function parseRange(header: string | null, diverging: boolean): ColourRange {
  const parts = (header ?? "").split(",").map((s) => Number(s.trim()));
  if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n))) return { vmin: 0, vmax: 1, diverging };
  return { vmin: parts[0], vmax: parts[1], diverging };
}

export function parseRangePerDepth(header: string | null, diverging: boolean): ColourRange[] | null {
  if (!header) return null;
  try {
    const raw: unknown = JSON.parse(header);
    if (!Array.isArray(raw)) return null;
    return raw.map((pair) => {
      const [vmin, vmax] = pair as [number, number];
      return { vmin: Number(vmin), vmax: Number(vmax), diverging };
    });
  } catch {
    return null;
  }
}

/** Decode a `format=f32` response (headers + body) into a {@link Volume}. */
export function decodeVolume(headers: Headers, buffer: ArrayBuffer): Volume {
  const dtype = headers.get("X-Dtype");
  const order = headers.get("X-Byte-Order");
  if (dtype && dtype !== "float32") throw new Error(`unsupported dtype ${dtype}`);
  if (order && order !== "little") throw new Error(`unsupported byte order ${order}`);
  const shape = parseShape(headers.get("X-Shape"));
  const [nz, ny, nx] = shape.length === 3 ? shape : [1, shape[0], shape[1]];
  const data = decodeFloat32LE(buffer);
  if (data.length !== nz * ny * nx) {
    throw new Error(`payload has ${data.length} values, X-Shape says ${nz * ny * nx}`);
  }
  const diverging = headers.get("X-Color-Diverging") === "1";
  return {
    nz,
    ny,
    nx,
    data,
    kind: headers.get("X-Kind") ?? "",
    date: headers.get("X-Date") ?? "",
    hasTarget: headers.get("X-Has-Target") === "1",
    range: parseRange(headers.get("X-Color-Range"), diverging),
    rangePerDepth: parseRangePerDepth(headers.get("X-Color-Range-Per-Depth"), diverging),
  };
}

/** View (no copy) of one depth level of a volume: row-major (lat, lon). */
export function levelOf(volume: Volume, depthIndex: number): Float32Array {
  const size = volume.ny * volume.nx;
  const k = Math.min(volume.nz - 1, Math.max(0, depthIndex));
  return volume.data.subarray(k * size, (k + 1) * size);
}

/**
 * Decode the packbits ocean mask (docs/api.md): base64 of `np.packbits` (MSB first) of the
 * row-major mask of one depth; bit 1 = ocean. Returns one byte (0/1) per cell.
 */
export function decodePackedMask(base64: string, nCells: number): Uint8Array {
  const bin = atob(base64);
  const out = new Uint8Array(nCells);
  for (let c = 0; c < nCells; c++) {
    const byte = bin.charCodeAt(c >> 3);
    out[c] = (byte >> (7 - (c & 7))) & 1;
  }
  return out;
}
