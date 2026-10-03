/**
 * Colormaps. Control points are the cmocean maps (Thyng et al. 2016) sampled at 12 evenly spaced
 * positions, plus matplotlib's viridis at 16; they are interpolated to 256-entry lookup tables.
 * Perceptually uniform in lightness, colour-vision-deficiency friendly.
 */

export type ColormapName =
  | "thermal"
  | "haline"
  | "balance"
  | "speed"
  | "amp"
  | "deep"
  | "tempo"
  | "curl"
  | "viridis";

type RGB = readonly [number, number, number];

const STOPS: Record<ColormapName, readonly RGB[]> = {
  thermal: [
    [3, 35, 51], [13, 48, 100], [53, 50, 155], [93, 62, 153], [126, 77, 143], [158, 89, 135],
    [193, 100, 121], [225, 113, 97], [246, 139, 69], [251, 173, 60], [246, 211, 70], [231, 250, 90],
  ],
  haline: [
    [41, 24, 107], [42, 35, 160], [15, 71, 153], [18, 95, 142], [38, 116, 137], [53, 136, 136],
    [65, 157, 133], [81, 178, 124], [111, 198, 107], [160, 214, 91], [212, 225, 112], [253, 238, 153],
  ],
  balance: [
    [23, 28, 66], [41, 58, 143], [11, 102, 189], [69, 144, 185], [142, 181, 194], [210, 216, 219],
    [230, 210, 204], [213, 157, 137], [196, 101, 72], [172, 43, 36], [120, 14, 40], [60, 9, 17],
  ],
  speed: [
    [254, 252, 205], [239, 225, 156], [221, 201, 106], [194, 182, 59], [157, 167, 21], [116, 153, 5],
    [75, 138, 20], [35, 121, 36], [11, 100, 44], [18, 78, 43], [25, 56, 34], [23, 35, 18],
  ],
  amp: [
    [241, 236, 236], [230, 209, 203], [221, 182, 170], [213, 156, 137], [205, 129, 103], [196, 102, 73],
    [186, 74, 47], [172, 44, 36], [149, 19, 39], [120, 14, 40], [89, 13, 31], [60, 9, 17],
  ],
  deep: [
    [253, 253, 204], [206, 236, 179], [156, 219, 165], [111, 201, 163], [86, 177, 163], [76, 153, 160],
    [68, 130, 155], [62, 108, 150], [62, 82, 143], [64, 60, 115], [54, 43, 77], [39, 26, 44],
  ],
  tempo: [
    [254, 245, 244], [222, 224, 210], [189, 206, 181], [153, 189, 156], [110, 173, 138], [65, 157, 129],
    [25, 137, 125], [18, 116, 117], [25, 94, 106], [28, 72, 93], [25, 51, 80], [20, 29, 67],
  ],
  curl: [
    [20, 29, 67], [28, 72, 93], [18, 115, 117], [63, 156, 129], [153, 189, 156], [223, 225, 211],
    [241, 218, 206], [224, 160, 137], [203, 101, 99], [164, 54, 96], [111, 23, 91], [51, 13, 53],
  ],
  viridis: [
    [68, 1, 84], [72, 26, 108], [71, 47, 125], [65, 68, 135], [57, 86, 140], [49, 104, 142],
    [42, 120, 142], [35, 136, 142], [31, 152, 139], [34, 168, 132], [53, 183, 121], [84, 197, 104],
    [122, 209, 81], [165, 219, 54], [210, 226, 27], [253, 231, 37],
  ],
};

/** Diverging maps have an even number of stops, so their exact (lightest) centre is not sampled. */
const CENTRE: Partial<Record<ColormapName, RGB>> = {
  balance: [241, 236, 235],
  curl: [254, 246, 245],
};

export const DIVERGING: ReadonlySet<ColormapName> = new Set<ColormapName>(["balance", "curl"]);

export const LUT_SIZE = 256;

/** Colour of a colormap at t in [0, 1] (clamped), as RGB 0-255 floats. */
export function sampleColormap(name: ColormapName, t: number): [number, number, number] {
  const stops = STOPS[name];
  const centre = CENTRE[name];
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const n = stops.length - 1;
  const pos = x * n;
  const i = Math.min(n - 1, Math.floor(pos));
  let f = pos - i;
  let a = stops[i];
  let b = stops[i + 1];
  if (centre && i === n / 2 - 0.5) {
    // the middle interval is split at the true centre colour
    if (f < 0.5) {
      b = centre;
      f *= 2;
    } else {
      a = centre;
      f = (f - 0.5) * 2;
    }
  }
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const lutCache = new Map<string, Uint8ClampedArray>();

/** 256-entry RGBA lookup table (alpha 255). `reversed` flips the direction. */
export function getLut(name: ColormapName, reversed = false): Uint8ClampedArray {
  const key = `${name}:${reversed ? 1 : 0}`;
  const hit = lutCache.get(key);
  if (hit) return hit;
  const lut = new Uint8ClampedArray(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    const [r, g, b] = sampleColormap(name, reversed ? 1 - t : t);
    lut[i * 4] = Math.round(r);
    lut[i * 4 + 1] = Math.round(g);
    lut[i * 4 + 2] = Math.round(b);
    lut[i * 4 + 3] = 255;
  }
  lutCache.set(key, lut);
  return lut;
}

/** LUT index of a value for the range [vmin, vmax]; -1 for NaN (never drawn as data). */
export function lutIndex(value: number, vmin: number, vmax: number): number {
  if (value !== value) return -1;
  const span = vmax - vmin;
  if (!(span > 0)) return (LUT_SIZE - 1) >> 1;
  const t = (value - vmin) / span;
  return t <= 0 ? 0 : t >= 1 ? LUT_SIZE - 1 : Math.round(t * (LUT_SIZE - 1));
}

export function rgbCss(name: ColormapName, t: number, reversed = false): string {
  const [r, g, b] = sampleColormap(name, reversed ? 1 - t : t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

/** CSS linear-gradient for a colour bar. */
export function cssGradient(name: ColormapName, reversed = false, direction = "to right", steps = 24): string {
  const parts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    parts.push(`${rgbCss(name, t, reversed)} ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${direction}, ${parts.join(", ")})`;
}

/**
 * Colour a row-major (ny, nx) float field into RGBA pixels, north-up (row 0 of the field is the
 * southernmost latitude, so rows are flipped). NaN cells stay fully transparent.
 */
export function colorizeField(
  field: Float32Array,
  nx: number,
  ny: number,
  vmin: number,
  vmax: number,
  lut: Uint8ClampedArray,
  out: Uint8ClampedArray,
): void {
  const span = vmax - vmin;
  const scale = span > 0 ? (LUT_SIZE - 1) / span : 0;
  for (let j = 0; j < ny; j++) {
    const src = j * nx;
    let dst = (ny - 1 - j) * nx * 4;
    for (let i = 0; i < nx; i++, dst += 4) {
      const v = field[src + i];
      if (v !== v) {
        out[dst + 3] = 0;
        continue;
      }
      let k = span > 0 ? Math.round((v - vmin) * scale) : (LUT_SIZE - 1) >> 1;
      if (k < 0) k = 0;
      else if (k > LUT_SIZE - 1) k = LUT_SIZE - 1;
      k *= 4;
      out[dst] = lut[k];
      out[dst + 1] = lut[k + 1];
      out[dst + 2] = lut[k + 2];
      out[dst + 3] = 255;
    }
  }
}
