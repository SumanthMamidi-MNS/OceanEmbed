/** Small numeric helpers over typed arrays; NaN means "no data" everywhere. */

/** q-th quantile (0..1) of the finite values, linear interpolation; NaN if there are none. */
export function quantile(values: ArrayLike<number>, q: number): number {
  const finite: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) finite.push(v);
  }
  if (finite.length === 0) return NaN;
  finite.sort((a, b) => a - b);
  const pos = Math.min(1, Math.max(0, q)) * (finite.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(finite.length - 1, lo + 1);
  return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo);
}

/** Symmetric limit for a diverging field: the q-th quantile of |x| (never 0). */
export function symmetricLimit(values: ArrayLike<number>, q = 0.99): number {
  const abs = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) abs[i] = Math.abs(values[i]);
  const lim = quantile(abs, q);
  return Number.isFinite(lim) && lim > 0 ? lim : 1;
}

export interface FieldStats {
  n: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  meanAbs: number;
}

export function fieldStats(values: ArrayLike<number>): FieldStats {
  let n = 0;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let sumAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== v) continue;
    n++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
    sumAbs += Math.abs(v);
  }
  if (n === 0) return { n: 0, min: NaN, max: NaN, mean: NaN, rms: NaN, meanAbs: NaN };
  return { n, min, max, mean: sum / n, rms: Math.sqrt(sumSq / n), meanAbs: sumAbs / n };
}

/** a - b element-wise (NaN where either is NaN). */
export function subtract(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  const res = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) res[i] = a[i] - b[i];
  return res;
}

/** sqrt(u^2 + v^2) element-wise. */
export function magnitude(u: Float32Array, v: Float32Array): Float32Array {
  const res = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) res[i] = Math.hypot(u[i], v[i]);
  return res;
}

/** Pearson correlation over the positions where both arrays are finite; NaN if fewer than 3. */
export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let n = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const x = a[i];
    const y = b[i];
    if (x !== x || y !== y) continue;
    n++;
    sa += x;
    sb += y;
    saa += x * x;
    sbb += y * y;
    sab += x * y;
  }
  if (n < 3) return NaN;
  const cov = sab - (sa * sb) / n;
  const va = saa - (sa * sa) / n;
  const vb = sbb - (sb * sb) / n;
  const den = Math.sqrt(va * vb);
  return den > 0 ? cov / den : NaN;
}

/**
 * NaN-aware block mean of a row-major (ny, nx) field onto (ny / fy, nx / fx) blocks.
 * A block is NaN when it has no finite cell.
 */
export function blockMean(field: Float32Array, nx: number, ny: number, fx: number, fy: number): Float32Array {
  const ox = Math.floor(nx / fx);
  const oy = Math.floor(ny / fy);
  const out = new Float32Array(ox * oy);
  for (let J = 0; J < oy; J++) {
    for (let I = 0; I < ox; I++) {
      let sum = 0;
      let n = 0;
      for (let dj = 0; dj < fy; dj++) {
        const row = (J * fy + dj) * nx;
        for (let di = 0; di < fx; di++) {
          const v = field[row + I * fx + di];
          if (v === v) {
            sum += v;
            n++;
          }
        }
      }
      out[J * ox + I] = n > 0 ? sum / n : NaN;
    }
  }
  return out;
}

/** Flatten a JSON matrix (null = NaN) into a row-major Float32Array. */
export function flatten2d(rows: ReadonlyArray<ReadonlyArray<number | null>>): Float32Array {
  const ny = rows.length;
  const nx = ny > 0 ? rows[0].length : 0;
  const out = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const row = rows[j];
    for (let i = 0; i < nx; i++) {
      const v = row[i];
      out[j * nx + i] = v == null ? NaN : v;
    }
  }
  return out;
}

export function mean(values: ArrayLike<number | null>): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v != null && Number.isFinite(v)) {
      s += v;
      n++;
    }
  }
  return n > 0 ? s / n : NaN;
}

export function extent(values: ArrayLike<number | null>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v == null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : [NaN, NaN];
}
