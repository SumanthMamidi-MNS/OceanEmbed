/** Scale helpers shared by the SVG charts and the canvas heat maps. */

/**
 * Depth axis position in [0, 1] on a square-root scale: the upper ocean, where the
 * thermocline lives, gets most of the axis while 1000 m still fits. 0 = surface (top).
 */
export function depthFraction(depth: number, maxDepth: number): number {
  if (!(maxDepth > 0)) return 0;
  return Math.sqrt(Math.max(0, depth) / maxDepth);
}

/** Inverse of {@link depthFraction}. */
export function depthFromFraction(fraction: number, maxDepth: number): number {
  const f = Math.min(1, Math.max(0, fraction));
  return f * f * maxDepth;
}

/**
 * Edges of the bands that represent the depth levels on the square-root axis:
 * midpoints between neighbouring levels (in axis space), clamped to [0, 1]. Length n + 1.
 */
export function depthBandEdges(depths: readonly number[]): number[] {
  const n = depths.length;
  if (n === 0) return [];
  const max = depths[n - 1];
  const pos = depths.map((d) => depthFraction(d, max));
  const edges = new Array<number>(n + 1);
  edges[0] = 0;
  for (let k = 1; k < n; k++) edges[k] = (pos[k - 1] + pos[k]) / 2;
  edges[n] = 1;
  return edges;
}

/** Subset of the depth levels that can be labelled without collisions on a sqrt axis. */
export function depthTicks(depths: readonly number[], axisPx: number, minGapPx = 18): number[] {
  if (depths.length === 0) return [];
  const max = depths[depths.length - 1];
  const out: number[] = [];
  const taken: number[] = [];
  // the two ends first, then the roundest numbers, so the labels read 0, 50, 100, 200, 500, 1000
  const roundness = (d: number) => (d === 0 || d === max ? 0 : d % 500 === 0 ? 1 : d % 100 === 0 ? 2 : d % 50 === 0 ? 3 : d % 10 === 0 ? 4 : 5);
  const order = [...depths].sort((a, b) => roundness(a) - roundness(b) || b - a);
  for (const d of order) {
    const y = depthFraction(d, max) * axisPx;
    if (taken.every((c) => Math.abs(c - y) >= minGapPx)) {
      taken.push(y);
      out.push(d);
    }
  }
  return out.sort((a, b) => a - b);
}

/** Index of the level nearest to a depth in metres. */
export function nearestDepthIndex(depths: readonly number[], depth: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let k = 0; k < depths.length; k++) {
    const d = Math.abs(depths[k] - depth);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/** "Nice" tick values covering [lo, hi] (1-2-5 steps), about `count` of them. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
  const raw = (b - a) / Math.max(1, count);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const err = raw / pow;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * pow;
  const start = Math.ceil(a / step - 1e-9);
  const end = Math.floor(b / step + 1e-9);
  const ticks: number[] = [];
  for (let i = start; i <= end; i++) ticks.push(i === 0 ? 0 : Number((i * step).toPrecision(12)));
  return ticks;
}

/** Decimals needed to print ticks of a given step without losing information. */
export function tickDigits(ticks: readonly number[]): number {
  if (ticks.length < 2) return 1;
  const step = Math.abs(ticks[1] - ticks[0]);
  if (step >= 1) return 0;
  return Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
}

/** Pad a [lo, hi] range by a fraction of its span (and never return a zero span). */
export function padRange(lo: number, hi: number, frac = 0.05): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) {
    const d = Math.abs(lo) > 0 ? Math.abs(lo) * 0.1 : 1;
    return [lo - d, hi + d];
  }
  const pad = (hi - lo) * frac;
  return [lo - pad, hi + pad];
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
