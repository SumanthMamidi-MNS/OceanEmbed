import { describe, expect, it } from "vitest";
import { makeGeom } from "./geo";
import { POINT_BINS, buildPointLayer, countByCell, nearestPoint, pointBin } from "./points";

describe("point layer", () => {
  it("bins values from the lowest to the highest, with a bin of its own for missing values", () => {
    expect(pointBin(NaN, 0, 2)).toBe(0);
    expect(pointBin(0, 0, 2)).toBe(1);
    expect(pointBin(1.99, 0, 2)).toBe(POINT_BINS);
    // beyond the range: clamped, never dropped
    expect(pointBin(9, 0, 2)).toBe(POINT_BINS);
    expect(pointBin(-1, 0, 2)).toBe(1);
  });

  it("orders the points by bin so the largest values are drawn last", () => {
    const layer = buildPointLayer([10, 11, 12, 13], [60, 61, 62, 63], [1.9, null, 0.1, 1.0], { vmin: 0, vmax: 2, cmap: "amp" }, "#000");
    expect(layer.n).toBe(4);
    // the point without a value first, then ascending value
    expect(Array.from(layer.order)).toEqual([1, 2, 3, 0]);
    expect(layer.binStart[0]).toBe(0);
    expect(layer.binStart[layer.binStart.length - 1]).toBe(4);
    expect(layer.colors).toHaveLength(POINT_BINS + 1);
    expect(layer.colors[0]).toBe("#000");
    expect(layer.colors[1]).not.toBe(layer.colors[POINT_BINS]);
  });

  it("keeps every point in one neutral batch when nothing colours them", () => {
    const layer = buildPointLayer([10, 11], [60, 61], null, null, "#123456");
    expect(layer.value).toBeNull();
    expect(layer.binStart[1]).toBe(2);
    expect(new Set(layer.colors)).toEqual(new Set(["#123456"]));
  });

  it("stays fast and complete with tens of thousands of points", () => {
    const n = 40000;
    const lat = Float32Array.from({ length: n }, (_, i) => 5 + ((i * 7) % 2500) / 100);
    const lon = Float32Array.from({ length: n }, (_, i) => 45 + ((i * 13) % 6000) / 100);
    const val = Float32Array.from({ length: n }, (_, i) => (i % 97) / 30);
    const t0 = performance.now();
    const layer = buildPointLayer(lat, lon, val, { vmin: 0, vmax: 3, cmap: "amp" }, "#000");
    const elapsed = performance.now() - t0;
    expect(layer.binStart[layer.binStart.length - 1]).toBe(n);
    expect(new Set(layer.order).size).toBe(n);
    expect(elapsed).toBeLessThan(500);
  });

  it("finds the point nearest to a screen position, within a radius", () => {
    const layer = buildPointLayer([10, 20], [60, 80], null, null, "#000");
    const X = (lon: number) => lon * 10;
    const Y = (lat: number) => 300 - lat * 10;
    expect(nearestPoint(layer, X, Y, 603, 198)).toBe(0);
    expect(nearestPoint(layer, X, Y, 795, 104)).toBe(1);
    expect(nearestPoint(layer, X, Y, 700, 150)).toBe(-1);
  });

  it("counts the points of each grid cell and averages their values", () => {
    const g = makeGeom(2, 2, [0, 1, 2], [10, 11, 12]);
    const layer = buildPointLayer([0.2, 0.4, 1.5, 9], [10.1, 10.9, 11.5, 11], [1, 3, null, 5], { vmin: 0, vmax: 4, cmap: "amp" }, "#000");
    const cells = countByCell(layer, g);
    expect(cells.get(0)).toEqual({ n: 2, mean: 2 });
    const upper = cells.get(1 * 2 + 1)!;
    expect(upper.n).toBe(1);
    expect(Number.isNaN(upper.mean)).toBe(true);
    // the point outside the grid is not counted
    expect([...cells.values()].reduce((a, c) => a + c.n, 0)).toBe(3);
  });
});
