import { describe, expect, it } from "vitest";
import { LUT_SIZE, colorizeField, cssGradient, getLut, lutIndex, sampleColormap } from "./colormaps";

function luminance([r, g, b]: number[]): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

describe("colormaps", () => {
  it("builds 256-entry opaque lookup tables", () => {
    const lut = getLut("thermal");
    expect(lut.length).toBe(LUT_SIZE * 4);
    expect(lut[3]).toBe(255);
    expect([lut[0], lut[1], lut[2]]).toEqual([3, 35, 51]);
    expect([lut[255 * 4], lut[255 * 4 + 1], lut[255 * 4 + 2]]).toEqual([231, 250, 90]);
  });

  it("reverses a map", () => {
    const fwd = getLut("viridis");
    const rev = getLut("viridis", true);
    expect([rev[0], rev[1], rev[2]]).toEqual([fwd[255 * 4], fwd[255 * 4 + 1], fwd[255 * 4 + 2]]);
  });

  it("thermal gets lighter monotonically enough to read as a sequence", () => {
    const start = luminance(sampleColormap("thermal", 0));
    const mid = luminance(sampleColormap("thermal", 0.5));
    const end = luminance(sampleColormap("thermal", 1));
    expect(start).toBeLessThan(mid);
    expect(mid).toBeLessThan(end);
  });

  it("diverging maps are lightest exactly at the centre and dark at both ends", () => {
    for (const name of ["balance", "curl"] as const) {
      const centre = luminance(sampleColormap(name, 0.5));
      for (const t of [0, 0.25, 0.4, 0.6, 0.75, 1]) expect(luminance(sampleColormap(name, t))).toBeLessThan(centre);
    }
    // the two halves have different hues: negative is blue, positive is red
    const [r0, , b0] = sampleColormap("balance", 0.2);
    const [r1, , b1] = sampleColormap("balance", 0.8);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(b1);
  });

  it("clamps out-of-range values and maps NaN to no colour", () => {
    expect(lutIndex(-5, 0, 1)).toBe(0);
    expect(lutIndex(7, 0, 1)).toBe(LUT_SIZE - 1);
    expect(lutIndex(0.5, 0, 1)).toBe(128);
    expect(lutIndex(NaN, 0, 1)).toBe(-1);
    expect(sampleColormap("thermal", NaN)).toEqual([3, 35, 51]);
  });

  it("colours a field north-up and leaves NaN transparent", () => {
    // 2 columns x 2 rows; row 0 is the SOUTH row
    const field = new Float32Array([0, NaN, 1, 0.5]);
    const out = new Uint8ClampedArray(2 * 2 * 4);
    const lut = getLut("thermal");
    colorizeField(field, 2, 2, 0, 1, lut, out);
    // image row 0 (north) = field row 1 = [1, 0.5]
    expect([out[0], out[1], out[2], out[3]]).toEqual([231, 250, 90, 255]);
    expect(out[7]).toBe(255);
    // image row 1 (south) = field row 0 = [0, NaN]
    expect([out[8], out[9], out[10], out[11]]).toEqual([3, 35, 51, 255]);
    expect(out[15]).toBe(0);
  });

  it("a zero-width range paints the middle of the map instead of dividing by zero", () => {
    const out = new Uint8ClampedArray(4);
    colorizeField(new Float32Array([3]), 1, 1, 3, 3, getLut("viridis"), out);
    expect(out[3]).toBe(255);
    expect(lutIndex(3, 3, 3)).toBe(127);
  });

  it("makes a CSS gradient with a stop at both ends", () => {
    const g = cssGradient("amp", false, "to right", 4);
    expect(g.startsWith("linear-gradient(to right, rgb(241, 236, 236) 0.0%")).toBe(true);
    expect(g.endsWith("100.0%)")).toBe(true);
  });
});
