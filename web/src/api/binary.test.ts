import { describe, expect, it } from "vitest";
import { decodeFloat32LE, decodePackedMask, decodeVolume, levelOf, parseRange, parseRangePerDepth, parseShape } from "./binary";

/** Little-endian float32 bytes written explicitly, whatever the host byte order. */
function le(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function headers(h: Record<string, string>): Headers {
  return new Headers(h);
}

describe("binary float32 transport", () => {
  it("decodes little-endian float32 and preserves NaN", () => {
    const out = decodeFloat32LE(le([1.5, -2.25, NaN, 0]));
    expect(Array.from(out.slice(0, 2))).toEqual([1.5, -2.25]);
    expect(Number.isNaN(out[2])).toBe(true);
    expect(out[3]).toBe(0);
  });

  it("rejects a payload that is not a whole number of float32", () => {
    expect(() => decodeFloat32LE(new ArrayBuffer(7))).toThrow(/multiple of 4/);
  });

  it("parses shapes and refuses malformed ones", () => {
    expect(parseShape("15,100,240")).toEqual([15, 100, 240]);
    expect(parseShape(" 100, 240 ")).toEqual([100, 240]);
    expect(() => parseShape(null)).toThrow();
    expect(() => parseShape("15")).toThrow();
    expect(() => parseShape("a,b")).toThrow();
    expect(() => parseShape("0,10")).toThrow();
  });

  it("parses colour-range hints", () => {
    expect(parseRange("6.6955,30.0829", false)).toEqual({ vmin: 6.6955, vmax: 30.0829, diverging: false });
    expect(parseRange(null, true)).toEqual({ vmin: 0, vmax: 1, diverging: true });
    expect(parseRangePerDepth("[[1, 2], [3.5, 4]]", false)).toEqual([
      { vmin: 1, vmax: 2, diverging: false },
      { vmin: 3.5, vmax: 4, diverging: false },
    ]);
    expect(parseRangePerDepth("not json", false)).toBeNull();
    expect(parseRangePerDepth(null, false)).toBeNull();
  });

  it("decodes a volume response with its headers", () => {
    // 2 depths x 1 lat x 3 lon
    const vol = decodeVolume(
      headers({
        "X-Shape": "2,1,3",
        "X-Dtype": "float32",
        "X-Byte-Order": "little",
        "X-Kind": "prediction",
        "X-Date": "2024-03-16",
        "X-Color-Range": "1,6",
        "X-Color-Diverging": "0",
        "X-Color-Range-Per-Depth": "[[1,3],[4,6]]",
        "X-Has-Target": "1",
      }),
      le([1, 2, 3, 4, NaN, 6]),
    );
    expect([vol.nz, vol.ny, vol.nx]).toEqual([2, 1, 3]);
    expect(vol.kind).toBe("prediction");
    expect(vol.date).toBe("2024-03-16");
    expect(vol.hasTarget).toBe(true);
    expect(vol.range).toEqual({ vmin: 1, vmax: 6, diverging: false });
    expect(vol.rangePerDepth).toHaveLength(2);
    const deep = levelOf(vol, 1);
    expect(deep.length).toBe(3);
    expect(deep[0]).toBe(4);
    expect(Number.isNaN(deep[1])).toBe(true);
    // a level is a view on the volume, not a copy
    expect(deep.buffer).toBe(vol.data.buffer);
    // out-of-range indices are clamped
    expect(levelOf(vol, 99)[2]).toBe(6);
  });

  it("treats a single level as one depth", () => {
    const vol = decodeVolume(headers({ "X-Shape": "2,2", "X-Color-Diverging": "1", "X-Color-Range": "-2,2" }), le([1, 2, 3, 4]));
    expect([vol.nz, vol.ny, vol.nx]).toEqual([1, 2, 2]);
    expect(vol.range.diverging).toBe(true);
    expect(vol.rangePerDepth).toBeNull();
  });

  it("refuses a body that does not match the declared shape, dtype or byte order", () => {
    expect(() => decodeVolume(headers({ "X-Shape": "2,2" }), le([1, 2, 3]))).toThrow(/X-Shape/);
    expect(() => decodeVolume(headers({ "X-Shape": "1,1", "X-Dtype": "float64" }), le([1]))).toThrow(/dtype/);
    expect(() => decodeVolume(headers({ "X-Shape": "1,1", "X-Byte-Order": "big" }), le([1]))).toThrow(/byte order/);
  });

  it("decodes the packbits ocean mask MSB first", () => {
    // bits 1010 0000 | 1100 0000 -> cells 0 and 2, then 8 and 9
    const b64 = btoa(String.fromCharCode(0b10100000, 0b11000000));
    expect(Array.from(decodePackedMask(b64, 10))).toEqual([1, 0, 1, 0, 0, 0, 0, 0, 1, 1]);
  });
});
