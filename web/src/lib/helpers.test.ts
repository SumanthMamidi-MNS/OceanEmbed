import { describe, expect, it } from "vitest";
import { daysInclusive, defaultDateIndex, fmtDate, fmtSpan, isIsoDate, nearestDateIndex } from "./dates";
import { componentFieldCorrelation, cumulative, decodeEmbedding, decodeSimilarity } from "./embedding";
import { fmt, fmtBytes, fmtCompact, fmtInt, fmtLat, fmtLon, fmtSigned, prettyText, prettyUnits } from "./format";
import { aspectRatio, cellAt, cellCentre, coastlineSegments, graticule, graticuleStep, makeGeom, nearestOceanCell } from "./geo";
import { ablationKeysOf, canonicalMethod, methodStyle, shortLabel, sortMethods } from "./methods";
import { metricDomain, metricMeta } from "./metricMeta";
import { depthBandEdges, depthFraction, depthFromFraction, depthTicks, nearestDepthIndex, niceTicks, tickDigits } from "./scales";
import { blockMean, fieldStats, flatten2d, pearson, quantile, symmetricLimit } from "./stats";

const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

describe("depth scale", () => {
  it("is a square-root axis with the surface at 0 and the deepest level at 1", () => {
    expect(depthFraction(0, 1000)).toBe(0);
    expect(depthFraction(1000, 1000)).toBe(1);
    expect(depthFraction(250, 1000)).toBeCloseTo(0.5);
    expect(depthFromFraction(0.5, 1000)).toBeCloseTo(250);
    // the upper 200 m get almost half of the axis
    expect(depthFraction(200, 1000)).toBeGreaterThan(0.44);
  });

  it("gives each level a band, increasing downward and covering the axis", () => {
    const e = depthBandEdges(DEPTHS);
    expect(e).toHaveLength(DEPTHS.length + 1);
    expect(e[0]).toBe(0);
    expect(e[e.length - 1]).toBe(1);
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThan(e[i - 1]);
  });

  it("labels both ends and prefers round depths when space is short", () => {
    const t = depthTicks(DEPTHS, 150, 18);
    expect(t[0]).toBe(0);
    expect(t[t.length - 1]).toBe(1000);
    expect(t).toContain(500);
    expect(t).toContain(100);
    expect(t).not.toContain(125);
    // with room, every level is labelled
    expect(depthTicks(DEPTHS, 4000, 10)).toEqual(DEPTHS);
  });

  it("snaps a depth to the nearest level", () => {
    expect(nearestDepthIndex(DEPTHS, 100)).toBe(7);
    expect(nearestDepthIndex(DEPTHS, 110)).toBe(7);
    expect(nearestDepthIndex(DEPTHS, 99999)).toBe(14);
  });
});

describe("ticks", () => {
  it("makes 1-2-5 ticks inside the range", () => {
    expect(niceTicks(0, 5, 5)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(niceTicks(-0.83, 0.83, 4)).toEqual([-0.5, 0, 0.5]);
    expect(niceTicks(21.2, 30.4, 5)).toEqual([22, 24, 26, 28, 30]);
    expect(niceTicks(3, 3)).toEqual([3]);
    expect(niceTicks(NaN, 1)).toEqual([]);
  });

  it("knows how many decimals the ticks need", () => {
    expect(tickDigits([0, 1, 2])).toBe(0);
    expect(tickDigits([0, 0.5, 1])).toBe(1);
    expect(tickDigits([0.2, 0.25, 0.3])).toBe(2);
  });
});

describe("formatting", () => {
  it("uses a real minus sign, fixed decimals and an en dash for missing values", () => {
    expect(fmt(-0.356, 2)).toBe("−0.36");
    expect(fmt(-0.001, 2)).toBe("0.00");
    expect(fmt(null)).toBe("–");
    expect(fmt(NaN)).toBe("–");
    expect(fmtSigned(0.26)).toBe("+0.26");
    expect(fmtSigned(-0.26)).toBe("−0.26");
    expect(fmtSigned(0)).toBe("0.00");
  });

  it("groups thousands and abbreviates large counts", () => {
    expect(fmtInt(12686)).toBe("12\u202F686");
    expect(fmtInt(900)).toBe("900");
    expect(fmtCompact(30769032)).toBe("30.8\u00A0M");
    expect(fmtCompact(12686)).toBe("13\u00A0k");
    expect(fmtBytes(13807902)).toBe("13.2\u00A0MB");
  });

  it("writes coordinates and units typographically", () => {
    expect(fmtLat(15.125)).toBe("15.13°N");
    expect(fmtLat(-5, 0)).toBe("5°S");
    expect(fmtLon(61.625)).toBe("61.63°E");
    expect(fmtLon(-30, 0)).toBe("30°W");
    expect(prettyUnits("degC")).toBe("°C");
    expect(prettyUnits("m s-1")).toBe("m/s");
    expect(prettyUnits("PSU")).toBe("PSU");
    expect(prettyText("0.25 deg, 5x5 block mean, 3 degC")).toBe("0.25°, 5×5 block mean, 3 °C");
  });
});

describe("dates", () => {
  it("validates ISO dates", () => {
    expect(isIsoDate("2024-03-16")).toBe(true);
    expect(isIsoDate("2024-02-30")).toBe(false);
    expect(isIsoDate("16/03/2024")).toBe(false);
    expect(isIsoDate(null)).toBe(false);
  });

  it("counts days inclusively", () => {
    expect(daysInclusive("2024-01-01", "2024-02-15")).toBe(46);
    expect(daysInclusive("2023-07-01", "2023-12-31")).toBe(184);
    expect(daysInclusive("2024-03-02", "2024-03-01")).toBe(0);
    expect(daysInclusive(null, "2024-03-01")).toBe(0);
  });

  it("formats dates and spans", () => {
    expect(fmtDate("2024-03-16")).toBe("16\u00A0Mar\u00A02024");
    expect(fmtDate("2023-07-01T09:17:47Z")).toBe("1\u00A0Jul\u00A02023");
    expect(fmtSpan("2024-03-01", "2024-03-31")).toBe("1\u00A0Mar\u00A02024 – 31\u00A0Mar\u00A02024");
  });

  it("finds the nearest available day", () => {
    const d = ["2024-03-01", "2024-03-02", "2024-03-05"];
    expect(nearestDateIndex(d, "2024-03-02")).toBe(1);
    expect(nearestDateIndex(d, "2024-03-04")).toBe(2);
    expect(nearestDateIndex(d, "1999-01-01")).toBe(0);
    expect(nearestDateIndex(d, null)).toBe(0);
    expect(nearestDateIndex([], "2024-03-02")).toBe(-1);
  });
});

describe("default day", () => {
  const days = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"];
  it("is the middle of the period when nothing restricts it", () => {
    expect(defaultDateIndex(days)).toBe(2);
    expect(defaultDateIndex(days, [new Set()])).toBe(2);
    expect(defaultDateIndex([])).toBe(-1);
  });
  it("moves to the nearest day that has a target and an embedding", () => {
    expect(defaultDateIndex(days, [new Set(["2024-01-01", "2024-01-04"]), new Set(["2024-01-04", "2024-01-05"])])).toBe(3);
    expect(defaultDateIndex(days, [new Set(["2024-01-02"])])).toBe(1);
    // nothing satisfies every set: fall back to the middle
    expect(defaultDateIndex(days, [new Set(["2024-01-01"]), new Set(["2024-01-05"])])).toBe(2);
  });
});

describe("statistics", () => {
  it("computes quantiles ignoring NaN", () => {
    expect(quantile([1, NaN, 3, 2], 0.5)).toBe(2);
    expect(quantile([0, 10], 0.25)).toBe(2.5);
    expect(Number.isNaN(quantile([NaN], 0.5))).toBe(true);
  });

  it("gives a symmetric limit that is never zero", () => {
    expect(symmetricLimit(new Float32Array([-4, 1, 2, NaN]), 1)).toBe(4);
    expect(symmetricLimit(new Float32Array([0, 0]))).toBe(1);
  });

  it("summarises a field (rms of a difference is its RMSE, mean its bias)", () => {
    const s = fieldStats(new Float32Array([1, -1, NaN, 3]));
    expect(s.n).toBe(3);
    expect(s.mean).toBeCloseTo(1);
    expect(s.rms).toBeCloseTo(Math.sqrt(11 / 3));
    expect(s.meanAbs).toBeCloseTo(5 / 3);
    expect(fieldStats(new Float32Array([NaN])).n).toBe(0);
  });

  it("correlates and block-averages with NaN awareness", () => {
    expect(pearson([1, 2, 3, 4], [2, 4, 6, 8])).toBeCloseTo(1);
    expect(pearson([1, 2, 3, NaN], [3, 2, 1, 5])).toBeCloseTo(-1);
    expect(Number.isNaN(pearson([1, 2], [1, 2]))).toBe(true);
    // 4x2 field -> 2x1 blocks of 2x2
    const f = new Float32Array([1, 3, NaN, NaN, 5, 7, NaN, 4]);
    const out = blockMean(f, 4, 2, 2, 2);
    expect(out[0]).toBe(4);
    expect(out[1]).toBe(4);
    expect(Array.from(flatten2d([[1, null], [3, 4]])).map((v) => (Number.isNaN(v) ? null : v))).toEqual([1, null, 3, 4]);
  });
});

describe("grid geometry", () => {
  const g = makeGeom(100, 240, [5, 30], [45, 105]);

  it("has the true aspect of the domain with square degrees", () => {
    expect(aspectRatio(g)).toBeCloseTo(2.4);
    expect(g.dLat).toBeCloseTo(0.25);
  });

  it("finds the cell of a point and its centre", () => {
    expect(cellAt(g, 5.1, 45.1)).toEqual({ j: 0, i: 0 });
    expect(cellAt(g, 30, 105)).toEqual({ j: 99, i: 239 });
    expect(cellAt(g, 4.9, 60)).toBeNull();
    expect(cellCentre(g, { j: 40, i: 66 })).toEqual({ lat: 15.125, lon: 61.625 });
  });

  it("derives the coastline from the mask as cell edges", () => {
    // 3x2 grid: one ocean cell at (j=0, i=1)
    const mask = new Uint8Array([0, 1, 0, 0, 0, 0]);
    const seg = Array.from(coastlineSegments(mask, 3, 2));
    // west edge, east edge and north edge of that cell; the southern domain border is not a coast
    expect(seg).toEqual([1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 2, 1]);
  });

  it("finds the nearest ocean cell from land", () => {
    const small = makeGeom(2, 3, [0, 2], [0, 3]);
    const mask = new Uint8Array([0, 0, 1, 0, 0, 0]);
    expect(nearestOceanCell(mask, small, 1.5, 0.5)).toEqual({ j: 0, i: 2 });
    expect(nearestOceanCell(new Uint8Array(6), small, 1, 1)).toBeNull();
  });

  it("places the graticule on round degrees", () => {
    expect(graticule(45, 105, 20)).toEqual([60, 80, 100]);
    expect(graticuleStep(60, 6)).toBe(10);
    expect(graticuleStep(5, 6)).toBe(1);
  });
});

describe("the per-pixel MLP", () => {
  it("has its own identity, between the ablations and ridge in the legend", () => {
    const mlp = methodStyle("mlp");
    expect(mlp.color).toBe("#D95FA8");
    expect(mlp.marker).toBe("pentagon");
    expect(new Set([methodStyle("model"), methodStyle("ridge"), methodStyle("glorys"), methodStyle("climatology"), mlp].map((x) => `${x.color}|${x.dash}|${x.marker}`)).size).toBe(5);
    expect(sortMethods(["glorys", "ridge", "mlp", "climatology", "model_pretrained", "model"])).toEqual(["model", "model_pretrained", "mlp", "ridge", "climatology", "glorys"]);
    // the label comes from the API; the pretrained ablation has a short name of its own
    expect(shortLabel("mlp", "Per-pixel MLP")).toBe("Per-pixel MLP");
    expect(shortLabel("model_pretrained")).toBe("Pretrained");
  });
});

describe("methods", () => {
  it("keeps one identity per method, also under API aliases", () => {
    expect(canonicalMethod("clim")).toBe("climatology");
    expect(methodStyle("clim")).toBe(methodStyle("climatology"));
    expect(methodStyle("prediction")).toBe(methodStyle("model"));
    expect(methodStyle("target")).toBe(methodStyle("glorys"));
  });

  it("never tells methods apart by colour alone", () => {
    const keys = ["model", "model_scratch", "ridge", "climatology", "glorys", "obs"];
    const styles = keys.map((k) => methodStyle(k, ["model_scratch"]));
    expect(new Set(styles.map((s) => s.color)).size).toBe(keys.length);
    expect(new Set(styles.map((s) => s.marker)).size).toBe(keys.length);
    expect(new Set(styles.filter((s) => s.marker !== "ring").map((s) => s.dash)).size).toBe(keys.length - 1);
  });

  it("gives ablations stable slots and readable names", () => {
    const abl = ablationKeysOf(["model", "model_scratch", "ridge", "model_big"]);
    expect(abl).toEqual(["model_big", "model_scratch"]);
    expect(methodStyle("model_big", abl)).not.toBe(methodStyle("model_scratch", abl));
    expect(shortLabel("model_scratch")).toBe("No pretraining");
    expect(shortLabel("model_big")).toBe("Ablation: big");
    expect(sortMethods(["glorys", "ridge", "model", "clim", "model_scratch"])).toEqual(["model", "model_scratch", "ridge", "clim", "glorys"]);
  });
});

describe("metric axes", () => {
  it("starts error metrics at zero and centres bias on zero", () => {
    expect(metricDomain(metricMeta("rmse"), [0.2, 3.4])[0]).toBe(0);
    const [lo, hi] = metricDomain(metricMeta("bias"), [-0.5, 0.1]);
    expect(lo).toBeCloseTo(-hi);
    expect(metricDomain(metricMeta("corr_anom"), [0.2, 0.9])[1]).toBeLessThanOrEqual(1);
    expect(metricDomain(metricMeta("rmse"), [null, null])).toEqual([0, 1]);
  });
});

describe("embedding helpers", () => {
  const emb = decodeEmbedding({
    run: "r",
    date: "2024-03-16",
    shape: [3, 1, 3],
    // one row, three cells; the middle one is land
    data: [[[0, 0.5, 1]], [[0, 0.5, 1]], [[1, 0.5, 0]]],
    explained_variance_ratio: [0.3, 0.1, 0.05],
    lat: [5.5],
    lon: [45.5, 46.5, 47.5],
    cell_size: [1, 1],
    ocean: [[1, 0, 1]],
  });

  it("decodes to north-up RGB with land transparent", () => {
    expect([emb.ny, emb.nx]).toEqual([1, 3]);
    expect(Array.from(emb.rgba.slice(0, 4))).toEqual([0, 0, 255, 255]);
    expect(emb.rgba[7]).toBe(0);
    expect(Array.from(emb.rgba.slice(8, 12))).toEqual([255, 255, 0, 255]);
    expect(emb.geom.lon0).toBe(45);
    expect(emb.geom.lat1).toBe(6);
  });

  it("keeps every served component and accumulates their variance", () => {
    const eight = decodeEmbedding({
      run: "r",
      date: "2024-03-16",
      shape: [5, 1, 2],
      data: [[[0, 1]], [[1, 0]], [[0.5, 0.5]], [[0.2, 0.8]], [[0.9, 0.1]]],
      explained_variance_ratio: [0.3, 0.2, 0.1, 0.05, 0.05],
      lat: [5.5],
      lon: [45.5, 46.5],
      cell_size: [1, 1],
      ocean: [[1, 1]],
    });
    expect(eight.pcs).toHaveLength(5);
    expect(eight.pcs[4][0]).toBeCloseTo(0.9);
    // only the first three components make the colour
    expect(Array.from(eight.rgba.slice(0, 4))).toEqual([0, 255, 128, 255]);
    const cum = cumulative(eight.explained);
    expect(cum[2]).toBeCloseTo(0.6);
    expect(cum[4]).toBeCloseTo(0.7);
  });

  it("decodes the API's similarity map: land is null, the range over the other ocean cells is the API's", () => {
    const s = decodeSimilarity({
      run: "r",
      date: "2024-03-16",
      requested: { lat: 5.5, lon: 45.5 },
      lat: 5.5,
      lon: 45.5,
      y_index: 0,
      x_index: 0,
      emb_dim: 128,
      centered: false,
      reference_is_ocean: true,
      shape: [1, 4],
      data: [[1, null, 0.7, -0.2]],
      similarity_range: { min: -0.2, max: 0.7 },
      lat_values: [5.5],
      lon_values: [45.5, 46.5, 47.5, 48.5],
      cell_size: [1, 1],
      ocean: [[1, 0, 1, 1]],
    });
    expect(s.data[0]).toBe(1);
    expect(Number.isNaN(s.data[1])).toBe(true);
    expect([s.min, s.max]).toEqual([expect.closeTo(-0.2), expect.closeTo(0.7)]);
    expect(s.alike).toBeCloseTo(0.5);
    expect(s.embDim).toBe(128);
    expect(s.referenceIsOcean).toBe(true);
  });

  it("correlates components with fields on the embedding grid", () => {
    const big = makeEmb5();
    const up = new Float32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    const r = componentFieldCorrelation(big, [{ data: up, nx: 10, ny: 1 }]);
    expect(r[0][0]).toBeCloseTo(1);
    expect(r[2][0]).toBeCloseTo(-1);
  });
});

function makeEmb5() {
  const up = [0, 0.25, 0.5, 0.75, 1];
  return decodeEmbedding({
    run: "r",
    date: "2024-03-16",
    shape: [3, 1, 5],
    data: [[up], [up], [[...up].reverse()]],
    explained_variance_ratio: [0.3, 0.1, 0.05],
    lat: [5.5],
    lon: [45.5, 46.5, 47.5, 48.5, 49.5],
    cell_size: [1, 1],
    ocean: [[1, 1, 1, 1, 1]],
  });
}
