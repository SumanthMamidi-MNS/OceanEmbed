import { describe, expect, it } from "vitest";
import type { RangesResponse } from "@/api/types";
import { byDepth, heldRanges, periodRanges, sharedMatrixLimit, sharedRanges, timeDepth } from "./fields";

describe("colour ranges of the explorer", () => {
  it("shares one range between estimates shown side by side: the larger limit wins", () => {
    const r = sharedRanges([
      { main: [14, 28], diff: 4.6 },
      { main: [14, 28], diff: 5.0 },
      { main: [13.5, 27], diff: 4.4 },
    ]);
    expect(r.main).toEqual([13.5, 28]);
    expect(r.diff).toBe(5.0);
    expect(sharedRanges([])).toEqual({ main: [0, 1], diff: 1 });
  });

  it("holds a range by only ever widening it when the API knows no period range", () => {
    const first = heldRanges({ main: [18, 26], diff: 2 }, undefined, null, "temp");
    expect(first).toEqual({ main: [18, 26], diff: 2 });
    // a later, milder day keeps the wider limits; a more extreme one widens them
    expect(heldRanges({ main: [19, 25], diff: 1.5 }, first, null, "temp")).toEqual({ main: [18, 26], diff: 2 });
    expect(heldRanges({ main: [17, 27], diff: 3 }, first, null, "temp")).toEqual({ main: [17, 27], diff: 3 });
  });

  it("uses the period-wide limits of the API, so a seasonal cycle stays on one scale", () => {
    const period = { temperature: [16, 29] as [number, number], anomaly: 4.7, difference: 3.5 };
    expect(heldRanges({ main: [22, 26], diff: 2 }, undefined, period, "temp")).toEqual({ main: [16, 29], diff: 3.5 });
    // an extreme day does not move a held scale: its values clip, as the pointed colour bar says
    expect(heldRanges({ main: [15, 31], diff: 5 }, undefined, period, "temp")).toEqual({ main: [16, 29], diff: 3.5 });
    expect(heldRanges({ main: [-3, 3], diff: 2 }, undefined, period, "anom").main).toEqual([-4.7, 4.7]);
    // a missing part falls back to the widening rule
    expect(heldRanges({ main: [-3, 3], diff: 2 }, { main: [-4, 4], diff: 2.5 }, { temperature: [16, 29], anomaly: null, difference: null }, "anom")).toEqual({ main: [-4, 4], diff: 2.5 });
  });

  it("reads the period limits of one depth for the methods on screen: the largest symmetric limit wins", () => {
    const r = (vmax: number) => ({ vmin: -vmax, vmax, diverging: true });
    const ranges = {
      temperature: [{ vmin: 23, vmax: 32, diverging: false }, { vmin: 17, vmax: 28, diverging: false }],
      methods: {
        model: { label: "m", kind: "model", difference: [r(1.6), r(3.45)], anomaly: [r(2), r(4.73)] },
        ridge: { label: "r", kind: "baseline", difference: [r(1.9), r(3.91)], anomaly: [r(2), r(4.41)] },
        model_scratch: { label: "s", kind: "ablation", difference: null, anomaly: null },
      },
    } as unknown as RangesResponse;
    expect(periodRanges(ranges, 1, ["model"])).toEqual({ temperature: [17, 28], anomaly: 4.73, difference: 3.45 });
    expect(periodRanges(ranges, 1, ["model", "ridge"])).toEqual({ temperature: [17, 28], anomaly: 4.73, difference: 3.91 });
    expect(periodRanges(ranges, 0, ["model_scratch"])).toEqual({ temperature: [23, 32], anomaly: null, difference: null });
    expect(periodRanges(undefined, 0, ["model"])).toBeNull();
  });
});

describe("time-depth panels", () => {
  const prediction = [
    [20, 10],
    [22, 11],
    [21, null],
  ];
  const target = [[19, 10.5], [null, null], [20, 9]];
  const climatology = [
    [20, 10],
    [20, 10],
    [20, 10],
  ];

  it("transposes [time][depth] rows into [depth][time]", () => {
    expect(byDepth(prediction, 2)).toEqual([
      [20, 22, 21],
      [10, 11, null],
    ]);
    expect(byDepth(null, 2)).toEqual([[], []]);
  });

  it("shows temperatures and the error in temperature mode", () => {
    const td = timeDepth(prediction, target, climatology, 2, false);
    expect(td.estimate[0]).toEqual([20, 22, 21]);
    expect(td.error[0]).toEqual([1, null, 1]);
    expect(td.error[1]).toEqual([-0.5, null, null]);
    expect(td.hasTarget).toBe(true);
  });

  it("removes the climatology from both sides in anomaly mode and leaves the error unchanged", () => {
    const temp = timeDepth(prediction, target, climatology, 2, false);
    const anom = timeDepth(prediction, target, climatology, 2, true);
    expect(anom.estimate[0]).toEqual([0, 2, 1]);
    expect(anom.target[0]).toEqual([-1, null, 0]);
    expect(anom.error).toEqual(temp.error);
    expect(anom.hasClimatology).toBe(true);
    // one symmetric limit for the two anomaly panels
    expect(anom.anomalyLimit).toBeGreaterThan(1);
    expect(anom.anomalyLimit).toBeLessThanOrEqual(2);
  });

  it("falls back to temperature when the run has no climatology at the point", () => {
    const td = timeDepth(prediction, target, null, 2, true);
    expect(td.hasClimatology).toBe(false);
    expect(td.estimate[0]).toEqual([20, 22, 21]);
    const empty = timeDepth(prediction, target, [[null, null], [null, null], [null, null]], 2, true);
    expect(empty.hasClimatology).toBe(false);
  });

  it("computes one symmetric limit over several matrices", () => {
    expect(sharedMatrixLimit([[[1, -2]], [[null, 0.5]]], 1)).toBe(2);
    expect(sharedMatrixLimit([[[null]]])).toBe(1);
  });
});
