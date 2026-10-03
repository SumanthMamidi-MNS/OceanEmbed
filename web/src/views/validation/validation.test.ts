import { describe, expect, it } from "vitest";
import type { RangeInfo } from "@/api/types";
import { rmseMethods } from "./ArgoSection";
import { outsideNote } from "./MapsSection";

const info = (over: Partial<RangeInfo>): RangeInfo => ({ n_valid: 9870, data_min: -3.041, data_max: 0.923, n_below: 337, n_above: 586, fraction_outside: 0.0935, exceeds_range: true, ...over });

describe("colour-range note of an error map", () => {
  it("says how much of the map lies beyond the scale, and how far", () => {
    expect(outsideNote(info({}), 2)).toBe("9\u00A0% of the cells lie beyond the colour scale (lowest \u22123.04, highest 0.92)");
  });
  it("names only the end that is exceeded and does not round a small share to zero", () => {
    expect(outsideNote(info({ n_above: 0, fraction_outside: 0.002 }), 2)).toBe("under 1\u00A0% of the cells lie beyond the colour scale (lowest \u22123.04)");
  });
});

describe("methods of a per-profile RMSE block", () => {
  it("keeps one climatology key when the API sends both spellings", () => {
    expect(rmseMethods({ model: 1, model_scratch: 1, ridge: 1, clim: 2, glorys: 0.5, climatology: 2 })).toEqual(["model", "model_scratch", "ridge", "climatology", "glorys"]);
  });
  it("still reads an older payload that only has the legacy key", () => {
    expect(rmseMethods({ model: 1, clim: 2 })).toEqual(["model", "clim"]);
    expect(rmseMethods(null)).toEqual([]);
  });
});
