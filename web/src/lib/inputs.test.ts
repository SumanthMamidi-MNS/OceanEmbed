import { describe, expect, it } from "vitest";
import type { DataProduct } from "@/api/types";
import { inputNames, inputUse, panelUsed } from "./inputs";

const product = (variable: string, long_name: string, used: boolean, role = "input"): DataProduct =>
  ({ variable, long_name, role, used_by_model: used, product: "", dataset_ids: [], native_resolution: "", regridding: "" }) as unknown as DataProduct;

const products = [
  product("sst", "Sea surface temperature", true),
  product("sss", "Sea surface salinity", false),
  product("sla", "Sea level anomaly", true),
  product("uo", "Eastward surface current", false),
  product("vo", "Northward surface current", false),
  product("temp", "Sea water potential temperature", true, "target"),
];

describe("what the model uses", () => {
  it("separates the inputs the model uses from the ones that are only available", () => {
    const use = inputUse({ products });
    expect(use.used.map((p) => p.variable)).toEqual(["sst", "sla"]);
    expect(use.unused.map((p) => p.variable)).toEqual(["sss", "uo", "vo"]);
    expect(use.partial).toBe(true);
    expect(inputNames(use.used)).toBe("sea surface temperature and sea level anomaly");
    expect(inputNames(use.unused)).toBe("sea surface salinity, eastward surface current and northward surface current");
  });

  it("treats a run that uses everything as such, and an older payload without the flag as used", () => {
    const all = inputUse({ products: products.map((p) => ({ ...p, used_by_model: undefined }) as unknown as DataProduct) });
    expect(all.partial).toBe(false);
    expect(all.used).toHaveLength(5);
  });

  it("marks a two-component panel as an input only if both components are", () => {
    const use = inputUse({ products });
    expect(panelUsed("sst", use)).toBe(true);
    expect(panelUsed("uo+vo", use)).toBe(false);
  });
});
