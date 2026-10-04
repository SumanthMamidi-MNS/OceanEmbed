import { describe, expect, it } from "vitest";
import { scopeMethods } from "./runContext";

describe("which methods a view shows", () => {
  const all = ["model", "model_pretrained", "mlp", "ridge", "climatology", "glorys"];

  it("shows the product, the seasonal climatology and the references on the product views", () => {
    expect(scopeMethods(all, "product")).toEqual(["model", "climatology", "glorys"]);
    // the aliases some payloads use for the same entities
    expect(scopeMethods(["model", "ridge", "clim", "target", "argo", "model_scratch"], "product")).toEqual(["model", "clim", "target", "argo"]);
  });

  it("shows every method in the Research area, in the order given", () => {
    expect(scopeMethods(all, "research")).toEqual(all);
  });
});
