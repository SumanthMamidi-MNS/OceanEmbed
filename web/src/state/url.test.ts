import { describe, expect, it } from "vitest";
import { EMPTY_STATE, applyPatch, formatUrl, parseUrl, viewHref, type UrlState } from "./url";

describe("URL state", () => {
  it("parses a full deep link", () => {
    const s = parseUrl("/explore", "?run=poc&date=2024-03-16&depth=100&lat=15.125&lon=61.625&q=anom&sec=merid");
    expect(s).toEqual({
      view: "explore",
      run: "poc",
      date: "2024-03-16",
      depth: 100,
      lat: 15.125,
      lon: 61.625,
      est: null,
      opts: { q: "anom", sec: "merid" },
    });
  });

  it("round-trips through formatUrl", () => {
    const url = "/validation?run=poc_trial&date=2024-03-02&depth=75&lat=10.375&lon=88.125&est=model_scratch&basin=bay_of_bengal&metric=bias";
    const [path, query] = url.split("?");
    expect(formatUrl(parseUrl(path, `?${query}`))).toBe(url);
  });

  it("opens on the overview for the root and for unknown paths", () => {
    expect(parseUrl("/", "").view).toBe("overview");
    expect(parseUrl("/nope/deeper", "").view).toBe("overview");
    expect(parseUrl("/explore/", "").view).toBe("explore");
    expect(formatUrl(EMPTY_STATE)).toBe("/");
  });

  it("drops malformed values instead of failing", () => {
    const s = parseUrl("/explore", "?run=../../etc&date=2024-02-30&depth=abc&lat=95&lon=60&BAD=1&q=");
    expect(s.run).toBeNull();
    expect(s.date).toBeNull();
    expect(s.depth).toBeNull();
    expect(s.lat).toBeNull();
    expect(s.lon).toBeNull();
    expect(s.opts).toEqual({});
  });

  it("needs both coordinates for a point", () => {
    expect(parseUrl("/", "?lat=10").lat).toBeNull();
    expect(parseUrl("/", "?lon=60").lon).toBeNull();
    const cleared = applyPatch(parseUrl("/", "?lat=10&lon=60"), { lat: null });
    expect([cleared.lat, cleared.lon]).toEqual([null, null]);
  });

  it("writes compact numbers and a stable option order", () => {
    const s: UrlState = { ...EMPTY_STATE, view: "explore", depth: 100, lat: 15.125001, lon: 61.6, opts: { z: "1", a: "2" } };
    expect(formatUrl(s)).toBe("/explore?depth=100&lat=15.125&lon=61.6&a=2&z=1");
  });

  it("keeps the shared selection and drops the options when the view changes", () => {
    const s = parseUrl("/explore", "?run=poc&date=2024-03-16&depth=100&lat=15&lon=60&q=anom");
    const next = applyPatch(s, { view: "validation" });
    expect(next.view).toBe("validation");
    expect([next.run, next.date, next.depth, next.lat, next.lon]).toEqual(["poc", "2024-03-16", 100, 15, 60]);
    expect(next.opts).toEqual({});
    expect(viewHref(s, "validation")).toBe("/validation?run=poc&date=2024-03-16&depth=100&lat=15&lon=60");
  });

  it("keeps the selected estimate across views and drops a malformed one", () => {
    const s = parseUrl("/explore", "?run=poc&est=ridge&q=anom");
    expect(s.est).toBe("ridge");
    expect(applyPatch(s, { view: "validation" }).est).toBe("ridge");
    expect(viewHref(s, "validation")).toBe("/validation?run=poc&est=ridge");
    expect(formatUrl(applyPatch(s, { est: null }))).toBe("/explore?run=poc&q=anom");
    expect(parseUrl("/explore", "?est=../x").est).toBeNull();
  });

  it("drops the date, the estimate and the options when the run changes, because they belong to the old run", () => {
    const s = parseUrl("/explore", "?run=a&date=2024-03-16&depth=100&est=ridge&q=anom");
    const next = applyPatch(s, { run: "b" });
    expect(next.run).toBe("b");
    expect(next.est).toBeNull();
    expect(next.date).toBeNull();
    expect(next.depth).toBe(100);
    expect(next.opts).toEqual({});
  });

  it("merges options and removes the ones set to null", () => {
    const s = parseUrl("/explore", "?q=anom&sec=merid");
    const next = applyPatch(s, { opts: { q: null, mode: "surf" } });
    expect(next.opts).toEqual({ sec: "merid", mode: "surf" });
    // same view given again: options survive
    expect(applyPatch(s, { view: "explore", depth: 50 }).opts).toEqual({ q: "anom", sec: "merid" });
  });
});
