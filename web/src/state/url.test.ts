import { describe, expect, it } from "vitest";
import { EMPTY_STATE, RESEARCH_VIEWS, VIEWS, applyPatch, formatUrl, isResearchView, parseUrl, primaryViews, viewHref, type UrlState } from "./url";

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
    const url = "/research/scores?run=poc_trial&date=2024-03-02&depth=75&lat=10.375&lon=88.125&est=model_scratch&basin=bay_of_bengal&metric=bias";
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
    const next = applyPatch(s, { view: "accuracy" });
    expect(next.view).toBe("accuracy");
    expect([next.run, next.date, next.depth, next.lat, next.lon]).toEqual(["poc", "2024-03-16", 100, 15, 60]);
    expect(next.opts).toEqual({});
    expect(viewHref(s, "accuracy")).toBe("/accuracy?run=poc&date=2024-03-16&depth=100&lat=15&lon=60");
  });

  it("keeps the selected estimate inside the Research area and drops it on the product views", () => {
    const s = parseUrl("/research/maps", "?run=poc&est=ridge&q=anom");
    expect(s.est).toBe("ridge");
    expect(applyPatch(s, { view: "research_scores" }).est).toBe("ridge");
    expect(viewHref(s, "research_scores")).toBe("/research/scores?run=poc&est=ridge");
    // the product views show the product: no estimate in their links
    expect(applyPatch(s, { view: "accuracy" }).est).toBeNull();
    expect(viewHref(s, "explore")).toBe("/explore?run=poc");
    expect(parseUrl("/accuracy", "?est=ridge").est).toBeNull();
    expect(applyPatch(parseUrl("/explore", "?run=poc"), { est: "ridge" }).est).toBeNull();
    expect(formatUrl(applyPatch(s, { est: null }))).toBe("/research/maps?run=poc&q=anom");
    expect(parseUrl("/research/maps", "?est=../x").est).toBeNull();
  });

  it("drops the date, the estimate and the options when the run changes, because they belong to the old run", () => {
    const s = parseUrl("/research/maps", "?run=a&date=2024-03-16&depth=100&est=ridge&q=anom");
    expect(s.est).toBe("ridge");
    const next = applyPatch(s, { run: "b" });
    expect(next.run).toBe("b");
    expect(next.est).toBeNull();
    expect(next.date).toBeNull();
    expect(next.depth).toBe(100);
    expect(next.opts).toEqual({});
  });

  it("routes the product views and the Research pages", () => {
    expect(parseUrl("/accuracy", "").view).toBe("accuracy");
    expect(parseUrl("/data", "").view).toBe("data");
    expect(parseUrl("/research", "").view).toBe("research");
    expect(parseUrl("/research/scores/", "").view).toBe("research_scores");
    expect(parseUrl("/research/maps", "").view).toBe("research_maps");
    expect(parseUrl("/research/embedding", "").view).toBe("research_embedding");
    expect(primaryViews(false).map((v) => v.label)).toEqual(["Overview", "Explorer", "Accuracy", "Data & downloads"]);
    expect(RESEARCH_VIEWS.map((v) => v.path)).toEqual(["/research", "/research/scores", "/research/maps", "/research/embedding"]);
    expect(RESEARCH_VIEWS.every((v) => isResearchView(v.key))).toBe(true);
    expect(primaryViews(true).some((v) => isResearchView(v.key))).toBe(false);
  });

  it("keeps the links of the earlier layout working", () => {
    const redirect = (path: string, search = "") => formatUrl(parseUrl(path, search));
    expect(redirect("/validation", "?run=final&depth=100&metric=bias&yr=2023")).toBe("/accuracy?run=final&depth=100&metric=bias&yr=2023");
    expect(redirect("/validation/", "?run=final&prof=5906527_112")).toBe("/accuracy?run=final&prof=5906527_112");
    expect(redirect("/experiments", "?run=final&cmp=final.poc")).toBe("/research?run=final&cmp=final.poc");
    expect(redirect("/representation", "?run=final&date=2024-06-10&lat=15.125&lon=61.625&sim=rel")).toBe(
      "/research/embedding?run=final&date=2024-06-10&lat=15.125&lon=61.625&sim=rel",
    );
    // an Explorer link that compared methods, or showed a baseline, is a Research link now
    expect(redirect("/explore", "?run=final&depth=100&sbs=1")).toBe("/research/maps?run=final&depth=100&sbs=1");
    expect(redirect("/explore", "?run=final&est=ridge&q=anom")).toBe("/research/maps?run=final&est=ridge&q=anom");
    // a plain Explorer link stays where it was
    expect(redirect("/explore", "?run=final&date=2024-06-10&depth=100&q=anom")).toBe("/explore?run=final&date=2024-06-10&depth=100&q=anom");
  });

  it("lists Live third, and only when the API reports a live run", () => {
    expect(VIEWS.map((v) => v.key).indexOf("live")).toBe(2);
    expect(primaryViews(true).map((v) => v.label)).toEqual(["Overview", "Explorer", "Live", "Accuracy", "Data & downloads"]);
    expect(primaryViews(false).some((v) => v.key === "live")).toBe(false);
    // the route always parses; the app sends it to the Overview when there is no live run
    expect(parseUrl("/live", "?run=final").view).toBe("live");
  });

  it("drops the day when a link enters or leaves the Live view: the live run has its own days", () => {
    const s = parseUrl("/explore", "?run=final&date=2024-06-10&depth=100&lat=15&lon=60");
    const live = applyPatch(s, { view: "live" });
    expect([live.run, live.date, live.depth, live.lat]).toEqual(["final", null, 100, 15]);
    expect(viewHref(s, "live")).toBe("/live?run=final&depth=100&lat=15&lon=60");
    const back = applyPatch(parseUrl("/live", "?run=final&date=2026-10-03&depth=50&vb=0_30"), { view: "accuracy" });
    expect([back.date, back.depth, back.opts]).toEqual([null, 50, {}]);
    // inside the Live view the day is kept
    expect(applyPatch(parseUrl("/live", "?date=2026-10-03"), { depth: 200 }).date).toBe("2026-10-03");
    // between two evaluated views nothing changes
    expect(applyPatch(s, { view: "accuracy" }).date).toBe("2024-06-10");
  });

  it("merges options and removes the ones set to null", () => {
    const s = parseUrl("/explore", "?q=anom&sec=merid");
    const next = applyPatch(s, { opts: { q: null, mode: "surf" } });
    expect(next.opts).toEqual({ sec: "merid", mode: "surf" });
    // same view given again: options survive
    expect(applyPatch(s, { view: "explore", depth: 50 }).opts).toEqual({ q: "anom", sec: "merid" });
  });
});
