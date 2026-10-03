import { describe, expect, it } from "vitest";
import type { MetricsResponse, RunSummary } from "@/api/types";
import {
  ablationFindings,
  argoSentence,
  basinSentence,
  chooseEvidence,
  climatologyText,
  compare,
  correlationSentence,
  defaultRun,
  depthSkill,
  describeComparison,
  describeDepths,
  headline,
  methodList,
  runCaveats,
} from "./narrative";

function method(key: string, label: string, extra: Record<string, unknown> = {}) {
  return { key, label, kind: key.startsWith("model") ? "model" : "baseline", in_glorys_metrics: true, in_argo_metrics: true, ...extra };
}

function metrics(pooled: Record<string, number>, over: Partial<MetricsResponse> = {}): MetricsResponse {
  const block = (rmse: number) => ({ n: 100, rmse, bias: 0, mae: rmse * 0.8, corr_raw: 0.93, corr_anom: 0.74, skill_vs_clim: 0.5 });
  return {
    run: "r",
    reference: "GLORYS",
    metadata: {},
    methods: Object.keys(pooled).map((k) => method(k, k === "model_scratch" ? "OceanEmbed (no pretraining)" : k, k === "model_scratch" ? { pretrained: false } : {})),
    metric_names: ["rmse"],
    overall: Object.fromEntries(Object.entries(pooled).map(([k, v]) => [k, block(v * 0.6)])),
    pooled: Object.fromEntries(Object.entries(pooled).map(([k, v]) => [k, block(v)])),
    pooled_range_m: [50, 200],
    depths: [0, 100, 500, 1000],
    per_depth: {},
    per_basin: {},
    ...over,
  } as MetricsResponse;
}

function run(over: Partial<RunSummary>): RunSummary {
  return {
    name: "r",
    run_name: "r",
    data_source: "real",
    updated: null,
    period: {},
    split: { train: { start: "2018-01-01", end: "2022-12-31" }, test: { start: "2024-01-01", end: "2024-12-15" } },
    label: "Demo run",
    description: "",
    n_train_days: 1826,
    n_val_days: 365,
    n_test_days: 350,
    n_harmonic_terms: 5,
    grid: { resolution: 0.25, n_lat: 100, n_lon: 240, n_depth: 15, lat: [], lon: [], depths: [] },
    artefacts: { predictions: true, metrics_glorys: true, metrics_argo: true, argo_matchups: true, maps: true, embeddings: true, report: true, training_logs: [], n_figures: 0, n_product_files: 0 },
    n_prediction_days: 350,
    ...over,
  } as RunSummary;
}

describe("compare", () => {
  it("reports how far a value is below or above a reference", () => {
    expect(compare(2, 4)).toEqual({ pct: 50, direction: "lower" });
    expect(compare(5, 4)?.direction).toBe("higher");
    expect(compare(4.02, 4)?.direction).toBe("same");
    expect(compare(null, 4)).toBeNull();
    expect(compare(1, 0)).toBeNull();
  });

  it("puts it in words", () => {
    expect(describeComparison({ pct: 32.6, direction: "lower" }, "climatology")).toBe("33\u00A0% lower than climatology");
    expect(describeComparison({ pct: -4.4, direction: "higher" }, "ridge regression")).toBe("4\u00A0% higher than ridge regression");
    expect(describeComparison({ pct: 0.2, direction: "same" }, "climatology")).toBe("about the same as climatology");
  });
});

describe("headline", () => {
  it("states the gain over both baselines when the model wins", () => {
    const h = headline(metrics({ model: 2.28, ridge: 2.64, climatology: 3.39 }));
    expect(h.beatsBaselines).toBe(true);
    expect(h.range).toBe("50–200\u00A0m");
    expect(h.sentence).toContain("2.28\u00A0°C");
    expect(h.sentence).toContain("33\u00A0% lower than climatology (3.39\u00A0°C)");
    expect(h.sentence).toContain("14\u00A0% lower than ridge regression (2.64\u00A0°C)");
  });

  it("says so when the model loses to a baseline: nothing is pre-written", () => {
    const h = headline(metrics({ model: 3.0, ridge: 2.5, climatology: 2.9 }));
    expect(h.beatsBaselines).toBe(false);
    expect(h.sentence).toContain("3\u00A0% higher than climatology");
    expect(h.sentence).toContain("20\u00A0% higher than ridge regression");
    expect(h.sentence).not.toContain("lower");
  });

  it("reports a tie as a tie", () => {
    const h = headline(metrics({ model: 1.31, ridge: 1.315, climatology: 1.53 }));
    expect(h.vsRidge?.direction).toBe("same");
    expect(h.beatsBaselines).toBe(false);
    expect(h.sentence).toContain("about the same as ridge regression");
  });

  it("degrades without metrics", () => {
    const h = headline(metrics({}));
    expect(h.model).toBeNull();
    expect(h.sentence).toMatch(/No pooled metrics/);
  });

  it("always pairs the anomaly correlation with the raw one", () => {
    const s = correlationSentence(metrics({ model: 2, climatology: 3 }));
    expect(s).toContain("anomaly correlation");
    expect(s).toContain("0.74");
    expect(s).toContain("raw correlation 0.93");
    expect(s).toContain("inflated");
  });
});

describe("ablation findings", () => {
  it("reports that pretraining does not help when the scratch model is clearly better", () => {
    const [f] = ablationFindings(metrics({ model: 2.4, model_scratch: 2.18, ridge: 2.64, climatology: 3.39 }));
    expect(f.key).toBe("model_scratch");
    expect(f.comparison.direction).toBe("higher");
    expect(f.sentence).toMatch(/^Pretraining does not help in this run/);
    expect(f.sentence).toContain("2.40 vs 2.18");
  });

  it("does not turn a single-seed difference of a few percent into a finding, either way", () => {
    // the real run: 1.077 vs 1.041, the ablation about 3 % better
    const [worse] = ablationFindings(metrics({ model: 1.077, model_scratch: 1.041 }));
    expect(worse.sentence).toMatch(/^No measurable gain from pretraining in this run/);
    expect(worse.sentence).toContain("3\u00A0% higher than the same network trained from scratch (1.08 vs 1.04\u00A0°C)");
    expect(worse.sentence).toContain("training seed");
    const [better] = ablationFindings(metrics({ model: 1.04, model_scratch: 1.077 }));
    expect(better.sentence).toMatch(/^No measurable gain from pretraining in this run/);
    expect(better.sentence).toContain("3\u00A0% lower");
  });

  it("reports a gain when there is a clear one, and a tie when there is none", () => {
    expect(ablationFindings(metrics({ model: 2.0, model_scratch: 2.4 }))[0].sentence).toMatch(/^Pretraining helps in this run/);
    expect(ablationFindings(metrics({ model: 2.0, model_scratch: 2.01 }))[0].sentence).toMatch(/^No measurable gain from pretraining/);
  });

  it("is empty for a run without ablations", () => {
    expect(ablationFindings(metrics({ model: 2, ridge: 3, climatology: 4 }))).toEqual([]);
  });
});

describe("depth skill", () => {
  it("names the best depth, the depths without skill and the depths where ridge wins", () => {
    const m = metrics(
      { model: 2, ridge: 3, climatology: 4 },
      {
        per_depth: {
          skill_vs_clim: { model: [0.75, 0.54, 0.1, -0.11] },
          rmse: { model: [0.24, 2.87, 0.19, 0.07], ridge: [0.32, 3.25, 0.18, 0.06] },
        },
      },
    );
    const d = depthSkill(m);
    expect(d.bestDepth).toBe(0);
    expect(d.bestSkill).toBe(0.75);
    expect(d.noSkillDepths).toEqual([1000]);
    expect(d.ridgeWinsDepths).toEqual([500, 1000]);
    expect(d.sentence).toContain("highest at 0\u00A0m (0.75) and at least 0.1 at 0–500\u00A0m");
    expect(d.clearDepths).toEqual([0, 100, 500]);
    expect(d.sentence).toContain("does not beat climatology at 1000\u00A0m");
    expect(d.sentence).toContain("ridge regression has the lower RMSE at 500\u00A0m and below");
  });

  it("keeps the sentence short on a 15-level grid: ranges, not a list of depths", () => {
    const depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
    const skill = depths.map((d) => (d >= 300 ? -0.05 : 0.4));
    const model = depths.map(() => 1);
    const ridge = depths.map((d) => (d <= 30 || d >= 500 ? 0.9 : 1.2));
    const m = metrics({ model: 2, ridge: 3, climatology: 4 }, { depths, per_depth: { skill_vs_clim: { model: skill }, rmse: { model, ridge } } });
    const s = depthSkill(m).sentence!;
    expect(s).toContain("does not beat climatology at 300\u00A0m and below");
    expect(s).toContain("ridge regression has the lower RMSE at 0–30\u00A0m and 500\u00A0m and below");
    expect(s.length).toBeLessThan(240);
  });

  it("says where the skill is clear, where it is marginal and where there is none (the real run's shape)", () => {
    const depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
    const skill = [0.479, 0.488, 0.482, 0.421, 0.369, 0.39, 0.493, 0.456, 0.364, 0.319, 0.219, 0.045, -0.034, -0.056, -0.152];
    const model = [0.572, 0.567, 0.57, 0.631, 0.725, 0.921, 1.1, 1.222, 1.241, 1.121, 0.79, 0.532, 0.357, 0.366, 0.408];
    const ridge = [0.727, 0.728, 0.726, 0.762, 0.828, 0.997, 1.213, 1.313, 1.303, 1.199, 0.827, 0.532, 0.351, 0.354, 0.38];
    const s = depthSkill(metrics({ model: 1.08, ridge: 1.15, climatology: 1.39 }, { depths, per_depth: { skill_vs_clim: { model: skill }, rmse: { model, ridge } } })).sentence!;
    expect(s).toBe(
      "Skill against climatology is highest at 75\u00A0m (0.49) and at least 0.1 at 0–200\u00A0m; it is marginal (below 0.1) at 300\u00A0m; " +
        "the model does not beat climatology at 500\u00A0m and below; ridge regression has the lower RMSE at 500\u00A0m and below.",
    );
  });

  it("says it plainly when there is no skill anywhere", () => {
    const m = metrics({ model: 2 }, { per_depth: { skill_vs_clim: { model: [-0.2, -0.05, -0.4, -0.3] }, rmse: {} } });
    expect(depthSkill(m).sentence).toMatch(/^The model does not beat climatology at any depth/);
  });
});

describe("depths in words", () => {
  const levels = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
  it("merges neighbouring levels and names an open-ended tail", () => {
    expect(describeDepths([100], levels)).toBe("100\u00A0m");
    expect(describeDepths([50, 75, 100], levels)).toBe("50–100\u00A0m");
    expect(describeDepths([0, 5, 300, 500, 700, 1000], levels)).toBe("0–5\u00A0m and 300\u00A0m and below");
    expect(describeDepths([0, 20, 30, 1000], levels)).toBe("0\u00A0m, 20–30\u00A0m and 1000\u00A0m");
    expect(describeDepths(levels, levels)).toBe("every depth");
    expect(describeDepths([], levels)).toBe("");
  });

  it("summarises a scattered set instead of listing it", () => {
    expect(describeDepths([5, 20, 50, 100, 150, 300, 700], levels)).toBe("7 of the 15 levels between 5\u00A0m and 700\u00A0m");
  });
});

describe("Argo sentence", () => {
  const blocks = (rows: Record<string, [number, number]>) => Object.fromEntries(Object.entries(rows).map(([k, [rmse, bias]]) => [k, { n: 16241, rmse, bias }]));
  it("uses the pooled range, names GLORYS as the floor and states the bias both share", () => {
    const m = metrics(
      { model: 1.415 },
      {
        metadata: { n_profiles_used: 2826, n_matchups: 40059 },
        pooled: blocks({ model: [1.415, 0.639], ridge: [1.531, 0.68], climatology: [1.669, 0.511], glorys: [1.079, 0.494] }),
      },
    );
    const s = argoSentence(m)!;
    expect(s).toContain("Against 2\u202F826 Argo profiles, between 50–200\u00A0m (16\u202F241 profile–depth matchups), OceanEmbed's RMSE is 1.42\u00A0°C");
    expect(s).toContain("15\u00A0% lower than climatology (1.67\u00A0°C)");
    expect(s).toContain("GLORYS itself differs from the same profiles by 1.08\u00A0°C: that is the floor");
    expect(s).toContain("GLORYS is on average 0.49\u00A0°C warmer than Argo there, and the reconstruction 0.64\u00A0°C warmer.");
  });

  it("says nothing about a bias that is not there, and can speak about all depths", () => {
    const m = metrics({ model: 2.5, ridge: 2.9, climatology: 3.8, glorys: 0.45 }, { metadata: { n_profiles_used: 900, n_matchups: 12686 } });
    expect(argoSentence(m)).not.toContain("warmer");
    const all = argoSentence(m, "overall")!;
    expect(all).toContain("900 Argo profiles, over all depths (12\u202F686 profile–depth matchups)");
    expect(all).toContain("GLORYS itself differs from the same profiles by 0.27");
  });
});

describe("basin sentence", () => {
  const basin = (model: number, clim: number, skill: number) => ({ overall: {}, pooled: { model: { rmse: model, skill_vs_clim: skill }, climatology: { rmse: clim, skill_vs_clim: 0 } }, per_depth: {} });
  it("contrasts the basins, the more skilful one first", () => {
    const m = metrics({ model: 1.08 }, { per_basin: { arabian_sea: basin(1.112, 1.317, 0.287), bay_of_bengal: basin(1.021, 1.527, 0.553) } });
    expect(basinSentence(m, { arabian_sea: "Arabian Sea", bay_of_bengal: "Bay of Bengal" })).toBe(
      "Over 50–200\u00A0m the skill against climatology is 0.55 in the Bay of Bengal (RMSE 1.02 vs 1.53\u00A0°C for climatology) and 0.29 in the Arabian Sea (1.11 vs 1.32\u00A0°C).",
    );
    expect(basinSentence(metrics({ model: 1 }), {})).toBeNull();
  });
});

describe("evidence for the overview", () => {
  const depths = [0, 50, 100, 200, 500];
  const base = { depths, per_depth: { rmse: { model: [0.57, 0.92, 1.22, 0.79, 0.36], climatology: [0.79, 1.18, 1.66, 0.89, 0.35] } } };
  it("takes the depth of the pooled range where climatology errs most, never the surface", () => {
    const e = chooseEvidence(metrics({ model: 1 }, base));
    expect(e.depth).toBe(100);
    expect(e.depthClimatologyRmse).toBe(1.66);
    // even when the surface has the largest error and there is no pooled range
    const surf = chooseEvidence(metrics({ model: 1 }, { depths, pooled_range_m: [], per_depth: { rmse: { climatology: [9, 1, 2, 1, 1] } } }));
    expect(surf.depth).toBe(100);
  });

  it("takes the day whose pooled error is the median of the test days", () => {
    const daily = { dates: ["d1", "d2", "d3", "d4", "d5"], depths, pooled_rmse: { model: [1.3, 0.9, 1.1, null, 1.0] } };
    const e = chooseEvidence(metrics({ model: 1 }, { ...base, daily }));
    // values 0.9, 1.0, 1.1, 1.3: the lower median
    expect([e.date, e.dayRmse, e.nDays, e.dayScope]).toEqual(["d5", 1.0, 4, "pooled"]);
    // only days that can be shown qualify
    expect(chooseEvidence(metrics({ model: 1 }, { ...base, daily }), new Set(["d1", "d2", "d3"])).date).toBe("d3");
  });

  it("falls back to the daily error at the chosen depth, and to nothing without daily metrics", () => {
    const daily_rmse = { dates: ["a", "b", "c"], depths, methods: { model: [[0, 0, 2.0, 0, 0], [0, 0, 1.0, 0, 0], [0, 0, 1.5, 0, 0]] } };
    const e = chooseEvidence(metrics({ model: 1 }, { ...base, daily_rmse }));
    expect([e.date, e.dayScope]).toEqual(["c", "depth"]);
    const none = chooseEvidence(metrics({ model: 1 }, base));
    expect([none.date, none.dayScope, none.nDays]).toEqual([null, null, 0]);
  });
});

describe("run caveats", () => {
  it("flags a synthetic run from its data source, not its name", () => {
    const c = runCaveats(run({ name: "anything", data_source: "synthetic", note: "SYNTHETIC DATA: demo" }));
    expect(c.synthetic).toBe(true);
    expect(c.syntheticNote).toBe("SYNTHETIC DATA: demo");
    expect(c.shortTraining).toBe(false);
  });

  it("flags a run trained on less than one annual cycle from the day count the API reports", () => {
    const c = runCaveats(run({ name: "looks_fine", n_train_days: 46, n_harmonic_terms: 1 }));
    expect(c.trainDays).toBe(46);
    expect(c.shortTraining).toBe(true);
    expect(c.shortTrainingNote).toContain("46 days");
    expect(c.shortTrainingNote).toContain("constant mean");
    expect(c.synthetic).toBe(false);
  });

  it("does not flag a multi-year real run", () => {
    const c = runCaveats(run({}));
    expect(c.trainDays).toBe(1826);
    expect(c.shortTraining).toBe(false);
    expect(c.shortTrainingNote).toBeNull();
  });

  it("does not guess when the API reports no day count", () => {
    const c = runCaveats(run({ n_train_days: null, split: { train: { start: "2024-01-01", end: "2024-02-15" } } }));
    expect(c.trainDays).toBeNull();
    expect(c.shortTraining).toBe(false);
  });

  it("names the climatology fit from its number of terms", () => {
    expect(climatologyText(1)).toMatch(/constant mean/);
    expect(climatologyText(3)).toBe("mean and annual cycle");
    expect(climatologyText(5)).toBe("mean, annual and semi-annual cycle");
    expect(climatologyText(null)).toBe("harmonic fit");
  });
});

describe("default run", () => {
  const arte = (over: Partial<RunSummary["artefacts"]>) => ({ ...run({}).artefacts, ...over });
  it("prefers the long real run, then the short real one, then the synthetic demo", () => {
    const full = run({ name: "poc" });
    const trial = run({ name: "poc_trial", n_train_days: 46, n_prediction_days: 31 });
    const synthetic = run({ name: "synthetic", data_source: "synthetic", n_prediction_days: 184 });
    expect(defaultRun([synthetic, trial, full])?.name).toBe("poc");
    expect(defaultRun([synthetic, trial])?.name).toBe("poc_trial");
    expect(defaultRun([])).toBeNull();
  });

  it("never lands on a run that is still being produced", () => {
    const building = run({ name: "poc", n_prediction_days: 0, artefacts: arte({ predictions: false, metrics_glorys: false }) });
    const trial = run({ name: "poc_trial", n_train_days: 46, n_prediction_days: 31 });
    expect(defaultRun([building, trial])?.name).toBe("poc_trial");
    // once it has a reconstruction it takes over, even before it is evaluated
    const predicted = run({ name: "poc", artefacts: arte({ metrics_glorys: false }) });
    expect(defaultRun([predicted, trial])?.name).toBe("poc");
  });
});

describe("method list", () => {
  it("lists OceanEmbed first, then ablations, ridge, climatology and GLORYS", () => {
    const m = metrics({ glorys: 0.2, climatology: 3, ridge: 2.5, model_scratch: 2.1, model: 2 });
    expect(methodList(m).map((x) => x.key)).toEqual(["model", "model_scratch", "ridge", "climatology", "glorys"]);
    expect(methodList(m).map((x) => x.short)).toEqual(["OceanEmbed", "No pretraining", "Ridge", "Climatology", "GLORYS"]);
  });
});
