import { describe, expect, it } from "vitest";
import type { LiveBandScore, LiveInput, LiveInputShift, LiveRevision, LiveVerification } from "@/api/types";
import { bandKeys, bandLabel, inputShiftSentence, pendingSentences, profilesPerDay, revisionPolicy, revisionSentence, rollingBands, verificationSentence, waitingFor } from "./live";

const score = (model: number, clim: number, bias = 0, n = 1000): LiveBandScore => ({ n, model_rmse: model, model_bias: bias, clim_rmse: clim, clim_bias: 0, n_days: 30 });
const bands = { "50_200": { label: "50-200 m", depth_range_m: [50, 200] }, "0_30": { label: "0-30 m", depth_range_m: [0, 30] }, "300_1000": { label: "300-1000 m", depth_range_m: [300, 1000] } };
const day = (date: string, n: number | null, rolling: Record<string, LiveBandScore>) => ({ date, n_profiles: n, bands: Object.fromEntries(Object.entries(rolling).map(([k, v]) => [k, { rolling: v }])) });

// the shape of the first month of the live run
const argo = { "0_30": score(0.965, 1.257, -0.11, 1358), "50_200": score(1.4314, 1.4313, 0.832, 1634), "300_1000": score(0.327, 0.32, 0.11, 1042) };
const analysis = { "0_30": score(0.75, 1.06), "50_200": score(1.24, 1.53, 0.5), "300_1000": score(0.36, 0.37) };
const v: LiveVerification = {
  rolling_days: 30,
  bands,
  argo: { daily: [day("2026-10-01", 4, argo), day("2026-10-02", 20, argo), day("2026-10-03", 7, argo)], latest: day("2026-10-03", 7, argo) },
  analysis: { daily: [day("2026-10-03", null, analysis)], latest: day("2026-10-03", null, analysis) },
};
const clim = "the seasonal climatology";

describe("live verification", () => {
  it("orders the depth bands from the surface down and names them", () => {
    expect(bandKeys(v)).toEqual(["0_30", "50_200", "300_1000"]);
    expect(bandLabel(v, "50_200")).toBe("50–200\u00A0m");
    expect(rollingBands(v, v.argo).map((b) => b.verdict?.direction)).toEqual(["lower", "same", "higher"]);
  });

  it("says where the reconstruction does not beat the climatology against the floats, with the bias", () => {
    expect(verificationSentence(v, "argo", clim)).toBe(
      "Against Argo floats over the last 30 days: at 0–30\u00A0m its error is 0.96\u00A0°C, 23\u00A0% lower than the seasonal climatology (1.26\u00A0°C); " +
        "at 50–200\u00A0m its error is about the same as that of the seasonal climatology (1.43 vs 1.43\u00A0°C); " +
        "at 300–1000\u00A0m its error is 0.33\u00A0°C, 2\u00A0% higher than the seasonal climatology (0.32\u00A0°C). " +
        "It does not beat the climatology at 50–200\u00A0m and 300–1000\u00A0m. " +
        "There the reconstruction is on average 0.83\u00A0°C warmer than the floats at 50–200\u00A0m.",
    );
  });

  it("says so when every band beats the climatology, and names the analysis as what it is compared with", () => {
    const s = verificationSentence(v, "analysis", clim)!;
    expect(s).toMatch(/^Against the operational analysis over the last 30 days: /);
    expect(s).toContain("at 50–200\u00A0m its error is 1.24\u00A0°C, 19\u00A0% lower than the seasonal climatology (1.53\u00A0°C)");
    expect(s).toMatch(/It beats the climatology in every depth band\.$/);
  });

  it("has nothing to say without a verification", () => {
    expect(verificationSentence(null, "argo", clim)).toBeNull();
    expect(verificationSentence({ bands, argo: { daily: [] } }, "argo", clim)).toBeNull();
    expect(rollingBands(null, null)).toEqual([]);
  });

  it("counts the profiles per day over the rolling window", () => {
    expect(profilesPerDay(v.argo, 30)).toEqual({ min: 4, max: 20 });
    expect(profilesPerDay(v.argo, 1)).toEqual({ min: 7, max: 7 });
    expect(profilesPerDay(v.analysis, 30)).toBeNull();
  });
});

describe("input shift", () => {
  const shift: LiveInputShift = {
    headline: {
      period: ["2025-10-01", "2026-03-31"],
      n_days: 182,
      recon_difference_pooled_50_200m: { bias: 0.03, rmse: 0.1707 },
      vs_glorys_all: { rmse_reprocessed: 1.0356, rmse_nrt: 1.0263 },
      measurably_worse: false,
    },
  };

  it("states the change, the overlap period and both errors against the reanalysis", () => {
    expect(inputShiftSentence(shift)).toBe(
      "With near-real-time instead of reprocessed inputs the reconstruction changes by 0.17\u00A0°C RMSE at 50–200\u00A0m (182 days, 1\u00A0Oct\u00A02025 – 31\u00A0Mar\u00A02026). " +
        "Its error against the GLORYS reanalysis over that period is 1.026\u00A0°C with near-real-time inputs and 1.036\u00A0°C with reprocessed ones: near-real-time inputs do not degrade it.",
    );
  });

  it("says so when near-real-time inputs make it worse, and nothing when the check has not run", () => {
    const worse = { headline: { ...shift.headline, vs_glorys_all: { rmse_reprocessed: 1.0, rmse_nrt: 1.2 }, measurably_worse: true } };
    expect(inputShiftSentence(worse)).toMatch(/near-real-time inputs make it measurably worse\.$/);
    expect(inputShiftSentence(null)).toBeNull();
    expect(inputShiftSentence({ headline: null })).toBeNull();
  });
});

describe("revisions and pending days", () => {
  const rev: LiveRevision = {
    policy: { revision_days: 7, rule: "the newest revision_days reconstructed days are fetched again on every update; a day whose inputs changed is reconstructed again" },
    n_days_checked: 7,
    n_days_revised: 0,
    by_age: [
      { age_days: 2, n_checks: 2, n_changed_since_first: 0, recon_rmse_50_200_max: 0 },
      { age_days: 3, n_checks: 1, n_changed_since_first: 0, recon_rmse_50_200_max: 0 },
    ],
  };

  it("states the policy with its number of days, and the statistics", () => {
    expect(revisionPolicy(rev)).toBe("The newest 7 reconstructed days are fetched again on every update; a day whose inputs changed is reconstructed again.");
    expect(revisionSentence(rev)).toBe("7 days have been re-checked so far (3 checks); none changed after first publication.");
    const changed = { ...rev, n_days_revised: 2, by_age: [{ age_days: 2, n_checks: 4, n_changed_since_first: 2, recon_rmse_50_200_max: 0.031 }] };
    expect(revisionSentence(changed)).toBe("7 days have been re-checked so far (4 checks); 2 changed after first publication, by at most 0.03\u00A0°C RMSE at 50–200\u00A0m.");
    expect(revisionSentence({ n_days_checked: 0 })).toBe("No day has been re-checked yet.");
    expect(revisionSentence(null)).toBeNull();
    expect(revisionPolicy({})).toBeNull();
  });

  it("names the input a pending day is waiting for", () => {
    const inputs = [
      { product: "sst", dataset: "a", version: null, last: "2026-10-03", available: true },
      { product: "sla", dataset: "b", version: null, last: "2026-10-04", available: true },
    ] as LiveInput[];
    expect(waitingFor("2026-10-04", inputs)).toEqual(["sst"]);
    const name = (p: string) => (p === "sst" ? "sea surface temperature" : "sea level anomaly");
    expect(pendingSentences({ pending: ["2026-10-04"], inputs }, name)).toEqual(["4\u00A0Oct\u00A02026 is waiting for sea surface temperature."]);
    expect(pendingSentences({ pending: [], inputs }, name)).toEqual([]);
  });
});
