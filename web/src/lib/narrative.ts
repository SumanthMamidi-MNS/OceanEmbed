/**
 * Metric-to-sentence helpers for the product views (Overview, Accuracy). Every statement of a
 * result in the interface is produced from the run's own metrics, so the text can never disagree
 * with the numbers and nothing is pre-written: when the reconstruction is no better than the
 * seasonal climatology, the sentence says so.
 *
 * What a user needs is here: how large the error is against the climatology (the estimate one has
 * without this product), where not to trust it, the basins, the test years, the Argo floats.
 * Sentences that compare methods (ridge regression, the per-pixel network, pretraining) are in
 * lib/research.ts and appear only in the Research area.
 */
import type { MetricBlock, MetricsResponse, RunSummary } from "@/api/types";
import { fmt, fmtDepth, fmtInt } from "./format";
import { shortLabel, sortMethods } from "./methods";

/** Differences smaller than this (in % of the reference RMSE) are reported as "about the same". */
export const TIE_PCT = 1;

/** A training period shorter than one annual cycle cannot constrain a seasonal climatology. */
export const MIN_TRAIN_DAYS = 365;

export interface Comparison {
  /** percentage by which `value` is below the reference (negative = above) */
  pct: number;
  direction: "lower" | "higher" | "same";
}

export function compare(value: number | null | undefined, reference: number | null | undefined): Comparison | null {
  if (value == null || reference == null || !Number.isFinite(value) || !Number.isFinite(reference) || reference === 0) {
    return null;
  }
  const pct = ((reference - value) / Math.abs(reference)) * 100;
  const direction = Math.abs(pct) < TIE_PCT ? "same" : pct > 0 ? "lower" : "higher";
  return { pct, direction };
}

/** "33 % lower than climatology" / "4 % higher than ridge regression" / "about the same as ..." */
export function describeComparison(c: Comparison, referenceName: string): string {
  if (c.direction === "same") return `about the same as ${referenceName}`;
  return `${fmt(Math.abs(c.pct), 0)}\u00A0% ${c.direction} than ${referenceName}`;
}

export function rmseOf(block: MetricBlock | undefined): number | null {
  const v = block?.rmse;
  return v != null && Number.isFinite(v) ? v : null;
}

export function rangeText(range: readonly number[] | null | undefined): string {
  if (!range || range.length < 2) return "the pooled depth range";
  return `${fmt(range[0], 0)}–${fmt(range[1], 0)}\u00A0m`;
}

/** Calendar years of a test period that spans several (empty for a single-year period). */
export function yearsOf(metrics: MetricsResponse | null | undefined): string[] {
  const keys = Object.keys(metrics?.per_year ?? {}).sort();
  return keys.length >= 2 ? keys : [];
}

/**
 * The same payload restricted to one test year: the whole-period blocks are replaced by that
 * year's, so every table, chart and sentence built from a metrics payload works per year too.
 * The daily series are left as they are (they already carry their dates).
 */
export function scopeToYear(metrics: MetricsResponse, year: string | null | undefined): MetricsResponse {
  const block = year ? metrics.per_year?.[year] : null;
  if (!block) return metrics;
  return {
    ...metrics,
    // the Argo counts of that year, when the block carries them
    metadata: {
      ...metrics.metadata,
      ...(block.n_profiles != null ? { n_profiles_used: block.n_profiles } : {}),
      ...(block.n_matchups != null ? { n_matchups: block.n_matchups } : {}),
      ...(block.n_days != null ? { n_days: block.n_days } : {}),
    },
    overall: block.overall ?? {},
    pooled: block.pooled ?? {},
    per_depth: block.per_depth ?? {},
    per_basin: block.per_basin ?? {},
    per_year: null,
  };
}

/** What the climatology is called in a sentence when it has a seasonal cycle. */
const SEASONAL = "the seasonal climatology";

/**
 * The climatology's name in the product's sentences: "seasonal" only when the fit has a seasonal
 * cycle (a run trained on less than a year has a constant mean, which must not be called seasonal).
 */
export function climatologyName(terms: number | null | undefined): string {
  return terms != null && terms <= 1 ? "the mean climatology" : SEASONAL;
}

export interface Accuracy {
  /** pooled depth range, e.g. "50–200 m" */
  range: string;
  model: number | null;
  climatology: number | null;
  vsClimatology: Comparison | null;
  /** the reconstruction's error is below the climatology's by more than the tie threshold */
  beatsClimatology: boolean;
  sentence: string;
}

/**
 * How accurate, in the simplest truthful form: the error over the pooled (thermocline) range
 * against the reference, each test year, and how it compares with the seasonal climatology.
 */
export function accuracyHeadline(metrics: MetricsResponse, clim: string = SEASONAL, referenceName = "GLORYS reanalysis"): Accuracy {
  const range = rangeText(metrics.pooled_range_m);
  const model = rmseOf(metrics.pooled.model);
  const climatology = rmseOf(metrics.pooled.climatology);
  const vsClimatology = compare(model, climatology);
  const years = yearsOf(metrics)
    .map((y) => ({ y, v: rmseOf(metrics.per_year?.[y]?.pooled?.model) }))
    .filter((x) => x.v != null);
  const perYear = years.length >= 2 ? ` (${years.map((x) => `${fmt(x.v)} in ${x.y}`).join(", ")})` : "";
  const sentence =
    model == null
      ? `No scores are available for ${range}.`
      : `Between ${range} the reconstruction differs from the ${referenceName} by ${fmt(model)}\u00A0°C RMSE${perYear}` +
        (vsClimatology ? `: ${describeComparison(vsClimatology, clim)} (${fmt(climatology)}\u00A0°C).` : ".");
  return { range, model, climatology, vsClimatology, beatsClimatology: vsClimatology?.direction === "lower", sentence };
}

/**
 * Skill against climatology (1 − MSE / MSE of the climatology) of a depth level, in three bands:
 * clearly useful at or above {@link CLEAR_SKILL}, marginal from {@link MARGINAL_SKILL} up to it,
 * and no better than the climatology below that.
 */
export const CLEAR_SKILL = 0.2;
export const MARGINAL_SKILL = 0.05;

/**
 * Down to this depth the model is close to its own input: it is given the sea surface temperature,
 * so a high skill there says little. The depth where the reconstruction "helps most" is looked for
 * below it.
 */
export const SURFACE_INPUT_DEPTH_M = 30;

export interface DepthClasses {
  /** the level with the highest skill among the levels below {@link SURFACE_INPUT_DEPTH_M} (any level if there is none) */
  bestDepth: number | null;
  bestSkill: number | null;
  /** depths where the skill against climatology is at least {@link CLEAR_SKILL} */
  clearDepths: number[];
  /** depths where it is at least {@link MARGINAL_SKILL} but below {@link CLEAR_SKILL} */
  marginalDepths: number[];
  /** depths where it is below {@link MARGINAL_SKILL}: no better than the climatology */
  noSkillDepths: number[];
}

/** The depth levels sorted by the reconstruction's skill against climatology: clear, marginal, none. */
export function classifyDepths(metrics: MetricsResponse): DepthClasses {
  const skill = metrics.per_depth.skill_vs_clim?.model ?? [];
  const out: DepthClasses = { bestDepth: null, bestSkill: null, clearDepths: [], marginalDepths: [], noSkillDepths: [] };
  const below = metrics.depths.some((d, k) => d > SURFACE_INPUT_DEPTH_M && skill[k] != null && Number.isFinite(skill[k]));
  metrics.depths.forEach((d, k) => {
    const s = skill[k];
    if (s == null || !Number.isFinite(s)) return;
    if ((!below || d > SURFACE_INPUT_DEPTH_M) && (out.bestSkill == null || s > out.bestSkill)) {
      out.bestSkill = s;
      out.bestDepth = d;
    }
    if (s < MARGINAL_SKILL) out.noSkillDepths.push(d);
    else if (s < CLEAR_SKILL) out.marginalDepths.push(d);
    else out.clearDepths.push(d);
  });
  return out;
}

export interface TrustLimits extends DepthClasses {
  /** where the reconstruction is worth using */
  use: string | null;
  /** where it is not: the note a user must read before using a value */
  limit: string | null;
  /** true when some depth has no skill or only a marginal one */
  limited: boolean;
}

/** "down to 200 m" when the levels are the top of the column, "from 500 m" when they are its bottom, else "at ...". */
function where(selected: readonly number[], levels: readonly number[]): string {
  const sorted = [...selected].sort((a, b) => a - b);
  const index = sorted.map((d) => levels.indexOf(d));
  const contiguous = index.every((k, i) => k >= 0 && (i === 0 || k === index[i - 1] + 1));
  if (contiguous && sorted.length > 1 && sorted.length < levels.length) {
    if (index[0] === 0) return `down to ${fmtDepth(sorted[sorted.length - 1])}`;
    if (index[index.length - 1] === levels.length - 1) return `from ${fmtDepth(sorted[0])}`;
  }
  return `at ${describeDepths(sorted, levels)}`;
}

/**
 * Where to use the reconstruction and where not to trust it, from the skill against the
 * climatology at each depth: clearly useful, marginal, no better than the climatology. The depth
 * where it helps most is named among the levels below the near-surface layer, because near the
 * surface the model is given the sea surface temperature.
 */
export function trustLimits(metrics: MetricsResponse, clim: string = SEASONAL): TrustLimits {
  const c = classifyDepths(metrics);
  const levels = metrics.depths;
  if (c.bestDepth == null || c.bestSkill == null) return { ...c, use: null, limit: null, limited: false };
  if (c.clearDepths.length === 0) {
    const any = c.marginalDepths.length > 0;
    return {
      ...c,
      use: null,
      limit: any
        ? `The reconstruction is only marginally better than ${clim} (best skill ${fmt(c.bestSkill)} at ${fmtDepth(c.bestDepth)}): treat it as no more reliable than the climatology.`
        : `The reconstruction does not beat ${clim} at any depth (best skill ${fmt(c.bestSkill)} at ${fmtDepth(c.bestDepth)}): do not rely on it.`,
      limited: true,
    };
  }
  const surface = levels.some((d) => d <= SURFACE_INPUT_DEPTH_M) && c.bestDepth > SURFACE_INPUT_DEPTH_M ? "; near the surface the model is given the sea surface temperature" : "";
  const use = `It clearly improves on ${clim} ${where(c.clearDepths, levels)} (most at ${fmtDepth(c.bestDepth)}, skill ${fmt(c.bestSkill)})${surface}.`;
  if (c.marginalDepths.length + c.noSkillDepths.length === 0) return { ...c, use, limit: null, limited: false };
  const parts: string[] = [];
  if (c.marginalDepths.length > 0) parts.push(`the gain is marginal ${where(c.marginalDepths, levels)}`);
  if (c.noSkillDepths.length > 0) parts.push(`it is no better than ${clim} ${where(c.noSkillDepths, levels)}`);
  const text = parts.join(" and ");
  return { ...c, use, limit: `${text[0].toUpperCase()}${text.slice(1)}: there the climatology is as good an estimate.`, limited: true };
}

/** Anomaly correlation next to the raw one, with the reason the raw one is inflated. */
export function correlationSentence(metrics: MetricsResponse): string | null {
  const block = metrics.pooled.model;
  const anom = block?.corr_anom;
  const raw = block?.corr_raw;
  if (anom == null || raw == null) return null;
  return (
    `Its anomaly correlation over ${rangeText(metrics.pooled_range_m)} is ${fmt(anom)} ` +
    `(raw correlation ${fmt(raw)}; the raw value is inflated by the seasonal cycle and by spatial and ` +
    `vertical gradients that any climatology already reproduces).`
  );
}

/**
 * Do the test years agree? One sentence: whether the reconstruction beats the seasonal climatology
 * in each year, with each year's numbers. With `baselines = "all"` (the Research area) ridge
 * regression is held to the same test. Null for a single-year test period.
 */
export function yearStability(metrics: MetricsResponse, baselines: "climatology" | "all" = "climatology", clim: string = SEASONAL): string | null {
  const years = yearsOf(metrics);
  if (years.length < 2) return null;
  const rows = years
    .map((y) => {
      const p = metrics.per_year?.[y]?.pooled ?? {};
      const model = rmseOf(p.model);
      const clim = rmseOf(p.climatology);
      const ridge = baselines === "all" ? rmseOf(p.ridge) : null;
      return { y, model, clim, ridge, beatsClim: compare(model, clim)?.direction === "lower", beatsRidge: ridge == null || compare(model, ridge)?.direction === "lower" };
    })
    .filter((r) => r.model != null && r.clim != null);
  if (rows.length < 2) return null;
  const numbers = rows.map((r) => `${r.y}: ${fmt(r.model)} vs ${fmt(r.clim)}${r.ridge != null ? ` and ${fmt(r.ridge)}` : ""}\u00A0°C`).join("; ");
  const hasRidge = rows.every((r) => r.ridge != null);
  const names = hasRidge ? "climatology and ridge regression" : baselines === "all" ? "climatology" : clim;
  const failing = rows.filter((r) => !r.beatsClim || !r.beatsRidge);
  if (failing.length === 0) return `It beats ${names} in each test year (${numbers}).`;
  const which = failing.map((r) => `${r.y} (${[!r.beatsClim ? "climatology" : null, !r.beatsRidge ? "ridge regression" : null].filter(Boolean).join(" and ")})`);
  if (baselines !== "all") return `It does not beat ${clim} in every test year: not in ${joinList(failing.map((r) => r.y))} (${numbers}).`;
  return `It does not beat every baseline in every test year: not in ${joinList(which)} (${numbers}).`;
}

export function joinList(parts: string[]): string {
  if (parts.length <= 2) return parts.join(" and ");
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** More separate depth ranges than this are summarised as a count, not listed. */
export const MAX_DEPTH_SPANS = 3;

/**
 * A set of depth levels in words. Neighbouring levels are merged into ranges ("50–100 m"), a
 * range that reaches the deepest level reads "300 m and below", and a scattered set is
 * summarised ("7 of the 15 levels between 5 m and 700 m") instead of listed, so the sentence
 * stays readable however many levels the grid has.
 */
export function describeDepths(selected: readonly number[], levels: readonly number[]): string {
  const index = new Map(levels.map((d, k) => [d, k]));
  const ks = selected
    .map((d) => index.get(d))
    .filter((k): k is number => k != null)
    .sort((a, b) => a - b);
  if (ks.length === 0) return "";
  if (ks.length === levels.length && levels.length > 1) return "every depth";
  const spans: [number, number][] = [];
  for (const k of ks) {
    const last = spans[spans.length - 1];
    if (last && k === last[1] + 1) last[1] = k;
    else spans.push([k, k]);
  }
  if (spans.length > MAX_DEPTH_SPANS) {
    return `${ks.length} of the ${levels.length} levels between ${fmtDepth(levels[ks[0]])} and ${fmtDepth(levels[ks[ks.length - 1]])}`;
  }
  const deepest = levels.length - 1;
  return joinList(
    spans.map(([a, b]) => {
      if (a === b) return fmtDepth(levels[a]);
      if (b === deepest) return `${fmtDepth(levels[a])} and below`;
      return `${fmt(levels[a], 0)}–${fmtDepth(levels[b])}`;
    }),
  );
}

/** Mean differences against Argo smaller than this (°C) are not called a bias. */
export const BIAS_NOTE = 0.1;

function warmer(bias: number): string {
  return `${fmt(Math.abs(bias))}\u00A0°C ${bias > 0 ? "warmer" : "colder"}`;
}

/**
 * The Argo comparison in words, over the pooled depth range when the payload has it (else over all
 * depths): the reconstruction next to the climatology (and to ridge regression with
 * `baselines = "all"`, the Research area), GLORYS itself as the floor, and the mean difference
 * that GLORYS and the model share, when there is one. Numbers only; no interpretation beyond them.
 */
export function argoSentence(
  argo: MetricsResponse,
  scope: "pooled" | "overall" = "pooled",
  baselines: "climatology" | "all" = "climatology",
  clim: string = SEASONAL,
): string | null {
  const pooled = scope === "pooled" && rmseOf(argo.pooled?.model) != null;
  const blocks = pooled ? argo.pooled : argo.overall;
  const model = rmseOf(blocks.model);
  if (model == null) return null;
  const nProfiles = argo.metadata.n_profiles_used ?? null;
  const matchups = pooled ? (blocks.model?.n ?? null) : (argo.metadata.n_matchups ?? blocks.model?.n ?? null);
  const where = pooled ? `between ${rangeText(argo.pooled_range_m)}` : "over all depths";
  const climRmse = rmseOf(blocks.climatology);
  const ridge = baselines === "all" ? rmseOf(blocks.ridge) : null;
  const glorys = rmseOf(blocks.glorys);
  const bits: string[] = [];
  const vsClim = compare(model, climRmse);
  if (vsClim) bits.push(`${describeComparison(vsClim, clim)} (${fmt(climRmse)}\u00A0°C)`);
  const vsRidge = compare(model, ridge);
  if (vsRidge) bits.push(`${describeComparison(vsRidge, "ridge regression")} (${fmt(ridge)}\u00A0°C)`);
  const sample = nProfiles != null ? `Against ${fmtInt(nProfiles)} Argo profiles` : "Against the Argo profiles";
  let s = `${sample}, ${where}${matchups != null ? ` (${fmtInt(matchups)} profile–depth matchups)` : ""}, OceanEmbed's RMSE is ${fmt(model)}\u00A0°C`;
  if (bits.length) s += `: ${bits.join(" and ")}`;
  s += ".";
  if (glorys != null) {
    s += ` GLORYS itself differs from the same profiles by ${fmt(glorys)}\u00A0°C: that is the floor, since a model trained on GLORYS cannot agree with Argo better than GLORYS does.`;
    const bg = blocks.glorys?.bias;
    const bm = blocks.model?.bias;
    if (bg != null && bm != null && Math.abs(bg) >= BIAS_NOTE && Math.sign(bg) === Math.sign(bm)) {
      s += ` GLORYS is on average ${warmer(bg)} than Argo there, and the reconstruction ${warmer(bm)}.`;
    }
  }
  return s;
}

export interface BasinContrast {
  key: string;
  label: string;
  skill: number | null;
  model: number | null;
  climatology: number | null;
}

/** Pooled skill and RMSE of the model in each evaluation basin, the most skilful basin first. */
export function basinContrast(metrics: MetricsResponse, labels: Readonly<Record<string, string>>): BasinContrast[] {
  return Object.entries(metrics.per_basin ?? {})
    .map(([key, b]) => ({
      key,
      label: labels[key] ?? key.replace(/_/g, " "),
      skill: b.pooled?.model?.skill_vs_clim ?? null,
      model: rmseOf(b.pooled?.model),
      climatology: rmseOf(b.pooled?.climatology),
    }))
    .filter((b) => b.model != null)
    .sort((a, b) => (b.skill ?? -Infinity) - (a.skill ?? -Infinity));
}

/** "Over 50–200 m the skill against climatology is 0.55 in the Bay of Bengal (...) and 0.29 in the Arabian Sea (...)." */
export function basinSentence(metrics: MetricsResponse, labels: Readonly<Record<string, string>>): string | null {
  const basins = basinContrast(metrics, labels).filter((b) => b.skill != null);
  if (basins.length < 2) return null;
  const parts = basins.map((b, i) => `${fmt(b.skill)} in the ${b.label} (${i === 0 ? "RMSE " : ""}${fmt(b.model)} vs ${fmt(b.climatology)}\u00A0°C${i === 0 ? " for climatology" : ""})`);
  return `Over ${rangeText(metrics.pooled_range_m)} the skill against climatology is ${joinList(parts)}.`;
}

export interface Evidence {
  /** depth shown as evidence, metres (null: no metrics to choose from) */
  depth: number | null;
  /** the climatology's RMSE at that depth: why it was chosen */
  depthClimatologyRmse: number | null;
  date: string | null;
  /** the daily error that made the day typical, and the median it was compared with */
  dayRmse: number | null;
  nDays: number;
  /** what the daily error is measured over */
  dayScope: "pooled" | "depth" | null;
}

/**
 * The day and depth the overview shows as evidence, chosen from the metrics so that the picture is
 * neither flattering nor unkind. Depth: inside the pooled (thermocline) range, where the
 * climatology errs most, i.e. where there is most to gain (never the surface, which is an input).
 * Day: the test day whose error is the median of the daily errors (pooled range if the run has
 * that series, else at the chosen depth); only days in `available` qualify.
 */
export function chooseEvidence(metrics: MetricsResponse, available?: ReadonlySet<string> | null): Evidence {
  const depths = metrics.depths ?? [];
  const range = metrics.pooled_range_m;
  const inRange = (d: number) => !range || range.length < 2 || (d >= range[0] && d <= range[1]);
  const pick = (values: readonly (number | null)[] | undefined): number => {
    let best = -1;
    depths.forEach((d, k) => {
      const v = values?.[k];
      if (!inRange(d) || d <= 0 || v == null || !Number.isFinite(v)) return;
      if (best < 0 || v > (values?.[best] as number)) best = k;
    });
    return best;
  };
  const clim = metrics.per_depth?.rmse?.climatology;
  let k = pick(clim);
  if (k < 0) k = pick(metrics.per_depth?.rmse?.model);
  const depth = k >= 0 ? depths[k] : null;

  const daily = metrics.daily;
  const days = daily?.dates ?? metrics.daily_rmse?.dates ?? [];
  let values: (number | null)[] | null = null;
  let dayScope: Evidence["dayScope"] = null;
  if (daily?.pooled_rmse?.model) {
    values = daily.pooled_rmse.model;
    dayScope = "pooled";
  } else {
    const rows = daily?.rmse?.model ?? metrics.daily_rmse?.methods?.model;
    const levels = daily?.depths ?? metrics.daily_rmse?.depths ?? [];
    const kk = depth != null ? levels.indexOf(depth) : -1;
    if (rows && kk >= 0) {
      values = rows.map((r) => r?.[kk] ?? null);
      dayScope = "depth";
    }
  }
  const ranked: { date: string; v: number }[] = [];
  if (values) {
    days.forEach((d, t) => {
      const v = values?.[t];
      if (v != null && Number.isFinite(v) && (!available || available.size === 0 || available.has(d))) ranked.push({ date: d, v });
    });
  }
  ranked.sort((a, b) => a.v - b.v || a.date.localeCompare(b.date));
  const mid = ranked.length > 0 ? ranked[Math.floor((ranked.length - 1) / 2)] : null;
  return {
    depth,
    depthClimatologyRmse: k >= 0 ? (clim?.[k] ?? null) : null,
    date: mid?.date ?? null,
    dayRmse: mid?.v ?? null,
    nDays: ranked.length,
    dayScope: mid ? dayScope : null,
  };
}

/** The harmonic climatology in words, from the number of fitted terms (1, 3 or 5). */
export function climatologyText(terms: number | null | undefined): string {
  if (terms == null) return "harmonic fit";
  if (terms <= 1) return "constant mean (no seasonal cycle)";
  if (terms === 3) return "mean and annual cycle";
  if (terms === 5) return "mean, annual and semi-annual cycle";
  return `harmonic fit with ${terms} terms`;
}

export interface RunCaveats {
  synthetic: boolean;
  /** days of the training split present in the data (null when the API does not report it) */
  trainDays: number | null;
  /** trained on less than one annual cycle: maps are real, skill numbers are not meaningful */
  shortTraining: boolean;
  syntheticNote: string | null;
  shortTrainingNote: string | null;
}

/**
 * What a reader must know before trusting a run's numbers, from the run summary alone: the data
 * source, the number of training days and the terms of the climatology fit, all as reported by
 * the API (never guessed from the run's name or its dates).
 */
export function runCaveats(run: RunSummary): RunCaveats {
  const synthetic = run.data_source === "synthetic";
  const trainDays = run.n_train_days ?? null;
  const shortTraining = trainDays != null && trainDays > 0 && trainDays < MIN_TRAIN_DAYS;
  let shortTrainingNote: string | null = null;
  if (shortTraining) {
    const terms = run.n_harmonic_terms;
    const clim = terms != null && terms <= 1 ? " The climatology baseline is a constant mean, not a seasonal cycle." : "";
    shortTrainingNote =
      `This run was trained on ${fmtInt(trainDays)} days, less than one annual cycle.` +
      ` Its maps are real; its skill numbers are not meaningful.${clim}`;
  }
  return {
    synthetic,
    trainDays,
    shortTraining,
    syntheticNote: synthetic
      ? (run.note ?? "Synthetic data: the numbers only demonstrate that the pipeline works; they are not scientific skill.")
      : null,
    shortTrainingNote,
  };
}

/**
 * Run shown when the link names none. A run that has a reconstruction to show comes first (a run
 * still being produced must not be the landing page), then real data before synthetic, a run
 * trained on at least one annual cycle before a shorter one, an evaluated run before an
 * unevaluated one, and the longest predicted period. The live run is never the default: it has no
 * evaluation against the reanalysis and its days are revised; it has its own view.
 */
export function defaultRun(all: readonly RunSummary[]): RunSummary | null {
  const runs = evaluatedRuns(all);
  if (runs.length === 0) return null;
  const score = (r: RunSummary) =>
    (r.artefacts.predictions && r.n_prediction_days > 0 ? 8 : 0) +
    (r.data_source === "real" ? 4 : 0) +
    (runCaveats(r).shortTraining ? 0 : 2) +
    (r.artefacts.metrics_glorys ? 1 : 0);
  return [...runs].sort((a, b) => score(b) - score(a) || b.n_prediction_days - a.n_prediction_days || a.name.localeCompare(b.name))[0];
}

/** The runs the evaluated pages can show: every run except the live (rolling, near-real-time) one. */
export function evaluatedRuns(runs: readonly RunSummary[]): RunSummary[] {
  return runs.filter((r) => !r.live);
}

/** The live run, when the API reports one. */
export function liveRun(runs: readonly RunSummary[]): RunSummary | null {
  return runs.find((r) => r.live) ?? null;
}

/** Methods of a metrics payload in legend order, with short labels. */
export function methodList(metrics: MetricsResponse): { key: string; label: string; short: string }[] {
  const byKey = new Map(metrics.methods.map((m) => [m.key, m]));
  return sortMethods(Object.keys(metrics.overall)).map((key) => ({
    key,
    label: byKey.get(key)?.label ?? key,
    short: shortLabel(key, byKey.get(key)?.label),
  }));
}
