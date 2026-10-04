/**
 * Metric-to-sentence helpers. Every statement of a finding in the interface is produced here from
 * the run's own metrics, so the text can never disagree with the numbers and nothing is
 * pre-written: when the model loses to a baseline, the sentence says so.
 */
import type { MetricBlock, MetricsResponse, RunSummary } from "@/api/types";
import { fmt, fmtDepth, fmtInt } from "./format";
import { isAblation, shortLabel, sortMethods } from "./methods";

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

function rmseOf(block: MetricBlock | undefined): number | null {
  const v = block?.rmse;
  return v != null && Number.isFinite(v) ? v : null;
}

function rangeText(range: readonly number[] | undefined): string {
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

export interface Headline {
  /** pooled depth range, e.g. "50–200 m" */
  range: string;
  model: number | null;
  ridge: number | null;
  climatology: number | null;
  vsClimatology: Comparison | null;
  vsRidge: Comparison | null;
  /** the model beats both baselines by more than the tie threshold */
  beatsBaselines: boolean;
  sentence: string;
}

/** The headline skill statement over the pooled (thermocline) depth range. */
export function headline(metrics: MetricsResponse, referenceName = "GLORYS"): Headline {
  const range = rangeText(metrics.pooled_range_m);
  const model = rmseOf(metrics.pooled.model);
  const ridge = rmseOf(metrics.pooled.ridge);
  const climatology = rmseOf(metrics.pooled.climatology);
  const vsClimatology = compare(model, climatology);
  const vsRidge = compare(model, ridge);
  // each test year next to the pooled number, when the test period spans several
  const years = yearsOf(metrics)
    .map((y) => ({ y, v: rmseOf(metrics.per_year?.[y]?.pooled?.model) }))
    .filter((x) => x.v != null);
  const perYear = years.length >= 2 ? ` (${years.map((x) => `${fmt(x.v)} in ${x.y}`).join(", ")})` : "";
  const parts: string[] = [];
  if (vsClimatology) parts.push(`${describeComparison(vsClimatology, "climatology")} (${fmt(climatology)}\u00A0°C)`);
  if (vsRidge) parts.push(`${describeComparison(vsRidge, "ridge regression")} (${fmt(ridge)}\u00A0°C)`);
  const sentence =
    model == null
      ? `No pooled metrics are available for ${range}.`
      : `Between ${range}, OceanEmbed's RMSE against ${referenceName} is ${fmt(model)}\u00A0°C${perYear}` +
        (parts.length ? `: ${parts.join(" and ")}.` : ".");
  return {
    range,
    model,
    ridge,
    climatology,
    vsClimatology,
    vsRidge,
    beatsBaselines: vsClimatology?.direction === "lower" && vsRidge?.direction === "lower",
    sentence,
  };
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

export interface AblationFinding {
  key: string;
  label: string;
  model: number;
  ablation: number;
  /** positive = the main model is better */
  comparison: Comparison;
  sentence: string;
}

/**
 * Two trainings of the same network differ by a few percent from the random seed alone. A pooled
 * RMSE difference below this (in % of the ablation's RMSE) between the main model and an
 * ablation trained once is therefore reported as "no measurable gain", in either direction.
 */
export const SEED_PCT = 5;

/** "level", or how far `value` is from `reference`, with the single-seed threshold. */
export function versus(value: number | null | undefined, reference: number | null | undefined): { c: Comparison; level: boolean } | null {
  const c = compare(value, reference);
  return c ? { c, level: c.direction === "same" || Math.abs(c.pct) < SEED_PCT } : null;
}

/**
 * The main model against each ablation over the pooled range, stated as it is. When the two differ
 * in pretraining the sentence answers "does pretraining help?" whichever of them is the ablation
 * (the pretrained network can be the main model or the ablation, depending on the run).
 */
export function ablationFindings(metrics: MetricsResponse): AblationFinding[] {
  const model = rmseOf(metrics.pooled.model);
  if (model == null) return [];
  const main = metrics.methods.find((m) => m.key === "model");
  const range = rangeText(metrics.pooled_range_m);
  const out: AblationFinding[] = [];
  for (const m of metrics.methods) {
    if (!isAblation(m.key)) continue;
    const abl = rmseOf(metrics.pooled[m.key]);
    const c = compare(model, abl);
    if (abl == null || !c) continue;
    let sentence: string;
    const aboutPretraining = main?.pretrained != null && m.pretrained != null && main.pretrained !== m.pretrained;
    if (aboutPretraining) {
      // read the pair as "with pretraining" against "without", whichever is the main model
      const mainPretrained = main?.pretrained === true;
      const withPre = mainPretrained ? model : abl;
      const without = mainPretrained ? abl : model;
      const v = versus(withPre, without)!;
      const numbers = `${fmt(withPre)}\u00A0°C with the pretrained encoder, ${fmt(without)}\u00A0°C trained from scratch`;
      const which = `the main model is the one ${mainPretrained ? "with pretraining" : "trained from scratch"}`;
      if (v.level) {
        sentence =
          `No measurable gain from pretraining in this run: pooled RMSE (${range}) is ${numbers} (${which}). ` +
          (v.c.direction === "same" ? "" : `The ${fmt(Math.abs(v.c.pct), 0)}\u00A0% between them is `) +
          (v.c.direction === "same" ? "Each was trained once." : "within what a different training seed can produce; each was trained once.");
      } else {
        sentence =
          `${v.c.direction === "lower" ? "Pretraining helps" : "Pretraining does not help"} in this run: pooled RMSE (${range}) is ` +
          `${fmt(Math.abs(v.c.pct), 0)}\u00A0% ${v.c.direction} with the pretrained encoder (${numbers}; ${which}).`;
      }
    } else {
      const v = versus(model, abl)!;
      const numbers = `${fmt(model)} vs ${fmt(abl)}\u00A0°C`;
      sentence = v.level
        ? `No measurable difference in this run: pooled RMSE (${range}) is ${numbers} for ${m.label}, within what a different training seed can produce.`
        : `The main model is ${v.c.direction === "lower" ? "ahead" : "behind"} in this run: its pooled RMSE (${range}) is ${fmt(Math.abs(v.c.pct), 0)}\u00A0% ${v.c.direction} than ${m.label} (${numbers}).`;
    }
    out.push({ key: m.key, label: m.label, model, ablation: abl, comparison: c, sentence });
  }
  return out;
}

/**
 * Do the test years agree? One sentence: whether the model beats climatology and ridge regression
 * in each year, with each year's numbers. Null for a single-year test period.
 */
export function yearStability(metrics: MetricsResponse): string | null {
  const years = yearsOf(metrics);
  if (years.length < 2) return null;
  const rows = years
    .map((y) => {
      const p = metrics.per_year?.[y]?.pooled ?? {};
      const model = rmseOf(p.model);
      const clim = rmseOf(p.climatology);
      const ridge = rmseOf(p.ridge);
      return { y, model, clim, ridge, beatsClim: compare(model, clim)?.direction === "lower", beatsRidge: ridge == null || compare(model, ridge)?.direction === "lower" };
    })
    .filter((r) => r.model != null && r.clim != null);
  if (rows.length < 2) return null;
  const numbers = rows.map((r) => `${r.y}: ${fmt(r.model)} vs ${fmt(r.clim)}${r.ridge != null ? ` and ${fmt(r.ridge)}` : ""}\u00A0°C`).join("; ");
  const hasRidge = rows.every((r) => r.ridge != null);
  const names = hasRidge ? "climatology and ridge regression" : "climatology";
  const failing = rows.filter((r) => !r.beatsClim || !r.beatsRidge);
  if (failing.length === 0) return `It beats ${names} in each test year (${numbers}).`;
  const which = failing.map((r) => `${r.y} (${[!r.beatsClim ? "climatology" : null, !r.beatsRidge ? "ridge regression" : null].filter(Boolean).join(" and ")})`);
  return `It does not beat every baseline in every test year: not in ${joinList(which)} (${numbers}).`;
}

function lowerFirst(label: string): string {
  return /^[A-Z][a-z]/.test(label) ? label[0].toLowerCase() + label.slice(1) : label;
}

/**
 * The network against the per-pixel MLP (same inputs, no spatial context): what the spatial model
 * adds. Pooled, then basin by basin; a difference under the single-seed threshold reads "level".
 * Null when the run has no MLP baseline.
 */
export function mlpSentence(metrics: MetricsResponse, labels: Readonly<Record<string, string>>): string | null {
  const model = rmseOf(metrics.pooled.model);
  const mlp = rmseOf(metrics.pooled.mlp);
  const whole = versus(model, mlp);
  if (!whole) return null;
  const name = `the ${lowerFirst(metrics.methods.find((m) => m.key === "mlp")?.label ?? "per-pixel MLP")}`;
  const say = (v: { c: Comparison; level: boolean }, a: number | null, b: number | null) =>
    `${v.level ? "level" : `${fmt(Math.abs(v.c.pct), 0)}\u00A0% ${v.c.direction}`} (${fmt(a)} vs ${fmt(b)}\u00A0°C)`;
  let s = `Against ${name}, which sees the same inputs one cell at a time, OceanEmbed's pooled RMSE over ${rangeText(metrics.pooled_range_m)} is ${say(whole, model, mlp)}`;
  const basins = Object.entries(metrics.per_basin ?? {})
    .map(([key, b]) => ({ label: labels[key] ?? key.replace(/_/g, " "), model: rmseOf(b.pooled?.model), mlp: rmseOf(b.pooled?.mlp) }))
    .map((b) => ({ ...b, v: versus(b.model, b.mlp) }))
    .filter((b) => b.v)
    .sort((a, b) => (b.v!.c.pct ?? 0) - (a.v!.c.pct ?? 0));
  if (basins.length >= 2) s += `; by basin: ${joinList(basins.map((b) => `${say(b.v!, b.model, b.mlp)} in the ${b.label}`))}`;
  return `${s}. “Level” means within what a different training seed can produce.`;
}

export interface DepthSkill {
  bestDepth: number | null;
  bestSkill: number | null;
  /** depths where the model's skill vs climatology is <= 0 */
  noSkillDepths: number[];
  /** depths where the skill is at least {@link CLEAR_SKILL} */
  clearDepths: number[];
  /** depths where ridge has a lower RMSE than the model (beyond the tie threshold) */
  ridgeWinsDepths: number[];
  sentence: string | null;
}

function joinList(parts: string[]): string {
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

/** A skill against climatology below this is called marginal: the MSE is within 10 % of the climatology's. */
export const CLEAR_SKILL = 0.1;

/** Where in the water column the skill is, and where it is not. */
export function depthSkill(metrics: MetricsResponse): DepthSkill {
  const skill = metrics.per_depth.skill_vs_clim?.model ?? [];
  const rmseModel = metrics.per_depth.rmse?.model ?? [];
  const rmseRidge = metrics.per_depth.rmse?.ridge ?? [];
  const depths = metrics.depths;
  let bestDepth: number | null = null;
  let bestSkill: number | null = null;
  const noSkillDepths: number[] = [];
  const clearDepths: number[] = [];
  const marginalDepths: number[] = [];
  const ridgeWinsDepths: number[] = [];
  depths.forEach((d, k) => {
    const s = skill[k];
    if (s != null && Number.isFinite(s)) {
      if (bestSkill == null || s > bestSkill) {
        bestSkill = s;
        bestDepth = d;
      }
      if (s <= 0) noSkillDepths.push(d);
      else if (s < CLEAR_SKILL) marginalDepths.push(d);
      else clearDepths.push(d);
    }
    const c = compare(rmseModel[k], rmseRidge[k]);
    if (c && c.direction === "higher") ridgeWinsDepths.push(d);
  });
  if (bestDepth == null || bestSkill == null) {
    return { bestDepth, bestSkill, noSkillDepths, clearDepths, ridgeWinsDepths, sentence: null };
  }
  const parts: string[] = [];
  if (bestSkill > 0) {
    const clear = clearDepths.length > 0 && clearDepths.length < depths.length ? ` and at least ${fmt(CLEAR_SKILL, 1)} at ${describeDepths(clearDepths, depths)}` : "";
    parts.push(`Skill against climatology is highest at ${fmtDepth(bestDepth)} (${fmt(bestSkill)})${clear}`);
    if (marginalDepths.length > 0) parts.push(`it is marginal (below ${fmt(CLEAR_SKILL, 1)}) at ${describeDepths(marginalDepths, depths)}`);
  } else {
    parts.push(`The model does not beat climatology at any depth (best skill ${fmt(bestSkill)} at ${fmtDepth(bestDepth)})`);
  }
  if (bestSkill > 0 && noSkillDepths.length > 0) {
    parts.push(`the model does not beat climatology at ${describeDepths(noSkillDepths, depths)}`);
  }
  if (ridgeWinsDepths.length > 0) {
    parts.push(`ridge regression has the lower RMSE at ${describeDepths(ridgeWinsDepths, depths)}`);
  }
  return { bestDepth, bestSkill, noSkillDepths, clearDepths, ridgeWinsDepths, sentence: `${parts.join("; ")}.` };
}

/** Mean differences against Argo smaller than this (°C) are not called a bias. */
export const BIAS_NOTE = 0.1;

function warmer(bias: number): string {
  return `${fmt(Math.abs(bias))}\u00A0°C ${bias > 0 ? "warmer" : "colder"}`;
}

/**
 * The Argo comparison in words, over the pooled depth range when the payload has it (else over all
 * depths): the model next to its baselines, GLORYS itself as the floor, and the mean difference
 * that GLORYS and the model share, when there is one. Numbers only; no interpretation beyond them.
 */
export function argoSentence(argo: MetricsResponse, scope: "pooled" | "overall" = "pooled"): string | null {
  const pooled = scope === "pooled" && rmseOf(argo.pooled?.model) != null;
  const blocks = pooled ? argo.pooled : argo.overall;
  const model = rmseOf(blocks.model);
  if (model == null) return null;
  const nProfiles = argo.metadata.n_profiles_used ?? null;
  const matchups = pooled ? (blocks.model?.n ?? null) : (argo.metadata.n_matchups ?? blocks.model?.n ?? null);
  const where = pooled ? `between ${rangeText(argo.pooled_range_m)}` : "over all depths";
  const clim = rmseOf(blocks.climatology);
  const ridge = rmseOf(blocks.ridge);
  const glorys = rmseOf(blocks.glorys);
  const bits: string[] = [];
  const vsClim = compare(model, clim);
  if (vsClim) bits.push(`${describeComparison(vsClim, "climatology")} (${fmt(clim)}\u00A0°C)`);
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
 * unevaluated one, and the longest predicted period.
 */
export function defaultRun(runs: readonly RunSummary[]): RunSummary | null {
  if (runs.length === 0) return null;
  const score = (r: RunSummary) =>
    (r.artefacts.predictions && r.n_prediction_days > 0 ? 8 : 0) +
    (r.data_source === "real" ? 4 : 0) +
    (runCaveats(r).shortTraining ? 0 : 2) +
    (r.artefacts.metrics_glorys ? 1 : 0);
  return [...runs].sort((a, b) => score(b) - score(a) || b.n_prediction_days - a.n_prediction_days || a.name.localeCompare(b.name))[0];
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
