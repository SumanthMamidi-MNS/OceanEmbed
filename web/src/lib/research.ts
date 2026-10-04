/**
 * Method-comparison sentences, for the Research area only: the model against ridge regression,
 * against the per-pixel network and against its ablation (pretraining). Computed from the run's
 * metrics like every other sentence (lib/narrative.ts); they answer "why this model", which a
 * user of the product does not need, so no product view imports this file.
 */
import type { MetricsResponse } from "@/api/types";
import { fmt, fmtDepth } from "./format";
import { isAblation } from "./methods";
import { CLEAR_SKILL, compare, describeComparison, describeDepths, joinList, rangeText, rmseOf, yearsOf, type Comparison } from "./narrative";

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

