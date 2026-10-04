/**
 * Sentences and tables of the Live view, computed from the payload of GET /runs/{run}/live: how
 * the same-day reconstruction is checked (against Argo floats and against the operational
 * analysis, next to the climatology, band by band, including the bands where it does not beat
 * the climatology), what near-real-time inputs do to it, how days are revised and which day is
 * waiting for an input. Nothing is pre-written: every number and every verdict is read from the
 * payload.
 */
import type { LiveBandScore, LiveInput, LiveInputShift, LiveReference, LiveResponse, LiveRevision, LiveVerification } from "@/api/types";
import { fmtDate, fmtSpan } from "./dates";
import { fmt, fmtInt, fmtSigned } from "./format";
import { compare, describeComparison, joinList, type Comparison } from "./narrative";

/** A mean difference from the reference below this (°C) is not mentioned. */
export const LIVE_BIAS_NOTE = 0.3;

export interface LiveBand {
  key: string;
  /** "50–200 m" */
  label: string;
  score: LiveBandScore | null;
  /** the reconstruction's error against the climatology's (null without both) */
  verdict: Comparison | null;
}

/** Depth bands of the verification in the API's order, shallowest first. */
export function bandKeys(v: LiveVerification | null | undefined): string[] {
  const keys = Object.keys(v?.bands ?? {});
  const top = (k: string) => v?.bands?.[k]?.depth_range_m?.[0] ?? Number.POSITIVE_INFINITY;
  return keys.sort((a, b) => top(a) - top(b));
}

export function bandLabel(v: LiveVerification | null | undefined, key: string): string {
  const r = v?.bands?.[key]?.depth_range_m;
  if (r && r.length >= 2) return `${fmt(r[0], 0)}–${fmt(r[1], 0)}\u00A0m`;
  return (v?.bands?.[key]?.label ?? key.replace(/_/g, "–")).replace(/-/g, "–");
}

/** The rolling scores of the newest day of one reference, band by band. */
export function rollingBands(v: LiveVerification | null | undefined, ref: LiveReference | null | undefined): LiveBand[] {
  const latest = ref?.latest ?? ref?.daily?.[ref.daily.length - 1] ?? null;
  if (!v || !latest) return [];
  return bandKeys(v).map((key) => {
    const score = latest.bands?.[key]?.rolling ?? null;
    return { key, label: bandLabel(v, key), score, verdict: compare(score?.model_rmse, score?.clim_rmse) };
  });
}

/** Days of the rolling window, as the newest rolling block reports them. */
export function rollingDays(v: LiveVerification | null | undefined, ref: LiveReference | null | undefined): number | null {
  const latest = ref?.latest ?? null;
  const n = Object.values(latest?.bands ?? {})
    .map((b) => b.rolling?.n_days)
    .filter((x): x is number => x != null);
  return n.length ? Math.max(...n) : (v?.rolling_days ?? null);
}

/** Smallest and largest number of Argo profiles on a day, over the days of the rolling window. */
export function profilesPerDay(ref: LiveReference | null | undefined, days: number | null): { min: number; max: number } | null {
  const counts = (ref?.daily ?? [])
    .slice(days != null && days > 0 ? -days : 0)
    .map((d) => d.n_profiles)
    .filter((n): n is number => n != null);
  return counts.length ? { min: Math.min(...counts), max: Math.max(...counts) } : null;
}

/**
 * One reference in words: band by band, the reconstruction's rolling error next to the
 * climatology's, stated as it is. When a band is not better than the climatology the sentence
 * says so, and names the mean difference when there is one.
 */
export function verificationSentence(
  v: LiveVerification | null | undefined,
  which: "argo" | "analysis",
  clim: string,
): string | null {
  const ref = which === "argo" ? v?.argo : v?.analysis;
  const bands = rollingBands(v, ref).filter((b) => b.score?.model_rmse != null && b.verdict);
  if (bands.length === 0) return null;
  const days = rollingDays(v, ref);
  const against = which === "argo" ? "Argo floats" : "the operational analysis";
  const parts = bands.map((b) => {
    const s = b.score!;
    const c = b.verdict!;
    return c.direction === "same"
      ? `at ${b.label} its error is about the same as that of ${clim} (${fmt(s.model_rmse)} vs ${fmt(s.clim_rmse)}\u00A0°C)`
      : `at ${b.label} its error is ${fmt(s.model_rmse)}\u00A0°C, ${describeComparison(c, clim)} (${fmt(s.clim_rmse)}\u00A0°C)`;
  });
  let out = `Against ${against} over the last ${days != null ? `${fmtInt(days)} days` : "days"}: ${parts.join("; ")}.`;
  const lost = bands.filter((b) => b.verdict!.direction !== "lower");
  if (lost.length > 0) {
    out += ` It does not beat the climatology at ${joinList(lost.map((b) => b.label))}.`;
    const warm = lost.filter((b) => b.score?.model_bias != null && Math.abs(b.score.model_bias) >= LIVE_BIAS_NOTE);
    if (warm.length > 0) {
      const other = which === "argo" ? "the floats" : "the analysis";
      out += ` There the reconstruction is on average ${joinList(
        warm.map((b) => `${fmt(Math.abs(b.score!.model_bias!))}\u00A0°C ${b.score!.model_bias! > 0 ? "warmer" : "colder"} than ${other} at ${b.label}`),
      )}.`;
    }
  } else {
    out += " It beats the climatology in every depth band.";
  }
  return out;
}

/**
 * What near-real-time inputs do to the reconstruction, measured on an overlap period where both
 * the near-real-time and the reprocessed inputs exist: how much the reconstruction changes, and
 * its error against the reanalysis with either input set.
 */
export function inputShiftSentence(shift: LiveInputShift | null | undefined): string | null {
  const h = shift?.headline;
  const diff = h?.recon_difference_pooled_50_200m?.rmse;
  if (!h || diff == null) return null;
  const period = h.period && h.period.length >= 2 ? fmtSpan(h.period[0], h.period[1]) : null;
  const over = [h.n_days != null ? `${fmtInt(h.n_days)} days` : null, period].filter(Boolean).join(", ");
  let out = `With near-real-time instead of reprocessed inputs the reconstruction changes by ${fmt(diff)}\u00A0°C RMSE at 50–200\u00A0m${over ? ` (${over})` : ""}.`;
  const g = h.vs_glorys_all;
  if (g?.rmse_nrt != null && g.rmse_reprocessed != null) {
    const verdict =
      h.measurably_worse === true
        ? "near-real-time inputs make it measurably worse"
        : h.measurably_worse === false
          ? "near-real-time inputs do not degrade it"
          : `a difference of ${fmtSigned(g.rmse_nrt - g.rmse_reprocessed, 3)}\u00A0°C`;
    out += ` Its error against the GLORYS reanalysis over that period is ${fmt(g.rmse_nrt, 3)}\u00A0°C with near-real-time inputs and ${fmt(g.rmse_reprocessed, 3)}\u00A0°C with reprocessed ones: ${verdict}.`;
  }
  return out;
}

/** The revision policy in one line (the API's rule with its number of days). */
export function revisionPolicy(r: LiveRevision | null | undefined): string | null {
  const rule = r?.policy?.rule;
  if (!rule) return null;
  const n = r?.policy?.revision_days;
  const text = n != null ? rule.replace(/\brevision_days\b/g, String(n)) : rule;
  return `${text[0].toUpperCase()}${text.slice(1)}.`;
}

/** How often days were looked at again, and how many of them changed. */
export function revisionSentence(r: LiveRevision | null | undefined): string | null {
  if (!r || r.n_days_checked == null) return null;
  const checks = (r.by_age ?? []).reduce((sum, a) => sum + (a.n_checks ?? 0), 0);
  const days = r.n_days_checked;
  if (days === 0) return "No day has been re-checked yet.";
  const changed = r.n_days_revised ?? 0;
  const largest = Math.max(0, ...(r.by_age ?? []).map((a) => a.recon_rmse_50_200_max ?? 0));
  return (
    `${fmtInt(days)} day${days === 1 ? " has" : "s have"} been re-checked so far${checks > 0 ? ` (${fmtInt(checks)} checks)` : ""}; ` +
    (changed === 0
      ? "none changed after first publication."
      : `${fmtInt(changed)} changed after first publication${largest > 0 ? `, by at most ${fmt(largest)}\u00A0°C RMSE at 50–200\u00A0m` : ""}.`)
  );
}

/** Which input a pending day is waiting for: the inputs whose catalogue does not reach that day yet. */
export function waitingFor(date: string, inputs: readonly LiveInput[]): string[] {
  return inputs.filter((i) => !i.last || i.last < date).map((i) => i.product);
}

/** "4 Oct 2026 is waiting for sea surface temperature." for each pending day. */
export function pendingSentences(live: Pick<LiveResponse, "pending" | "inputs">, nameOf: (product: string) => string): string[] {
  return (live.pending ?? []).map((d) => {
    const missing = waitingFor(d, live.inputs).map(nameOf);
    return missing.length > 0 ? `${fmtDate(d)} is waiting for ${joinList(missing)}.` : `${fmtDate(d)} is not reconstructed yet.`;
  });
}
