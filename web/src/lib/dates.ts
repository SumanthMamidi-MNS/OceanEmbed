/** Dates are ISO `YYYY-MM-DD` strings in UTC everywhere (the API's convention). */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const NBSP = "\u00A0";

export function isIsoDate(s: string | null | undefined): s is string {
  if (!s || !/^\d{4}-\d{2}-\d{2}$/.test(s)) return false;
  const t = Date.parse(`${s}T00:00:00Z`);
  return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === s;
}

export function toUtcMs(date: string): number {
  return Date.parse(`${date.slice(0, 10)}T00:00:00Z`);
}

export function fromUtcMs(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

/** Inclusive number of days from start to end (0 if either is missing or reversed). */
export function daysInclusive(start: string | null | undefined, end: string | null | undefined): number {
  if (!isIsoDate(start) || !isIsoDate(end)) return 0;
  const d = Math.round((toUtcMs(end) - toUtcMs(start)) / 86400000) + 1;
  return d > 0 ? d : 0;
}

/** "16 Mar 2024" */
export function fmtDate(date: string | null | undefined): string {
  if (!date || !isIsoDate(date.slice(0, 10))) return "–";
  const [y, m, d] = date.slice(0, 10).split("-").map(Number);
  return `${d}${NBSP}${MONTHS[m - 1]}${NBSP}${y}`;
}

/** "Mar 2024" */
export function fmtMonth(date: string): string {
  const [y, m] = date.slice(0, 7).split("-").map(Number);
  return `${MONTHS[m - 1]}${NBSP}${y}`;
}

export function monthShort(date: string): string {
  return MONTHS[Number(date.slice(5, 7)) - 1];
}

/** "1 Jan 2018 – 15 Dec 2024" */
export function fmtSpan(start: string | null | undefined, end: string | null | undefined): string {
  if (!start || !end) return "–";
  return `${fmtDate(start)} – ${fmtDate(end)}`;
}

/**
 * Day shown when the link names none: the day nearest the middle of the period that every given
 * set contains (e.g. the days with a GLORYS target and the days with an embedding), so the first
 * view of a run has every panel filled. Empty sets are ignored; -1 for an empty list.
 */
export function defaultDateIndex(dates: readonly string[], required: readonly ReadonlySet<string>[] = []): number {
  if (dates.length === 0) return -1;
  const mid = Math.floor(dates.length / 2);
  const sets = required.filter((s) => s.size > 0);
  if (sets.length === 0) return mid;
  for (let off = 0; off < dates.length; off++) {
    for (const i of off === 0 ? [mid] : [mid + off, mid - off]) {
      if (i >= 0 && i < dates.length && sets.every((s) => s.has(dates[i]))) return i;
    }
  }
  return mid;
}

/** Index of the date in a sorted list nearest to `date` (0 when no date is wished, -1 for an empty list). */
export function nearestDateIndex(dates: readonly string[], date: string | null | undefined): number {
  if (dates.length === 0) return -1;
  if (!date || !isIsoDate(date)) return 0;
  const exact = dates.indexOf(date);
  if (exact >= 0) return exact;
  const t = toUtcMs(date);
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < dates.length; i++) {
    const d = Math.abs(toUtcMs(dates[i]) - t);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}
