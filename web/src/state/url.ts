/**
 * Deep-linkable application state <-> URL. Pure functions (tested): the path selects the view and
 * the query string carries the shared selection (run, date, depth, point, estimate) plus per-view
 * options.
 */
import { isIsoDate } from "@/lib/dates";

export const VIEWS = [
  { key: "overview", path: "/", label: "Overview" },
  { key: "explore", path: "/explore", label: "Explorer" },
  { key: "validation", path: "/validation", label: "Validation" },
  { key: "representation", path: "/representation", label: "Representation" },
  { key: "experiments", path: "/experiments", label: "Experiments" },
] as const;

export type ViewKey = (typeof VIEWS)[number]["key"];

export interface UrlState {
  view: ViewKey;
  run: string | null;
  /** YYYY-MM-DD */
  date: string | null;
  /** metres */
  depth: number | null;
  lat: number | null;
  lon: number | null;
  /** the estimate shown where one method is shown: `model`, `ridge` or an ablation key (null = the main model) */
  est: string | null;
  /** per-view options (tab, metric, basin, ...), plain strings */
  opts: Readonly<Record<string, string>>;
}

export const EMPTY_STATE: UrlState = { view: "overview", run: null, date: null, depth: null, lat: null, lon: null, est: null, opts: {} };

const RUN_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const METHOD_RE = /^[a-z][a-z0-9_]{0,39}$/;
const SHARED = new Set(["run", "date", "depth", "lat", "lon", "est"]);
const OPT_KEY_RE = /^[a-z][a-z0-9_]{0,23}$/;

function viewFromPath(pathname: string): ViewKey {
  const clean = pathname.replace(/\/+$/, "") || "/";
  const hit = VIEWS.find((v) => v.path === clean);
  return hit ? hit.key : "overview";
}

function num(value: string | null, lo: number, hi: number): number | null {
  if (value == null || value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && n >= lo && n <= hi ? n : null;
}

/** Parse a location. Anything malformed is dropped (never thrown), so a bad link still opens. */
export function parseUrl(pathname: string, search: string): UrlState {
  const q = new URLSearchParams(search);
  const run = q.get("run");
  const date = q.get("date");
  const lat = num(q.get("lat"), -90, 90);
  const lon = num(q.get("lon"), -180, 360);
  const est = q.get("est");
  const opts: Record<string, string> = {};
  for (const [k, v] of q) {
    if (!SHARED.has(k) && OPT_KEY_RE.test(k) && v.length > 0 && v.length <= 120) opts[k] = v;
  }
  return {
    view: viewFromPath(pathname),
    run: run && RUN_RE.test(run) ? run : null,
    date: isIsoDate(date) ? date : null,
    depth: num(q.get("depth"), 0, 11000),
    // a point needs both coordinates
    lat: lat != null && lon != null ? lat : null,
    lon: lat != null && lon != null ? lon : null,
    est: est && METHOD_RE.test(est) ? est : null,
    opts,
  };
}

function trimNumber(n: number, digits: number): string {
  return String(Number(n.toFixed(digits)));
}

/** Serialise a state to `path?query` (stable key order, no empty values). */
export function formatUrl(state: UrlState): string {
  const path = VIEWS.find((v) => v.key === state.view)?.path ?? "/";
  const q = new URLSearchParams();
  if (state.run) q.set("run", state.run);
  if (state.date) q.set("date", state.date);
  if (state.depth != null) q.set("depth", trimNumber(state.depth, 1));
  if (state.lat != null && state.lon != null) {
    q.set("lat", trimNumber(state.lat, 3));
    q.set("lon", trimNumber(state.lon, 3));
  }
  if (state.est) q.set("est", state.est);
  for (const k of Object.keys(state.opts).sort()) {
    if (state.opts[k]) q.set(k, state.opts[k]);
  }
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

export interface UrlPatch {
  view?: ViewKey;
  run?: string | null;
  date?: string | null;
  depth?: number | null;
  lat?: number | null;
  lon?: number | null;
  est?: string | null;
  /** merged into the current options; a null value removes the key */
  opts?: Record<string, string | null>;
}

/**
 * Apply a patch. Changing the view drops the per-view options (the shared selection survives);
 * changing the run drops the date, the estimate and the options, which belong to the old run.
 */
export function applyPatch(state: UrlState, patch: UrlPatch): UrlState {
  const viewChanged = patch.view !== undefined && patch.view !== state.view;
  const runChanged = patch.run !== undefined && patch.run !== state.run;
  let opts: Record<string, string> = viewChanged || runChanged ? {} : { ...state.opts };
  if (patch.opts) {
    opts = { ...opts };
    for (const [k, v] of Object.entries(patch.opts)) {
      if (v == null || v === "") delete opts[k];
      else opts[k] = v;
    }
  }
  const next: UrlState = {
    view: patch.view ?? state.view,
    run: patch.run !== undefined ? patch.run : state.run,
    date: patch.date !== undefined ? patch.date : runChanged ? null : state.date,
    depth: patch.depth !== undefined ? patch.depth : state.depth,
    lat: patch.lat !== undefined ? patch.lat : state.lat,
    lon: patch.lon !== undefined ? patch.lon : state.lon,
    est: patch.est !== undefined ? patch.est : runChanged ? null : state.est,
    opts,
  };
  if (next.lat == null || next.lon == null) {
    next.lat = null;
    next.lon = null;
  }
  return next;
}

/** Link to another view that keeps the shared selection. */
export function viewHref(state: UrlState, view: ViewKey, extra?: UrlPatch): string {
  return formatUrl(applyPatch(state, { ...extra, view }));
}
