/** Number and label formatting. Real minus sign, fixed decimals, en dash for missing values. */

export const MISSING = "–";
const MINUS = "−";
const NBSP = "\u00A0";

export function fmt(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  const s = Math.abs(value).toFixed(digits);
  const isZero = Number(s) === 0;
  return value < 0 && !isZero ? MINUS + s : s;
}

/** Signed number: always shows + or the real minus sign. */
export function fmtSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  const s = Math.abs(value).toFixed(digits);
  if (Number(s) === 0) return s;
  return (value < 0 ? MINUS : "+") + s;
}

export function fmtInt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  const s = Math.abs(Math.round(value))
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, "\u202F");
  return value < 0 && Math.round(value) !== 0 ? MINUS + s : s;
}

export function fmtPct(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  return `${fmt(value, digits)}${NBSP}%`;
}

/** Compact count: 30769032 -> "30.8 M". */
export function fmtCompact(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  const a = Math.abs(value);
  if (a >= 1e9) return `${(value / 1e9).toFixed(1)}${NBSP}G`;
  if (a >= 1e6) return `${(value / 1e6).toFixed(1)}${NBSP}M`;
  if (a >= 1e4) return `${(value / 1e3).toFixed(0)}${NBSP}k`;
  return fmtInt(value);
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return MISSING;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)}${NBSP}GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)}${NBSP}MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)}${NBSP}kB`;
  return `${bytes}${NBSP}B`;
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return MISSING;
  if (seconds < 90) return `${Math.round(seconds)}${NBSP}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}${NBSP}min`;
  return `${(seconds / 3600).toFixed(1)}${NBSP}h`;
}

export function fmtDepth(depth: number): string {
  return `${Number.isInteger(depth) ? depth : depth.toFixed(1)}${NBSP}m`;
}

export function fmtLat(lat: number, digits = 2): string {
  return `${Math.abs(lat).toFixed(digits)}°${lat < 0 ? "S" : "N"}`;
}

export function fmtLon(lon: number, digits = 2): string {
  const l = ((((lon + 180) % 360) + 360) % 360) - 180;
  return `${Math.abs(l).toFixed(digits)}°${l < 0 ? "W" : "E"}`;
}

export function fmtLatLon(lat: number, lon: number, digits = 2): string {
  return `${fmtLat(lat, digits)}, ${fmtLon(lon, digits)}`;
}

/** API unit strings -> typographic units. */
export function prettyUnits(units: string | null | undefined): string {
  if (!units) return "";
  const u = units.trim();
  if (u === "degC" || u === "degree_Celsius") return "°C";
  if (u === "m s-1") return "m/s";
  return u;
}

/** Fix ASCII unit / degree spellings inside API-provided prose ("0.25 deg" -> "0.25°"). */
export function prettyText(text: string): string {
  return text
    .replace(/\bdegC\b/g, "°C")
    .replace(/(\d)\s*deg(?:rees?)?\b/g, "$1°")
    .replace(/\bm s-1\b/g, "m/s")
    .replace(/(\d)x(\d)/g, "$1×$2");
}
