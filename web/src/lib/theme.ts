/**
 * Design tokens — the single source of truth for colour. `applyTheme()` publishes them as CSS
 * custom properties (`--paper`, `--ink`, ...); canvas and SVG code reads the same object, so a map,
 * a chart and a button can never drift apart. Documented in docs/design.md.
 */

export const color = {
  /** page background: warm paper white */
  paper: "#F7F6F2",
  /** figure and panel surface */
  surface: "#FFFFFF",
  /** wells: table headers, inactive segments, code */
  sunken: "#EFEDE6",
  /** hairlines and borders */
  rule: "#DDD9CE",
  /** axis lines, control borders */
  ruleStrong: "#B4AFA2",
  /** primary text: blue-black ink (15.2:1 on paper) */
  ink: "#14212B",
  /** secondary text, axis labels (7.6:1) */
  ink2: "#42515C",
  /** captions, overlines, tick labels (5.3:1; 4.9:1 on the sunken tone) */
  ink3: "#5A6872",
  /** the one accent: links, active control, focus, selection */
  accent: "#0A5A78",
  accentStrong: "#06445C",
  accentSoft: "#DDEBF0",
  /** synthetic-data treatment only */
  warn: "#F3BC3D",
  warnInk: "#2B1D00",
  warnSoft: "#FDF3D7",
  warnRule: "#C99512",
  /** short-training caution only */
  cautionSoft: "#ECE9F4",
  cautionRule: "#6A5FA3",
  cautionInk: "#2B2552",
  /** error text */
  danger: "#A3261C",
  dangerSoft: "#FAE9E6",
  /** maps: land is a flat neutral, never a data colour */
  land: "#D6D1C4",
  /** maps: ocean at the surface but below the sea floor at this depth */
  seafloor: "#B9BCB7",
  coast: "#2E3B44",
  graticule: "rgba(20, 33, 43, 0.16)",
  /** chart grid lines (one step off the surface) */
  grid: "#ECE9E1",
  /** selection marker on maps: ink with a paper halo */
  marker: "#14212B",
  markerHalo: "#FFFFFF",
} as const;

export type ColorToken = keyof typeof color;

export const font = {
  display: '"Newsreader Variable", "Newsreader", "Iowan Old Style", Georgia, serif',
  sans: '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif',
  mono: '"IBM Plex Mono", "Cascadia Mono", Consolas, monospace',
} as const;

/** Canvas font shorthands (the canvas API cannot read CSS variables). */
export const canvasFont = {
  tick: `400 11px ${font.sans}`,
  tickSmall: `400 10px ${font.sans}`,
  label: `500 11px ${font.sans}`,
  basin: `500 10px ${font.mono}`,
} as const;

function kebab(name: string): string {
  return name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
}

/** Publish the tokens as CSS custom properties on the root element. */
export function applyTheme(root: HTMLElement = document.documentElement): void {
  for (const [name, value] of Object.entries(color)) root.style.setProperty(`--${kebab(name)}`, value);
  root.style.setProperty("--font-display", font.display);
  root.style.setProperty("--font-sans", font.sans);
  root.style.setProperty("--font-mono", font.mono);
}
