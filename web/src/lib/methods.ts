/**
 * One fixed visual identity per method, everywhere: colour + dash + marker, so nothing depends on
 * colour alone. Colours follow the entity (its key), never its rank in a list.
 * The chromatic slots pass the dataviz palette validator on the light surface (lightness band,
 * chroma floor, CVD separation >= 8, contrast >= 3:1).
 */
import { color } from "./theme";

export type MarkerShape = "circle" | "square" | "diamond" | "triangle" | "triangle-down" | "pentagon" | "cross" | "ring";

export interface MethodStyle {
  color: string;
  /** SVG stroke-dasharray ("" = solid) */
  dash: string;
  marker: MarkerShape;
  width: number;
  /** drawing order: higher is drawn later (on top) */
  z: number;
}

const MODEL: MethodStyle = { color: "#0B5FA5", dash: "", marker: "circle", width: 2.25, z: 10 };
const RIDGE: MethodStyle = { color: "#C8431F", dash: "7 2.5 1.5 2.5", marker: "square", width: 1.75, z: 4 };
/** The per-pixel MLP baseline: same inputs as the network, no spatial context. */
const MLP: MethodStyle = { color: "#D95FA8", dash: "3 2", marker: "pentagon", width: 1.75, z: 5 };
const GLORYS: MethodStyle = { color: "#0E8A6A", dash: "10 3", marker: "diamond", width: 1.75, z: 6 };
const CLIM: MethodStyle = { color: "#6B7680", dash: "1.5 3", marker: "cross", width: 1.75, z: 2 };
const OBS: MethodStyle = { color: color.ink, dash: "", marker: "ring", width: 1.5, z: 12 };
const ABLATIONS: readonly MethodStyle[] = [
  { color: "#B8730A", dash: "5 3", marker: "triangle", width: 1.75, z: 8 },
  { color: "#8A4FB0", dash: "12 3 2 3", marker: "triangle-down", width: 1.75, z: 7 },
];

/** Aliases used by different API payloads for the same entity. */
const CANONICAL: Record<string, string> = {
  clim: "climatology",
  prediction: "model",
  predicted: "model",
  target: "glorys",
  argo: "obs",
};

export function canonicalMethod(key: string): string {
  return CANONICAL[key] ?? key;
}

/**
 * Style of a method. Ablations (`model_<tag>`) take the ablation slots in the order of
 * `ablationKeys` (the run's sorted ablation list), so an ablation keeps its look on every view.
 */
export function methodStyle(key: string, ablationKeys: readonly string[] = []): MethodStyle {
  const k = canonicalMethod(key);
  if (k === "model") return MODEL;
  if (k === "ridge") return RIDGE;
  if (k === "mlp") return MLP;
  if (k === "glorys") return GLORYS;
  if (k === "climatology") return CLIM;
  if (k === "obs") return OBS;
  if (k.startsWith("model_")) {
    const idx = Math.max(0, ablationKeys.indexOf(k));
    return ABLATIONS[idx % ABLATIONS.length];
  }
  return CLIM;
}

export function isAblation(key: string): boolean {
  return canonicalMethod(key).startsWith("model_");
}

/** Sorted ablation keys among a list of method keys. */
export function ablationKeysOf(keys: readonly string[]): string[] {
  return keys.map(canonicalMethod).filter(isAblation).sort();
}

/** Legend order: OceanEmbed first, then ablations, the MLP, ridge, climatology, GLORYS, observations. */
export function methodOrder(key: string): number {
  const k = canonicalMethod(key);
  if (k === "model") return 0;
  if (k.startsWith("model_")) return 1;
  if (k === "mlp") return 2;
  if (k === "ridge") return 3;
  if (k === "climatology") return 4;
  if (k === "glorys") return 5;
  if (k === "obs") return 6;
  return 7;
}

export function sortMethods<T extends string>(keys: readonly T[]): T[] {
  return [...keys].sort((a, b) => methodOrder(a) - methodOrder(b) || a.localeCompare(b));
}

const SHORT: Record<string, string> = {
  model: "OceanEmbed",
  ridge: "Ridge",
  climatology: "Climatology",
  glorys: "GLORYS",
  obs: "Argo",
};

/** Short label for legends and table heads; falls back to the API label. */
export function shortLabel(key: string, apiLabel?: string | null): string {
  const k = canonicalMethod(key);
  if (SHORT[k]) return SHORT[k];
  if (k.startsWith("model_")) {
    const tag = k.slice("model_".length);
    return tag === "scratch" ? "No pretraining" : tag === "pretrained" ? "Pretrained" : `Ablation: ${tag}`;
  }
  return apiLabel ?? key;
}

/** What an estimate is, in a few words, for the line under a map title. */
export function estimateRole(key: string): string {
  const k = canonicalMethod(key);
  if (k === "model") return "from surface fields only";
  if (k === "ridge") return "linear baseline on the same surface fields";
  if (k === "mlp") return "per-pixel baseline: no spatial context";
  if (k === "model_pretrained") return "same network, pretrained encoder";
  if (k === "model_scratch") return "same network, no pretraining";
  if (k.startsWith("model_")) return "ablation of the network";
  return "";
}
