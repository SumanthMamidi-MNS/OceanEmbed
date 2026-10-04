/**
 * Which surface products the run's model actually uses. The API marks each input product with
 * `used_by_model`; a run can carry products that are harmonised and shown but not fed to the
 * network, and the interface must never count those as inputs.
 */
import type { DataProduct, RunDetail } from "@/api/types";

export interface InputUse {
  /** every surface input product of the run */
  all: DataProduct[];
  used: DataProduct[];
  unused: DataProduct[];
  /** variable keys of the used products */
  usedKeys: ReadonlySet<string>;
  /** true when some available product is not used */
  partial: boolean;
}

export function inputUse(detail: Pick<RunDetail, "products">): InputUse {
  const all = detail.products.filter((p) => p.role === "input");
  const used = all.filter((p) => p.used_by_model !== false);
  const unused = all.filter((p) => p.used_by_model === false);
  return { all, used, unused, usedKeys: new Set(used.map((p) => p.variable)), partial: unused.length > 0 };
}

function lowerFirst(s: string): string {
  return /^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s;
}

/** "sea surface temperature and sea level anomaly" */
export function inputNames(products: readonly DataProduct[]): string {
  const names = products.map((p) => lowerFirst(p.long_name));
  if (names.length <= 2) return names.join(" and ");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** A panel of several variables (currents: uo + vo) feeds the model only if all of them do. */
export function panelUsed(panelKey: string, use: InputUse): boolean {
  return panelKey.split("+").every((k) => use.usedKeys.has(k));
}
