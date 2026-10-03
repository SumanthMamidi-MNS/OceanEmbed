/**
 * Horizontal bars comparing methods on one number. Every bar is labelled with its method and its
 * value, so colour is redundant; bars share one zero-based axis.
 */
import { fmt } from "@/lib/format";
import type { MethodStyle } from "@/lib/methods";

export interface MethodBarRow {
  key: string;
  label: string;
  value: number | null;
  style: MethodStyle;
  /** short remark after the value, e.g. "−33 % vs climatology" */
  note?: string;
  emphasis?: boolean;
}

export function MethodBars(props: {
  rows: readonly MethodBarRow[];
  unit: string;
  digits?: number;
  label: string;
  max?: number;
  /** one line per bar (label, track, value) instead of the value above the track */
  inline?: boolean;
}) {
  const { rows, unit, digits = 2, label, inline } = props;
  const max = props.max ?? Math.max(0, ...rows.map((r) => (r.value != null && Number.isFinite(r.value) ? r.value : 0)));
  return (
    <div className={`bars ${inline ? "bars--inline" : ""}`} role="table" aria-label={label}>
      {rows.map((r) => {
        const w = r.value != null && max > 0 ? Math.max(0, Math.min(100, (r.value / max) * 100)) : 0;
        return (
          <div key={r.key} className={`bars__row ${r.emphasis ? "is-lead" : ""}`} role="row">
            <span className="bars__label" role="rowheader">
              {r.label}
            </span>
            <span className="bars__track" role="presentation">
              <span className="bars__bar" style={{ width: `${w}%`, background: r.style.color }} />
            </span>
            <span className="bars__value num" role="cell">
              {fmt(r.value, digits)}
              <span className="bars__unit"> {unit}</span>
              {r.note && <span className="bars__note">{r.note}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}
