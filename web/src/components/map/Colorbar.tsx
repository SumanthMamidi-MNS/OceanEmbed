/** Horizontal colour bar with ticks and units. Pointed ends mean the range is clipped (percentiles). */
import { cssGradient, type ColormapName } from "@/lib/colormaps";
import { fmt } from "@/lib/format";
import { niceTicks, tickDigits } from "@/lib/scales";

export interface ColorbarSpec {
  cmap: ColormapName;
  vmin: number;
  vmax: number;
  /** already typographic, e.g. "°C" */
  units: string;
  reversed?: boolean;
  /** what the colour encodes, e.g. "Temperature" */
  label?: string;
  /** which ends are clipped */
  extend?: "both" | "max" | "none";
  /** sign always shown (differences, anomalies) */
  signed?: boolean;
}

export function Colorbar({ spec, ticks = 5 }: { spec: ColorbarSpec; ticks?: number }) {
  const { cmap, vmin, vmax, units, reversed, label, extend = "both", signed } = spec;
  const ok = Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin;
  let values = ok ? niceTicks(vmin, vmax, ticks) : [];
  // never more labels than asked for: thin them, keeping zero on diverging scales
  while (values.length > ticks + 1) {
    const zero = values.indexOf(0);
    const keep = zero >= 0 ? zero % 2 : 0;
    values = values.filter((_, i) => i % 2 === keep);
  }
  const digits = tickDigits(values);
  const text = (v: number) => {
    const s = fmt(v, digits);
    return signed && v > 0 ? `+${s}` : s;
  };
  return (
    <div className="colorbar" role="img" aria-label={`Colour scale${label ? ` for ${label}` : ""}: ${ok ? `${text(vmin)} to ${text(vmax)}` : "no data"} ${units}`}>
      <div className={`colorbar__bar colorbar__bar--${extend}`}>
        <div className="colorbar__fill" style={{ background: cssGradient(cmap, reversed) }} />
      </div>
      <div className="colorbar__ticks" aria-hidden="true">
        {values.map((v) => (
          <span key={v} className="colorbar__tick num" style={{ left: `${((v - vmin) / (vmax - vmin)) * 100}%` }}>
            {text(v)}
          </span>
        ))}
      </div>
      <div className="colorbar__label" aria-hidden="true">
        {label ? `${label} ` : ""}
        <span className="colorbar__units">{units}</span>
      </div>
    </div>
  );
}
