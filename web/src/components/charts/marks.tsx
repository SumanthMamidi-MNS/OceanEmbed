/** Marker glyphs and legend swatches shared by every chart, so a method looks the same everywhere. */
import type { MarkerShape, MethodStyle } from "@/lib/methods";

export function markerPath(shape: MarkerShape, r: number): string {
  switch (shape) {
    case "square":
      return `M${-r * 0.86},${-r * 0.86}h${r * 1.72}v${r * 1.72}h${-r * 1.72}z`;
    case "diamond":
      return `M0,${-r * 1.2}L${r * 1.2},0L0,${r * 1.2}L${-r * 1.2},0z`;
    case "triangle":
      return `M0,${-r * 1.15}L${r * 1.1},${r * 0.85}L${-r * 1.1},${r * 0.85}z`;
    case "triangle-down":
      return `M0,${r * 1.15}L${r * 1.1},${-r * 0.85}L${-r * 1.1},${-r * 0.85}z`;
    case "cross":
      return `M${-r},${-r}L${r},${r}M${-r},${r}L${r},${-r}`;
    case "circle":
    case "ring":
    default:
      return `M${-r},0a${r},${r} 0 1,0 ${r * 2},0a${r},${r} 0 1,0 ${-r * 2},0`;
  }
}

/** One marker centred on (x, y). Filled glyphs get a surface ring so overlapping marks separate. */
export function Marker(props: { shape: MarkerShape; x: number; y: number; r?: number; color: string; ring?: string }) {
  const { shape, x, y, r = 3.25, color, ring = "var(--surface)" } = props;
  const open = shape === "cross" || shape === "ring";
  return (
    <path
      d={markerPath(shape, r)}
      transform={`translate(${x.toFixed(1)},${y.toFixed(1)})`}
      fill={open ? (shape === "ring" ? ring : "none") : color}
      stroke={open ? color : ring}
      strokeWidth={open ? 1.6 : 1}
      strokeLinecap="round"
    />
  );
}

/** Line + marker sample for legends and tooltips. */
export function Swatch({ style, width = 30 }: { style: MethodStyle; width?: number }) {
  return (
    <svg className="swatch" width={width} height={12} viewBox={`0 0 ${width} 12`} aria-hidden="true">
      <line
        x1={1}
        x2={width - 1}
        y1={6}
        y2={6}
        stroke={style.color}
        strokeWidth={style.width}
        strokeDasharray={style.dash || undefined}
        strokeLinecap={style.dash ? "butt" : "round"}
      />
      <Marker shape={style.marker} x={width / 2} y={6} r={3} color={style.color} />
    </svg>
  );
}

export interface LegendItem {
  key: string;
  label: string;
  style: MethodStyle;
  note?: string;
}

/** A legend is always present for two or more series; identity never depends on colour alone. */
export function Legend({ items, label = "Legend", band }: { items: readonly LegendItem[]; label?: string; band?: string | null }) {
  return (
    <ul className="legend" aria-label={label}>
      {items.map((it) => (
        <li key={it.key} className="legend__item" title={it.note}>
          <Swatch style={it.style} />
          <span>{it.label}</span>
        </li>
      ))}
      {band && (
        <li className="legend__item">
          <span className="legend__band" aria-hidden="true" />
          <span>{band}</span>
        </li>
      )}
    </ul>
  );
}
