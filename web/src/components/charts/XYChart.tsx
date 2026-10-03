/**
 * The one line chart of the application (SVG). It draws depth profiles (depth on a square-root
 * axis, increasing downward), time series and training curves with the same axes, grid, hover and
 * tooltip, so every chart reads the same way. One axis pair only: never a dual axis.
 */
import { useMemo, useRef, useState, type ReactNode } from "react";
import { fromUtcMs, monthShort } from "@/lib/dates";
import type { MethodStyle } from "@/lib/methods";
import { depthFraction, depthFromFraction, depthTicks, niceTicks, tickDigits } from "@/lib/scales";
import { fmt } from "@/lib/format";
import { useSize } from "@/lib/useSize";
import { Marker, Swatch } from "./marks";

export interface ChartPoint {
  x: number | null;
  y: number | null;
}

export interface ChartSeries {
  key: string;
  label: string;
  style: MethodStyle;
  points: ChartPoint[];
  markers?: boolean;
}

export type ScaleKind = "linear" | "depth" | "time";

export interface AxisSpec {
  /** axis title including units, e.g. "RMSE (°C)" */
  label: string;
  domain: [number, number];
  scale?: ScaleKind;
  ticks?: number[];
  format?: (v: number) => string;
}

export interface XYChartProps {
  series: readonly ChartSeries[];
  x: AxisSpec;
  y: AxisSpec;
  height?: number;
  /** axis the cross-hair snaps along: "y" for depth profiles, "x" for time series */
  hover?: "x" | "y";
  /** emphasised zero line across the given axis (for bias / skill) */
  zero?: "x" | "y" | null;
  /** shaded range, e.g. the pooled depth range */
  band?: { axis: "x" | "y"; from: number; to: number; label?: string } | null;
  /** reference line, e.g. the depth or day selected elsewhere */
  reference?: { axis: "x" | "y"; value: number; label?: string } | null;
  onPick?: (value: number) => void;
  ariaLabel: string;
  /** title of the tooltip for a hovered axis value */
  hoverTitle?: (value: number) => string;
  /** extra rows under the tooltip */
  tooltipExtra?: (value: number) => ReactNode;
  compact?: boolean;
}

interface Scale {
  (v: number): number;
  invert: (px: number) => number;
}

function makeScale(spec: AxisSpec, lo: number, hi: number, flip: boolean): Scale {
  const [d0, d1] = spec.domain;
  const kind = spec.scale ?? "linear";
  const toT = (v: number) => (kind === "depth" ? depthFraction(v - d0, d1 - d0) : d1 === d0 ? 0.5 : (v - d0) / (d1 - d0));
  const fromT = (t: number) => (kind === "depth" ? d0 + depthFromFraction(t, d1 - d0) : d0 + t * (d1 - d0));
  const fn = ((v: number) => {
    const t = toT(v);
    return flip ? hi - t * (hi - lo) : lo + t * (hi - lo);
  }) as Scale;
  fn.invert = (px: number) => {
    const t = flip ? (hi - px) / (hi - lo) : (px - lo) / (hi - lo);
    return fromT(t);
  };
  return fn;
}

/** Month (or day) ticks for a time axis given in UTC milliseconds. */
export function timeTicks(lo: number, hi: number, maxTicks: number): number[] {
  const DAY = 86400000;
  const days = (hi - lo) / DAY;
  const out: number[] = [];
  if (days <= 75) {
    const step = [1, 2, 5, 7, 14, 28].find((s) => days / s <= maxTicks) ?? 28;
    for (let t = Math.ceil(lo / DAY) * DAY; t <= hi; t += step * DAY) out.push(t);
    return out;
  }
  const months = days / 30.44;
  const stepM = [1, 2, 3, 6, 12, 24].find((s) => months / s <= maxTicks) ?? 24;
  const d0 = new Date(lo);
  let y = d0.getUTCFullYear();
  let m = Math.ceil(d0.getUTCMonth() / stepM) * stepM;
  for (;;) {
    const t = Date.UTC(y, m, 1);
    if (t > hi) break;
    if (t >= lo) out.push(t);
    m += stepM;
    y += Math.floor(m / 12);
    m %= 12;
  }
  return out;
}

export function formatTimeTick(ms: number, span: number): string {
  const iso = fromUtcMs(ms);
  const day = Number(iso.slice(8, 10));
  if (span / 86400000 <= 75) return `${day} ${monthShort(iso)}`;
  return iso.slice(5, 7) === "01" ? `${monthShort(iso)} ${iso.slice(0, 4)}` : monthShort(iso);
}

function linePath(points: readonly ChartPoint[], X: Scale, Y: Scale): string {
  let d = "";
  let pen = false;
  for (const p of points) {
    if (p.x == null || p.y == null || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function XYChart(props: XYChartProps) {
  const { series, x, y, height = 320, hover = "x", zero = null, band, reference, onPick, ariaLabel, compact } = props;
  const [wrapRef, size] = useSize<HTMLDivElement>();
  const [hoverValue, setHoverValue] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const width = size.width;
  const m = { top: 26, right: 14, bottom: 40, left: compact ? 40 : 46 };
  const pw = Math.max(10, width - m.left - m.right);
  const ph = Math.max(10, height - m.top - m.bottom);
  const yDown = (y.scale ?? "linear") === "depth";
  const X = useMemo(() => makeScale(x, m.left, m.left + pw, false), [x, m.left, pw]);
  const Y = useMemo(() => makeScale(y, m.top, m.top + ph, !yDown), [y, m.top, ph, yDown]);

  const xTicks = useMemo(() => {
    if (x.ticks) return x.ticks;
    if (x.scale === "time") return timeTicks(x.domain[0], x.domain[1], Math.max(2, Math.floor(pw / 86)));
    return niceTicks(x.domain[0], x.domain[1], Math.max(2, Math.floor(pw / 62)));
  }, [x, pw]);
  const yTicks = useMemo(() => {
    if (y.ticks) return y.scale === "depth" ? depthTicks(y.ticks, ph, 17) : y.ticks;
    return niceTicks(y.domain[0], y.domain[1], Math.max(2, Math.floor(ph / 44)));
  }, [y, ph]);
  const xDigits = tickDigits(xTicks);
  const yDigits = tickDigits(yTicks);
  const xSpan = x.domain[1] - x.domain[0];
  const fx = x.format ?? ((v: number) => (x.scale === "time" ? formatTimeTick(v, xSpan) : fmt(v, xDigits)));
  const fy = y.format ?? ((v: number) => fmt(v, yDigits));

  const ordered = useMemo(() => [...series].sort((a, b) => a.style.z - b.style.z), [series]);

  /** distinct values along the hover axis */
  const hoverValues = useMemo(() => {
    const set = new Set<number>();
    for (const s of series) {
      for (const p of s.points) {
        const v = hover === "x" ? p.x : p.y;
        if (v != null && Number.isFinite(v)) set.add(v);
      }
    }
    return [...set].sort((a, b) => a - b);
  }, [series, hover]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || hoverValues.length === 0) return;
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px < m.left - 6 || px > m.left + pw + 6 || py < m.top - 6 || py > m.top + ph + 6) {
      setHoverValue(null);
      return;
    }
    const scale = hover === "x" ? X : Y;
    const pos = hover === "x" ? px : py;
    let best = hoverValues[0];
    let bestD = Infinity;
    for (const v of hoverValues) {
      const d = Math.abs(scale(v) - pos);
      if (d < bestD) {
        bestD = d;
        best = v;
      }
    }
    setHoverValue(best);
    setPointer({ x: px, y: py });
  };

  const hovered = useMemo(() => {
    if (hoverValue == null) return null;
    return series
      .map((s) => {
        const p = s.points.find((q) => (hover === "x" ? q.x : q.y) === hoverValue);
        const v = p ? (hover === "x" ? p.y : p.x) : null;
        return { s, p, v };
      })
      .filter((r) => r.p && r.v != null && Number.isFinite(r.v));
  }, [hoverValue, series, hover]);

  const inX = (v: number) => v >= Math.min(...x.domain) - 1e-9 && v <= Math.max(...x.domain) + 1e-9;
  const inY = (v: number) => v >= Math.min(...y.domain) - 1e-9 && v <= Math.max(...y.domain) + 1e-9;

  const tooltipLeft = pointer && pointer.x > width * 0.58;

  return (
    <div ref={wrapRef} className="chart" style={{ height }}>
      {width > 0 && (
        <svg
          ref={svgRef}
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          onPointerMove={onMove}
          onPointerLeave={() => setHoverValue(null)}
          onClick={() => hoverValue != null && onPick?.(hoverValue)}
          style={{ cursor: onPick ? "pointer" : "default" }}
        >
          {band && (
            <g>
              {band.axis === "y" ? (
                <rect x={m.left} width={pw} y={Math.min(Y(band.from), Y(band.to))} height={Math.abs(Y(band.to) - Y(band.from))} className="chart__band" />
              ) : (
                <rect y={m.top} height={ph} x={Math.min(X(band.from), X(band.to))} width={Math.abs(X(band.to) - X(band.from))} className="chart__band" />
              )}
            </g>
          )}

          {/* grid */}
          <g className="chart__grid">
            {xTicks.filter(inX).map((t) => (
              <line key={`x${t}`} x1={X(t)} x2={X(t)} y1={m.top} y2={m.top + ph} />
            ))}
            {yTicks.filter(inY).map((t) => (
              <line key={`y${t}`} x1={m.left} x2={m.left + pw} y1={Y(t)} y2={Y(t)} />
            ))}
          </g>

          {zero === "x" && inX(0) && <line className="chart__zero" x1={X(0)} x2={X(0)} y1={m.top} y2={m.top + ph} />}
          {zero === "y" && inY(0) && <line className="chart__zero" x1={m.left} x2={m.left + pw} y1={Y(0)} y2={Y(0)} />}

          {/* axes */}
          <g className="chart__axis">
            <line x1={m.left} x2={m.left + pw} y1={m.top + ph} y2={m.top + ph} />
            {yDown && <line className="chart__surface" x1={m.left} x2={m.left + pw} y1={m.top} y2={m.top} />}
            <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + ph} />
            {xTicks.filter(inX).map((t) => (
              <text key={`xt${t}`} x={X(t)} y={m.top + ph + 15} textAnchor="middle" className="chart__tick">
                {fx(t)}
              </text>
            ))}
            {yTicks.filter(inY).map((t) => (
              <text key={`yt${t}`} x={m.left - 7} y={Y(t)} dy="0.34em" textAnchor="end" className="chart__tick">
                {fy(t)}
              </text>
            ))}
            <text x={m.left + pw / 2} y={height - 6} textAnchor="middle" className="chart__label">
              {x.label}
            </text>
            <text x={2} y={12} textAnchor="start" className="chart__label">
              {y.label}
            </text>
          </g>

          {reference && (
            <g className="chart__ref">
              {reference.axis === "y" && inY(reference.value) && (
                <>
                  <line x1={m.left} x2={m.left + pw} y1={Y(reference.value)} y2={Y(reference.value)} />
                  {reference.label && (
                    <text x={m.left + pw - 3} y={Y(reference.value) - 4} textAnchor="end">
                      {reference.label}
                    </text>
                  )}
                </>
              )}
              {reference.axis === "x" && inX(reference.value) && (
                <>
                  <line y1={m.top} y2={m.top + ph} x1={X(reference.value)} x2={X(reference.value)} />
                  {reference.label && (
                    <text x={X(reference.value) + 4} y={m.top + 10} textAnchor="start">
                      {reference.label}
                    </text>
                  )}
                </>
              )}
            </g>
          )}

          {/* series, back to front */}
          {ordered.map((s) => (
            <g key={s.key}>
              <path
                d={linePath(s.points, X, Y)}
                fill="none"
                stroke={s.style.color}
                strokeWidth={s.style.width}
                strokeDasharray={s.style.dash || undefined}
                strokeLinejoin="round"
                strokeLinecap={s.style.dash ? "butt" : "round"}
              />
              {s.markers !== false &&
                s.points.length <= 40 &&
                s.points.map((p, i) =>
                  p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y) ? (
                    <Marker key={i} shape={s.style.marker} x={X(p.x)} y={Y(p.y)} color={s.style.color} r={compact ? 2.6 : 3.1} />
                  ) : null,
                )}
            </g>
          ))}

          {/* hover */}
          {hoverValue != null && (
            <g className="chart__hover" pointerEvents="none">
              {hover === "x" ? (
                <line x1={X(hoverValue)} x2={X(hoverValue)} y1={m.top} y2={m.top + ph} />
              ) : (
                <line x1={m.left} x2={m.left + pw} y1={Y(hoverValue)} y2={Y(hoverValue)} />
              )}
              {hovered?.map(({ s, p }) => (
                <Marker key={s.key} shape={s.style.marker} x={X(p!.x!)} y={Y(p!.y!)} color={s.style.color} r={4.4} />
              ))}
            </g>
          )}
        </svg>
      )}

      {hoverValue != null && hovered && hovered.length > 0 && pointer && (
        <div
          className="tooltip"
          style={{
            top: Math.min(Math.max(pointer.y - 12, 4), Math.max(4, height - 40 - hovered.length * 20)),
            ...(tooltipLeft ? { right: width - pointer.x + 14 } : { left: pointer.x + 14 }),
          }}
          role="presentation"
        >
          <p className="tooltip__title">
            {props.hoverTitle ? props.hoverTitle(hoverValue) : hover === "x" ? fx(hoverValue) : fy(hoverValue)}
          </p>
          <table className="tooltip__rows">
            <tbody>
              {[...hovered]
                .sort((a, b) => b.s.style.z - a.s.style.z)
                .map(({ s, v }) => (
                  <tr key={s.key}>
                    <td>
                      <Swatch style={s.style} width={22} />
                    </td>
                    <td>{s.label}</td>
                    <td className="num right">{hover === "x" ? fy(v!) : fx(v!)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          {props.tooltipExtra?.(hoverValue)}
        </div>
      )}
    </div>
  );
}
