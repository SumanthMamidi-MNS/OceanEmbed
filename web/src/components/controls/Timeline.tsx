/**
 * Day selector with meaning: the strip shows the daily RMSE of the reconstruction at the selected
 * depth (and of climatology, as the reference), so the hard days are visible before you step to
 * them. Click or drag to scrub; it is also a keyboard slider.
 */
import { useMemo, useRef, useState } from "react";
import { fmtDate, toUtcMs } from "@/lib/dates";
import { fmt } from "@/lib/format";
import { methodStyle, type MethodStyle } from "@/lib/methods";
import { useSize } from "@/lib/useSize";
import { formatTimeTick, timeTicks } from "../charts/XYChart";

export interface TimelineProps {
  dates: readonly string[];
  index: number;
  onChange: (index: number) => void;
  /** per day, same length as dates (null = no target that day): the selected estimate */
  model?: readonly (number | null)[] | null;
  /** name and identity of that estimate (default: the main model) */
  modelLabel?: string;
  modelStyle?: MethodStyle;
  climatology?: readonly (number | null)[] | null;
  /** e.g. "RMSE at 100 m (°C)" */
  seriesLabel?: string;
  height?: number;
}

function path(values: readonly (number | null)[], X: (i: number) => number, Y: (v: number) => number): string {
  let d = "";
  let pen = false;
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

export function Timeline({ dates, index, onChange, model, modelLabel = "OceanEmbed", modelStyle, climatology, seriesLabel, height = 78 }: TimelineProps) {
  const [ref, size] = useSize<HTMLDivElement>();
  const dragging = useRef(false);
  const [hover, setHover] = useState<number | null>(null);
  const n = dates.length;
  const w = size.width;
  const m = { left: 10, right: 10, top: 18, bottom: 20 };
  const pw = Math.max(10, w - m.left - m.right);
  const ph = Math.max(10, height - m.top - m.bottom);
  const X = (i: number) => m.left + (n <= 1 ? 0.5 : i / (n - 1)) * pw;

  const hasSeries = !!model && model.some((v) => v != null);
  const vmax = useMemo(() => {
    let hi = 0;
    for (const arr of [model, climatology]) {
      if (!arr) continue;
      for (const v of arr) if (v != null && Number.isFinite(v) && v > hi) hi = v;
    }
    return hi > 0 ? hi * 1.08 : 1;
  }, [model, climatology]);
  const Y = (v: number) => m.top + ph - (v / vmax) * ph;

  const ms = useMemo(() => dates.map(toUtcMs), [dates]);
  const ticks = useMemo(() => (n > 1 ? timeTicks(ms[0], ms[n - 1], Math.max(2, Math.floor(pw / 92))) : []), [ms, n, pw]);
  const tickX = (t: number) => m.left + ((t - ms[0]) / (ms[n - 1] - ms[0])) * pw;

  const indexAt = (clientX: number): number => {
    const el = ref.current;
    if (!el || n === 0) return index;
    const rect = el.getBoundingClientRect();
    const f = (clientX - rect.left - m.left) / pw;
    return Math.min(n - 1, Math.max(0, Math.round(f * (n - 1))));
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const targets: Record<string, number> = {
      ArrowRight: index + 1,
      ArrowLeft: index - 1,
      PageDown: index + 7,
      PageUp: index - 7,
      Home: 0,
      End: n - 1,
    };
    if (!(e.key in targets)) return;
    e.preventDefault();
    e.stopPropagation();
    const next = Math.min(n - 1, Math.max(0, targets[e.key]));
    if (next !== index) onChange(next);
  };

  const clim = methodStyle("climatology");
  const mod = modelStyle ?? methodStyle("model");
  const shown = hover ?? index;
  const shownModel = model?.[shown];
  const shownClim = climatology?.[shown];

  return (
    <div
      ref={ref}
      className="timeline"
      style={{ height }}
      role="slider"
      tabIndex={0}
      aria-label="Day"
      aria-valuemin={0}
      aria-valuemax={Math.max(0, n - 1)}
      aria-valuenow={index}
      aria-valuetext={n > 0 ? fmtDate(dates[index]) : undefined}
      aria-keyshortcuts="ArrowLeft ArrowRight"
      onKeyDown={onKeyDown}
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        const i = indexAt(e.clientX);
        if (i !== index) onChange(i);
      }}
      onPointerMove={(e) => {
        const i = indexAt(e.clientX);
        setHover(i);
        if (dragging.current && i !== index) onChange(i);
      }}
      onPointerUp={(e) => {
        dragging.current = false;
        if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
      }}
      onPointerLeave={() => setHover(null)}
    >
      {w > 0 && n > 0 && (
        <svg width={w} height={height} aria-hidden="true">
          <line x1={m.left} x2={m.left + pw} y1={m.top + ph + 0.5} y2={m.top + ph + 0.5} className="timeline__base" />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={tickX(t)} x2={tickX(t)} y1={m.top} y2={m.top + ph + 4} className="timeline__tick" />
              {tickX(t) < m.left + pw - 44 && (
                <text x={tickX(t) + 4} y={height - 5} className="timeline__ticklabel">
                  {formatTimeTick(t, ms[n - 1] - ms[0])}
                </text>
              )}
            </g>
          ))}
          {hasSeries && climatology && (
            <path d={path(climatology, X, Y)} fill="none" stroke={clim.color} strokeWidth={1.25} strokeDasharray={clim.dash} />
          )}
          {hasSeries && model && (
            <path d={path(model, X, Y)} fill="none" stroke={mod.color} strokeWidth={1.5} strokeLinejoin="round" strokeDasharray={mod.dash || undefined} />
          )}
          {hover != null && hover !== index && <line x1={X(hover)} x2={X(hover)} y1={m.top} y2={m.top + ph} className="timeline__hover" />}
          <line x1={X(index)} x2={X(index)} y1={m.top - 4} y2={m.top + ph} className="timeline__cursor" />
          <path d={`M${X(index) - 5},${m.top - 9}h10l-5,6z`} className="timeline__thumb" />
          {shownModel != null && hasSeries && <circle cx={X(shown)} cy={Y(shownModel)} r={3.25} fill={mod.color} stroke="var(--surface)" strokeWidth={1} />}
        </svg>
      )}
      <div className="timeline__legend" aria-hidden="true">
        {hasSeries ? (
          <>
            <span className="timeline__series">{seriesLabel}</span>
            <span className="num">
              <i style={{ background: mod.color }} /> {modelLabel} {fmt(shownModel)}
            </span>
            {climatology && (
              <span className="num">
                <i className="is-dotted" style={{ color: clim.color }} /> Climatology {fmt(shownClim)}
              </span>
            )}
            {hover != null && hover !== index && <span className="timeline__hoverdate">{fmtDate(dates[hover])}</span>}
          </>
        ) : (
          <span className="timeline__series">{n} days with a reconstruction</span>
        )}
      </div>
    </div>
  );
}
