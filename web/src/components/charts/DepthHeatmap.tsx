/**
 * Depth-vs-something heat map: vertical sections (x = longitude or latitude), time-depth
 * (Hovmoeller) plots and day-by-depth error. Depth is on the same square-root axis as the
 * profile charts and increases downward; each level is a band (no interpolation between levels).
 * Cells are drawn on a canvas, axes and marks in SVG on top.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { LUT_SIZE, getLut, type ColormapName } from "@/lib/colormaps";
import { fmtDate, fromUtcMs } from "@/lib/dates";
import { fmtDepth, fmtLat, fmtLon } from "@/lib/format";
import { depthBandEdges, depthFraction, depthTicks, niceTicks } from "@/lib/scales";
import { color } from "@/lib/theme";
import { useSize } from "@/lib/useSize";
import { formatTimeTick, timeTicks } from "./XYChart";

export interface DepthHeatmapProps {
  /** values[depthIndex][xIndex]; null = no data (below the sea floor, land, no target) */
  values: ReadonlyArray<ReadonlyArray<number | null>>;
  depths: readonly number[];
  /** degrees for sections, UTC milliseconds for time */
  xValues: readonly number[];
  xKind: "lon" | "lat" | "time" | "index";
  /** axis title under the plot (used with xKind "index", e.g. "Epoch") */
  xLabel?: string;
  cmap: ColormapName;
  vmin: number;
  vmax: number;
  reversed?: boolean;
  height?: number;
  /** x position to mark (the selected point, the selected day) */
  markX?: number | null;
  markDepth?: number | null;
  onPick?: (xIndex: number, depthIndex: number) => void;
  formatValue: (v: number) => string;
  ariaLabel: string;
  /** fill for null cells */
  nullFill?: string;
  showYAxis?: boolean;
  showXAxis?: boolean;
  /** small title at the top right of the plot */
  title?: string;
}

export function DepthHeatmap(props: DepthHeatmapProps) {
  const {
    values,
    depths,
    xValues,
    xKind,
    cmap,
    vmin,
    vmax,
    reversed,
    height = 240,
    markX,
    markDepth,
    onPick,
    formatValue,
    ariaLabel,
    nullFill = color.seafloor,
    showYAxis = true,
    showXAxis = true,
    title,
    xLabel,
  } = props;
  const [wrapRef, size] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ xi: number; di: number; px: number; py: number } | null>(null);

  const width = size.width;
  const m = { top: 22, right: 8, bottom: showXAxis ? (xLabel ? 40 : 26) : 6, left: showYAxis ? 46 : 8 };
  const pw = Math.max(10, width - m.left - m.right);
  const ph = Math.max(10, height - m.top - m.bottom);
  const nx = xValues.length;
  const nz = depths.length;
  const maxDepth = nz > 0 ? depths[nz - 1] : 1;
  const edges = useMemo(() => depthBandEdges(depths), [depths]);

  const xLo = nx > 0 ? xValues[0] : 0;
  const xHi = nx > 0 ? xValues[nx - 1] : 1;
  const xStep = nx > 1 ? (xHi - xLo) / (nx - 1) : 1;
  const d0 = xLo - xStep / 2;
  const d1 = xHi + xStep / 2;
  const X = (v: number) => m.left + ((v - d0) / (d1 - d0)) * pw;
  const Yd = (depth: number) => m.top + depthFraction(depth, maxDepth) * ph;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || nx === 0 || nz === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(pw * dpr);
    canvas.height = Math.round(ph * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const off = document.createElement("canvas");
    off.width = nx;
    off.height = nz;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(nx, nz);
    const lut = getLut(cmap, reversed);
    const span = vmax - vmin;
    for (let k = 0; k < nz; k++) {
      const row = values[k];
      for (let i = 0; i < nx; i++) {
        const v = row?.[i];
        const dst = (k * nx + i) * 4;
        if (v == null || !Number.isFinite(v)) {
          img.data[dst + 3] = 0;
          continue;
        }
        let idx = span > 0 ? Math.round(((v - vmin) / span) * (LUT_SIZE - 1)) : (LUT_SIZE - 1) >> 1;
        idx = (idx < 0 ? 0 : idx > LUT_SIZE - 1 ? LUT_SIZE - 1 : idx) * 4;
        img.data[dst] = lut[idx];
        img.data[dst + 1] = lut[idx + 1];
        img.data[dst + 2] = lut[idx + 2];
        img.data[dst + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = nullFill;
    ctx.fillRect(0, 0, pw, ph);
    ctx.imageSmoothingEnabled = false;
    for (let k = 0; k < nz; k++) {
      const y0 = Math.round(edges[k] * ph);
      const y1 = Math.round(edges[k + 1] * ph);
      ctx.drawImage(off, 0, k, nx, 1, 0, y0, pw, Math.max(1, y1 - y0));
    }
  }, [values, depths, nx, nz, cmap, reversed, vmin, vmax, pw, ph, width, edges, nullFill]);

  const xTicks = useMemo(() => {
    if (xKind === "time") return timeTicks(d0, d1, Math.max(2, Math.floor(pw / 84)));
    const t = niceTicks(d0, d1, Math.max(2, Math.floor(pw / 90)));
    return xKind === "index" ? t.filter((v) => Number.isInteger(v)) : t;
  }, [xKind, d0, d1, pw]);
  const fx = (v: number) =>
    xKind === "index" ? String(v) : xKind === "time" ? formatTimeTick(v, d1 - d0) : xKind === "lon" ? fmtLon(v, Number.isInteger(v) ? 0 : 1) : fmtLat(v, Number.isInteger(v) ? 0 : 1);
  const yTicks = useMemo(() => depthTicks(depths, ph, 17), [depths, ph]);

  const locate = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px < m.left || px > m.left + pw || py < m.top || py > m.top + ph || nx === 0) return null;
    const xi = Math.min(nx - 1, Math.max(0, Math.floor(((px - m.left) / pw) * nx)));
    const f = (py - m.top) / ph;
    let di = nz - 1;
    for (let k = 0; k < nz; k++) {
      if (f < edges[k + 1]) {
        di = k;
        break;
      }
    }
    return { xi, di, px, py };
  };

  const hv = hover ? values[hover.di]?.[hover.xi] : null;
  const hoverX = hover ? xValues[hover.xi] : null;
  const xText = (v: number) => (xKind === "index" ? `${xLabel ?? ""} ${v}`.trim() : xKind === "time" ? fmtDate(fromUtcMs(v)) : xKind === "lon" ? fmtLon(v) : fmtLat(v));

  return (
    <div ref={wrapRef} className="chart heat" style={{ height }}>
      {width > 0 && (
        <>
          <canvas ref={canvasRef} className="heat__canvas" style={{ left: m.left, top: m.top, width: pw, height: ph }} aria-hidden="true" />
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={ariaLabel}
            className="heat__svg"
            onPointerMove={(e) => setHover(locate(e))}
            onPointerLeave={() => setHover(null)}
            onClick={(e) => {
              const h = locate(e as unknown as React.PointerEvent<SVGSVGElement>);
              if (h && onPick) onPick(h.xi, h.di);
            }}
            style={{ cursor: onPick ? "pointer" : "crosshair" }}
          >
            <rect x={m.left + 0.5} y={m.top + 0.5} width={pw - 1} height={ph - 1} fill="none" stroke="var(--rule-strong)" />
            <g className="chart__axis">
              {xTicks.map((t) => (
                <g key={t}>
                  <line x1={X(t)} x2={X(t)} y1={m.top + ph} y2={m.top + ph + 4} />
                  {showXAxis && (
                    <text x={X(t)} y={m.top + ph + 16} textAnchor="middle" className="chart__tick">
                      {fx(t)}
                    </text>
                  )}
                </g>
              ))}
              {showXAxis && xLabel && (
                <text x={m.left + pw / 2} y={height - 6} textAnchor="middle" className="chart__label">
                  {xLabel}
                </text>
              )}
              {showYAxis &&
                yTicks.map((t) => (
                  <g key={t}>
                    <line x1={m.left - 4} x2={m.left} y1={Yd(t)} y2={Yd(t)} />
                    <text x={m.left - 7} y={Yd(t)} dy="0.34em" textAnchor="end" className="chart__tick">
                      {t}
                    </text>
                  </g>
                ))}
              {showYAxis && (
                <text x={2} y={12} className="chart__label">
                  Depth (m)
                </text>
              )}
              {title && (
                <text x={m.left + pw} y={12} textAnchor="end" className="heat__title">
                  {title}
                </text>
              )}
            </g>
            {markDepth != null && (
              <g className="heat__mark">
                <line x1={m.left} x2={m.left + pw} y1={Yd(markDepth)} y2={Yd(markDepth)} className="heat__mark-halo" />
                <line x1={m.left} x2={m.left + pw} y1={Yd(markDepth)} y2={Yd(markDepth)} />
              </g>
            )}
            {markX != null && markX >= d0 && markX <= d1 && (
              <g className="heat__mark">
                <line x1={X(markX)} x2={X(markX)} y1={m.top} y2={m.top + ph} className="heat__mark-halo" />
                <line x1={X(markX)} x2={X(markX)} y1={m.top} y2={m.top + ph} />
                <path d={`M${X(markX) - 4},${m.top - 6}h8l-4,6z`} fill="var(--ink)" />
              </g>
            )}
            {hover && (
              <rect
                x={m.left + (hover.xi / nx) * pw}
                y={m.top + edges[hover.di] * ph}
                width={Math.max(2, pw / nx)}
                height={Math.max(2, (edges[hover.di + 1] - edges[hover.di]) * ph)}
                className="heat__cell"
                pointerEvents="none"
              />
            )}
          </svg>
          {hover && hoverX != null && (
            <div
              className="tooltip"
              style={{
                top: Math.max(4, Math.min(hover.py - 10, height - 64)),
                ...(hover.px > width * 0.6 ? { right: width - hover.px + 14 } : { left: hover.px + 14 }),
              }}
              role="presentation"
            >
              <p className="tooltip__title">
                {xText(hoverX)} · {fmtDepth(depths[hover.di])}
              </p>
              <p className="tooltip__value num">{hv != null && Number.isFinite(hv) ? formatValue(hv) : "no data"}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
