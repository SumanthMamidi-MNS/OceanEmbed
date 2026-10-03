/**
 * Observation-vs-estimate scatter for thousands of matchups, drawn as a binned density (so
 * overplotting cannot hide the bulk) or coloured by the mean depth of each bin (so the depth range
 * that carries the error is visible). Square, shared axes, 1:1 line.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { LUT_SIZE, getLut, type ColormapName } from "@/lib/colormaps";
import { fmt, fmtInt } from "@/lib/format";
import { niceTicks, tickDigits } from "@/lib/scales";
import { useSize } from "@/lib/useSize";

export interface ScatterDensityProps {
  x: ArrayLike<number | null>;
  y: ArrayLike<number | null>;
  /** depth of each matchup, for the "depth" colouring */
  depth?: ArrayLike<number>;
  mode: "density" | "depth";
  maxDepth?: number;
  domain: [number, number];
  xLabel: string;
  yLabel: string;
  bins?: number;
  ariaLabel: string;
  maxSize?: number;
}

export const DENSITY_CMAP: ColormapName = "tempo";
export const DEPTH_CMAP: ColormapName = "deep";

export function ScatterDensity(props: ScatterDensityProps) {
  const { x, y, depth, mode, maxDepth = 1000, domain, xLabel, yLabel, bins = 72, ariaLabel, maxSize = 460 } = props;
  const [wrapRef, size] = useSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ bx: number; by: number; px: number; py: number } | null>(null);

  const m = { top: 10, right: 12, bottom: 40, left: 46 };
  const side = Math.max(60, Math.min(maxSize, size.width) - m.left - m.right);
  const width = side + m.left + m.right;
  const height = side + m.top + m.bottom;
  const [lo, hi] = domain;

  const grid = useMemo(() => {
    const count = new Uint32Array(bins * bins);
    const depthSum = new Float64Array(bins * bins);
    let max = 0;
    const n = Math.min(x.length, y.length);
    for (let k = 0; k < n; k++) {
      const a = x[k];
      const b = y[k];
      if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
      const i = Math.floor(((a - lo) / (hi - lo)) * bins);
      const j = Math.floor(((b - lo) / (hi - lo)) * bins);
      if (i < 0 || j < 0 || i >= bins || j >= bins) continue;
      const c = ++count[j * bins + i];
      if (depth) depthSum[j * bins + i] += depth[k];
      if (c > max) max = c;
    }
    return { count, depthSum, max };
  }, [x, y, depth, lo, hi, bins]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(side * dpr);
    canvas.height = Math.round(side * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const off = document.createElement("canvas");
    off.width = bins;
    off.height = bins;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(bins, bins);
    const lut = getLut(mode === "density" ? DENSITY_CMAP : DEPTH_CMAP);
    const logMax = Math.log(grid.max + 1);
    for (let j = 0; j < bins; j++) {
      for (let i = 0; i < bins; i++) {
        const c = grid.count[j * bins + i];
        const dst = ((bins - 1 - j) * bins + i) * 4;
        if (c === 0) continue;
        let t: number;
        if (mode === "density") t = logMax > 0 ? 0.12 + 0.88 * (Math.log(c + 1) / logMax) : 1;
        else t = 0.08 + 0.92 * Math.sqrt(Math.min(1, grid.depthSum[j * bins + i] / c / maxDepth));
        const idx = Math.round(Math.min(1, Math.max(0, t)) * (LUT_SIZE - 1)) * 4;
        img.data[dst] = lut[idx];
        img.data[dst + 1] = lut[idx + 1];
        img.data[dst + 2] = lut[idx + 2];
        img.data[dst + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, side, side);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, side, side);
  }, [grid, mode, bins, side, size.width, maxDepth]);

  const ticks = niceTicks(lo, hi, Math.max(3, Math.floor(side / 70)));
  const digits = tickDigits(ticks);
  const P = (v: number) => ((v - lo) / (hi - lo)) * side;
  const hb = hover ? hover.by * bins + hover.bx : -1;
  const hc = hb >= 0 ? grid.count[hb] : 0;
  const binW = (hi - lo) / bins;

  return (
    <div ref={wrapRef} className="chart scatter">
      {size.width > 0 && (
        <div style={{ position: "relative", width, height }}>
          <canvas ref={canvasRef} className="heat__canvas" style={{ left: m.left, top: m.top, width: side, height: side }} aria-hidden="true" />
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={ariaLabel}
            className="heat__svg"
            onPointerMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const px = e.clientX - rect.left - m.left;
              const py = e.clientY - rect.top - m.top;
              if (px < 0 || py < 0 || px >= side || py >= side) return setHover(null);
              setHover({ bx: Math.floor((px / side) * bins), by: bins - 1 - Math.floor((py / side) * bins), px: px + m.left, py: py + m.top });
            }}
            onPointerLeave={() => setHover(null)}
          >
            <g className="chart__grid">
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={m.left + P(t)} x2={m.left + P(t)} y1={m.top} y2={m.top + side} />
                  <line x1={m.left} x2={m.left + side} y1={m.top + side - P(t)} y2={m.top + side - P(t)} />
                </g>
              ))}
            </g>
            <line x1={m.left} y1={m.top + side} x2={m.left + side} y2={m.top} className="scatter__identity" />
            <rect x={m.left + 0.5} y={m.top + 0.5} width={side - 1} height={side - 1} fill="none" stroke="var(--rule-strong)" />
            <g className="chart__axis">
              {ticks.map((t) => (
                <g key={t}>
                  <text x={m.left + P(t)} y={m.top + side + 15} textAnchor="middle" className="chart__tick">
                    {fmt(t, digits)}
                  </text>
                  <text x={m.left - 7} y={m.top + side - P(t)} dy="0.34em" textAnchor="end" className="chart__tick">
                    {fmt(t, digits)}
                  </text>
                </g>
              ))}
              <text x={m.left + side / 2} y={height - 6} textAnchor="middle" className="chart__label">
                {xLabel}
              </text>
              <text transform={`translate(11, ${m.top + side / 2}) rotate(-90)`} textAnchor="middle" className="chart__label">
                {yLabel}
              </text>
              <text x={m.left + side - 22} y={m.top + 14} textAnchor="end" className="scatter__idlabel">
                1 : 1 line
              </text>
            </g>
          </svg>
          {hover && hc > 0 && (
            <div
              className="tooltip"
              style={{ top: Math.max(4, hover.py - 44), ...(hover.px > width * 0.6 ? { right: width - hover.px + 12 } : { left: hover.px + 12 }) }}
              role="presentation"
            >
              <p className="tooltip__title">
                {fmtInt(hc)} matchup{hc === 1 ? "" : "s"}
              </p>
              <p className="num">
                observed {fmt(lo + hover.bx * binW, 1)}–{fmt(lo + (hover.bx + 1) * binW, 1)} °C
              </p>
              <p className="num">
                estimated {fmt(lo + hover.by * binW, 1)}–{fmt(lo + (hover.by + 1) * binW, 1)} °C
              </p>
              {depth && <p className="num">mean depth {fmt(grid.depthSum[hb] / hc, 0)} m</p>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
