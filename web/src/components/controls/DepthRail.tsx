/**
 * Depth selector: the standard levels laid out on the same square-root depth axis as every
 * profile chart, surface at the top. Optionally draws the basin-mean temperature profile of the
 * current day behind the ticks, so the thermocline is visible where you scrub.
 */
import { useRef } from "react";
import { fmtDepth } from "@/lib/format";
import { depthFraction, depthTicks, nearestDepthIndex } from "@/lib/scales";
import { useSize } from "@/lib/useSize";

export interface DepthRailProps {
  depths: readonly number[];
  index: number;
  onChange: (index: number) => void;
  /** mean temperature per level (same length as depths), drawn as context */
  profile?: readonly (number | null)[] | null;
  /** optional shaded depth range (the pooled metrics range) */
  band?: readonly number[] | null;
}

const PAD_TOP = 22;
const PAD_BOTTOM = 10;

export function DepthRail({ depths, index, onChange, profile, band }: DepthRailProps) {
  const [ref, size] = useSize<HTMLDivElement>();
  const dragging = useRef(false);
  const n = depths.length;
  const max = n > 0 ? depths[n - 1] : 1;
  const h = Math.max(40, size.height - PAD_TOP - PAD_BOTTOM);
  const w = Math.max(48, size.width);
  const y = (depth: number) => PAD_TOP + depthFraction(depth, max) * h;
  const railX = 46;
  const labels = new Set(depthTicks(depths, h, 15));
  labels.add(depths[index]);

  const pick = (clientY: number) => {
    const el = ref.current;
    if (!el || n === 0) return;
    const rect = el.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientY - rect.top - PAD_TOP) / h));
    const next = nearestDepthIndex(depths, f * f * max);
    if (next !== index) onChange(next);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const targets: Record<string, number> = {
      ArrowDown: index + 1,
      ArrowUp: index - 1,
      Home: 0,
      End: n - 1,
      PageDown: index + 3,
      PageUp: index - 3,
    };
    if (!(e.key in targets)) return;
    e.preventDefault();
    e.stopPropagation();
    const next = Math.min(n - 1, Math.max(0, targets[e.key]));
    if (next !== index) onChange(next);
  };

  // context profile: mean temperature per level, scaled into the space right of the rail
  let profilePath = "";
  if (profile && profile.length === n) {
    const vals = profile.filter((v): v is number => v != null && Number.isFinite(v));
    if (vals.length > 1) {
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const x0 = railX + 8;
      const x1 = w - 4;
      const px = (v: number) => x0 + (hi === lo ? 0.5 : (v - lo) / (hi - lo)) * (x1 - x0);
      const pts: string[] = [];
      profile.forEach((v, k) => {
        if (v != null && Number.isFinite(v)) pts.push(`${px(v).toFixed(1)},${y(depths[k]).toFixed(1)}`);
      });
      if (pts.length > 1) {
        const firstY = pts[0].split(",")[1];
        const lastY = pts[pts.length - 1].split(",")[1];
        profilePath = `M${x0},${firstY}L${pts.join("L")}L${x0},${lastY}Z`;
      }
    }
  }

  return (
    <div
      ref={ref}
      className="depthrail"
      role="slider"
      tabIndex={0}
      aria-label="Depth level"
      aria-orientation="vertical"
      aria-valuemin={0}
      aria-valuemax={Math.max(0, n - 1)}
      aria-valuenow={index}
      aria-valuetext={n > 0 ? fmtDepth(depths[index]) : undefined}
      aria-keyshortcuts="ArrowUp ArrowDown"
      onKeyDown={onKeyDown}
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        pick(e.clientY);
      }}
      onPointerMove={(e) => dragging.current && pick(e.clientY)}
      onPointerUp={(e) => {
        dragging.current = false;
        if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
      }}
      onPointerCancel={() => {
        dragging.current = false;
      }}
    >
      {size.height > 0 && n > 0 && (
        <svg width={w} height={size.height} aria-hidden="true">
          <text x={railX} y={11} textAnchor="end" className="depthrail__head">
            DEPTH
          </text>
          {band && band.length >= 2 && (
            <rect x={railX - 3} width={w - railX + 3} y={y(band[0])} height={y(band[1]) - y(band[0])} className="depthrail__band" />
          )}
          {profilePath && (
            <>
              <path d={profilePath} className="depthrail__profile">
                <title>Mean reconstructed temperature of each level on this day</title>
              </path>
              <text x={w - 2} y={11} textAnchor="end" className="depthrail__note">
                mean T
              </text>
            </>
          )}
          <line x1={railX} x2={railX} y1={y(0)} y2={y(max)} className="depthrail__track" />
          {depths.map((d, k) => (
            <g key={d} className={k === index ? "depthrail__level is-active" : "depthrail__level"}>
              <line x1={railX - (k === index ? 7 : 4)} x2={railX + (k === index ? 7 : 4)} y1={y(d)} y2={y(d)} />
              {labels.has(d) && (k === index || Math.abs(y(d) - y(depths[index])) >= 13) && (
                <text x={railX - 10} y={y(d)} dy="0.34em" textAnchor="end">
                  {d}
                </text>
              )}
            </g>
          ))}
          <circle cx={railX} cy={y(depths[index])} r={5.5} className="depthrail__thumb" />
          <text x={railX + 12} y={y(depths[index])} dy="0.34em" className="depthrail__unit">
            m
          </text>
        </svg>
      )}
    </div>
  );
}
