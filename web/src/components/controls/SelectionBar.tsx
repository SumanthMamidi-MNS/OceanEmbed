/**
 * The selection bar: one compact control surface for what is being looked at (day, depth, water
 * column, estimate). It is docked under the top bar on every view where the shared selection
 * matters, always in the same place and with the same behaviour; a view only says which groups
 * apply to it. The values live in the URL (state/url), so the bar and the keyboard shortcuts are
 * two handles on the same state.
 */
import { createContext, useContext, useId, useMemo, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Icon, IconButton, Segmented } from "@/components/ui/primitives";
import { fmtDate, fmtMonth } from "@/lib/dates";
import { fmtDepth, fmtInt } from "@/lib/format";
import { useRunContext } from "@/state/runContext";

/** The element under the top bar that the bar is rendered into (provided by the shell). */
export const SelectionBarSlot = createContext<HTMLElement | null>(null);

export interface SelectionBarProps {
  /** which groups this view uses (default: all that the run supports) */
  day?: boolean;
  depth?: boolean;
  point?: boolean;
  estimate?: boolean;
  /** restrict the day control to these days (e.g. the days that have an embedding) */
  days?: readonly string[];
  /** shown on the explorer: playback of the days */
  play?: { playing: boolean; onToggle: () => void };
  /** arrow-key shortcuts are active on this view (the explorer), so the tooltips name them */
  arrowKeys?: boolean;
  /** view-specific switches, right-aligned */
  children?: ReactNode;
}

function groupByMonth(dates: readonly string[]): { month: string; items: { date: string; index: number }[] }[] {
  const out: { month: string; items: { date: string; index: number }[] }[] = [];
  dates.forEach((date, index) => {
    const month = date.slice(0, 7);
    const last = out[out.length - 1];
    if (last && last.month === month) last.items.push({ date, index });
    else out.push({ month, items: [{ date, index }] });
  });
  return out;
}

export function SelectionBar(props: SelectionBarProps) {
  const slot = useContext(SelectionBarSlot);
  const ctx = useRunContext();
  const { day = true, depth = true, point = true, estimate = true, days, play, arrowKeys, children } = props;
  const { dates, date, setDateIndex, depths, depthIndex, setDepthIndex, geom, setPoint, fieldMethods, labelOf, setEstimate, detail } = ctx;
  const dayId = useId();
  const depthId = useId();
  const latId = useId();
  const lonId = useId();

  // the day control may be limited to a subset of the days (still writing the shared date)
  const list = days ?? dates;
  const at = date ? list.indexOf(date) : -1;
  const months = useMemo(() => groupByMonth(list), [list]);
  const goto = (i: number) => {
    const d = list[Math.min(list.length - 1, Math.max(0, i))];
    const target = d ? dates.indexOf(d) : -1;
    if (target >= 0) setDateIndex(target);
  };
  const step = (delta: number) => {
    if (at >= 0) return goto(at + delta);
    // the selected day is not in the subset: move to the nearest day on that side
    if (date == null) return;
    let next = -1;
    if (delta > 0) next = list.findIndex((d) => d > date);
    else for (let i = list.length - 1; i >= 0 && next < 0; i--) if (list[i] < date) next = i;
    if (next >= 0) goto(next);
  };
  const dayOption = (d: string, index: number) => (
    <option key={d} value={index}>
      {fmtDate(d)}
    </option>
  );

  const pt = ctx.point;
  const res = detail.grid.resolution;

  const bar = (
    <div className="selbar" role="region" aria-label="Selection: day, depth, water column and estimate">
      <div className="selbar__inner">
        {day && list.length > 0 && (
          <div className="selbar__group">
            <label htmlFor={dayId} className="selbar__label">
              Day
            </label>
            <IconButton label="Previous day" shortcut={arrowKeys ? "ArrowLeft" : ","} onClick={() => step(-1)} disabled={at === 0 || list.length < 2}>
              <Icon name="prev" size={14} />
            </IconButton>
            <div className="select">
              <select id={dayId} className="num" value={at >= 0 ? at : ""} onChange={(e) => goto(Number(e.target.value))}>
                {at < 0 && <option value="">{date ? fmtDate(date) : "–"}</option>}
                {months.length > 2
                  ? months.map((m) => (
                      <optgroup key={m.month} label={fmtMonth(m.month)}>
                        {m.items.map((it) => dayOption(it.date, it.index))}
                      </optgroup>
                    ))
                  : list.map(dayOption)}
              </select>
              <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true">
                <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </div>
            <IconButton label="Next day" shortcut={arrowKeys ? "ArrowRight" : "."} onClick={() => step(1)} disabled={at === list.length - 1 || list.length < 2}>
              <Icon name="next" size={14} />
            </IconButton>
            {play && (
              <IconButton label={play.playing ? "Pause the animation" : "Play through the days"} shortcut="Space" pressed={play.playing} onClick={play.onToggle}>
                <Icon name={play.playing ? "pause" : "play"} size={14} />
              </IconButton>
            )}
            <span className="selbar__note num" aria-hidden="true">
              {at >= 0 ? `${fmtInt(at + 1)} / ${fmtInt(list.length)}` : `of ${fmtInt(list.length)}`}
            </span>
          </div>
        )}

        {depth && depths.length > 0 && (
          <div className="selbar__group">
            <label htmlFor={depthId} className="selbar__label">
              Depth
            </label>
            <IconButton label="Shallower level" shortcut={arrowKeys ? "ArrowUp" : "["} onClick={() => setDepthIndex(depthIndex - 1)} disabled={depthIndex <= 0}>
              <Icon name="up" size={14} />
            </IconButton>
            <div className="select">
              <select id={depthId} className="num" value={depthIndex} onChange={(e) => setDepthIndex(Number(e.target.value))}>
                {depths.map((d, k) => (
                  <option key={d} value={k}>
                    {fmtDepth(d)}
                  </option>
                ))}
              </select>
              <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true">
                <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </div>
            <IconButton label="Deeper level" shortcut={arrowKeys ? "ArrowDown" : "]"} onClick={() => setDepthIndex(depthIndex + 1)} disabled={depthIndex >= depths.length - 1}>
              <Icon name="down" size={14} />
            </IconButton>
          </div>
        )}

        {point && (
          <div className="selbar__group" role="group" aria-label="Water column: the point the profiles are read at" title="The water column read by the profiles, sections and time series. Click any map to move it.">
            <span className="selbar__label">Point</span>
            <label htmlFor={latId} className="selbar__unit">
              Lat
            </label>
            <input
              id={latId}
              type="number"
              className="selbar__num num"
              value={pt ? Number(pt.lat.toFixed(3)) : ""}
              disabled={!pt}
              min={geom.lat0}
              max={geom.lat1}
              step={res}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (pt && e.target.value !== "" && Number.isFinite(v) && v >= geom.lat0 && v <= geom.lat1) setPoint(v, pt.lon);
              }}
            />
            <label htmlFor={lonId} className="selbar__unit">
              Lon
            </label>
            <input
              id={lonId}
              type="number"
              className="selbar__num num"
              value={pt ? Number(pt.lon.toFixed(3)) : ""}
              disabled={!pt}
              min={geom.lon0}
              max={geom.lon1}
              step={res}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (pt && e.target.value !== "" && Number.isFinite(v) && v >= geom.lon0 && v <= geom.lon1) setPoint(pt.lat, v);
              }}
            />
          </div>
        )}

        {estimate && fieldMethods.length > 1 && (
          <div className="selbar__group">
            <span className="selbar__label" aria-hidden="true">
              Estimate
            </span>
            <Segmented
              label="Estimate shown"
              size="sm"
              value={ctx.estimate}
              onChange={setEstimate}
              options={fieldMethods.map((m) => ({ value: m.key, label: labelOf(m.key), title: m.label }))}
            />
          </div>
        )}

        {children && <div className="selbar__extra">{children}</div>}
      </div>
    </div>
  );

  // without a slot (isolated rendering) the bar sits where it is declared
  return slot ? createPortal(bar, slot) : bar;
}
