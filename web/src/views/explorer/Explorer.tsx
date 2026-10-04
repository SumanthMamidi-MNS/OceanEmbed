/**
 * Ocean explorer: one day, one depth, the selected estimate against GLORYS on linked maps (or every
 * estimate side by side), the surface inputs of the same day, and the water column under a chosen
 * point (profile, vertical section, time-depth). Volumes are cached client-side, so scrubbing
 * depth and stepping days only re-colours arrays that are already in memory.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRunRanges } from "@/api/queries";
import { peekDay, prefetchDay, useDayVolumes } from "@/api/volumes";
import { DepthRail } from "@/components/controls/DepthRail";
import { SelectionBar } from "@/components/controls/SelectionBar";
import { Timeline } from "@/components/controls/Timeline";
import { ZoomControls } from "@/components/map/ZoomControls";
import { runLabel } from "@/components/shell/Shell";
import { Empty, ErrorState, Loading, Segmented } from "@/components/ui/primitives";
import { fmtDate } from "@/lib/dates";
import { fmtDepth } from "@/lib/format";
import { prefersReducedMotion } from "@/lib/useSize";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";
import { levelMeans, type Quantity } from "./fields";
import { PointPanels } from "./PointPanels";
import { ProfilePanel } from "./ProfilePanel";
import { CompareMaps, SubsurfaceMaps } from "./SubsurfaceMaps";
import { SurfaceMaps } from "./SurfaceMaps";
import { useDailyStrip } from "./useDailyStrip";

type Mode = "sub" | "surf";

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target.isContentEditable;
}

export default function Explorer() {
  const ctx = useRunContext();
  const { run, dates, dateIndex, date, depths, depthIndex, targetDates, geom, mask, estimate, fieldMethods, labelOf, styleOf } = ctx;
  const [url, setUrl] = useUrlState();
  const link = useLinkedView(geom);

  const mode: Mode = url.opts.mode === "surf" ? "surf" : "sub";
  const quantity: Quantity = url.opts.q === "anom" ? "anom" : "temp";
  const showBasins = url.opts.basins === "1";
  const canCompare = fieldMethods.length > 1;
  const sideBySide = canCompare && url.opts.sbs === "1" && mode === "sub";
  const profileAll = url.opts.pall !== "0";
  const [playing, setPlaying] = useState(false);
  const [fixedRange, setFixedRange] = useState(false);
  const holdRange = fixedRange || playing;

  // the selected estimate first (it is the day's primary); the others only where they are drawn
  const methods = useMemo(() => {
    const others = sideBySide || profileAll ? fieldMethods.map((m) => m.key).filter((k) => k !== estimate) : [];
    return [estimate, ...others];
  }, [estimate, fieldMethods, sideBySide, profileAll]);
  const compareMethods = useMemo(() => fieldMethods.map((m) => m.key), [fieldMethods]);

  // neighbours to prefetch: a few days ahead (animation direction), one behind
  const neighbours = useMemo(() => {
    const out: { date: string; wantTarget: boolean }[] = [];
    for (const off of [1, 2, 3, -1]) {
      const d = dates[dateIndex + off];
      if (d) out.push({ date: d, wantTarget: targetDates.has(d) });
    }
    return out;
  }, [dates, dateIndex, targetDates]);

  const wantTarget = date ? targetDates.has(date) : false;
  const { day, loading, error } = useDayVolumes(run.name, date, wantTarget, neighbours, methods);
  // the period ranges take seconds to compute on a long run: the maps keep their limits meanwhile
  const rangesQ = useRunRanges(run.name, holdRange);
  const ranges = rangesQ.data;

  // ---- stepping, animation, keyboard ----------------------------------------------------
  const stateRef = useRef({ dateIndex, depthIndex, n: dates.length, nz: depths.length });
  stateRef.current = { dateIndex, depthIndex, n: dates.length, nz: depths.length };
  const { setDateIndex, setDepthIndex } = ctx;

  const stepDay = useCallback(
    (delta: number) => {
      const s = stateRef.current;
      setDateIndex(Math.min(s.n - 1, Math.max(0, s.dateIndex + delta)));
    },
    [setDateIndex],
  );
  const stepDepth = useCallback(
    (delta: number) => {
      const s = stateRef.current;
      setDepthIndex(Math.min(s.nz - 1, Math.max(0, s.depthIndex + delta)));
    },
    [setDepthIndex],
  );

  const methodKey = methods.join(",");
  useEffect(() => {
    if (!playing) return;
    const interval = prefersReducedMotion() ? 600 : 180;
    const wanted = methodKey.split(",");
    const timer = setInterval(() => {
      const s = stateRef.current;
      if (s.n <= 1) return;
      const next = (s.dateIndex + 1) % s.n;
      const d = dates[next];
      // advance only when the next frame is already in memory: no blank or stale frames
      if (peekDay(run.name, d, targetDates.has(d), wanted)) setDateIndex(next);
      else prefetchDay(run.name, d, targetDates.has(d), wanted);
    }, interval);
    return () => clearInterval(timer);
  }, [playing, dates, run.name, targetDates, setDateIndex, methodKey]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || isTyping(e.target)) return;
      if (e.key === "ArrowLeft") stepDay(-1);
      else if (e.key === "ArrowRight") stepDay(1);
      else if (e.key === "ArrowUp") stepDepth(-1);
      else if (e.key === "ArrowDown") stepDepth(1);
      else if (e.key === " " && !(e.target instanceof HTMLButtonElement) && !(e.target instanceof HTMLAnchorElement)) {
        setPlaying((p) => !p);
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepDay, stepDepth]);

  const strip = useDailyStrip(estimate);

  const meanProfile = useMemo(() => (day ? levelMeans(day.prediction) : null), [day]);

  if (dates.length === 0 || !date) {
    return (
      <Empty title="This run has no reconstruction yet" height={280}>
        Its prediction product has not been written (“oceanembed predict”, or the run is still being produced). The views fill in as the
        pipeline writes its outputs; choose another run in the selector meanwhile.
      </Empty>
    );
  }

  const depth = depths[depthIndex];
  const mapProps = day ? { day, link, quantity, holdRange, ranges, showBasins, pending: loading } : null;

  return (
    <div className="explorer">
      <SelectionBar play={{ playing, onToggle: () => setPlaying((p) => !p) }} arrowKeys>
        {canCompare && (
          <button
            type="button"
            className="toggle"
            aria-pressed={sideBySide}
            disabled={mode !== "sub"}
            title="Every estimate of this run beside GLORYS, on shared colour scales"
            onClick={() => setUrl({ opts: { sbs: sideBySide ? null : "1" } })}
          >
            Compare
          </button>
        )}
      </SelectionBar>

      <header className="viewhead viewhead--tight">
        <div className="viewhead__row">
          <div>
            <p className="overline">Explorer · {runLabel(run)}</p>
            <h1 className="h1">
              {fmtDate(date)} <span className="explorer__at">at</span> {fmtDepth(depth)}
            </h1>
          </div>
          <p className="caption explorer__hint">
            <kbd>←</kbd> <kbd>→</kbd> day · <kbd>↑</kbd> <kbd>↓</kbd> depth · <kbd>Space</kbd> play · click a map: water column · scroll: zoom ·
            drag: pan
          </p>
        </div>
      </header>

      <Timeline
        dates={dates}
        index={dateIndex}
        onChange={setDateIndex}
        model={strip?.estimate}
        modelLabel={labelOf(estimate)}
        modelStyle={styleOf(estimate)}
        climatology={strip?.climatology}
        seriesLabel={`Daily RMSE vs GLORYS at ${fmtDepth(depth)} (°C)`}
      />

      <div className="toolbar explorer__options">
        <Segmented
          label="Fields"
          value={mode}
          onChange={(v) => setUrl({ opts: { mode: v === "sub" ? null : v } })}
          options={[
            { value: "sub", label: "Subsurface" },
            { value: "surf", label: "Surface inputs" },
          ]}
        />
        {mode === "sub" && (
          <Segmented
            label="Quantity"
            value={quantity}
            onChange={(v) => setUrl({ opts: { q: v === "temp" ? null : v, td: null } })}
            options={[
              { value: "temp", label: "Temperature" },
              { value: "anom", label: "Anomaly", title: "Departure from the harmonic climatology of the training period" },
            ]}
          />
        )}
        {mode === "surf" && (
          <Segmented
            label="Currents and winds"
            value={url.opts.vec === "uv" ? "uv" : "dir"}
            onChange={(v) => setUrl({ opts: { vec: v === "dir" ? null : v } })}
            options={[
              { value: "dir", label: "Speed + direction" },
              { value: "uv", label: "U, V components" },
            ]}
          />
        )}
        <Segmented
          label="Colour range"
          value={holdRange ? "fixed" : "day"}
          onChange={(v) => setFixedRange(v === "fixed")}
          options={[
            { value: "day", label: "Range per day", title: "Colour limits follow each day's 1st–99th percentile" },
            {
              value: "fixed",
              label: "Hold range",
              title:
                "One colour scale for every day at this depth: the limits cover the whole period, so a seasonal cycle moves through fixed colours. Always on during playback.",
            },
          ]}
        />
        {holdRange && rangesQ.isPending && (
          <span className="caption toolbar__status" role="status">
            period limits loading…
          </span>
        )}
        <button type="button" className="toggle" aria-pressed={showBasins} onClick={() => setUrl({ opts: { basins: showBasins ? null : "1" } })}>
          Basin outlines
        </button>
        <div className="toolbar__spacer" />
        <ZoomControls link={link} />
      </div>

      {error && !day ? (
        <ErrorState error={new Error(error)} what="The reconstruction for this day" height={320} />
      ) : !day || !mask || !mapProps ? (
        <Loading height={420} label="Loading the reconstruction" />
      ) : sideBySide ? (
        <>
          <div className="explorer__stage explorer__stage--wide">
            <CompareMaps {...mapProps} methods={compareMethods} />
          </div>
          <PointPanels day={day} quantity={quantity} lead={<ProfilePanel day={day} compact />} />
        </>
      ) : (
        <>
          <div className="explorer__main">
            <div className="explorer__stage">
              <div className="explorer__maps">
                {mode === "sub" ? (
                  <SubsurfaceMaps
                    {...mapProps}
                    rail={<DepthRail depths={depths} index={depthIndex} onChange={setDepthIndex} profile={meanProfile} band={ctx.pooledRange} />}
                  />
                ) : (
                  <SurfaceMaps day={day} link={link} components={url.opts.vec === "uv"} showBasins={showBasins} />
                )}
              </div>
            </div>
            <ProfilePanel day={day} />
          </div>
          <PointPanels day={day} quantity={quantity} />
        </>
      )}
    </div>
  );
}
