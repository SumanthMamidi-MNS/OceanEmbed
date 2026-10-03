/** Application frame: sticky top bar (wordmark, views, run), honesty bands, selection bar, colophon. */
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { RunSummary } from "@/api/types";
import { SelectionBarSlot } from "@/components/controls/SelectionBar";
import { fmtSpan } from "@/lib/dates";
import { fmtInt } from "@/lib/format";
import type { RunCaveats } from "@/lib/narrative";
import { useUrlState } from "@/state/router";
import { VIEWS, viewHref, type UrlPatch, type ViewKey } from "@/state/url";

function Wordmark() {
  // three shortening strokes: the surface and the levels beneath it
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <path d="M2 5.5h18" stroke="var(--ink)" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M2 11h13" stroke="var(--accent)" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M2 16.5h8" stroke="var(--accent)" strokeWidth="2.4" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}

function plainLeftClick(e: React.MouseEvent): boolean {
  return e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;
}

/** In-app link to a view that keeps the shared selection (run, day, depth, point). */
export function ViewLink(props: {
  view: ViewKey;
  children: ReactNode;
  className?: string;
  opts?: Record<string, string | null>;
  /** selection the link carries with it (e.g. the day and depth shown where the link is) */
  patch?: Pick<UrlPatch, "date" | "depth" | "lat" | "lon" | "est">;
  current?: boolean;
}) {
  const [url, setUrl] = useUrlState();
  const { view, children, className, opts, patch, current } = props;
  return (
    <a
      className={className}
      href={viewHref(url, view, { ...patch, opts })}
      aria-current={current ? "page" : undefined}
      onClick={(e) => {
        if (!plainLeftClick(e)) return;
        e.preventDefault();
        setUrl({ ...patch, view, opts }, "push");
      }}
    >
      {children}
    </a>
  );
}

/** Display name of a run: the label from its configuration, else its identifier. */
export function runLabel(r: Pick<RunSummary, "name" | "label">): string {
  return r.label?.trim() || r.name;
}

export function Shell(props: {
  runs: RunSummary[];
  run: RunSummary | null;
  caveats: RunCaveats | null;
  children: ReactNode;
}) {
  const { runs, run, caveats, children } = props;
  const [url, setUrl] = useUrlState();
  const sticky = useRef<HTMLDivElement | null>(null);
  // the views render their selection bar into this element, so it is docked in one place
  const [barSlot, setBarSlot] = useState<HTMLDivElement | null>(null);

  // publish the sticky header height so anchors and sticky toolbars clear it
  useLayoutEffect(() => {
    const el = sticky.current;
    if (!el) return;
    const apply = () => document.documentElement.style.setProperty("--sticky-h", `${el.offsetHeight}px`);
    apply();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const test = run?.split?.test;

  return (
    <div className="app" data-source={run?.data_source ?? "unknown"}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <div className="sticky" ref={sticky}>
        <header className="topbar">
          <div className="topbar__inner">
            <ViewLink view="overview" className="wordmark">
              <Wordmark />
              OceanEmbed
            </ViewLink>
            <nav className="nav" aria-label="Views">
              {VIEWS.map((v) => (
                <ViewLink key={v.key} view={v.key} className="nav__link" current={url.view === v.key}>
                  {v.label}
                </ViewLink>
              ))}
            </nav>
            <div className="runpick">
              {run && (
                <span className="runpick__meta num">
                  {test?.start ? `Test ${fmtSpan(test.start, test.end)}` : ""}
                  {run.n_test_days != null ? `${test?.start ? " · " : ""}${fmtInt(run.n_test_days)} days` : ""}
                </span>
              )}
              {runs.length > 0 && (
                <div className="select select--run" title={run?.description || undefined}>
                  <label htmlFor="run-select" className="select__prefix">
                    Run
                  </label>
                  <select id="run-select" value={run?.name ?? ""} onChange={(e) => setUrl({ run: e.target.value }, "push")}>
                    {runs.map((r) => (
                      <option key={r.name} value={r.name}>
                        {runLabel(r)}
                      </option>
                    ))}
                  </select>
                  <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true">
                    <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </div>
              )}
            </div>
          </div>
        </header>
        {caveats?.synthetic && (
          <div className="band band--synthetic" role="note" aria-label="Synthetic data warning">
            <div className="band__inner">
              <span className="band__tag">Synthetic data</span>
              <span className="band__text">
                Pipeline demonstration, not scientific skill. Every map and number in this run comes from an analytic test
                ocean generated by the project itself.
              </span>
            </div>
          </div>
        )}
        {caveats?.shortTraining && (
          <div className="band band--caution" role="note" aria-label="Short training period warning">
            <div className="band__inner">
              <span className="band__tag">Short training period</span>
              <span className="band__text">{caveats.shortTrainingNote}</span>
            </div>
          </div>
        )}
        <div className="selbar-slot" ref={setBarSlot} />
      </div>
      <main id="main" className="page" tabIndex={-1}>
        <SelectionBarSlot.Provider value={barSlot}>{children}</SelectionBarSlot.Provider>
      </main>
      <footer className="colophon">
        <div className="colophon__inner">
          <span>
            OceanEmbed · subsurface ocean temperature reconstructed from surface satellite fields · read-only view of{" "}
            <a href="/api/docs">the data API</a>
          </span>
          <span>
            <kbd>,</kbd> <kbd>.</kbd> previous / next day · <kbd>[</kbd> <kbd>]</kbd> shallower / deeper
          </span>
        </div>
      </footer>
    </div>
  );
}
