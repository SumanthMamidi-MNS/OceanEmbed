/** Small, quiet building blocks. The data is the hero; these stay out of its way. */
import { useId, type ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { ApiError, errorMessage } from "@/api/client";
import { prettyText } from "@/lib/format";

// ---- controls -------------------------------------------------------------------------------

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

export function Segmented<T extends string>(props: {
  label: string;
  value: T;
  options: readonly SegmentedOption<T>[];
  onChange: (value: T) => void;
  showLabel?: boolean;
  size?: "sm" | "md";
}) {
  const { label, value, options, onChange, showLabel = false, size = "md" } = props;
  return (
    <div className="field">
      {showLabel && <span className="field__label">{label}</span>}
      <div className={`segmented segmented--${size}`} role="group" aria-label={label}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className="segmented__btn"
            aria-pressed={o.value === value}
            disabled={o.disabled}
            title={o.title}
            onClick={() => o.value !== value && onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Select<T extends string>(props: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
  showLabel?: boolean;
  className?: string;
}) {
  const { label, value, options, onChange, showLabel = true, className } = props;
  const id = useId();
  return (
    <div className={`field ${className ?? ""}`}>
      <label htmlFor={id} className={showLabel ? "field__label" : "visually-hidden"}>
        {label}
      </label>
      <div className="select">
        <select id={id} value={value} onChange={(e) => onChange(e.target.value as T)}>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true">
          <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </div>
    </div>
  );
}

export function IconButton(props: {
  label: string;
  onClick: () => void;
  children: ReactNode;
  disabled?: boolean;
  pressed?: boolean;
  shortcut?: string;
}) {
  const { label, onClick, children, disabled, pressed, shortcut } = props;
  return (
    <button
      type="button"
      className="iconbtn"
      aria-label={label}
      title={shortcut ? `${label} (${shortcut})` : label}
      aria-pressed={pressed}
      aria-keyshortcuts={shortcut}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

const ICONS = {
  prev: "M10 3L5 8l5 5",
  next: "M6 3l5 5-5 5",
  first: "M11 3L6 8l5 5M4.5 3v10",
  last: "M5 3l5 5-5 5M11.5 3v10",
  up: "M3 10l5-5 5 5",
  down: "M3 6l5 5 5-5",
  plus: "M8 3v10M3 8h10",
  minus: "M3 8h10",
  reset: "M3 8a5 5 0 1 0 1.6-3.7M3 2.5v2.8h2.8",
  download: "M8 2v8m0 0L5 7m3 3l3-3M3 13h10",
  arrow: "M3 8h9m0 0L8.5 4.5M12 8l-3.5 3.5",
} as const;

export function Icon({ name, size = 16 }: { name: keyof typeof ICONS | "play" | "pause"; size?: number }) {
  if (name === "play") {
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
        <path d="M5 3.2v9.6L13 8z" fill="currentColor" />
      </svg>
    );
  }
  if (name === "pause") {
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
        <path d="M4.5 3h2.5v10H4.5zM9 3h2.5v10H9z" fill="currentColor" />
      </svg>
    );
  }
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <path d={ICONS[name]} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ---- containers -----------------------------------------------------------------------------

/** A figure: title + optional subtitle, optional actions, body, optional caption. */
export function Panel(props: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  caption?: ReactNode;
  children: ReactNode;
  className?: string;
  flush?: boolean;
  id?: string;
  /** the figure is not about the selected run (cross-run comparison): no automatic data-source tag */
  untagged?: boolean;
}) {
  const { title, subtitle, actions, caption, children, className, flush, id, untagged } = props;
  return (
    <figure id={id} className={`panel ${flush ? "panel--flush" : ""} ${untagged ? "panel--untagged" : ""} ${className ?? ""}`}>
      {(title || actions) && (
        <header className="panel__head">
          <div className="panel__titles">
            {title && <h3 className="panel__title">{title}</h3>}
            {subtitle && <p className="panel__sub">{subtitle}</p>}
          </div>
          {actions && <div className="panel__actions">{actions}</div>}
        </header>
      )}
      <div className="panel__body">{children}</div>
      {caption && <figcaption className="panel__caption caption">{caption}</figcaption>}
    </figure>
  );
}

export type NoteKind = "plain" | "honesty" | "caveat";

/** Explanations and honesty statements. `honesty` is used for the binding disclosure rules. */
export function Note(props: { kind?: NoteKind; title?: string; children: ReactNode; className?: string }) {
  const { kind = "plain", title, children, className } = props;
  return (
    <aside className={`note note--${kind} ${className ?? ""}`}>
      {title && <p className="note__title">{title}</p>}
      <div className="note__body">{children}</div>
    </aside>
  );
}

export function Stat(props: { label: string; value: ReactNode; unit?: string; sub?: ReactNode; emphasis?: boolean }) {
  const { label, value, unit, sub, emphasis } = props;
  return (
    <div className={`stat ${emphasis ? "stat--emphasis" : ""}`}>
      <dt className="stat__label">{label}</dt>
      <dd className="stat__value num">
        {value}
        {unit && <span className="stat__unit">{unit}</span>}
      </dd>
      {sub && <dd className="stat__sub">{sub}</dd>}
    </div>
  );
}

// ---- states ---------------------------------------------------------------------------------

export function Loading({ height = 160, label = "Loading" }: { height?: number | string; label?: string }) {
  return (
    <div className="state state--loading" style={{ minHeight: height }} role="status" aria-live="polite">
      <span className="state__pulse" aria-hidden="true" />
      <span className="visually-hidden">{label}</span>
    </div>
  );
}

/**
 * An API error. A 404 is a missing artefact, not a crash: the server's hint is shown as it is.
 */
export function ErrorState(props: { error: unknown; what?: string; onRetry?: () => void; height?: number | string }) {
  const { error, what, onRetry, height } = props;
  const missing = error instanceof ApiError && error.isMissing;
  const offline = error instanceof ApiError && error.status === 0;
  return (
    <div className={`state ${missing ? "state--missing" : "state--error"}`} style={{ minHeight: height }} role="alert">
      <p className="state__title">
        {missing ? `${what ?? "This"} is not available for this run` : offline ? "The data API is not reachable" : `Could not load ${what ?? "this"}`}
      </p>
      <p className="state__detail">{prettyText(errorMessage(error))}</p>
      {onRetry && !missing && (
        <button type="button" className="btn btn--quiet" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty(props: { title: string; children?: ReactNode; height?: number | string }) {
  return (
    <div className="state state--missing" style={{ minHeight: props.height }}>
      <p className="state__title">{props.title}</p>
      {props.children && <p className="state__detail">{props.children}</p>}
    </div>
  );
}

/** Loading / error / data switch for one query. */
export function QueryState<T>(props: {
  query: UseQueryResult<T, Error>;
  what: string;
  height?: number | string;
  children: (data: T) => ReactNode;
}) {
  const { query, what, height, children } = props;
  if (query.data !== undefined) return <>{children(query.data)}</>;
  if (query.isError) return <ErrorState error={query.error} what={what} height={height} onRetry={() => void query.refetch()} />;
  // a query that is switched off (the run lacks the artefact) is an empty state, never an endless spinner
  if (query.fetchStatus === "idle") {
    return (
      <Empty title={`${what} is not available for this run`} height={typeof height === "number" ? Math.min(height, 160) : height}>
        The run has not produced this artefact yet.
      </Empty>
    );
  }
  return <Loading height={height} label={`Loading ${what}`} />;
}

// ---- tables ---------------------------------------------------------------------------------

export interface Column<R> {
  key: string;
  label: ReactNode;
  align?: "left" | "right";
  render: (row: R) => ReactNode;
  title?: string;
}

export function DataTable<R>(props: {
  columns: readonly Column<R>[];
  rows: readonly R[];
  rowKey: (row: R, index: number) => string;
  caption?: string;
  dense?: boolean;
  rowClass?: (row: R) => string | undefined;
}) {
  const { columns, rows, rowKey, caption, dense, rowClass } = props;
  return (
    <div className="tablewrap" tabIndex={0} role="region" aria-label={caption ?? "Table"}>
      <table className={`table ${dense ? "table--dense" : ""}`}>
        {caption && <caption className="visually-hidden">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={c.align === "right" ? "num right" : undefined} title={c.title}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={rowKey(row, i)} className={rowClass?.(row)}>
              {columns.map((c, ci) =>
                ci === 0 ? (
                  <th key={c.key} scope="row">
                    {c.render(row)}
                  </th>
                ) : (
                  <td key={c.key} className={c.align === "right" ? "num right" : undefined}>
                    {c.render(row)}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The table twin of a chart: every plotted number is also available as text. */
export function TableTwin({ label = "Show the numbers as a table", children }: { label?: string; children: ReactNode }) {
  return (
    <details className="twin">
      <summary>{label}</summary>
      <div className="twin__body">{children}</div>
    </details>
  );
}
