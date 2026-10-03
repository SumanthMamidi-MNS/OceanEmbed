/**
 * Tiny URL-backed store. The in-memory state is the truth and updates synchronously; the address
 * bar follows: view changes push a history entry, selection changes (day, depth, point, options)
 * replace it, debounced so scrubbing and animation never flood the History API.
 */
import { useCallback, useSyncExternalStore } from "react";
import { applyPatch, formatUrl, parseUrl, type UrlPatch, type UrlState } from "./url";

type Listener = () => void;

const REPLACE_DELAY_MS = 180;

class Router {
  private state: UrlState;
  private readonly listeners = new Set<Listener>();
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.state = parseUrl(window.location.pathname, window.location.search);
    window.addEventListener("popstate", () => {
      this.cancelPending();
      this.state = parseUrl(window.location.pathname, window.location.search);
      this.emit();
    });
  }

  get = (): UrlState => this.state;

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  set(patch: UrlPatch, mode: "push" | "replace" = "replace"): void {
    const next = applyPatch(this.state, patch);
    const url = formatUrl(next);
    if (url === formatUrl(this.state)) return;
    const viewChanged = next.view !== this.state.view;
    this.state = next;
    if (mode === "push" || viewChanged) {
      this.cancelPending();
      window.history.pushState(null, "", url);
      if (viewChanged) window.scrollTo(0, 0);
    } else {
      this.scheduleReplace();
    }
    this.emit();
  }

  private scheduleReplace(): void {
    if (this.timer) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      window.history.replaceState(null, "", formatUrl(this.state));
    }, REPLACE_DELAY_MS);
  }

  private cancelPending(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }
}

let router: Router | null = null;

function getRouter(): Router {
  router ??= new Router();
  return router;
}

/** Current URL state and a setter (`mode: "push"` adds a history entry). */
export function useUrlState(): [UrlState, (patch: UrlPatch, mode?: "push" | "replace") => void] {
  const r = getRouter();
  const state = useSyncExternalStore(r.subscribe, r.get);
  const set = useCallback((patch: UrlPatch, mode?: "push" | "replace") => r.set(patch, mode), [r]);
  return [state, set];
}

/** Imperative access for event handlers outside React (keyboard shortcuts). */
export function navigate(patch: UrlPatch, mode: "push" | "replace" = "replace"): void {
  getRouter().set(patch, mode);
}

export function currentUrlState(): UrlState {
  return getRouter().get();
}
