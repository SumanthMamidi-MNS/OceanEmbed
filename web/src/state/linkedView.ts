/**
 * Shared camera and cursor for a group of maps. Every map of a group draws the same geographic
 * window and the same hover cross-hair, so prediction, target and difference can be read cell for
 * cell. State lives outside React: maps redraw on a subscription, nothing re-renders per frame.
 */

export interface GeoWindow {
  lon0: number;
  lon1: number;
  lat0: number;
  lat1: number;
}

export interface GeoPoint {
  lat: number;
  lon: number;
}

type Listener = () => void;

export const MAX_ZOOM = 12;

export class LinkedView {
  private domain: GeoWindow;
  private zoom = 1;
  private cx: number;
  private cy: number;
  private hoverPoint: GeoPoint | null = null;
  private readonly viewListeners = new Set<Listener>();
  private readonly hoverListeners = new Set<Listener>();

  constructor(domain: GeoWindow) {
    this.domain = domain;
    this.cx = (domain.lon0 + domain.lon1) / 2;
    this.cy = (domain.lat0 + domain.lat1) / 2;
  }

  setDomain(domain: GeoWindow): void {
    const d = this.domain;
    if (d.lon0 === domain.lon0 && d.lon1 === domain.lon1 && d.lat0 === domain.lat0 && d.lat1 === domain.lat1) return;
    this.domain = domain;
    this.reset();
  }

  getDomain(): GeoWindow {
    return this.domain;
  }

  get k(): number {
    return this.zoom;
  }

  /** The visible window in degrees. */
  window(): GeoWindow {
    const { domain, zoom } = this;
    const w = (domain.lon1 - domain.lon0) / zoom;
    const h = (domain.lat1 - domain.lat0) / zoom;
    return { lon0: this.cx - w / 2, lon1: this.cx + w / 2, lat0: this.cy - h / 2, lat1: this.cy + h / 2 };
  }

  /** Zoom by `factor` keeping the geographic point (lon, lat) fixed on screen. */
  zoomAt(factor: number, lon: number, lat: number): void {
    const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * factor));
    if (next === this.zoom) return;
    const ratio = this.zoom / next;
    this.cx = lon + (this.cx - lon) * ratio;
    this.cy = lat + (this.cy - lat) * ratio;
    this.zoom = next;
    this.clampCentre();
    this.emitView();
  }

  zoomBy(factor: number): void {
    this.zoomAt(factor, this.cx, this.cy);
  }

  panBy(dLon: number, dLat: number): void {
    if (this.zoom === 1) return;
    const { cx, cy } = this;
    this.cx += dLon;
    this.cy += dLat;
    this.clampCentre();
    if (this.cx !== cx || this.cy !== cy) this.emitView();
  }

  reset(): void {
    const { domain } = this;
    this.zoom = 1;
    this.cx = (domain.lon0 + domain.lon1) / 2;
    this.cy = (domain.lat0 + domain.lat1) / 2;
    this.emitView();
  }

  get hover(): GeoPoint | null {
    return this.hoverPoint;
  }

  setHover(point: GeoPoint | null): void {
    const prev = this.hoverPoint;
    if (prev === point) return;
    if (prev && point && prev.lat === point.lat && prev.lon === point.lon) return;
    this.hoverPoint = point;
    for (const l of this.hoverListeners) l();
  }

  subscribeView = (listener: Listener): (() => void) => {
    this.viewListeners.add(listener);
    return () => this.viewListeners.delete(listener);
  };

  subscribeHover = (listener: Listener): (() => void) => {
    this.hoverListeners.add(listener);
    return () => this.hoverListeners.delete(listener);
  };

  private clampCentre(): void {
    const { domain, zoom } = this;
    const halfW = (domain.lon1 - domain.lon0) / zoom / 2;
    const halfH = (domain.lat1 - domain.lat0) / zoom / 2;
    this.cx = Math.min(domain.lon1 - halfW, Math.max(domain.lon0 + halfW, this.cx));
    this.cy = Math.min(domain.lat1 - halfH, Math.max(domain.lat0 + halfH, this.cy));
  }

  private emitView(): void {
    for (const l of this.viewListeners) l();
  }
}
