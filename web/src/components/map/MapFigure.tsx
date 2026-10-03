/**
 * A map with its title, a live read-out of the hovered cell and its colour bar. The read-out is
 * written straight to the DOM from the linked cursor, so hovering never re-renders React.
 */
import { useEffect, useRef, type ReactNode } from "react";
import { fmt, fmtLatLon, MISSING } from "@/lib/format";
import { Colorbar, type ColorbarSpec } from "./Colorbar";
import { MapCanvas, type MapCanvasProps } from "./MapCanvas";

export interface MapFigureProps extends MapCanvasProps {
  title: ReactNode;
  subtitle?: ReactNode;
  colorbar?: ColorbarSpec | null;
  /** decimals of the read-out value */
  digits?: number;
  /** custom read-out text for a hovered position (defaults to the raster value + units) */
  valueAt?: (lat: number, lon: number) => string | null;
  /** line under the title, e.g. field statistics */
  stats?: ReactNode;
  actions?: ReactNode;
  /** dim the map while newer data loads */
  pending?: boolean;
  size?: "lg" | "md" | "sm";
  /** "value": the read-out shows only this map's value (the coordinates are on a neighbouring map) */
  readout?: "full" | "value";
  /** false: the colour bar is drawn once for a group of maps elsewhere; it still gives the read-out its units */
  showColorbar?: boolean;
  children?: ReactNode;
}

/** Key of the two neutrals every subsurface map uses. */
export function MaskKey() {
  return (
    <div className="maskkey" aria-hidden="true">
      <span>
        <i className="maskkey__land" /> land
      </span>
      <span>
        <i className="maskkey__floor" /> below the sea floor at this depth
      </span>
    </div>
  );
}

export function MapFigure(props: MapFigureProps) {
  const { title, subtitle, colorbar, digits = 2, valueAt, stats, actions, pending, size = "md", readout: readoutMode, showColorbar = true, children, ...map } = props;
  const valueOnly = (readoutMode ?? (size === "sm" ? "value" : "full")) === "value";
  const readout = useRef<HTMLSpanElement | null>(null);
  const latest = useRef({ raster: map.raster, geom: map.geom, valueAt, colorbar, digits, valueOnly });
  latest.current = { raster: map.raster, geom: map.geom, valueAt, colorbar, digits, valueOnly };
  const link = map.link;

  useEffect(() => {
    const update = () => {
      const el = readout.current;
      if (!el) return;
      const h = link.hover;
      if (!h) {
        el.textContent = "";
        return;
      }
      const { raster, geom, valueAt: custom, colorbar: cb, digits: d, valueOnly: only } = latest.current;
      let value: string | null = null;
      if (custom) value = custom(h.lat, h.lon);
      else if (raster) {
        const i = Math.min(raster.nx - 1, Math.max(0, Math.floor(((h.lon - geom.lon0) / (geom.lon1 - geom.lon0)) * raster.nx)));
        const j = Math.min(raster.ny - 1, Math.max(0, Math.floor(((h.lat - geom.lat0) / (geom.lat1 - geom.lat0)) * raster.ny)));
        const v = raster.data[j * raster.nx + i];
        const s = cb?.signed && v > 0 ? `+${fmt(v, d)}` : fmt(v, d);
        value = Number.isFinite(v) ? `${s}\u00A0${cb?.units ?? ""}` : MISSING;
      }
      if (only) el.textContent = value ?? "";
      else el.textContent = value == null ? fmtLatLon(h.lat, h.lon) : `${fmtLatLon(h.lat, h.lon)}  ·  ${value}`;
    };
    update();
    return link.subscribeHover(update);
  }, [link]);

  return (
    <figure className={`mapfig mapfig--${size} ${map.axes === "none" ? "mapfig--bare" : ""} ${pending ? "is-pending" : ""}`}>
      <header className="mapfig__head">
        <div className="mapfig__titles">
          <h3 className="mapfig__title">{title}</h3>
          {subtitle && <span className="mapfig__sub">{subtitle}</span>}
        </div>
        <span ref={readout} className="mapfig__readout mono" aria-hidden="true" />
        {actions}
      </header>
      <div className="mapfig__map">
        <MapCanvas {...map} />
        {children}
      </div>
      {((colorbar && showColorbar) || stats) && (
        <footer className="mapfig__foot">
          {colorbar && showColorbar && <Colorbar spec={colorbar} ticks={size === "sm" ? 3 : 5} />}
          {map.levelMask && map.surfaceMask && map.levelMask !== map.surfaceMask && size === "lg" && <MaskKey />}
          {stats && <div className="mapfig__stats num">{stats}</div>}
        </footer>
      )}
    </figure>
  );
}
