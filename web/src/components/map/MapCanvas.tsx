/**
 * Georeferenced raster map on a 2-D canvas.
 *
 * - Equirectangular with square degrees: the canvas height follows from the width and the domain.
 * - Cells are drawn as they are (nearest neighbour, no smoothing): the grid is the measurement.
 * - Land is a flat neutral from the surface mask; "below the sea floor at this depth" is a second
 *   neutral; NaN is transparent and therefore never drawn as data.
 * - Camera and cursor come from a shared {@link LinkedView}, so a group of maps pans, zooms and
 *   hovers together. Drawing is imperative (requestAnimationFrame); React renders only on resize.
 */
import { memo, useCallback, useEffect, useRef } from "react";
import type { Basin } from "@/api/types";
import { colorizeField, getLut, type ColormapName } from "@/lib/colormaps";
import { fmtLat, fmtLon } from "@/lib/format";
import { aspectRatio, cellAt, cellCentre, graticule, graticuleStep, type GridGeom } from "@/lib/geo";
import { nearestPoint, type PointLayer } from "@/lib/points";
import { canvasFont, color } from "@/lib/theme";
import { useSize } from "@/lib/useSize";
import type { GeoPoint, GeoWindow, LinkedView } from "@/state/linkedView";

export interface RasterLayer {
  /** row-major (ny, nx), row 0 = southern edge */
  data: Float32Array;
  nx: number;
  ny: number;
  vmin: number;
  vmax: number;
  cmap: ColormapName;
  reversed?: boolean;
}

/** Pre-coloured pixels (north-up RGBA), e.g. the PCA-RGB embedding. */
export interface RgbaLayer {
  pixels: Uint8ClampedArray;
  nx: number;
  ny: number;
}

export interface VectorLayer {
  u: Float32Array;
  v: Float32Array;
  nx: number;
  ny: number;
  /** speed drawn as a full-length arrow */
  refSpeed: number;
}

export interface MapCanvasProps {
  geom: GridGeom;
  link: LinkedView;
  raster?: RasterLayer | null;
  rgba?: RgbaLayer | null;
  /** surface ocean mask on the `geom` grid (1 = ocean); the rest is land */
  surfaceMask?: Uint8Array | null;
  /** ocean mask at the displayed depth; surface ocean outside it is drawn as sea floor */
  levelMask?: Uint8Array | null;
  /** coastline segments in cell-corner units of the `coastGeom` grid (default: `geom`) */
  coast?: Float32Array | null;
  /** grid the coastline was derived on, when the raster grid is coarser (the embedding) */
  coastGeom?: GridGeom | null;
  /** light halo under the coastline, for fields that also cover land (winds) */
  coastHalo?: boolean;
  basins?: readonly Basin[] | null;
  vectors?: VectorLayer | null;
  /** observation points (Argo profiles), drawn in colour-binned batches */
  points?: PointLayer | null;
  /** index into `points` of the selected one (ringed) */
  selectedPoint?: number | null;
  onPickPoint?: (index: number) => void;
  marker?: GeoPoint | null;
  sectionLine?: { along: "zonal" | "meridional"; value: number } | null;
  /** outlined rectangle, e.g. the footprint of one embedding cell */
  highlightBox?: GeoWindow | null;
  onPick?: (lat: number, lon: number) => void;
  axes?: "full" | "none";
  /** "plain": the wheel zooms; "modifier": only Ctrl/Cmd + wheel zooms (page keeps scrolling) */
  wheel?: "plain" | "modifier";
  ariaLabel: string;
}

const MARGIN_FULL = { top: 6, right: 6, bottom: 20, left: 38 };
const MARGIN_NONE = { top: 0, right: 0, bottom: 0, left: 0 };

interface Layout {
  width: number;
  height: number;
  x0: number;
  y0: number;
  pw: number;
  ph: number;
}

function layoutFor(width: number, aspect: number, axes: "full" | "none"): Layout {
  const m = axes === "full" ? MARGIN_FULL : MARGIN_NONE;
  const pw = Math.max(10, width - m.left - m.right);
  const ph = Math.max(10, Math.round(pw / aspect));
  return { width, height: ph + m.top + m.bottom, x0: m.left, y0: m.top, pw, ph };
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function MapCanvasImpl(props: MapCanvasProps) {
  const { geom, link, axes = "full", wheel = "modifier", ariaLabel } = props;
  const [wrapRef, size] = useSize<HTMLDivElement>();
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const rasterCanvas = useRef<HTMLCanvasElement | null>(null);
  const floorCanvas = useRef<HTMLCanvasElement | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const frame = useRef(0);

  const aspect = aspectRatio(geom);
  const layout = layoutFor(size.width, aspect, axes);
  const layoutRef = useRef(layout);
  layoutRef.current = layout;

  /** geographic -> pixel transform for the current camera */
  const projection = useCallback(() => {
    const l = layoutRef.current;
    const win = propsRef.current.link.window();
    const sx = l.pw / (win.lon1 - win.lon0);
    const sy = l.ph / (win.lat1 - win.lat0);
    return {
      win,
      sx,
      sy,
      X: (lon: number) => l.x0 + (lon - win.lon0) * sx,
      Y: (lat: number) => l.y0 + (win.lat1 - lat) * sy,
      lonAt: (px: number) => win.lon0 + (px - l.x0) / sx,
      latAt: (py: number) => win.lat1 - (py - l.y0) / sy,
    };
  }, []);

  const drawBase = useCallback(() => {
    const canvas = baseRef.current;
    const l = layoutRef.current;
    if (!canvas || l.width <= 0) return;
    const p = propsRef.current;
    const g = p.geom;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, l.width, l.height);
    const { win, sx, sy, X, Y } = projection();

    ctx.save();
    ctx.beginPath();
    ctx.rect(l.x0, l.y0, l.pw, l.ph);
    ctx.clip();
    ctx.fillStyle = color.land;
    ctx.fillRect(l.x0, l.y0, l.pw, l.ph);

    const dx = X(g.lon0);
    const dy = Y(g.lat1);
    const dw = (g.lon1 - g.lon0) * sx;
    const dh = (g.lat1 - g.lat0) * sy;
    ctx.imageSmoothingEnabled = false;
    if (floorCanvas.current && p.levelMask && p.surfaceMask) ctx.drawImage(floorCanvas.current, dx, dy, dw, dh);
    if (rasterCanvas.current && (p.raster || p.rgba)) ctx.drawImage(rasterCanvas.current, dx, dy, dw, dh);

    // graticule
    const lonStep = graticuleStep(win.lon1 - win.lon0, Math.max(3, l.pw / 110));
    const latStep = graticuleStep(win.lat1 - win.lat0, Math.max(2, l.ph / 60));
    const lons = graticule(win.lon0, win.lon1, lonStep);
    const lats = graticule(win.lat0, win.lat1, latStep);
    ctx.strokeStyle = color.graticule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const lon of lons) {
      const x = Math.round(X(lon)) + 0.5;
      ctx.moveTo(x, l.y0);
      ctx.lineTo(x, l.y0 + l.ph);
    }
    for (const lat of lats) {
      const y = Math.round(Y(lat)) + 0.5;
      ctx.moveTo(l.x0, y);
      ctx.lineTo(l.x0 + l.pw, y);
    }
    ctx.stroke();

    // coastline (cell edges between ocean and land at the surface)
    if (p.coast && p.coast.length > 0) {
      const c = p.coast;
      const cg = p.coastGeom ?? g;
      const cx = cg.dLon * sx;
      const cy = cg.dLat * sy;
      const ox = X(cg.lon0);
      const oy = Y(cg.lat0);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let s = 0; s < c.length; s += 4) {
        ctx.moveTo(ox + c[s] * cx, oy - c[s + 1] * cy);
        ctx.lineTo(ox + c[s + 2] * cx, oy - c[s + 3] * cy);
      }
      if (p.coastHalo) {
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      ctx.strokeStyle = color.coast;
      ctx.lineWidth = p.coastHalo ? 1.25 : l.pw < 420 ? 0.75 : 1;
      ctx.stroke();
    }

    // basin outlines
    if (p.basins && p.basins.length > 0) {
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1;
      ctx.font = canvasFont.basin;
      ctx.textBaseline = "top";
      for (const b of p.basins) {
        const bx = X(b.box.lon_min);
        const by = Y(b.box.lat_max);
        const bw = (b.box.lon_max - b.box.lon_min) * sx;
        const bh = (b.box.lat_max - b.box.lat_min) * sy;
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineDashOffset = 4.5;
        ctx.strokeRect(Math.round(bx) + 0.5, Math.round(by) + 0.5, Math.round(bw), Math.round(bh));
        ctx.strokeStyle = color.ink;
        ctx.lineDashOffset = 0;
        ctx.strokeRect(Math.round(bx) + 0.5, Math.round(by) + 0.5, Math.round(bw), Math.round(bh));
        // label inside the top-left corner, haloed so it reads on any colour
        const text = b.label.toUpperCase();
        ctx.setLineDash([]);
        ctx.textAlign = "left";
        ctx.lineJoin = "round";
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(255,255,255,0.92)";
        ctx.strokeText(text, Math.round(bx) + 7, Math.round(by) + 7);
        ctx.fillStyle = color.ink;
        ctx.fillText(text, Math.round(bx) + 7, Math.round(by) + 7);
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1;
      }
      ctx.setLineDash([]);
    }

    if (p.vectors) drawVectors(ctx, p.vectors, g, l, X, Y, sx, sy);

    // observation points
    if (p.points && p.points.n > 0) {
      const pts = p.points;
      // smaller dots when there are thousands, larger again when zoomed in
      const zoom = (g.lon1 - g.lon0) / (win.lon1 - win.lon0);
      const base = pts.n > 6000 ? 1.5 : pts.n > 1500 ? 2 : l.pw < 500 ? 2 : 2.75;
      const r = Math.min(4.5, base * Math.sqrt(zoom));
      const xLo = l.x0 - r;
      const xHi = l.x0 + l.pw + r;
      const yLo = l.y0 - r;
      const yHi = l.y0 + l.ph + r;
      // the halo is stroked first and the dots filled over it, so overlapping dots never erase each other
      ctx.lineWidth = r < 2 ? 1 : 1.5;
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      // one path per colour bin, lowest values first: the largest errors are drawn last, on top
      for (let b = 0; b + 1 < pts.binStart.length; b++) {
        const from = pts.binStart[b];
        const to = pts.binStart[b + 1];
        if (to === from) continue;
        ctx.beginPath();
        for (let o = from; o < to; o++) {
          const i = pts.order[o];
          const x = X(pts.lon[i]);
          const y = Y(pts.lat[i]);
          if (x < xLo || x > xHi || y < yLo || y > yHi) continue;
          ctx.moveTo(x + r, y);
          ctx.arc(x, y, r, 0, Math.PI * 2);
        }
        ctx.fillStyle = pts.colors[b];
        ctx.stroke();
        ctx.fill();
      }
      const s = p.selectedPoint;
      if (s != null && s >= 0 && s < pts.n) drawRing(ctx, X(pts.lon[s]), Y(pts.lat[s]), 7);
    }

    // section line through the selected point
    if (p.sectionLine) {
      ctx.setLineDash([6, 4]);
      for (const [stroke, width] of [
        ["rgba(255,255,255,0.9)", 3],
        [color.ink, 1.25],
      ] as const) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = width;
        ctx.beginPath();
        if (p.sectionLine.along === "zonal") {
          const y = Y(p.sectionLine.value);
          ctx.moveTo(l.x0, y);
          ctx.lineTo(l.x0 + l.pw, y);
        } else {
          const x = X(p.sectionLine.value);
          ctx.moveTo(x, l.y0);
          ctx.lineTo(x, l.y0 + l.ph);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    if (p.highlightBox) {
      const b = p.highlightBox;
      const bx = X(b.lon0);
      const by = Y(b.lat1);
      const bw = (b.lon1 - b.lon0) * sx;
      const bh = (b.lat1 - b.lat0) * sy;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.strokeRect(bx, by, bw, bh);
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = color.ink;
      ctx.strokeRect(bx, by, bw, bh);
    }

    if (p.marker) drawRing(ctx, X(p.marker.lon), Y(p.marker.lat), 6);
    ctx.restore();

    // frame and axis labels
    ctx.strokeStyle = color.ruleStrong;
    ctx.lineWidth = 1;
    ctx.strokeRect(l.x0 + 0.5, l.y0 + 0.5, l.pw - 1, l.ph - 1);
    if ((p.axes ?? "full") === "full") {
      ctx.fillStyle = color.ink3;
      ctx.font = canvasFont.tick;
      ctx.textBaseline = "top";
      ctx.textAlign = "center";
      const lonDigits = lonStep < 1 ? 1 : 0;
      for (const lon of lons) {
        const x = X(lon);
        if (x < l.x0 + 12 || x > l.x0 + l.pw - 12) continue;
        ctx.fillText(fmtLon(lon, lonDigits), x, l.y0 + l.ph + 5);
      }
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const latDigits = latStep < 1 ? 1 : 0;
      for (const lat of lats) {
        const y = Y(lat);
        if (y < l.y0 + 6 || y > l.y0 + l.ph - 6) continue;
        ctx.fillText(fmtLat(lat, latDigits), l.x0 - 5, y);
      }
    }
  }, [projection]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    const l = layoutRef.current;
    if (!canvas || l.width <= 0) return;
    const p = propsRef.current;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, l.width, l.height);
    const hover = p.link.hover;
    if (!hover) return;
    const { X, Y, sx, sy } = projection();
    const x = X(hover.lon);
    const y = Y(hover.lat);
    if (x < l.x0 || x > l.x0 + l.pw || y < l.y0 || y > l.y0 + l.ph) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(l.x0, l.y0, l.pw, l.ph);
    ctx.clip();
    for (const [stroke, width] of [
      ["rgba(255,255,255,0.7)", 3],
      ["rgba(20,33,43,0.75)", 1],
    ] as const) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, l.y0);
      ctx.lineTo(Math.round(x) + 0.5, l.y0 + l.ph);
      ctx.moveTo(l.x0, Math.round(y) + 0.5);
      ctx.lineTo(l.x0 + l.pw, Math.round(y) + 0.5);
      ctx.stroke();
    }
    // the hovered cell
    const cw = Math.max(4, p.geom.dLon * sx);
    const ch = Math.max(4, p.geom.dLat * sy);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 3;
    ctx.strokeRect(x - cw / 2, y - ch / 2, cw, ch);
    ctx.strokeStyle = color.ink;
    ctx.lineWidth = 1.25;
    ctx.strokeRect(x - cw / 2, y - ch / 2, cw, ch);
    ctx.restore();
  }, [projection]);

  const schedule = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      drawBase();
      drawOverlay();
    });
  }, [drawBase, drawOverlay]);

  // --- offscreen layers -------------------------------------------------------------------
  const { raster, rgba, surfaceMask, levelMask } = props;
  const rData = raster?.data;
  const rNx = raster?.nx;
  const rNy = raster?.ny;
  const rMin = raster?.vmin;
  const rMax = raster?.vmax;
  const rCmap = raster?.cmap;
  const rRev = raster?.reversed;
  useEffect(() => {
    const nx = rgba ? rgba.nx : rNx;
    const ny = rgba ? rgba.ny : rNy;
    if (!nx || !ny) {
      rasterCanvas.current = null;
      schedule();
      return;
    }
    let canvas = rasterCanvas.current;
    if (!canvas || canvas.width !== nx || canvas.height !== ny) {
      canvas = makeCanvas(nx, ny);
      rasterCanvas.current = canvas;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(nx, ny);
    if (rgba) img.data.set(rgba.pixels);
    else if (rData && rCmap && rMin != null && rMax != null) {
      colorizeField(rData, nx, ny, rMin, rMax, getLut(rCmap, rRev), img.data);
    }
    ctx.putImageData(img, 0, 0);
    schedule();
  }, [rgba, rData, rNx, rNy, rMin, rMax, rCmap, rRev, schedule]);

  useEffect(() => {
    if (!surfaceMask || !levelMask) {
      floorCanvas.current = null;
      schedule();
      return;
    }
    const { nLon: nx, nLat: ny } = geom;
    const canvas = makeCanvas(nx, ny);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(nx, ny);
    const r = parseInt(color.seafloor.slice(1, 3), 16);
    const gg = parseInt(color.seafloor.slice(3, 5), 16);
    const b = parseInt(color.seafloor.slice(5, 7), 16);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const src = j * nx + i;
        if (surfaceMask[src] && !levelMask[src]) {
          const dst = ((ny - 1 - j) * nx + i) * 4;
          img.data[dst] = r;
          img.data[dst + 1] = gg;
          img.data[dst + 2] = b;
          img.data[dst + 3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    floorCanvas.current = canvas;
    schedule();
  }, [surfaceMask, levelMask, geom, schedule]);

  // --- canvas size ------------------------------------------------------------------------
  useEffect(() => {
    const dpr = window.devicePixelRatio || 1;
    for (const canvas of [baseRef.current, overlayRef.current]) {
      if (!canvas || layout.width <= 0) continue;
      const w = Math.round(layout.width * dpr);
      const h = Math.round(layout.height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }
    drawBase();
    drawOverlay();
  }, [layout.width, layout.height, drawBase, drawOverlay]);

  // redraw when any drawn prop changes
  useEffect(() => {
    schedule();
  }, [
    schedule,
    props.coast,
    props.coastGeom,
    props.coastHalo,
    props.basins,
    props.vectors,
    props.points,
    props.selectedPoint,
    props.marker,
    props.sectionLine,
    props.highlightBox,
    geom,
    axes,
  ]);

  useEffect(() => {
    const unView = link.subscribeView(schedule);
    const unHover = link.subscribeHover(drawOverlay);
    // web fonts may arrive after the first paint: redraw the axis labels once they do
    let alive = true;
    document.fonts?.ready.then(() => alive && schedule()).catch(() => undefined);
    return () => {
      alive = false;
      unView();
      unHover();
      if (frame.current) cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [link, schedule, drawOverlay]);

  // --- interaction ------------------------------------------------------------------------
  const drag = useRef<{ x: number; y: number; moved: boolean; id: number } | null>(null);

  const geoAt = useCallback(
    (e: { clientX: number; clientY: number }): (GeoPoint & { inside: boolean; px: number; py: number }) | null => {
      const canvas = overlayRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const l = layoutRef.current;
      const { lonAt, latAt } = projection();
      const inside = px >= l.x0 && px <= l.x0 + l.pw && py >= l.y0 && py <= l.y0 + l.ph;
      return { lon: lonAt(px), lat: latAt(py), inside, px, py };
    },
    [projection],
  );

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (d && d.id === e.pointerId) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (d.moved || Math.hypot(dx, dy) > 3) {
        const { sx, sy } = projection();
        d.moved = true;
        d.x = e.clientX;
        d.y = e.clientY;
        link.panBy(-dx / sx, dy / sy);
        return;
      }
    }
    const pos = geoAt(e);
    if (!pos || !pos.inside) {
      link.setHover(null);
      return;
    }
    const cell = cellAt(geom, pos.lat, pos.lon);
    link.setHover(cell ? cellCentre(geom, cell) : null);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!d || d.moved) return;
    const pos = geoAt(e);
    if (!pos || !pos.inside) return;
    const p = propsRef.current;
    if (p.points && p.onPickPoint) {
      const { X, Y } = projection();
      const best = nearestPoint(p.points, X, Y, pos.px, pos.py);
      if (best >= 0) {
        p.onPickPoint(best);
        return;
      }
    }
    if (p.onPick) {
      const cell = cellAt(p.geom, pos.lat, pos.lon);
      if (cell) {
        const c = cellCentre(p.geom, cell);
        p.onPick(c.lat, c.lon);
      }
    }
  };

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      if (wheel === "modifier" && !(e.ctrlKey || e.metaKey)) return;
      const pos = geoAt(e);
      if (!pos || !pos.inside) return;
      e.preventDefault();
      link.zoomAt(Math.exp(-e.deltaY * 0.0018), pos.lon, pos.lat);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [wheel, link, geoAt]);

  const onDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = geoAt(e);
    if (pos?.inside) link.zoomAt(e.shiftKey ? 0.5 : 2, pos.lon, pos.lat);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "+" || e.key === "=") link.zoomBy(1.5);
    else if (e.key === "-" || e.key === "_") link.zoomBy(1 / 1.5);
    else if (e.key === "0") link.reset();
    else return;
    e.preventDefault();
  };

  return (
    <div
      ref={wrapRef}
      className="mapcanvas"
      style={{ height: size.width > 0 ? layout.height : undefined, aspectRatio: size.width > 0 ? undefined : `${aspect}` }}
      tabIndex={0}
      role="group"
      aria-label={`${ariaLabel}. Map: press plus or minus to zoom, 0 to reset the view.`}
      onKeyDown={onKeyDown}
    >
      <canvas ref={baseRef} className="mapcanvas__layer" role="img" aria-label={ariaLabel} />
      <canvas
        ref={overlayRef}
        className="mapcanvas__layer mapcanvas__layer--top"
        aria-hidden="true"
        onPointerMove={onPointerMove}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onPointerLeave={() => link.setHover(null)}
        onDoubleClick={onDoubleClick}
      />
    </div>
  );
}

function drawRing(ctx: CanvasRenderingContext2D, x: number, y: number, r: number): void {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.lineWidth = 4;
  ctx.strokeStyle = color.markerHalo;
  ctx.stroke();
  ctx.lineWidth = 1.75;
  ctx.strokeStyle = color.marker;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = color.marker;
  ctx.fill();
}

/** Direction glyphs: block-averaged arrows on a regular screen lattice, length ~ speed. */
function drawVectors(
  ctx: CanvasRenderingContext2D,
  vec: VectorLayer,
  g: GridGeom,
  l: Layout,
  X: (lon: number) => number,
  Y: (lat: number) => number,
  sx: number,
  sy: number,
): void {
  const cellPx = ((g.lon1 - g.lon0) / vec.nx) * sx;
  const spacing = l.pw < 360 ? 16 : l.pw < 700 ? 20 : 24;
  const step = Math.max(1, Math.round(spacing / cellPx));
  const dLon = (g.lon1 - g.lon0) / vec.nx;
  const dLat = (g.lat1 - g.lat0) / vec.ny;
  const maxLen = step * cellPx * 0.92;
  const segs: number[] = [];
  for (let J = 0; J + step <= vec.ny; J += step) {
    for (let I = 0; I + step <= vec.nx; I += step) {
      let su = 0;
      let sv = 0;
      let n = 0;
      for (let dj = 0; dj < step; dj++) {
        for (let di = 0; di < step; di++) {
          const idx = (J + dj) * vec.nx + I + di;
          const u = vec.u[idx];
          const v = vec.v[idx];
          if (u === u && v === v) {
            su += u;
            sv += v;
            n++;
          }
        }
      }
      if (n * 2 < step * step) continue;
      const u = su / n;
      const v = sv / n;
      const cx = X(g.lon0 + (I + step / 2) * dLon);
      const cy = Y(g.lat0 + (J + step / 2) * dLat);
      if (cx < l.x0 - maxLen || cx > l.x0 + l.pw + maxLen || cy < l.y0 - maxLen || cy > l.y0 + l.ph + maxLen) continue;
      const speed = Math.hypot(u, v);
      if (!(speed > 0)) continue;
      // length grows with the square root of speed: slow flow still shows its direction, colour carries the magnitude
      const len = Math.max(3, Math.sqrt(Math.min(1, speed / vec.refSpeed)) * maxLen);
      const ux = u / speed;
      const uy = -v / speed;
      // tail, tip, direction, length, and whether the colour underneath is dark (fast flow)
      segs.push(cx - (ux * len) / 2, cy - (uy * len) / 2, cx + (ux * len) / 2, cy + (uy * len) / 2, ux, uy, len, speed / vec.refSpeed > 0.52 ? 1 : 0);
    }
  }
  void sy;
  const head = Math.min(4, Math.max(2.25, maxLen * 0.3));
  // ink arrows on light colours, white arrows on dark ones: direction stays readable over the whole speed range
  for (const [stroke, width, onDark] of [
    ["rgba(255,255,255,0.55)", 2.5, 0],
    ["rgba(20,33,43,0.92)", 1, 0],
    ["rgba(10,20,26,0.5)", 2.5, 1],
    ["rgba(255,255,255,0.96)", 1.1, 1],
  ] as const) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let s = 0; s < segs.length; s += 8) {
      if (segs[s + 7] !== onDark) continue;
      const [x0, y0, x1, y1, ux, uy, len] = [segs[s], segs[s + 1], segs[s + 2], segs[s + 3], segs[s + 4], segs[s + 5], segs[s + 6]];
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      if (len > 4) {
        const h = Math.min(head, len * 0.5);
        // two barbs at +-150 degrees from the direction of flow
        ctx.moveTo(x1 - h * (ux * 0.866 - uy * 0.5), y1 - h * (uy * 0.866 + ux * 0.5));
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 - h * (ux * 0.866 + uy * 0.5), y1 - h * (uy * 0.866 - ux * 0.5));
      }
    }
    ctx.stroke();
  }
}

export const MapCanvas = memo(MapCanvasImpl);
