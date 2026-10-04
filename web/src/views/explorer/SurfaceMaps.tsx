/**
 * The surface inputs of the selected day: what the model actually sees. Currents and winds are
 * vector fields, so they are drawn as speed (colour) plus direction (arrows) rather than as two
 * flat signed maps; the U and V components remain available.
 */
import { useMemo } from "react";
import { levelOf } from "@/api/binary";
import { useSurface, type SurfaceData, type SurfaceFieldData } from "@/api/queries";
import type { DayVolumes } from "@/api/volumes";
import type { ColorbarSpec } from "@/components/map/Colorbar";
import type { RasterLayer, VectorLayer } from "@/components/map/MapCanvas";
import { MapFigure } from "@/components/map/MapFigure";
import { QueryState } from "@/components/ui/primitives";
import type { ColormapName } from "@/lib/colormaps";
import { fmtDepth, prettyUnits } from "@/lib/format";
import { inputUse, panelUsed } from "@/lib/inputs";
import { magnitude, quantile } from "@/lib/stats";
import type { LinkedView } from "@/state/linkedView";
import { useRunContext } from "@/state/runContext";

export interface SurfacePanelSpec {
  key: string;
  title: string;
  subtitle: string;
  raster: RasterLayer;
  colorbar: ColorbarSpec;
  vectors?: VectorLayer;
  digits: number;
  /** the product also has values over land (winds): the coastline gets a halo to stay visible */
  coversLand: boolean;
}

function coversLand(data: Float32Array): boolean {
  let n = 0;
  for (let i = 0; i < data.length; i++) if (data[i] === data[i]) n++;
  return n > 0.9 * data.length;
}

const SCALAR_CMAP: Record<string, ColormapName> = { sst: "thermal", sss: "haline" };

function scalarPanel(f: SurfaceFieldData, nx: number, ny: number, product: string): SurfacePanelSpec {
  const cmap: ColormapName = f.range.diverging ? "balance" : (SCALAR_CMAP[f.key] ?? "viridis");
  const units = prettyUnits(f.units);
  return {
    key: f.key,
    title: f.longName,
    subtitle: product,
    raster: { data: f.data, nx, ny, vmin: f.range.vmin, vmax: f.range.vmax, cmap },
    colorbar: { cmap, vmin: f.range.vmin, vmax: f.range.vmax, units, extend: "both", signed: f.range.diverging },
    digits: f.range.vmax - f.range.vmin < 2 ? 3 : 2,
    coversLand: coversLand(f.data),
  };
}

function vectorPanel(u: SurfaceFieldData, v: SurfaceFieldData, nx: number, ny: number, title: string, product: string): SurfacePanelSpec {
  const speed = magnitude(u.data, v.data);
  const vmax = quantile(speed, 0.99) || 1;
  const units = prettyUnits(u.units);
  return {
    key: `${u.key}+${v.key}`,
    title,
    subtitle: `${product} · colour: speed, arrows: direction${coversLand(speed) ? " · also defined over land" : ""}`,
    raster: { data: speed, nx, ny, vmin: 0, vmax, cmap: "speed" },
    colorbar: { cmap: "speed", vmin: 0, vmax, units, extend: "max", label: "Speed" },
    vectors: { u: u.data, v: v.data, nx, ny, refSpeed: vmax },
    digits: vmax < 2 ? 3 : 2,
    coversLand: coversLand(speed),
  };
}

/** Panels for a day's surface fields, in the order the story tells them. */
export function surfacePanels(surface: SurfaceData, components: boolean, productOf: (variable: string) => string): SurfacePanelSpec[] {
  const { fields, nx, ny } = surface;
  const out: SurfacePanelSpec[] = [];
  const done = new Set<string>();
  const pairs: [string, string, string][] = [
    ["uo", "vo", "Surface currents"],
    ["uw", "vw", "10 m winds"],
  ];
  for (const key of Object.keys(fields)) {
    if (done.has(key)) continue;
    const pair = pairs.find(([a, b]) => (a === key || b === key) && fields[a] && fields[b]);
    if (pair && !components) {
      out.push(vectorPanel(fields[pair[0]], fields[pair[1]], nx, ny, pair[2], productOf(pair[0])));
      done.add(pair[0]).add(pair[1]);
    } else {
      out.push(scalarPanel(fields[key], nx, ny, productOf(key)));
      done.add(key);
    }
  }
  return out;
}

export function SurfaceMaps(props: { day: DayVolumes; link: LinkedView; components: boolean; showBasins: boolean }) {
  const { day, link, components, showBasins } = props;
  const ctx = useRunContext();
  const { run, geom, mask, point, setPoint, detail, depthIndex, depths, labelOf } = ctx;
  const surface = useSurface(run.name, day.date);

  const productOf = useMemo(() => {
    const map = new Map(detail.products.map((p) => [p.variable, p.product]));
    return (variable: string) => map.get(variable) ?? "";
  }, [detail.products]);

  const panels = useMemo(
    () => (surface.data ? surfacePanels(surface.data, components, productOf) : []),
    [surface.data, components, productOf],
  );
  const use = useMemo(() => inputUse(detail), [detail]);

  // the reconstruction at the selected depth closes the grid: inputs on the surface, output below
  const reconRaster = useMemo<RasterLayer>(() => {
    const hint = day.prediction.rangePerDepth?.[depthIndex] ?? day.prediction.range;
    return { data: levelOf(day.prediction, depthIndex), nx: day.prediction.nx, ny: day.prediction.ny, vmin: hint.vmin, vmax: hint.vmax, cmap: "thermal" };
  }, [day, depthIndex]);

  const common = {
    geom,
    link,
    coast: mask?.coast ?? null,
    basins: showBasins ? detail.basins : null,
    marker: point,
    onPick: setPoint,
    wheel: "plain" as const,
    size: "md" as const,
  };

  return (
    <QueryState query={surface} what="The surface input fields" height={360}>
      {() => (
        <div className={`surfgrid ${components ? "surfgrid--uv" : ""}`}>
          {panels.map((p) => (
            <MapFigure
              key={p.key}
              {...common}
              title={
                <>
                  {p.title}
                  {use.partial && <span className={`chip ${panelUsed(p.key, use) ? "chip--lead" : ""}`}>{panelUsed(p.key, use) ? "model input" : "not used by the model"}</span>}
                </>
              }
              subtitle={p.subtitle}
              raster={p.raster}
              vectors={p.vectors ?? null}
              coastHalo={p.coversLand}
              colorbar={p.colorbar}
              digits={p.digits}
              pending={surface.isPlaceholderData}
              ariaLabel={`${p.title} on ${day.date}`}
            />
          ))}
          <MapFigure
            {...common}
            surfaceMask={mask?.levels[0] ?? null}
            levelMask={mask?.levels[depthIndex] ?? null}
            title={day.primary === "model" ? "Reconstruction" : labelOf(day.primary)}
            subtitle={`what ${day.primary === "model" ? "the model" : "this estimate"} makes of these fields · ${fmtDepth(depths[depthIndex])}`}
            raster={reconRaster}
            colorbar={{ cmap: "thermal", vmin: reconRaster.vmin, vmax: reconRaster.vmax, units: "°C", extend: "both" }}
            ariaLabel={`Reconstructed temperature at ${fmtDepth(depths[depthIndex])}`}
          />
        </div>
      )}
    </QueryState>
  );
}
