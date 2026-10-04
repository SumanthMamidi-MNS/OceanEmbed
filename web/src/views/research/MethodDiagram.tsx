/**
 * Method diagram: inputs -> encoder -> embedding -> decoder -> temperature -> validation.
 * The three data stages show real fields of the selected day; the numbers come from the run's
 * model configuration.
 */
import { useMemo } from "react";
import { levelOf } from "@/api/binary";
import { useEmbedding, useSurface } from "@/api/queries";
import type { DayVolumes } from "@/api/volumes";
import { MapCanvas, type RasterLayer, type RgbaLayer } from "@/components/map/MapCanvas";
import { ViewLink } from "@/components/shell/Shell";
import { decodeEmbedding } from "@/lib/embedding";
import { inputNames, inputUse } from "@/lib/inputs";
import { fmtDepth, fmtInt } from "@/lib/format";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";

function Arrow() {
  return (
    <svg className="flow__arrow" width="28" height="14" viewBox="0 0 28 14" aria-hidden="true">
      <path d="M0 7h25m0 0l-5-5m5 5l-5 5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function MethodDiagram(props: { day: DayVolumes | null; /** day and level of the thumbnails (default: the shared selection) */ date?: string | null; depthIndex?: number }) {
  const ctx = useRunContext();
  const { run, geom, mask, detail, depths } = ctx;
  const { day } = props;
  const date = props.date ?? ctx.date;
  const depthIndex = props.depthIndex ?? ctx.depthIndex;
  const link = useLinkedView(geom);
  const surface = useSurface(run.name, date);
  const emb = useEmbedding(run.name, date, run.artefacts.embeddings);
  const m = detail.model;
  const use = inputUse(detail);
  const nInputs = use.used.length;

  // the thumbnail is a field the model really reads
  const first = surface.data ? (Object.values(surface.data.fields).find((f) => use.usedKeys.has(f.key)) ?? Object.values(surface.data.fields)[0]) : null;
  const inputRaster = useMemo<RasterLayer | null>(() => {
    if (!first || !surface.data) return null;
    return { data: first.data, nx: surface.data.nx, ny: surface.data.ny, vmin: first.range.vmin, vmax: first.range.vmax, cmap: first.range.diverging ? "balance" : "thermal" };
  }, [first, surface.data]);

  const embMap = useMemo(() => (emb.data ? decodeEmbedding(emb.data) : null), [emb.data]);
  const embLayer = useMemo<RgbaLayer | null>(() => (embMap ? { pixels: embMap.rgba, nx: embMap.nx, ny: embMap.ny } : null), [embMap]);

  const outRaster = useMemo<RasterLayer | null>(() => {
    if (!day) return null;
    const hint = day.prediction.rangePerDepth?.[depthIndex] ?? day.prediction.range;
    return { data: levelOf(day.prediction, depthIndex), nx: day.prediction.nx, ny: day.prediction.ny, vmin: hint.vmin, vmax: hint.vmax, cmap: "thermal" };
  }, [day, depthIndex]);

  const coast = mask?.coast ?? null;
  const grid = `${detail.grid.n_lat} × ${detail.grid.n_lon}`;
  const pretrained = m.main_init !== "scratch";
  const maskPct = pretrained && m.pretrain?.mask_ratio != null ? Math.round(m.pretrain.mask_ratio * 100) : null;
  // the channel count of the configuration is only split into fields and context when every field is used
  const extra = !use.partial && m.model?.in_channels != null ? m.model.in_channels - nInputs : null;

  return (
    <ol className="flow" aria-label="Method, step by step">
      <li className="flow__step flow__step--data">
        <p className="flow__kicker">Input</p>
        <div className="flow__thumb">{inputRaster && <MapCanvas geom={geom} link={link} raster={inputRaster} coast={coast} axes="none" ariaLabel={first?.longName ?? "Surface field"} />}</div>
        <h3 className="flow__title">Surface state</h3>
        <p className="flow__text num">
          {use.partial
            ? `${inputNames(use.used)}: ${nInputs} of the ${use.all.length} surface fields; the other ${use.unused.length} are available but not used by this model`
            : `${nInputs} satellite fields`}
          {extra != null && extra > 0 ? ` + ${extra} context channels (ocean mask, day of year, position)` : ""} · {grid} cells
        </p>
      </li>
      <li className="flow__link">
        <Arrow />
      </li>
      <li className="flow__step flow__step--model">
        <p className="flow__kicker">Model</p>
        <h3 className="flow__title">Encoder</h3>
        <p className="flow__text num">
          Convolutional stem, then a Transformer
          {m.model?.depth != null ? ` of ${m.model.depth} layers` : ""}
          {m.model?.heads != null ? `, ${m.model.heads} attention heads` : ""}
          {m.model?.dim != null ? `, width ${m.model.dim}` : ""}.
        </p>
        {maskPct != null ? (
          <p className="flow__aside">
            Pretrained without labels: {maskPct}&nbsp;% of the surface hidden, reconstructed from the rest.
          </p>
        ) : (
          m.main_init === "scratch" && <p className="flow__aside">Trained from scratch together with the decoder, without pretraining.</p>
        )}
      </li>
      <li className="flow__link">
        <Arrow />
      </li>
      <li className="flow__step flow__step--data">
        <p className="flow__kicker">Representation</p>
        <div className="flow__thumb">
          {embLayer && embMap ? (
            <MapCanvas geom={geom} link={link} rgba={embLayer} coast={coast} axes="none" ariaLabel="Satellite embedding, first three principal components as red, green and blue" />
          ) : (
            <span className="flow__missing">no embedding exported for this day</span>
          )}
        </div>
        <h3 className="flow__title">Satellite embedding</h3>
        <p className="flow__text num">
          {m.model?.emb_dim != null ? `${fmtInt(m.model.emb_dim)} features` : "learned features"}
          {embMap ? ` · ${embMap.ny} × ${embMap.nx} map` : ""} · shown as its 3 leading components.{" "}
          <ViewLink view="research_embedding">Look inside</ViewLink>
        </p>
      </li>
      <li className="flow__link">
        <Arrow />
      </li>
      <li className="flow__step flow__step--model">
        <p className="flow__kicker">Model</p>
        <h3 className="flow__title">Decoder</h3>
        <p className="flow__text">
          Upsamples the embedding back to the full grid, with skip connections from the convolutional stem, and predicts the
          anomaly from climatology at every depth.
        </p>
      </li>
      <li className="flow__link">
        <Arrow />
      </li>
      <li className="flow__step flow__step--data">
        <p className="flow__kicker">Output</p>
        <div className="flow__thumb">
          {outRaster && (
            <MapCanvas
              geom={geom}
              link={link}
              raster={outRaster}
              surfaceMask={mask?.levels[0] ?? null}
              levelMask={mask?.levels[depthIndex] ?? null}
              coast={coast}
              axes="none"
              ariaLabel={`Reconstructed temperature at ${fmtDepth(depths[depthIndex])}`}
            />
          )}
        </div>
        <h3 className="flow__title">Temperature</h3>
        <p className="flow__text num">
          {m.model?.n_depths ?? depths.length} depths, 0–{fmtInt(depths[depths.length - 1])} m · {grid} cells · shown at {fmtDepth(depths[depthIndex])}
        </p>
      </li>
      <li className="flow__link">
        <Arrow />
      </li>
      <li className="flow__step flow__step--check">
        <p className="flow__kicker">Validation</p>
        <h3 className="flow__title">Scored on held-out days</h3>
        <p className="flow__text">
          Against the GLORYS reanalysis and against Argo float profiles, always next to ridge regression and climatology.{" "}
          <ViewLink view="research_scores">See the scores</ViewLink>
        </p>
      </li>
    </ol>
  );
}
