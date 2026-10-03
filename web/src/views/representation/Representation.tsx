/**
 * Representation: the satellite embedding made tangible. Its three leading principal components
 * as an RGB map; how much of the embedding those (and the next) components carry; which places
 * the encoder treats alike (cosine similarity over all features, from the API); which surface
 * field each component follows; and how well the pretraining task can be solved.
 */
import { useMemo } from "react";
import { useEmbedding, useEmbeddingSimilarity, useSurface, useTraining, type SurfaceData } from "@/api/queries";
import type { TrainingLog } from "@/api/types";
import { MethodBars } from "@/components/charts/MethodBars";
import { SelectionBar } from "@/components/controls/SelectionBar";
import { Colorbar } from "@/components/map/Colorbar";
import type { RasterLayer, RgbaLayer } from "@/components/map/MapCanvas";
import { MapFigure } from "@/components/map/MapFigure";
import { ZoomControls } from "@/components/map/ZoomControls";
import { runLabel } from "@/components/shell/Shell";
import { Empty, Note, Panel, QueryState, Segmented } from "@/components/ui/primitives";
import { rgbCss } from "@/lib/colormaps";
import { fmtDate, nearestDateIndex } from "@/lib/dates";
import { componentFieldCorrelation, cumulative, decodeEmbedding, decodeSimilarity, type EmbeddingMap } from "@/lib/embedding";
import { fmt, fmtInt, fmtLatLon, fmtPct, fmtSigned } from "@/lib/format";
import { cellAt, cellCentre } from "@/lib/geo";
import { methodStyle } from "@/lib/methods";
import type { GeoWindow, LinkedView } from "@/state/linkedView";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";
import { surfacePanels } from "../explorer/SurfaceMaps";

/** Components asked from the API: the three that are drawn as colour, and the next ones for context. */
const N_COMPONENTS = 8;
const PC_CSS = ["#C22F2F", "#1F8A3B", "#2B55C7"] as const;
const PC_CHANNEL = ["red", "green", "blue"] as const;
const NEUTRAL = "#6B7680";

function pcName(i: number): string {
  return i < 3 ? `Component ${i + 1} (${PC_CHANNEL[i]})` : `Component ${i + 1}`;
}

export default function Representation() {
  const ctx = useRunContext();
  const { run, embeddingDates, date, geom, detail } = ctx;
  const link = useLinkedView(geom);
  const hasEmb = run.artefacts.embeddings && embeddingDates.length > 0;
  // the embedding exists for the test days; use the selected day if it has one, else the nearest
  const embIndex = hasEmb ? nearestDateIndex(embeddingDates, date) : -1;
  const embDate = embIndex >= 0 ? embeddingDates[embIndex] : null;
  const wanted = Math.min(N_COMPONENTS, detail.model.model?.emb_dim ?? N_COMPONENTS);
  const emb = useEmbedding(run.name, embDate, hasEmb, wanted);
  const surface = useSurface(run.name, embDate);
  const map = useMemo(() => (emb.data ? decodeEmbedding(emb.data) : null), [emb.data]);
  const m = detail.model;

  return (
    <div className="repr">
      {hasEmb && <SelectionBar depth={false} estimate={false} days={embeddingDates} />}
      <header className="viewhead">
        <div className="viewhead__row">
          <div>
            <p className="overline">Representation · {runLabel(run)}</p>
            <h1 className="h1">Inside the satellite embedding</h1>
          </div>
          {hasEmb && <ZoomControls link={link} />}
        </div>
        <p className="caption">
          The encoder compresses each day's surface state into {m.model?.emb_dim ? `${fmtInt(m.model.emb_dim)} learned numbers` : "a set of learned numbers"}
          {map ? ` in each cell of a ${map.ny} × ${map.nx} map` : ""}. Nobody chose what they mean; these figures show what they turned out to hold
          {embDate && embDate !== date ? ` (${fmtDate(embDate)}, the nearest day with an embedding)` : ""}.
        </p>
      </header>

      {!hasEmb ? (
        <Empty title="This run has no exported embeddings" height={200}>
          Run “oceanembed embed” for this configuration to write embeddings.zarr for the test period.
        </Empty>
      ) : (
        <>
          <QueryState query={emb} what="The embedding of this day" height={420}>
            {() => (map ? <Body map={map} link={link} surface={surface.data} date={embDate!} pending={emb.isPlaceholderData} /> : null)}
          </QueryState>
          <Pretraining />
        </>
      )}
    </div>
  );
}

function Body(props: { map: EmbeddingMap; link: LinkedView; surface: SurfaceData | undefined; date: string; pending: boolean }) {
  const { map, link, surface, date, pending } = props;
  const { run, geom, mask, detail, point, setPoint } = useRunContext();
  const [url, setUrl] = useUrlState();
  const explained = map.explained;
  const cum = useMemo(() => cumulative(explained), [explained]);
  const shown = cum[Math.min(2, cum.length - 1)] ?? 0;
  const all = cum[cum.length - 1] ?? 0;
  const rgbaLayer = useMemo<RgbaLayer>(() => ({ pixels: map.rgba, nx: map.nx, ny: map.ny }), [map]);

  // reference cell of the similarity map: the embedding cell under the shared point
  const refCell = point ? cellAt(map.geom, point.lat, point.lon) : null;
  const refIsOcean = !!refCell && map.ocean[refCell.j * map.nx + refCell.i] === 1;
  const refCentre = useMemo(() => (refCell ? cellCentre(map.geom, refCell) : null), [refCell?.j, refCell?.i, map.geom]); // eslint-disable-line react-hooks/exhaustive-deps
  const refBox = useMemo<GeoWindow | null>(() => {
    if (!refCentre) return null;
    return { lon0: refCentre.lon - map.geom.dLon / 2, lon1: refCentre.lon + map.geom.dLon / 2, lat0: refCentre.lat - map.geom.dLat / 2, lat1: refCentre.lat + map.geom.dLat / 2 };
  }, [refCentre, map.geom]);

  // exact cosine similarity over every feature of the embedding, computed by the API
  const centered = url.opts.sim === "rel";
  const simQ = useEmbeddingSimilarity(run.name, date, refCentre?.lat ?? null, refCentre?.lon ?? null, centered, refIsOcean);
  const sim = useMemo(() => (simQ.data ? decodeSimilarity(simQ.data) : null), [simQ.data]);
  const simRaster = useMemo<RasterLayer | null>(() => {
    if (!sim) return null;
    return sim.centered
      ? { data: sim.data, nx: sim.nx, ny: sim.ny, vmin: -1, vmax: 1, cmap: "balance" }
      : { data: sim.data, nx: sim.nx, ny: sim.ny, vmin: 0, vmax: 1, cmap: "viridis" };
  }, [sim]);

  const productOf = useMemo(() => {
    const p = new Map(detail.products.map((x) => [x.variable, x.product]));
    return (v: string) => p.get(v) ?? "";
  }, [detail.products]);
  const panels = useMemo(() => (surface ? surfacePanels(surface, false, productOf) : []), [surface, productOf]);

  const corr = useMemo(() => {
    if (!surface) return null;
    const keys = Object.keys(surface.fields);
    const fields = keys.map((k) => ({ data: surface.fields[k].data, nx: surface.nx, ny: surface.ny }));
    return { keys, names: keys.map((k) => surface.fields[k].longName), r: componentFieldCorrelation(map, fields) };
  }, [surface, map]);

  const valueAtEmb = (lat: number, lon: number) => {
    const c = cellAt(map.geom, lat, lon);
    if (!c) return null;
    const idx = c.j * map.nx + c.i;
    if (!map.ocean[idx]) return "land";
    return `PC 1–3: ${fmt(map.pcs[0]?.[idx])} / ${fmt(map.pcs[1]?.[idx])} / ${fmt(map.pcs[2]?.[idx])}`;
  };

  const pick = (lat: number, lon: number) => {
    const c = cellAt(map.geom, lat, lon);
    if (!c) return;
    const centre = cellCentre(map.geom, c);
    setPoint(centre.lat, centre.lon);
  };

  const coast = mask?.coast ?? null;
  const n = map.pcs.length;

  return (
    <div className={pending ? "is-pending" : ""}>
      <div className="repr__top">
        <Panel
          title="The embedding, seen through its three strongest directions"
          subtitle={`${fmtDate(date)} · same colour, same surface state as the encoder describes it · click a cell to ask which places it treats alike`}
        >
          <MapFigure
            geom={map.geom}
            link={link}
            size="lg"
            title="Embedding (principal components 1–3 as red, green, blue)"
            subtitle={`${map.ny} × ${map.nx} cells of ${Number(map.geom.dLat.toFixed(2))}°`}
            rgba={rgbaLayer}
            coast={coast}
            coastGeom={geom}
            marker={point}
            highlightBox={refBox}
            onPick={pick}
            valueAt={valueAtEmb}
            ariaLabel="Satellite embedding: first three principal components mapped to red, green and blue"
          />
        </Panel>
        <Panel
          title="How much of the embedding the picture shows"
          subtitle="share of the variance carried by each principal component, fitted once for the run so colours mean the same on every day"
        >
          <MethodBars
            inline
            label={`Explained variance of the first ${n} principal components`}
            unit="%"
            digits={1}
            max={Math.max(...explained) * 100}
            rows={explained.map((v, i) => ({
              key: `pc${i}`,
              label: pcName(i),
              value: v * 100,
              style: { ...methodStyle("climatology"), color: PC_CSS[i] ?? NEUTRAL },
              note: `Σ ${fmtPct((cum[i] ?? 0) * 100, 0)}`,
            }))}
          />
          <p className="caption gap-top-sm">
            Σ is the running total. The three coloured components carry {fmtPct(shown * 100, 0)} of the variance{n > 3 ? `, the first ${n} carry ${fmtPct(all * 100, 0)}` : ""}. The rest lives in
            directions no picture shows, so the map is a summary of the embedding; the similarity map below uses all of it.
          </p>
        </Panel>
      </div>

      <div className="twocol gap-top">
        <Panel
          title="Which places does the encoder treat alike?"
          subtitle={
            refIsOcean && refCentre
              ? `cosine similarity between the embedding at ${fmtLatLon(refCentre.lat, refCentre.lon, 1)} and at every other cell, over all ${sim ? fmtInt(sim.embDim) : ""} features`
              : "click an ocean cell of the embedding map to choose a reference"
          }
          actions={
            <Segmented
              label="Similarity measure"
              size="sm"
              value={centered ? "rel" : "raw"}
              onChange={(v) => setUrl({ opts: { sim: v === "raw" ? null : v } })}
              options={[
                { value: "raw", label: "As it is", title: "Cosine similarity of the embedding vectors themselves" },
                { value: "rel", label: "Relative to the average", title: "The run's mean embedding is subtracted first: what makes a place different from the average ocean" },
              ]}
            />
          }
        >
          {!refIsOcean ? (
            <Empty title="No reference cell" height={220}>
              Click an ocean cell on the embedding map above. The map will show how close every other cell is to it.
            </Empty>
          ) : (
            <QueryState query={simQ} what="The similarity map" height={260}>
              {() =>
                simRaster && sim ? (
                  <MapFigure
                    geom={map.geom}
                    link={link}
                    title="Similarity to the selected cell"
                    subtitle={`1 = same direction in the embedding space · ${fmtPct(sim.alike * 100, 0)} of the other cells above 0.5`}
                    raster={simRaster}
                    coast={coast}
                    coastGeom={geom}
                    marker={point}
                    highlightBox={refBox}
                    onPick={pick}
                    pending={simQ.isPlaceholderData}
                    colorbar={
                      sim.centered
                        ? { cmap: "balance", vmin: -1, vmax: 1, units: "", label: "Cosine similarity, mean removed", extend: "none", signed: true }
                        : { cmap: "viridis", vmin: 0, vmax: 1, units: "", label: "Cosine similarity", extend: "none" }
                    }
                    stats={`other cells: ${fmtSigned(sim.min)} to ${fmtSigned(sim.max)}`}
                    ariaLabel="Cosine similarity of every embedding cell to the selected cell"
                  />
                ) : null
              }
            </QueryState>
          )}
          <p className="caption gap-top-sm">
            {centered
              ? "Every cell shares a large common part. With the average embedding removed, red cells depart from the average ocean in the same way as the selected cell and blue cells in the opposite way."
              : "All cells share a large common part, so values are mostly positive; switch to “relative to the average” to see only what makes places differ."}{" "}
            If the pattern follows a water mass or a current system, the encoder found that structure by itself.
          </p>
        </Panel>

        <Panel title="Which surface field does each component follow?" subtitle={`correlation across ocean cells on ${fmtDate(date)}, fields averaged to the embedding grid`}>
          {corr ? <CorrelationTable corr={corr} /> : <Empty title="Surface fields are loading or missing" height={160} />}
          <p className="caption gap-top-sm">
            ±1: the component is, on this day, a copy of that field. Several moderate values in a row: a learned mixture of fields.
          </p>
        </Panel>
      </div>

      <Panel
        className="gap-top"
        title={`The first ${n} components, one at a time`}
        subtitle="each scaled 0–1 over the fitted sample; the first three are the red, green and blue of the map above · hover any map to read all of them at the same place"
        actions={<Colorbar spec={{ cmap: "viridis", vmin: 0, vmax: 1, units: "", label: "Component value", extend: "none" }} ticks={3} />}
      >
        <div className="mapgrid mapgrid--4">
          {map.pcs.map((pc, i) => (
            <MapFigure
              key={i}
              geom={map.geom}
              link={link}
              size="sm"
              title={pcName(i)}
              subtitle={`${fmtPct((explained[i] ?? 0) * 100, 1)} of the variance`}
              raster={{ data: pc, nx: map.nx, ny: map.ny, vmin: 0, vmax: 1, cmap: "viridis" }}
              coast={coast}
              coastGeom={geom}
              highlightBox={refBox}
              onPick={pick}
              colorbar={{ cmap: "viridis", vmin: 0, vmax: 1, units: "", extend: "none" }}
              showColorbar={false}
              ariaLabel={`Principal component ${i + 1} of the embedding`}
            />
          ))}
        </div>
      </Panel>

      <Panel
        className="gap-top"
        title="The surface fields it encodes"
        subtitle={`the inputs of ${fmtDate(date)} on the ${detail.grid.resolution}° grid · the box is the footprint of the selected embedding cell`}
      >
        {panels.length > 0 ? (
          <div className="mapgrid mapgrid--surface">
            {panels.map((p) => (
              <MapFigure
                key={p.key}
                geom={geom}
                link={link}
                size="sm"
                title={p.title}
                subtitle={p.subtitle.split(" · ")[0]}
                raster={p.raster}
                vectors={p.vectors ?? null}
                coastHalo={p.coversLand}
                coast={coast}
                highlightBox={refBox}
                colorbar={p.colorbar}
                digits={p.digits}
                ariaLabel={p.title}
              />
            ))}
          </div>
        ) : (
          <Empty title="Surface fields are loading or missing" height={160} />
        )}
      </Panel>
    </div>
  );
}

function CorrelationTable({ corr }: { corr: { keys: string[]; names: string[]; r: number[][] } }) {
  return (
    <div className="tablewrap">
      <table className="table corrtable">
        <caption className="visually-hidden">Correlation between embedding components and surface fields</caption>
        <thead>
          <tr>
            <th scope="col">Component</th>
            {corr.names.map((n, i) => (
              <th key={corr.keys[i]} scope="col" className="right" title={n}>
                {corr.keys[i].toUpperCase()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {corr.r.map((row, i) => (
            <tr key={i}>
              <th scope="row">
                <span className="methodcell">
                  <i className="dot" style={{ background: PC_CSS[i] ?? NEUTRAL }} />
                  {pcName(i)}
                </span>
              </th>
              {row.map((v, j) => {
                const t = Number.isFinite(v) ? (v + 1) / 2 : 0.5;
                const strong = Math.abs(v) > 0.55;
                return (
                  <td key={j} className="num right corrtable__cell" style={{ background: Number.isFinite(v) ? rgbCss("balance", t) : undefined, color: strong ? "#fff" : undefined }}>
                    {fmtSigned(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption gap-top-sm">{corr.keys.map((k, i) => `${k.toUpperCase()}: ${corr.names[i]}`).join(" · ")}</p>
    </div>
  );
}

/** Pretraining: how much of each hidden surface field the encoder can fill in from the rest. */
function Pretraining() {
  const { run, detail } = useRunContext();
  const hasLog = run.artefacts.training_logs.includes("pretrain");
  const training = useTraining(run.name, hasLog);
  const log: TrainingLog | undefined = training.data?.logs.pretrain;
  const result = useMemo(() => {
    if (!log) return null;
    const mse = log.columns.val_mse_per_channel as Record<string, number>[] | undefined;
    const base = log.columns.val_meanfill_per_channel as Record<string, number>[] | undefined;
    const epochs = log.columns.epoch as number[] | undefined;
    if (!mse || !base || !epochs || mse.length === 0) return null;
    const bestEpoch = log.best?.epoch ?? epochs[epochs.length - 1];
    const at = Math.max(0, epochs.indexOf(bestEpoch));
    const rows = Object.keys(mse[at]).map((k) => {
      const b = base[at]?.[k];
      const recovered = b ? (1 - mse[at][k] / b) * 100 : NaN;
      return { key: k, recovered };
    });
    return { rows, bestEpoch };
  }, [log]);

  if (!hasLog) return null;
  const names = new Map(detail.products.map((p) => [p.variable, p.long_name]));
  const mask = detail.model.pretrain?.mask_ratio;
  const block = detail.model.pretrain?.block;
  return (
    <section className="section" aria-labelledby="h-pretrain">
      <div className="section__head">
        <h2 id="h-pretrain" className="h2">
          How the embedding was learned
        </h2>
        <p className="caption">
          Pretraining uses the surface fields alone
          {mask != null && block != null ? `: ${Math.round(mask * 100)} % of the map is hidden in ${block} × ${block} cell blocks and must be filled in from the rest` : ""}
          . The bars show how much of each hidden field's variance is recovered on validation days, against filling the hole with the mean.
        </p>
      </div>
      <QueryState query={training} what="The pretraining log" height={200}>
        {() =>
          result ? (
            <div className="twocol">
              <Panel title="Hidden blocks filled in, by field" subtitle={`share of variance recovered on validation days · best epoch (${result.bestEpoch})`}>
                <MethodBars
                  label="Share of the variance of hidden blocks recovered, by surface field"
                  unit="%"
                  digits={0}
                  max={100}
                  rows={result.rows.map((r) => ({
                    key: r.key,
                    label: names.get(r.key) ?? r.key,
                    value: Math.max(0, r.recovered),
                    style: methodStyle("model"),
                    note: r.recovered < 0 ? "worse than filling with the mean" : undefined,
                  }))}
                />
              </Panel>
              <Note title="Reading the bars">
                A field filled in well is one the encoder can infer from its surroundings and from the other fields; near zero it is, to the
                encoder, noise. No subsurface data is used here. Whether pretraining then helps the reconstruction is answered by the ablation
                on the Experiments view, whichever way it turns out.
              </Note>
            </div>
          ) : (
            <Empty title="The pretraining log has no per-field validation error" height={120} />
          )
        }
      </QueryState>
    </section>
  );
}
