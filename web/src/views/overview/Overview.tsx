/**
 * Overview: what the product gives you, at a glance. It opens on the reconstruction itself, beside
 * the reanalysis and their difference, on a typical test day at a thermocline depth, both chosen
 * from the metrics (lib/narrative chooseEvidence), never the surface, which is an input. Then how
 * accurate it is in the simplest truthful form (error by depth against the seasonal climatology,
 * the two basins, per test year) and where not to trust it (the depths without skill, computed
 * from the metrics, and the Argo floats with the reanalysis as the floor).
 *
 * Nothing here compares methods: ridge regression, the per-pixel network, the ablation and the
 * embedding are in the Research area. Every statement of a result is computed from the run's
 * metrics, so the text cannot overstate what the numbers say.
 */
import { useMemo, type ReactNode } from "react";
import { useArgoMetrics, useGlorysMetrics } from "@/api/queries";
import type { MetricsResponse } from "@/api/types";
import { useDayVolumes } from "@/api/volumes";
import { Legend, Swatch } from "@/components/charts/marks";
import { MethodBars } from "@/components/charts/MethodBars";
import { MetricProfile } from "@/components/charts/MetricProfile";
import { YearSwitch, useYear } from "@/components/controls/YearSwitch";
import { MapFigure, MaskKey } from "@/components/map/MapFigure";
import { ViewLink, runLabel } from "@/components/shell/Shell";
import { Empty, ErrorState, Icon, Loading, Note, Panel, QueryState } from "@/components/ui/primitives";
import { defaultDateIndex, fmtDate, fmtSpan } from "@/lib/dates";
import { fmt, fmtDepth, fmtInt, fmtSigned, prettyText } from "@/lib/format";
import { inputNames, inputUse } from "@/lib/inputs";
import {
  accuracyHeadline,
  argoSentence,
  basinContrast,
  basinSentence,
  chooseEvidence,
  climatologyName,
  climatologyText,
  compare,
  correlationSentence,
  methodList,
  rangeText,
  scopeToYear,
  trustLimits,
  yearStability,
  yearsOf,
  type Evidence,
} from "@/lib/narrative";
import { nearestDepthIndex } from "@/lib/scales";
import { symmetricLimit } from "@/lib/stats";
import { defaultDepthIndex, useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";
import { divergingBar, levelFields, raster, tempBar } from "../explorer/fields";

function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** One block of the page: a number, a short title, a one-line caption and its figures. */
function Chapter(props: { num: string; id: string; title: string; caption?: ReactNode; children: ReactNode }) {
  return (
    <section className="chapter" aria-labelledby={props.id}>
      <header className="chapter__head">
        <span className="chapter__num">{props.num}</span>
        <h2 id={props.id} className="chapter__title">
          {props.title}
        </h2>
        {props.caption && <p className="chapter__caption">{props.caption}</p>}
      </header>
      {props.children}
    </section>
  );
}

export default function Overview() {
  const ctx = useRunContext();
  const { run, detail, depths, dates, targetDates, embeddingDates, pooledRange, scoped } = ctx;
  const evaluated = run.artefacts.metrics_glorys;
  const metrics = useGlorysMetrics(run.name, evaluated);
  const argo = useArgoMetrics(run.name, run.artefacts.metrics_argo);

  // days that can be shown as evidence: predicted, with a GLORYS target (the same rule as before the embedding left this page)
  const available = useMemo(() => {
    const emb = new Set(embeddingDates);
    const both = dates.filter((d) => targetDates.has(d) && (emb.size === 0 || emb.has(d)));
    return new Set(both.length > 0 ? both : dates.filter((d) => targetDates.has(d)));
  }, [dates, targetDates, embeddingDates]);

  // the day and depth of the evidence, from the metrics; a run that is not evaluated yet gets the plain defaults
  const evidence = useMemo<Evidence | null>(() => (metrics.data ? chooseEvidence(metrics.data, available) : null), [metrics.data, available]);
  const waiting = evaluated && !metrics.data && !metrics.isError;
  const fallbackDate = dates[defaultDateIndex(dates, [targetDates, new Set(embeddingDates)])] ?? null;
  const date = evidence?.date && dates.includes(evidence.date) ? evidence.date : fallbackDate;
  const depthIndex = evidence?.depth != null ? nearestDepthIndex(depths, evidence.depth) : defaultDepthIndex(depths, pooledRange);

  const grid = detail.grid;
  const basinNames = detail.basins.map((b) => b.label);
  const use = useMemo(() => inputUse(detail), [detail]);
  const years = yearsOf(metrics.data);
  const year = useYear(years);
  const scoped1 = useMemo(() => (metrics.data ? scopeToYear(metrics.data, year) : undefined), [metrics.data, year]);
  const argoScoped = useMemo(() => (argo.data ? scopeToYear(argo.data, year) : undefined), [argo.data, year]);
  const test = run.split?.test;

  if (dates.length === 0) {
    return (
      <Empty title="This run has no reconstruction yet" height={280}>
        Its prediction product has not been written (“oceanembed predict”, or the run is still being produced). The views fill in as the
        pipeline writes its outputs; choose another run in the selector meanwhile.
      </Empty>
    );
  }

  return (
    <article className="story">
      <header className="viewhead viewhead--tight">
        <p className="overline">{runLabel(run)}</p>
        <h1 className="h1">Ocean temperature below the surface, every day</h1>
        <p className="caption">
          Daily temperature at {grid.n_depth} depths (0–{fmtInt(depths[depths.length - 1])}&nbsp;m) on a {grid.resolution}° grid
          {basinNames.length > 0 ? ` of the ${joinNames(basinNames)}` : ""}, reconstructed from {use.used.length} satellite surface field
          {use.used.length === 1 ? "" : "s"} ({inputNames(use.used)}). No measurement from below the surface goes in.
        </p>
      </header>

      {waiting || !date ? (
        <Loading height={360} label="Loading the evaluation" />
      ) : (
        <EvidencePanel date={date} depthIndex={depthIndex} evidence={evidence} metrics={metrics.data} />
      )}

      <Chapter
        num="01"
        id="ch-skill"
        title="How accurate is it"
        caption={
          <>
            Scored on {fmtInt(run.n_test_days)} days it never saw ({fmtSpan(test?.start, test?.end)}) against the GLORYS reanalysis, next to the{" "}
            <strong>{climatologyName(run.n_harmonic_terms).slice(4)}</strong> ({climatologyText(run.n_harmonic_terms)} of each cell): what you would use without it.
          </>
        }
      >
        {evaluated ? (
          <QueryState query={metrics} what="The evaluation against GLORYS" height={320}>
            {(mm) => <AccuracyBlock metrics={scoped1 ?? mm} whole={mm} year={year} methods={scoped(methodList(scoped1 ?? mm).map((x) => x.key))} />}
          </QueryState>
        ) : (
          <Empty title="This run has not been evaluated yet" height={140}>
            Run “oceanembed evaluate” for this configuration; the accuracy figures appear here as soon as metrics_glorys.json exists.
          </Empty>
        )}
      </Chapter>

      {(metrics.data || argo.data) && (
        <Chapter num="02" id="ch-limits" title="Where not to trust it" caption="Read this before using a value.">
          <div className="twocol">
            <LimitsPanel metrics={scoped1} resolution={grid.resolution} />
            {run.artefacts.metrics_argo ? (
              <QueryState query={argo} what="The Argo comparison" height={260}>
                {(a) => <ArgoPanel argo={argoScoped ?? a} year={year} />}
              </QueryState>
            ) : (
              <Empty title="This run has not been compared with Argo floats yet" height={160}>
                Run “oceanembed validate-argo” for this configuration.
              </Empty>
            )}
          </div>
        </Chapter>
      )}

      <nav className="nextviews" aria-label="Continue">
        <ViewLink view="accuracy" className="nextviews__item">
          <span className="overline">Accuracy</span>
          <span className="nextviews__title">Error by depth, basin, place and day</span>
          <span className="caption">Error maps, daily error, Argo float profiles one by one.</span>
        </ViewLink>
        <ViewLink view="data" className="nextviews__item">
          <span className="overline">Data &amp; downloads</span>
          <span className="nextviews__title">Take the fields away</span>
          <span className="caption">NetCDF files by month, the inputs used, the released model.</span>
        </ViewLink>
      </nav>
      {run.description && (
        <p className="caption story__about">
          <span className="overline">This run</span> {prettyText(run.description)}
        </p>
      )}
    </article>
  );
}

// ---- the evidence: three linked maps of one typical day at a thermocline depth ---------------------

function EvidencePanel(props: { date: string; depthIndex: number; evidence: Evidence | null; metrics: MetricsResponse | undefined }) {
  const { date, depthIndex, evidence, metrics } = props;
  const { run, geom, mask, depths, targetDates, detail, pooledRange } = useRunContext();
  const link = useLinkedView(geom);
  const { day, error } = useDayVolumes(run.name, date, targetDates.has(date));
  const depth = depths[depthIndex];
  const depthText = fmtDepth(depth);
  const fields = useMemo(() => (day && day.date === date ? levelFields(day, depthIndex, "temp") : null), [day, date, depthIndex]);
  const view = useMemo(() => {
    if (!day || !fields) return null;
    const hint = day.prediction.rangePerDepth?.[depthIndex] ?? day.prediction.range;
    return { hint, diffLimit: fields.diff ? symmetricLimit(fields.diff) : 1 };
  }, [day, fields, depthIndex]);

  // why this day and this depth, in the figure's own words
  const why: string[] = [];
  if (evidence?.date === date && evidence.dayRmse != null) {
    const over = evidence.dayScope === "pooled" ? `${rangeText(metrics?.pooled_range_m ?? pooledRange)} RMSE` : `RMSE at ${depthText}`;
    why.push(`a typical day: its ${over} (${fmt(evidence.dayRmse)}\u00A0°C) is the median of the ${fmtInt(evidence.nDays)} test days`);
  }
  if (evidence?.depth === depth && evidence.depthClimatologyRmse != null) {
    why.push(`${depthText} is where climatology errs most (${fmt(evidence.depthClimatologyRmse)}\u00A0°C), so where there is most to gain`);
  }

  const common = {
    geom,
    link,
    size: "md" as const,
    surfaceMask: mask?.levels[0] ?? null,
    levelMask: mask?.levels[depthIndex] ?? null,
    coast: mask?.coast ?? null,
  };
  const ds = fields?.diffStats ?? null;
  const tBar = view ? tempBar(view.hint.vmin, view.hint.vmax) : null;
  const dBar = view ? divergingBar(view.diffLimit) : null;

  return (
    <Panel
      className="evidence"
      title={`${fmtDate(date)} at ${depthText}: the reconstruction beside the reanalysis it never saw`}
      subtitle={why.length > 0 ? `${why.join(" · ")} · the maps are linked` : "the maps are linked: hover one to read the same cell in all"}
      actions={
        <ViewLink view="explore" className="btn" patch={{ date, depth }}>
          Open in Explorer <Icon name="arrow" />
        </ViewLink>
      }
    >
      {error && !day ? (
        <ErrorState error={new Error(error)} what="The reconstruction of this day" height={260} />
      ) : !day || !fields || !view || !tBar || !dBar || !mask ? (
        <Loading height={300} label="Loading the reconstruction" />
      ) : (
        <>
          <div className="evidence__maps">
            <MapFigure
              {...common}
              title="Reconstruction"
              subtitle="OceanEmbed, from surface fields only"
              raster={raster(fields.recon, day.prediction, view.hint.vmin, view.hint.vmax, "thermal")}
              colorbar={tBar}
              stats={`mean ${fmt(fields.reconStats.mean)} °C`}
              ariaLabel={`Reconstructed temperature at ${depthText} on ${date}`}
            />
            {fields.target ? (
              <MapFigure
                {...common}
                title="GLORYS"
                subtitle={`the target: reanalysis on the ${detail.grid.resolution}° grid`}
                raster={raster(fields.target, day.prediction, view.hint.vmin, view.hint.vmax, "thermal")}
                colorbar={tBar}
                stats="same colour scale"
                ariaLabel={`GLORYS temperature at ${depthText} on ${date}`}
              />
            ) : (
              <Empty title="No GLORYS target for this day" height={160} />
            )}
            {fields.diff && ds ? (
              <MapFigure
                {...common}
                title="Difference"
                subtitle="reconstruction − GLORYS"
                raster={raster(fields.diff, day.prediction, -view.diffLimit, view.diffLimit, "balance")}
                colorbar={dBar}
                stats={`RMSE ${fmt(ds.rms)} · bias ${fmtSigned(ds.mean)} °C`}
                ariaLabel={`Reconstruction minus GLORYS at ${depthText} on ${date}`}
              />
            ) : (
              <Empty title="No difference without a target" height={160} />
            )}
          </div>
          <div className="gap-top-sm">
            <MaskKey />
          </div>
        </>
      )}
    </Panel>
  );
}

// ---- how accurate: against the seasonal climatology ---------------------------------------------------

function AccuracyBlock(props: { metrics: MetricsResponse; /** the unscoped payload (every test year) */ whole: MetricsResponse; year: string | null; methods: string[] }) {
  const { metrics, whole, year, methods: keys } = props;
  const { depths, styleOf, labelOf, caveats, detail, run } = useRunContext();
  const climName = climatologyName(run.n_harmonic_terms);
  const years = yearsOf(whole);
  const stability = useMemo(() => yearStability(whole, "climatology", climName), [whole, climName]);
  const labels = useMemo(() => Object.fromEntries(detail.basins.map((b) => [b.key, b.label])), [detail.basins]);
  const period = year ? `test days of ${year}` : "test period";
  const head = useMemo(() => accuracyHeadline(metrics, climName), [metrics, climName]);
  const corr = useMemo(() => correlationSentence(metrics), [metrics]);
  const basins = useMemo(() => basinContrast(metrics, labels), [metrics, labels]);
  const basinText = useMemo(() => basinSentence(metrics, labels), [metrics, labels]);
  const clim = metrics.pooled.climatology?.rmse ?? null;
  const legend = keys.map((k) => ({ key: k, label: labelOf(k), style: styleOf(k) }));
  const basinMax = Math.max(0, ...basins.flatMap((b) => keys.map((k) => metrics.per_basin[b.key]?.pooled[k]?.rmse ?? 0)));

  return (
    <div className="skill">
      {years.length >= 2 && (
        <div className="toolbar skill__scope">
          <YearSwitch years={years} />
          <span className="caption">applies to the figures of this page</span>
        </div>
      )}
      <div className="skill__grid">
        <Panel title={`Error over ${head.range}`} subtitle={`against GLORYS, ${period} · the depth range where the temperature varies most`}>
          <p className={`skill__headline ${head.beatsClimatology ? "" : "is-negative"}`}>{head.sentence}</p>
          <div className="tablewrap gap-top-sm">
            <table className="table table--snug">
              <caption className="visually-hidden">
                Error over {head.range}: the reconstruction and {climName}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Estimate</th>
                  <th scope="col" className="num right">
                    RMSE (°C)
                  </th>
                  {!year &&
                    years.map((y) => (
                      <th key={y} scope="col" className="num right" title={`RMSE over the test days of ${y}`}>
                        {y}
                      </th>
                    ))}
                  <th scope="col" className="num right" title="RMSE relative to the climatology's">
                    vs climatology
                  </th>
                  <th scope="col" className="num right" title="Correlation after removing the climatology from both sides: day-to-day skill">
                    Anomaly corr.
                  </th>
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => {
                  const b = metrics.pooled[k];
                  const c = k === "climatology" ? null : compare(b?.rmse, clim);
                  return (
                    <tr key={k} className={k === "model" ? "is-lead" : undefined}>
                      <th scope="row">
                        <span className="methodcell">
                          <Swatch style={styleOf(k)} width={26} />
                          {k === "climatology" ? `${climName[4].toUpperCase()}${climName.slice(5)}` : labelOf(k)}
                        </span>
                      </th>
                      <td className="num right">{fmt(b?.rmse)}</td>
                      {!year &&
                        years.map((y) => (
                          <td key={y} className="num right table__sub">
                            {fmt(whole.per_year?.[y]?.pooled?.[k]?.rmse)}
                          </td>
                        ))}
                      <td className="num right">{k === "climatology" ? "the reference" : c ? (c.direction === "same" ? "about the same" : `${fmtSigned(-c.pct, 0)}\u00A0%`) : fmt(null)}</td>
                      <td className="num right">{fmt(b?.corr_anom)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!year && stability && <p className="caption gap-top-sm skill__depths">{stability}</p>}
          {corr && <p className="caption gap-top-sm">{corr}</p>}
        </Panel>
        <Panel title="Error by depth" subtitle={`RMSE against GLORYS, ${period} · lower is better`}>
          <Legend items={legend} band={`${head.range}`} />
          <MetricProfile perDepth={metrics.per_depth} depths={depths} metric="rmse" methods={keys} styleOf={styleOf} labelOf={labelOf} pooledRange={metrics.pooled_range_m} reference="GLORYS" height={310} />
        </Panel>
        <Panel title="Basin by basin" subtitle={`RMSE over ${head.range}${year ? `, ${year}` : ""} · one axis for all basins`}>
          {basins.length === 0 ? (
            <Empty title="No basin breakdown in this run's metrics" height={160} />
          ) : (
            <>
              <div className="basins basins--stack">
                {basins.map((b) => (
                  <div key={b.key}>
                    <p className="multiples__title">{b.label}</p>
                    <MethodBars
                      inline
                      label={`RMSE over ${head.range} in the ${b.label}`}
                      unit="°C"
                      max={basinMax}
                      rows={keys.map((k) => {
                        const block = metrics.per_basin[b.key]?.pooled[k];
                        return {
                          key: k,
                          label: labelOf(k),
                          value: block?.rmse ?? null,
                          style: styleOf(k),
                          emphasis: k === "model",
                          note: k === "climatology" ? undefined : `skill ${fmt(block?.skill_vs_clim)}`,
                        };
                      })}
                    />
                  </div>
                ))}
              </div>
              {basinText && <p className="caption gap-top-sm">{basinText}</p>}
            </>
          )}
        </Panel>
      </div>

      {(caveats.synthetic || caveats.shortTraining) && (
        <div className="skill__notes">
          <Note kind="caveat" title={caveats.synthetic ? "Synthetic data" : "Short training period"}>
            {caveats.synthetic
              ? "These numbers come from the analytic test ocean. They show that the pipeline and the evaluation work; they say nothing about the real ocean."
              : caveats.shortTrainingNote}
          </Note>
        </div>
      )}
    </div>
  );
}

// ---- where not to trust it ----------------------------------------------------------------------------

function LimitsPanel({ metrics, resolution }: { metrics: MetricsResponse | undefined; resolution: number }) {
  const { run } = useRunContext();
  const climName = climatologyName(run.n_harmonic_terms);
  const trust = useMemo(() => (metrics ? trustLimits(metrics, climName) : null), [metrics, climName]);
  return (
    <Panel title="By depth" subtitle={`from the skill against ${climName} at each of the depth levels`}>
      {trust?.limit ? (
        <p className="trust is-limited">{trust.limit}</p>
      ) : trust?.use ? (
        <p className="trust">It improves on {climName} at every depth level.</p>
      ) : (
        <p className="trust">This run has no scores by depth yet.</p>
      )}
      {trust?.use && <p className="caption gap-top-sm skill__depths">{trust.use}</p>}
      <ul className="limits limits--stack gap-top">
        <li>
          <strong>It reproduces a reanalysis, not the ocean itself.</strong> It was trained on GLORYS, a model constrained by observations, and can be at
          best as good as GLORYS; the floats see a somewhat different ocean, as the Argo comparison shows.
        </li>
        <li>
          <strong>One day at a time, {resolution}° cells.</strong> Each day is reconstructed from that day's surface fields alone; anything smaller than
          a cell is averaged out.
        </li>
      </ul>
      <p className="gap-top">
        <ViewLink view="accuracy" className="btn btn--quiet">
          Every score, by depth, place and day <Icon name="arrow" />
        </ViewLink>
      </p>
    </Panel>
  );
}

function ArgoPanel({ argo, year }: { argo: MetricsResponse; year: string | null }) {
  const { styleOf, labelOf, scoped } = useRunContext();
  const keys = useMemo(() => scoped(methodList(argo).map((m) => m.key)), [argo, scoped]);
  const pooled = keys.some((k) => argo.pooled?.[k]?.rmse != null);
  const blocks = pooled ? argo.pooled : argo.overall;
  const sentence = useMemo(() => argoSentence(argo), [argo]);
  const where = pooled ? `over ${rangeText(argo.pooled_range_m)}` : "over all depths";
  return (
    <Panel title="Against Argo float profiles" subtitle={`measurements it never saw · RMSE ${where}${year ? `, ${year}` : ""}, same matchups for every bar · GLORYS itself is the floor`}>
      {sentence && <p className="caption evidence__lead">{sentence}</p>}
      <MethodBars
        inline
        label={`RMSE against Argo ${where}`}
        unit="°C"
        rows={keys.map((k) => ({
          key: k,
          label: k === "glorys" ? "GLORYS (the floor)" : labelOf(k),
          value: blocks[k]?.rmse ?? null,
          style: styleOf(k),
          emphasis: k === "model",
          note: blocks[k]?.bias != null ? `bias ${fmtSigned(blocks[k]?.bias)}` : undefined,
        }))}
      />
      <Note kind="honesty" title="How independent is this?" className="gap-top-sm">
        {argo.metadata.independence_note
          ? prettyText(argo.metadata.independence_note)
          : "Argo profiles are independent of the model's inputs, but GLORYS assimilates Argo and the model is trained on GLORYS."}
      </Note>
    </Panel>
  );
}
