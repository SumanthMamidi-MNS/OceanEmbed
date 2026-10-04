/**
 * Overview: does it work, at a glance. A fixed, curated result panel, not an exploration tool
 * (that is the Explorer). It opens on the evidence: reconstruction, GLORYS and their difference
 * side by side on a typical test day at a thermocline depth, both chosen from the metrics
 * (lib/narrative chooseEvidence), never the surface, which is an input. Then the headline numbers
 * of every method, the skill by depth, the basin contrast and the Argo comparison with its floor,
 * a short strip of how it works, and what it does not show. Every statement of a result is
 * computed from the run's metrics, so the text cannot overstate what the numbers say.
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
import { fmt, fmtDepth, fmtInt, fmtLat, fmtLon, fmtSigned, prettyText } from "@/lib/format";
import { inputNames, inputUse } from "@/lib/inputs";
import {
  ablationFindings,
  argoSentence,
  basinContrast,
  basinSentence,
  chooseEvidence,
  climatologyText,
  compare,
  correlationSentence,
  depthSkill,

  headline,
  methodList,
  mlpSentence,
  scopeToYear,
  yearStability,
  yearsOf,
  type Evidence,
} from "@/lib/narrative";
import { nearestDepthIndex } from "@/lib/scales";
import { symmetricLimit } from "@/lib/stats";
import { defaultDepthIndex, useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";
import { divergingBar, levelFields, raster, tempBar } from "../explorer/fields";
import { MethodDiagram } from "./MethodDiagram";

function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

function rangeText(range: readonly number[] | null | undefined): string {
  return range && range.length >= 2 ? `${fmt(range[0], 0)}–${fmt(range[1], 0)}\u00A0m` : "the pooled depth range";
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
  const { run, detail, depths, dates, targetDates, embeddingDates, pooledRange, geom, mask } = ctx;
  const evaluated = run.artefacts.metrics_glorys;
  const metrics = useGlorysMetrics(run.name, evaluated);
  const argo = useArgoMetrics(run.name, run.artefacts.metrics_argo);

  // days that can be shown as evidence: predicted, with a GLORYS target (and an embedding for the method strip)
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
  const m = detail.model;
  // the masked-surface pretraining is part of the main model only when its encoder starts from it
  const maskPct = m.main_init !== "scratch" && m.pretrain?.mask_ratio != null ? Math.round(m.pretrain.mask_ratio * 100) : null;
  const years = yearsOf(metrics.data);
  const year = useYear(years);
  const scoped = useMemo(() => (metrics.data ? scopeToYear(metrics.data, year) : undefined), [metrics.data, year]);
  const argoScoped = useMemo(() => (argo.data ? scopeToYear(argo.data, year) : undefined), [argo.data, year]);
  const nArgo = detail.counts.n_argo_profiles ?? null;
  const nOcean = mask?.nOcean[0] ?? null;
  const perDay = nArgo && run.n_test_days ? nArgo / run.n_test_days : null;

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
        <h1 className="h1">Subsurface temperature, reconstructed from the surface</h1>
        <p className="caption">
          {use.used.length} daily satellite surface field{use.used.length === 1 ? "" : "s"} in
          {use.partial ? ` (${inputNames(use.used)})` : ""}, temperature at {grid.n_depth} depths (0–{fmtInt(depths[depths.length - 1])}&nbsp;m) out, on a{" "}
          {grid.resolution}° grid{basinNames.length > 0 ? ` of the ${joinNames(basinNames)}` : ""} ({fmtLat(geom.lat0, 0)}–{fmtLat(geom.lat1, 0)},{" "}
          {fmtLon(geom.lon0, 0)}–{fmtLon(geom.lon1, 0)}). No subsurface measurement goes in. Trained on {fmtInt(run.n_train_days)} days (
          {fmtSpan(run.split?.train?.start, run.split?.train?.end)}), tested on {fmtInt(run.n_test_days)} days it never saw (
          {fmtSpan(run.split?.test?.start, run.split?.test?.end)}).
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
        title="How good is it"
        caption={
          <>
            Scored on the test period against GLORYS and against what is free: <strong>climatology</strong> ({climatologyText(run.n_harmonic_terms)} of each
            cell, fitted on the training period) and <strong>ridge regression</strong> (a linear model on the same surface fields).
          </>
        }
      >
        {evaluated ? (
          <QueryState query={metrics} what="The evaluation against GLORYS" height={320}>
            {(mm) => <SkillBlock metrics={scoped ?? mm} whole={mm} year={year} />}
          </QueryState>
        ) : (
          <Empty title="This run has not been evaluated yet" height={140}>
            Run “oceanembed evaluate” for this configuration; the skill figures appear here as soon as metrics_glorys.json exists.
          </Empty>
        )}
      </Chapter>

      {(metrics.data || argo.data) && (
        <Chapter
          num="02"
          id="ch-where"
          title="Where it holds, and against observations"
          caption="The same pooled score in each basin, and against Argo float profiles: in-situ measurements the model never saw as input."
        >
          <div className="twocol">
            {scoped && <BasinPanel metrics={scoped} year={year} />}
            {run.artefacts.metrics_argo ? (
              <QueryState query={argo} what="The Argo validation" height={260}>
                {(a) => <ArgoPanel argo={argoScoped ?? a} year={year} />}
              </QueryState>
            ) : (
              <Empty title="This run has not been validated against Argo yet" height={160}>
                Run “oceanembed validate-argo” for this configuration.
              </Empty>
            )}
          </div>
        </Chapter>
      )}

      <Chapter
        num="03"
        id="ch-method"
        title="How it works"
        caption={
          <>
            {nArgo && perDay && nOcean
              ? `Below the surface the ocean is measured only where an instrument goes down: ${fmtInt(nArgo)} Argo profiles in the test period, about ${fmt(perDay, perDay < 10 ? 1 : 0)} a day, against ${fmtInt(nOcean)} ocean cells. `
              : ""}
            An encoder compresses the day's surface state{use.partial ? ` (${inputNames(use.used)})` : ""} into a satellite embedding
            {m.model?.emb_dim ? ` of ${m.model.emb_dim} features` : ""}
            {maskPct != null ? ` (first trained without subsurface data, by filling in ${maskPct}\u00A0% hidden surface)` : ""}; a decoder expands it into the
            temperature anomaly at each depth.
          </>
        }
      >
        {date && <MethodStrip date={date} depthIndex={depthIndex} />}
      </Chapter>

      <Chapter num="04" id="ch-limits" title="What this does not show">
        <ul className="limits">
          <li>
            <strong>The target is a reanalysis.</strong> GLORYS is a model constrained by observations, Argo included. The reconstruction can
            be at best as good as GLORYS.
          </li>
          <li>
            <strong>One day at a time.</strong> The network sees a single daily snapshot and has no memory of the days before.
          </li>
          <li>
            <strong>Deep water barely varies.</strong> Below a few hundred metres climatology is hard to beat and skill scores are noisy.
          </li>
          <li>
            <strong>Sub-grid detail is averaged out.</strong> Fine satellite products are block-averaged to {grid.resolution}°.
          </li>
        </ul>
        {run.description && (
          <p className="caption story__about">
            <span className="overline">This run</span> {prettyText(run.description)}
          </p>
        )}
        <nav className="nextviews" aria-label="Continue">
          <ViewLink view="explore" className="nextviews__item" patch={date ? { date, depth: depths[depthIndex] } : undefined}>
            <span className="overline">Explorer</span>
            <span className="nextviews__title">Any day, depth, point and method</span>
            <span className="caption">Linked maps, profiles, sections and time–depth plots.</span>
          </ViewLink>
          <ViewLink view="validation" className="nextviews__item">
            <span className="overline">Validation</span>
            <span className="nextviews__title">Every metric, by depth and basin</span>
            <span className="caption">Error maps, daily error, Argo float profiles.</span>
          </ViewLink>
          <ViewLink view="representation" className="nextviews__item">
            <span className="overline">Representation</span>
            <span className="nextviews__title">Inside the embedding</span>
            <span className="caption">What the encoder keeps of the surface, and what it groups together.</span>
          </ViewLink>
          <ViewLink view="experiments" className="nextviews__item">
            <span className="overline">Experiments</span>
            <span className="nextviews__title">Runs, ablations, the product</span>
            <span className="caption">Training curves, the comparison table, NetCDF files and the report.</span>
          </ViewLink>
        </nav>
      </Chapter>
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

// ---- headline numbers and the skill by depth ---------------------------------------------------------

function SkillBlock({ metrics, whole, year }: { metrics: MetricsResponse; /** the unscoped payload (every test year) */ whole: MetricsResponse; year: string | null }) {
  const { depths, styleOf, labelOf, caveats, detail } = useRunContext();
  const years = yearsOf(whole);
  const stability = useMemo(() => yearStability(whole), [whole]);
  const labels = useMemo(() => Object.fromEntries(detail.basins.map((b) => [b.key, b.label])), [detail.basins]);
  const mlp = useMemo(() => mlpSentence(metrics, labels), [metrics, labels]);
  const period = year ? `test days of ${year}` : "test period";
  const head = useMemo(() => headline(metrics), [metrics]);
  const corr = useMemo(() => correlationSentence(metrics), [metrics]);
  const skill = useMemo(() => depthSkill(metrics), [metrics]);
  const ablations = useMemo(() => ablationFindings(metrics), [metrics]);
  const methods = useMemo(() => methodList(metrics), [metrics]);
  const keys = methods.map((m) => m.key);
  const clim = metrics.pooled.climatology?.rmse ?? null;
  const candidates = methods.filter((m) => metrics.pooled[m.key]?.rmse != null);
  const best = candidates.length ? candidates.reduce((a, b) => ((metrics.pooled[a.key].rmse as number) <= (metrics.pooled[b.key].rmse as number) ? a : b)).key : null;
  const legend = keys.map((k) => ({ key: k, label: labelOf(k), style: styleOf(k) }));

  return (
    <div className="skill">
      {years.length >= 2 && (
        <div className="toolbar skill__scope">
          <YearSwitch years={years} />
          <span className="caption">applies to the tables, the depth curves, the basins and the Argo comparison below</span>
        </div>
      )}
      <div className="skill__grid">
        <Panel title={`Pooled over ${head.range}`} subtitle={`against GLORYS, ${period} · the depth range where the temperature varies most`}>
          <p className={`skill__headline ${head.beatsBaselines ? "" : "is-negative"}`}>{head.sentence}</p>
          <div className="tablewrap gap-top-sm">
            <table className="table table--snug">
              <caption className="visually-hidden">Pooled scores over {head.range} by method</caption>
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col" className="num right">
                    RMSE (°C)
                  </th>
                  {!year &&
                    years.map((y) => (
                      <th key={y} scope="col" className="num right" title={`Pooled RMSE over the test days of ${y}`}>
                        {y}
                      </th>
                    ))}
                  <th scope="col" className="num right" title="Pooled RMSE relative to the climatology's">
                    vs climatology
                  </th>
                  <th scope="col" className="num right" title="Correlation after removing the climatology from both sides">
                    Anom. corr.
                  </th>
                  <th scope="col" className="num right" title="1 − MSE / MSE of the climatology">
                    Skill
                  </th>
                </tr>
              </thead>
              <tbody>
                {methods.map((mm) => {
                  const b = metrics.pooled[mm.key];
                  const c = mm.key === "climatology" ? null : compare(b?.rmse, clim);
                  return (
                    <tr key={mm.key} className={mm.key === "model" ? "is-lead" : undefined}>
                      <th scope="row">
                        <span className="methodcell">
                          <Swatch style={styleOf(mm.key)} width={26} />
                          {mm.short}
                        </span>
                      </th>
                      <td className="num right">{mm.key === best ? <span className="best">{fmt(b?.rmse)}</span> : fmt(b?.rmse)}</td>
                      {!year &&
                        years.map((y) => (
                          <td key={y} className="num right table__sub">
                            {fmt(whole.per_year?.[y]?.pooled?.[mm.key]?.rmse)}
                          </td>
                        ))}
                      <td className="num right">{mm.key === "climatology" ? "the reference" : c ? (c.direction === "same" ? "about the same" : `${fmtSigned(-c.pct, 0)}\u00A0%`) : fmt(null)}</td>
                      <td className="num right">{fmt(b?.corr_anom)}</td>
                      <td className="num right">{fmt(b?.skill_vs_clim)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!year && stability && <p className="caption gap-top-sm skill__depths">{stability}</p>}
          {skill.sentence && <p className="caption gap-top-sm skill__depths">{skill.sentence}</p>}
          {corr && <p className="caption gap-top-sm">{corr}</p>}
        </Panel>
        <Panel title="RMSE by depth" subtitle={`against GLORYS, ${period} · lower is better`}>
          <Legend items={legend} band={`pooled range ${head.range}`} />
          <MetricProfile perDepth={metrics.per_depth} depths={depths} metric="rmse" methods={keys} styleOf={styleOf} labelOf={labelOf} pooledRange={metrics.pooled_range_m} reference="GLORYS" height={360} />
        </Panel>
        <Panel title="Skill against climatology by depth" subtitle="1 − MSE / MSE of climatology · right of the zero line beats climatology">
          <Legend items={legend.filter((l) => l.key !== "climatology")} band={`pooled range ${head.range}`} />
          <MetricProfile
            perDepth={metrics.per_depth}
            depths={depths}
            metric="skill_vs_clim"
            methods={keys.filter((k) => k !== "climatology")}
            styleOf={styleOf}
            labelOf={labelOf}
            pooledRange={metrics.pooled_range_m}
            reference="GLORYS"
            height={360}
          />
        </Panel>
      </div>

      <div className="skill__notes">
        {mlp && (
          <Note kind="honesty" title="What the spatial model adds">
            {mlp}
          </Note>
        )}
        {ablations.map((a) => (
          <Note key={a.key} kind="honesty" title="Ablation: does pretraining help?">
            {a.sentence}
          </Note>
        ))}
        {(caveats.synthetic || caveats.shortTraining) && (
          <Note kind="caveat" title={caveats.synthetic ? "Synthetic data" : "Short training period"}>
            {caveats.synthetic
              ? "These numbers come from the analytic test ocean. They show that the pipeline and the evaluation work; they say nothing about the real ocean."
              : caveats.shortTrainingNote}
          </Note>
        )}
      </div>
      <p className="chapter__more">
        <ViewLink view="validation" className="btn btn--quiet">
          Every metric, map and day <Icon name="arrow" />
        </ViewLink>
      </p>
    </div>
  );
}

// ---- basins and Argo ---------------------------------------------------------------------------------

function BasinPanel({ metrics, year }: { metrics: MetricsResponse; year: string | null }) {
  const { detail, styleOf } = useRunContext();
  const labels = useMemo(() => Object.fromEntries(detail.basins.map((b) => [b.key, b.label])), [detail.basins]);
  const basins = useMemo(() => basinContrast(metrics, labels), [metrics, labels]);
  const sentence = useMemo(() => basinSentence(metrics, labels), [metrics, labels]);
  const methods = useMemo(() => methodList(metrics), [metrics]);
  const range = rangeText(metrics.pooled_range_m);
  if (basins.length === 0) return <Empty title="No basin breakdown in this run's metrics" height={160} />;
  const max = Math.max(...basins.flatMap((b) => methods.map((mm) => metrics.per_basin[b.key]?.pooled[mm.key]?.rmse ?? 0)));
  return (
    <Panel title="Basin by basin" subtitle={`pooled RMSE over ${range} against GLORYS${year ? `, ${year}` : ""}, with the skill against climatology · one axis for all basins`}>
      {sentence && <p className="caption evidence__lead">{sentence}</p>}
      <div className="basins">
        {basins.map((b) => (
          <div key={b.key}>
            <p className="multiples__title">{b.label}</p>
            <MethodBars
              inline
              label={`Pooled RMSE over ${range} in the ${b.label} by method`}
              unit="°C"
              max={max}
              rows={methods.map((mm) => {
                const block = metrics.per_basin[b.key]?.pooled[mm.key];
                return {
                  key: mm.key,
                  label: mm.short,
                  value: block?.rmse ?? null,
                  style: styleOf(mm.key),
                  emphasis: mm.key === "model",
                  note: mm.key === "climatology" ? undefined : `skill ${fmt(block?.skill_vs_clim)}`,
                };
              })}
            />
          </div>
        ))}
      </div>
    </Panel>
  );
}

function ArgoPanel({ argo, year }: { argo: MetricsResponse; year: string | null }) {
  const { styleOf } = useRunContext();
  const methods = useMemo(() => methodList(argo), [argo]);
  const pooled = methods.some((mm) => argo.pooled?.[mm.key]?.rmse != null);
  const blocks = pooled ? argo.pooled : argo.overall;
  const sentence = useMemo(() => argoSentence(argo), [argo]);
  const where = pooled ? `over ${rangeText(argo.pooled_range_m)}` : "over all depths";
  return (
    <Panel title="Against Argo float profiles" subtitle={`RMSE ${where}${year ? `, ${year}` : ""}, same matchups for every method · GLORYS itself is the floor`}>
      {sentence && <p className="caption evidence__lead">{sentence}</p>}
      <MethodBars
        inline
        label={`RMSE against Argo ${where} by method`}
        unit="°C"
        rows={methods.map((mm) => ({
          key: mm.key,
          label: mm.key === "glorys" ? "GLORYS (the floor)" : mm.short,
          value: blocks[mm.key]?.rmse ?? null,
          style: styleOf(mm.key),
          emphasis: mm.key === "model",
          note: blocks[mm.key]?.bias != null ? `bias ${fmtSigned(blocks[mm.key]?.bias)}` : undefined,
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

// ---- how it works -------------------------------------------------------------------------------------

function MethodStrip({ date, depthIndex }: { date: string; depthIndex: number }) {
  const { run, targetDates } = useRunContext();
  // the same day as the evidence, already in memory
  const { day } = useDayVolumes(run.name, date, targetDates.has(date));
  return <MethodDiagram day={day && day.date === date ? day : null} date={date} depthIndex={depthIndex} />;
}
