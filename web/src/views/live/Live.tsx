/**
 * Live: the latest reconstructed day. A same-day reconstruction (nowcast) from near-real-time
 * satellite fields, updated daily; not a forecast. The map of the latest day at a chosen depth is
 * the hero, beside the climatology and the anomaly from it (no reanalysis exists yet for these
 * days). Under it: how fresh each input is, how the reconstruction is checked while it runs
 * (against Argo floats and against the operational analysis, next to the climatology, band by
 * band, including the bands where it does not beat the climatology), what near-real-time inputs
 * do to it, and how days are revised.
 *
 * The view always shows the live run, whatever run the other views show, and never the evaluated
 * runs' skill numbers. Every statement of a result is computed from the payload (lib/live.ts).
 */
import { useMemo } from "react";
import { useLive } from "@/api/queries";
import type { LiveBand } from "@/lib/live";
import type { LiveInput, LiveResponse, LiveVerification } from "@/api/types";
import { useDayVolumes, type DayVolumes } from "@/api/volumes";
import { Legend, Swatch } from "@/components/charts/marks";
import { XYChart, type ChartSeries } from "@/components/charts/XYChart";
import { DepthRail } from "@/components/controls/DepthRail";
import { SelectionBar } from "@/components/controls/SelectionBar";
import { MapFigure } from "@/components/map/MapFigure";
import { ZoomControls } from "@/components/map/ZoomControls";
import { Empty, ErrorState, Loading, Note, Panel, QueryState, Segmented } from "@/components/ui/primitives";
import { fmtDate, fromUtcMs, toUtcMs } from "@/lib/dates";
import { fmt, fmtDepth, fmtInt, fmtSigned, prettyText } from "@/lib/format";
import { bandKeys, bandLabel, inputShiftSentence, pendingSentences, profilesPerDay, revisionPolicy, revisionSentence, rollingBands, rollingDays, verificationSentence } from "@/lib/live";
import { climatologyName, climatologyText } from "@/lib/narrative";
import { symmetricLimit } from "@/lib/stats";
import type { LinkedView } from "@/state/linkedView";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";
import { divergingBar, levelFields, levelMeans, raster, tempBar } from "../explorer/fields";
import { ProfilePanel } from "../explorer/ProfilePanel";

/** "4 Oct 2026, 09:06 UTC" */
function fmtStamp(ts: string | null | undefined): string {
  if (!ts) return "–";
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(ts);
  return m ? `${fmtDate(m[1])}, ${m[2]}\u00A0UTC` : ts;
}

export default function Live() {
  const ctx = useRunContext();
  const { run, detail, dates, date, depths, depthIndex, geom } = ctx;
  const live = useLive(run.name, !!run.live);
  const link = useLinkedView(geom);
  const { day, loading, error } = useDayVolumes(run.name, date, false);
  const lastDay = live.data?.last_day ?? run.live_last_day ?? dates[dates.length - 1] ?? null;
  const names = useMemo(() => new Map(detail.products.map((p) => [p.variable, p.long_name])), [detail.products]);
  const nameOf = (product: string) => {
    const n = names.get(product) ?? product;
    return /^[A-Z][a-z]/.test(n) ? n[0].toLowerCase() + n.slice(1) : n;
  };

  // the app shows this view only for a live run; while it redirects, nothing of another run is drawn
  if (!run.live) return <Loading height={420} label="Opening the live reconstruction" />;

  if (dates.length === 0 || !date) {
    return (
      <Empty title="The live run has no reconstructed day yet" height={280}>
        “oceanembed live update” fetches the newest satellite fields and reconstructs the days that have every input.
      </Empty>
    );
  }
  const isLatest = date === lastDay;

  return (
    <div className="live">
      <SelectionBar estimate={false} />
      <header className="viewhead viewhead--tight">
        <div className="viewhead__row">
          <div>
            <p className="overline">
              Live <span className="chip chip--lead">nowcast · not a forecast</span>
            </p>
            <h1 className="h1">Latest reconstruction: {fmtDate(lastDay)}</h1>
          </div>
          <ZoomControls link={link} />
        </div>
        <p className="caption">
          A same-day reconstruction from near-real-time satellite fields ({detail.products
            .filter((p) => p.role === "input" && p.used_by_model !== false)
            .map((p) => nameOf(p.variable))
            .join(" and ")}
          ), updated daily. It says what the ocean below the surface was on that day; it does not predict a later one. Last updated{" "}
          <span className="num">{fmtStamp(live.data?.last_update)}</span>.
        </p>
      </header>

      {live.data && <WindowStrip live={live.data} nameOf={nameOf} />}

      {error && !day ? (
        <ErrorState error={new Error(error)} what="The reconstruction of this day" height={320} />
      ) : !day ? (
        <Loading height={420} label="Loading the reconstruction" />
      ) : (
        <div className="explorer__main">
          <div className="explorer__stage">
            <div className="explorer__maps">
              <LiveMaps day={day} link={link} pending={loading} latest={isLatest} />
            </div>
          </div>
          <ProfilePanel day={day} />
        </div>
      )}
      <p className="caption live__noref">
        No reanalysis exists yet for these days, so the reconstruction is shown beside the {climatologyName(run.n_harmonic_terms).slice(4)} (
        {climatologyText(run.n_harmonic_terms)} of each cell at {fmtDepth(depths[depthIndex])}) instead of a reference. How far to trust it is measured below, as the
        days come in.
      </p>

      <QueryState query={live} what="The state of the live reconstruction" height={320}>
        {(l) => (
          <>
            <section className="section" aria-labelledby="h-fresh">
              <div className="section__head">
                <h2 id="h-fresh" className="h2">
                  How fresh the inputs are
                </h2>
                <p className="caption">A day is reconstructed only when every input the model reads has been published for it.</p>
              </div>
              <Freshness live={l} nameOf={nameOf} />
            </section>

            <section className="section" aria-labelledby="h-check">
              <div className="section__head">
                <h2 id="h-check" className="h2">
                  How it is checked
                </h2>
                <p className="caption">
                  Each new day is scored as measurements arrive, next to the {climatologyName(run.n_harmonic_terms).slice(4)}. These are running numbers of the
                  latest weeks, not the evaluation of the model.
                </p>
              </div>
              <Verification live={l} />
            </section>

            <section className="section" aria-labelledby="h-rev">
              <div className="section__head">
                <h2 id="h-rev" className="h2">
                  Revisions
                </h2>
                <p className="caption">Near-real-time maps are corrected by their producers for a few days after they first appear.</p>
              </div>
              <Revisions live={l} />
            </section>
          </>
        )}
      </QueryState>
    </div>
  );
}

// ---- the window: every day, reconstructed or waiting ----------------------------------------------------

function WindowStrip({ live, nameOf }: { live: LiveResponse; nameOf: (product: string) => string }) {
  const { dates, date, setDateIndex } = useRunContext();
  const pending = pendingSentences(live, nameOf);
  const days = live.window_days ?? [];
  const waiting = (live.pending ?? []).filter((d) => !days.some((w) => w.date === d));
  const n = days.filter((d) => d.reconstructed).length;
  return (
    <div className="window" role="group" aria-label="Days of the live window">
      <div className="window__head">
        <span className="overline">
          Window · {fmtInt(n)} reconstructed day{n === 1 ? "" : "s"}
          {live.window?.start && live.window.end ? `, ${fmtDate(live.window.start)} – ${fmtDate(live.window.end)}` : ""}
        </span>
        <span className="window__key" aria-hidden="true">
          <i className="window__sw is-done" /> reconstructed <i className="window__sw is-wait" /> waiting for an input
        </span>
      </div>
      <div className="window__days" style={{ gridTemplateColumns: `repeat(${Math.max(1, days.length + waiting.length)}, minmax(0, 1fr))` }}>
        {days.map((d) => {
          const i = dates.indexOf(d.date);
          const state = d.reconstructed && i >= 0 ? "is-done" : "is-wait";
          const missing = Object.entries(d.inputs ?? {})
            .filter(([, ok]) => !ok)
            .map(([k]) => nameOf(k));
          return (
            <button
              key={d.date}
              type="button"
              className={`window__day ${state}`}
              aria-pressed={d.date === date}
              disabled={i < 0}
              title={`${fmtDate(d.date)}${d.reconstructed ? "" : missing.length ? ` · waiting for ${missing.join(" and ")}` : " · not reconstructed yet"}`}
              aria-label={`${fmtDate(d.date)}${d.reconstructed ? "" : ", waiting for an input"}`}
              onClick={() => i >= 0 && setDateIndex(i)}
            />
          );
        })}
        {waiting.map((d) => (
          <span key={d} className="window__day is-wait" role="img" title={`${fmtDate(d)} · waiting for an input`} aria-label={`${fmtDate(d)}, waiting for an input`} />
        ))}
      </div>
      {pending.length > 0 && <p className="caption window__pending">{pending.join(" ")}</p>}
    </div>
  );
}

// ---- the maps: reconstruction, climatology, anomaly ---------------------------------------------------

function LiveMaps({ day, link, pending, latest }: { day: DayVolumes; link: LinkedView; pending: boolean; latest: boolean }) {
  const { run, depths, depthIndex, setDepthIndex, geom, mask, point, setPoint, pooledRange } = useRunContext();
  const depthText = fmtDepth(depths[depthIndex]);
  const temp = useMemo(() => levelFields(day, depthIndex, "temp"), [day, depthIndex]);
  const anom = useMemo(() => levelFields(day, depthIndex, "anom"), [day, depthIndex]);
  const clim = useMemo(() => {
    const size = day.climatology.nx * day.climatology.ny;
    return day.climatology.data.subarray(depthIndex * size, (depthIndex + 1) * size);
  }, [day, depthIndex]);
  const hint = day.prediction.rangePerDepth?.[depthIndex] ?? day.prediction.range;
  const limit = useMemo(() => symmetricLimit(anom.recon, 0.99) || 1, [anom]);
  const meanProfile = useMemo(() => levelMeans(day.prediction), [day]);
  const common = {
    geom,
    link,
    surfaceMask: mask?.levels[0] ?? null,
    levelMask: mask?.levels[depthIndex] ?? null,
    coast: mask?.coast ?? null,
    marker: point,
    onPick: setPoint,
    wheel: "plain" as const,
    pending,
  } as const;
  const tBar = tempBar(hint.vmin, hint.vmax);
  const rs = temp.reconStats;
  const as = anom.reconStats;
  const climName = climatologyName(run.n_harmonic_terms).slice(4);
  return (
    <div className="submaps">
      <div className="submaps__rail">
        <DepthRail depths={depths} index={depthIndex} onChange={setDepthIndex} profile={meanProfile} band={pooledRange} />
      </div>
      <div className="submaps__cell submaps__cell--main">
        <MapFigure
          {...common}
          size="lg"
          title={`Reconstruction · ${fmtDate(day.date)}`}
          subtitle={`${latest ? "the latest day · " : ""}OceanEmbed, from that day's satellite fields · ${depthText}`}
          raster={raster(temp.recon, day.prediction, hint.vmin, hint.vmax, "thermal")}
          colorbar={tBar}
          stats={`mean ${fmt(rs.mean)} · min ${fmt(rs.min, 1)} · max ${fmt(rs.max, 1)} °C`}
          ariaLabel={`Reconstructed temperature at ${depthText} on ${day.date}`}
        />
      </div>
      <div className="submaps__cell submaps__cell--a">
        <MapFigure
          {...common}
          size="md"
          readout="value"
          title="Anomaly"
          subtitle={`reconstruction − ${climName} · ${depthText}`}
          raster={raster(anom.recon, day.prediction, -limit, limit, "balance")}
          colorbar={divergingBar(limit)}
          stats={`mean ${fmtSigned(as.mean)} · RMS ${fmt(as.rms)} °C`}
          ariaLabel={`Reconstruction minus climatology at ${depthText} on ${day.date}`}
        />
      </div>
      <div className="submaps__cell submaps__cell--b">
        <MapFigure
          {...common}
          size="md"
          readout="value"
          title={`${climName[0].toUpperCase()}${climName.slice(1)}`}
          subtitle={`for that day of the year · ${depthText}`}
          raster={raster(clim as Float32Array, day.prediction, hint.vmin, hint.vmax, "thermal")}
          colorbar={tBar}
          stats="same colour scale as the reconstruction"
          ariaLabel={`Climatology at ${depthText} for ${day.date}`}
        />
      </div>
    </div>
  );
}

// ---- input freshness ---------------------------------------------------------------------------------

function Freshness({ live, nameOf }: { live: LiveResponse; nameOf: (product: string) => string }) {
  const pending = pendingSentences(live, nameOf);
  const age = (i: LiveInput) => (i.age_days == null ? "–" : `${fmtInt(i.age_days)} day${i.age_days === 1 ? "" : "s"}`);
  const cap = (s: string) => `${s[0].toUpperCase()}${s.slice(1)}`;
  return (
    <Panel title="Inputs of the latest day" subtitle={`as of the last update, ${fmtStamp(live.last_update)}`}>
      {live.inputs.length === 0 ? (
        <Empty title="The live state lists no inputs" height={100} />
      ) : (
        <div className="tablewrap">
          <table className="table">
            <caption className="visually-hidden">Near-real-time inputs: dataset, the day of the data used, its age, and how far the catalogue reaches</caption>
            <thead>
              <tr>
                <th scope="col">Input</th>
                <th scope="col">Dataset</th>
                <th scope="col" className="num right">
                  Data used
                </th>
                <th scope="col" className="num right" title="Days between the last update and the day of the data used">
                  Age
                </th>
                <th scope="col" className="num right" title="The newest day the catalogue had published at the last update">
                  Published through
                </th>
                <th scope="col">Latest day</th>
              </tr>
            </thead>
            <tbody>
              {live.inputs.map((i) => (
                <tr key={i.product}>
                  <th scope="row">{cap(nameOf(i.product))}</th>
                  <td>
                    <span className="mono ids ids--inline">{i.dataset}</span>
                    {i.version && i.version !== "default" ? <span className="chip">{i.version}</span> : null}
                  </td>
                  <td className="num right">{fmtDate(i.latest_data_date)}</td>
                  <td className="num right">{age(i)}</td>
                  <td className="num right">{fmtDate(i.last)}</td>
                  <td>{i.available ? "available" : <strong>missing</strong>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="caption gap-top-sm skill__depths">
        {pending.length > 0 ? pending.join(" ") : "No day is waiting for an input."}
      </p>
    </Panel>
  );
}

// ---- running verification ------------------------------------------------------------------------------

function BandTable({ title, note, bands, counts }: { title: string; note: string; bands: LiveBand[]; counts: string }) {
  const { styleOf } = useRunContext();
  return (
    <Panel title={title} subtitle={note}>
      <div className="tablewrap">
        <table className="table table--snug">
          <caption className="visually-hidden">{title}: rolling error of the reconstruction and of climatology by depth band</caption>
          <thead>
            <tr>
              <th scope="col">Depth band</th>
              <th scope="col" className="num right">
                <span className="methodcell methodcell--head">
                  <Swatch style={styleOf("model")} width={22} />
                  OceanEmbed
                </span>
              </th>
              <th scope="col" className="num right">
                <span className="methodcell methodcell--head">
                  <Swatch style={styleOf("climatology")} width={22} />
                  Climatology
                </span>
              </th>
              <th scope="col" className="right">
                vs climatology
              </th>
              <th scope="col" className="num right" title="Mean of reconstruction minus reference">
                Bias
              </th>
              <th scope="col" className="num right" title="Values compared in the rolling window">
                n
              </th>
            </tr>
          </thead>
          <tbody>
            {bands.map((b) => {
              const c = b.verdict;
              const cls = !c ? "" : c.direction === "lower" ? "" : "is-notbetter";
              return (
                <tr key={b.key} className={cls}>
                  <th scope="row">{b.label}</th>
                  <td className="num right">{fmt(b.score?.model_rmse)}</td>
                  <td className="num right">{fmt(b.score?.clim_rmse)}</td>
                  <td className="right">
                    {!c ? fmt(null) : c.direction === "same" ? <strong>about the same</strong> : c.direction === "lower" ? `${fmt(c.pct, 0)}\u00A0% lower` : <strong>{fmt(-c.pct, 0)}&nbsp;% higher</strong>}
                  </td>
                  <td className="num right">{fmtSigned(b.score?.model_bias)}</td>
                  <td className="num right">{fmtInt(b.score?.n)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="caption gap-top-sm">RMSE in °C · {counts}</p>
    </Panel>
  );
}

function RollingChart({ v, bandKey }: { v: LiveVerification; bandKey: string }) {
  const { styleOf, date, dates, setDateIndex } = useRunContext();
  const charts = (
    [
      ["argo", "against Argo floats", v.argo],
      ["analysis", "against the operational analysis", v.analysis],
    ] as const
  ).filter(([, , ref]) => (ref?.daily?.length ?? 0) > 1);
  if (charts.length === 0) return null;
  const all = charts.flatMap(([, , ref]) => (ref?.daily ?? []).map((d) => toUtcMs(d.date)));
  const x = { label: "", domain: [Math.min(...all), Math.max(...all)] as [number, number], scale: "time" as const };
  const label = bandLabel(v, bandKey);
  const top = Math.max(
    0,
    ...charts.flatMap(([, , ref]) => (ref?.daily ?? []).flatMap((d) => [d.bands?.[bandKey]?.rolling?.model_rmse ?? 0, d.bands?.[bandKey]?.rolling?.clim_rmse ?? 0])),
  );
  const series = (ref: NonNullable<LiveVerification["argo"]>): ChartSeries[] =>
    (
      [
        ["climatology", "Climatology", "clim_rmse"],
        ["model", "OceanEmbed", "model_rmse"],
      ] as const
    ).map(([key, name, field]) => ({
      key,
      label: name,
      style: styleOf(key),
      markers: false,
      points: (ref.daily ?? []).map((d) => ({ x: toUtcMs(d.date), y: d.bands?.[bandKey]?.rolling?.[field] ?? null })),
    }));
  return (
    <div className="twocol">
      {charts.map(([key, where, ref]) => (
        <div key={key} className="multiples__cell">
          <p className="multiples__title">
            Rolling RMSE at {label}, {where}
          </p>
          <XYChart
            series={series(ref!)}
            x={x}
            y={{ label: "RMSE (°C)", domain: [0, top > 0 ? top * 1.1 : 1] }}
            hover="x"
            height={200}
            reference={date ? { axis: "x", value: toUtcMs(date) } : null}
            onPick={(ms) => {
              const i = dates.indexOf(fromUtcMs(ms));
              if (i >= 0) setDateIndex(i);
            }}
            hoverTitle={(ms) => fmtDate(fromUtcMs(ms))}
            ariaLabel={`Rolling RMSE at ${label} ${where}: reconstruction and climatology by day`}
          />
        </div>
      ))}
    </div>
  );
}

function Verification({ live }: { live: LiveResponse }) {
  const { run, styleOf } = useRunContext();
  const [url, setUrl] = useUrlState();
  const v = live.verification;
  const clim = climatologyName(run.n_harmonic_terms);
  const shift = useMemo(() => inputShiftSentence(live.input_shift), [live.input_shift]);
  const keys = bandKeys(v);
  const bandKey = keys.includes(url.opts.vb) ? url.opts.vb : (keys.find((k) => k.startsWith("50")) ?? keys[0]);

  const shiftNote = (
    <Note kind="honesty" title="Near-real-time against reprocessed inputs" className="gap-top">
      {shift ??
        "The effect of near-real-time inputs on the reconstruction has not been measured yet (“oceanembed live input-shift”). The model was trained and evaluated on reprocessed inputs."}
    </Note>
  );

  if (!v || (!v.argo?.latest && !v.analysis?.latest)) {
    return (
      <>
        <Empty title="No verification yet" height={140}>
          The running check starts with the first update that finds Argo profiles or the operational analysis for a reconstructed day.
        </Empty>
        {shiftNote}
      </>
    );
  }

  const argoBands = rollingBands(v, v.argo);
  const anaBands = rollingBands(v, v.analysis);
  const argoDays = rollingDays(v, v.argo);
  const anaDays = rollingDays(v, v.analysis);
  const perDay = profilesPerDay(v.argo, argoDays);
  const argoText = verificationSentence(v, "argo", clim);
  const anaText = verificationSentence(v, "analysis", clim);

  return (
    <>
      <div className="twocol">
        {argoBands.length > 0 ? (
          <div className="rightcol">
            <BandTable
              title="Against Argo floats"
              note={`measurements · error over the last ${argoDays ?? "–"} days`}
              bands={argoBands}
              counts={`n is profile–depth matchups${perDay ? ` · ${perDay.min === perDay.max ? fmtInt(perDay.min) : `${fmtInt(perDay.min)}–${fmtInt(perDay.max)}`} profiles a day` : ""}${
                v.argo?.n_profiles != null ? ` · ${fmtInt(v.argo.n_profiles)} profiles in the window` : ""
              }`}
            />
            {argoText && <p className="caption skill__depths">{argoText}</p>}
            {v.argo?.note && <p className="caption">{prettyText(v.argo.note[0].toUpperCase() + v.argo.note.slice(1))}</p>}
          </div>
        ) : (
          <Empty title="No Argo profiles for the window yet" height={140}>
            Near-real-time profiles arrive with a delay.
          </Empty>
        )}
        {anaBands.length > 0 ? (
          <div className="rightcol">
            <BandTable
              title="Against the operational analysis"
              note={`a model analysis, not an observation · error over the last ${anaDays ?? "–"} days`}
              bands={anaBands}
              counts="n is grid cells × levels × days"
            />
            {anaText && <p className="caption skill__depths">{anaText}</p>}
            {v.analysis?.note && <p className="caption">This reference is {prettyText(v.analysis.note)}.</p>}
          </div>
        ) : (
          <Empty title="No operational analysis for the window yet" height={140} />
        )}
      </div>

      <Panel
        className="gap-top"
        title="Day by day"
        subtitle={`rolling ${v.rolling_days ?? argoDays ?? ""}-day error ending on each day · lower is better · click a day to show it on the map`}
        actions={
          keys.length > 1 && (
            <Segmented
              label="Depth band of the rolling error"
              size="sm"
              value={bandKey}
              onChange={(k) => setUrl({ opts: { vb: k === (keys.find((x) => x.startsWith("50")) ?? keys[0]) ? null : k } })}
              options={keys.map((k) => ({ value: k, label: bandLabel(v, k) }))}
            />
          )
        }
      >
        <Legend
          items={[
            { key: "model", label: "OceanEmbed", style: styleOf("model") },
            { key: "climatology", label: "Climatology", style: styleOf("climatology") },
          ]}
        />
        <RollingChart v={v} bandKey={bandKey} />
      </Panel>
      {shiftNote}
    </>
  );
}

// ---- revisions -----------------------------------------------------------------------------------------

function Revisions({ live }: { live: LiveResponse }) {
  const policy = revisionPolicy(live.revision);
  const stats = revisionSentence(live.revision);
  const rows = live.revision?.by_age ?? [];
  return (
    <Panel title="How days are revised" subtitle={policy ?? "the revision policy is not recorded for this run"}>
      <p className="caption skill__depths">{stats ?? "No revision statistics yet."}</p>
      {rows.length > 0 && (
        <div className="tablewrap gap-top-sm">
          <table className="table table--snug">
            <caption className="visually-hidden">Re-checks by the age of the day when it was checked again</caption>
            <thead>
              <tr>
                <th scope="col">Age of the day when re-checked</th>
                {rows.map((r) => (
                  <th key={r.age_days} scope="col" className="num right">
                    {r.age_days} d
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Checks</th>
                {rows.map((r) => (
                  <td key={r.age_days} className="num right">
                    {fmtInt(r.n_checks)}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Changed since first published</th>
                {rows.map((r) => (
                  <td key={r.age_days} className="num right">
                    {fmtInt(r.n_changed_since_first)}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Largest change, RMSE at 50–200 m (°C)</th>
                {rows.map((r) => (
                  <td key={r.age_days} className="num right">
                    {fmt(r.recon_rmse_50_200_max)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
