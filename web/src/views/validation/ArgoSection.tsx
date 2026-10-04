/**
 * Validation against Argo float profiles. The independence statement comes before any number:
 * Argo is independent of the model's inputs, but GLORYS (the training target) assimilates Argo.
 */
import { useId, useMemo } from "react";
import { ARGO_MAP_LIMIT, useArgoMetrics, useArgoProfile, useArgoProfiles, useMatchups, type ArgoProfileQuery } from "@/api/queries";
import type { ArgoProfileDetail, ArgoProfileSummary, MatchupsResponse, MetricsResponse } from "@/api/types";
import { Legend } from "@/components/charts/marks";
import { MetricProfile } from "@/components/charts/MetricProfile";
import { MetricsTable } from "@/components/charts/MetricsTable";
import { DENSITY_CMAP, DEPTH_CMAP, ScatterDensity } from "@/components/charts/ScatterDensity";
import { XYChart, type ChartSeries } from "@/components/charts/XYChart";
import { MapFigure } from "@/components/map/MapFigure";
import { ZoomControls } from "@/components/map/ZoomControls";
import { Empty, ErrorState, Icon, IconButton, Loading, Note, Panel, QueryState, Segmented, Select } from "@/components/ui/primitives";
import { rgbCss } from "@/lib/colormaps";
import { fmtDate, isIsoDate } from "@/lib/dates";
import { fmt, fmtDepth, fmtInt, fmtLatLon, fmtSigned, prettyText } from "@/lib/format";
import { metricMeta } from "@/lib/metricMeta";
import { canonicalMethod, sortMethods } from "@/lib/methods";
import { YearSwitch, useYear } from "@/components/controls/YearSwitch";
import { methodList, scopeToYear, yearsOf } from "@/lib/narrative";
import { buildPointLayer, countByCell } from "@/lib/points";
import { padRange } from "@/lib/scales";
import { extent, quantile } from "@/lib/stats";
import { color } from "@/lib/theme";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { useLinkedView } from "@/state/useLinkedView";

const ARGO_METRICS = ["rmse", "bias", "corr_anom", "corr_raw"] as const;

export function ArgoSection() {
  const { run } = useRunContext();
  const metrics = useArgoMetrics(run.name, run.artefacts.metrics_argo);
  if (!run.artefacts.metrics_argo) {
    return (
      <Empty title="This run has not been validated against Argo yet" height={160}>
        Run “oceanembed validate-argo” for this configuration to produce the matchups and metrics.
      </Empty>
    );
  }
  return (
    <QueryState query={metrics} what="The Argo validation" height={280}>
      {(m) => <ArgoBody metrics={m} />}
    </QueryState>
  );
}

function ArgoBody({ metrics: whole }: { metrics: MetricsResponse }) {
  const { run, depths, styleOf, labelOf } = useRunContext();
  const years = yearsOf(whole);
  const year = useYear(years);
  const metrics = useMemo(() => scopeToYear(whole, year), [whole, year]);
  const md = metrics.metadata;
  const methods = useMemo(() => methodList(metrics).map((m) => m.key), [metrics]);
  const legend = methods.map((k) => ({ key: k, label: labelOf(k), style: styleOf(k) }));
  const matchups = useMatchups(run.name, 20000, run.artefacts.argo_matchups);
  const dropped = Object.entries(md.dropped_profiles ?? {}).filter(([, n]) => n > 0);
  const range = metrics.pooled_range_m;
  const rangeText = range?.length >= 2 ? `${fmt(range[0], 0)}–${fmt(range[1], 0)} m` : "pooled range";

  return (
    <>
      <Note kind="honesty" title="How independent is this?">
        {md.independence_note
          ? prettyText(md.independence_note)
          : "Argo profiles are independent of the model's inputs, but GLORYS assimilates Argo and the model is trained on GLORYS."}{" "}
        The GLORYS row and line below are therefore a floor, not a competitor.
        {md.note ? <> {prettyText(md.note)}</> : null}
      </Note>

      {years.length >= 2 && (
        <div className="toolbar gap-top">
          <YearSwitch years={years} />
          <span className="caption">applies to the counts, the tables and the by-depth charts; the profile map has its own date filter</span>
        </div>
      )}
      <dl className="factrow gap-top">
        <div>
          <dt>Profiles used</dt>
          <dd className="num">{fmtInt(md.n_profiles_used)}</dd>
        </div>
        <div>
          <dt>Profile–depth matchups</dt>
          <dd className="num">{fmtInt(md.n_matchups)}</dd>
        </div>
        <div>
          <dt>Loaded for the period</dt>
          <dd className="num">{fmtInt(md.n_profiles_loaded)}</dd>
        </div>
        <div>
          <dt>Dropped</dt>
          <dd className="num">{dropped.length === 0 ? "none" : dropped.map(([k, n]) => `${fmtInt(n)} ${k.replace(/_/g, " ")}`).join(", ")}</dd>
        </div>
        <div className="factrow__wide">
          <dt>Collocation</dt>
          <dd>
            {md.collocation ? prettyText(md.collocation) : "containing cell, same day"}
            {md.interpolation_rule ? `. Vertical interpolation: ${prettyText(md.interpolation_rule)}.` : ""}
          </dd>
        </div>
      </dl>

      <div className="gap-top">{run.artefacts.argo_matchups ? <ProfileBrowser /> : <Empty title="This run has no Argo matchup table" height={120} />}</div>

      <div className="twocol gap-top">
        <QueryState query={matchups} what="The Argo matchups" height={320}>
          {(m) => <MatchupScatter matchups={m} columnFor={(md.column_for_method as Record<string, string> | undefined) ?? {}} methods={methods} />}
        </QueryState>
        <div className="rightcol">
          <Panel title="All depths" subtitle={`every matchup${year ? ` of ${year}` : ""} · bias is method minus Argo`}>
            <MetricsTable blocks={metrics.overall} methods={methods} styleOf={styleOf} labelOf={labelOf} caption="Metrics against Argo over all depths" />
          </Panel>
          <Panel title={`Pooled over ${rangeText}`} subtitle={`matchups in the pooled depth range${year ? `, ${year}` : ""}`}>
            <MetricsTable blocks={metrics.pooled} methods={methods} styleOf={styleOf} labelOf={labelOf} caption={`Metrics against Argo over ${rangeText}`} />
          </Panel>
        </div>
      </div>

      <Panel className="gap-top" title="Against Argo, by depth" subtitle={`same matchups for every method${year ? ` · ${year}` : ""} · the shaded band is the pooled range`}>
        <Legend items={legend} band={`pooled range ${rangeText}`} />
        <div className="multiples multiples--4">
          {ARGO_METRICS.filter((k) => metrics.per_depth[k]).map((k) => (
            <div key={k} className="multiples__cell">
              <p className="multiples__title">{metricMeta(k).label}</p>
              <p className="multiples__sub">{metricMeta(k).explain}</p>
              <MetricProfile
                perDepth={metrics.per_depth}
                depths={depths}
                metric={k}
                methods={methods}
                styleOf={styleOf}
                labelOf={labelOf}
                pooledRange={range}
                reference="Argo"
                height={340}
                compact
              />
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

// ---- profile browser --------------------------------------------------------------------------

/** Rows of the profile table per page. */
export const ARGO_PAGE = 10;

type SortKey = "time" | "worst" | "best";

/** Methods of a per-profile RMSE block. The API repeats the climatology under its legacy key `clim`: keep one. */
export function rmseMethods(rmse: Record<string, number | null> | null | undefined): string[] {
  const keys = Object.keys(rmse ?? {});
  return sortMethods(keys.includes("climatology") ? keys.filter((k) => k !== "clim") : keys);
}

/**
 * Profile by profile, for a run with a few hundred or many thousands of profiles: the server
 * filters (basin, dates), sorts (time or per-profile RMSE) and pages the list; the map draws every
 * match on canvas; one profile at a time is opened against every estimate.
 */
function ProfileBrowser() {
  const { run, geom, mask, detail, estimate, labelOf } = useRunContext();
  const [url, setUrl] = useUrlState();
  const link = useLinkedView(geom);
  const startId = useId();
  const endId = useId();
  const test = run.split?.test;

  const basin = detail.basins.some((b) => b.key === url.opts.ab) ? url.opts.ab : null;
  const start = isIsoDate(url.opts.as) ? url.opts.as : null;
  const end = isIsoDate(url.opts.ae) ? url.opts.ae : null;
  const sort: SortKey = url.opts.asort === "worst" || url.opts.asort === "best" ? url.opts.asort : "time";
  // an explicit page (the pager was used), else the page that contains the opened profile
  const explicitPage = url.opts.ap != null ? Math.max(0, Math.floor(Number(url.opts.ap) || 0)) : null;
  const filters: ArgoProfileQuery = { basin, start, end };
  const key = estimate;

  // every match, for the map (positions and RMSE only: a few hundred bytes per profile)
  const all = useArgoProfiles(run.name, { ...filters, limit: ARGO_MAP_LIMIT });
  // one page of the list, in the chosen order
  const pageQ = useArgoProfiles(run.name, {
    ...filters,
    sort: sort === "time" ? "time" : "rmse",
    order: sort === "worst" ? "desc" : "asc",
    method: key,
    limit: ARGO_PAGE,
    offset: (explicitPage ?? 0) * ARGO_PAGE,
    around: explicitPage == null ? (url.opts.prof ?? null) : null,
  });
  const page = pageQ.data ? Math.floor(pageQ.data.offset / ARGO_PAGE) : (explicitPage ?? 0);

  const list = useMemo<ArgoProfileSummary[]>(() => all.data?.profiles ?? [], [all.data]);
  const vmax = useMemo(() => quantile(list.map((p) => p.rmse[key] ?? NaN), 0.95) || 1, [list, key]);
  const layer = useMemo(
    () =>
      buildPointLayer(
        list.map((p) => p.lat),
        list.map((p) => p.lon),
        list.map((p) => p.rmse[key]),
        { vmin: 0, vmax, cmap: "amp", floor: 0.12 },
        color.ink,
      ),
    [list, key, vmax],
  );
  const cells = useMemo(() => countByCell(layer, geom), [layer, geom]);

  const rows = pageQ.data?.profiles ?? [];
  const selectedId = url.opts.prof ?? rows[0]?.profile_id ?? list[0]?.profile_id ?? null;
  const selectedIndex = useMemo(() => (selectedId ? list.findIndex((p) => p.profile_id === selectedId) : -1), [list, selectedId]);
  const selected = rows.find((p) => p.profile_id === selectedId) ?? (selectedIndex >= 0 ? list[selectedIndex] : null);
  const detailQ = useArgoProfile(run.name, selectedId);

  // a new filter or order starts at the first page (an explicit page: the list does not jump to the opened profile)
  const set = (opts: Record<string, string | null>) => setUrl({ opts: { ap: "0", ...opts } });
  const nMatch = pageQ.data?.n_profiles ?? all.data?.n_profiles ?? 0;
  const nTotal = pageQ.data?.n_total ?? all.data?.n_total ?? 0;
  const nPages = Math.max(1, Math.ceil(nMatch / ARGO_PAGE));
  const filtered = !!(basin || start || end);
  const methods = rmseMethods(rows[0]?.rmse ?? list[0]?.rmse);
  // with many methods the table keeps its RMSE columns and drops the level count (it is on the opened profile)
  const showLevels = methods.length <= 5;

  const valueAt = (lat: number, lon: number) => {
    if (!(lat >= geom.lat0 && lat <= geom.lat1 && lon >= geom.lon0 && lon <= geom.lon1)) return null;
    const j = Math.min(geom.nLat - 1, Math.floor((lat - geom.lat0) / geom.dLat));
    const i = Math.min(geom.nLon - 1, Math.floor((lon - geom.lon0) / geom.dLon));
    const c = cells.get(j * geom.nLon + i);
    if (!c) return "no profile in this cell";
    return `${c.n} profile${c.n === 1 ? "" : "s"} in this cell${Number.isFinite(c.mean) ? `  ·  mean RMSE ${fmt(c.mean)}\u00A0°C` : ""}`;
  };

  return (
    <Panel
      title="Profile by profile"
      subtitle={`one dot per Argo profile, coloured by the RMSE of ${labelOf(estimate)} over that profile (the estimate of the selection bar) · click a dot or a row to open it`}
      actions={<ZoomControls link={link} />}
    >
      <div className="toolbar argo__filters" role="group" aria-label="Filter and order the Argo profiles">
        {detail.basins.length > 0 && (
          <Segmented
            label="Basin"
            size="sm"
            value={basin ?? "all"}
            onChange={(v) => set({ ab: v === "all" ? null : v })}
            options={[{ value: "all", label: "All basins" }, ...detail.basins.map((b) => ({ value: b.key, label: b.label }))]}
          />
        )}
        <div className="field">
          <label htmlFor={startId} className="field__label">
            From
          </label>
          <input
            id={startId}
            type="date"
            className="dateinput num"
            value={start ?? ""}
            min={test?.start ?? undefined}
            max={end ?? test?.end ?? undefined}
            onChange={(e) => set({ as: isIsoDate(e.target.value) ? e.target.value : null })}
          />
          <label htmlFor={endId} className="field__label">
            to
          </label>
          <input
            id={endId}
            type="date"
            className="dateinput num"
            value={end ?? ""}
            min={start ?? test?.start ?? undefined}
            max={test?.end ?? undefined}
            onChange={(e) => set({ ae: isIsoDate(e.target.value) ? e.target.value : null })}
          />
        </div>
        <Segmented
          label="Order of the list"
          size="sm"
          value={sort}
          onChange={(v) => set({ asort: v === "time" ? null : v })}
          options={[
            { value: "time", label: "By date" },
            { value: "worst", label: "Largest error first", title: `Highest RMSE of ${labelOf(estimate)} first` },
            { value: "best", label: "Smallest first" },
          ]}
        />
        {filtered && (
          <button type="button" className="linkbtn" onClick={() => set({ ab: null, as: null, ae: null })}>
            Clear filters
          </button>
        )}
      </div>

      <div className="argo">
        <div className="argo__left">
          {all.isError ? (
            <ErrorState error={all.error} what="The Argo profile list" height={240} onRetry={() => void all.refetch()} />
          ) : !all.data ? (
            <Loading height={280} label="Loading the Argo profiles" />
          ) : (
            <MapFigure
              geom={geom}
              link={link}
              title={`${fmtInt(all.data.n_profiles)} profile${all.data.n_profiles === 1 ? "" : "s"}`}
              subtitle={
                (filtered ? `of ${fmtInt(nTotal)} in the test period` : "test period") +
                (all.data.has_more ? ` · the first ${fmtInt(all.data.n_returned)} are drawn` : "")
              }
              surfaceMask={mask?.levels[0] ?? null}
              coast={mask?.coast ?? null}
              basins={detail.basins}
              points={layer}
              selectedPoint={selectedIndex >= 0 ? selectedIndex : null}
              onPickPoint={(i) => setUrl({ opts: { prof: list[i].profile_id, ap: null } })}
              valueAt={valueAt}
              pending={all.isPlaceholderData}
              colorbar={{ cmap: "amp", vmin: 0, vmax, units: "°C", label: `Profile RMSE, ${labelOf(estimate)}`, extend: "max" }}
              ariaLabel={`Positions of ${all.data.n_profiles} Argo profiles coloured by the error of ${labelOf(estimate)}`}
            />
          )}

          {pageQ.isError ? (
            <ErrorState error={pageQ.error} what="This page of profiles" height={160} onRetry={() => void pageQ.refetch()} />
          ) : !pageQ.data ? (
            <Loading height={ARGO_PAGE * 28} label="Loading the profile list" />
          ) : nMatch === 0 ? (
            <Empty title="No Argo profile matches these filters" height={140} />
          ) : (
            <div className={`argo__list ${pageQ.isPlaceholderData ? "is-pending" : ""}`}>
              <div className="tablewrap" tabIndex={0} role="region" aria-label="Argo profiles">
                <table className="table table--dense table--pick">
                  <caption className="visually-hidden">Argo profiles matching the filters, with the RMSE of every method over each profile in °C</caption>
                  <thead>
                    <tr>
                      <th scope="col">Profile</th>
                      <th scope="col">Date</th>
                      <th scope="col">Position</th>
                      {showLevels && (
                        <th scope="col" className="num right" title="Matched depth levels">
                          Levels
                        </th>
                      )}
                      {methods.map((m) => (
                        <th key={m} scope="col" className="num right" title={`RMSE of ${labelOf(m)} over the profile, °C`}>
                          {labelOf(m)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((p) => (
                      <tr key={p.profile_id} className={p.profile_id === selectedId ? "is-selected" : undefined} onClick={() => setUrl({ opts: { prof: p.profile_id } })}>
                        <th scope="row">
                          <button type="button" className="linkbtn mono" aria-pressed={p.profile_id === selectedId} onClick={() => setUrl({ opts: { prof: p.profile_id } })}>
                            {p.profile_id}
                          </button>
                        </th>
                        <td className="num">{fmtDate(p.time)}</td>
                        <td className="num">{fmtLatLon(p.lat, p.lon, 1)}</td>
                        {showLevels && <td className="num right">{p.n_levels}</td>}
                        {methods.map((m) => (
                          <td key={m} className={`num right ${m === key ? "is-key" : ""}`}>
                            {fmt(p.rmse[m])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="pager">
                <span className="num">
                  {fmtInt(page * ARGO_PAGE + 1)}–{fmtInt(page * ARGO_PAGE + rows.length)} of {fmtInt(nMatch)}
                  {filtered ? ` (${fmtInt(nTotal)} in all)` : ""} · RMSE over each profile, °C
                </span>
                <span className="pager__buttons">
                  <IconButton label="First page" onClick={() => setUrl({ opts: { ap: "0" } })} disabled={page === 0}>
                    <Icon name="first" size={14} />
                  </IconButton>
                  <IconButton label="Previous page" onClick={() => setUrl({ opts: { ap: String(Math.max(0, page - 1)) } })} disabled={page === 0}>
                    <Icon name="prev" size={14} />
                  </IconButton>
                  <span className="num pager__page">
                    page {fmtInt(page + 1)} of {fmtInt(nPages)}
                  </span>
                  <IconButton label="Next page" onClick={() => setUrl({ opts: { ap: String(page + 1) } })} disabled={!pageQ.data.has_more}>
                    <Icon name="next" size={14} />
                  </IconButton>
                  <IconButton label="Last page" onClick={() => setUrl({ opts: { ap: String(nPages - 1) } })} disabled={!pageQ.data.has_more}>
                    <Icon name="last" size={14} />
                  </IconButton>
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="argo__right">
          {!selectedId ? (
            <Empty title="No profile selected" height={300}>
              Click a dot on the map or a row of the list.
            </Empty>
          ) : (
            <QueryState query={detailQ} what="This Argo profile" height={420}>
              {(d) => <ProfileChart profile={d} pending={detailQ.isPlaceholderData} rmse={selected?.rmse ?? null} />}
            </QueryState>
          )}
        </div>
      </div>
    </Panel>
  );
}

function ProfileChart({ profile, pending, rmse }: { profile: ArgoProfileDetail; pending: boolean; rmse: Record<string, number | null> | null }) {
  const { depths, styleOf, labelOf, detail } = useRunContext();
  // the API repeats the climatology under its legacy key `clim`: draw it once
  const all = Object.keys(profile.series);
  const keys = sortMethods(all.includes("climatology") ? all.filter((k) => k !== "clim") : all);
  const series: ChartSeries[] = [
    ...keys.map((k) => ({
      key: k,
      label: labelOf(k),
      style: styleOf(k),
      points: profile.depth.map((d, i) => ({ x: profile.series[k][i] ?? null, y: d })),
    })),
    {
      key: "obs",
      label: "Argo observation",
      style: styleOf("obs"),
      points: profile.depth.map((d, i) => ({ x: profile.obs[i] ?? null, y: d })),
    },
  ];
  const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p.x)));
  const basin = detail.basins.find((b) => b.key === profile.basin)?.label;
  const maxDepth = depths[depths.length - 1];
  return (
    <div className={pending ? "is-pending" : ""}>
      <p className="caption num">
        <strong>{profile.profile_id}</strong> · {fmtDate(profile.time)} · {fmtLatLon(profile.lat, profile.lon)}
        {basin ? ` · ${basin}` : ""} · {profile.depth.length} matched levels
      </p>
      <Legend items={[series[series.length - 1], ...series.slice(0, -1)].map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
      <XYChart
        series={series}
        x={{ label: "Temperature (°C)", domain: padRange(lo, hi, 0.06) }}
        y={{ label: "Depth (m)", domain: [0, maxDepth], scale: "depth", ticks: [...depths] }}
        hover="y"
        height={440}
        hoverTitle={(d) => fmtDepth(d)}
        ariaLabel={`Argo profile ${profile.profile_id} against ${keys.map((k) => labelOf(k)).join(", ")}`}
      />
      {rmse && (
        <p className="caption num">
          RMSE over this profile:{" "}
          {rmseMethods(rmse)
            .map((k) => `${labelOf(canonicalMethod(k))} ${fmt(rmse[k])}`)
            .join(" · ")}{" "}
          °C
        </p>
      )}
    </div>
  );
}

// ---- scatter ----------------------------------------------------------------------------------

function MatchupScatter(props: { matchups: MatchupsResponse; columnFor: Record<string, string>; methods: readonly string[] }) {
  const { matchups, columnFor, methods } = props;
  const { labelOf, depths, estimate } = useRunContext();
  const [url, setUrl] = useUrlState();
  const cols = matchups.columns;
  const colOf = (m: string) => columnFor[m] ?? (m === "climatology" ? "clim" : m);
  const usable = methods.filter((m) => Array.isArray(cols[colOf(m)]));
  // the estimate of the selection bar, unless another series (climatology, GLORYS) was chosen here
  const method = usable.includes(url.opts.sm) ? url.opts.sm : usable.includes(estimate) ? estimate : (usable[0] ?? "model");
  const mode = url.opts.sc === "depth" ? "depth" : "density";
  const estCol = colOf(method);
  const est = useMemo(() => (cols[estCol] ?? []) as (number | null)[], [cols, estCol]);
  const obs = cols.obs;

  const stats = useMemo(() => {
    let n = 0;
    let se = 0;
    let sd = 0;
    for (let i = 0; i < obs.length; i++) {
      const e = est[i];
      const o = obs[i];
      if (e == null || o == null) continue;
      n++;
      se += (e - o) ** 2;
      sd += e - o;
    }
    return n > 0 ? { n, rmse: Math.sqrt(se / n), bias: sd / n } : null;
  }, [obs, est]);

  const domain = useMemo<[number, number]>(() => {
    const [lo, hi] = extent(obs);
    const all = usable.flatMap((m) => extent((cols[colOf(m)] ?? []) as (number | null)[]));
    const lo2 = Math.min(lo, ...all.filter(Number.isFinite));
    const hi2 = Math.max(hi, ...all.filter(Number.isFinite));
    return [Math.floor(lo2), Math.ceil(hi2)];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [obs, cols]);

  const maxDepth = depths[depths.length - 1];
  return (
    <Panel
      title="Observed against estimated"
      subtitle={`each matchup is one Argo level and the ${labelOf(method)} value in the same cell and day`}
      actions={
        <>
          <Select label="Series" value={method} onChange={(v) => setUrl({ opts: { sm: v === estimate ? null : v } })} options={usable.map((m) => ({ value: m, label: labelOf(m) }))} />
          <Segmented
            label="Colour"
            size="sm"
            value={mode}
            onChange={(v) => setUrl({ opts: { sc: v === "density" ? null : v } })}
            options={[
              { value: "density", label: "Density" },
              { value: "depth", label: "Depth" },
            ]}
          />
        </>
      }
    >
      <div className="scatterwrap">
      <ScatterDensity
        x={obs}
        y={est}
        depth={cols.depth}
        mode={mode}
        maxDepth={maxDepth}
        domain={domain}
        xLabel="Argo observation (°C)"
        yLabel={`${labelOf(method)} (°C)`}
        ariaLabel={`Argo temperature against ${labelOf(method)} for ${matchups.n_returned} matchups`}
      />
      <div className="scatter__foot">
        {mode === "density" ? (
          <div className="scalekey">
            <span className="scalekey__bar" style={{ background: `linear-gradient(to right, ${rgbCss(DENSITY_CMAP, 0.12)}, ${rgbCss(DENSITY_CMAP, 0.55)}, ${rgbCss(DENSITY_CMAP, 1)})` }} />
            <span>few</span>
            <span>many matchups per bin (log scale)</span>
          </div>
        ) : (
          <div className="scalekey">
            <span className="scalekey__bar" style={{ background: `linear-gradient(to right, ${rgbCss(DEPTH_CMAP, 0.08)}, ${rgbCss(DEPTH_CMAP, 0.54)}, ${rgbCss(DEPTH_CMAP, 1)})` }} />
            <span>surface</span>
            <span>
              {fmtDepth(maxDepth)} · mean depth of the bin's matchups (square-root scale)
            </span>
          </div>
        )}
        <p className="caption num">
          {stats ? (
            <>
              n = {fmtInt(stats.n)} · RMSE {fmt(stats.rmse)} °C · bias {fmtSigned(stats.bias)} °C
            </>
          ) : (
            "no matchups"
          )}
          {matchups.downsampled ? ` · an evenly spaced sample of ${fmtInt(matchups.n_total)} matchups is drawn; the tables use all of them` : ""}
        </p>
      </div>
      </div>
    </Panel>
  );
}
