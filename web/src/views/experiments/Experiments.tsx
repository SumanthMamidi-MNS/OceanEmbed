/**
 * Experiments and results: the comparison table of methods and ablations, the same table across
 * runs, the training curves, the data that went in and the product and report that came out.
 */
import { useMemo } from "react";
import { useArgoMetrics, useCompare, useExperiments, useGlorysMetrics } from "@/api/queries";
import type { CompareRun, DataProduct, ExperimentRow, ExperimentsResponse, MetricsResponse, RunSummary } from "@/api/types";
import { Swatch } from "@/components/charts/marks";
import { MethodBars } from "@/components/charts/MethodBars";
import { DataTable, Empty, Note, Panel, QueryState, type Column } from "@/components/ui/primitives";
import { runLabel } from "@/components/shell/Shell";
import { daysInclusive, fmtSpan } from "@/lib/dates";
import { fmt, fmtDepth, fmtInt, fmtSigned, prettyText } from "@/lib/format";
import { ablationKeysOf, isAblation, methodStyle, shortLabel, sortMethods } from "@/lib/methods";
import { inputNames, inputUse } from "@/lib/inputs";
import { ablationFindings, methodList, runCaveats, yearStability, yearsOf } from "@/lib/narrative";
import { useUrlState } from "@/state/router";
import { useRunContext } from "@/state/runContext";
import { ProductSection, ReportSection } from "./ReportSection";
import { TrainingSection } from "./TrainingSection";

/** Readable names for the configuration keys the API reports (unknown keys are shown as they are). */
const CONFIG_LABELS: Record<string, string> = {
  in_channels: "Input channels",
  emb_dim: "Embedding features per cell",
  dim: "Transformer width",
  depth: "Transformer layers",
  heads: "Attention heads",
  mlp_ratio: "MLP ratio",
  arch: "Architecture",
  input_groups: "Input groups used",
  main_init: "Encoder initialisation",
  stem_channels: "Stem channels",
  n_depths: "Output depth levels",
  epochs: "Epochs (maximum)",
  batch_size: "Batch size",
  lr: "Learning rate",
  mask_ratio: "Hidden fraction of the surface",
  block: "Hidden block size (cells)",
  encoder_lr_scale: "Encoder learning-rate scale",
  patience: "Early-stopping patience (epochs)",
  vertical_grad_weight: "Vertical-gradient loss weight",
};

function sortRows(rows: readonly ExperimentRow[]): ExperimentRow[] {
  const order = sortMethods(rows.map((r) => r.method));
  return [...rows].sort((a, b) => order.indexOf(a.method) - order.indexOf(b.method));
}

function gainText(pct: number | null | undefined): string {
  if (pct == null || !Number.isFinite(pct)) return fmt(null);
  if (pct === 0) return "0 %";
  // a positive gain is a lower RMSE
  return `${fmtSigned(-pct, 1)} %`;
}

export default function Experiments() {
  const { run, detail } = useRunContext();
  const exp = useExperiments(run.name, run.artefacts.metrics_glorys);
  const glorys = useGlorysMetrics(run.name, run.artefacts.metrics_glorys);
  const argo = useArgoMetrics(run.name, run.artefacts.metrics_argo);

  return (
    <div className="experiments">
      <header className="viewhead">
        <p className="overline">Experiments · {runLabel(run)}</p>
        <h1 className="h1">Methods, runs and deliverables</h1>
        {run.description && <p className="caption">{prettyText(run.description)}</p>}
        <nav className="subnav" aria-label="Sections of this view">
          <a href="#methods">Methods and ablations</a>
          <a href="#runs">Across runs</a>
          <a href="#training">Training</a>
          <a href="#data">Data and model</a>
          <a href="#product">Product and report</a>
        </nav>
      </header>

      <section id="methods" className="section" aria-labelledby="h-methods">
        <div className="section__head">
          <h2 id="h-methods" className="h2">
            Methods and ablations in this run
          </h2>
          <p className="caption">One row per method, all scored on the same test days and cells. RMSE in °C; lower is better.</p>
        </div>
        {run.artefacts.metrics_glorys ? (
          <QueryState query={exp} what="The experiments table" height={260}>
            {(e) => <MethodsTable exp={e} glorys={glorys.data} argo={argo.data} />}
          </QueryState>
        ) : (
          <Empty title="This run has not been evaluated yet" height={140}>
            Run “oceanembed evaluate” to produce the metrics this table is built from.
          </Empty>
        )}
      </section>

      <section id="runs" className="section" aria-labelledby="h-runs">
        <div className="section__head">
          <h2 id="h-runs" className="h2">
            Across runs
          </h2>
          <p className="caption">
            Runs differ in data, period and training length, so absolute RMSE is not comparable between them; the reduction against each
            run's own climatology is.
          </p>
        </div>
        <CompareSection />
      </section>

      <section id="training" className="section" aria-labelledby="h-training">
        <div className="section__head">
          <h2 id="h-training" className="h2">
            Training
          </h2>
          <p className="caption">Straight from the training logs. The dashed vertical line marks the epoch whose weights were kept (best validation score).</p>
        </div>
        <TrainingSection />
      </section>

      <section id="data" className="section" aria-labelledby="h-data">
        <div className="section__head">
          <h2 id="h-data" className="h2">
            Data and model
          </h2>
          <p className="caption">
            Every product is regridded to the {detail.grid.resolution}° daily grid before the model sees it. Dataset identifiers are the
            ones in this run's configuration.
          </p>
        </div>
        <DataSection />
      </section>

      <section id="product" className="section" aria-labelledby="h-product">
        <div className="section__head">
          <h2 id="h-product" className="h2">
            Product and report
          </h2>
          <p className="caption">The gridded temperature product, the baseline and ablation fields for comparison, and the generated report.</p>
        </div>
        <ProductSection />
        <div className="gap-top">
          <ReportSection />
        </div>
      </section>
    </div>
  );
}

// ---- methods table ----------------------------------------------------------------------------

/** Pooled RMSE of every method in each test year, against GLORYS and against Argo, next to the whole period. */
function YearTable({ glorys, argo }: { glorys: MetricsResponse; argo: MetricsResponse | undefined }) {
  const { styleOf } = useRunContext();
  const years = yearsOf(glorys);
  const stability = yearStability(glorys);
  if (years.length < 2) return null;
  const argoYears = yearsOf(argo);
  const methods = methodList(argo ?? glorys);
  const range = glorys.pooled_range_m.length >= 2 ? `${fmt(glorys.pooled_range_m[0], 0)}–${fmt(glorys.pooled_range_m[1], 0)} m` : "pooled";
  const cell = (v: number | null | undefined) => <td className="num right">{fmt(v)}</td>;
  return (
    <Panel
      className="gap-top"
      title="Year by year"
      subtitle={`pooled RMSE over ${range} in °C, each test year and the whole test period · the same methods against GLORYS and against Argo`}
    >
      <div className="tablewrap" tabIndex={0} role="region" aria-label="Pooled RMSE by test year">
        <table className="table">
          <caption className="visually-hidden">Pooled RMSE by method and test year, against GLORYS and against Argo</caption>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Method
              </th>
              <th scope="colgroup" colSpan={years.length + 1} className="right table__group">
                against GLORYS
              </th>
              {argo && (
                <th scope="colgroup" colSpan={argoYears.length + 1} className="right table__group">
                  against Argo
                </th>
              )}
            </tr>
            <tr>
              {years.map((y) => (
                <th key={`g${y}`} scope="col" className="num right">
                  {y}
                </th>
              ))}
              <th scope="col" className="num right">
                all
              </th>
              {argo &&
                argoYears.map((y) => (
                  <th key={`a${y}`} scope="col" className="num right">
                    {y}
                  </th>
                ))}
              {argo && (
                <th scope="col" className="num right">
                  all
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {methods.map((m) => (
              <tr key={m.key} className={m.key === "model" ? "is-lead" : undefined}>
                <th scope="row">
                  <span className="methodcell">
                    <Swatch style={styleOf(m.key)} width={26} />
                    {m.key === "glorys" ? `${m.short} (the floor against Argo)` : m.short}
                  </span>
                </th>
                {years.map((y) => (
                  <td key={`g${y}`} className="num right">
                    {fmt(glorys.per_year?.[y]?.pooled?.[m.key]?.rmse)}
                  </td>
                ))}
                {cell(glorys.pooled[m.key]?.rmse)}
                {argo &&
                  argoYears.map((y) => (
                    <td key={`a${y}`} className="num right">
                      {fmt(argo.per_year?.[y]?.pooled?.[m.key]?.rmse)}
                    </td>
                  ))}
                {argo && cell(argo.pooled?.[m.key]?.rmse)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {stability && <p className="caption gap-top-sm">Against GLORYS: {stability.replace(/^It /, "OceanEmbed ")}</p>}
    </Panel>
  );
}

function MethodsTable({ exp, glorys, argo }: { exp: ExperimentsResponse; glorys: MetricsResponse | undefined; argo: MetricsResponse | undefined }) {
  const { styleOf, labelOf, caveats } = useRunContext();
  const rows = useMemo(() => sortRows(exp.rows), [exp.rows]);
  const range = exp.pooled_range_m.length >= 2 ? `${fmt(exp.pooled_range_m[0], 0)}–${fmt(exp.pooled_range_m[1], 0)} m` : "pooled";
  const candidates = rows.filter((r) => r.glorys_pooled?.rmse != null);
  const bestPooled = candidates.length ? candidates.reduce((a, b) => ((a.glorys_pooled!.rmse as number) <= (b.glorys_pooled!.rmse as number) ? a : b)).method : null;
  const findings = glorys ? ablationFindings(glorys) : [];

  const columns: Column<ExperimentRow>[] = [
    {
      key: "method",
      label: "Method",
      render: (r) => (
        <span className="methodcell">
          <Swatch style={styleOf(r.method)} width={26} />
          <span>
            {r.label}
            {r.kind === "reanalysis" && <span className="chip">reference</span>}
            {(r.kind === "ablation" || isAblation(r.method)) && <span className="chip">ablation</span>}
          </span>
        </span>
      ),
    },
    {
      key: "pooled",
      label: `RMSE ${range}`,
      align: "right",
      title: "Pooled RMSE against GLORYS over the pooled depth range",
      render: (r) => (r.method === bestPooled ? <span className="best">{fmt(r.glorys_pooled?.rmse)}</span> : fmt(r.glorys_pooled?.rmse)),
    },
    { key: "gclim", label: "vs climatology", align: "right", title: "Change of pooled RMSE relative to climatology (negative is better)", render: (r) => (r.method === "climatology" ? "reference" : gainText(r.pooled_rmse_gain_vs_climatology_pct)) },
    { key: "gridge", label: "vs ridge", align: "right", title: "Change of pooled RMSE relative to ridge regression (negative is better)", render: (r) => (r.method === "ridge" ? "reference" : gainText(r.pooled_rmse_gain_vs_ridge_pct)) },
    { key: "acc", label: "Anom. corr.", align: "right", title: "Anomaly correlation over the pooled range (climatology removed from both sides)", render: (r) => fmt(r.glorys_pooled?.corr_anom) },
    { key: "raw", label: "Raw corr.", align: "right", title: "Raw correlation over the pooled range (inflated by gradients; for reference)", render: (r) => fmt(r.glorys_pooled?.corr_raw, 3) },
    ...exp.selected_depths.map<Column<ExperimentRow>>((d) => ({
      key: `d${d}`,
      label: fmtDepth(d),
      align: "right",
      title: `RMSE against GLORYS at ${d} m`,
      render: (r) => fmt(r.glorys_depths?.[String(Math.round(d))]?.rmse ?? r.glorys_depths?.[String(d)]?.rmse),
    })),
    { key: "all", label: "All depths", align: "right", title: "RMSE against GLORYS over all depths", render: (r) => fmt(r.glorys_overall?.rmse) },
    { key: "argo", label: "vs Argo", align: "right", title: "RMSE against Argo profiles over all depths", render: (r) => fmt(r.argo_overall?.rmse) },
    { key: "val", label: "Val. RMSE", align: "right", title: "Best validation RMSE during training (neural models only)", render: (r) => fmt(r.val_rmse, 3) },
    { key: "epoch", label: "Epoch", align: "right", title: "Epoch of the kept checkpoint", render: (r) => (r.epoch != null ? String(r.epoch) : fmt(null)) },
  ];

  const bars = candidates.map((r) => ({
    key: r.method,
    label: labelOf(r.method),
    value: r.glorys_pooled?.rmse ?? null,
    style: styleOf(r.method),
    emphasis: r.method === "model",
    note: r.method === "climatology" ? "the reference" : gainText(r.pooled_rmse_gain_vs_climatology_pct) + " vs climatology",
  }));

  return (
    <>
      <Panel title="Comparison table" subtitle={`RMSE in °C against GLORYS unless stated · pooled range ${range} · best pooled RMSE underlined · hover a column head for its definition`}>
        <DataTable columns={columns} rows={rows} rowKey={(r) => r.method} caption="Methods and ablations" rowClass={(r) => (r.method === "model" ? "is-lead" : undefined)} />
      </Panel>
      {glorys && <YearTable glorys={glorys} argo={argo} />}
      <div className="twocol gap-top">
        <Panel title={`Pooled RMSE, ${range}`} subtitle="against GLORYS · lower is better">
          <MethodBars rows={bars} unit="°C" label={`Pooled RMSE over ${range} by method`} />
        </Panel>
        <div className="rightcol">
        {findings.map((f) => (
          <Note key={f.key} kind="honesty" title={`Ablation: ${f.label}`}>
            {f.sentence}
          </Note>
        ))}
        {exp.notes
          .filter((n) => !(caveats.synthetic && /synthetic/i.test(n)))
          .map((n) => (
            <Note key={n}>{prettyText(n)}</Note>
          ))}
        </div>
      </div>
    </>
  );
}

// ---- across runs ------------------------------------------------------------------------------

function CompareSection() {
  const { runs, run } = useRunContext();
  const [url, setUrl] = useUrlState();
  const evaluated = runs.filter((r) => r.artefacts.metrics_glorys);
  const picked = useMemo(() => {
    const wished = (url.opts.cmp ?? "").split(".").filter((n) => evaluated.some((r) => r.name === n));
    return wished.length > 0 ? wished : evaluated.map((r) => r.name).slice(0, 8);
  }, [url.opts.cmp, evaluated]);
  const cmp = useCompare(picked);

  if (evaluated.length === 0) return <Empty title="No evaluated run to compare" height={120} />;

  const toggle = (name: string) => {
    const next = picked.includes(name) ? picked.filter((n) => n !== name) : [...picked, name];
    if (next.length === 0) return;
    setUrl({ opts: { cmp: next.length === evaluated.length ? null : next.join(".") } });
  };

  return (
    <>
      <div className="toolbar" role="group" aria-label="Runs to compare">
        <span className="field__label">Runs</span>
        {evaluated.map((r) => (
          <button key={r.name} type="button" className="toggle" aria-pressed={picked.includes(r.name)} onClick={() => toggle(r.name)}>
            {runLabel(r)}
            {r.name === run.name ? " (selected)" : ""}
          </button>
        ))}
      </div>
      <QueryState query={cmp} what="The run comparison" height={240}>
        {(c) => <CompareBody runs={c.runs} summaries={runs} />}
      </QueryState>
    </>
  );
}

interface CompareRow {
  run: CompareRun;
  row: ExperimentRow;
  first: boolean;
  span: number;
}

function CompareBody({ runs, summaries }: { runs: CompareRun[]; summaries: RunSummary[] }) {
  const flat: CompareRow[] = runs.flatMap((r) => {
    const rows = sortRows(r.experiments.rows).filter((x) => x.kind !== "reanalysis");
    return rows.map((row, i) => ({ run: r, row, first: i === 0, span: rows.length }));
  });
  const summaryOf = (name: string) => summaries.find((x) => x.name === name);
  const caveatOf = (name: string) => {
    const s = summaryOf(name);
    return s ? runCaveats(s) : null;
  };
  const nameOf = (name: string) => runLabel(summaryOf(name) ?? { name, label: "" });
  const testDaysOf = (r: CompareRun) => summaryOf(r.name)?.n_test_days ?? r.n_days ?? daysInclusive(r.test_period.start, r.test_period.end);
  const styleFor = (r: CompareRun, method: string) => methodStyle(method, ablationKeysOf(r.experiments.rows.map((x) => x.method)));

  const columns: Column<CompareRow>[] = [
    {
      key: "run",
      label: "Run",
      render: (x) =>
        x.first ? (
          <span className="runcell">
            <strong>{nameOf(x.run.name)}</strong>
            <span className={`chip ${x.run.data_source === "synthetic" ? "chip--warn" : ""}`}>{x.run.data_source === "synthetic" ? "synthetic" : "real data"}</span>
            {caveatOf(x.run.name)?.shortTraining && <span className="chip chip--caution">short training</span>}
          </span>
        ) : (
          <span className="visually-hidden">{x.run.name}</span>
        ),
    },
    {
      key: "test",
      label: "Test period",
      render: (x) => (x.first ? `${fmtSpan(x.run.test_period.start, x.run.test_period.end)} · ${fmtInt(testDaysOf(x.run))} d` : ""),
    },
    {
      key: "method",
      label: "Method",
      render: (x) => (
        <span className="methodcell">
          <Swatch style={styleFor(x.run, x.row.method)} width={26} />
          {shortLabel(x.row.method, x.row.label)}
        </span>
      ),
    },
    { key: "pooled", label: "Pooled RMSE (°C)", align: "right", render: (x) => fmt(x.row.glorys_pooled?.rmse) },
    { key: "gain", label: "vs climatology", align: "right", render: (x) => (x.row.method === "climatology" ? "reference" : gainText(x.row.pooled_rmse_gain_vs_climatology_pct)) },
    { key: "ridge", label: "vs ridge", align: "right", render: (x) => (x.row.method === "ridge" ? "reference" : gainText(x.row.pooled_rmse_gain_vs_ridge_pct)) },
    { key: "acc", label: "Anom. corr.", align: "right", render: (x) => fmt(x.row.glorys_pooled?.corr_anom) },
    { key: "skill", label: "Skill", align: "right", title: "1 − MSE / MSE of climatology over the pooled range", render: (x) => fmt(x.row.glorys_pooled?.skill_vs_clim) },
    { key: "argo", label: "Argo RMSE (°C)", align: "right", render: (x) => fmt(x.row.argo_overall?.rmse) },
    { key: "nargo", label: "Argo profiles", align: "right", render: (x) => (x.first ? fmtInt(x.run.n_argo_profiles) : "") },
  ];

  return (
    <>
      <div className="compare">
        {runs.map((r) => {
          const rows = sortRows(r.experiments.rows).filter((x) => x.kind !== "reanalysis" && x.method !== "climatology");
          const cav = caveatOf(r.name);
          const max = Math.max(1, ...runs.flatMap((q) => q.experiments.rows.map((x) => x.pooled_rmse_gain_vs_climatology_pct ?? 0)));
          return (
            <Panel
              key={r.name}
              untagged
              title={
                <>
                  {nameOf(r.name)}
                  <span className={`chip ${r.data_source === "synthetic" ? "chip--warn" : ""}`}>{r.data_source === "synthetic" ? "synthetic" : "real data"}</span>
                  {cav?.shortTraining && <span className="chip chip--caution">short training</span>}
                </>
              }
              subtitle="reduction of pooled RMSE against the climatology of the same run"
            >
              <MethodBars
                label={`Pooled RMSE reduction against climatology in run ${nameOf(r.name)}`}
                unit="%"
                digits={1}
                max={max}
                rows={rows.map((x) => ({
                  key: x.method,
                  label: shortLabel(x.method, x.label),
                  value: x.pooled_rmse_gain_vs_climatology_pct != null ? Math.max(0, x.pooled_rmse_gain_vs_climatology_pct) : null,
                  style: styleFor(r, x.method),
                  emphasis: x.method === "model",
                  note: (x.pooled_rmse_gain_vs_climatology_pct ?? 0) < 0 ? `worse than climatology by ${fmt(-(x.pooled_rmse_gain_vs_climatology_pct ?? 0), 1)} %` : undefined,
                }))}
              />
            </Panel>
          );
        })}
      </div>
      <Panel className="gap-top" untagged title="Run by run" subtitle="pooled metrics against GLORYS on each run's own test period">
        <DataTable columns={columns} rows={flat} rowKey={(x) => `${x.run.name}:${x.row.method}`} caption="Comparison across runs" rowClass={(x) => (x.first ? "is-group" : undefined)} />
      </Panel>
    </>
  );
}

// ---- data and model ---------------------------------------------------------------------------

function DataSection() {
  const { detail } = useRunContext();
  const m = detail.model;
  const use = inputUse(detail);
  const scratch = m.main_init === "scratch";
  const columns: Column<DataProduct>[] = [
    {
      key: "var",
      label: "Variable",
      render: (p) => (
        <span>
          {p.long_name} <span className="chip">{p.role}</span>
          {p.role === "input" && use.partial && (
            <span className={`chip ${p.used_by_model !== false ? "chip--lead" : ""}`}>{p.used_by_model !== false ? "used by the model" : "available, not used"}</span>
          )}
        </span>
      ),
    },
    {
      key: "product",
      label: "Product and dataset identifiers",
      render: (p) => (
        <span>
          {prettyText(p.product)}
          {p.dataset_ids.length > 0 && <span className="mono ids">{p.dataset_ids.join(", ")}</span>}
        </span>
      ),
    },
    { key: "res", label: "Native resolution", render: (p) => prettyText(p.native_resolution) },
    { key: "regrid", label: "Brought to the grid by", render: (p) => prettyText(p.regridding) },
  ];
  const kv = (obj: Record<string, unknown> | undefined) =>
    Object.entries(obj ?? {}).map(([k, v]) => (
      <div key={k}>
        <dt>{CONFIG_LABELS[k] ?? k.replace(/_/g, " ")}</dt>
        <dd className="num">{typeof v === "number" ? (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)))) : Array.isArray(v) ? v.join(", ") : String(v)}</dd>
      </div>
    ));
  return (
    <>
      <Panel
        title="Data products"
        subtitle={
          use.partial
            ? `the model uses ${use.used.length} of the ${use.all.length} surface products (${inputNames(use.used)}); the others are harmonised and shown, but not fed to it`
            : `the ${use.all.length} surface inputs, the training target and the validation data of this run`
        }
      >
        <DataTable columns={columns} rows={detail.products} rowKey={(p) => p.variable} caption="Data products" />
      </Panel>
      <div className="trio gap-top">
        <Panel title="Encoder" subtitle={`CNN stem + Transformer${scratch ? ", trained from scratch in the main model" : ", shared by pretraining and reconstruction"}`}>
          <dl className="kv">{kv(m.model as Record<string, unknown>)}</dl>
        </Panel>
        <Panel title="Pretraining" subtitle={`masked reconstruction of the surface fields${scratch ? " · used by the pretrained ablation only, not by the main model" : ""}`}>
          <dl className="kv">{kv(m.pretrain as Record<string, unknown>)}</dl>
        </Panel>
        <Panel title="Reconstruction training" subtitle="supervised, anomaly from climatology at every depth">
          <dl className="kv">{kv(m.train as Record<string, unknown>)}</dl>
        </Panel>
      </div>
      {m.output && <p className="caption gap-top-sm">Output: {prettyText(m.output)}.</p>}
    </>
  );
}
