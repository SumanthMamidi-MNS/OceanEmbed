/**
 * Data & downloads: what the product is made of and how to take it away. The surface fields the
 * model reads (and the ones the run carries but does not use), the period, the grid and the depth
 * levels, the NetCDF product files, the released model and the generated report.
 */
import { ViewLink, runLabel } from "@/components/shell/Shell";
import { DataTable, Panel, type Column } from "@/components/ui/primitives";
import type { DataProduct } from "@/api/types";
import { fmtSpan } from "@/lib/dates";
import { fmtDepth, fmtInt, fmtLat, fmtLon, prettyText } from "@/lib/format";
import { inputNames, inputUse } from "@/lib/inputs";
import { useRunContext } from "@/state/runContext";
import { ProductSection, ReportSection } from "./ReportSection";

/** The surface fields, the training target and the validation data of the run, with what the model uses. */
export function DataProducts() {
  const { detail } = useRunContext();
  const use = inputUse(detail);
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
  return (
    <Panel
      title="Data products"
      subtitle={
        use.partial
          ? `the model uses ${use.used.length} of the ${use.all.length} surface products (${inputNames(use.used)}); the others are harmonised and shown in the Explorer, but not fed to it`
          : `the ${use.all.length} surface inputs, the training target and the validation data of this run`
      }
    >
      <DataTable columns={columns} rows={detail.products} rowKey={(p) => p.variable} caption="Data products" />
    </Panel>
  );
}

export default function Data() {
  const { run, detail, depths, geom } = useRunContext();
  const g = detail.grid;
  const m = detail.model;
  const use = inputUse(detail);
  const split = run.split ?? {};
  const main = detail.methods.find((x) => x.key === "model");
  const span = (key: "train" | "val" | "test", days: number | null | undefined) =>
    split[key]?.start ? `${fmtSpan(split[key]?.start, split[key]?.end)}${days != null ? ` · ${fmtInt(days)} days` : ""}` : "–";

  return (
    <div className="datapage">
      <header className="viewhead">
        <p className="overline">Data &amp; downloads · {runLabel(run)}</p>
        <h1 className="h1">What goes in, what you can take away</h1>
        <p className="caption">
          {use.used.length} daily satellite surface field{use.used.length === 1 ? "" : "s"} in ({inputNames(use.used)}), temperature at {g.n_depth} depths out, one field a
          day on a {g.resolution}° grid, as CF NetCDF files.
        </p>
        <nav className="subnav" aria-label="Sections of this view">
          <a href="#coverage">Period and grid</a>
          <a href="#inputs">Inputs</a>
          <a href="#product">Download</a>
          <a href="#model">The model</a>
          <a href="#report">Report</a>
        </nav>
      </header>

      <section id="coverage" className="section" aria-labelledby="h-coverage">
        <div className="section__head">
          <h2 id="h-coverage" className="h2">
            Period and grid
          </h2>
        </div>
        <dl className="factrow factrow--grid">
          <div>
            <dt>Reconstructed days</dt>
            <dd className="num">
              {fmtInt(run.n_prediction_days)}
              {detail.prediction_dates.length > 0 ? ` · ${fmtSpan(detail.prediction_dates[0], detail.prediction_dates[detail.prediction_dates.length - 1])}` : ""}
            </dd>
          </div>
          <div>
            <dt>Trained on</dt>
            <dd className="num">{span("train", run.n_train_days)}</dd>
          </div>
          <div>
            <dt>Validated on</dt>
            <dd className="num">{span("val", run.n_val_days)}</dd>
          </div>
          <div>
            <dt>Tested on (never seen)</dt>
            <dd className="num">{span("test", run.n_test_days)}</dd>
          </div>
          <div>
            <dt>Grid</dt>
            <dd className="num">
              {g.resolution}° · {g.n_lat} × {g.n_lon} cells · {fmtLat(geom.lat0, 0)}–{fmtLat(geom.lat1, 0)}, {fmtLon(geom.lon0, 0)}–{fmtLon(geom.lon1, 0)}
            </dd>
          </div>
          <div>
            <dt>Basins</dt>
            <dd>{detail.basins.length > 0 ? detail.basins.map((b) => b.label).join(", ") : "–"}</dd>
          </div>
          <div className="factrow__wide">
            <dt>
              {depths.length} depth levels (0–{fmtInt(depths[depths.length - 1])} m)
            </dt>
            <dd className="num">{depths.map((d) => fmtDepth(d)).join(" · ")}</dd>
          </div>
        </dl>
        {run.description && <p className="caption gap-top-sm">{prettyText(run.description)}</p>}
      </section>

      <section id="inputs" className="section" aria-labelledby="h-inputs">
        <div className="section__head">
          <h2 id="h-inputs" className="h2">
            What the model reads
          </h2>
          <p className="caption">
            Every product is regridded to the {g.resolution}° daily grid before the model sees it. No subsurface measurement goes in: the reanalysis is the
            training target, the Argo floats are used only to check the result.
          </p>
        </div>
        <DataProducts />
      </section>

      <section id="product" className="section" aria-labelledby="h-product">
        <div className="section__head">
          <h2 id="h-product" className="h2">
            Download the product
          </h2>
          <p className="caption">One NetCDF file per month; every file opens with xarray, Panoply or any CF-aware tool.</p>
        </div>
        <ProductSection />
      </section>

      <section id="model" className="section" aria-labelledby="h-model">
        <div className="section__head">
          <h2 id="h-model" className="h2">
            The model
          </h2>
        </div>
        <div className="twocol">
          <Panel title="What it is" subtitle="one network, one day of surface fields in, the temperature at every depth out">
            <dl className="kv">
              <div>
                <dt>Inputs</dt>
                <dd>{inputNames(use.used) || "–"}</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>{m.output ? prettyText(m.output) : `temperature at ${g.n_depth} depths`}</dd>
              </div>
              <div>
                <dt>Network</dt>
                <dd>
                  convolutional encoder with a Transformer, convolutional decoder
                  {m.main_init === "scratch" ? ", trained from scratch" : m.main_init === "pretrained" ? ", encoder pretrained on the surface fields" : ""}
                </dd>
              </div>
              {main?.epoch != null && (
                <div>
                  <dt>Weights kept</dt>
                  <dd className="num">epoch {main.epoch} (best validation score)</dd>
                </div>
              )}
            </dl>
          </Panel>
          <Panel title="Released weights" subtitle="kept in the repository, not served by this site">
            <p className="caption">
              The weights of the released model and its model card are in the repository under <code>models/final/</code>. To reconstruct other days from them,
              run <code>oceanembed predict --weights</code>; <code>docs/reproduce.md</code> describes how to rebuild every number shown here.
            </p>
            <p className="caption gap-top-sm">
              Why this network, against which alternatives, and what its training looked like:{" "}
<ViewLink view="research">the Research pages</ViewLink>.
            </p>
          </Panel>
        </div>
      </section>

      <section id="report" className="section" aria-labelledby="h-report">
        <div className="section__head">
          <h2 id="h-report" className="h2">
            Report
          </h2>
          <p className="caption">Written by the pipeline from the same metric files this site reads.</p>
        </div>
        <ReportSection />
      </section>
    </div>
  );
}
