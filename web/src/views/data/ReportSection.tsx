/**
 * The run's deliverables: the NetCDF product files and the generated report with its figures
 * (Data & downloads), and the baseline and ablation fields written for comparison (Research).
 */
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useProduct, useReport } from "@/api/queries";
import type { FigureLike } from "./types";
import { DataTable, Empty, Icon, Loading, Panel, QueryState, TableTwin, type Column } from "@/components/ui/primitives";
import type { ExtraProduct, ProductFile } from "@/api/types";
import { fmtMonth, fmtSpan } from "@/lib/dates";
import { fmtBytes, fmtInt, prettyText } from "@/lib/format";
import { estimateRole } from "@/lib/methods";
import { useRunContext } from "@/state/runContext";

const ReportMarkdown = lazy(() => import("./ReportMarkdown"));

function fileColumns(): Column<ProductFile>[] {
  return [
    { key: "month", label: "Month", render: (f) => fmtMonth(f.month) },
    { key: "span", label: "Days covered", render: (f) => `${fmtSpan(f.start, f.end)} (${f.n_days})` },
    { key: "name", label: "File", render: (f) => <span className="mono">{f.name}</span> },
    { key: "size", label: "Size", align: "right", render: (f) => fmtBytes(f.size) },
    {
      key: "get",
      label: "",
      align: "right",
      render: (f) => (
        <a className="dl" href={f.url} download={f.name} aria-label={`Download ${f.name}, ${fmtMonth(f.month)}`}>
          <Icon name="download" size={14} /> Download
        </a>
      ),
    },
  ];
}

/** What an extra product is, stated on its panel so a downloaded file is never mistaken for the product. */
function extraKind(p: ExtraProduct): { chip: string; text: string } {
  const role = estimateRole(p.method);
  if (p.kind === "ablation") {
    return { chip: "ablation", text: `An ablation of the model${role ? ` (${role})` : ""}, written for comparison. It is not the product.` };
  }
  if (p.kind === "baseline") {
    return { chip: "baseline", text: `A baseline${role ? ` (${role})` : ""}, written for comparison. It is not the product.` };
  }
  return { chip: p.kind, text: "Written for comparison with the main product." };
}

/**
 * `show="product"`: the product a user downloads. `show="comparison"`: the baseline and ablation
 * fields, each labelled as not the product (Research area).
 */
export function ProductSection({ show = "product" }: { show?: "product" | "comparison" }) {
  const { run, detail } = useRunContext();
  const product = useProduct(run.name, run.artefacts.n_product_files > 0);
  if (run.artefacts.n_product_files === 0) {
    if (show === "comparison") return null;
    return (
      <Empty title="No product files yet" height={140}>
        “oceanembed predict” writes one CF-1.8 NetCDF file per month.
      </Empty>
    );
  }
  const columns = fileColumns();
  const g = detail.grid;
  const shape = `temperature(time, depth, lat, lon) in °C · ${g.n_depth} depths × ${g.n_lat} × ${g.n_lon} cells · daily`;
  const count = (n: number) => `${fmtInt(n)} monthly file${n === 1 ? "" : "s"}`;
  return (
    <QueryState query={product} what="The product files" height={200}>
      {(p) => {
        const extras = p.extra_products ?? [];
        if (show === "comparison") {
          if (extras.length === 0) return <Empty title="This run wrote no baseline or ablation fields" height={120} />;
          return (
            <div className="extras">
              <p className="extras__head">
                <span className="caption">Same grid, days and format as the product, for comparison only. The monthly file names repeat across methods.</span>
              </p>
              <div className="twocol">
                {extras.map((e) => {
                  const kind = extraKind(e);
                  return (
                    <Panel
                      key={e.method}
                      title={
                        <>
                          {e.label}
                          <span className="chip">{kind.chip}</span>
                        </>
                      }
                      subtitle={`${count(e.files.length)}, ${fmtBytes(e.total_size)} in total`}
                    >
                      <TableTwin label={`Show the ${e.files.length} file${e.files.length === 1 ? "" : "s"}`}>
                        <DataTable columns={columns} rows={e.files} rowKey={(f) => f.name} caption={`NetCDF files of ${e.label}`} dense />
                      </TableTwin>
                      <p className="caption gap-top-sm">{kind.text}</p>
                    </Panel>
                  );
                })}
              </div>
            </div>
          );
        }
        return (
          <>
            <Panel
              title={
                <>
                  Reconstructed temperature, NetCDF (CF-1.8)<span className="chip chip--lead">the product</span>
                </>
              }
              subtitle={`OceanEmbed · ${shape} · ${count(p.files.length)}, ${fmtBytes(p.total_size)} in total`}
            >
              <DataTable columns={columns} rows={p.files} rowKey={(f) => f.name} caption="NetCDF product files of the main model" dense />
              <p className="caption gap-top-sm">
                Land and cells below the sea floor carry the fill value. The global attributes record the data source, the checkpoint and
                the full configuration, so a file identifies the run that made it.
              </p>
            </Panel>
          </>
        );
      }}
    </QueryState>
  );
}

export function ReportSection() {
  const { run } = useRunContext();
  const report = useReport(run.name, run.artefacts.report);
  const [open, setOpen] = useState<FigureLike | null>(null);
  const [showText, setShowText] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const figureUrl = useMemo(() => {
    const byName = new Map((report.data?.figures ?? []).map((f) => [f.name, f.url]));
    return (src: string) => {
      const name = src.split("/").pop() ?? src;
      return byName.get(name) ?? src;
    };
  }, [report.data]);

  if (!run.artefacts.report) {
    return (
      <Empty title="No report for this run" height={140}>
        “oceanembed report” writes report.md and its figures.
      </Empty>
    );
  }

  return (
    <QueryState query={report} what="The report" height={240}>
      {(r) => (
        <>
          <Panel title="Figures of the generated report" subtitle={`${r.figures.length} static figures written by the pipeline (matplotlib) · click to enlarge`}>
            {r.figures.length === 0 ? (
              <Empty title="The report has no figures" />
            ) : (
              <ul className="gallery">
                {r.figures.map((f) => {
                  const label = prettyText(f.name.replace(/\.png$/i, "").replace(/_/g, " "));
                  return (
                    <li key={f.name}>
                      <button type="button" className="gallery__item" onClick={() => setOpen({ url: f.url, label })}>
                        <img src={f.url} alt={`Report figure: ${label}`} loading="lazy" decoding="async" />
                        <span className="gallery__label">{label}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>

          <Panel
            className="gap-top"
            title="Report text"
            subtitle="report.md as written by “oceanembed report”: setup, metric tables, figures, limitations"
            actions={
              <button type="button" className="btn btn--quiet" aria-expanded={showText} onClick={() => setShowText((v) => !v)}>
                {showText ? "Hide the report" : "Read the report"}
              </button>
            }
          >
            {showText ? (
              <Suspense fallback={<Loading height={200} label="Loading the report" />}>
                <ReportMarkdown markdown={r.markdown} resolve={figureUrl} />
              </Suspense>
            ) : (
              <p className="caption">
                {fmtInt(r.markdown.split(/\s+/).length)} words. The report is generated from the same metric files this dashboard
                reads, so the two cannot disagree.
              </p>
            )}
          </Panel>

          {open && (
            <div className="lightbox" role="dialog" aria-modal="true" aria-label={open.label} onClick={() => setOpen(null)}>
              <figure onClick={(e) => e.stopPropagation()}>
                <img src={open.url} alt={`Report figure: ${open.label}`} />
                <figcaption>
                  <span>{open.label}</span>
                  <button type="button" className="btn btn--quiet" onClick={() => setOpen(null)} autoFocus>
                    Close
                  </button>
                </figcaption>
              </figure>
            </div>
          )}
        </>
      )}
    </QueryState>
  );
}
