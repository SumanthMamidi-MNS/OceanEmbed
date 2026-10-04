/**
 * Accuracy: how far to trust a value. The reconstruction scored on days it never saw, against the
 * GLORYS reanalysis and against Argo float profiles, by depth, by basin, per test year, in space
 * and in time, next to the seasonal climatology: the estimate a user has without this product.
 *
 * The same sections serve the Research area (route /research/scores), where every method of the
 * run is shown; which methods appear comes from the run context (`scope`, `scoped`).
 */
import { useMemo } from "react";
import { useGlorysMetrics } from "@/api/queries";
import type { MetricsResponse } from "@/api/types";
import { SelectionBar } from "@/components/controls/SelectionBar";
import { runLabel } from "@/components/shell/Shell";
import { Empty, Note, QueryState } from "@/components/ui/primitives";
import { fmtSpan } from "@/lib/dates";
import { fmtInt, prettyText } from "@/lib/format";
import { accuracyHeadline, climatologyName, climatologyText, trustLimits } from "@/lib/narrative";
import { useRunContext } from "@/state/runContext";
import { ArgoSection } from "./ArgoSection";
import { DailySection } from "./DailySection";
import { GlorysSection } from "./GlorysSection";
import { MapsSection } from "./MapsSection";

export default function Accuracy() {
  return <AccuracyBody />;
}

/** The answer before the tables: how large the error is, and where not to rely on it. */
function Verdict({ metrics }: { metrics: MetricsResponse }) {
  const { run } = useRunContext();
  const clim = climatologyName(run.n_harmonic_terms);
  const head = useMemo(() => accuracyHeadline(metrics, clim), [metrics, clim]);
  const trust = useMemo(() => trustLimits(metrics, clim), [metrics, clim]);
  return (
    <div className="verdict gap-bottom">
      <p className={`verdict__lead ${head.beatsClimatology ? "" : "is-negative"}`}>{head.sentence}</p>
      {(trust.use || trust.limit) && (
        <p className="verdict__limit">
          {trust.use} {trust.limit && <strong>{trust.limit}</strong>}
        </p>
      )}
    </div>
  );
}

export function AccuracyBody() {
  const { run, detail, caveats, scope } = useRunContext();
  const research = scope === "research";
  const metrics = useGlorysMetrics(run.name, run.artefacts.metrics_glorys);
  const md = detail.metrics_metadata;
  const test = run.split?.test;
  const clim = md?.climatology;

  return (
    <div className="validation">
      <SelectionBar estimate={research} />
      <header className="viewhead">
        <p className="overline">
          {research ? "Every score, every method" : "Accuracy"} · {runLabel(run)}
        </p>
        <h1 className="h1">{research ? "Every method on the same test days" : "How far to trust it"}</h1>
        <p className="caption">
          Scored on {run.n_test_days != null ? `${fmtInt(run.n_test_days)} days` : "days"} the model never saw ({fmtSpan(test?.start, test?.end)}), against{" "}
          {prettyText(md?.reference ?? "the GLORYS reanalysis on the model grid")} and against Argo floats,{" "}
          {research ? "next to ridge regression, the other baselines and the ablation" : `next to ${climatologyName(run.n_harmonic_terms)}: what you would use without this product`}. Climatology:{" "}
          {climatologyText(run.n_harmonic_terms)} of each cell, fitted on the {fmtInt(run.n_train_days)} training days
          {clim?.train_start ? ` (${fmtSpan(clim.train_start, clim.train_end)})` : ""}.
        </p>
        <nav className="subnav" aria-label="Sections of this view">
          <a href="#glorys">By depth and basin</a>
          <a href="#where">Where: error maps</a>
          <a href="#when">When: daily error</a>
          <a href="#argo">Against Argo floats</a>
        </nav>
      </header>

      {(caveats.synthetic || caveats.shortTraining) && (
        <Note kind="caveat" title={caveats.synthetic ? "Synthetic data" : "Short training period"} className="gap-bottom">
          {caveats.synthetic
            ? "Every score on this page is measured on the analytic test ocean. It verifies the evaluation code, not the model's skill in the real ocean."
            : caveats.shortTrainingNote}
        </Note>
      )}

      {!research && metrics.data && <Verdict metrics={metrics.data} />}

      <section id="glorys" className="section" aria-labelledby="h-glorys">
        <div className="section__head">
          <h2 id="h-glorys" className="h2">
            Against the GLORYS reanalysis
          </h2>
          <p className="caption">The reanalysis the model was trained to reproduce: every cell, every depth, every test day.</p>
        </div>
        {run.artefacts.metrics_glorys ? (
          <QueryState query={metrics} what="The evaluation against GLORYS" height={320}>
            {(m) => <GlorysSection metrics={m} />}
          </QueryState>
        ) : (
          <Empty title="This run has not been evaluated yet" height={160}>
            Run “oceanembed evaluate” for this configuration to produce metrics_glorys.json.
          </Empty>
        )}
      </section>

      <section id="where" className="section" aria-labelledby="h-where">
        <div className="section__head">
          <h2 id="h-where" className="h2">
            Where the error is
          </h2>
          <p className="caption">The same scores at every grid point, over the test period. The maps are linked; click one to move the water column.</p>
        </div>
        {run.artefacts.maps ? (
          <MapsSection />
        ) : (
          <Empty title="No error maps for this run" height={160}>
            maps_glorys.nc is written by “oceanembed evaluate”.
          </Empty>
        )}
      </section>

      <section id="when" className="section" aria-labelledby="h-when">
        <div className="section__head">
          <h2 id="h-when" className="h2">
            When the error is
          </h2>
          <p className="caption">Domain-wide scores of each test day. A model that only reproduced the seasons would track the climatology line.</p>
        </div>
        {metrics.data ? <DailySection metrics={metrics.data} /> : <Empty title="No daily error series" height={120} />}
      </section>

      <section id="argo" className="section" aria-labelledby="h-argo">
        <div className="section__head">
          <h2 id="h-argo" className="h2">
            Against Argo float profiles
          </h2>
          <p className="caption">In-situ profiles, interpolated to the standard depths and compared in the same cell on the same day.</p>
        </div>
        <ArgoSection />
      </section>
    </div>
  );
}
