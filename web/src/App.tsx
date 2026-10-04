import { lazy, Suspense, useEffect, useMemo } from "react";
import { useDates, useOceanMask, useRun, useRuns } from "@/api/queries";
import { Shell } from "@/components/shell/Shell";
import { Empty, ErrorState, Loading } from "@/components/ui/primitives";
import { defaultRun, evaluatedRuns, liveRun, runCaveats } from "@/lib/narrative";
import { navigate, useUrlState } from "@/state/router";
import { RunProvider, useRunContext } from "@/state/runContext";
import type { ViewKey } from "@/state/url";

const Overview = lazy(() => import("@/views/overview/Overview"));
const Explorer = lazy(() => import("@/views/explorer/Explorer"));
const Live = lazy(() => import("@/views/live/Live"));
const Accuracy = lazy(() => import("@/views/accuracy/Accuracy"));
const Data = lazy(() => import("@/views/data/Data"));
const Research = lazy(() => import("@/views/research/Research"));
const ResearchScores = lazy(() => import("@/views/research/ResearchScores"));
const ResearchMaps = lazy(() => import("@/views/research/ResearchMaps"));
const Embedding = lazy(() => import("@/views/research/Embedding"));

const VIEW_COMPONENTS: Record<ViewKey, React.LazyExoticComponent<() => React.JSX.Element>> = {
  overview: Overview,
  explore: Explorer,
  live: Live,
  accuracy: Accuracy,
  data: Data,
  research: Research,
  research_scores: ResearchScores,
  research_maps: ResearchMaps,
  research_embedding: Embedding,
};

const VIEW_TITLES: Record<ViewKey, string> = {
  overview: "Overview",
  explore: "Ocean explorer",
  live: "Live",
  accuracy: "Accuracy",
  data: "Data & downloads",
  research: "Research · Methods",
  research_scores: "Research · Every score",
  research_maps: "Research · Methods on the map",
  research_embedding: "Research · Embedding",
};

/** `,` `.` step the day and `[` `]` the depth on every view (the explorer adds the arrow keys). */
function GlobalShortcuts() {
  const { dateIndex, depthIndex, setDateIndex, setDepthIndex, dates, depths } = useRunContext();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target;
      if (t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "," && dateIndex > 0) setDateIndex(dateIndex - 1);
      else if (e.key === "." && dateIndex < dates.length - 1) setDateIndex(dateIndex + 1);
      else if (e.key === "[" && depthIndex > 0) setDepthIndex(depthIndex - 1);
      else if (e.key === "]" && depthIndex < depths.length - 1) setDepthIndex(depthIndex + 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dateIndex, depthIndex, setDateIndex, setDepthIndex, dates.length, depths.length]);
  return null;
}

export function App() {
  const [url] = useUrlState();
  const runsQ = useRuns();
  const runs = useMemo(() => runsQ.data ?? [], [runsQ.data]);
  // the evaluated pages show an evaluated run; the Live view always shows the live run, whatever run the link names
  const live = useMemo(() => liveRun(runs), [runs]);
  const evaluated = useMemo(() => evaluatedRuns(runs), [runs]);
  const wanted = url.run ? (evaluated.find((r) => r.name === url.run) ?? null) : null;
  const namesLive = !!url.run && live?.name === url.run;
  const selected = wanted ?? (url.run && !namesLive ? null : defaultRun(runs));
  const onLive = url.view === "live" && !!live;
  const run = onLive ? live : selected;
  const detailQ = useRun(run?.name ?? null);
  const datesQ = useDates(run?.name ?? null);
  const maskQ = useOceanMask(run?.artefacts.predictions ? run.name : null);

  // pin the resolved run in the URL so every link that is copied names it
  useEffect(() => {
    if (runsQ.isPending || runsQ.isError) return;
    // a link that names the live run belongs to the Live view; the run parameter is the evaluated run
    if (namesLive) navigate({ view: "live", run: selected?.name ?? null, date: url.date }, "replace");
    // without a live run there is no Live view: its route opens the Overview
    else if (url.view === "live" && !live) navigate({ view: "overview" }, "replace");
    // naming the run must not cost the link its day or its options
    else if (selected && !url.run) navigate({ run: selected.name, date: url.date, est: url.est, opts: url.opts }, "replace");
  }, [runsQ.isPending, runsQ.isError, namesLive, live, selected, url.run, url.view, url.date, url.est, url.opts]);

  useEffect(() => {
    document.title = run ? `${VIEW_TITLES[url.view]} · ${run.label || run.name} · OceanEmbed` : "OceanEmbed";
  }, [url.view, run]);

  const caveats = useMemo(() => (run ? runCaveats(run) : null), [run]);
  const View = VIEW_COMPONENTS[url.view];

  let body: React.ReactNode;
  if (runsQ.isError) {
    body = <ErrorState error={runsQ.error} what="the list of runs" height={280} onRetry={() => void runsQ.refetch()} />;
  } else if (runsQ.isPending) {
    body = <Loading height={420} label="Loading runs" />;
  } else if (runs.length === 0) {
    body = (
      <Empty title="No finished runs yet" height={280}>
        The outputs folder has no run with a run_meta.json. Produce one with “oceanembed run-all --config configs/synthetic.yaml”.
      </Empty>
    );
  } else if (url.view === "live" && !live) {
    body = <Loading height={420} label="Opening the overview" />;
  } else if (!run) {
    body = (
      <Empty title={`Run “${url.run}” was not found`} height={280}>
        Available runs: {evaluated.map((r) => (r.label ? `${r.label} (${r.name})` : r.name)).join(", ")}. Choose one in the run selector above.
      </Empty>
    );
  } else if (detailQ.isError) {
    body = <ErrorState error={detailQ.error} what="this run" height={280} onRetry={() => void detailQ.refetch()} />;
  } else if (!detailQ.data) {
    body = <Loading height={420} label="Loading the run" />;
  } else {
    body = (
      <RunProvider key={run.name} runs={runs} run={run} detail={detailQ.data} dates={datesQ.data} mask={maskQ.data}>
        <GlobalShortcuts />
        <Suspense fallback={<Loading height={420} label="Loading the view" />}>
          <View />
        </Suspense>
      </RunProvider>
    );
  }

  return (
    <Shell runs={evaluated} run={run} selected={selected} live={live} caveats={caveats}>
      {body}
    </Shell>
  );
}
