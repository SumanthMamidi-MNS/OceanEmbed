/** Training curves from the run's logs: pretraining, reconstruction training and its ablations. */
import { useMemo } from "react";
import { useTraining } from "@/api/queries";
import type { TrainingLog } from "@/api/types";
import { DepthHeatmap } from "@/components/charts/DepthHeatmap";
import { Legend } from "@/components/charts/marks";
import { XYChart, type ChartSeries } from "@/components/charts/XYChart";
import { Colorbar } from "@/components/map/Colorbar";
import { DataTable, Empty, Panel, QueryState, type Column } from "@/components/ui/primitives";
import { fmt, fmtDuration } from "@/lib/format";
import type { MethodStyle } from "@/lib/methods";
import { padRange } from "@/lib/scales";
import { extent } from "@/lib/stats";
import { useRunContext } from "@/state/runContext";

function numbers(log: TrainingLog, column: string): (number | null)[] {
  const col = log.columns[column];
  if (!Array.isArray(col)) return [];
  return col.map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
}

function line(key: string, label: string, style: MethodStyle, x: (number | null)[], y: (number | null)[]): ChartSeries {
  return { key, label, style, markers: x.length <= 40, points: x.map((e, i) => ({ x: e, y: y[i] ?? null })) };
}

const TRAIN_STYLE: MethodStyle = { color: "#6B7680", dash: "5 3", marker: "square", width: 1.5, z: 2 };
const VAL_STYLE: MethodStyle = { color: "#0B5FA5", dash: "", marker: "circle", width: 2, z: 5 };
const BASE_STYLE: MethodStyle = { color: "#6B7680", dash: "1.5 3", marker: "cross", width: 1.5, z: 1 };

export function TrainingSection() {
  const { run } = useRunContext();
  const has = run.artefacts.training_logs.length > 0;
  const training = useTraining(run.name, has);
  if (!has) {
    return (
      <Empty title="This run has no training logs" height={140}>
        logs/pretrain.jsonl and logs/train.jsonl are written by “oceanembed pretrain” and “oceanembed train”.
      </Empty>
    );
  }
  return (
    <QueryState query={training} what="The training log" height={300}>
      {(t) => <Curves logs={t.logs} />}
    </QueryState>
  );
}

function Curves({ logs }: { logs: Record<string, TrainingLog> }) {
  const { depths, styleOf, labelOf } = useRunContext();
  const pre = logs.pretrain;
  const trainKeys = Object.keys(logs)
    .filter((k) => logs[k].kind === "train")
    .sort((a, b) => (a === "train" ? -1 : b === "train" ? 1 : a.localeCompare(b)));
  const methodOf = (stem: string) => (stem === "train" ? "model" : `model_${logs[stem].tag ?? stem.replace(/^train_/, "")}`);

  const valRmse = useMemo<ChartSeries[]>(
    () => trainKeys.map((k) => line(k, labelOf(methodOf(k)), styleOf(methodOf(k)), numbers(logs[k], "epoch"), numbers(logs[k], "val_rmse"))),
    [logs], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const rmseDomain = padRange(...extent(valRmse.flatMap((s) => s.points.map((p) => p.y))), 0.08);
  const maxEpoch = (series: ChartSeries[]) => Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.x ?? 0)));

  const main = logs.train;
  const lossSeries = main
    ? [line("train_loss", "Training loss", TRAIN_STYLE, numbers(main, "epoch"), numbers(main, "train_loss")), line("val_loss", "Validation loss", VAL_STYLE, numbers(main, "epoch"), numbers(main, "val_loss"))]
    : [];
  const preSeries = pre
    ? [
        line("meanfill", "Filling with the mean", BASE_STYLE, numbers(pre, "epoch"), numbers(pre, "val_meanfill")),
        line("train_loss", "Training loss", TRAIN_STYLE, numbers(pre, "epoch"), numbers(pre, "train_loss")),
        line("val_loss", "Validation loss", VAL_STYLE, numbers(pre, "epoch"), numbers(pre, "val_loss")),
      ].filter((s) => s.points.some((p) => p.y != null))
    : [];

  const perDepth = main?.columns.val_rmse_per_depth as (number | null)[][] | undefined;
  const epochs = main ? numbers(main, "epoch").filter((v): v is number => v != null) : [];
  const heat = useMemo(() => {
    if (!perDepth || perDepth.length === 0) return null;
    const nz = perDepth[0].length;
    // relative to the first epoch, so every depth is readable on one scale
    const rel: (number | null)[][] = Array.from({ length: nz }, (_, k) =>
      perDepth.map((row) => {
        const first = perDepth[0][k];
        const v = row[k];
        return v == null || first == null || first === 0 ? null : (v / first - 1) * 100;
      }),
    );
    const flat = rel.flat().filter((v): v is number => v != null).map(Math.abs);
    return { rel, lim: Math.max(5, Math.ceil(Math.max(...flat, 0) / 5) * 5) };
  }, [perDepth]);

  const summaryRows = [...(pre ? ["pretrain"] : []), ...trainKeys].map((stem) => {
    const log = logs[stem];
    const best = log.best ?? {};
    const bestKey = Object.keys(best).find((k) => k !== "epoch");
    return {
      stem,
      name: stem === "pretrain" ? "Pretraining (masked reconstruction)" : `Reconstruction: ${labelOf(methodOf(stem), false)}`,
      epochs: log.n_epochs,
      bestEpoch: best.epoch ?? null,
      bestValue: bestKey ? `${fmt(best[bestKey], 3)} ${bestKey === "val_rmse" ? "°C val. RMSE" : bestKey.replace(/_/g, " ")}` : fmt(null),
      seconds: numbers(log, "epoch_seconds").reduce<number>((acc, v) => acc + (v ?? 0), 0),
      gpu: Math.max(0, ...numbers(log, "peak_gpu_mb").map((v) => v ?? 0)),
    };
  });
  type SummaryRow = (typeof summaryRows)[number];
  const summaryColumns: Column<SummaryRow>[] = [
    { key: "name", label: "Stage", render: (r) => r.name },
    { key: "epochs", label: "Epochs run", align: "right", render: (r) => String(r.epochs) },
    { key: "best", label: "Kept epoch", align: "right", render: (r) => (r.bestEpoch != null ? String(r.bestEpoch) : fmt(null)) },
    { key: "value", label: "Best validation score", align: "right", render: (r) => r.bestValue },
    { key: "time", label: "Wall time", align: "right", render: (r) => fmtDuration(r.seconds) },
    { key: "gpu", label: "Peak GPU memory", align: "right", render: (r) => (r.gpu > 0 ? `${fmt(r.gpu / 1024, 1)} GB` : fmt(null)) },
  ];

  return (
    <>
      <Panel title="Training stages" subtitle="from the logs of this run">
        <DataTable columns={summaryColumns} rows={summaryRows} rowKey={(r) => r.stem} caption="Training stages" dense />
      </Panel>
      <div className="twocol gap-top">
      {valRmse.length > 0 && (
        <Panel title="Validation RMSE during training" subtitle="reconstruction model and its ablations · lower is better">
          <Legend items={valRmse.map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
          <XYChart
            series={valRmse}
            x={{ label: "Epoch", domain: [1, maxEpoch(valRmse)], format: (v) => fmt(v, 0) }}
            y={{ label: "Validation RMSE (°C)", domain: rmseDomain }}
            hover="x"
            height={300}
            reference={main?.best?.epoch != null ? { axis: "x", value: main.best.epoch, label: "kept" } : null}
            hoverTitle={(v) => `Epoch ${v}`}
            ariaLabel="Validation RMSE by epoch for the model and its ablations"
          />
        </Panel>
      )}
      {lossSeries.length > 0 && (
        <Panel title="Reconstruction loss" subtitle="main model · standardised anomaly units · a widening gap between the two lines is overfitting">
          <Legend items={[...lossSeries].reverse().map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
          <XYChart
            series={lossSeries}
            x={{ label: "Epoch", domain: [1, maxEpoch(lossSeries)], format: (v) => fmt(v, 0) }}
            y={{ label: "Loss", domain: padRange(0, Math.max(...lossSeries.flatMap((s) => s.points.map((p) => p.y ?? 0))), 0.06) }}
            hover="x"
            height={300}
            reference={main?.best?.epoch != null ? { axis: "x", value: main.best.epoch, label: "kept" } : null}
            hoverTitle={(v) => `Epoch ${v}`}
            ariaLabel="Training and validation loss of the reconstruction model by epoch"
          />
        </Panel>
      )}
      {preSeries.length > 0 && pre && (
        <Panel title="Pretraining: masked surface reconstruction" subtitle="error on hidden blocks · the dotted line is what filling the holes with the mean would score">
          <Legend items={[...preSeries].reverse().map((s) => ({ key: s.key, label: s.label, style: s.style }))} />
          <XYChart
            series={preSeries}
            x={{ label: "Epoch", domain: [1, maxEpoch(preSeries)], format: (v) => fmt(v, 0) }}
            y={{ label: "Masked MSE (standardised)", domain: padRange(0, Math.max(...preSeries.flatMap((s) => s.points.map((p) => p.y ?? 0))), 0.06) }}
            hover="x"
            height={300}
            reference={pre.best?.epoch != null ? { axis: "x", value: pre.best.epoch, label: "kept" } : null}
            hoverTitle={(v) => `Epoch ${v}`}
            ariaLabel="Pretraining loss by epoch against the mean-fill baseline"
          />
        </Panel>
      )}
      {heat && epochs.length > 1 && (
        <Panel title="Validation RMSE by depth, as training proceeds" subtitle="main model · change relative to the first epoch, so quiet deep levels and the thermocline share one scale">
          <DepthHeatmap
            values={heat.rel}
            depths={depths}
            xValues={epochs}
            xKind="index"
            xLabel="Epoch"
            cmap="balance"
            vmin={-heat.lim}
            vmax={heat.lim}
            height={266}
            markX={main?.best?.epoch ?? null}
            formatValue={(v) => `${v > 0 ? "+" : ""}${fmt(v, 0)} % vs epoch 1`}
            ariaLabel="Validation RMSE by depth and epoch, relative to the first epoch"
          />
          <div className="stack__bars">
            <Colorbar spec={{ cmap: "balance", vmin: -heat.lim, vmax: heat.lim, units: "%", label: "RMSE vs epoch 1", extend: "both", signed: true }} />
          </div>
        </Panel>
      )}
      </div>
    </>
  );
}
