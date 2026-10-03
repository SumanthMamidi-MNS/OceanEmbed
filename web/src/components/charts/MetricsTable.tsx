/**
 * Methods x metrics table for one metric block (overall or pooled). Baselines sit next to the
 * model by construction; the anomaly correlation comes before the raw one. The best value of
 * each column among the candidate methods is underlined (the GLORYS reference row is not a
 * candidate: it is the target the model was trained on).
 */
import type { MetricBlock } from "@/api/types";
import { DataTable, type Column } from "@/components/ui/primitives";
import { fmt, fmtCompact, fmtSigned } from "@/lib/format";
import type { MethodStyle } from "@/lib/methods";
import { metricMeta } from "@/lib/metricMeta";
import { Swatch } from "./marks";

export interface MetricsTableProps {
  blocks: Record<string, MetricBlock>;
  methods: readonly string[];
  styleOf: (key: string) => MethodStyle;
  labelOf: (key: string) => string;
  metrics?: readonly string[];
  caption: string;
  /** methods that are references rather than candidates (never marked as best) */
  references?: readonly string[];
}

const DEFAULT_METRICS = ["rmse", "bias", "mae", "corr_anom", "corr_raw", "skill_vs_clim"] as const;

export function bestMethod(
  blocks: Record<string, MetricBlock>,
  methods: readonly string[],
  metric: string,
  references: readonly string[] = [],
): string | null {
  const meta = metricMeta(metric);
  let best: string | null = null;
  let bestV = NaN;
  for (const m of methods) {
    if (references.includes(m)) continue;
    const v = blocks[m]?.[metric];
    if (v == null || !Number.isFinite(v)) continue;
    const score = meta.better === "lower" ? -v : meta.better === "higher" ? v : -Math.abs(v);
    if (best == null || score > bestV) {
      best = m;
      bestV = score;
    }
  }
  return best;
}

export function MetricsTable(props: MetricsTableProps) {
  const { blocks, methods, styleOf, labelOf, metrics = DEFAULT_METRICS, caption, references = ["glorys"] } = props;
  const best = Object.fromEntries(metrics.map((k) => [k, bestMethod(blocks, methods, k, references)]));
  const columns: Column<string>[] = [
    {
      key: "method",
      label: "Method",
      render: (m) => (
        <span className="methodcell">
          <Swatch style={styleOf(m)} width={26} />
          {labelOf(m)}
        </span>
      ),
    },
    ...metrics.map<Column<string>>((k) => {
      const meta = metricMeta(k);
      return {
        key: k,
        label: meta.unit ? `${meta.short} (${meta.unit})` : meta.short,
        title: meta.explain,
        align: "right",
        render: (m) => {
          const v = blocks[m]?.[k];
          const text = k === "bias" ? fmtSigned(v, meta.digits) : fmt(v, meta.digits);
          return best[k] === m && v != null ? <span className="best">{text}</span> : text;
        },
      };
    }),
    { key: "n", label: "n", align: "right", title: "Number of values compared", render: (m) => fmtCompact(blocks[m]?.n) },
  ];
  return <DataTable columns={columns} rows={methods} rowKey={(m) => m} caption={caption} rowClass={(m) => (m === "model" ? "is-lead" : undefined)} />;
}
