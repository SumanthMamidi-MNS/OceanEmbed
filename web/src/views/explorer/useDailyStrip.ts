/** The daily RMSE of one estimate and of climatology at the selected depth, aligned with the run's days (for the timeline). */
import { useMemo } from "react";
import { useGlorysMetrics } from "@/api/queries";
import { useRunContext } from "@/state/runContext";

export interface DailyStrip {
  estimate: (number | null)[] | null;
  climatology: (number | null)[] | null;
}

export function useDailyStrip(method: string): DailyStrip | null {
  const { run, dates, depths, depthIndex } = useRunContext();
  const metrics = useGlorysMetrics(run.name, run.artefacts.metrics_glorys);
  return useMemo(() => {
    const m = metrics.data;
    const daily = m?.daily?.rmse ? { dates: m.daily.dates, depths: m.daily.depths, methods: m.daily.rmse } : m?.daily_rmse;
    if (!daily) return null;
    const idx = new Map(daily.dates.map((d, i) => [d, i]));
    const k = daily.depths.indexOf(depths[depthIndex]);
    if (k < 0) return null;
    const pick = (key: string) => {
      const rows = daily.methods[key];
      if (!rows) return null;
      return dates.map((d) => {
        const i = idx.get(d);
        return i == null ? null : (rows[i]?.[k] ?? null);
      });
    };
    return { estimate: pick(method), climatology: pick("climatology") };
  }, [metrics.data, dates, depths, depthIndex, method]);
}
