/**
 * Client-side cache of daily 3-D volumes (float32, ~1.4 MB each for the full grid). Depth
 * scrubbing re-colours a cached volume; day stepping reads the cache and prefetches neighbours.
 * A small LRU bounds memory. A prediction volume belongs to a method (`model`, `ridge` or an
 * ablation); the GLORYS target and the climatology do not depend on it.
 */
import { useEffect, useState } from "react";
import { decodeVolume, type Volume } from "./binary";
import { ApiError, errorMessage, getBinary, runPath } from "./client";
import type { FieldKind } from "./types";

/** The run's own reconstruction: the default `method` of every field endpoint. */
export const MAIN_METHOD = "model";

type Entry = { promise: Promise<Volume>; volume?: Volume; bytes: number };

export class VolumeCache {
  private readonly entries = new Map<string, Entry>();
  private readonly maxBytes: number;

  constructor(maxBytes = 200 * 1024 * 1024) {
    this.maxBytes = maxBytes;
  }

  static key(run: string, date: string, kind: FieldKind, method: string = MAIN_METHOD): string {
    // only the kinds built from a prediction depend on the method
    const m = kind === "target" || kind === "climatology" || kind === "anomaly_target" ? MAIN_METHOD : method;
    return `${run}|${date}|${kind}|${m}`;
  }

  peek(run: string, date: string, kind: FieldKind, method: string = MAIN_METHOD): Volume | undefined {
    const key = VolumeCache.key(run, date, kind, method);
    const e = this.entries.get(key);
    if (e?.volume) this.touch(key, e);
    return e?.volume;
  }

  get(run: string, date: string, kind: FieldKind, method: string = MAIN_METHOD): Promise<Volume> {
    const key = VolumeCache.key(run, date, kind, method);
    const hit = this.entries.get(key);
    if (hit) {
      this.touch(key, hit);
      return hit.promise;
    }
    const entry: Entry = { promise: this.load(run, date, kind, method), bytes: 0 };
    entry.promise.then(
      (volume) => {
        entry.volume = volume;
        entry.bytes = volume.data.byteLength;
        this.evict();
      },
      () => {
        // failed loads are not cached, so a later request retries
        if (this.entries.get(key) === entry) this.entries.delete(key);
      },
    );
    this.entries.set(key, entry);
    return entry.promise;
  }

  /** Fire-and-forget load (errors are swallowed; the foreground request will surface them). */
  prefetch(run: string, date: string, kind: FieldKind, method: string = MAIN_METHOD): void {
    this.get(run, date, kind, method).catch(() => undefined);
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }

  private async load(run: string, date: string, kind: FieldKind, method: string): Promise<Volume> {
    // the default method is not sent, so the main product keeps its canonical URL
    const res = await getBinary(runPath(run, "/fields"), { date, kind, format: "f32", method: method === MAIN_METHOD ? undefined : method });
    return decodeVolume(res.headers, await res.arrayBuffer());
  }

  private touch(key: string, entry: Entry): void {
    this.entries.delete(key);
    this.entries.set(key, entry);
  }

  private evict(): void {
    let total = 0;
    for (const e of this.entries.values()) total += e.bytes;
    for (const [key, e] of this.entries) {
      if (total <= this.maxBytes) break;
      if (!e.volume) continue;
      this.entries.delete(key);
      total -= e.bytes;
    }
  }
}

export const volumeCache = new VolumeCache();

export interface DayVolumes {
  run: string;
  date: string;
  /** one volume per requested method (`model`, `ridge`, an ablation) */
  estimates: Readonly<Record<string, Volume>>;
  /** the method the views show first: the first of the requested methods */
  primary: string;
  /** `estimates[primary]` */
  prediction: Volume;
  climatology: Volume;
  /** null when the day has no GLORYS target */
  target: Volume | null;
}

const ONLY_MAIN: readonly string[] = [MAIN_METHOD];

function assemble(
  run: string,
  date: string,
  methods: readonly string[],
  volumes: readonly Volume[],
  climatology: Volume,
  target: Volume | null,
): DayVolumes {
  const estimates: Record<string, Volume> = {};
  methods.forEach((m, i) => {
    estimates[m] = volumes[i];
  });
  return { run, date, estimates, primary: methods[0], prediction: volumes[0], climatology, target };
}

/** The day if every volume it needs is already in memory, else null (never starts a request). */
export function peekDay(run: string, date: string, wantTarget: boolean, methods: readonly string[] = ONLY_MAIN): DayVolumes | null {
  const volumes: Volume[] = [];
  for (const m of methods) {
    const v = volumeCache.peek(run, date, "prediction", m);
    if (!v) return null;
    volumes.push(v);
  }
  const climatology = volumeCache.peek(run, date, "climatology");
  const target = wantTarget ? volumeCache.peek(run, date, "target") : null;
  if (!climatology || (wantTarget && !target)) return null;
  return assemble(run, date, methods, volumes, climatology, target ?? null);
}

export async function loadDay(run: string, date: string, wantTarget: boolean, methods: readonly string[] = ONLY_MAIN): Promise<DayVolumes> {
  const [volumes, climatology, target] = await Promise.all([
    Promise.all(methods.map((m) => volumeCache.get(run, date, "prediction", m))),
    volumeCache.get(run, date, "climatology"),
    wantTarget
      ? volumeCache.get(run, date, "target").catch((err: unknown) => {
          // the day has no target after all: show the prediction alone
          if (err instanceof ApiError && err.isMissing) return null;
          throw err;
        })
      : Promise.resolve(null),
  ]);
  return assemble(run, date, methods, volumes, climatology, target);
}

export function prefetchDay(run: string, date: string, wantTarget: boolean, methods: readonly string[] = ONLY_MAIN): void {
  for (const m of methods) volumeCache.prefetch(run, date, "prediction", m);
  volumeCache.prefetch(run, date, "climatology");
  if (wantTarget) volumeCache.prefetch(run, date, "target");
}

export interface DayVolumesState {
  /** the most recent fully loaded day (kept on screen while the next one loads) */
  day: DayVolumes | null;
  /** true while `day` is not the requested day (or not the requested methods) yet */
  loading: boolean;
  error: string | null;
}

/**
 * Volumes of one day for the given methods (the first one is the day's `primary`). While a new
 * day or method loads, the previous one stays on screen (no blank frame); neighbouring days are
 * prefetched so stepping and animation stay instant.
 */
export function useDayVolumes(
  run: string | null,
  date: string | null,
  wantTarget: boolean,
  neighbours: readonly { date: string; wantTarget: boolean }[] = [],
  methods: readonly string[] = ONLY_MAIN,
): DayVolumesState {
  const [state, setState] = useState<DayVolumesState>({ day: null, loading: false, error: null });
  const methodKey = methods.join(",");

  useEffect(() => {
    if (!run || !date) {
      setState({ day: null, loading: false, error: null });
      return;
    }
    const wanted = methodKey.split(",");
    const cached = peekDay(run, date, wantTarget, wanted);
    if (cached) {
      setState({ day: cached, loading: false, error: null });
      return;
    }
    let stale = false;
    setState((prev) => ({ day: prev.day && prev.day.run === run ? prev.day : null, loading: true, error: null }));
    loadDay(run, date, wantTarget, wanted).then(
      (day) => {
        if (!stale) setState({ day, loading: false, error: null });
      },
      (err: unknown) => {
        if (!stale) setState({ day: null, loading: false, error: errorMessage(err) });
      },
    );
    return () => {
      stale = true;
    };
  }, [run, date, wantTarget, methodKey]);

  const neighbourKey = neighbours.map((n) => `${n.date}:${n.wantTarget ? 1 : 0}`).join(",");
  useEffect(() => {
    if (!run || !neighbourKey) return;
    const wanted = methodKey.split(",");
    for (const item of neighbourKey.split(",")) {
      const [d, t] = item.split(":");
      prefetchDay(run, d, t === "1", wanted);
    }
  }, [run, neighbourKey, methodKey]);

  return state;
}
