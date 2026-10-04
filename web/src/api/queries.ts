/** React Query hooks, one per endpoint. Every hook is keyed by the run so switching runs is clean. */
import { QueryClient, keepPreviousData, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { coastlineSegments } from "@/lib/geo";
import { flatten2d } from "@/lib/stats";
import { decodePackedMask } from "./binary";
import { ApiError, getJson, runPath } from "./client";
import type {
  ArgoProfileDetail,
  ArgoProfilesResponse,
  ColourRange,
  CompareResponse,
  DatesResponse,
  EmbeddingResponse,
  EmbeddingSimilarityResponse,
  ExperimentsResponse,
  Health,
  MapResponse,
  MapsIndex,
  MaskResponse,
  MatchupsResponse,
  MetricsResponse,
  LiveResponse,
  ProductResponse,
  ProfileResponse,
  RangeInfo,
  RangesResponse,
  ReportResponse,
  RunDetail,
  RunSummary,
  SectionResponse,
  SurfaceResponse,
  TimeseriesResponse,
  TrainingResponse,
} from "./types";

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
        // a 4xx is an answer (missing artefact, bad parameter), not a failure worth retrying
        retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 1,
      },
    },
  });
}

type Q<T> = UseQueryResult<T, Error>;

export function useHealth(): Q<Health> {
  return useQuery({ queryKey: ["health"], queryFn: ({ signal }) => getJson<Health>("/health", undefined, signal) });
}

export function useRuns(): Q<RunSummary[]> {
  return useQuery({
    queryKey: ["runs"],
    queryFn: async ({ signal }) => (await getJson<{ runs: RunSummary[] }>("/runs", undefined, signal)).runs,
  });
}

export function useRun(run: string | null): Q<RunDetail> {
  return useQuery({
    queryKey: ["run", run],
    queryFn: ({ signal }) => getJson<RunDetail>(runPath(run!), undefined, signal),
    enabled: !!run,
  });
}

export function useDates(run: string | null): Q<DatesResponse> {
  return useQuery({
    queryKey: ["dates", run],
    queryFn: ({ signal }) => getJson<DatesResponse>(runPath(run!, "/dates"), undefined, signal),
    enabled: !!run,
  });
}

/** Colour ranges per depth that hold over the whole predicted period (for "hold range"). */
export function useRunRanges(run: string | null, enabled = true): Q<RangesResponse> {
  return useQuery({
    queryKey: ["ranges", run],
    queryFn: ({ signal }) => getJson<RangesResponse>(runPath(run!, "/ranges"), undefined, signal),
    enabled: enabled && !!run,
    staleTime: 10 * 60_000,
  });
}

export interface OceanMask {
  nz: number;
  ny: number;
  nx: number;
  depths: number[];
  /** one 0/1 byte per cell, per depth level */
  levels: Uint8Array[];
  nOcean: number[];
  /** coastline of the surface level, cell-corner units (see lib/geo) */
  coast: Float32Array;
}

function decodeMask(res: MaskResponse): OceanMask {
  const [nz, ny, nx] = res.shape;
  const levels = res.data.map((b64) => decodePackedMask(b64, ny * nx));
  return { nz, ny, nx, depths: res.depths, levels, nOcean: res.n_ocean, coast: coastlineSegments(levels[0], nx, ny) };
}

export function useOceanMask(run: string | null): Q<OceanMask> {
  return useQuery({
    queryKey: ["mask", run],
    queryFn: async ({ signal }) => decodeMask(await getJson<MaskResponse>(runPath(run!, "/mask"), undefined, signal)),
    enabled: !!run,
    staleTime: 10 * 60_000,
  });
}

export interface SurfaceFieldData {
  key: string;
  longName: string;
  units: string;
  data: Float32Array;
  range: ColourRange;
}

export interface SurfaceData {
  date: string;
  ny: number;
  nx: number;
  fields: Record<string, SurfaceFieldData>;
}

function decodeSurface(res: SurfaceResponse): SurfaceData {
  const fields: Record<string, SurfaceFieldData> = {};
  for (const [key, f] of Object.entries(res.fields)) {
    fields[key] = { key, longName: f.long_name, units: f.units, data: flatten2d(f.data), range: f.color_range };
  }
  return { date: res.date, ny: res.shape[0], nx: res.shape[1], fields };
}

export function useSurface(run: string | null, date: string | null): Q<SurfaceData> {
  return useQuery({
    queryKey: ["surface", run, date],
    queryFn: async ({ signal }) =>
      decodeSurface(await getJson<SurfaceResponse>(runPath(run!, "/surface"), { date }, signal)),
    enabled: !!run && !!date,
    placeholderData: keepPreviousData,
  });
}

/** `method` (here and below): `model`, `ridge` or an ablation key; one of the run's `field_methods`. */
export function useProfile(run: string | null, date: string | null, lat: number | null, lon: number | null, method = "model"): Q<ProfileResponse> {
  return useQuery({
    queryKey: ["profile", run, date, lat, lon, method],
    queryFn: ({ signal }) => getJson<ProfileResponse>(runPath(run!, "/profile"), { date, lat, lon, method }, signal),
    enabled: !!run && !!date && lat != null && lon != null,
    placeholderData: keepPreviousData,
  });
}

export function useSection(
  run: string | null,
  date: string | null,
  along: "zonal" | "meridional",
  lat: number | null,
  lon: number | null,
  method = "model",
): Q<SectionResponse> {
  const params = along === "zonal" ? { date, lat, method } : { date, lon, method };
  const fixed = along === "zonal" ? lat : lon;
  return useQuery({
    queryKey: ["section", run, date, along, fixed, method],
    queryFn: ({ signal }) => getJson<SectionResponse>(runPath(run!, "/section"), params, signal),
    enabled: !!run && !!date && fixed != null,
    placeholderData: keepPreviousData,
  });
}

export function useTimeseries(run: string | null, lat: number | null, lon: number | null, method = "model", enabled = true): Q<TimeseriesResponse> {
  return useQuery({
    queryKey: ["timeseries", run, lat, lon, method],
    queryFn: ({ signal }) => getJson<TimeseriesResponse>(runPath(run!, "/timeseries"), { lat, lon, method }, signal),
    enabled: enabled && !!run && lat != null && lon != null,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function useGlorysMetrics(run: string | null, enabled = true): Q<MetricsResponse> {
  return useQuery({
    queryKey: ["metrics", "glorys", run],
    queryFn: ({ signal }) => getJson<MetricsResponse>(runPath(run!, "/metrics/glorys"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export function useArgoMetrics(run: string | null, enabled = true): Q<MetricsResponse> {
  return useQuery({
    queryKey: ["metrics", "argo", run],
    queryFn: ({ signal }) => getJson<MetricsResponse>(runPath(run!, "/metrics/argo"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export function useMapsIndex(run: string | null, enabled = true): Q<MapsIndex> {
  return useQuery({
    queryKey: ["maps-index", run],
    queryFn: ({ signal }) => getJson<MapsIndex>(runPath(run!, "/metrics/maps/index"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export interface ErrorMap {
  metric: string;
  method: string;
  label: string;
  depth: number;
  units: string;
  data: Float32Array;
  range: ColourRange;
  /** what the colour range does to this map: how many cells lie outside it, and the true extremes */
  rangeInfo: RangeInfo | null;
}

export function useErrorMap(run: string | null, metric: string | null, method: string | null, depth: number | null): Q<ErrorMap> {
  return useQuery({
    queryKey: ["map", run, metric, method, depth],
    queryFn: async ({ signal }) => {
      const res = await getJson<MapResponse>(runPath(run!, "/metrics/maps"), { metric, method, depth }, signal);
      return {
        metric: res.metric,
        method: res.method,
        label: res.label,
        depth: res.depth,
        units: res.units,
        data: flatten2d(res.data),
        range: res.color_range,
        rangeInfo: res.range_info ?? null,
      };
    },
    enabled: !!run && !!metric && !!method && depth != null,
    placeholderData: keepPreviousData,
  });
}

export function useMatchups(run: string | null, maxPoints = 20000, enabled = true): Q<MatchupsResponse> {
  return useQuery({
    queryKey: ["matchups", run, maxPoints],
    queryFn: ({ signal }) =>
      getJson<MatchupsResponse>(runPath(run!, "/argo/matchups"), { max_points: maxPoints }, signal),
    enabled: enabled && !!run,
  });
}

/** Filters, ordering and paging of the Argo profile list (all server-side). */
export interface ArgoProfileQuery {
  basin?: string | null;
  /** inclusive days, YYYY-MM-DD */
  start?: string | null;
  end?: string | null;
  sort?: "time" | "rmse";
  order?: "asc" | "desc";
  /** method whose per-profile RMSE orders the list when `sort` is "rmse" */
  method?: string | null;
  limit?: number;
  offset?: number;
  /** ask for the page that contains this profile (the offset is then the server's) */
  around?: string | null;
}

/** Most profiles one request asks for: enough for every profile of a multi-year run on one map. */
export const ARGO_MAP_LIMIT = 50000;

export function useArgoProfiles(run: string | null, query: ArgoProfileQuery = {}, enabled = true): Q<ArgoProfilesResponse> {
  const { basin = null, start = null, end = null, sort = "time", order = "asc", method = null, limit = ARGO_MAP_LIMIT, offset = 0, around = null } = query;
  // the method only matters to the server when it orders by RMSE
  const sortMethod = sort === "rmse" ? method : null;
  return useQuery({
    queryKey: ["argo-profiles", run, basin, start, end, sort, order, sortMethod, limit, around ? `around:${around}` : offset],
    queryFn: async ({ signal }) => {
      const params = { basin, start, end, sort, order, method: sortMethod, limit };
      if (around) {
        try {
          return await getJson<ArgoProfilesResponse>(runPath(run!, "/argo/profiles"), { ...params, around }, signal);
        } catch (err) {
          // the profile is not among the matches of the current filters: show the first page instead
          if (!(err instanceof ApiError && err.isMissing)) throw err;
          return getJson<ArgoProfilesResponse>(runPath(run!, "/argo/profiles"), { ...params, offset: 0 }, signal);
        }
      }
      return getJson<ArgoProfilesResponse>(runPath(run!, "/argo/profiles"), { ...params, offset }, signal);
    },
    enabled: enabled && !!run,
    placeholderData: keepPreviousData,
  });
}

export function useArgoProfile(run: string | null, profileId: string | null): Q<ArgoProfileDetail> {
  return useQuery({
    queryKey: ["argo-profile", run, profileId],
    queryFn: ({ signal }) =>
      getJson<ArgoProfileDetail>(runPath(run!, `/argo/profiles/${encodeURIComponent(profileId!)}`), undefined, signal),
    enabled: !!run && !!profileId,
    placeholderData: keepPreviousData,
  });
}

/** PCA view of the embedding; the first three components are the same whatever `nComponents` is. */
export function useEmbedding(run: string | null, date: string | null, enabled = true, nComponents = 3): Q<EmbeddingResponse> {
  return useQuery({
    queryKey: ["embedding", run, date, nComponents],
    queryFn: ({ signal }) => getJson<EmbeddingResponse>(runPath(run!, "/embeddings"), { date, n_components: nComponents }, signal),
    enabled: enabled && !!run && !!date,
    placeholderData: keepPreviousData,
  });
}

/** Cosine similarity of every embedding cell to the cell under a point, over all features. */
export function useEmbeddingSimilarity(
  run: string | null,
  date: string | null,
  lat: number | null,
  lon: number | null,
  center: boolean,
  enabled = true,
): Q<EmbeddingSimilarityResponse> {
  return useQuery({
    queryKey: ["embedding-similar", run, date, lat, lon, center],
    queryFn: ({ signal }) => getJson<EmbeddingSimilarityResponse>(runPath(run!, "/embeddings/similar"), { date, lat, lon, center }, signal),
    enabled: enabled && !!run && !!date && lat != null && lon != null,
    placeholderData: keepPreviousData,
  });
}

export function useTraining(run: string | null, enabled = true): Q<TrainingResponse> {
  return useQuery({
    queryKey: ["training", run],
    queryFn: ({ signal }) => getJson<TrainingResponse>(runPath(run!, "/training"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export function useExperiments(run: string | null, enabled = true): Q<ExperimentsResponse> {
  return useQuery({
    queryKey: ["experiments", run],
    queryFn: ({ signal }) => getJson<ExperimentsResponse>(runPath(run!, "/experiments"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export function useCompare(runs: readonly string[]): Q<CompareResponse> {
  const list = [...runs].sort().join(",");
  return useQuery({
    queryKey: ["compare", list],
    queryFn: ({ signal }) => getJson<CompareResponse>("/compare", { runs: list }, signal),
    enabled: runs.length > 0,
    placeholderData: keepPreviousData,
  });
}

export function useReport(run: string | null, enabled = true): Q<ReportResponse> {
  return useQuery({
    queryKey: ["report", run],
    queryFn: ({ signal }) => getJson<ReportResponse>(runPath(run!, "/report"), undefined, signal),
    enabled: enabled && !!run,
  });
}

/** State of the live nowcast: freshness of the inputs, the window, revisions and the running verification. */
export function useLive(run: string | null, enabled = true): Q<LiveResponse> {
  return useQuery({
    queryKey: ["live", run],
    queryFn: ({ signal }) => getJson<LiveResponse>(runPath(run!, "/live"), undefined, signal),
    enabled: enabled && !!run,
  });
}

export function useProduct(run: string | null, enabled = true): Q<ProductResponse> {
  return useQuery({
    queryKey: ["product", run],
    queryFn: ({ signal }) => getJson<ProductResponse>(runPath(run!, "/product"), undefined, signal),
    enabled: enabled && !!run,
  });
}
