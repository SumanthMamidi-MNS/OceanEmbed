/** Minimal typed fetch layer over the read-only OceanEmbed API (always same-origin `/api`). */

export const API_BASE = "/api";

/** An API error. `detail` is the server's own message (for a 404 it is the hint to show). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly url: string;

  constructor(status: number, detail: string, url: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.url = url;
  }

  /** True when the server says the artefact does not exist (not a failure of the app). */
  get isMissing(): boolean {
    return this.status === 404;
  }
}

export type QueryParams = Record<string, string | number | boolean | null | undefined>;

export function buildUrl(path: string, params?: QueryParams): string {
  const url = `${API_BASE}${path}`;
  if (!params) return url;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === "") continue;
    qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `${url}?${s}` : url;
}

export function runPath(run: string, rest = ""): string {
  return `/runs/${encodeURIComponent(run)}${rest}`;
}

async function toApiError(res: Response, url: string): Promise<ApiError> {
  let detail = `${res.status} ${res.statusText}`.trim();
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const d = (body as { detail: unknown }).detail;
      detail = typeof d === "string" ? d : JSON.stringify(d);
    }
  } catch {
    // not JSON: keep the status text
  }
  return new ApiError(res.status, detail, url);
}

async function request(url: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError(0, "The OceanEmbed API is not reachable. Start it with: oceanembed serve", url);
  }
  if (!res.ok) throw await toApiError(res, url);
  return res;
}

export async function getJson<T>(path: string, params?: QueryParams, signal?: AbortSignal): Promise<T> {
  const url = buildUrl(path, params);
  const res = await request(url, { signal, headers: { Accept: "application/json" } });
  return (await res.json()) as T;
}

export async function getBinary(path: string, params?: QueryParams, signal?: AbortSignal): Promise<Response> {
  const url = buildUrl(path, params);
  return request(url, { signal, headers: { Accept: "application/octet-stream" } });
}

/** Human-readable message for any thrown value. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return String(err);
}
