/**
 * Checks the hand-written client against the committed OpenAPI snapshot (src/api/openapi.json,
 * refreshed by `npm run gen:api`): every path the client calls must exist as a GET operation, and
 * every query parameter it sends must be declared.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import spec from "./openapi.json";

interface Operation {
  parameters?: { name: string; in: string; required?: boolean }[];
}
const paths = spec.paths as unknown as Record<string, { get?: Operation }>;

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
}

/** Paths the client builds: runPath(run, "/suffix") and getJson("/top-level"). */
function clientPaths(): string[] {
  const text = source("./queries.ts") + source("./volumes.ts");
  const out = new Set<string>();
  for (const m of text.matchAll(/runPath\(run!?(?:,\s*[`"]([^`"]*)[`"])?\)/g)) {
    const suffix = (m[1] ?? "").replace(/\$\{[^}]+\}/g, "{profile_id}");
    out.add(`/api/runs/{run}${suffix}`);
  }
  for (const m of text.matchAll(/getJson<[^(]+>\(\s*"(\/[a-z]+)"/g)) out.add(`/api${m[1]}`);
  return [...out].sort();
}

describe("API client vs OpenAPI snapshot", () => {
  const used = clientPaths();

  it("finds the client's endpoints", () => {
    expect(used.length).toBeGreaterThanOrEqual(20);
    expect(used).toContain("/api/runs/{run}/fields");
    expect(used).toContain("/api/runs/{run}/argo/profiles/{profile_id}");
    expect(used).toContain("/api/compare");
  });

  it.each(clientPaths())("%s is a documented GET endpoint", (path) => {
    expect(Object.keys(paths)).toContain(path);
    expect(paths[path].get).toBeDefined();
  });

  it("sends only declared query parameters", () => {
    const sent: Record<string, string[]> = {
      "/api/runs/{run}/fields": ["date", "kind", "format", "method"],
      "/api/runs/{run}/surface": ["date"],
      "/api/runs/{run}/profile": ["date", "lat", "lon", "method"],
      "/api/runs/{run}/section": ["date", "lat", "lon", "method"],
      "/api/runs/{run}/timeseries": ["lat", "lon", "method"],
      "/api/runs/{run}/metrics/maps": ["metric", "method", "depth"],
      "/api/runs/{run}/argo/matchups": ["max_points"],
      "/api/runs/{run}/argo/profiles": ["basin", "start", "end", "sort", "order", "method", "limit", "offset", "around"],
      "/api/runs/{run}/embeddings": ["date", "n_components"],
      "/api/runs/{run}/embeddings/similar": ["date", "lat", "lon", "center"],
      "/api/compare": ["runs"],
    };
    for (const [path, names] of Object.entries(sent)) {
      const declared = (paths[path].get?.parameters ?? []).filter((p) => p.in === "query").map((p) => p.name);
      for (const name of names) expect(declared, `${path} ?${name}`).toContain(name);
    }
  });

  it("covers the response schemas the client narrows", () => {
    const schemas = Object.keys((spec as unknown as { components: { schemas: Record<string, unknown> } }).components.schemas);
    for (const name of [
      "RunDetail",
      "MetricsResponse",
      "FieldResponse",
      "MaskResponse",
      "EmbeddingResponse",
      "EmbeddingSimilarityResponse",
      "MatchupsResponse",
      "TrainingResponse",
      "FieldMethod",
      "ExtraProduct",
      "RangesResponse",
      "RangeInfo",
    ]) {
      expect(schemas).toContain(name);
    }
  });
});
