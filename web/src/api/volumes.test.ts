import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, buildUrl, errorMessage, getJson } from "./client";
import { VolumeCache, loadDay, volumeCache } from "./volumes";

function volumeResponse(value: number, cells = 4): Response {
  const data = new Float32Array(cells).fill(value);
  return new Response(data.buffer, {
    status: 200,
    headers: { "X-Shape": `1,1,${cells}`, "X-Dtype": "float32", "X-Byte-Order": "little", "X-Color-Range": "0,1", "X-Has-Target": "1" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("volume cache", () => {
  it("fetches a volume once and serves it from memory afterwards", async () => {
    const fetchMock = vi.fn(async (_url: string) => volumeResponse(7));
    vi.stubGlobal("fetch", fetchMock);
    const cache = new VolumeCache();
    expect(cache.peek("r", "2024-03-01", "prediction")).toBeUndefined();
    const [a, b] = await Promise.all([cache.get("r", "2024-03-01", "prediction"), cache.get("r", "2024-03-01", "prediction")]);
    expect(a).toBe(b);
    expect(a.data[0]).toBe(7);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toBe("/api/runs/r/fields?date=2024-03-01&kind=prediction&format=f32");
    expect(cache.peek("r", "2024-03-01", "prediction")).toBe(a);
  });

  it("evicts the least recently used volumes beyond its byte budget", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => volumeResponse(1, 4)));
    const cache = new VolumeCache(40); // room for two 16-byte volumes
    await cache.get("r", "d1", "prediction");
    await cache.get("r", "d2", "prediction");
    cache.peek("r", "d1", "prediction"); // d1 is now the most recent
    await cache.get("r", "d3", "prediction");
    expect(cache.peek("r", "d2", "prediction")).toBeUndefined();
    expect(cache.peek("r", "d1", "prediction")).toBeDefined();
    expect(cache.peek("r", "d3", "prediction")).toBeDefined();
  });

  it("does not cache a failed load, so the next request retries", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "date 2020-01-01 is outside the available range" }), { status: 400 }))
      .mockResolvedValueOnce(volumeResponse(3));
    vi.stubGlobal("fetch", fetchMock);
    const cache = new VolumeCache();
    await expect(cache.get("r", "2020-01-01", "prediction")).rejects.toMatchObject({ status: 400 });
    expect(cache.size).toBe(0);
    expect((await cache.get("r", "2020-01-01", "prediction")).data[0]).toBe(3);
  });
});

describe("methods", () => {
  it("asks for another method's prediction with method=, and caches it apart from the main model", async () => {
    const fetchMock = vi.fn(async (url: string) => volumeResponse(url.includes("method=ridge") ? 5 : 7));
    vi.stubGlobal("fetch", fetchMock);
    const cache = new VolumeCache();
    const model = await cache.get("r", "2024-03-01", "prediction");
    const ridge = await cache.get("r", "2024-03-01", "prediction", "ridge");
    expect(model.data[0]).toBe(7);
    expect(ridge.data[0]).toBe(5);
    expect(String(fetchMock.mock.calls[1][0])).toBe("/api/runs/r/fields?date=2024-03-01&kind=prediction&format=f32&method=ridge");
    expect(cache.peek("r", "2024-03-01", "prediction", "ridge")).toBe(ridge);
    expect(cache.peek("r", "2024-03-01", "prediction", "model_scratch")).toBeUndefined();
  });

  it("shares the target and the climatology between methods", () => {
    expect(VolumeCache.key("r", "d", "target", "ridge")).toBe(VolumeCache.key("r", "d", "target"));
    expect(VolumeCache.key("r", "d", "climatology", "ridge")).toBe(VolumeCache.key("r", "d", "climatology"));
    expect(VolumeCache.key("r", "d", "prediction", "ridge")).not.toBe(VolumeCache.key("r", "d", "prediction"));
  });

  it("loads a day for several methods; the first one is the day's primary estimate", async () => {
    volumeCache.clear();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => volumeResponse(url.includes("method=ridge") ? 5 : url.includes("method=model_scratch") ? 6 : url.includes("kind=prediction") ? 7 : 1)),
    );
    const day = await loadDay("multi", "2024-03-01", true, ["ridge", "model", "model_scratch"]);
    expect(day.primary).toBe("ridge");
    expect(day.prediction.data[0]).toBe(5);
    expect(Object.keys(day.estimates)).toEqual(["ridge", "model", "model_scratch"]);
    expect(day.estimates.model.data[0]).toBe(7);
    expect(day.estimates.model_scratch.data[0]).toBe(6);
    expect(day.target?.data[0]).toBe(1);
  });
});

describe("day loading", () => {
  it("shows the prediction alone when the day has no GLORYS target (404)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.includes("kind=target") ? new Response(JSON.stringify({ detail: "no GLORYS target for 2099-01-01" }), { status: 404 }) : volumeResponse(2),
      ),
    );
    const day = await loadDay("solo", "2099-01-01", true);
    expect(day.target).toBeNull();
    expect(day.prediction.data[0]).toBe(2);
    expect(day.climatology.data[0]).toBe(2);
  });
});

describe("client", () => {
  it("builds URLs without empty parameters", () => {
    expect(buildUrl("/runs/x/fields", { date: "2024-03-01", kind: "prediction", depth: null, format: undefined })).toBe(
      "/api/runs/x/fields?date=2024-03-01&kind=prediction",
    );
    expect(buildUrl("/runs")).toBe("/api/runs");
  });

  it("turns an API error body into an ApiError carrying the server's hint", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "run 'nope' not found" }), { status: 404 })));
    const err = await getJson("/runs/nope").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).isMissing).toBe(true);
    expect(errorMessage(err)).toBe("run 'nope' not found");
  });

  it("explains an unreachable API", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    const err = (await getJson("/runs").catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(0);
    expect(err.detail).toMatch(/oceanembed serve/);
  });
});
