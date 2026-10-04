import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/api/client";
import type { RunSummary } from "@/api/types";
import { methodStyle } from "@/lib/methods";
import { runCaveats } from "@/lib/narrative";
import { Legend } from "./charts/marks";
import { MethodBars } from "./charts/MethodBars";
import { MetricsTable, bestMethod } from "./charts/MetricsTable";
import { XYChart, timeTicks } from "./charts/XYChart";
import { DepthRail } from "./controls/DepthRail";
import { Colorbar } from "./map/Colorbar";
import { Shell } from "./shell/Shell";
import { ErrorState, Segmented } from "./ui/primitives";

const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

function run(over: Partial<RunSummary>): RunSummary {
  return {
    name: "demo",
    run_name: "demo",
    data_source: "real",
    updated: null,
    period: {},
    split: { train: { start: "2018-01-01", end: "2022-12-31" }, test: { start: "2024-01-01", end: "2024-12-15" } },
    label: "Demo run, 2018 to 2024",
    description: "Seven years of real data.",
    n_train_days: 1826,
    n_val_days: 365,
    n_test_days: 350,
    n_harmonic_terms: 5,
    grid: { resolution: 0.25, n_lat: 100, n_lon: 240, n_depth: 15, lat: [], lon: [], depths: [] },
    artefacts: { predictions: true, metrics_glorys: true, metrics_argo: true, argo_matchups: true, maps: true, embeddings: true, report: true, training_logs: [], n_figures: 0, n_product_files: 0 },
    n_prediction_days: 350,
    ...over,
  } as RunSummary;
}

describe("Segmented", () => {
  it("marks the active option and reports a change", () => {
    const onChange = vi.fn();
    render(
      <Segmented
        label="Quantity"
        value="temp"
        onChange={onChange}
        options={[
          { value: "temp", label: "Temperature" },
          { value: "anom", label: "Anomaly" },
        ]}
      />,
    );
    expect(screen.getByRole("group", { name: "Quantity" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Temperature" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Anomaly" }));
    expect(onChange).toHaveBeenCalledWith("anom");
    fireEvent.click(screen.getByRole("button", { name: "Temperature" }));
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});

describe("ErrorState", () => {
  it("shows a missing artefact as missing, with the server's hint and no retry", () => {
    render(<ErrorState error={new ApiError(404, "run 'x' has no embeddings (run `oceanembed embed`)", "/api/x")} what="The embedding" onRetry={() => undefined} />);
    expect(screen.getByText("The embedding is not available for this run")).toBeTruthy();
    expect(screen.getByText(/oceanembed embed/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("offers a retry for a real failure", () => {
    const retry = vi.fn();
    render(<ErrorState error={new ApiError(500, "internal error", "/api/x")} what="the metrics" onRetry={retry} />);
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).toContain("Could not load the metrics");
  });
});

describe("Shell honesty bands", () => {
  it("shows the synthetic band for a synthetic run", () => {
    const r = run({ data_source: "synthetic", note: "SYNTHETIC DATA" });
    render(
      <Shell runs={[r]} run={r} caveats={runCaveats(r)}>
        <p>content</p>
      </Shell>,
    );
    const band = screen.getByRole("note", { name: "Synthetic data warning" });
    expect(band.textContent).toContain("not scientific skill");
    expect(screen.queryByRole("note", { name: "Short training period warning" })).toBeNull();
  });

  it("shows the short-training band from the reported training days, whatever the run is called", () => {
    const r = run({ name: "final_run", n_train_days: 46, n_harmonic_terms: 1 });
    render(
      <Shell runs={[r]} run={r} caveats={runCaveats(r)}>
        <p>content</p>
      </Shell>,
    );
    expect(screen.getByRole("note", { name: "Short training period warning" }).textContent).toContain("46 days");
    expect(screen.queryByRole("note", { name: "Synthetic data warning" })).toBeNull();
  });

  it("shows neither band for a long real run, and lists the views", () => {
    const r = run({});
    render(
      <Shell runs={[r]} run={r} caveats={runCaveats(r)}>
        <p>content</p>
      </Shell>,
    );
    expect(screen.queryAllByRole("note")).toHaveLength(0);
    const nav = screen.getByRole("navigation", { name: "Views" });
    // the product's views only; the Research area is linked from the colophon
    expect([...nav.querySelectorAll("a")].map((a) => a.textContent)).toEqual(["Overview", "Explorer", "Accuracy", "Data & downloads"]);
    expect(nav.querySelector('[aria-current="page"]')?.textContent).toBe("Overview");
    expect(nav.textContent).not.toMatch(/Research|Live/);
    const research = screen.getByRole("link", { name: /^Research/ });
    expect(research.closest("footer")).not.toBeNull();
    expect(research.getAttribute("href")).toMatch(/^\/research/);
    const select = screen.getByLabelText("Run") as HTMLSelectElement;
    expect(select.value).toBe("demo");
    // the selector shows the run's display name; its value stays the URL identifier
    expect(select.selectedOptions[0].textContent).toBe("Demo run, 2018 to 2024");
    expect(screen.getByText(/350 days/)).toBeTruthy();
  });
});

describe("charts", () => {
  it("draws one path and one legend entry per series, and labels the axes with units", () => {
    const series = ["model", "ridge", "climatology"].map((k, i) => ({
      key: k,
      label: k,
      style: methodStyle(k),
      points: DEPTHS.map((d) => ({ x: 1 + i + d / 1000, y: d })),
    }));
    const { container } = render(
      <>
        <Legend items={series.map((s) => ({ key: s.key, label: s.label, style: s.style }))} band="pooled range" />
        <XYChart
          series={series}
          x={{ label: "RMSE (°C)", domain: [0, 5] }}
          y={{ label: "Depth (m)", domain: [0, 1000], scale: "depth", ticks: DEPTHS }}
          hover="y"
          ariaLabel="RMSE by depth"
        />
      </>,
    );
    const svg = screen.getByRole("img", { name: "RMSE by depth" });
    expect(svg.querySelectorAll('path[fill="none"][stroke-width]').length).toBeGreaterThanOrEqual(3);
    expect(svg.textContent).toContain("RMSE (°C)");
    expect(svg.textContent).toContain("Depth (m)");
    expect(container.querySelectorAll(".legend__item")).toHaveLength(4);
    // depth increases downward: the surface is drawn above 1000 m
    const ticks = [...svg.querySelectorAll("text.chart__tick")].filter((t) => t.getAttribute("text-anchor") === "end");
    const y = (label: string) => Number(ticks.find((t) => t.textContent === label)?.getAttribute("y"));
    expect(y("0")).toBeLessThan(y("1000"));
  });

  it("puts time ticks on days for short periods and on months for long ones", () => {
    const day = 86400000;
    const mar1 = Date.UTC(2024, 2, 1);
    const short = timeTicks(mar1, mar1 + 30 * day, 6);
    expect(short.length).toBeGreaterThan(2);
    expect(short.length).toBeLessThanOrEqual(7);
    const long = timeTicks(Date.UTC(2024, 0, 1), Date.UTC(2024, 11, 15), 6);
    expect(long.every((t) => new Date(t).getUTCDate() === 1)).toBe(true);
    expect(long.length).toBeLessThanOrEqual(6);
  });

  it("labels every bar with its method and value", () => {
    render(
      <MethodBars
        label="Pooled RMSE"
        unit="°C"
        rows={[
          { key: "model", label: "OceanEmbed", value: 2.28, style: methodStyle("model"), emphasis: true },
          { key: "climatology", label: "Climatology", value: 3.39, style: methodStyle("climatology") },
          { key: "none", label: "Missing", value: null, style: methodStyle("ridge") },
        ]}
      />,
    );
    const table = screen.getByRole("table", { name: "Pooled RMSE" });
    expect(table.textContent).toContain("OceanEmbed");
    expect(table.textContent).toContain("2.28");
    expect(table.textContent).toContain("3.39");
    expect(table.textContent).toContain("–");
  });

  it("underlines the best candidate per metric and never the GLORYS reference", () => {
    const blocks = {
      model: { rmse: 1.5, bias: -0.14, corr_anom: 0.74, n: 100 },
      ridge: { rmse: 1.7, bias: -0.1, corr_anom: 0.71, n: 100 },
      glorys: { rmse: 0.2, bias: 0.01, corr_anom: 1.0, n: 100 },
    };
    const methods = ["model", "ridge", "glorys"];
    expect(bestMethod(blocks, methods, "rmse", ["glorys"])).toBe("model");
    expect(bestMethod(blocks, methods, "bias", ["glorys"])).toBe("ridge");
    expect(bestMethod(blocks, methods, "corr_anom", ["glorys"])).toBe("model");
    const { container } = render(
      <MetricsTable blocks={blocks} methods={methods} styleOf={(k) => methodStyle(k)} labelOf={(k) => k} metrics={["rmse", "corr_anom"]} caption="t" />,
    );
    const best = [...container.querySelectorAll(".best")].map((n) => n.textContent);
    expect(best).toEqual(["1.50", "0.74"]);
  });
});

describe("Colorbar", () => {
  it("states its range and units, and signs a diverging scale", () => {
    render(<Colorbar spec={{ cmap: "balance", vmin: -2, vmax: 2, units: "°C", signed: true }} />);
    const bar = screen.getByRole("img");
    expect(bar.getAttribute("aria-label")).toContain("−2 to +2 °C");
    expect(bar.textContent).toContain("+1");
    expect(bar.textContent).toContain("°C");
  });

  it("never prints more tick labels than asked for", () => {
    const { container } = render(<Colorbar spec={{ cmap: "balance", vmin: -1.7, vmax: 1.7, units: "°C", signed: true }} ticks={3} />);
    const labels = [...container.querySelectorAll(".colorbar__tick")].map((n) => n.textContent);
    expect(labels.length).toBeLessThanOrEqual(4);
    expect(labels).toContain("0");
  });
});

describe("DepthRail", () => {
  it("is a vertical slider that steps one level per arrow key, deeper downward", () => {
    const onChange = vi.fn();
    render(<DepthRail depths={DEPTHS} index={7} onChange={onChange} />);
    const slider = screen.getByRole("slider", { name: "Depth level" });
    expect(slider.getAttribute("aria-orientation")).toBe("vertical");
    expect(slider.getAttribute("aria-valuetext")).toBe("100\u00A0m");
    fireEvent.keyDown(slider, { key: "ArrowDown" });
    expect(onChange).toHaveBeenLastCalledWith(8);
    fireEvent.keyDown(slider, { key: "ArrowUp" });
    expect(onChange).toHaveBeenLastCalledWith(6);
    fireEvent.keyDown(slider, { key: "End" });
    expect(onChange).toHaveBeenLastCalledWith(14);
    fireEvent.keyDown(slider, { key: "a" });
    expect(onChange).toHaveBeenCalledTimes(3);
  });
});
