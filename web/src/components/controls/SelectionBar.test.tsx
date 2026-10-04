import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { RunDetail, RunSummary } from "@/api/types";
import { currentUrlState, navigate } from "@/state/router";
import { RunProvider } from "@/state/runContext";
import { SelectionBar } from "./SelectionBar";

const DEPTHS = [0, 50, 100, 200];
const DAYS = ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"];

const summary = {
  name: "demo",
  run_name: "demo",
  label: "Demo run",
  description: "",
  data_source: "real",
  updated: null,
  period: {},
  split: {},
  grid: { resolution: 1, n_lat: 2, n_lon: 2, n_depth: 4, lat: [], lon: [], depths: DEPTHS },
  artefacts: { predictions: true, metrics_glorys: false, metrics_argo: false, argo_matchups: false, maps: false, embeddings: false, report: false, training_logs: [], n_figures: 0, n_product_files: 0 },
  n_prediction_days: DAYS.length,
  n_train_days: 800,
  n_test_days: DAYS.length,
  n_harmonic_terms: 5,
} as unknown as RunSummary;

function detail(fieldMethods: string[]): RunDetail {
  return {
    summary,
    grid: { resolution: 1, n_lat: 2, n_lon: 2, n_depth: 4, lat_values: [5.5, 6.5], lon_values: [60.5, 61.5], depth_values: DEPTHS, lat_edges: [5, 6, 7], lon_edges: [60, 61, 62] },
    basins: [],
    prediction_dates: DAYS,
    methods: [
      { key: "model", label: "OceanEmbed (pretrained encoder)", kind: "model" },
      { key: "ridge", label: "Ridge regression", kind: "baseline" },
    ],
    field_methods: fieldMethods.map((key) => ({ key, label: key, kind: "model", n_days: DAYS.length, first: DAYS[0], last: DAYS[4] })),
    model: {},
    training_summary: {},
    products: [],
    counts: {},
  } as unknown as RunDetail;
}

function renderBar(fieldMethods: string[], props: Parameters<typeof SelectionBar>[0] = {}) {
  return render(
    <RunProvider runs={[summary]} run={summary} detail={detail(fieldMethods)} dates={undefined} mask={undefined}>
      <SelectionBar {...props} />
    </RunProvider>,
  );
}

afterEach(() => {
  navigate({ view: "overview", run: null, date: null, depth: null, lat: null, lon: null, est: null }, "push");
});

describe("SelectionBar", () => {
  it("steps the day and the depth and writes them to the shared selection", () => {
    navigate({ date: "2024-03-02", depth: 50 });
    renderBar(["model"]);
    const bar = screen.getByRole("region", { name: /^Selection/ });
    expect((within(bar).getByLabelText("Day") as HTMLSelectElement).selectedOptions[0].textContent).toContain("2\u00A0Mar\u00A02024");
    fireEvent.click(within(bar).getByRole("button", { name: "Next day" }));
    expect(currentUrlState().date).toBe("2024-03-03");
    fireEvent.click(within(bar).getByRole("button", { name: "Deeper level" }));
    expect(currentUrlState().depth).toBe(100);
    fireEvent.change(within(bar).getByLabelText("Depth"), { target: { value: "3" } });
    expect(currentUrlState().depth).toBe(200);
  });

  it("offers no estimate on the product views, whatever methods the run has", () => {
    renderBar(["model", "ridge", "mlp"]);
    expect(screen.queryByRole("group", { name: "Estimate shown" })).toBeNull();
    // a baseline cannot be selected from a product link either
    navigate({ est: "ridge" });
    expect(currentUrlState().est).toBeNull();
  });

  it("offers in the Research area only the estimates the run has, and none when there is a single one", () => {
    navigate({ view: "research_maps" }, "push");
    const one = renderBar(["model"]);
    expect(screen.queryByRole("group", { name: "Estimate shown" })).toBeNull();
    one.unmount();
    renderBar(["model", "ridge"]);
    const group = screen.getByRole("group", { name: "Estimate shown" });
    expect(within(group).getAllByRole("button").map((b) => b.textContent)).toEqual(["OceanEmbed", "Ridge"]);
    fireEvent.click(within(group).getByRole("button", { name: "Ridge" }));
    expect(currentUrlState().est).toBe("ridge");
    // the main model is the default and is not written to the link
    fireEvent.click(within(screen.getByRole("group", { name: "Estimate shown" })).getByRole("button", { name: "OceanEmbed" }));
    expect(currentUrlState().est).toBeNull();
  });

  it("shows only the groups a view asks for, and can limit the days", () => {
    navigate({ date: "2024-03-03" });
    renderBar(["model", "ridge"], { depth: false, estimate: false, days: ["2024-03-01", "2024-03-03", "2024-03-05"] });
    expect(screen.queryByLabelText("Depth")).toBeNull();
    expect(screen.queryByRole("group", { name: "Estimate shown" })).toBeNull();
    expect((screen.getByLabelText("Day") as HTMLSelectElement).options).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "Next day" }));
    expect(currentUrlState().date).toBe("2024-03-05");
    expect((screen.getByRole("button", { name: "Next day" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("ignores an estimate the run does not have", () => {
    navigate({ view: "research_scores", est: "model_other" }, "push");
    renderBar(["model", "ridge"]);
    const pressed = within(screen.getByRole("group", { name: "Estimate shown" })).getByRole("button", { pressed: true });
    expect(pressed.textContent).toBe("OceanEmbed");
  });
});
