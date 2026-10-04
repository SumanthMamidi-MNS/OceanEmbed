/**
 * Test-year switch for a run whose test period spans several calendar years: the pooled result
 * of all years, or one year at a time. The choice is a per-view URL option (`yr`), so every
 * switch on a view shows and changes the same thing. Renders nothing for a single-year period.
 */
import { Segmented } from "@/components/ui/primitives";
import { useUrlState } from "@/state/router";

/** The selected test year (null: all years together), validated against the years the metrics have. */
export function useYear(years: readonly string[]): string | null {
  const [url] = useUrlState();
  return url.opts.yr && years.includes(url.opts.yr) ? url.opts.yr : null;
}

export function YearSwitch({ years, showLabel = true }: { years: readonly string[]; showLabel?: boolean }) {
  const [, setUrl] = useUrlState();
  const year = useYear(years);
  if (years.length < 2) return null;
  return (
    <Segmented
      label="Test year"
      showLabel={showLabel}
      size="sm"
      value={year ?? "all"}
      onChange={(v) => setUrl({ opts: { yr: v === "all" ? null : v } })}
      options={[
        { value: "all", label: years.length === 2 ? "Both years" : "All years", title: `The whole test period: ${years.join(", ")} together` },
        ...years.map((y) => ({ value: y, label: y, title: `Scores of the test days of ${y} only` })),
      ]}
    />
  );
}
