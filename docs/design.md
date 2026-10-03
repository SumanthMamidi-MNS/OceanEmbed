# Design system — OceanEmbed dashboard

The design system of the final dashboard: the React single-page app in `web/`, served by
`oceanembed serve`. The Streamlit app in `app/` is the earlier proof of concept and keeps its own
tokens in `app/ui/theme.py`; it is a fallback, not the product, and nothing below describes it.

## Brief

A deliberately designed, production-grade scientific product: strong visual hierarchy, polished
typography, intentional spacing, professional scientific visualisation, clean maps, meaningful
controls, depth and profile exploration, validation views, representation views, experiment views
and clear scientific storytelling, built on the actual outputs of a run. Audience: a jury of ocean
scientists and engineers, reading on a laptop or a projector.

## Feedback on the first version (binding)

1. **Keep the light, clean scientific look.** It reads as more professional; the direction below
   stands.
2. **Keep the top navigation** for the five sections: Overview, Explorer, Validation,
   Representation, Experiments.
3. **One small contextual control panel** for date, depth, location and methods, instead of
   controls scattered through the page or a full-height sidebar. This is the *selection bar*
   (see Controls): the same component, in the same place, with the same behaviour on every view
   where the shared selection matters.
4. **The data and the map are the hero, especially on the Overview:** less news article, more
   scientific instrument. The Overview opens on a large live map of the reconstruction with its
   controls and read-outs, not on a headline with a figure beside it.
5. **Keep the story, shorten the text.** The order (problem, inputs, method, output, skill,
   limits) stays, carried by figures with a short title, a one-line caption and computed
   read-outs. Large serif headline blocks and long paragraphs are gone. Every honesty treatment
   stays.

## Direction: a printed atlas that answers back

The PoC was a dark navy dashboard with a cyan accent. The final product is **light**: warm paper,
blue-black ink, serif headings, figures framed like plates in a journal. The reasons:

- **The data needs the dark end of the scale.** The thermal colormap starts near black. On a dark
  page the coldest water merges with the background and the land (the PoC had to cut the darkest
  10 % of the map). On paper the full range is usable and land can be a quiet, light neutral.
- **It will be projected.** Dark interfaces wash out in a lit room; ink on paper survives.
- **It should read as science, not as a monitoring console.** The jury reads papers. A figure with
  a title, a caption, units and a frame is the format they trust; screenshots of this app can go
  straight into a report.
- **It is the opposite of generic.** Dark navy with a cyan accent is what every generated dashboard
  looks like. A paper page with a text face, a data face and one ink colour is a decision.

Principles, in order:

1. **The data is the hero.** Maps and charts take the area; chrome is hairlines and text.
2. **Cells are shown as they are.** No interpolation, no smoothing: one grid cell is one flat
   square, the coastline follows cell edges. The grid is the measurement.
3. **Everything is linked.** One day, one depth and one water column are shared by every view and
   live in the URL; maps of a group share camera and cursor.
4. **Honest by construction.** Sentences that state a result are computed from the metrics; the
   baselines are never off screen when the model is on it; caveats are part of the layout.
5. **Something you use, not something you read.** A title is a few words, a caption one or two
   lines, a result a computed sentence or a number with its unit. The Overview tells the story in
   figures; no view opens on a paragraph.

## Information architecture

One top bar: wordmark, five views, run selector (the run's `label` from the API; its `description`
is the tooltip and is printed on the Overview and Experiments views). Under it, part of the same
sticky header: the honesty bands, then the selection bar. The shared selection (run, day, depth,
point, estimate) survives navigation and is written to the URL.

| View | Path | Job |
|---|---|---|
| Overview | `/` | **Does it work, at a glance**: a fixed, curated result panel, not an exploration tool. It opens on the evidence (reconstruction, GLORYS and their difference side by side, linked, on shared scales) for one typical test day at one thermocline depth, both chosen from the metrics (see "The evidence on the Overview"). Then the pooled scores of every method with the computed headline, RMSE and skill by depth for all methods, the basin contrast, the Argo comparison with GLORYS as its floor and the independence note, the method strip (inputs, embedding, reconstruction, validation) and what it does not show. No depth rail, no point, no layer switch, no selection bar: one "Open in Explorer" action carries the day and depth. |
| Explorer | `/explore` | **The instrument** for free exploration. One day at one depth for the selected estimate (reconstruction, ridge, ablation): estimate, GLORYS and difference on linked maps, or every estimate side by side; temperature or anomaly; the surface inputs; the water column under a point (profile with every estimate, vertical section, time–depth as temperature or anomaly). Timeline with daily RMSE, depth rail, playback. |
| Validation | `/validation` | Against GLORYS (tables, every metric by depth, basins), where (error maps per method for every metric), when (daily RMSE, pooled RMSE, bias and spatial anomaly correlation as lines; a day × depth field), against Argo (independence note, profile map with server-side filters, sortable paged list, one profile against every estimate, density scatter, per-depth metrics). |
| Representation | `/representation` | The embedding as PCA-RGB, a scree of the first eight components, the exact cosine-similarity map from a chosen cell, component-to-field correlations, the components one at a time, the surface fields, and what the pretraining task recovered. |
| Experiments | `/experiments` | Methods and ablations table, the same across runs (`/api/compare`), training curves, data products and model configuration, NetCDF downloads (the product first, baseline and ablation fields labelled as such), generated report and figures. |

URL: `/<view>?run=&date=YYYY-MM-DD&depth=<m>&lat=&lon=&est=<method>` plus per-view options
(`mode`, `q`, `td`, `sbs`, `pall`, `sec`, `focus`, `basins`, `vec`, `layer`, `basin`, `metric`,
`dh`, `ab`, `as`, `ae`, `asort`, `ap`, `prof`, `sm`, `sc`, `sim`, `cmp`). `est` is the estimate
shown wherever one method is shown (omitted for the main model). Changing view pushes a history
entry; changing the selection replaces it (debounced), so Back leaves a view rather than undoing a
scrub. Changing the run drops the date, the estimate and the options, which belong to the old run.

Defaults when the link names nothing: the run that has a reconstruction, is real, long-trained
and evaluated; the day nearest the middle of the period that has a GLORYS target and an embedding;
the centre of the pooled depth range (a thermocline depth, never the surface); the ocean cell
nearest the centre of the first basin whose column reaches the deepest level.

### The evidence on the Overview

The Overview and the Explorer do different jobs, so they must not open on the same picture. The
Overview shows one fixed piece of evidence, chosen by a rule from the run's metrics
(`chooseEvidence` in `lib/narrative.ts`, tested), and says in the figure's subtitle why:

- **Depth:** inside the pooled (thermocline) range, the level where the climatology's RMSE is
  largest, i.e. where there is most to gain. Never the surface: the surface temperature is an
  input, so a surface map flatters the model.
- **Day:** the test day whose pooled daily RMSE is the median of the test days (the daily RMSE at
  that depth for runs without the pooled series), among the days that have a GLORYS target and an
  embedding. A typical day, neither the best nor the worst.

The Overview ignores the day and depth of the URL; "Open in Explorer" hands its own day and depth
to the Explorer.

## Tokens

Single source of truth: `web/src/lib/theme.ts`. `applyTheme()` publishes the colours as CSS custom
properties; canvas and SVG code import the same object. Spacing, type and motion tokens are in
`web/src/styles/base.css`.

### Colour

| Token | Value | Use | Contrast on paper |
|---|---|---|---|
| `paper` | `#F7F6F2` | page background | |
| `surface` | `#FFFFFF` | figure and panel surface | |
| `sunken` | `#EFEDE6` | wells, loading blocks, bar tracks | |
| `rule` | `#DDD9CE` | hairlines, panel borders | |
| `ruleStrong` | `#B4AFA2` | axis lines, control borders | |
| `grid` | `#ECE9E1` | chart grid (one step off the surface) | |
| `ink` | `#14212B` | text, active controls, section rules | 15.2 : 1 |
| `ink2` | `#42515C` | secondary text, axis titles | 7.6 : 1 |
| `ink3` | `#5A6872` | captions, overlines, tick labels | 5.3 : 1 (4.9 : 1 on `sunken`) |
| `accent` | `#0A5A78` | links, focus ring, chapter numbers, pooled-range band | 7.1 : 1 |
| `accentSoft` | `#DDEBF0` | band fill, validation step of the method diagram | |
| `warn` / `warnInk` | `#F3BC3D` / `#2B1D00` | synthetic-data band and tags only | 9.4 : 1 (ink on warn) |
| `cautionSoft` / `cautionRule` / `cautionInk` | `#ECE9F4` / `#6A5FA3` / `#2B2552` | short-training band and tags only | 11.8 : 1 |
| `danger` / `dangerSoft` | `#A3261C` / `#FAE9E6` | failed requests only | 6.3 : 1 |
| `land` | `#D6D1C4` | land on every map | |
| `seafloor` | `#B9BCB7` | ocean at the surface but below the sea floor at the shown depth; no-data cells of sections | |
| `coast` | `#2E3B44` | coastline (cell edges of the surface mask) | |

One accent. The two status hues (amber, violet-grey) are reserved for the two honesty treatments
and are never used for data.

### Type

Bundled locally (`@fontsource`, latin subsets; no network request for fonts).

| Role | Face | Size / weight |
|---|---|---|
| View title `h1` | Newsreader, opsz 30 | 1.625rem / 500, a few words, one line |
| Section, chapter `h2` | Newsreader, opsz 30 | 1.25rem / 500 |
| Computed headline (skill sentence) | Newsreader, opsz 20 | 1.1875rem / 400 |
| Read-out number | IBM Plex Sans | 1.375rem / 600, tabular; unit in 11px `ink3` |
| Figure title `h3` | IBM Plex Sans | 1rem / 600 |
| Body, prose | IBM Plex Sans | 0.9375rem / 400, line-height 1.55 |
| UI, controls | IBM Plex Sans | 0.8125–0.875rem / 500 |
| Caption | IBM Plex Sans | 0.8125rem, `ink2` |
| Overline, tags, keys | IBM Plex Mono | 0.6875rem / 500, uppercase, 0.09em tracking |
| Chart ticks / labels | IBM Plex Sans | 11px / 11.5px 500 |
| Read-outs, coordinates | IBM Plex Mono | 0.75rem, tabular |

Serif for titles and for the one computed finding, sans for what is used (controls, captions,
charts, read-outs), mono for what is looked up (coordinates, tags, control labels). There is no
display size and no lede paragraph: the largest text on a page is a 26px title. Every number is set with tabular lining figures, a
real minus sign (−), fixed decimals, thin-space thousands and an en dash for a missing value
(`lib/format.ts`). The root size is 16px, 17px from 1800px and 19px from 2300px, so a projector
gets larger text without a separate layout.

### Space, shape, layout

- Spacing scale 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96 px (`--s1` … `--s9`).
- Radius 3px everywhere: plates, not cards. No shadows except the tooltip.
- Page: max width 1760px, gutter `clamp(16px, 2.6vw, 44px)`. Sections open with a 1px ink rule;
  figures are white panels with a hairline border and 16px padding.
- The Overview opens on one panel of three equal linked maps, then a row of three panels (pooled
  table with the computed headline, RMSE by depth, skill by depth), then two (basins, Argo). Its
  chapters are a number, a title and a caption on one line above their figures. Working views
  use panels in 2-column grids that collapse to one.

## Maps (`components/map/MapCanvas.tsx`)

Custom 2-D canvas renderer; no tile stack, nothing fetched but the arrays.

- **Projection:** equirectangular with square degrees. The canvas height is derived from its width
  and the domain, so the aspect is always true. At 5–30°N the east–west scale error of this
  projection is at most 13 % (cos 30°); a conformal projection would resample the grid, which
  principle 2 forbids.
- **Raster:** each field is coloured through a 256-entry lookup table into a grid-sized image and
  drawn with nearest-neighbour scaling. NaN is transparent.
- **Land and sea floor:** land is the flat `land` neutral behind the raster, from the surface mask;
  cells that are ocean at the surface but below the sea floor at the shown depth are `seafloor`.
  A key for the two neutrals sits under the main map.
- **Coastline:** the edges between ocean and land cells of the surface mask, 1px `coast`. Fields
  that are also defined over land (winds) get a light halo under it.
- **Graticule and axes:** hairlines on round degrees with °N / °E labels outside the frame.
- **Basins:** optional dashed outlines of the evaluation boxes from the API, with haloed labels.
- **Vectors:** currents and winds are speed as colour (`speed` colormap from zero) plus direction
  arrows on a regular screen lattice (block-averaged, so they do not alias). Arrow length grows
  with the square root of speed; arrows are ink on light colours and white on dark ones. U and V
  components remain available as diverging maps.
- **Markers:** the selected water column is a ringed dot with a white halo; the section through it
  is a dashed line.
- **Observation points** (Argo profiles, `lib/points.ts`): dots coloured by their own RMSE, sorted
  once into 24 colour bins and drawn as one canvas path per bin, lowest values first, so a redraw
  costs the same handful of state changes for 250 or 50 000 profiles and the largest errors end up
  on top. The halo is stroked under the fill, so overlapping dots never erase each other; the dot
  shrinks with the count and grows with the zoom. Hover reads the number of profiles in the cell
  and their mean RMSE; a click opens the nearest profile.
- **Linking:** maps of a group share a `LinkedView` (camera + cursor). Wheel zooms about the
  pointer (Explorer) or Ctrl/Cmd + wheel (scrolling pages), drag pans, double-click zooms, `+` `−`
  `0` on a focused map. Hover draws a cross-hair and the hovered cell on every map of the group and
  writes each map's value to its header; none of this re-renders React.
- **Colour bars:** under every map, with units; pointed ends mean the range is clipped at
  percentiles; diverging bars are symmetric and signed.

### Colormaps (`lib/colormaps.ts`)

cmocean maps (12 control points each, interpolated to 256) and viridis.

| Quantity | Colormap | Range |
|---|---|---|
| Temperature (maps, sections, time–depth, SST) | `thermal`, full range | API hint: 1st–99th percentile, shared by reconstruction and GLORYS, per depth |
| Difference, anomaly, bias, SLA, U/V components, RMSE-vs-climatology | `balance` | symmetric about zero (99th percentile of the absolute value) |
| Salinity | `haline` | API hint |
| Speed of currents and winds | `speed` | 0 to 99th percentile |
| RMSE, MAE (maps, day × depth), profile RMSE dots | `amp` | from 0 |
| Correlation maps, single PCA components | `viridis` | API hint / 0–1 |
| Embedding similarity, as it is | `viridis` | fixed 0–1 (comparable between cells, days and runs) |
| Embedding similarity, mean removed; daily spatial anomaly correlation | `balance` | symmetric, fixed −1…1 / 99th percentile |
| Skill vs climatology | `curl` reversed (teal = better than climatology) | API hint: symmetric about zero, capped at ±1 |
| Matchup density | `tempo`, log count | |
| Depth as colour (scatter) | `deep`, square-root of depth | 0 to deepest level |
| Embedding | PC1–3 as R, G, B | fixed per run by the API |

"Range per day" follows each day's percentiles. "Hold range" (always on during playback) keeps one
scale per depth for the whole period, from the API's `/ranges` (temperature, and per method the
symmetric difference and anomaly limits), so a seasonal cycle moves through fixed colours. Only
where the API has no period range do the limits fall back to "only ever widen".

**Maps whose values exceed the scale.** The error maps print the API's `range_info` under the map
when cells lie beyond the colour range ("9 % of the cells lie beyond the colour scale (lowest
−3.04)"), and the colour bar is pointed only at the ends that are really exceeded. Skill is at
most 1 and unbounded below, so its symmetric scale always clips the worst cells; the note says by
how much.

**Comparing methods.** The temperature range is the API's and is the same for every method. When
estimates are shown side by side their difference panels share one symmetric limit, the largest
of the methods' own limits, and each row of maps has a single colour bar.

## Charts

- **One line chart** (`XYChart`, SVG) for depth profiles, time series and training curves: same
  grid, axes, hover and tooltip everywhere. Never a dual axis.
- **Depth always increases downward** on a square-root axis (0 at the top), so the thermocline
  gets most of the height and 1000 m still fits. Tick labels are real depth levels, the roundest
  first. The surface is drawn as a line at the top. The same axis is used by the depth rail, the
  sections and the time–depth plots; levels are bands, never interpolated.
- **Units on every axis** in the axis title; error metrics start at zero, bias and skill show a
  zero line, the pooled depth range is a shaded band named in the legend, the depth or day
  selected elsewhere is a dotted reference line. Clicking a chart sets that depth or day.
- **Daily series** (validation): four small multiples on one time axis, one line per method:
  RMSE at the selected depth, RMSE pooled over the thermocline range, bias (zero line), and the
  *spatial* anomaly correlation of the day. Its title and caption say that it is spatial (across
  the cells of one day) and not the temporal correlation of the per-depth table.
- **Daily series scope**: the three daily charts are stacked on one full-width time axis and show
  either the selected depth or the pooled range (RMSE, bias and spatial anomaly correlation all
  exist pooled).
- **Scree** (representation): one line per principal component, bar and percentage, with the
  running total; the three components that make the colour are drawn in red, green and blue.
- **Methods have one identity everywhere** (`lib/methods.ts`), colour + dash + marker:

  | Method | Colour | Dash | Marker |
  |---|---|---|---|
  | OceanEmbed (`model`) | `#0B5FA5` | solid, 2.25px | circle |
  | Ridge regression | `#C8431F` | dash-dot | square |
  | GLORYS (target / reference) | `#0E8A6A` | long dash | diamond |
  | No-pretraining ablation (first `model_<tag>`) | `#B8730A` | dash | triangle |
  | Further ablation | `#8A4FB0` | long dash-dot | triangle down |
  | Climatology | `#6B7680` (neutral) | dot | cross |
  | Argo observation | ink | solid | open ring |

  The five chromatic slots pass the dataviz palette validator on the light surface (lightness
  band, chroma floor, worst adjacent colour-vision-deficiency ΔE 8.8, normal-vision ΔE 19.4,
  contrast ≥ 3 : 1). Colour follows the method key, never its rank. A legend is present whenever
  there are two or more series; tooltips repeat swatch, name and value.
- **Tables are first-class.** Summary tables put baselines next to the model, list the anomaly
  correlation before the raw one, and underline the best candidate per column (the GLORYS row of
  the Argo tables is a reference and is never "best"). Per-depth charts have a table twin.
- **Bars** (pooled RMSE, explained variance, pretraining recovery) are labelled with name and
  value, share one zero-based axis and never rely on colour.
- **Scatter** of thousands of matchups is a binned density (log count) or mean depth per bin, with
  the 1 : 1 line.

## Controls

- **Selection bar** (`components/controls/SelectionBar.tsx`): the one control surface for *what is
  being looked at*. A compact row docked under the top bar and the honesty bands, inside the same
  sticky header, so it is always in the same place and never over the data. Groups, left to right:
  **Day** (previous, date select grouped by month, next, position in the period; play on the
  Explorer), **Depth** (shallower, level select, deeper), **Point** (latitude and longitude of the
  water column), **Estimate** (the methods the run has day fields for, from `field_methods`; hidden
  when there is one), then view switches on the right (Explorer: Compare). A view declares which
  groups apply: Explorer and Validation all four; Representation day (limited to the days with
  an embedding) and point; the Overview and Experiments have no selection and no bar (the
  Overview's day and depth are chosen by rule, not by the reader).
  Every value is URL state, so the bar, the keyboard shortcuts and a pasted link are three handles
  on the same thing. Below 860 px the bar scrolls sideways instead of stacking.
- **Figure options** stay with the figure they change (how it is drawn, not what is selected):
  fields, quantity, colour range, basin outlines and zoom above the Explorer maps; map layer on
  the Overview; metric on the error maps; filters and order on the Argo list.
- **Timeline** (Explorer): a strip of all days with the daily RMSE of the selected
  estimate and of climatology at the selected depth. It is the day slider; the hard days are
  visible before you go to them. Play only advances to frames already in memory.
- **Depth rail** (Explorer): the levels on the shared square-root axis beside the main
  map, with the mean temperature of each level as a shape behind it and the pooled range shaded.
- **Compare** (Explorer): every estimate of the run beside GLORYS on one temperature scale, and
  every estimate minus GLORYS on one symmetric scale, with the day's RMSE, bias and MAE of each in
  a table; the water column moves into the row of point panels.
- **Argo list**: basin and date filters, order (date, largest or smallest error of the selected
  estimate) and paging are done by the API; the page shows ten rows. A profile opened from the
  map is shown on the page that contains it (`around=`); using the pager or changing a filter
  takes the list where the reader sends it.
- Segmented controls for exclusive modes, toggles for overlays, native selects for long lists.
  Active state is ink fill; no colour-only states.
- Keyboard: `←` `→` day, `↑` `↓` depth, `Space` play (Explorer); `,` `.` day and `[` `]` depth on
  every view; sliders also take Home / End / Page keys; `+` `−` `0` on a focused map.

## Honesty treatments

- **Synthetic run** (`data_source == "synthetic"`): an amber, striped band in the sticky header on
  every view ("pipeline demonstration, not scientific skill"), an amber `SYNTHETIC` tag after the
  title of every figure (so a cropped screenshot still says it), and a caveat note where findings
  are stated. Cross-run panels carry explicit per-run chips instead of the automatic tag.
- **Short training** (fewer than 365 training days, from `n_train_days` of the API, never from the
  run name or its dates): a violet-grey band in the sticky header stating the number of training
  days, and the same note beside the findings; if the climatology has a single harmonic term
  (`n_harmonic_terms`) the band says it is a constant mean. Captions name the climatology fit in
  words (constant mean; mean and annual cycle; mean, annual and semi-annual cycle).
- **Correlations:** the anomaly correlation is listed first and explained; the raw one is labelled
  as inflated wherever it appears.
- **Baselines:** climatology and ridge are in every table, chart and bar set that shows the model;
  the Overview read-outs show them beside the model's number. Baseline and ablation NetCDF files
  are listed after the product, each labelled "baseline" or "ablation" and "not the product".
- **Estimates:** a map, section or time–depth plot of ridge or of the ablation is titled with that
  method's name, never "Reconstruction"; while another estimate loads, the title stays that of the
  data on screen.
- **Argo:** the independence statement from the API precedes every Argo number; GLORYS-vs-Argo is
  shown as the floor.
- **Ablation:** the verdict sentence is computed with both numbers. The ablation is trained once,
  so a pooled-RMSE difference under 5 % in either direction reads "no measurable gain from
  pretraining" (within what another training seed can produce); only a larger one reads "helps"
  or "does not help".
- **Depth of the skill:** the computed sentence names where the skill is clear (at least 0.1),
  where it is marginal and where the model does not beat climatology, as depth ranges.
- **Argo floor:** the Argo sentence and bars are over the pooled range, list GLORYS itself as
  "the floor" (a model trained on GLORYS cannot agree with Argo better than GLORYS does), and
  state the mean difference GLORYS and the reconstruction share when there is one. Nothing is
  said beyond those numbers.
- **Computed text:** `lib/narrative.ts` builds every sentence that states a result; differences
  under 1 % are reported as "about the same". Depths are named as ranges ("300 m and below",
  "0–30 m"), and a scattered set as a count, so the sentence stays one line on a 15-level grid.
  Captions never contain a number that is not read from the API.

## States

Every figure has three states besides data: a quiet shimmer block while loading (same height as
the figure), a dashed "not available for this run" box that shows the API's own hint for a 404 or
for an artefact the run does not have, and a red-tinted error box with a retry for real failures
(API down, 5xx). A deep link to an unknown run lists the available runs. While a new day loads, the
previous day stays on screen slightly dimmed; maps never blank.

## Motion

Motion only where it explains: day playback (about 5 frames per second; 1.7 with
`prefers-reduced-motion`), 120 ms colour transitions on controls, the loading shimmer. With
`prefers-reduced-motion` all transitions and the shimmer are disabled. Nothing animates on scroll.

## Accessibility

- Text contrast is at least 4.9 : 1 on every surface it is used on (WCAG AA); method colours are at least 3.8 : 1 on the chart surface (non-text marks need 3 : 1).
- Colour is never the only channel: methods differ by dash and marker, bars and maps are labelled,
  the diverging maps are signed in their colour bars and read-outs.
- Landmarks, a skip link, one `h1` per view, labelled groups, `aria-pressed` on toggles,
  `role="slider"` with value text on the timeline and depth rail, `role="img"` with a descriptive
  label on every chart and map, table twins for the profile charts.
- Everything a pointer can do has a keyboard path: day and depth by keys, the point by the
  latitude / longitude fields, Argo profiles by a select, zoom by keys or buttons.
- Focus is always visible (2px accent ring).

## Responsiveness

Checked at 900, 1280, 1440 and 1920 px (content is capped at 1760 px; the root font size steps up at 1800 and 2300 px).

| Width | Layout |
|---|---|
| ≥ 1181 | Overview: three evidence maps in a row; table and two depth charts in a row. Explorer: maps left, water column right; sections and time–depth side by side; compare: one column per estimate. Validation: five metric charts in a row; Argo map and list beside the opened profile. |
| 861–1180 | Single column of panels; below 1021 px the evidence maps stack; compare maps two per row; metric charts three in a row; top bar wraps the views under the wordmark; the selection bar may wrap to a second row. |
| ≤ 860 | Point panels stack; two-up small multiples; narrower depth rail; the selection bar scrolls sideways. |
| ≤ 620 | One map per row. |

## Known trade-offs

- **Light only.** There is no dark theme; the tokens are CSS variables, so one can be added, but
  every colormap pairing and the method palette would have to be re-validated on a dark surface.
- **Plate carrée**, not an equal-area or conformal projection (see Maps).
- **Hover values are not exposed to assistive technology** point by point; the tables and the
  per-view summaries carry the numbers instead. Charts cannot be traversed with the keyboard;
  their table twins can.
- **The coastline is a staircase** at 0.25°: it is the model's coastline, not the real one.
- **Small multiples at laptop width are small** (about 250 px per map in the overview ladder);
  they are for comparison, the Explorer is for reading values.
- **Wheel zoom needs Ctrl/Cmd outside the Explorer**, so that long pages scroll normally.
- **The selection bar scrolls sideways below 860 px**, so a group can be off screen until it is
  scrolled to; the alternative, three stacked rows in a sticky header, would take the map.
- **The estimate applies where one method is shown.** Tables and by-depth charts always show every
  method; the estimate picks the map, section, time–depth plot, the day × depth field and the
  colour and order of the Argo profiles.
- **A held range clips extreme days**: the API's period range is a 1st–99th percentile over 24
  sampled days, so the most extreme cells of some days saturate (the pointed colour bar says so).
- **The evidence maps are modest at laptop width** (about 390 px each at 1440 px): three maps
  side by side on shared scales read as one comparison; the Explorer is one click away for size.
- **Side-by-side maps are small at laptop width** (about 280 px each at 1440 px): they are for
  comparing patterns, the single-estimate view is for reading values.
- **The raw similarity scale is fixed at 0–1**, so the few negative values clip to the darkest
  colour; the "relative to the average" mode shows the full signed range.
- **The Argo list shows ten profiles a page**; the map shows all of them.
- **The Overview no longer explains the science in prose.** A reader who wants the argument reads
  the generated report on the Experiments view.
