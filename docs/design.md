# Design system — OceanEmbed dashboard

The design system of the final dashboard: the React single-page app in `web/`, served by
`oceanembed serve`. The Streamlit app in `app/` is the earlier proof of concept and keeps its own
tokens in `app/ui/theme.py`; it is a fallback, not the product, and nothing below describes it.

## Brief

A deliberately designed, production-grade scientific product: strong visual hierarchy, polished
typography, intentional spacing, professional scientific visualisation, clean maps, meaningful
controls, depth and profile exploration and accuracy views, built on the actual outputs of a run.

**The site is for the people who use the product.** A user wants to see the temperature field for
a day, a depth and a place, to know how far to trust it, and to download it. The primary
navigation serves exactly that. The material that answers "why this model and how do we know"
(method comparisons, ablations, training curves, the embedding) is kept, complete, in one
secondary **Research** area that is not in the primary navigation, and in the repository
(`docs/research/`, `results/`). Audience: ocean scientists and engineers, on a laptop or a
projector.

## Feedback on the first version (binding)

1. **Keep the light, clean scientific look.** It reads as more professional; the direction below
   stands.
2. **Keep the top navigation.** It lists the product's views: Overview, Explorer, Accuracy,
   Data & downloads (and Live, when it ships). Research is not one of them.
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
6. **A prototype people use, not an experimental page.** No research material on the first page:
   no ablation verdicts, no method-comparison tables, no embedding. The first page says what the
   product gives, how accurate it is against the seasonal climatology, and where not to trust it.

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
   seasonal climatology (what a user would have without the product) is never off screen when
   the reconstruction's accuracy is on it; caveats are part of the layout.
5. **Something you use, not something you read.** A title is a few words, a caption one or two
   lines, a result a computed sentence or a number with its unit. The Overview tells the story in
   figures; no view opens on a paragraph.
6. **Product first, research apart.** A view either helps someone use the field (primary
   navigation) or documents how the model was chosen (Research). Nothing is on both.

## Information architecture

One top bar: wordmark, the product's views, run selector (the run's `label` from the API; its
`description` is the tooltip and is printed on the Overview and on Data & downloads). Under it,
part of the same sticky header: the honesty bands, then the selection bar. The shared selection
(run, day, depth, point) survives navigation and is written to the URL.

**Primary navigation: the product.**

| View | Path | Job |
|---|---|---|
| Overview | `/` | **What it gives you, at a glance.** One line of what the product is. The reconstruction as the hero: reconstruction, GLORYS and their difference side by side, linked, on shared scales, for one typical test day at one thermocline depth, both chosen from the metrics (see "The evidence on the Overview"), with the one action into the Explorer. Then *how accurate*: the error over the pooled range as a computed sentence and a two-row table (the reconstruction and the seasonal climatology, per test year), the error by depth, the two basins. Then *where not to trust it*: the depths without skill, computed from the metrics, two fixed limits, and the Argo comparison with GLORYS as its floor and the independence note. No method comparison, no ablation verdict, no embedding. No selection bar. |
| Explorer | `/explore` | **The instrument.** One day at one depth: reconstruction, GLORYS and difference on linked maps; temperature or anomaly; the surface inputs; the water column under a point (profile, vertical section, time–depth). Timeline with daily RMSE, depth rail, playback. It shows the product: no estimate selector, no Compare. |
| Live | `/live` | Reserved for the same-day reconstruction. The position (third) and the route exist; the entry is not listed and the route opens the Overview until the view ships (`LIVE_ENABLED` in `state/url.ts`), so there is no dead link. |
| Accuracy | `/accuracy` | **How far to trust a value.** The computed answer first (error against the climatology, where to use it, where not). Against GLORYS: tables, every metric by depth, basins, per test year. Where: error maps. When: daily RMSE, bias and spatial anomaly correlation, a day × depth field. Against Argo: independence note, profile map with server-side filters, sortable paged list, one profile opened, density scatter, per-depth metrics. Every figure shows the reconstruction next to the climatology (and GLORYS, the floor, against Argo); the other methods are not on this page. |
| Data & downloads | `/data` | **What goes in, what you can take away.** Period, splits, grid and depth levels; the surface products with "used by the model" / "available, not used"; the monthly NetCDF product files; the model in four lines and where the released weights are (`models/final/` in the repository; the site does not serve them); the generated report and its figures. Links to Research. |

**Secondary: Research.** Reached from a quiet link in the colophon ("Research: how the model was
chosen") and from Data & downloads; never from the primary navigation. Every Research page opens
with the same band: what the area is, that the research notes are in the repository under
`docs/research/` and the result tables under `results/`, its four pages as tabs, and "Back to
the product".

| Page | Path | Holds |
|---|---|---|
| Methods | `/research` | The comparison table of methods and ablations with the computed headline and verdicts (ridge regression, per-pixel network, pretraining), the year-by-year table against GLORYS and Argo, pooled bars, RMSE and skill by depth for every method, the method diagram (inputs, encoder, embedding, decoder, output), the same table across runs (`/api/compare`), training curves, the model configuration, and the baseline and ablation NetCDF fields, each labelled "not the product". |
| Every score | `/research/scores` | The Accuracy page with every method of the run in every table, chart, map and Argo figure, and the estimate selector. |
| On the map | `/research/maps` | The Explorer with every estimate: it opens in Compare (every estimate beside GLORYS on one temperature scale, every difference on one symmetric scale, the day's RMSE, bias and MAE) and offers the estimate selector. |
| Embedding | `/research/embedding` | The embedding as PCA-RGB, a scree of the first eight components, the exact cosine-similarity map from a chosen cell, component-to-field correlations, the components one at a time, the surface fields, and what the pretraining task recovered. |

The same components serve a product view and its Research twin (Accuracy / Every score, Explorer
/ On the map). Which methods a view shows is one rule in the run context (`scope`): the product
views show `model`, `climatology`, `glorys` and the observations; Research views show every
method.

**What moved.** The earlier layout had five views. Validation became Accuracy (model against
climatology) and Research · Every score (all methods). Representation became Research ·
Embedding. Experiments was split: data products, product files and the report went to Data &
downloads; methods, ablations, cross-run comparison, training, configuration and the comparison
fields went to Research · Methods. The Overview's method tables, by-depth charts of all methods,
ablation and per-pixel-network notes and the method strip went to Research · Methods. The
Explorer's estimate selector and Compare went to Research · On the map. Old links are rewritten
in place: `/validation` → `/accuracy`, `/experiments` → `/research`, `/representation` →
`/research/embedding`, and an Explorer link with `sbs=1` or `est=` → `/research/maps`.

URL: `/<view>?run=&date=YYYY-MM-DD&depth=<m>&lat=&lon=` plus per-view options (`mode`, `q`, `td`,
`pall`, `sec`, `focus`, `basins`, `vec`, `yr`, `basin`, `metric`, `ds`, `dh`, `ab`, `as`, `ae`,
`asort`, `ap`, `prof`, `sm`, `sc`, `sim`, `cmp`, `sbs`). On the Research pages `est=<method>` is
the estimate shown wherever one method is shown (omitted for the main model); it does not exist
on the product views and is dropped when a link leaves Research. Changing view pushes a history
entry; changing the selection replaces it (debounced), so Back leaves a view rather than undoing
a scrub. Changing the run drops the date, the estimate and the options, which belong to the old
run.

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

### How accurate, and where not to trust it

Both are computed (`lib/narrative.ts`), on the Overview and at the top of Accuracy:

- **How accurate** (`accuracyHeadline`): the RMSE over the pooled range against the reanalysis,
  each test year, and the difference from the seasonal climatology in percent. Only the
  climatology is named: it is the estimate a user has without the product.
- **Where not to trust it** (`trustLimits`): from the skill against the climatology at each
  depth. Levels with a skill of at least 0.1 are "use it"; levels with a positive skill below
  0.1 are "marginal"; levels at or below zero are "no better than the seasonal climatology". The
  limit is set in the display face, with an ink rule, as the first thing of the chapter. If no
  level has clear skill the sentence says not to rely on the reconstruction at all.

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
| `accentStrong` | `#06445C` | hover state of links and primary buttons | 10.6 : 1 |
| `warnSoft` / `warnRule` | `#FDF3D7` / `#C99512` | block quotes of the generated report; border of the synthetic band and chips | |
| `graticule` | `rgba(20, 33, 43, 0.16)` | graticule lines on maps | |
| `marker` / `markerHalo` | `#14212B` / `#FFFFFF` | selected water column and selected Argo profile: ink ring on a white halo | |

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
- The Overview opens on one panel of three equal linked maps, then a row of three panels (error
  over the pooled range with the computed sentence, error by depth, basins), then two (the limit
  by depth, Argo). Its chapters are a number, a title and a caption on one line above their
  figures. Working views use panels in 2-column grids that collapse to one.
- The Research band is a sunken strip with the overline "Research", one sentence, tabs for its
  pages and a quiet "Back to the product" button: the reader always knows this is the secondary
  area, without a second top bar.

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
| Skill vs climatology | `curl` reversed (teal = better than climatology) | API hint: symmetric about zero; the limit is the API's robust one (95th percentile of the absolute skill over all methods at that depth, kept within 0.5–3), and the note under the map says how many cells exceed it |
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
- **Daily series** (Accuracy): three charts stacked on one full-width time axis, one line per
  method: RMSE, bias (zero line) and the *spatial* anomaly correlation of the day, either at the
  selected depth or over the pooled range (a switch on the panel). The correlation's title and
  caption say that it is spatial (across the cells of one day) and not the temporal correlation
  of the per-depth table. Measured with 715 days and five methods: the charts stay one path per
  method, and stepping is unaffected.
- **Test years.** When the test period spans several calendar years (`per_year` in the metrics)
  a "Test year" switch (all years, or one) sits above the result tables of the Overview and of
  Accuracy and drives tables, by-depth charts, basins and the Argo numbers; the Overview table
  also carries one RMSE column per year, the computed sentence quotes each year, and Research ·
  Methods has a year-by-year table of every method against GLORYS and against Argo. Daily charts
  and error maps always cover the whole period.
- **Scree** (Research · Embedding): one line per principal component, bar and percentage, with the
  running total; the three components that make the colour are drawn in red, green and blue.
- **Methods have one identity everywhere** (`lib/methods.ts`), colour + dash + marker:

  | Method | Colour | Dash | Marker |
  |---|---|---|---|
  | OceanEmbed (`model`) | `#0B5FA5` | solid, 2.25px | circle |
  | Per-pixel MLP (`mlp`) | `#D95FA8` | short dash | pentagon |
  | Ridge regression | `#C8431F` | dash-dot | square |
  | GLORYS (target / reference) | `#0E8A6A` | long dash | diamond |
  | Ablation (first `model_<tag>`: the no-pretraining or the pretrained variant, whichever the run's ablation is) | `#B8730A` | dash | triangle |
  | Further ablation | `#8A4FB0` | long dash-dot | triangle down |
  | Climatology | `#6B7680` (neutral) | dot | cross |
  | Argo observation | ink | solid | open ring |

  The chromatic slots pass the dataviz palette validator on the light surface (lightness band,
  chroma floor, contrast ≥ 3 : 1) in legend order. The MLP pink was chosen by checking it pair
  by pair against every other chromatic slot (model, both ablation slots, ridge, GLORYS): each
  pair passes the colour-vision-deficiency check (ΔE ≥ 8) and the normal-vision floor (ΔE ≥ 15);
  its nearest neighbours, the model and the MLP, whose lines often run together, differ by
  ΔE 28 in normal vision and also by dash and marker. Method labels come from the API; the short
  forms are "OceanEmbed", "No pretraining" / "Pretrained" for the ablation, and the API label
  otherwise. Legend order: model, ablation, MLP, ridge, climatology, GLORYS, Argo. Colour follows
  the method key, never its rank. A legend is present whenever
  there are two or more series; tooltips repeat swatch, name and value.
- **Tables are first-class.** Summary tables put the climatology (and, in Research, every
  baseline) next to the model, list the anomaly
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
  water column), and, on the Research pages only, **Estimate** (the methods the run has day fields
  for, from `field_methods`; hidden when there is one) and the Compare switch on the right. A
  view declares which groups apply: Explorer and Accuracy day, depth and point; Research · On the
  map and Every score all four; Research · Embedding day (limited to the days with an embedding)
  and point; the Overview, Data & downloads and Research · Methods have no selection and no bar
  (the Overview's day and depth are chosen by rule, not by the reader).
  Every value is URL state, so the bar, the keyboard shortcuts and a pasted link are three handles
  on the same thing. Below 860 px the bar scrolls sideways instead of stacking.
- **Figure options** stay with the figure they change (how it is drawn, not what is selected):
  fields, quantity, colour range, basin outlines and zoom above the Explorer maps; the test-year
  switch above the result tables; metric on the error maps; depth scope on the daily charts;
  filters and order on the Argo list.
- **Timeline** (Explorer): a strip of all days with the daily RMSE of the reconstruction (in
  Research, of the selected estimate) and of climatology at the selected depth. It is the day slider; the hard days are
  visible before you go to them. Play only advances to frames already in memory.
- **Depth rail** (Explorer): the levels on the shared square-root axis beside the main
  map, with the mean temperature of each level as a shape behind it and the pooled range shaded.
- **Compare** (Research · On the map, on by default there; not offered in the Explorer): every
  estimate of the run beside GLORYS on one temperature scale, and
  every estimate minus GLORYS on one symmetric scale, with the day's RMSE, bias and MAE of each in
  a table; the water column moves into the row of point panels. Up to four maps share a row;
  with more (four estimates and GLORYS) they wrap into rows of three.
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
- **Baselines:** on the product views the seasonal climatology is in every table, chart and bar
  set that shows the reconstruction: it is the reference a user needs. Ridge regression, the
  per-pixel MLP and the ablation are in every such figure of the Research pages and nowhere
  else. Baseline and ablation NetCDF files are on Research · Methods, each labelled "baseline"
  or "ablation" and "not the product"; Data & downloads lists the product only.
- **The limit on the first page.** The Overview states where the reconstruction is no better
  than the climatology, computed from the per-depth skill, before anything else in its second
  chapter; the Accuracy page opens with the same sentence.
- **Estimates** (Research): a map, section or time–depth plot of ridge or of the ablation is titled with that
  method's name, never "Reconstruction"; while another estimate loads, the title stays that of the
  data on screen.
- **Argo:** the independence statement from the API precedes every Argo number; GLORYS-vs-Argo is
  shown as the floor.
- **Ablation** (Research · Methods only): the verdict sentence is computed with both numbers and reads the pair as "with
  the pretrained encoder" against "trained from scratch", using the `pretrained` flags of the
  two methods, so it is right whichever of them is the main model. Each is trained once, so a
  pooled-RMSE difference under 5 % in either direction reads "no measurable gain from
  pretraining"; only a larger one reads "helps" or "does not help".
- **Level, not better** (Research · Methods only). The same 5 % rule words every comparison between two networks trained
  once: the model against the per-pixel MLP reads "level" below it, pooled and basin by basin.
- **What the model uses.** Input counts and names come from `used_by_model` of the data products,
  never from the number of surface products. A product the run carries but does not feed to the
  model is shown and labelled "not used by the model" (Explorer surface maps, the data table of
  Data & downloads, method diagram) or "unused" (Embedding correlation table, whose caption says such a
  correlation was not built into the embedding). When the main model's encoder is trained from
  scratch (`main_init`), nothing says it was pretrained: Data & downloads and the method diagram
  say "trained from scratch", and the pretraining figures are titled as the ablation's.
- **Both test years.** The computed sentence quotes each year next to the pooled number, and a
  second one says whether the reconstruction beats the climatology in each year, with the numbers
  (in Research: climatology and ridge regression).
- **Depth of the skill:** the limit names where the skill is clear (at least 0.1), where it is
  marginal and where the reconstruction does not beat climatology, as depth ranges; the Research
  sentence adds where ridge regression has the lower error.
- **Argo floor:** the Argo sentence and bars are over the pooled range, list GLORYS itself as
  "the floor" (a model trained on GLORYS cannot agree with Argo better than GLORYS does), and
  state the mean difference GLORYS and the reconstruction share when there is one. Nothing is
  said beyond those numbers.
- **Computed text:** `lib/narrative.ts` builds every sentence of the product views and
  `lib/research.ts` the method-comparison sentences of the Research pages (no product view
  imports it); differences
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
| ≥ 1181 | Overview: three evidence maps in a row; table, depth chart and basins in a row (from 1281; below, the table spans the row). Explorer: maps left, water column right; sections and time–depth side by side. Research · On the map: one column per estimate. Accuracy: five metric charts in a row; Argo map and list beside the opened profile. |
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
- **The estimate (Research) applies where one method is shown.** Tables and by-depth charts there
  always show every method; the estimate picks the map, section, time–depth plot, the day × depth field and the
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
- **The Overview does not explain the science.** A reader who wants the argument reads the
  generated report on Data & downloads, the Research pages or `docs/research/`.
- **Research is one click further away, on purpose.** Its only entry points are the colophon and
  Data & downloads; a reader who looks for the ablations in the top bar will not find them.
- **No estimate selector outside Research.** A user cannot put the ridge field next to the product
  in the Explorer; that comparison is a Research page.
- **Two pages share one component** (Accuracy / Every score, Explorer / On the map), so a layout
  change to one changes the other.
