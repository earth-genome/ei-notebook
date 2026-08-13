# Automated footprint delineation from known point locations

## Goal

Given (a) a set of point lat/lons for facilities we already know exist and (b) a
patch-based geo-foundation-model embedding dataset covering those points, but
*no* spatial footprints, rebuild a linear-probe binary classifier from the known
points and use the merged patch-level inference output as the footprints.

CLI script `build_footprints.py` over the `footprints` package. Derived from
`ParquetEmbeddingsML-2026-08-10.ipynb` / `ml_utils.py` and
`DuckDBEmbeddingsML-NB-OKtest.ipynb` / `duckdb_ml_utils.py`, but does not import
them (they pull in ipyleaflet / IPython).

Developed as a single self-contained script and ported into this repo on
2026-08-13, when it was split along the section boundaries it had grown: one
module per stage, with the CLI reduced to argument parsing and the run sequence.
The split moved code without editing it — every function's AST was verified
identical before and after. Nothing in `automated/` imports `src/`; the two
workflows share the repo and `scripts/build_duck_assets.py`, not code.

## Measured properties of the reference dataset

Probed from `20230101-20240101_USA_Kansas-deduped.parquet` and
`USA_KS1133_2025-05-02*.geojson` before planning:

| Property | Value |
|---|---|
| Patches | 14,677,576 |
| Embedding dims | 384, `uint8` (quantized) |
| Parquet layout | **one single row group**, 7.4 GB uncompressed |
| Patch size | **326.8 m x 326.1 m** (10.4 ha), rotated quadrilaterals in EPSG:4326 (UTM-grid aligned) |
| Centroid stride | **159.4 m** -> patches overlap 50%, sliding window not a tiling |
| Data bbox | -102.779, 36.106 to -94.110, 40.653 (~384,000 km^2) |
| Kansas area | ~213,000 km^2 -> **the dataset is not clipped to the state** |
| `centroids.parquet` | same 14,677,576 rows, `tile_id` + centroid geometry, one row group |
| `embeddings.db` | DuckDB, table `embeddings`, `tile_id` + 384 cols, ~6 GB |
| Positives | 1,133 points, EPSG:4326 |
| Prior-run polygons `USA_KS1133_2025-05-02.geojson` | **grid-quantized, not independent ground truth** (see below) |
| Union of 3 km buffers around positives | 26,732 km^2 = ~12.5% of Kansas |

**The existing polygons are a prior model run on this same grid, not measured
footprints.** Their areas are exact grid multiples: minimum **10.24 ha =
320 x 320 m**, with **399 of 1,133 (35%) at exactly that value**, then 15.36
(320x480), 20.48, 23.04 (480x480), ... and nothing below 10.24 ha. They were
built with the `centroid.buffer(160, cap_style=3)` convention of
`duckdb_ml_utils.detections_to_rectpolys`. So they can tell us *which facilities
and how many* a new run recovers, but they carry no information about true
facility extent and cannot be used to assess area bias.

Note that `2 x stride = 318.8 m`, i.e. the prior `half_width=160` convention was
already effectively "two strides". `--footprint-geometry patch` is therefore
defined as `2 * stride` (~319 m) rather than the measured 326.8 m receptive
field, so new outputs stay directly comparable to the existing files.
`--patch-size-m` overrides.

Consequences carried into the design:

1. **Footprint areas are quantized, with a hard floor of one cell** — 10.2 ha in
   `patch` mode, 2.5 ha in `stride` mode — and grow in 160 m steps. Whether they
   are *biased* relative to real facility extent is not measurable from any file
   in this folder; it needs an independent source or visual inspection in QGIS.
   Stated as such in the stats file rather than asserted.
2. **`--boundary` matters.** Without it, ~45% of the inference area is
   out-of-state, has no known positives, and inflates the "fraction rejected"
   statistic for reasons unrelated to model quality.
3. **Single row group** means no cheap random row access in the parquet
   backend; row selection requires a streamed pass.
4. Environment: Python 3.11.5, geopandas 1.1.3 (use `union_all()`, not the
   deprecated `unary_union`), pandas 3.0.3 (`sjoin(predicate=)`, not `op=`),
   shapely 2.1.2 (vectorized `shapely.from_wkb` / `centroid`), duckdb 1.5.5,
   sklearn 1.9.0, pyarrow 11.0.0. 30 GB RAM, 20 cores.

## Backend abstraction (agnostic to parquet vs DuckDB)

Both storage layouts expose the same four operations, so the rest of the script
is backend-blind.

```
class EmbeddingBackend:
    n_patches: int
    ids: np.ndarray            # row indices (parquet) or tile_id strings (duckdb)
    centroids_ll: (N, 2) f8    # EPSG:4326 lon/lat
    centroids_m: (N, 2) f8     # metric CRS, built once
    def fetch(ids)             -> (k, 384) float32     # small gather, for training rows
    def iter_all(batch_size)   -> yields (positions, X) # one sequential pass, full AOI
```

**`ParquetBackend(embeddings_parquet)`** — the single-file layout of
`ParquetEmbeddingsML`.
- Pass A: `pq.ParquetFile.iter_batches(columns=['geometry'])`, vectorized
  `shapely.from_wkb` -> `shapely.centroid` -> `get_coordinates`, keep only the
  `(N, 2)` float array (~235 MB) and discard the shapely objects. The 14.7 M
  patch polygons never all exist at once.
- `fetch`: streamed pass over feature columns, gathering only the requested rows
  (~12.5 k of 14.7 M).
- `iter_all`: streamed pass over feature columns.
- `--cache-features` holds the 5.6 GB `uint8` matrix in RAM after the first
  feature pass so `iter_all` does not re-read; off by default. Peak RAM without
  it is ~2-3 GB.

**`DuckDBBackend(centroids_parquet, duckdb_path, table)`** — the layout of
`DuckDBEmbeddingsML` / `Build-Duck-assets.py`.
- Centroids from `centroids.parquet` (`tile_id` + point geometry).
- Connection opened `read_only=True` (the file may be mid-download / have a WAL).
- `fetch`: register the wanted `tile_id`s as an Arrow table and `JOIN`, one scan.
- `iter_all`: **`con.execute("SELECT tile_id, * FROM embeddings").fetch_record_batch(batch_size)`
  — a single sequential scan**, mapping `tile_id` -> centroid row via a dict
  built once. This replaces `duckdb_ml_utils.get_vectors`'s
  `WHERE tile_id IN (10k-tuple)` per batch, which is a full table scan per batch
  (1,468 scans of a 6 GB table for the full AOI).
- Verifies `COUNT(*)` against `len(centroids)` and warns on mismatch.

Selection: `--embeddings PATH.parquet` picks the parquet backend;
`--centroids PATH.parquet --duckdb PATH.db [--table embeddings]` picks DuckDB.
Exactly one of the two must be given.

Geometry is **synthesized from centroids in the metric CRS in both backends**, so
the two produce identical outputs. Stride is auto-detected as the median
nearest-neighbour centroid distance over a random sample (159.4 m here) and
reported; `--patch-size-m` / `--stride-m` override it.

## Pipeline

### Step 0 — setup
- Metric CRS: `--metric-crs`, default `estimate_utm_crs()` on the data bbox
  (EPSG:32614 for Kansas). Kansas spans UTM zones 13-15, so expect <=0.5% area
  error at the edges and up to ~4 deg of square rotation for out-of-zone patches
  — immaterial given 50% patch overlap. EPSG:5070 is the equal-area alternative.
- All distances and areas in metres / hectares from here on.
- `cKDTree` on `centroids_m` (~235 MB for the full set) rather than a 14.7 M
  element `GeoSeries.sindex`: faster, lighter, and metrically correct — degree
  space is anisotropic at 37-40 deg N, so `sindex.nearest` in EPSG:4326 subtly
  distorts the 3 km rule.
- `--boundary`: restrict the working set of patches to those whose centroid
  falls inside, before sampling and inference. Applies to both.

### Step 1 — positives
- Read `--positives`, take centroids of whatever geometry is supplied, drop all
  non-geometry columns, reproject to the metric CRS.
- Snap each to the nearest patch centroid via the KDTree.
- **`--max-snap-dist-m` (default `1.5 * stride` ~= 240 m)**: drop positives
  farther than this from any patch, and report the count. `sindex.nearest` /
  KDTree never fail; a point outside the embedding extent silently attaches
  itself to an edge patch, polluting training and inflating the recall
  denominator. Points inside the AOI are at most ~113 m (stride cell
  half-diagonal) from a centroid, so the default is generous but safe.
- **Dedup for training**: several positives can snap to one patch. Train on
  unique patches (a duplicated row is unintended sample weighting) but keep all
  original points for step 6/7 accounting.

### Step 2 — negatives
- Sample `round(--neg-ratio * n_pos_unique)` patches uniformly without
  replacement from the working set (default ratio **10.0**, seeded by `--seed`).
- Reject any whose centroid is within **`--neg-min-dist-m` (default 3000)** of
  any positive point, via `KDTree.query_ball_point`; oversample and top up until
  the target count is met or the frame is exhausted.
- Note recorded in the config file: the 3 km rule removes ~12.5% of the
  sampling frame and specifically the facility-adjacent hard negatives. For a
  detection task that would be a bias; for *footprint* generation it is the
  desirable one — we want the model to fire on the scruffy edges of a yard
  rather than to have learned sharp boundaries from near-misses. Shrinking this
  toward ~500 m *might* yield tighter, more fragmented polygons — **untested
  supposition, never measured**; worth one exploratory run if tighter footprints
  are ever wanted. Note that no run has yet produced footprints smaller than the
  reference polygons at all: the minimum area ratio observed across every
  experiment is 1.09.

### Step 3 — model
- `--model {logreg,mlp}`, default `logreg` =
  `LogisticRegression(max_iter=--max-iter (1000), class_weight=--class-weight (None))`.
  `mlp` = `MLPClassifier(hidden_layer_sizes=--hidden-layers (64,16), max_iter=1000, n_iter_no_change=40)`,
  included because past runs used it; expect saturated probabilities there
  (hence the historical t=0.999) and well-calibrated ones from logreg.
- **No feature scaling**, matching both notebooks (raw quantized `uint8`).
  Non-convergence is caught and written to the config file as a warning rather
  than swallowed.

### Step 4 — threshold
- **`--cv-folds` (default 5) stratified CV -> out-of-fold probabilities for all
  ~12.5 k labeled patches**, then refit on everything for inference. With only
  1,133 positives, an 80/20 split would both bias the reported metrics (the
  threshold is picked on the same 227 points it is scored on) and be
  high-variance. OOF is the better use of the labels; the residual optimism
  (threshold tuned on the set it is reported on) is stated in the config file.
- Candidate thresholds: unique OOF probabilities from
  `precision_recall_curve`, so the peak is found exactly rather than on a 0.025
  grid.
- **Plateau rule**: pick the highest threshold in the *contiguous* run of
  within-`--plateau-tol` (default 0.005) thresholds **that contains the F-beta
  maximum** — the automated form of "the rightmost edge of the central plateau".
  `--plateau-tol 0` gives the literal argmax. Favouring the high edge also
  favours tighter footprints.

  The contiguity requirement was added 2026-08-11 after New Mexico exposed the
  original rule (highest within-tolerance threshold *anywhere*) jumping across a
  dip into a disconnected bump: the eligible set split into [0.238..0.492],
  containing the peak, and [0.502..0.624], and the old rule chose 0.624. All
  Kansas runs are unaffected (their eligible sets are effectively one run), so
  the change is a correctness fix rather than a retuning. Note it moved New
  Mexico's threshold *down* to 0.492, which makes that AOI's over-detection
  worse — the rule was never the cause of that failure. The number of
  disconnected runs is now reported in the config file, since a ragged curve is
  itself evidence of a weakly separating model.
- Warn if the selected threshold exceeds `1 - 1e-6` (degenerate saturation,
  expected only for `mlp`); in that case also report the equivalent
  decision-function margin so the operating point is still interpretable.
- `--beta` (default 1.0). Lower beta -> precision-favouring -> higher threshold
  -> tighter polygons; **try `--beta 0.25` against the bloat concern.**
- PNGs written: `F1`, `F0.25`, `F{beta}` if beta is neither, plus a
  precision-recall curve. Curves are F-beta vs threshold, as in the notebooks.

### Step 5 — inference
- One `iter_all` pass over the working set; per batch `predict_proba`, keep rows
  above threshold, accumulate positions + probability. Casting `uint8` -> float
  is per batch, so RAM stays flat. `--batch-size` default 200,000.
- The model evaluation itself is a single ~5.6 GFLOP matmul; I/O dominates.

### Step 6 — filter against known positives
**Merge first, then keep a polygon if it contains a known positive point, or
lies within `--match-tol-m` of one.** Default tolerance = `1 * stride`
(~160 m); `0` gives strict containment.

Rationale: containment is parameter-free and self-scaling. A 440 ha facility
spans ~40 patches; a patch-level "within D of the point" rule with D = 500 m
truncates it, while a D large enough to keep it (>=1.5 km) sweeps in unrelated
blobs. The tolerance exists because the input points are centroids: for an
L-shaped or ring-shaped yard, or when the model misses the single central patch,
the centroid can fall in a hole or just outside the blob, and strict containment
would silently discard a correct footprint.

Merge details:
- Union performed in the **metric CRS with a real metre buffer**
  (`--merge-buffer-m`, default 2 m, mitre join), not the notebooks'
  `0.00001 deg` in lat/lon. Patches from adjacent MGRS tiles carry different
  rotations, so degree-space seams can fail to close.
- `--footprint-geometry {patch,stride}`, default `patch`:
  - `patch` — each detection contributes a `2 * stride` (~319 m) square,
    reproducing the `centroid.buffer(160, cap_style=3)` convention behind the
    existing outputs. Cell floor 10.2 ha.
  - `stride` — each detection contributes only its 159.4 m stride cell. Cells
    tile exactly, so connectivity is preserved, but an isolated detection
    becomes 2.5 ha rather than 10.2 ha. **Directly targets the bloat problem.**
    Note that this cannot be arbitrated by `--reference-polygons` IoU: the
    reference is itself built from 320 m squares, so `patch` agrees with it by
    construction. Choosing between the modes needs visual inspection against
    imagery, or an independent footprint source.
- `--gap-close-m` (default 0): morphological closing (buffer out then in) before
  the union, for facilities fragmented across a one-patch gap.
- Polygon `confidence` = mean probability of its member patches (as in both
  notebooks), plus `n_patches`, `area_ha`, `n_positives`.
- Two failure modes reported rather than silently handled:
  - **Merge collision** — two facilities 300 m apart become one polygon matched
    by two points. Reported as a positives-per-polygon distribution.
  - **Fragmentation** — a facility splits and only the point-containing blob
    survives. Reported as the count of retained polygons having a *rejected*
    polygon within one stride; that is the diagnostic for turning on
    `--gap-close-m`.

### Step 7 — statistics (`*_stats.txt`)
- AOI: patch count, stride, patch size, metric CRS, boundary applied.
- Positives: input count, dropped for snap distance, dropped outside boundary,
  unique patches used for training.
- Negatives: requested, rejected by the 3 km rule, final count, realized ratio.
- Model at the selected threshold, from **OOF** predictions: confusion matrix,
  precision, recall, F1, F-beta, plus per-fold spread.
- **Expected AOI-wide false positives = OOF FPR x n_patches.** At a 10:1 sample
  the negative prior is ~9%; over the full AOI it is ~1e-4, so an FPR of 0.1%
  looks excellent and means ~14,700 false patches statewide. The threshold is
  therefore not calibrated to the inference run. For this task that is mostly
  harmless — step 6 discards strays — the only cost being bloated footprints
  adjacent to true positives. Stated explicitly so nobody reads the validation
  precision as a field precision.
- Raw inference: detected patch count, % of AOI, polygon count before filtering.
- Filtering: polygons retained / rejected, patches retained / rejected, and both
  as fractions.
- **Re-detection reported three ways**, since "fraction of positives
  re-detected" is ambiguous:
  1. the snapped patch itself fires;
  2. any patch within one ring fires (the `predict_w_nbors` criterion);
  3. the positive is matched by a retained polygon (the headline number).
  Each alongside the honest OOF figure — every positive is in the training set,
  so the in-sample re-detection rate flatters the model relative to what it
  would do on unseen facilities.
- Areas (ha): count, sum, min, p10, median, mean, p90, max; patches-per-polygon
  distribution. With the caveat that areas are quantized in 160 m steps with a
  floor of 10.2 ha (`patch`) / 2.5 ha (`stride`), and that absolute area
  accuracy is not assessable from the inputs.

### Step 7b — evaluation against reference polygons (`--reference-polygons`)

A first-class, reusable module, not an afterthought: this is how each new
geography gets judged, and the harder AOI to come will be scored the same way.
Reported to `*_eval.txt`.

**Correspondence** (spatial join on intersection, both directions, since one
facility can fragment into several output polygons and two facilities can merge
into one):
- `n_reference`, `n_output`.
- Reference polygons with >=1 intersecting output polygon -> **recovery rate**.
- Output polygons with >=1 intersecting reference -> **agreement rate**;
  the complement is the count of output polygons with no reference counterpart.
- Correspondence classes: 1:1, 1:many (fragmentation), many:1 (merge collision),
  0:1 and 1:0 (missed / novel), each with counts.

**Overlap quality:**
- Per-reference **best-match IoU**: mean, median, p10, p25, p75, p90.
- Per-reference **union IoU** — reference against the union of all its
  intersecting output polygons; the fragmentation-robust version, and the one I
  would treat as the headline.
- **Global IoU** = total intersection area / total union area over all
  geometries; matching-free and insensitive to the classification above.
- Per-reference **coverage** (intersection / reference area) and **excess**
  (output area outside the reference / output area), which separate "we missed
  part of it" from "we spilled beyond it".
- Area ratio (matched output area / reference area): median and total. Read as
  agreement with the prior grid-based run, **not** as area accuracy.

**Companion geojson for QGIS** (the folder already has `AutomatedModeling.qgz`),
each written only when non-empty:
- `*_eval_missed_refs.geojson` — reference polygons with no output overlap.
- `*_eval_novel_polys.geojson` — output polygons with no reference overlap.
- `*_eval_matched.geojson` — output polygons with `ref_id`, `iou`, `coverage`,
  `excess` attached, for sorting by worst IoU.

Caveat recorded in the file: this comparison is in-sample (every positive point
is in the training set) and the reference is itself model output on the same
grid, so high agreement demonstrates reproduction of the prior workflow, not
independent accuracy.

### Step 8 — outputs
Into `--outdir` (default `./runs`), basenamed `{--tag}_{ISO timestamp}`:

| File | Contents |
|---|---|
| `*_detections_raw.geojson` | all detected patches, **as centroid points** + `probability` (can be 10^5-10^6 features; points keep the file small, as the notebook comment notes). `--raw-as-polygons` to write squares instead |
| `*_patches_filtered.geojson` | patches belonging to *retained* polygons, as squares + `probability`, `poly_id` |
| `*_footprints.geojson` | final merged polygons + `confidence`, `n_patches`, `area_ha`, `n_positives` |
| `*_config.txt` | every CLI parameter, input paths with sizes and short hashes, backend used, detected stride/patch size, metric CRS, selected threshold (+ margin), **OOF confusion matrix**, CV metrics, library versions, and any warnings |
| `*_stats.txt` | step 7 |
| `*_eval.txt` + `*_eval_{missed_refs,novel_polys,matched}.geojson` | step 7b, when `--reference-polygons` is given |
| `*_fbeta_F1.png`, `*_fbeta_F0.25.png`, (`*_fbeta_F{beta}.png`), `*_pr.png` | curves |
| `*_model.joblib` | model refit on all labeled data |
| `*_labels.geojson` | constructed labeled set: geometry, `int_class`, patch id, OOF probability — so any run is auditable and reproducible |

All geojson in EPSG:4326.

## CLI

```
python build_footprints.py \
  --positives USA_KS1133_2025-05-02_centroids.geojson \
  --embeddings 20230101-20240101_USA_Kansas-deduped.parquet \
  --boundary usa_kansas.geojson \
  --tag ks_feedlots

# or, DuckDB backend for large AOIs:
python build_footprints.py \
  --positives USA_KS1133_2025-05-02_centroids.geojson \
  --centroids centroids.parquet --duckdb embeddings.db \
  --boundary usa_kansas.geojson --tag ks_feedlots
```

| Flag | Default | Purpose |
|---|---|---|
| `--positives` | required | point (or any) geometry file of known facilities |
| `--embeddings` | — | parquet backend |
| `--centroids` / `--duckdb` / `--table` | — / — / `embeddings` | DuckDB backend |
| `--boundary` | none | clip the working patch set |
| `--outdir` / `--tag` | `./runs` / `run` | output location and basename |
| `--neg-ratio` | `10.0` | negatives per positive |
| `--neg-min-dist-m` | `3000` | min distance from any positive |
| `--beta` | `1.0` | F-beta for threshold selection |
| `--plateau-tol` | `0.005` | highest threshold within tol of peak F-beta; `0` = argmax |
| `--cv-folds` | `5` | stratified CV for OOF probabilities |
| `--model` | `logreg` | `logreg` or `mlp` |
| `--hidden-layers` | `64,16` | mlp only |
| `--class-weight` | `none` | or `balanced` |
| `--max-iter` | `1000` | solver iterations |
| `--footprint-geometry` | `patch` | `patch` (2 x stride, ~319 m) or `stride` (159 m) cells |
| `--patch-size-m` / `--stride-m` | auto-detected | override the grid geometry |
| `--match-tol-m` | `1 * stride` | polygon-to-positive matching tolerance; `0` = strict containment |
| `--gap-close-m` | `0` | morphological closing before merge |
| `--merge-buffer-m` | `2.0` | union tolerance in metres |
| `--max-snap-dist-m` | `1.5 * stride` | drop positives farther than this from any patch |
| `--metric-crs` | auto UTM | override, e.g. `EPSG:5070` |
| `--batch-size` | `200000` | inference batch |
| `--cache-features` | off | hold the 5.6 GB uint8 matrix (parquet backend) |
| `--raw-as-polygons` | off | write raw detections as squares, not points |
| `--reference-polygons` | none | optional IoU validation |
| `--seed` | `42` | sampling and CV |

## Code layout (single file)

```
backends:   EmbeddingBackend (ABC), ParquetBackend, DuckDBBackend, make_backend(args)
geo:        detect_stride, build_squares, to_metric, load_boundary
labels:     snap_positives, sample_negatives, build_label_set
model:      make_model, oof_probabilities, select_threshold, fbeta_curves
inference:  run_inference
polygons:   merge_to_polygons, filter_by_positives
evaluate:   correspond, iou_stats, evaluate_against_reference   # step 7b, reused per AOI
report:     summarize, write_config, write_stats, write_eval, write_outputs
main(args) / __main__ argparse
```

## Hard-negative mining (`--hard-negative-rounds`, default 2) — now core, not optional

**Updated 2026-08-11: this is an expected part of the design, and the default is
2 rounds (three training passes in total).** Random negatives alone leave the
model firing over open country; every acceptable result in this project used at
least one mining round. `--hard-negative-rounds 0` still gives the original
single-pass behaviour for comparison.

Motivated by the finding that uniform random negatives are too easy: they sample
the AOI at 1 patch per 740, so rare look-alikes (dairies, quarries, industrial
yards, bare-soil scrapes) rarely enter training, and the 3 km rule removes the
hardest cases by construction. Raising `--neg-ratio` from 10 to 50 did **not**
help (see results below), so the lever is negative *quality*, not quantity.

Each extra round adds the previous round's **rejected detections** — patches that
fired but fell outside every retained polygon, i.e. the AOI's own hardest
negatives, already localized for free by step 6 — and retrains.

Factoring, chosen so the single-pass path stays trivially revertible:

- `train_model(X, y, args)` — CV, threshold selection, refit. Sees only the
  labeled matrix, nothing about the embedding store.
- `detect_and_filter(env, model, threshold)` — inference, merge, filter. `env`
  holds the run-invariant setup (backend, geometry, positives).
- `mine_hard_negatives(env, result, known, min_dist, cap, rng)` — the only new
  logic.
- `main` loops `range(rounds + 1)` over the three. **`--hard-negative-rounds 0`
  executes exactly one iteration and is byte-identical to the single-pass
  workflow** (verified against a pre-refactor run). Reverting means deleting the
  three functions and inlining one iteration; nothing else changed.

Guards:

- Hard negatives are held at least `--hard-neg-min-dist-m` (default: the same
  `--neg-min-dist-m`, 3 km) from any known positive, so a patch that is really
  part of a *fragmented* facility is not taught as a negative. The count
  excluded by this rule is logged.
- `--max-hard-negatives` (default 50,000) caps the per-round addition; any
  excess is dropped at random and **warned about**, never silently truncated.
- Zero available hard negatives stops the loop early rather than repeating a
  round that cannot change.
- The labels geojson gains a `source` column (`positive` / `random` /
  `hard_r0` / `hard_r1` / ...), and the config file gains a per-round table.
  Metrics elsewhere in the config describe the final round only, which the table
  states explicitly.

## Measured results, Kansas (implemented 2026-08-10)

Parquet backend, `--boundary usa_kansas.geojson`, 1,133 positives, all three runs
reproducible and identical across repeats.

Grid auto-detection confirmed the convention exactly: **stride 160.00 m, cell
320.00 m (10.24 ha)** — the same geometry as the prior `half_width=160` outputs.
The boundary clip cut the AOI from 14,677,576 to **8,371,690 patches (57%)**; the
other 43% was out of state. All 1,133 positives snapped within tolerance.
Negative rejection at 3 km ran at **12.2%** (2,342 of 19,128 drawn), against the
12.5% predicted from buffer-union area. Runtime **~75 s** end to end, peak RSS
~3 GiB; the two feature passes were ~19 s and ~25 s.

| | beta=1 (default) | beta=0.25 | stride, beta=1 |
|---|---|---|---|
| Threshold selected | 0.999853 | 1.000000 (saturated, warned) | 0.999853 |
| Literal argmax would be | 0.881631 | 0.999939 | 0.881631 |
| Detected patches | 18,283 (0.22% of AOI) | 13,433 (0.16%) | 18,283 |
| Merged polygons, raw | 4,778 | 3,071 | 5,062 |
| Retained after filtering | 1,105 | 1,075 | 1,126 |
| Polygons rejected | 76.9% | 65.0% | 77.8% |
| Positives re-detected (polygon) | **99.3%** | 96.3% | 99.3% |
| Union IoU vs reference, median | 0.644 | 0.667 | 0.419 |
| Area ratio vs reference, median | **1.515** | 1.419 | 0.750 |
| Median footprint area | 28.16 ha | 23.04 ha | 10.24 ha |

Findings that matter for the next AOI:

1. **The threshold is not data-determined.** F1 is a flat plateau from ~0.01 to
   ~0.99 on the labeled set (see the linear panel of the F-beta PNGs): every
   threshold in that range scores the same, so sampled negatives cannot pick an
   operating point. The plateau rule is what actually chooses it, landing at
   0.9999 rather than the 0.5-0.8 band a calibrated probe would suggest — the
   linear probe on these embeddings is far more saturated than expected. The
   logit panel added to each PNG is where the real decision is visible.
2. **beta=0.25 ran off the end of the probability scale**, selecting exactly
   1.000000 and firing the saturation warning. It still detects (thresholding
   `>= 1.0` catches the patches whose float64 probability rounds to one), and it
   does tighten footprints — area ratio 1.42 vs 1.52 — but it costs 34
   facilities (96.3% vs 99.3% recovery). Not obviously a good trade.
3. **The implied AOI-wide false positive estimate held up**: 3,694 predicted
   from the out-of-fold FPR against 3,673 polygons actually rejected in step 6.
   Worth keeping as a sanity check on new geographies.
4. **Footprints run ~1.5x the prior run's area** in `patch` mode and ~0.75x in
   `stride` mode, with coverage of the reference at median 1.000 either way. The
   truth is between the two modes, and no file in this folder can say where.
5. Merge collisions are rare (19 polygons covering 2-3 positives each) and
   fragmentation is minor (32 retained polygons with a rejected neighbour within
   one stride), so `--gap-close-m 0` is a reasonable default here.

### Hard-negative mining, Kansas (`runs/ks_hard2_2026-08-11T0007_*`)

`--hard-negative-rounds 2`, all other defaults. Round 0 is the single-pass run.

| round | negatives | threshold | detections | polys raw | polys kept | rejected patches | redetected | mined |
|---|---|---|---|---|---|---|---|---|
| 0 | 11,330 | 0.999853 | 18,283 | 4,778 | 1,105 | 6,750 | 1,125 | 5,127 |
| 1 | 16,457 | **0.712830** | 7,642 | 1,114 | 991 | 151 | 993 | 33 |
| 2 | 16,490 | **0.585411** | 7,816 | 1,157 | 1,035 | 153 | 1,038 | 0 |

Round 0 → round 2, against the reference polygons:

| | single pass | 2 rounds |
|---|---|---|
| False positive patches | 6,750 | **153** (44x fewer) |
| Rejected polygons | 3,673 (76.9%) | 122 (10.5%) |
| Area ratio, median | 1.515 | **1.000** |
| Area ratio, total | 1.494 | 1.047 |
| Union IoU, median | 0.644 | **0.940** |
| Union IoU, mean | 0.624 | 0.858 |
| Median footprint | 28.16 ha | 15.36 ha |
| Positives recovered | **99.3%** | 91.6% |

1. **The threshold moves into the 0.5-0.8 band** — 0.9999 → 0.713 → 0.585 —
   exactly where a calibrated logistic regression was expected to sit. Hard
   negatives de-saturate the probe: once real look-alikes are in the training
   set the problem stops being linearly separable, probabilities spread out, and
   the F-beta curve acquires a genuine peak instead of an eight-decade plateau.
   The plateau rule stops doing the deciding.
2. **The bloat is essentially gone**: median area ratio 1.515 → 1.000 and median
   union IoU 0.644 → 0.940 against the prior workflow's polygons.
3. **The cost is recall: 99.3% → 91.6%**, i.e. 95 facilities unrecovered
   against 8 before. **The mechanism matters**: any real facility *absent from
   the known-positive list* becomes a mined hard negative, and since feedlots
   resemble feedlots, teaching one as negative suppresses the others.
   **So hard-negative mining is only safe to the extent the positive list is a
   complete inventory of the AOI.** The `--hard-neg-min-dist-m` guard (which
   withheld 1,623 rejected patches in round 0) protects the *known* facilities
   but can do nothing about unknown ones.
4. Almost all the gain lands in round 1; round 2 mined only 33 more negatives
   and partly walked recall back up (991 → 1,035 kept). One round is the sensible
   default if this is used.
5. The AOI-wide false-positive extrapolation becomes meaningless once mining
   starts, because the negatives are no longer a random sample of the AOI: it
   reads 97,475 predicted against 153 observed. The stats file now says so
   explicitly instead of printing a misleading ratio.

### Budgeted hard negatives: the recommended operating point

`--max-hard-negatives 150`, matching hand practice of mostly random negatives
plus a couple of hundred selected hard ones. Runs `ks_h150rand_*` (1 round) and
`ks_h150x2_*` (2 rounds).

| | single pass | 150 x1 | 150 x2 | unlimited (5,127) |
|---|---|---|---|---|
| Final threshold | 0.999853 | **0.752757** | 0.381010 | 0.585411 |
| Detections | 18,283 | 14,015 | 12,924 | 7,816 |
| FP patches | 6,750 | 3,428 | 2,798 | 153 |
| **Positives recovered** | 99.3% | **100.0%** | **100.0%** | 91.6% |
| Union IoU, median | 0.644 | 0.673 | 0.750 | 0.940 |
| Area ratio, median | 1.515 | 1.455 | 1.333 | 1.000 |

**A small budget is strictly better than unlimited mining on the metric that
matters most.** 150 hard negatives recover *all* 1,133 facilities — better than
the single pass (8 missed) — while halving false positives and de-saturating the
threshold into the 0.5-0.8 band. Unlimited mining buys tighter footprints at the
cost of 95 facilities, because at that volume the mined set inevitably contains
unlisted real facilities, and teaching those as negatives suppresses the listed
ones too. Expect unlisted facilities as the normal case.

Recommended default for a new AOI: **`--hard-negative-rounds 1
--max-hard-negatives 150`**. A second round trades a little more tightness
(area ratio 1.455 → 1.333, IoU 0.673 → 0.750) for a threshold drifting low
(0.38), which is worth watching but did not cost recall here.

Two selection knobs (`--hard-negative-selection dispersed`,
`--hard-neg-max-patches`) were built, measured, and then **removed** on
2026-08-11. Both were unnecessary and one was misconceived:

- Dispersion is automatic at this budget: uniform random already drew 147
  distinct polygons out of 150 patches, because the rejected population is 31%
  single-patch and 26% two-patch.
- Filtering out large blobs was justified as avoiding unlisted real facilities,
  but large rejected clusters are just as often **genuine** negatives worth
  learning from — suburban construction was the example that settled it — so
  size filtering would discard the most useful hard negatives. Sampling is now
  uniform over everything available, with no shape or size preference.

`--max-hard-negatives` now defaults to **150**, so enabling
`--hard-negative-rounds` gets the validated operating point rather than the
unlimited-mining one that costs 8% of recall. The round table in the config file
records both how many were mined and how many were available, so the sampling
fraction is never silent.

## Measured results, New Mexico (2026-08-11): a label-quality failure

DuckDB backend (`NMembeddings.db` / `NMcentroids.parquet`), 15,747,636 patches,
12,334,671 inside the state boundary. Targets: beef and dairy cattle facilities.
591 positive points, hand-cleaned and augmented; 379 reference polygons from a
prior run, on the same 320 m grid.

| | 591 positives | 591 + fixed round-0 t | **426 reference-backed** | Kansas (ref) |
|---|---|---|---|---|
| Peak F1 (out-of-fold) | 0.8013 | 0.8148 | **0.8773** | 0.9623 |
| Final threshold | 0.6244 | 0.9998 | 0.5522 | 0.7528 |
| Detections | 181,082 (1.47%) | 126,184 (1.02%) | **14,556 (0.118%)** | 0.17% |
| Raw polygons | 36,834 | 26,573 | **4,317** | 3,343 |
| Retained | 314 | 308 | 337 | 1,118 |
| Largest footprint | 47,099 ha | 31,585 ha | **719 ha** | — |
| Total area | 214,400 ha | 178,430 ha | **28,140 ha** | — |
| Area ratio, median | 6.75x | 6.33x | **1.510x** | 1.455x |
| Union IoU, median | 0.143 | 0.156 | **0.633** | 0.673 |
| Outputs with no reference | 70 | 69 | **0** | 0 |
| References recovered | 98.2% | 97.4% | 93.9% | 100% |

**The cause was contaminated positives, not any modeling parameter.** Diagnosis
path, in order:

1. The first run reported **100% of positives re-detected and 98.2% of
   references recovered** while producing garbage: 12x the reference area, a
   471 km^2 polygon containing 71 positives, IoU 0.143. Recovery statistics
   cannot see this failure — one giant blob "recovers" everything it covers.
2. Two hypotheses were tested and **disproved**. A fixed round-0 threshold of
   0.99 changed round-0 detections by 1% (495,277 vs 489,695): at round 0 the
   probabilities are saturated, so the threshold has almost no leverage. Raising
   random negatives from 5,910 to 10,000 moved peak F1 by 0.013.
3. The real signal was in the labels: **84 of 591 positives (14.2%) had
   out-of-fold probability below 0.01** — the model actively rejected them —
   against 1.9% in Kansas. Splitting by provenance located it exactly:
   **70 of 165 hand-added points (42.4%) were rejected, versus 14 of 426
   reference-backed points (3.3%).** A 13x difference. Beef/dairy heterogeneity
   was a minor secondary factor (44% of the 54 dairy points rejected, but they
   are only 24 of the 84).
4. Retraining on the 426 reference-backed positives alone reproduced Kansas-grade
   output (table above) with no other change.

So ~12% bad labels overall was enough to inflate footprint area 12x while every
headline metric still read as success. Conditions under which client-augmented
points are workable, stated as a check rather than a guess: **the share of
positives with out-of-fold probability below ~0.01 must stay under a few per
cent** (Kansas 1.9%, cleaned New Mexico 3.3%, failed New Mexico 14.2%).

**Nothing in the input point file predicts label quality.** The New Mexico
positives carry only `asset_identifier` (family plus a sequence number) and
geometry. Family carries weak signal — dairy 59% not-ok against beef 20.5% — but
dropping dairy wholesale would discard 22 good points to remove 32 bad ones and
still leave 110 bad beef points. The sequence number is near-useless
(r = +0.13 with badness; beef quintile 4 spikes to 48% not-ok, suggesting the
hand-added block sits mid-sequence, but quintile 5 falls back to 26%). The only
clean separator found was reference-polygon backing, which by construction will
not exist for newly contributed points.

Two consequences for the client handoff: the out-of-fold check is the only
generally available quality signal, since it needs no input property; and it is
worth **asking contributors for a provenance column** (`source` =
surveyed / digitized / inferred, or a confidence score) so runs can stratify the
rejection rate per source and name which batch is bad rather than just that one
is.

`review_rejected_positives.py` triages a point set against a trained probe,
scoring every patch within 3 x stride to separate the failure modes. On the full
591 with the clean 426-point model as judge: 449 `ok`, **48 `mislocated`** (own
patch low, neighbour high — median offset 319 m, i.e. one to two patches, with
east/north offsets given so the point can be dragged), 93 `no_signal` (nothing
nearby scores — wrong location, not a facility, or invisible in the imagery
behind these embeddings), 1 `weak`.

Open refinement: round 1 was the best round here (9,242 detections, 340 retained,
threshold 0.998) and round 2 drifted to threshold 0.552 with 14,556 detections.
The final-round-wins convention costs some quality; selecting the best round on a
stated criterion would be an improvement. Kansas showed the same mild
non-monotonicity.

### Negative count is not a variable worth exposing

Three runs, identical but for the random-negative count, on the clean 426
positives with 2 hard-negative rounds:

| | 4,260 (the 10:1 default) | 10,000 | 16,000 (Kansas per-tile parity) |
|---|---|---|---|
| Final threshold | 0.9151 | 0.5522 | 0.9984 |
| Detections | 10,228 (0.083%) | 14,556 (0.118%) | 7,665 (0.062%) |
| Polygons retained | 341 | 337 | 339 |
| Positives re-detected | 100% | 100% | 99.1% |
| References recovered | 93.1% | 93.9% | 91.8% |
| Union IoU, median | 0.667 | 0.633 | 0.714 |
| Area ratio, median | 1.500 | 1.510 | 1.321 |
| Largest footprint | 658 ha | 719 ha | 650 ha |

Retained polygons span 1.2% across a near-4x change in negative count. Raw
detections vary 2x, but step 6 absorbs nearly all of it. **Keep `--neg-ratio 10`
as the default and do not expose `--n-negatives` to a non-expert user.** One
guard is worth adding: a floor of a few thousand, since 10:1 on a 50-point
contribution would give 500 negatives for an entire state — a regime nothing here
tested. `max(10 x positives, ~4000)`.

Two cautions when reading that table:

- **Peak F1 is not comparable across the columns.** It is highest at 4,260
  (0.9313 vs 0.8773 / 0.8827) purely because fewer negatives means fewer chances
  to be wrong. This is a good illustration of why area ratio and IoU are the
  metrics to steer on.
- **Round-to-round thresholds are erratic even where outputs converge**: round 1
  chose 0.0276 at 4,260 negatives and 0.4158 at 16,000, before round 2 settled at
  0.9151 and 0.9984. Reassuring for robustness, but further evidence the
  threshold is poorly determined per round, and an argument for selecting the
  best round rather than the last.

For scale: negative count moved median IoU by 0.08; label quality moved it from
0.14 to 0.63.

### Label-quality gate (`--drop-positives-below`, default 0.01)

Implemented 2026-08-11. After round 0's cross-validation, positives whose
out-of-fold probability falls below the threshold are excluded and the model is
refitted once, before any inference. Excluded points are written as the *input*
geometry to `*_positives_excluded.geojson` with `oof_probability` and
`redetected_anyway`, for correction and reinsertion. `--label-quality-warn`
(default 0.05) controls the loud warning. Excluded points are **kept as step-6
filter seeds**: a point mislocated by 300 m may sit on a real facility, and
dropping it as a seed would discard a correct footprint. `redetected_anyway`
records which ones got a footprint regardless — 21 of 41 did.

Measured on the contaminated 591-point New Mexico set:

| | no gate | gate 0.01 | gate 0.5 | 426 curated by hand |
|---|---|---|---|---|
| Excluded | — | 41 (6.9%) | 53 (9.0%) | 165 |
| Detections | 1.47% of AOI | 0.48% | 0.34% | 0.12% |
| Area ratio, median | 6.75x | 2.98x | 2.80x | 1.51x |
| Union IoU, median | 0.143 | 0.333 | 0.345 | 0.633 |
| Largest footprint | 47,099 ha | 12,915 ha | 9,633 ha | 719 ha |
| Positives recovered | 100% | 96.3% | 93.9% | 100% |

**The gate halves area inflation with no human input, but is not a substitute for
curation.** Two findings bound it:

1. **Raising the floor does almost nothing.** The out-of-fold distribution is
   sharply bimodal — 451 of 550 survivors above 0.999, only 21 anywhere between
   0.01 and 0.9 — so any floor in that range excludes nearly the same points.
   0.01 to 0.5 bought 12 more exclusions, +0.012 IoU, and cost 2.4 points of
   recall. Keep the default at 0.01.
2. **A probability floor can only catch *isolated* bad labels — when the model
   doing the scoring is itself contaminated.** When a batch of bad points shares
   an appearance, the model learns that appearance as positive and then scores
   those points *high* out of fold, because other members of the same batch in
   other folds support them. **Corrected 2026-08-13: this is a property of the
   contaminated model, not a limit of the embeddings.** Scored by a model trained
   only on trusted points, the same points are rejected outright — see *The
   embeddings can disambiguate* below, where median probability for one
   contaminated batch moves from 1.000 to 0.000. The consequence is that the
   gate's measured 43% sensitivity is a floor set by self-assessment, and a
   seeded approach does far better.

Evidence for a possible improvement: under round 2's sharper boundary, 58 of the
550 retained positives had fallen below 0.01 — more than the 41 the round-0 gate
caught. Re-running the check each round would compound to ~99 exclusions without
changing the threshold. Not implemented.

Note the measured rejection fraction is configuration-dependent: 6.9% at 5,910
negatives versus 14.2% at 10,000 on the same 591 points. The fixed 5% warning
threshold is therefore softer than it looks.

## Round selection: per-round machinery built, criterion now supported by evidence

**Built 2026-08-12.** Every round's model, threshold, detections, polygons and
label snapshot are retained, so any round can be emitted without re-running.
`--select-round last|<N>`; `--save-round-outputs` writes each round's footprints.
A per-round quality table appears in every config file, all columns
reference-free: trained positives, recall(trained), recall(all), detections per
positive, median/max footprint area, ha per positive.

Scoring each round against the reference polygons (via the standalone evaluator)
across two Kansas point sets:

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| **derived 1,133**: ref IoU | 0.667 | 0.769 | **0.818** |
| ha/positive | 42.4 | 38.6 | **37.3** |
| det/positive | 13 | 10 | **9** |
| **client 1,371**: ref IoU | 0.429 | 0.500 | **0.644** |
| ha/positive | 61.4 | 55.4 | **48.3** |
| det/positive | 59 | 29 | **15** |

**A workable reference-free criterion: minimise `ha/positive` subject to
`recall(trained)` not dropping.** It picks the same round as reference IoU in all
four series measured so far (these two, plus New Mexico's round 1 → 0.444 vs
round 2 → 0.667). Interpretation: total footprint area per facility should settle
once the model is right, so inflation above that is bloat. This replaces
compactness, which correlated better across runs (rho +0.895) but is confounded --
it penalises legitimately adjacent facilities.

**RESOLVED 2026-08-12 by five-round runs.** `--select-round auto` is now the
default, and `--hard-negative-rounds` defaults to 4 (five passes) to give the
selector range. Criterion in `select_best_round()`: lowest ha/positive among
rounds whose recall(trained) is within 0.5 percentage points of the best achieved.

| round | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **KS derived 1,133** ha/pos | 42.4 | 38.6 | **37.3** | 38.1 | 41.3 |
| recall(trained) | 99.7% | 100% | 100% | 100% | 100% |
| reference IoU | 0.667 | 0.769 | **0.818** | 0.750 | 0.667 |
| **KS client 1,370** ha/pos | 61.4 | 55.4 | 48.3 | 47.7 | **45.8** |
| recall(trained) | 97.7% | 100% | 100% | 100% | 100% |
| reference IoU | 0.429 | 0.500 | 0.644 | 0.667 | **0.667** |
| **NM ref426** ha/pos | 119.6 | 56.6 | 54.3 | **51.0** | 45.4 |
| recall(trained) | 100% | 99.3% | 100% | **100%** | 97.6% |
| reference IoU | 0.400 | 0.714 | 0.727 | **0.750** | 0.760 |

Two things this settled, both of which were open:

1. **The optimum is not always terminal.** Kansas derived turns at round 2:
   ha/positive rises afterwards and reference IoU falls in step, ending at round
   0's value. Taking the last round would have discarded the whole gain, so the
   criterion is not an elaborate way of saying "last".
2. **The recall constraint binds, and cheaply.** New Mexico's round 4 has both the
   lowest ha/positive (45.4) and the best IoU (0.760), but recall(trained) dips to
   97.6%. The rule therefore picks round 3, giving up 0.010 of IoU to keep 10
   facilities. That is the exchange rate that was previously untested, and it
   argues for keeping the constraint.

Selection agreement with reference IoU: exact on KS derived (0.818), tied on KS
client (0.667), 0.010 below the maximum on NM. The 0.5 pp tolerance was chosen to
ignore one or two facilities of noise while excluding the 2.4-point drop seen in
NM round 4; exact-max eligibility gives the same three answers.

## OPEN QUESTION (filed 2026-08-11): how many rounds, and how to pick without references

Deferred deliberately -- the evidence is suggestive but confounded. Resume here.

**Per-round label gating works.** The gate now runs after every round, since each
round's hard negatives sharpen the boundary and surface more unsupported
positives. On the contaminated 591-point New Mexico set it excluded 52 + 53 + 48
= 153 (25.9%), against the 165 removed by hand, and reproduced hand-curated
footprint quality with no human input: area ratio median 1.500 (hand: 1.510),
union IoU 0.667 (hand: 0.633), largest footprint 640 ha (hand: 719). Headline
recall reads 85.3% of all 591, which decomposes as **100% of the 438 positives it
trained on, plus 66 of the 153 it excluded**.

**But the gate does not converge.** It removes a roughly constant share of the
*remaining* pool each round -- 8.8%, 9.8%, 9.9% -- so exclusions compound rather
than taper; a 4th and 5th round would drop ~48 and ~42 more. It is
self-reinforcing: dropping the hardest positives narrows the class, making the
next-hardest look unsupported. Landing at 153, close to the manual 165, is partly
an artefact of stopping at 2 rounds. **A brake is needed before this goes to a
client.** Options, cheapest first: cap cumulative exclusion (~20-25%) and warn
that the cap bound rather than convergence; require the per-round count to decay;
or gate only in the first and last rounds.

**Round count: evidence favours 3 passes (`--hard-negative-rounds 2`), not 2.**
Kansas 0/1/2 rounds gave IoU 0.644 / 0.673 / 0.750, and unlimited mining 0.940.
New Mexico per-round gating: round 1 gave 0.444, round 2 gave 0.667 (ha/positive
107 -> 65, largest footprint 1,823 -> 640 ha). Monotonic in both. **Confounded**,
though: New Mexico's round 2 also had 48 more positives gated out, so it is not a
controlled test of the round count.

**Reference-free round selection: unresolved.** Across 25 completed runs,
Spearman correlation with reference IoU: median compactness (4*pi*A/P^2) +0.895,
polygons rejected -0.819, detections per positive -0.778, ha per positive -0.700,
largest footprint -0.597, polygons per positive +0.466, top-5 area share -0.413.
Compactness ranks runs nearly as well as the reference does -- **but the
correlation is confounded**: in this sample sprawl and failure always co-occurred.
An AOI with genuinely adjacent facilities (side-by-side dairy complexes, rows of
pens) would merge them into legitimately ungainly polygons and be penalised.
`ks_stride` already shows the pattern: sound by every other measure, penalised on
geometry. Compactness needs validating on a clustered AOI before it can be
trusted as a selection criterion; it may only be a proxy for "New Mexico went
wrong".

To settle any of this properly, per-round footprints need saving (currently only
the final round is written), which is cheap since the geometry is already
computed.

## Files

| file | what it is |
|---|---|
| `automated/build_footprints.py` | The pipeline entry point: argument parsing and the run sequence for steps 1-8 plus the label-quality gate, hard-negative rounds and run assessment. `--help` shows standard and advanced options; grid-geometry and execution flags are `argparse.SUPPRESS`-ed and documented in a comment block in `parse_args`. |
| `automated/footprints/` | The stages, one module each: `backends` (streaming parquet/DuckDB access), `geometry`, `labels`, `modeling` (classifier, out-of-fold probabilities, threshold, curves), `inference` (full-AOI pass, patches to polygons), `pipeline` (one train-infer-filter pass and the mining/admission loops), `assessment`, `reporting`, `util`. Dependencies run one way, `util` -> stages -> `pipeline` -> CLI. |
| `automated/evaluate_footprints.py` | Step 7b. Compares predicted footprints to reference polygons: correspondence counts in both directions, best-match and fragmentation-robust union IoU, matching-free global IoU, coverage vs excess, plus QGIS layers for missed references, novel polygons and per-polygon IoU. **Runnable standalone**, which is how per-round footprints get scored: `python3 evaluate_footprints.py --footprints X_round1_footprints.geojson --reference REF.geojson --outdir . --tag r1`. |
| `automated/review_rejected_positives.py` | Triages a point set against a trained probe, scoring every patch within 3 x stride to separate **mislocated** (own patch low, neighbour high -- reports the east/north offset to drag the point) from **no_signal** (nothing nearby scores; wrong location, not a facility, or invisible in the imagery behind these embeddings). Takes any run's `_model.joblib` as the judge -- use a model trained on *clean* positives so it is not grading its own training data. This is what identified the address-derived points in both states. |
| `automated/smoke_test.py` | One-command end-to-end check, ~1 minute, 23 assertions. Builds a synthetic AOI matching the real grid structure (50%-overlapping 320 m patches on a 160 m stride, 384 uint8 features) with planted facilities **and confusable decoys**, so hard-negative mining is actually exercised rather than early-stopping. Asserts outputs exist, all trained positives recovered, assessment passes, gate excludes nothing on clean data, rounds run, `--select-round` obeys, and **the parquet and DuckDB backends agree exactly**. Run it after changing any default. |
| `scripts/build_duck_assets.py` | Converts embeddings parquets to DuckDB + centroids parquet. Shared with the interactive workflow; replaces the `Build-Duck-assets.py` used during development, which had the same CLI. |
| `src/*.py`, `*.ipynb` | The interactive notebook workflows, including the one this automates. Not imported here, and not to be edited from this side. |

Run outputs (`runs-key-*/`, and the working runs behind the tables below) stayed
in the development folder outside this repo; the paths quoted in this document
refer to those.

## Key runs, matched comparison (2026-08-13) -- `runs-key-2026-08-13/`

Six runs, both AOIs x three cleaning strategies, all on the client's full point
sets with provenance, all at current defaults (4 mining rounds, `--select-round
auto`, per-round and unfiltered outputs). Internally comparable, unlike the
2026-08-11 table below, whose runs predate auto round selection.

| run | excl | adm | round | recall(tr) | ha/pos | max_ha | IoU | area ratio | unfiltered | assessment |
|---|---|---|---|---|---|---|---|---|---|---|
| NM, no cleaning | 0 | -- | 1 | 99.0% | 351.7 | **39,982** | **0.089** | **11.08** | 46.1:1 | **6 of 9 flagged** |
| NM, CV gate | 118 | -- | 4 | 100% | 60.8 | 660 | 0.667 | 1.500 | 5.0:1 | 2 of 11 flagged |
| **NM, seeded (`ei`)** | 12 | 32 | 4 | 99.6% | **45.1** | 584 | **0.785** | **1.125** | **1.4:1** | **all pass** |
| KS, no cleaning | 0 | -- | 0 | 94.2% | 55.2 | 1,961 | 0.462 | 2.167 | 16.6:1 | 1 of 9 flagged |
| KS, CV gate | 274 | -- | 4 | 100% | 45.8 | 622 | 0.667 | 1.500 | 3.3:1 | 2 of 11 flagged |
| **KS, seeded (`ei`)** | 19 | 122 | 4 | 100% | **42.5** | 625 | 0.667 | **1.403** | **2.5:1** | **all pass** |

**Seeded admission wins in both AOIs and is the only strategy that passes every
check.** Decisive in New Mexico (IoU 0.785 vs 0.667, area ratio 1.125 vs 1.500);
in Kansas fidelity ties at 0.667 but seeded is tighter (1.403), has a cleaner
discovery set (2.5:1 vs 3.3:1), and gets there **excluding 19 points instead of
274** -- it keeps 122 facilities the gate discarded.

No cleaning fails badly on New Mexico: a **399 km^2** polygon, 11x the reference
area, while still reporting 99.0% recall(trained). That is the failure the
assessment block exists for, and it raised 6 flags. Kansas degrades more gently
(IoU 0.462, 1 flag), consistent with Kansas absorbing contamination that destroys
New Mexico.

### Admission scores by source, and what they mean

Kansas, trusting `ei` only (1,019 of 1,371):

| source | admitted | median probability | final recall (got a footprint) |
|---|---|---|---|
| `rsl` | 82/104 (79%) | **1.000** | 93.3% |
| `ct_raic` | 3/5 | 1.000 | 100% |
| `state_permit` | 30/203 (15%) | 0.000 | **35.0%** |
| `echo` | 7/39 (18%) | 0.000 | 55.0% |
| `ei` (trusted) | -- | -- | 99.6% |

Three things worth carrying forward:

1. **`rsl` scores like trusted data** (79% admitted, median 1.000), yet excluding it
   from the trusted set costs almost nothing: 93.3% of `rsl` points still get a
   footprint, because points set aside from *training* remain filter seeds. Being
   conservative about trust is cheap.
2. **The same source label means different things in different AOIs.** New Mexico
   `ct_raic` scored median 0.000; Kansas `ct_raic` scores 1.000. Do not port a trust
   list between regions without re-checking.
3. **`state_permit` at 35% final recall is where real loss sits** -- but at median
   probability 0.000, the seed is confident those points are not facilities, so
   most of that is probably correct rather than missed.

**Tested: trusting `rsl` as well** (`--trusted-source ei,rsl`, exploratory run
`ks_seeded_eirsl`, left in `runs/`). It changes almost nothing, and slightly for
the worse:

| | `ei` only | `ei,rsl` |
|---|---|---|
| trusted seed / admitted | 1,019 / 122 of 351 | 1,123 / 45 of 247 |
| trained positives | 1,140 | 1,156 |
| union IoU | 0.667 | 0.667 |
| area ratio | **1.403** | 1.500 |
| unfiltered ratio | **2.5:1** | 2.8:1 |

The trained set grew by only 16 net positives, because the 82 `rsl` points
admission had already let in are simply trusted by assertion instead -- so the
composition barely moved and neither did the output.

**The transferable lesson is the second-order effect.** `state_permit` admission
*rose* from 30 to 36 of 203 once `rsl` joined the seed, despite the seed being
larger and ostensibly better. Every point added to the trusted set widens the
positive class and loosens the bar for everything judged against it. **Trust is not
free even when the trusted points are good**, which argues for keeping the trusted
set to the provenance you are most confident in and letting admission earn the
rest. `ei` only is the recommended configuration.

### CV gate placement: after admission, and it is complementary

Of the 12 points the gate removed in the seeded New Mexico run, **10 were trusted
`ei` points** -- which admission never examines, since it only screens untrusted
candidates. So the two mechanisms partition the problem: admission handles
batch-level provenance failure, the gate catches individual stragglers anywhere,
including inside the trusted set (it removed 2.3% of `ei`, close to the 4.4%
measured independently).

One ordering wrinkle, not currently worth fixing: the 2 *admitted* points the gate
also removed were judged by a weaker model (trained on trusted+admitted,
self-assessed) than the sharpened trusted-only seed that admitted them. At 2 points
this is noise. **The interaction to watch is the per-round trigger**: if a run
admits many points and the gate then exceeds `--label-quality-warn`, per-round
gating engages and compounds at ~10% of the remainder per round over a set
admission has already screened. If that is ever observed, exempt admitted points
from gating and let the stronger judge stand. Not observed yet (New Mexico seeded
gated 2.6%, under the 5% trigger).

## Key runs (2026-08-11, predate auto round selection)

**Read `recall(trained)` as the headline, not `recall(all)`.** The gate removes
positives the embeddings cannot support -- points not co-located with a visible
facility -- and those are precisely the ones we do *not* want recovered. They stay
in the `recall(all)` denominator, which is why that column looks moderate.
Recovery of the points actually kept for training is ~100% throughout.

| run | positives | excluded by gate | trained on | footprints | recall(all) | **recall(trained)** | area ratio | union IoU |
|---|---|---|---|---|---|---|---|---|
| `ks_default` — first Kansas success, single pass, no mining | 1,133 | 0 | 1,133 | 1,105 | 99.3% | **99.3%** | 1.515 | 0.644 |
| `ks_h150x2` — Kansas, 2 mining rounds, no gate | 1,133 | 0 | 1,133 | 1,117 | 100% | **100%** | 1.333 | 0.750 |
| `nm_ref426` — New Mexico, hand-cleaned positives | 426 | 0 | 426 | 337 | 100% | **100%** | 1.510 | 0.633 |
| `ks_conditional` — Kansas, current full stack | 1,133 | 8 (0.7%) | 1,125 | 1,119 | 99.9% | **100%** | **1.174** | **0.818** |
| `ks1371` — Kansas *client record*, current stack | 1,370 | 266 (19.4%) | 1,104 | 1,143 | 90.2% | **100%** | 1.512 | 0.644 |
| `nm_current` — New Mexico, current stack | 591 | 118 (20.0%, capped) | 473 | 399 | 86.6% | **99.4%** | 1.510 | 0.642 |

Area ratio and union IoU are measured against the pre-existing Earth Index
polygons, which are themselves patch-grid outputs with a 10.24 ha floor -- so
they measure agreement with prior work, not absolute accuracy.

The current full stack is: 2 mining rounds, label gate at out-of-fold prob < 0.01
(conditional per-round, 20% ceiling), negative floor 1,000 per million AOI
patches, contiguous-plateau threshold selection, and the run-assessment block.
`ks_conditional` is the best result obtained in the project: 100% of trained
positives recovered at an area ratio of 1.17.

## MLP test, Kansas derived positives (2026-08-12)

`--model mlp --hidden-layers 64,16`, otherwise current defaults (5 passes,
conditional gating, auto round selection). Run `mlp_ks_derived_2026-08-12T1826`.
Runtime **480 s** against ~120 s for logistic regression -- 30 network fits
(5 folds x 5 passes, plus refits) instead of 30 trivial ones.

**Thresholds land in a sane range and the plateau rule works.** Round thresholds
0.635, 0.676, 0.576, 0.570, 0.266; emitted 0.576 with peak F1 0.921 and a plateau
spanning 0.268-0.576 (73 candidates). No saturation, unlike the historical MLP
runs that motivated `t=0.999` in the notebooks -- so automated threshold selection
is viable for this model, which was not obvious in advance.

Label gate: 4 of 1,133 excluded (0.4%), per-round gating correctly not triggered.

| round | recall(trained) | det/pos | median_ha | ha/pos | threshold |
|---|---|---|---|---|---|
| 0 | 100.0% | 19 | 28.2 | 47.6 | 0.635 |
| 1 | 99.9% | 10 | 20.5 | 38.7 | 0.676 |
| **2 (emitted)** | **99.7%** | **9** | **20.5** | **38.1** | **0.576** |
| 3 | 90.8% | 7 | 15.4 | 32.3 | 0.570 |
| 4 | 85.1% | 7 | 15.5 | 30.8 | 0.266 |

**The recall constraint in `select_best_round` is load-bearing here.** Rounds 3 and
4 have much lower ha/positive (32.3, 30.8) but shed 9-15% of the facilities to get
there. On bloat alone the selector would have taken round 4 and silently dropped
168 facilities. With logistic regression the constraint mostly idled; a
higher-capacity model can carve out the accumulated mined negatives at the expense
of nearby true positives, so it earns its place. Implication if MLP ever becomes
the default: it wants fewer mining rounds, or a smaller hard-negative budget.

Emitted round versus the logistic-regression equivalent (`ks_conditional`):

| | logreg | MLP |
|---|---|---|
| Filtered footprints | 1,119 | 1,119 |
| recall(trained) | **100%** | 99.7% |
| Union IoU | **0.818** | 0.792 |
| Area ratio, median | **1.174** | 1.250 |
| Detections | **9,410** | 10,674 |
| **Unfiltered polygons** | 2,254 | **1,948** |
| **Polygons per trained positive** | 2.0:1 | **1.7:1** |
| **Discovery candidates (unretained)** | 1,135 | **829** |
| Assessment | all pass | all pass |

**Verdict (client review, 2026-08-12): MLP is a win for discovery**, and the extra
area over known positives is not considered a drawback. Slightly behind on
footprint fidelity but comparable rather than inferior. The 1.7:1 ratio sits
inside the 2:1 usability threshold with 27% fewer candidates to review.

### MLP validated on New Mexico (2026-08-13)

`mlp_nm_ref426`, the 426 reference-backed positives, five passes, otherwise
current defaults. Runtime 560 s. Auto-selection emitted round 2.

| round | recall(trained) | det/pos | median_ha | ha/pos | threshold |
|---|---|---|---|---|---|
| 0 | 100.0% | 42 | 48.6 | 91.4 | 0.577 |
| 1 | 100.0% | 17 | 38.4 | 55.9 | 0.619 |
| **2 (emitted)** | **99.5%** | **14** | **35.8** | **50.8** | **0.637** |
| 3 | 97.6% | 12 | 33.3 | 48.0 | 0.471 |
| 4 | 94.1% | 10 | 30.7 | 43.6 | 0.598 |

**What replicated:**

- **Late-round recall erosion is an MLP property, not a Kansas artefact.** Kansas
  100 → 85.1%, New Mexico 100 → 94.1%, both monotonic from round 2, both with
  their lowest ha/positive at round 4. Auto-selection stopped at round 2 in both
  because of the recall constraint. Milder here but the same shape. **Do not
  override `--select-round auto` with an MLP.**
- **Thresholds stay well-behaved**: 0.47-0.64 here, 0.58-0.68 in Kansas. No
  saturation in either. Automated threshold selection is viable for this model,
  which was the open doubt.
- Assessment passes all 10 checks in both states.

**Discovery: MLP wins, confirmed in both states (settled 2026-08-13).** Compare
each model at *its own auto-selected round* -- comparing against whatever round
happened to be emitted produced a wrong answer once already (see below).

| AOI | logreg (its optimum) | MLP (its optimum) |
|---|---|---|
| Kansas derived | round 2: 2,254 polygons, **2.0:1** | round 2: 1,948, **1.7:1** |
| New Mexico 426 | round 3: 988 polygons, **2.4:1** | round 2: 740, **1.7:1** |

MLP lands at 1.7:1 in both AOIs, a 15-29% smaller candidate set. In New Mexico it
is also *better* on footprints (union IoU 0.777 vs 0.750, area ratio 1.222 vs
1.250) at 99.5% recall against logreg's 100%.

**Methodological note, worth remembering.** The first New Mexico comparison
concluded "no advantage" because it measured logreg at its **last** round (4:
4,799 detections, 706 polygons) rather than its **optimal** round (3: 5,987
detections, 988 polygons) -- a 40% error in the polygon count, in the direction
that erased the effect. Any cross-run comparison must use each run's selected
round. Several runs in `runs-key-2026-08-11/` predate `--select-round auto` and
therefore hold last-round output; their headline numbers are not necessarily at
their optimum.

Footprint fidelity is a genuine wash across the two AOIs: MLP was worse in Kansas
(union IoU 0.792 vs 0.818) and better in New Mexico (0.777 vs 0.750), at
comparable bloat and within one facility of the same recall.

**Verdict: MLP is a viable alternative, not a win.** Four times the runtime, no
discovery advantage, a recall-erosion pattern the linear model does not have, and
a coin-flip on footprint fidelity. Its value is as a cross-check, and as evidence
that the pipeline is not tied to a linear probe.

## Provenance: the `source` field (2026-08-13)

The client's `*_for_emission_modeling_*.csv` files carry a `source` column that
the derived point geojsons dropped. It resolves the provenance question the
geometric analysis could only approximate.

| source | New Mexico (591) | Kansas (1,371) |
|---|---|---|
| `ei` (Earth Index) | 427 | 1,019 |
| `ct_raic` | 116 | 5 |
| `echo` (EPA ECHO) | 48 | 40 |
| `state_permit` | — | 203 |
| `rsl` | — | 104 |

Conversion from CSV to geojson validated: row counts and identifier sets match
exactly, maximum positional error **0.0000 m** (differences ~1e-13 degrees, float64
noise), and the CSV's own WKT geometry agrees with its latitude/longitude columns.
No lat/lon swap, no CRS error. 50 of 52 columns were dropped, which is the only
information loss.

**New Mexico, overlap with the Earth Index reference polygons:**

| source | n | inside a polygon | % | within 320 m | median distance | polys hit |
|---|---|---|---|---|---|---|
| `ei` | 427 | 407 | **95.3%** | 99.1% | **0 m** | 351 |
| `echo` | 48 | 11 | 22.9% | 37.5% | 991 m | 10 |
| `ct_raic` | 116 | 8 | **6.9%** | 13.8% | **2,942 m** | 8 |
| all | 591 | 426 | 72.1% | 77.3% | 0 m | 352 |

Distance quantiles separate the three populations cleanly: `ei` is at 0 m through
p90; `echo` runs 16 m (p25) to 13 km (p90); `ct_raic` runs 907 m (p25) to 29 km
(p90). `ct_raic` is not a geocoding offset — it is a different set of places.

**Rejection rate by source** (out-of-fold prob < 0.01, contaminated 591-point
model): `ei` 4.4%, `ct_raic` 32.8%, `echo` 56.2% — a 13x spread, with `ei` at
essentially the clean-data baseline. The label gate was detecting source, not
noise in the abstract.

Two incidental findings: `location_CI` is `high` for all 591 and all 1,371, so the
register's own confidence flag carries no signal. And 44 of 48 `echo` points are
dairies against only 10 in the entire `ei` set, so the "dairy is harder" effect
measured earlier was substantially a source effect in disguise.

The hand-built 426 set approximated `source == ei` but was not identical: 407
points are both, 20 are `ei` but not reference-backed (small hand movement during
the client's disambiguation), 19 are reference-backed but not `ei`.

## The embeddings can disambiguate; contamination captures the boundary (2026-08-13)

Client review of the New Mexico points that fall outside any Earth Index polygon:
the `echo` ones are, without exception, not co-located with cattle facilities; the
`ct_raic` ones are mostly former cattle facilities, horse facilities and auction
yards.

Scoring those 145 expert-judged non-facilities two ways — the difference is only
whether *sibling* points from the same source were in training:

| source | n | | median probability | >= 0.99 | < 0.01 |
|---|---|---|---|---|---|
| `ct_raic` | 108 | contaminated model | **1.000** | 59.3% | 34.3% |
| | | clean model (`ei` only) | **0.000** | 11.1% | **81.5%** |
| `echo` | 37 | contaminated model | 0.000 | 29.7% | 67.6% |
| | | clean model | 0.000 | 2.7% | **94.6%** |

**Same points, same embeddings, median probability 1.000 -> 0.000.** The
embeddings distinguish former feedlots, horse operations and auction yards from
operating cattle facilities perfectly well. The contaminated model did not because
those 108 points were 19% of its positive class and helped draw the boundary they
then sat comfortably inside. The original Earth Index run is the independent
demonstration: it excluded these same places using these same embeddings.

This corrects an earlier and wrong framing in these notes, that appearance
under-determines the target class and the gate's blindness was a limit of the
approach. It is not. Consequences:

1. **The gate's 43% sensitivity is a floor set by self-assessment**, not a
   ceiling. Measured against the expert labels, a clean-seeded model reaches 81.5%
   on `ct_raic` and 94.6% on `echo`.
2. **Contamination *fraction* matters more than severity.** `ct_raic` at 19% of
   positives and near the positive distribution captured the boundary; `echo` at
   8% and far outside it did much less damage.
3. **Per-round gating works for the same reason**, incrementally: each removal
   cleans the boundary a little, which is why later rounds keep finding more.

**This motivates seeded admission** (next to implement): train on the trusted
source alone, score the rest against that model, and admit only points it already
accepts. That is what the hand-built 426 did by accident, and it is a far stronger
screen than one pass of self-assessment.

## Seeded admission (`--trusted-source`), implemented 2026-08-13

Motivated by the finding that a self-assessed gate cannot see a whole batch of
bad points, while a model trained only on trusted points can (median probability
1.000 -> 0.000 on the same 108 points; see *The embeddings can disambiguate*).

**Mechanism.** Name one or more trusted provenance values from `--source-field`
(default `source`). The pipeline trains on those positives alone, **sharpens that
seed with `--seed-rounds` hard-negative rounds (default 2)**, scores every
untrusted positive with the sharpened seed, and admits only those at or above
`--admit-above` (default 0.5). The main rounds then run on trusted + admitted.
Without `--trusted-source` the cross-validated gate is used instead, unchanged.

**The seed's mined negatives are discarded.** `mature_seed()` accumulates them
locally and returns only the model; the main pipeline's labelled set is only ever
*subset* by the admission mask, never extended. The seed is a disposable judge, so
extra seed rounds cost inference passes but leave no residue in training.

**Sharpening the seed is essential, not an optimisation.** New Mexico, 591 client
points, trusting `ei` (427 of them):

| seed rounds | admitted of 164 | `ct_raic` median prob | union IoU | area ratio | unfiltered ratio |
|---|---|---|---|---|---|
| 0 | 69 | 0.316 | **0.533** | 1.873 | 6.9:1 |
| 1 | 35 | 0.000 | 0.750 | 1.188 | 2.0:1 |
| **2 (default)** | 32 | 0.000 | **0.785** | **1.125** | **1.4:1** |

A round-0 seed is *worse than doing nothing* -- it admits 55 of 116 `ct_raic`
look-alikes at median 0.316 and lands at IoU 0.533, below both the CV gate (0.642)
and hand curation (0.750). One round fixes most of it; the second still buys IoU
+0.035 and cuts discovery candidates 2.0:1 -> 1.4:1. Admission counts barely move
between 1 and 2 rounds (35 -> 32), so the second round improves the *final model*
via sharper mined negatives rather than changing who is admitted. 3 rounds
untested; `det/pos` curves flatten after round 2 everywhere, so expect little.

**Seeded admission beats hand curation on New Mexico** -- the best result obtained
by any method:

| method | trained | recall(trained) | union IoU | area ratio | unfiltered ratio |
|---|---|---|---|---|---|
| hand-cleaned 426 | 420 | 100% | 0.750 | 1.250 | 2.4:1 |
| CV gate on 591 | 473 | 99.4% | 0.642 | 1.510 | -- |
| **seeded, 2 rounds** | **447** | **99.6%** | **0.785** | **1.125** | **1.4:1** |

It keeps 32 points hand curation discarded wholesale, and its 1.4:1 discovery
ratio beats MLP's 1.7:1 using the default linear model.

**Reading `recall(all)` under seeded admission.** It falls as admission gets
stricter (91.5% at 0 seed rounds, 81.7% at 1, 80.0% at 2) purely because
set-aside points stay in the denominator. `recall(trained)` holds at 99.6-100%
throughout, and the set-aside points are the ones that should not be recovered.
The declining number is the mechanism working; it is the one figure here that
could mislead a reader.

**Definition of `recall(trained)`**, since it feeds round selection: the fraction
of *trained positive patches* that a retained polygon matched, computed as
`|trained positive patches ∩ matched patches| / |trained positive patches|` from
the current labelled set. It must be derived from that set rather than by
subtracting known exclusions, because several mechanisms can remove a positive
(label gate, seeded admission) and any future one would be missed. A smoke-test
assertion fails on any recall above 100%.

## IMPLEMENTED: automatic "this run probably failed" warnings

New Mexico reported **100% of positives re-detected and 98.2% of references
recovered** while producing unusable output: 12x the reference area, a single
471 km^2 polygon swallowing 71 positives, median union IoU 0.143. Recovery
statistics cannot detect this, because one giant blob trivially "recovers"
everything it covers. The run needs to say so itself rather than relying on
someone reading the area distribution.

Candidate signals, with observed good/bad values from the two AOIs:

| Signal | Kansas (good) | New Mexico (failed) | Suggested warn |
|---|---|---|---|
| **Positives with out-of-fold prob < 0.01** | 1.9% (3.3% NM cleaned) | **14.2%** | > ~5% (**implemented**: the gate excludes them and refits rather than stopping — see below) |
| Detections as % of working AOI | 0.16-0.22% | **1.47%** (4.0% in round 0) | > ~0.75% |
| Retained polygons per positive | 0.99 | **0.53** (314 / 591) | < ~0.7 |
| Largest footprint area | 583 ha | **47,099 ha** | > ~2,000 ha, or > 5x the largest reference polygon |
| Max positives in one polygon | 3 | **71** | > ~5 |
| Share of total area in top 5 polygons | small | **63%** | > ~25% |
| F-beta curve raggedness (disconnected within-tolerance runs) | 1 real run | **2 runs** | > 1 non-degenerate run |
| Peak F-beta itself | 0.99 | **0.80** | < ~0.9 |
| % of raw polygons rejected by step 6 | 65-77% | **99.1%** | > ~95% |

**Built and validated 2026-08-11** as `assess_run()` / `write_assessment()`. The
block prints at the top of every stats file. Validated across every completed run:
the four catastrophic New Mexico runs raise 6-7 flags, every acceptable run raises
0 or 1, and the single flag is always just the gate reporting how many points it
set aside. Two signals were demoted to **context only** after measurement showed
they do not discriminate:

- **Peak F-beta**: good New Mexico runs sit at 0.877-0.883 and failed ones at
  0.801-0.869 -- overlapping, because a harder AOI lowers it legitimately.
- **F-beta curve raggedness**: not even monotonic. The best Kansas run
  (`ks_conditional`, IoU 0.818) shows 4 disconnected plateaus while the failed
  New Mexico runs show 2. Label gating reshapes the probability distribution
  enough to fragment the curve on good runs.

Two thresholds were recalibrated off Kansas-only guesses: detections are now
measured **per known positive** (50) rather than as a share of the AOI, which is
density-dependent and false-flags a dense region; and positives-per-footprint
moved from 5 to 20, because a legitimate New Mexico run merges 8 genuinely
adjacent facilities while failed runs reach 29-71.

Notes on implementation:

- **The label-quality check is the highest-value one and comes for free**: it is
  computable straight after cross-validation, before the expensive inference
  pass, so a contaminated point set can be caught in ~45 s rather than after a
  full run. It is also the check that actually diagnosed New Mexico. Pair the
  warning with a pointer to `review_rejected_positives.py`.
- These are heuristics, so they belong in a clearly-labelled block at the top of
  the stats file (and echoed to stdout), not as hard failures. Wording should be
  "this run looks unreliable because ..." with the offending numbers.
- Raggedness is already computed (`n_eligible_runs` in the threshold selection);
  it needs the degenerate near-zero singleton runs discounted, since Kansas
  shows 3 runs of which 2 are artefacts of saturated probabilities.
- "Largest reference polygon" is only available with `--reference-polygons`, so
  the area check needs an absolute fallback.
- The blob failure is arguably worth *preventing* as well as reporting, via a
  `--max-poly-area-ha` that rejects implausible merges. Discussed but not
  implemented; it would also stop one merge laundering dozens of positives into
  an apparent success.

## Tried and rejected -- do not redo these

Each was built or measured, then dropped. Re-running them is the main way to waste
time here.

| idea | why it was dropped |
|---|---|
| **Compactness** (4*pi*A/P^2) as the round-selection criterion | Best single correlate of reference IoU across 25 runs (rho +0.895), but confounded: in that sample sprawl and failure always co-occurred. It penalises legitimately adjacent facilities, and `ks_stride` already showed the pattern. Superseded by ha/positive. |
| **`--hard-negative-selection dispersed`** | Near no-op at a 150 budget: uniform random already drew 147 distinct polygons from 150 patches, because the rejected population is 31% single-patch. Removed. |
| **`--hard-neg-max-patches`** (skip large rejected blobs when mining) | Premise was that big blobs are unlisted facilities. Wrong: large rejected clusters are just as often genuine negatives worth learning from (suburban construction). Removed. |
| **Raising the label-gate threshold** above 0.01 | Out-of-fold distribution is sharply bimodal (451 of 550 above 0.999, only 21 between 0.01 and 0.9). 0.01 -> 0.5 bought 12 more exclusions, +0.012 IoU, and cost 2.4 points of recall. |
| **More random negatives** | 4,260 vs 10,000 vs 16,000 on clean data moved retained polygons by 1.2% and IoU by 0.08. Do not expose the knob. (On *contaminated* data it does matter -- the floor change helped there -- so the finding is conditional.) |
| **Fixed round-0 threshold** (`--round0-threshold`) | Null result: 0.99 vs the plateau rule changed round-0 detections by 1% (495,277 vs 489,695), because round-0 probabilities are saturated. Flag retained but pointless. |
| **Peak F-beta as a failure signal** | Does not discriminate: good New Mexico runs 0.877-0.883, failed ones 0.801-0.869. Demoted to context-only. |
| **F-beta curve raggedness as a failure signal** | Not even monotonic: the best Kansas run shows 4 disconnected plateaus, failed New Mexico runs show 2. Demoted to context-only. |
| **Detections as a share of the AOI** as a flag | Density-dependent; false-flags a facility-dense area and the synthetic fixture. Replaced by detections per known positive (threshold 50), which separates 12-34 good from 71-306 bad. |
| **Unlimited hard-negative mining** | 5,127 mined negatives cost 8% of recall in New Mexico by mining unlisted real facilities. 150 per round is better on every axis. |

## Operational notes for whoever runs this next

- **Runtimes** (this machine, 20 cores, NVMe): Kansas 8.4M patches ~75 s for 3
  passes, ~4 min for 5; New Mexico 12.3M patches ~4-6 min for 5 passes. MLP is
  ~4x slower. Seed rounds add one inference pass each. The two feature passes are
  I/O bound, ~250-380k patches/s.
- **Do not run two jobs against the same DuckDB file concurrently.** Measured: two
  parallel scans dropped to 93-104k patches/s each, i.e. *less combined* than one
  run alone at 290k. Run them sequentially.
- **`smoke_test.py` is self-contained** -- it generates its own fixture, because an
  earlier fixture living in a scratchpad got wiped mid-session. Run it after
  changing any default; it catches the invariants that have actually broken.
- **Comparing runs**: always compare each run at *its own selected round*. Runs in
  `runs-key-2026-08-11/` predate `--select-round auto` and hold last-round output.
  Also do not compare peak F1 across runs with different negative counts -- class
  balance inflates it.
- **Scoring points with a model**: watch in-sample vs out-of-fold. A model's opinion
  of its own training data is not evidence. `review_rejected_positives.py` must be
  given a model trained on *other* points, and the clean-vs-contaminated experiment
  only works because the trusted seed never saw the candidates.

## Data-specific cautions (this client's files)

- `*_for_emission_modeling_*.csv` carry the **`source`** column plus 50 others; the
  derived point geojsons keep only `asset_identifier`. Conversion validated at
  0.0000 m error, so rebuild from the CSVs when you need provenance.
- `feedlot_area_ha` (median 6.2 ha NM, 7.2 ha KS) and `retention_ponds_area_ha`
  exist in those CSVs. **Footprints are not expected to align with
  `feedlot_area_ha`** -- confirmed by the client -- so do not treat it as ground
  truth for area.
- `location_CI` is `high` for every row in both states: no signal.
- The Earth Index reference polygons in this folder are **prior model output**, not
  truth, and are **not necessarily identical to the set delivered to the client** --
  the client also moved points slightly when disambiguating polygons to facilities.
  Discrepancies at the ~5% level are expected and were confirmed as such.

## Deliberately out of scope

- Model ensembling / voting (present in the DuckDB notebook) — a single probe is
  what this workflow calls for; the saved `.joblib` keeps ensembling possible by
  hand.
- Interactive labeling (`GeoLabeler`) — there is nothing to label here.
- Sub-patch footprint refinement. 320 m is the resolution floor of this data;
  tighter boundaries need a different source, not better post-processing.

## Test plan

1. **Kansas, parquet backend, `--boundary usa_kansas.geojson`, defaults, with
   `--reference-polygons USA_KS1133_2025-05-02.geojson`.** Check runtime, peak
   RAM, and that the threshold lands in the expected 0.5-0.8 band for logreg.
   Pass criteria: high recovery rate against the reference, few novel polygons
   after step-6 filtering (by construction there should be almost none), and
   union IoU high enough that we are clearly reproducing the prior workflow.
2. **`--beta 0.25` vs `--beta 1.0`**, both in `patch` mode, compared on the step
   7b metrics. Here the reference comparison *is* informative, because both runs
   use the same cell geometry as the reference — the difference is purely how
   many patches fire, so IoU and excess directly measure whether beta=0.25
   tightens footprints without dropping facilities.
3. **`--footprint-geometry stride`**, one run, judged by eye in QGIS against
   imagery rather than by IoU (see above — the reference cannot arbitrate this).
4. **DuckDB backend** on the same AOI — **PASSED 2026-08-11** with
   `KSembeddings.db` / `KScentroids.parquet` (run `ks_duck2_*`). Every reported
   field matches the parquet run — stride, working AOI, negatives, threshold
   0.999853, 18,283 detections, 4,778 raw / 1,105 retained polygons, 98.3/99.3/
   99.3% re-detection, 51,126.36 ha total — and the symmetric difference between
   the two footprint unions is **0.000000 ha**, i.e. bit-identical rather than
   merely close. 62 s against 75 s for parquet: DuckDB skips the 22 s WKB
   geometry pass and pays it back in a slower row scan (210k vs 250-375k
   patches/s). Its real advantage is never needing the feature matrix resident.
   Note: duckdb 1.5 deprecates `fetch_record_batch` / `fetch_arrow_table` in
   favour of `to_arrow_reader` / `to_arrow_table`; both are now called behind
   `hasattr` shims so older duckdb still works, and the swap was verified to
   change nothing.
5. **Harder geography**, once Kansas passes: same invocation, same step 7b
   report, with or without reference polygons depending on what exists there.
   Nothing AOI-specific should need editing — that is the point of
   auto-detecting stride and the metric CRS.
6. Edge cases: a positive outside the boundary; a positive far outside the
   embedding extent; `--neg-ratio` high enough to exhaust the frame; zero
   detections at an absurd threshold; a reference file that overlaps nothing.
```
