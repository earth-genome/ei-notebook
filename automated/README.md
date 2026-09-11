# Automated footprints from known point locations

You have point locations for objects of interest — a register, a prior survey, an
earlier modelling run — and no polygons for them. This builds a linear probe on
patch-based geo-foundation-model embeddings, runs it across the whole area of
interest, merges the detected patches into polygons, and keeps the polygons that
land on your known points. The result is a footprint per known point, ready for
downstream work that needs an extent rather than a location.

It was developed and tested on cattle feeding facilities in Kansas and New Mexico,
where the footprints feed BSI pond extraction, and the worked examples throughout
are from that application. Nothing in the method is specific to it: the
requirements are a patch-embedding dataset covering the area, point locations for
the targets, and targets that are sparse relative to the area being searched. How
far it transfers to other target types is untested — see *Assumptions and
limitations*.

```bash
python3 automated/build_footprints.py \
    --positives POINTS.geojson \
    --embeddings EMBEDDINGS.parquet \
    --boundary AOI.geojson \
    --tag myaoi
```

For large areas use the DuckDB backend instead of the parquet one — same results,
never holds the feature matrix in memory:

```bash
    --centroids CENTROIDS.parquet --duckdb EMBEDDINGS.db
```

Name the column holding your facility identifier, so every footprint can be
attributed back to the facility it belongs to. You almost always want this:

```bash
    --positive-id-field asset_identifier
```

Without it, facilities are identified by their row number in the positives file,
which works but breaks the moment that file is re-sorted. See *Outputs* for how
the attribution is written.

If your points come from several sources of differing quality, and one of them
is trustworthy, name it:

```bash
    --source-field source --trusted-source ei
```

The model is then built from the trusted points alone and the rest are admitted
only if it scores them well. This is much the strongest way to handle a mixed
register — see *Cleaning a mixed-provenance point set*.

Every run also writes `_footprints_split.geojson`: one polygon per facility,
dividing any footprint that covers several between them, which is what
downstream work needs to allocate ponds. Nothing is discarded by default. To
disown ground too far from any facility to be part of it:

```bash
    --split-max-dist-m 790
```

Choose that number by looking. Each run prints how far its own single-facility
footprints reach — `p50 255 m, p90 790 m, p98 1,776 m` — and anything beyond the
cap becomes a *detected, not attributable to a listed facility* row rather than
being given to the nearest one. The right value differs sharply by AOI: Türkiye
wanted p90 and Kansas no cap at all, on near-identical distributions. See
*Outputs*.

Add `--reference-polygons REF.geojson` if you happen to have existing polygons to
compare against. You usually won't; everything important works without them.

### Getting the embeddings

If you don't already have an embeddings parquet for your area, `fetch_embeddings.py`
builds one from the STAC catalogue — searching, downloading, quantizing,
deduplicating where MGRS tiles overlap, and assigning the `tile_id` the pipeline
keys on:

```bash
python3 automated/fetch_embeddings.py \
    --region AOI.geojson \
    --start 2025-01-01 --end 2025-12-31 \
    --name myaoi --outdir . --jobs 6 --build-duckdb
```

Ask for the calendar year you want. Yearly embeddings run 1 January to 1 January,
so that window also touches the previous vintage and the catalogue returns both;
the one your window genuinely overlaps is kept, and the period the catalogue
reports is what lands in the filename.

`--jobs` is worth setting. One HTTP stream to the asset host measured 33 MB/s,
so downloads run concurrently; the useful number is however many streams it
takes to saturate your disk. On a 600 GB `pd-balanced` volume, which takes about
170 MB/s, that is 5 or 6 — beyond that the writes just queue. The run prints an
aggregate MB/s per completed tile, so you can see where it flattens. Tiles are
around 320 MB each, and a partly finished download resumes without re-fetching
what completed.

Two options matter for large areas:

- `--positives POINTS.geojson` fetches only tiles a known facility lands on. For
  eastern Australia that is 154 tiles instead of 286, because most of the region
  has no cattle in it. Use it where targets are clustered; skip it where they're
  spread out, as in Kansas or Türkiye.
- `--clip AOI.geojson` drops patches outside the boundary during assembly, rather
  than re-clipping on every run. Give it a real AOI boundary, **not** a tight
  buffer around the positives: negative sampling holds 3 km off every positive
  and hard-negative mining works by finding look-alikes away from facilities, so
  clipping to the points themselves starves both.

`--dry-run` lists the tiles it would fetch, which is worth doing before
committing to a continent's worth of download. `--build-duckdb` runs the repo's
`scripts/build_duck_assets.py` for you; you can also run that directly on
embedding parquets you already have.

`python3 tests/test_footprints_smoke.py` runs the whole pipeline on a synthetic
area in about a minute and checks 28 invariants. Run it after changing anything.

## Layout

`build_footprints.py` is the entry point and holds only argument parsing and the
run sequence. The stages live in the `footprints` package beside it:

| | |
|---|---|
| `backends.py` | Streaming embedding access, parquet or DuckDB, behind one interface. |
| `geometry.py` | Reprojection to a metric CRS, grid-stride inference, patch squares. |
| `labels.py` | Positives from points, sampled negatives. |
| `modeling.py` | Classifier, out-of-fold probabilities, threshold selection, curves. |
| `inference.py` | Full-AOI inference, patches to merged polygons, positive matching. |
| `pipeline.py` | One train-infer-filter pass, plus the mining and admission loops. |
| `assessment.py` | The reference-free checks behind the run assessment. |
| `reporting.py` | The config and stats files. |
| `util.py` | Progress logging and the run-wide warning list. |

The dependency order runs one way, `util` → stages → `pipeline` → CLI, with no
cycles. Nothing here imports `src/`; the automated pipeline and the interactive
notebook workflow share the repo and the asset builder, not code.

## What it does

1. **Snap** each input point to its nearest embedding patch, dropping points too
   far from any patch to be real.
2. **Sample negatives** at random from the AOI, at least `--neg-min-dist-m` from
   any known positive, `--neg-ratio` per positive with a floor of
   `--min-negatives-per-million` per million patches searched.
3. **Train** a logistic regression, cross-validated.
4. **Check the labels.** If your points carry a provenance field and you name a
   trusted value — `--source-field source --trusted-source ei` — the pipeline
   trains on the trusted points alone, sharpens that model with a couple of mining
   rounds, and admits the rest only if it scores them above `--admit-above`. This
   is much the stronger option: see *Cleaning a mixed-provenance point set*.
   Otherwise, positives the model scores near zero when held out are
   labels the imagery does not support — mislocated points, addresses rather than
   facilities, mistakes. They get excluded and the model refits, and the run says
   how many went. Capped at 20% of the point set; see *Assumptions and
   limitations*.
5. **Pick a threshold** at the top edge of the F-beta curve's peak plateau.
6. **Mine hard negatives.** Detected patches that fall nowhere near a known
   positive are the area's own hardest negatives — the look-alikes random
   sampling never finds. Add a sample of them and retrain. Repeat
   `--hard-negative-rounds` times (default 4). **This is not optional tuning:**
   without it the model fires over open country. The best of the resulting rounds
   is chosen automatically — see *Choosing a round*.
7. **Run inference** across every patch in the AOI.
8. **Merge and filter.** Detected patches become polygons; a polygon is kept if
   it contains, or nearly contains, one of your known points.
9. **Attribute and report.** Statistics, a self-assessment, and the output
   layers below.

## Outputs

Each run writes into its own `<outdir>/run_<tag>_<timestamp>/` folder, so
`--outdir` is a parent directory holding many runs rather than a pile of files.

| file | what it is |
|---|---|
| `_footprints.geojson` | the deliverable — one row per footprint, with `area_ha`, `n_positives` and `positive_ids` |
| `_positives_uncovered.geojson` | input facilities that got **no** footprint, with `excluded_from_training`, `excluded_by`, `oof_probability` and `dist_to_footprint_m` |
| `_footprints_split.geojson` | one row per facility, `split_id` = `<poly_id>-<NN>`, plus `-00` rows for ground no facility could claim. `--no-split-footprints` to skip |
| `_positive_to_footprint.csv` | the facility↔footprint relation, readable in either direction |
| `_positives_excluded.geojson` | positives dropped from *training*, with `oof_probability` and `redetected_anyway` |
| `_stats.txt`, `_config.txt` | the summary and the full parameter/provenance record |
| `_detections_raw`, `_patches_filtered` | patch-level layers, joinable to footprints on `poly_id` |
| `_model.joblib`, `_fbeta_*.png`, `_pr.png` | the fitted probe and its curves |

**Attributing footprints to facilities.** The relation is many-to-many: one
polygon can cover several neighbouring facilities, and one facility can be
bracketed by disjoint polygons. Where a facility's pieces belong to it alone
they are reassembled into a single MultiPolygon row; pieces shared with a
neighbour are left separate, because absorbing them would attribute that
neighbour's ground to the wrong facility. What remains is recorded explicitly:
`positive_ids` on each footprint, and `_positive_to_footprint.csv` for the
reverse direction. On a Kansas run, about 89% of pairs are a clean one-to-one
and the table only matters for the rest.

Facility identifiers come from `--positive-id-field` (e.g. `asset_identifier`),
defaulting to the row number in the positives file. The footprints and the
uncovered layer partition your register exactly: every input facility appears
in one or the other, so nothing falls silently between them.

## Assumptions and limitations

Read these before trusting any output. Most are structural rather than fixable.

**Your point set is assumed to be reasonably complete.** Negatives are sampled at
random from the AOI on the assumption that targets are sparse *and known*, so an
unlisted facility can be drawn as a negative — and hard-negative mining, which
looks for detections away from your points, will preferentially find exactly the
facilities missing from your register and teach the model they are background.
This is the single strongest assumption in the workflow. When it broke in testing
(mining thousands of hard negatives from an incomplete register) recall fell 8
points, because the mined "negatives" were real facilities.

**Discovery is possible, but buyer beware.** By default a polygon survives only if
it contains or nearly contains one of your known points, so the standard output
finds nothing new. `--save-unfiltered-polygons` writes every merged polygon with a
`retained` column, and the unretained ones are candidate unlisted facilities.
Whether they are worth anything depends entirely on the input points: they must be
almost all co-located with real facilities, *and* capture a large fraction of the
facilities that exist in the area.

You can judge this quickly from the polygon-to-positive ratio, which the run logs:

- **Around 2:1 or below** — if your points really do cover most facilities, the
  extra polygons are a manageable candidate set. On Kansas's own polygon centroids
  (2,254 polygons for 1,125 positives, 2.0:1) the unretained set picked up many
  small cattle facilities, other farms in roughly the right genre, and some
  suburban development and other false positives — usable with review.
- **Around 3:1 or above** — too noisy to be useful. The client's Kansas register
  gave 4,059 polygons for 1,104 positives (3.7:1), and that output is not worth
  reviewing.

The ratio is a symptom, not a cause: a high ratio means either the register misses
many real facilities or its points are poorly located, and both make the model's
extra detections untrustworthy. Note also that the run statistics, assessment
thresholds and round selection are all calibrated on *filtered* output, so they do
not describe the unfiltered set.

**If discovery is the goal, use `--model mlp --hidden-layers 64,16`.** Tested in
both states, it produces a materially cleaner candidate set — **1.7:1 in both**,
against logistic regression's 2.0:1 in Kansas and 2.4:1 in New Mexico, so 15–29%
fewer polygons to review. Footprint fidelity is a wash (union IoU 0.792 vs 0.818 in
Kansas, but 0.777 vs 0.750 in New Mexico). It costs about four times the runtime.
One caution: its recall erodes in later mining rounds far more than the linear
model's — down 15% by round 4 in Kansas, 6% in New Mexico — so **never override
`--select-round auto` with an MLP.**

**How much label noise is tolerable depends on whether you have provenance.**

*With* a provenance field and a trusted source, the tolerance is high and we have
not found a ceiling. New Mexico's register was 28% non-trusted — the same 28% that
caused total failure when trained on directly — and seeded admission handled it,
producing the best result of any method we tried. The binding constraint shifts
from *how much is bad* to *whether the trusted subset is good and large enough to
train on*; New Mexico's was 427 points, and we have not tested how small it can be.
Admission itself is uncapped, so it will set aside as much as it needs to.

*Without* provenance, the fallback gate is capped at 20% of the point set, and both
real client registers we tested reached or nearly reached that cap (19.4% and
20.0%) — so on data of that quality some noise stays in training, and the run flags
it. Untreated, 28% mismatch caused total failure in one state while 16% was
absorbed without trouble in another; we cannot say where that threshold lies, or
how much of the difference was terrain rather than data.

So the practical advice is to **ask contributors for a provenance column.** It
converts an unbounded risk into a manageable one.

**Footprints are quantised to the embedding grid, and are not measurements.** The
cell is ~320 m, so the smallest possible footprint is ~10.2 ha and areas grow in
~160 m steps. A facility smaller than one cell still gets a full cell. Where
reference polygons exist they are themselves patch-grid outputs, so an "area
ratio" of 1.0 means *agrees with prior work*, not *correct*.

**The embeddings are a fixed time window.** For these AOIs, a 2023 annual mosaic.
A facility built, demolished or substantially changed outside that window is
invisible to the model no matter how good the point is — one confirmed case in
New Mexico.

**Precision figures are not field precision.** The model is validated against
sampled negatives, where the negative prior is a few per cent; across a whole
state the true prior is ~1 in 10,000. The stats file scales this into an expected
false-positive count. The positives filter absorbs most of the consequence, which
is why it exists.

**Adjacent facilities merge and cannot be separated.** If two facilities are
within a cell of each other they become one polygon covering both. The run reports
how often this happens (positives per footprint) but does nothing about it.

**Tested on US cattle facilities only, so far.** Two states, a 320 m grid, semi-arid to
arid terrain, one target class (beef and dairy trained together). Untested: other
target types, targets much smaller or larger than a cell, non-US contexts, and —
notably — the no-reference-polygon path end to end, which is the configuration
production will actually use.

**Not validated against the downstream objective.** Everything here is tuned
toward reproducing existing polygons. Whether a more generous footprint is
actually better for BSI pond extraction has not been measured, and if it is, then
some of the metrics point the wrong way.

## Cleaning a mixed-provenance point set

Registers are usually assembled from several sources, and they are not equally
good. In New Mexico the client's 591 points came from three: 427 from prior Earth Index work, 116 from earlier modelling work by another group, and 48 from a regulatory compliance database. Reviewed against imagery, the first were almost all genuine cattle facilities. Among the modelled points, we found some errors, and a proportion were former feedlots, horse operations and auction yards — plausible livestock infrastructure, but not the target class. The compliance addresses were mostly not co-located with any facility at all, often landing in towns.

**If your points carry a provenance field, use it.** Name the source you trust:

```bash
    --source-field source --trusted-source ei
```

The pipeline then trains on the trusted points alone, sharpens that model with a
couple of hard-negative rounds, and admits each remaining point only if the
sharpened model scores it above `--admit-above` (default 0.5). Repeat the flag or
comma-separate to trust several sources.

This works far better than the automatic gate, for a reason worth understanding:
**a model trained on everything cannot recognise a whole batch of bad points**,
because those points define part of what it has learned "facility" to mean. On the
New Mexico data, 108 points scored a median probability of 1.000 under a model
trained on all of them, and 0.000 under one trained only on trusted points — the
same points, the same embeddings. Seeded from the trusted subset, the pipeline
produced a better result than a point set we cleaned by hand (union IoU 0.785 vs
0.750, area ratio 1.125 vs 1.250) while keeping 32 genuine facilities that hand
cleaning threw away.

Two things to know when reading a seeded run:

- **The seed must be sharpened.** With `--seed-rounds 0` the seed is too permissive
  and admits the very look-alikes the mechanism exists to exclude — it scored worse
  than doing nothing at all. The default of 2 is the tested setting.
- **`recall(all)` will look low, and that is correct.** Points set aside by
  admission stay in its denominator. Read `recall(trained)`, which stays at
  99–100%; the set-aside points are the ones you do not want footprints for.

Without a provenance field, the fallback runs automatically: positives the model
scores near zero when held out are excluded and it refits. That catches isolated
errors well and batches of similar errors poorly, which is exactly why the
provenance route is preferable when it is available.

### A trusted subset can be small and still worth using

Eastern Australia's register is 1,230 points: 1,095 from permit records and only
**135 from prior remote-sensed detections** (`tdx`). Seeding on that ninth of the
data still produced better footprints than training on everything — 122 trusted
points survived snapping, and the sharpened seed admitted 279 of the rest.

Training on all 1,230 instead made the model fire across open country. The reason
is visible in the imagery: many permitted sites are so small that they read as
ordinary rural scene to a human eye, and evidently to the model as well. Include
them as positives and "rural" becomes part of what the model has learned a
facility to be, so it predicts rural everywhere. That is the same failure as New
Mexico's compliance addresses, arriving by a different route — there the points
were in the wrong place, here they are in the right place but below what the
embeddings can distinguish.

Headline recall on that run was 51.6%, which is the honest figure for a register
where about half the entries are not reconstructible at this resolution. Do not
chase it by relaxing the trusted set.

### What to do with the facilities that get no footprint

Whether `positives_uncovered.geojson` is worth shipping alongside the footprints
depends on **why** the points are uncovered, and that differs by register:

- **Türkiye and eastern Australia: ship it.** Visual inspection shows the large
  majority of input locations are genuinely good. Where no footprint comes back,
  it is because the facility is very small or morphologically atypical — not
  because the point is wrong. Those locations are real and worth passing on, with
  the caveat that a footprint could not be constructed for them. For the Australian
  permit sites in particular, footprints are probably not reconstructible in this
  paradigm at all, and delivering the locations is the right answer rather than a
  consolation prize.
- **The US state registers: do not ship it.** There the uncovered points were
  typically *not* well situated over any facility, so the layer is mostly bad
  locations rather than hard ones. Turkiye's 58 uncovered points sat a median of
  4.6 km from the nearest detection, which is what that looks like in numbers.

The distinction is worth making explicitly because the layer is identical in both
cases. What differs is whether "no footprint" means "we could not draw this one"
or "this point was never a facility", and only review against imagery tells you
which register you have.

## The two numbers that matter

Neither needs reference polygons, which is the point — in production you have
none.

**`recall(trained)` — the fraction of the positives the model was trained on that
got a footprint.** Not `recall(all)`. The label gate deliberately removes points
that aren't on facilities, and those stay in the `recall(all)` denominator, so
that figure reads low for a good reason. On every healthy run so far
`recall(trained)` is 99–100%. If it is materially below that, the model is
failing to find facilities it was explicitly shown.

**`ha/pos` — total footprint area divided by trained positives.** This is the
bloat detector. A model that fires too freely doesn't just add stray polygons; it
merges neighbouring detections into sprawling blobs, and because any polygon
touching a known point is kept, those blobs are *retained* and still report
finding everything. Recall cannot see this failure. Area per facility can: once
the model is right, the area needed per facility settles, and anything above that
is spill.

Why trust `ha/pos`? Because where we *do* have reference polygons to check
against, **reference IoU rises monotonically with rounds and `ha/pos` falls
monotonically alongside it** — in every series measured so far, across two states
and three point sets. That correspondence is what licenses using `ha/pos` as a
stand-in for quality where no reference exists. `det/pos` (detected patches per
trained positive) tracks the same thing one step earlier in the pipeline and is
a useful cross-check.

## Reading the run assessment

Every run prints a block at the top of `*_stats.txt`:

```
RUN ASSESSMENT: all 9 checks passed
  ok  positives unsupported by the embeddings          0.7%  (flag if > 5.0%)
  ok  peak F-beta on the labeled set                  0.924  (context only)
  ok  detected patches per known positive                 9  (flag if > 50)
  ...
```

A flagged run is not necessarily wrong, but do not use it without looking. The
checks exist because **a broken run can report 100% recall while producing
garbage** — one New Mexico run reported finding all 591 facilities inside
polygons up to 471 km², one of which swallowed 71 separate facilities.

| flag | what it means | what to do |
|---|---|---|
| positives unsupported by the embeddings | many of your points don't look like facilities in the imagery | run `review_rejected_positives.py`; expect address data |
| label-quality exclusions hit their ceiling | more points were unsupported than the 20% cap allows removing, so some contamination is still in the training set | clean the point set before trusting the run |
| detected patches per known positive | the model fires far more than your points can explain | more mining rounds, or a lower `--beta` |
| raw polygons rejected by the positives filter | almost everything detected was spurious | as above |
| retained polygons per trained positive | fewer polygons than targets — they're merging together | check the largest footprints in GIS |
| largest footprint | bigger than any plausible target | look at it; it is probably several merged together |
| share of area in the 5 largest footprints | a few blobs dominate the output | as above |
| most positives inside one footprint | one polygon is covering many targets at once | as above |

Two entries are marked **context only** and never flag: peak F-beta and F-beta
curve raggedness. Both were measured and found not to distinguish good runs from
bad — a harder area lowers F-beta legitimately, and the raggedness count is not
even monotonic. They are printed because they are informative, not because they
are diagnostic.

## Choosing a round

Each mining round trains a new model, and the config file tabulates all of them:

```
  round  trained_pos  recall(trained)  recall(all)  det/pos  median_ha  max_ha  ha/pos
      0        1,286            97.7%        94.2%       59       41.0    1160    61.4
      1        1,167           100.0%        93.2%       29       33.3     630    55.4
 *    2        1,104           100.0%        90.2%       15       25.6     640    48.3
```

**This is done for you.** `--select-round auto` is the default: it takes the round
with the lowest `ha/pos` among those whose `recall(trained)` is within half a
percentage point of the best any round achieved. The chosen round is marked `*`,
and the log says why it was chosen. Note how `det/pos` and `median_ha` usually
fall alongside `ha/pos` — when the three disagree, look closer.

**The best round is often not the last one.** On Kansas's own polygon centroids
over five rounds, `ha/pos` bottomed out at round 2 (37.3) and rose again to 41.3
by round 4 — and reference IoU tracked it exactly, peaking at 0.818 on round 2 and
falling back to 0.667. Simply taking the final round would have discarded the
entire gain. That is the case the automatic rule exists for.

To override: `--select-round last`, or `--select-round <N>` for a specific one.
**Neither avoids the work** — every round is trained and inferred regardless, and
the setting only decides which round's outputs get *selected*. Every round's
polygons are written anyway, as `*_round<N>_footprints.geojson`, so you can
compare rounds in GIS or score them with `evaluate_footprints.py` without
re-running. Pass `--no-save-round-outputs` to suppress them.

The default is 4 mining rounds (five passes), which gives the selector enough
range to find a turning point. Extra rounds cost one inference pass each and are
cheap relative to being wrong.

## When results look wrong

- **Far too many detections.** This is the first and clearest tell, visible in the
  log while inference is still running. The worst run in testing produced
  **490,000 detected patches** across one state — around 4% of the entire area —
  where a healthy run sits near 0.1–0.2%. Watch `det/pos` in the per-round table:
  9–15 on healthy runs, 60–300 on broken ones. More mining rounds are the first
  remedy; if the count stays high, suspect the point set.
- **Footprints far too large, or a few enormous ones.** Almost always label
  quality. Run `review_rejected_positives.py` with a model trained on points you
  trust; it separates *mislocated* points (a facility is nearby — it reports the
  offset to drag the point) from *no signal* (nothing there resembles a facility).
- **Nothing detected.** Check that positives fall inside `--boundary`, and that
  the config file's "Dropped, snap distance" count is zero.

**On footprints being too small:** we have never observed it. Every run measured
produced footprints *larger* than the existing polygons — the smallest area ratio
seen is 1.09, and most sit between 1.2 and 1.5. So the failure mode this workflow
exhibits is over-generous footprints, not tight ones. If you do need tighter
output, `--footprint-geometry stride` reduces each detection's cell from the full
~320 m receptive field to its ~160 m centre, which is a 4x area reduction by
construction — but it has never been evaluated against imagery, and the
reference-comparison metrics structurally cannot judge it, because the reference
polygons are themselves built from full-size cells. Treat it as untested.
`--neg-min-dist-m` is *plausibly* another lever — a smaller exclusion radius puts
more facility-adjacent ground into the negative set — but that effect has not been
measured, and should be treated as a hypothesis rather than a setting with a known
result.

## Other tools here

| | |
|---|---|
| `fetch_embeddings.py` | Build an embeddings parquet from the STAC catalogue: search, download, quantize, dedupe at MGRS overlaps, assign `tile_id`, optionally build the DuckDB pair. See *Getting the embeddings*. |
| `evaluate_footprints.py` | Compare footprints against reference polygons. Runs standalone, so it also scores per-round outputs. |
| `--save-unfiltered-polygons` | Not a tool but worth knowing: writes every merged polygon with a `retained` flag, the only route to unlisted facilities. See the ratio guidance under *Assumptions and limitations*. |
| `review_rejected_positives.py` | Triage a point set against a trained model: which points are mislocated, and by how far, versus which have no signal at all. |
| [`tests/test_footprints_smoke.py`](../tests/test_footprints_smoke.py) | One-command end-to-end check on synthetic data. Builds its DuckDB fixture with the repo's `scripts/build_duck_assets.py`. |
| [`docs/automated-planning.md`](../docs/automated-planning.md) | Design decisions, measured results from every experiment, and the reasoning behind the defaults. Read this before changing a default. |

Defaults in this pipeline are evidence-based rather than guessed, and
`docs/automated-planning.md` records the measurements behind each one — including several
plausible-sounding ideas that were built, measured and then removed.
