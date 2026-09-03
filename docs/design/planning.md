## Design note: Range index + id column

**Timestamp:** 2026.03.05

---

### Context

In some places we use **positional indices** (e.g. sindex, Annoy). In others we use a **label index** (id_column, e.g. tile_id) to retrieve vectors from DuckDB. After moving id_column to be the index of the centroid GeoDataFrame (for a cleaner embedding_store contract), GeoLabeler and other positional use broke because `sindex.nearest` returns iloc positions but the code used `.loc` with them.

This note outlines reverting to a **range index everywhere** and keeping **id_column as a column** so that:
- Positional use (sindex, Annoy, GeoLabeler) consistently uses integer positions 0..n-1 with `.iloc`.
- Id-based use (DuckDB, get_vectors) uses the id column; the vector store API stays id-based.

---

### 1. embedding_store.py

**Centroid GeoDataFrame (embeddings.gdf)**

- **Index:** Always `np.arange(len(gdf))` (range index). Do not use id values as the index.
- **Id values:** Keep them in a column, e.g. `centroid_gdf[id_column] = ...` (or `centroid_gdf["tile_id"]` when `id_column` is None, using ordinal 0..n-1).
- Net: range index + explicit id column; the index is never "the id".

**from_dataframe / from_parquet**

- Build centroid geometry as today.
- Set `centroid_gdf.index = np.arange(len(gdf))` (and e.g. `index.name = None`).
- If `id_column` was provided and present in `gdf`: add `centroid_gdf[id_column] = gdf[id_column].values`.
- If `id_column` is None: add `centroid_gdf["tile_id"] = np.arange(len(gdf))` (or equivalent) so there is still a single id column name for downstream.
- For **InMemoryVectorStore**, the internal DataFrame must still be keyed by id for `get_vectors(ids)` to work: e.g. `vectors_df.index = centroid_gdf[id_column].values` (so the store continues to take ids, not positions).

**from_duckdb**

- Caller passes a centroid GeoDataFrame that may have `id_column` as **index** or as **column**.
- Normalize to "range index + id column":
  - If `id_column` is the index: add `centroid_gdf[id_column] = centroid_gdf.index`, then `centroid_gdf.index = np.arange(len(centroid_gdf))`.
  - If `id_column` is already a column: set `centroid_gdf.index = np.arange(len(centroid_gdf))`.
- EmbeddingMapper must know the id column name; it can't rely on `gdf.index.name` anymore, so it needs an explicit `id_column` argument (or we infer it and store it).

**EmbeddingMapper**

- **id_column:** Can't be `self.gdf.index.name` anymore. Store the id column name explicitly (e.g. `__init__(self, centroid_gdf, vector_store, id_column="tile_id")`) and have the property return that.
- **map_points:** Keep returning **ids** (values from the id column), not positions, so existing "assign to id column + call get_vectors(ids)" usage still works. Implementation: get nearest **position** from sindex, then `return self.gdf[self.id_column].iloc[nearest_positions]` (same shape as today, but coming from the column).
- **get_vectors(ids):** Unchanged: it receives ids and passes them to the store. No change to DuckDB/InMemory store APIs.

**DuckDBVectorStore / InMemoryVectorStore**

- No change: they still take ids and return embedding vectors. InMemory's internal DataFrame is still indexed by id (as set from `centroid_gdf[id_column]` above).

**get_vectors() return value: no id, no index**

- The contract that **ids never leak into embedding data** must remain. When building or returning vectors in `get_vectors()`:
  - **InMemoryVectorStore:** Build the internal DataFrame from **embedding columns only** (exclude `id_column` and geometry when building `vectors_df`). Use `centroid_gdf[id_column].values` only as the **index** of that DataFrame for lookup; do not add `id_column` as a column. On return, use `reset_index(drop=True)` so the returned DataFrame has a default range index and **no id column**—only embedding dimensions.
  - **DuckDBVectorStore:** After `fetchdf()`, **explicitly drop the id column** (e.g. `df.drop(columns=[self._id_column], errors="ignore")`) and then reindex by the requested ids; return with `reset_index(drop=True)` so the result has **no id column and no index**—only embedding dimensions.
- So when moving to range index + id column: ensure `from_dataframe` still builds `vectors_df` from `embedding_cols` that exclude `id_column` (and geometry / tile_id when applicable). The returned DataFrame from `get_vectors()` must never contain the id column or expose the lookup index; it must be embedding-only.

---

### 2. ui.py (GeoLabeler refactor)

**EmbeddingMapper relocation**

- `EmbeddingMapper` now lives in `embedding_store.py` (see §1). `ui.py` imports it only indirectly through notebooks; the labeler receives a centroid `gdf`, not a DuckDB connection or Annoy index.

**GeoLabeler constructor**

- Signature: `GeoLabeler(gdf, geojson_path, custom_baselayer_url=None, custom_attribution=None, save_dir=None)`.
- `gdf`: centroid GeoDataFrame (typically `embeddings.gdf` from an `EmbeddingMapper`).
- `geojson_path`: AOI boundary drawn on the map; initial viewport is fit to this boundary.
- `custom_baselayer_url`: optional tile URL added as a `CUSTOM` basemap entry in the toggle queue.
- `save_dir`: directory for GeoJSON exports from `save_dataset()` (default: current working directory).
- Default basemap is **GOOGLE_HYBRID** (Maptiler, Google Hybrid, and Mapbox are always available via toggle).

**pos_indices / neg_indices and iloc**

- Store **positions** (results of `sindex.nearest`), not id values.
- Layer updates and save use **`.iloc`** (e.g. `self.gdf.iloc[self.pos_indices][["geometry"]]`).
- `save_dataset(b=None)` is callable from the notebook without a button click.

**add_ee_basemaps (opt-in)**

- EE basemaps are **not** created in `__init__`. Call `labeler.add_ee_basemaps(geojson_path, start_date, end_date)` to append `HSV_MEDIAN` and `RGB_MEDIAN` to the basemap toggle queue. This triggers Earth Engine initialization and authentication. The current basemap is left unchanged until the user toggles.

---

### 3. scripts/build_duck_assets.py

Builds out-of-memory DuckDB assets from one or more embedding parquet files.

**CLI**

```
python scripts/build_duck_assets.py PARQUET [PARQUET ...] \
  [--clip_path GEOJSON] \
  [--db_path embeddings.db] \
  [--table_name embeddings] \
  [--centroids_path centroids.parquet] \
  [--id_column tile_id] \
  [--geometry_col geometry]
```

**Requirements**

- Every parquet must contain `id_column` (default `tile_id`) and `geometry_col` (default `geometry`). The id column is **required** for deduplication across overlapping regions.
- Optional `--clip_path` clips each parquet to a boundary before insert.

**Outputs**

- `embeddings.db` with a table of id + embedding columns (geometry stripped).
- `centroids.parquet` with id column and centroid geometry (written as a GeoDataFrame).

These outputs feed `embedding_store.from_duckdb()` in `duckdb_ei.ipynb` or `ei_alt_workflow.ipynb`.

---

### 4. src/alt_workflow_ml_utils.py

ML and validation helpers used by `ei_alt_workflow.ipynb`. All functions expect an `EmbeddingMapper` with `.gdf`, `.get_vectors(ids)`, and `.id_column`.

| Function | Purpose |
|----------|---------|
| `predict(X, model, threshold=0.5)` | Binary inference from feature matrix |
| `predict_df(df, embeddings, model, threshold=0.5)` | Inference on labeled points; resolves ids via `embeddings.id_column` |
| `score(y_pred, y_true)` | Prints accuracy, precision, recall, confusion matrix |
| `f1_curve`, `prec_rec_curve`, `roc_curve` | Threshold / validation plots |
| `get_detections(embeddings, model, threshold, boundary_path=None, batch_size=10000)` | Run model over all centroids; returns GeoDataFrame with id column, geometry, and `probability` |
| `detections_to_rectpolys(embeddings, detections, patch_width=320, ...)` | Merge point detections into polygons with mean confidence |

**get_detections batching**

- Iterates over `embeddings.gdf` by **range-index positions** in batches of `batch_size`.
- Converts each batch to ids via `gdf[id_column].iloc[batch_positions]`, then calls `get_vectors(ids)`.
- Returns an empty GeoDataFrame (with `probability` column) when no tiles exceed the threshold.

**detections_to_rectpolys buffering**

- Builds axis-aligned squares in **EPSG:4326** by converting `patch_width` meters to degrees at the centroid of `embeddings.gdf` bounds (not UTM).
- Merges boxes with a small degree-based buffer, then assigns polygon confidence as the mean of overlapping detection probabilities.

Default prediction threshold is **0.5** throughout.

---

### 5. Annoy and notebooks

- Annoy item ids are 0..n-1 (row order) → **positions**.
- When you get nearest items from Annoy, you get positions. To call `get_vectors(...)` you need **ids**: `ids = embeddings.gdf[embeddings.id_column].iloc[positions]`, then `get_vectors(ids)`.
- So any place that builds an Annoy index from `embeddings.gdf` (or equivalent) and then uses the returned indices to fetch vectors should convert **position → id** via the id column before calling `get_vectors`. That's the only change where Annoy is used.

---

### 6. ml_utils and other callers

- **predict_df / get_detections / detections_to_rectpolys:** Implemented in `alt_workflow_ml_utils.py` (see §4). They use `embeddings.id_column` to find the id column in DataFrames; the mapper's gdf holds ids in a column, not the index.
- Any code that does **index-based** access on `embeddings.gdf` (e.g. `gdf.loc[some_id]`) should use id-column semantics: `gdf[gdf[id_column] == some_id]`.

---

### 7. Summary table

| Component              | Current (id as index)     | After (range index + id column)                    |
|------------------------|---------------------------|----------------------------------------------------|
| centroid_gdf.index     | id values (tile_id etc.)   | `np.arange(n)`                                     |
| centroid_gdf[id_column]| N/A (id is index)         | Column of id values                                |
| map_points() return   | index labels (= ids)      | Same: ids from `gdf[id_column].iloc[positions]`    |
| get_vectors(ids)       | ids                       | Unchanged: still ids                               |
| Vector stores          | Keyed by id               | Unchanged                                          |
| GeoLabeler             | Must use iloc (fix)       | Same iloc fix; gdf has range index                |
| Annoy → get_vectors    | N/A or by id              | Convert positions to ids via id column, then call  |
| EmbeddingMapper.id_column | `gdf.index.name`      | Explicit stored name (e.g. constructor arg)        |

---

### 8. Why this resolves the clash

- **Positional use (sindex, Annoy, GeoLabeler):** All use integer positions 0..n-1; `gdf.iloc[positions]` is always correct.
- **Id-based use (DuckDB, get_vectors, saving by tile_id):** All use the **column** `id_column`; no dependence on the index.
- **API:** `map_points()` and `get_vectors(ids)` stay the same from the caller's perspective; only the internal representation of the centroid gdf (index vs column) and the iloc/loc usage in the labeler (and position→id for Annoy) change.
