"""Unified embedding vector store and centroid-based mapper.

Two backends: DuckDB (id -> vectors via SQL) and in-memory (id -> vectors via
DataFrame). EmbeddingMapper holds a centroid GeoDataFrame and a VectorStore;
map_points() snaps geometry to nearest centroid id, get_vectors(ids) delegates
to the store.

Helpers: from_parquet() splits a combined parquet into centroid gdf +
InMemoryVectorStore; from_duckdb() builds an EmbeddingMapper from a centroid
gdf and DuckDB connection. get_annoy_index() loads an Annoy index from disk or
builds it from vectors and saves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union
import warnings

from annoy import AnnoyIndex

import geopandas as gpd
import numpy as np
import pandas as pd


def _require_ids_present(requested, found, source: str) -> None:
    """Raise if any requested id is absent from the store.

    Without this, reindexing to the requested ids turns a missing id into a row
    of NaN, which surfaces later as an opaque failure inside the model -- or not
    at all, if something imputes on the way.
    """
    requested = pd.Index(requested)
    missing_mask = ~requested.isin(pd.Index(found))
    if not missing_mask.any():
        return
    missing = requested[missing_mask].unique()
    examples = ", ".join(repr(m) for m in missing[:5])
    if len(missing) > 5:
        examples += ", ..."
    raise KeyError(
        f"{len(missing)} requested id(s) are not in {source}: {examples}. "
        "The centroids and the embedding store are out of sync."
    )


class VectorStore(Protocol):
    """Protocol for id -> embedding vectors. Rows returned in same order as ids."""

    def get_vectors(self, ids: Union[pd.Series, pd.Index, list, np.ndarray]) -> pd.DataFrame:
        """Return embedding vectors for the given ids.

        Returns a DataFrame of numeric embedding columns only (no id column).
        Row order matches the order of ids.
        """
        ...


class InMemoryVectorStore:
    """Vector store backed by a DataFrame indexed by id (embedding columns only)."""

    def __init__(self, df: pd.DataFrame):
        """df: DataFrame with index = id and columns = embedding dimensions."""
        self._df = df

    def get_vectors(
        self, ids: Union[pd.Series, pd.Index, list, np.ndarray]
    ) -> pd.DataFrame:
        ids = pd.Index(ids) if not isinstance(ids, (pd.Series, pd.Index)) else ids
        _require_ids_present(ids, self._df.index, "the embedding table")
        out = self._df.reindex(ids)
        return out.reset_index(drop=True)

    def iter_all(self, batch_size: int = 200_000):
        """Yield (ids, vectors) over the whole store, in storage order.

        For scanning every vector once -- see DuckDBVectorStore.iter_all for why
        this exists rather than looping over get_vectors().
        """
        for start in range(0, len(self._df), batch_size):
            chunk = self._df.iloc[start:start + batch_size]
            yield chunk.index.to_numpy(), chunk.reset_index(drop=True)


class DuckDBVectorStore:
    """Vector store backed by a DuckDB table with an id column and embedding columns."""

    def __init__(
        self,
        connection,  # duckdb.DuckDBPyConnection
        table_name: str,
        id_column: str = "tile_id",
    ):
        self._con = connection
        self._table_name = table_name
        self._id_column = id_column

    def get_vectors(
        self, ids: Union[pd.Series, pd.Index, list, np.ndarray]
    ) -> pd.DataFrame:
        ids_full = pd.Series(ids) if not isinstance(ids, pd.Series) else ids
        if len(ids_full) == 0:
            df = self._con.execute(
                f"SELECT * FROM {self._table_name} LIMIT 0"
            ).fetchdf()
            return df.drop(columns=[self._id_column], errors="ignore")
        def _quote(tid):
            if isinstance(tid, str):
                # Double any single quote, else an id containing one ends the
                # string literal and the query fails to parse.
                escaped = tid.replace("'", "''")
                return f"'{escaped}'"
            return str(int(tid))
        placeholders = ", ".join(_quote(tid) for tid in ids_full)
        query = f"""
            SELECT * FROM {self._table_name}
            WHERE {self._id_column} IN ({placeholders})
        """
        df = self._con.execute(query).fetchdf()
        ids_from_db = df[self._id_column].values
        _require_ids_present(
            ids_full, ids_from_db, f"DuckDB table '{self._table_name}'"
        )
        df = df.drop(columns=[self._id_column], errors="ignore")
        df.index = ids_from_db
        df = df.reindex(ids_full)
        return df.reset_index(drop=True)

    def _embedding_columns(self) -> list[str]:
        """Table columns except the id column, in table order.

        Matches the column order that get_vectors returns via SELECT *, so a
        model fitted on one can be applied to the other.
        """
        cols = [r[0] for r in
                self._con.execute(f"DESCRIBE {self._table_name}").fetchall()]
        return [c for c in cols if c != self._id_column]

    def iter_all(self, batch_size: int = 200_000):
        """Yield (ids, vectors) over the whole table in one sequential scan.

        For covering the whole table, this is the cheap path. Looping over
        get_vectors() instead pays, per batch, a pass over the id column plus a
        random-access fetch of the matching rows; measured on a persisted file at
        384 dims and batch_size=10,000 that is 4-5x slower overall, and the
        per-batch cost itself creeps up with table size (85ms at 100k rows,
        104ms at 800k), so the gap widens as the AOI grows.

        Vectors come back as a DataFrame with the same columns, in the same
        order, as get_vectors() returns. All non-id columns must be numeric.
        """
        cols = self._embedding_columns()
        select = ", ".join(f'"{c}"' for c in cols)
        result = self._con.execute(
            f'SELECT "{self._id_column}", {select} FROM {self._table_name}'
        )
        # to_arrow_reader is current; fetch_record_batch is deprecated in duckdb
        # 1.5 but is the only one present in older versions.
        reader = (result.to_arrow_reader(batch_size)
                  if hasattr(result, "to_arrow_reader")
                  else result.fetch_record_batch(batch_size))
        for batch in reader:
            # Column-wise to_numpy is zero-copy for null-free numeric Arrow
            # columns; to_pydict/to_pandas would be an order of magnitude slower
            # at 384 columns.
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            arr = np.stack(
                [batch.column(i + 1).to_numpy(zero_copy_only=False)
                 for i in range(len(cols))],
                axis=1,
            )
            yield ids, pd.DataFrame(arr, columns=cols, copy=False)


class EmbeddingMapper:
    """Map points to nearest centroid id; get vectors by id from a VectorStore.

    centroid_gdf: GeoDataFrame with geometry; has range index and id_column as
    a column. vector_store: DuckDB or in-memory backend implementing
    get_vectors(ids).
    """

    def __init__(
        self,
        centroid_gdf: gpd.GeoDataFrame,
        vector_store: VectorStore,
        id_column: str = "tile_id",
    ):
        self.gdf = centroid_gdf
        self.sindex = self.gdf.sindex
        self._store = vector_store
        self._id_column = id_column

    @property
    def id_column(self) -> str | None:
        """Name of the id/tile column. Use for predict_df, get_detections output, detections_to_rectpolys."""
        return self._id_column

    def map_points(self, df: gpd.GeoDataFrame) -> pd.Series:
        """Map geometry to nearest centroid. Returns Series of ids, index = df.index."""
        nearest_idxs = self.sindex.nearest(df.geometry, return_all=False)[1]
        ids = self.gdf[self._id_column].iloc[nearest_idxs]
        return pd.Series(ids.values, index=df.index)

    def get_vectors(
        self, ids: Union[pd.Series, pd.Index, list, np.ndarray]
    ) -> pd.DataFrame:
        """Return embedding vectors for the given ids (order preserved)."""
        return self._store.get_vectors(ids)

    def iter_all(self, batch_size: int = 200_000):
        """Yield (ids, vectors) over the whole store in one pass, if supported.

        Raises AttributeError for stores that do not implement it, so callers can
        fall back to get_vectors().
        """
        return self._store.iter_all(batch_size)


def from_parquet(
    path: Union[str, Path],
    geometry_col: str = "geometry",
    id_column: str = "tile_id",
    embedding_cols: list[str] | None = None,
    return_mapper: bool = True,
) -> Union[tuple[gpd.GeoDataFrame, InMemoryVectorStore], EmbeddingMapper]:
    """Load a parquet (centroids + embeddings) and return mapper or (gdf, store).

    Delegates to from_dataframe after gpd.read_parquet(path). If return_mapper
    is True, returns an EmbeddingMapper; else returns (centroid_gdf, InMemoryVectorStore).
    """
    gdf = gpd.read_parquet(path)
    return from_dataframe(
        gdf,
        geometry_col=geometry_col,
        id_column=id_column,
        embedding_cols=embedding_cols,
        return_mapper=return_mapper,
    )


def from_duckdb(
    centroid_gdf: gpd.GeoDataFrame,
    connection,  # duckdb.DuckDBPyConnection
    table_name: str,
    id_column: str = "tile_id",
) -> EmbeddingMapper:
    """Build an EmbeddingMapper from a centroid GeoDataFrame and DuckDB table.

    centroid_gdf must have id_column as a column or as index. It is normalized
    to range index + id_column as column (consistent with map_points and get_detections).
    """
    centroid_gdf = centroid_gdf.copy()
    if id_column in centroid_gdf.columns:
        centroid_gdf.index = np.arange(len(centroid_gdf))
    elif centroid_gdf.index.name == id_column:
        centroid_gdf[id_column] = centroid_gdf.index.values
        centroid_gdf.index = np.arange(len(centroid_gdf))
    else:
        raise ValueError(
            f"centroid_gdf must have '{id_column}' as column or index"
        )
    store = DuckDBVectorStore(connection, table_name, id_column=id_column)
    return EmbeddingMapper(centroid_gdf, store, id_column=id_column)


def from_dataframe(
    gdf: gpd.GeoDataFrame,
    geometry_col: str = "geometry",
    id_column: str = "tile_id",
    embedding_cols: list[str] | None = None,
    return_mapper: bool = True,
) -> Union[tuple[gpd.GeoDataFrame, InMemoryVectorStore], EmbeddingMapper]:
    """Build an EmbeddingMapper (or centroid gdf + store) from a GeoDataFrame.

    Geometry is converted to centroids (points); if already points, unchanged.
    Centroid gdf has range index and an id column: id_column (default
    'tile_id'). If that column exists in gdf, it is used; otherwise an ordinal
    id (0..n-1) is created with that name and a warning is issued. If
    return_mapper is False, returns (centroid_gdf, InMemoryVectorStore);
    otherwise returns EmbeddingMapper.
    """
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not in DataFrame")
    if embedding_cols is None:
        exclude = {geometry_col, id_column}
        embedding_cols = [c for c in gdf.columns if c not in exclude]
    centroid_gdf = gdf[[geometry_col]].copy()
    centroid_gdf[geometry_col] = centroid_gdf[geometry_col].centroid
    centroid_gdf.index = np.arange(len(gdf))
    if id_column in gdf.columns:
        centroid_gdf[id_column] = gdf[id_column].values
    else:
        warnings.warn(
            f"Column '{id_column}' not in DataFrame; creating ordinal id (0..n-1) with that name.",
            UserWarning,
            stacklevel=2,
        )
        centroid_gdf[id_column] = np.arange(len(gdf))
    vectors_df = gdf[embedding_cols].copy()
    vectors_df.index = centroid_gdf[id_column].values
    store = InMemoryVectorStore(vectors_df)
    if return_mapper:
        return EmbeddingMapper(centroid_gdf, store, id_column=id_column)
    return centroid_gdf, store


def get_annoy_index(
    path: Union[str, Path],
    dim: int | None = None,
    *,
    vectors: Union[np.ndarray, pd.DataFrame, None] = None,
    n_trees: int = 10,
    metric: str = "angular",
):
    """Load Annoy index from path if it exists; otherwise build from vectors and save.

    path: Path to .ann file.
    dim: Embedding dimension. Required when loading from path; when building from
        vectors, inferred from vectors if omitted.
    vectors: If provided and path does not exist, build index from these (shape (n, dim)).
        Caller must pass embedding-only data (e.g. from get_vectors); no id column.
    n_trees: Number of trees when building (default 10).
    metric: 'angular' or 'euclidean' (default 'angular').

    Returns:
        AnnoyIndex instance (loaded or newly built). Item ids in the built index
        are 0..n-1 corresponding to the rows of vectors.
    """
    path = Path(path)
    if path.exists():
        if dim is None:
            raise ValueError("dim is required when loading an existing Annoy index from path")
        idx = AnnoyIndex(dim, metric)
        idx.load(str(path))
        return idx
    if vectors is not None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be 2-dimensional, got shape {arr.shape}")
        inferred_dim = arr.shape[1]
        if dim is None:
            dim = inferred_dim
        elif dim != inferred_dim:
            raise ValueError(
                f"vectors must have shape (n, {dim}), got {arr.shape}"
            )
        idx = AnnoyIndex(dim, metric)
        for i, row in enumerate(arr):
            idx.add_item(i, row)
        idx.build(n_trees)
        path.parent.mkdir(parents=True, exist_ok=True)
        idx.save(str(path))
        return idx
    raise FileNotFoundError(f"Annoy index not found at {path} and no vectors provided to build.")
