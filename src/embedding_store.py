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

from annoy import AnnoyIndex

import geopandas as gpd
import numpy as np
import pandas as pd


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
        out = self._df.reindex(ids)
        return out.reset_index(drop=True)


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
        ids_unique = ids_full.unique()
        if len(ids_unique) == 0:
            df = self._con.execute(
                f"SELECT * FROM {self._table_name} LIMIT 0"
            ).fetchdf()
            return df.drop(columns=[self._id_column], errors="ignore")
        def _quote(tid):
            if isinstance(tid, str):
                return f"'{tid}'"
            return str(int(tid))
        placeholders = ", ".join(_quote(tid) for tid in ids_unique)
        query = f"""
            SELECT * FROM {self._table_name}
            WHERE {self._id_column} IN ({placeholders})
        """
        df = self._con.execute(query).fetchdf()
        ids_from_db = df[self._id_column].values
        df = df.drop(columns=[self._id_column], errors="ignore")
        df.index = ids_from_db
        df = df.reindex(ids_full)
        return df.reset_index(drop=True)


class EmbeddingMapper:
    """Map points to nearest centroid id; get vectors by id from a VectorStore.

    centroid_gdf: GeoDataFrame with geometry; index is the id space (integer
    or tile_id). vector_store: DuckDB or in-memory backend implementing
    get_vectors(ids).
    """

    def __init__(self, centroid_gdf: gpd.GeoDataFrame, vector_store: VectorStore):
        self.gdf = centroid_gdf
        self.sindex = self.gdf.sindex
        self._store = vector_store

    @property
    def id_column(self) -> str | None:
        """Name of the id/tile column (same as gdf.index.name). Use for predict_df, get_detections output, detections_to_rectpolys."""
        return self.gdf.index.name

    def map_points(self, df: gpd.GeoDataFrame) -> pd.Series:
        """Map geometry to nearest centroid. Returns Series of ids, index = df.index."""
        nearest_idxs = self.sindex.nearest(df.geometry, return_all=False)[1]
        ids = self.gdf.iloc[nearest_idxs].index
        return pd.Series(ids, index=df.index)

    def get_vectors(
        self, ids: Union[pd.Series, pd.Index, list, np.ndarray]
    ) -> pd.DataFrame:
        """Return embedding vectors for the given ids (order preserved)."""
        return self._store.get_vectors(ids)


def from_parquet(
    path: Union[str, Path],
    geometry_col: str = "geometry",
    id_column: str | None = None,
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

    centroid_gdf must have id_column as a column or as index. If it is a
    column, it is set as index so gdf.index is the id space (consistent with
    map_points and get_detections).
    """
    centroid_gdf = centroid_gdf.copy()
    if id_column in centroid_gdf.columns:
        centroid_gdf = centroid_gdf.set_index(id_column)
    elif centroid_gdf.index.name != id_column:
        raise ValueError(
            f"centroid_gdf must have '{id_column}' as column or index"
        )
    store = DuckDBVectorStore(connection, table_name, id_column=id_column)
    return EmbeddingMapper(centroid_gdf, store)


def from_dataframe(
    gdf: gpd.GeoDataFrame,
    geometry_col: str = "geometry",
    id_column: str | None = None,
    embedding_cols: list[str] | None = None,
    return_mapper: bool = True,
) -> Union[tuple[gpd.GeoDataFrame, InMemoryVectorStore], EmbeddingMapper]:
    """Build an EmbeddingMapper (or centroid gdf + store) from a GeoDataFrame.

    Geometry is converted to centroids (points); if already points, unchanged.
    If id_column is None, uses
    integer index 0..n-1 with index.name = 'tile_id'. If return_mapper is False,
    returns (centroid_gdf, InMemoryVectorStore); otherwise returns EmbeddingMapper.
    """
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not in DataFrame")
    if embedding_cols is None:
        exclude = {geometry_col} | ({id_column} if id_column else {"tile_id"})
        embedding_cols = [c for c in gdf.columns if c not in exclude]
    centroid_gdf = gdf[[geometry_col]].copy()
    centroid_gdf[geometry_col] = centroid_gdf[geometry_col].centroid
    if id_column and id_column in gdf.columns:
        centroid_gdf.index = gdf[id_column].values
        centroid_gdf.index.name = id_column
    else:
        centroid_gdf.index = np.arange(len(gdf))
        centroid_gdf.index.name = "tile_id"  # ordinal ids
    vectors_df = gdf[embedding_cols].copy()
    vectors_df.index = centroid_gdf.index
    store = InMemoryVectorStore(vectors_df)
    if return_mapper:
        return EmbeddingMapper(centroid_gdf, store)
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
