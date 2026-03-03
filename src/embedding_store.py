"""Unified embedding vector store and centroid-based mapper.

Two backends: DuckDB (id -> vectors via SQL) and in-memory (id -> vectors via
DataFrame). EmbeddingMapper holds a centroid GeoDataFrame and a VectorStore;
map_points() snaps geometry to nearest centroid id, get_vectors(ids) delegates
to the store.

Helpers: from_parquet() splits a combined parquet into centroid gdf +
InMemoryVectorStore; from_duckdb() builds an EmbeddingMapper from a centroid
gdf and DuckDB connection.
"""

from __future__ import annotations

from typing import Protocol, Union

import geopandas as gpd
import numpy as np
import pandas as pd


class VectorStore(Protocol):
    """Protocol for id -> embedding vectors. Rows returned in same order as ids."""

    def get_vectors(self, ids: Union[pd.Series, pd.Index, list, np.ndarray]) -> pd.DataFrame:
        """Return embedding vectors for the given ids. DataFrame has no id column."""
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
            return self._con.execute(
                f"SELECT * FROM {self._table_name} LIMIT 0"
            ).fetchdf()
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
        df = df.set_index(self._id_column)
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
    path: str,
    geometry_col: str = "geometry",
    id_col: str | None = None,
    embedding_cols: list[str] | None = None,
    return_mapper: bool = True,
) -> Union[tuple[gpd.GeoDataFrame, InMemoryVectorStore], EmbeddingMapper]:
    """Split a parquet (centroids + embeddings) into centroid gdf and vector store.

    If id_col is None, uses integer index 0..n-1. embedding_cols defaults to
    all columns except geometry and id_col. If return_mapper is True, returns
    an EmbeddingMapper; else returns (centroid_gdf, InMemoryVectorStore).
    """
    gdf = gpd.read_parquet(path)
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not in parquet columns")
    if embedding_cols is None:
        exclude = {geometry_col} | ({id_col} if id_col else set())
        embedding_cols = [c for c in gdf.columns if c not in exclude]
    centroid_gdf = gdf[[geometry_col]].copy()
    if id_col and id_col in gdf.columns:
        centroid_gdf.index = gdf[id_col].values
    else:
        centroid_gdf.index = np.arange(len(gdf))
    vectors_df = gdf[embedding_cols].copy()
    vectors_df.index = centroid_gdf.index
    store = InMemoryVectorStore(vectors_df)
    if return_mapper:
        return EmbeddingMapper(centroid_gdf, store)
    return centroid_gdf, store


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
    id_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> EmbeddingMapper:
    """Build an EmbeddingMapper from a single GeoDataFrame (centroids + embeddings in memory).

    Splits into centroid gdf and InMemoryVectorStore. Same semantics as
    from_parquet; use when you already have the DataFrame loaded.
    """
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not in DataFrame")
    if embedding_cols is None:
        exclude = {geometry_col} | ({id_col} if id_col else set())
        embedding_cols = [c for c in gdf.columns if c not in exclude]
    centroid_gdf = gdf[[geometry_col]].copy()
    if id_col and id_col in gdf.columns:
        centroid_gdf.index = gdf[id_col].values
    else:
        centroid_gdf.index = np.arange(len(gdf))
    vectors_df = gdf[embedding_cols].copy()
    vectors_df.index = centroid_gdf.index
    return EmbeddingMapper(centroid_gdf, InMemoryVectorStore(vectors_df))
