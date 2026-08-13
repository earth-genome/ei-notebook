"""Streaming access to patch embeddings, over parquet or DuckDB.

Two storage layouts, one interface: a single parquet holding features plus
tile_id and patch geometry, or a DuckDB table of features keyed by tile_id
alongside a centroids parquet. Both stream in batches, so neither needs the full
embedding matrix in RAM. make_backend() picks one from the parsed arguments.
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS

from .util import log, warn


class EmbeddingBackend:
    """Common interface over the two embedding storage layouts.

    Attributes:
        n_patches: Number of patches.
        n_features: Embedding dimension.
        ids: Per-patch identifier (row index or tile_id), positionally aligned
            with centroids_ll.
        centroids_ll: (N, 2) array of lon/lat patch centroids.
        source_crs: CRS of the stored geometry.

    Methods:
        fetch: Gather embedding vectors for a set of positions.
        iter_all: Stream (positions, vectors) over the whole dataset.
    """

    n_patches = 0
    n_features = 0
    ids = None
    centroids_ll = None
    source_crs = None
    kind = 'abstract'

    def fetch(self, positions):
        raise NotImplementedError

    def iter_all(self, batch_size, keep_mask=None):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError


def _col_to_numpy(col):
    """Arrow Array or ChunkedArray to numpy, across pyarrow versions."""
    if isinstance(col, pa.ChunkedArray):
        return col.to_numpy()
    return col.to_numpy(zero_copy_only=False)


def _batch_to_array(batch, n_features, columns=None):
    """Convert an Arrow batch or table of numeric columns to a uint8 array."""
    n_rows = batch.num_rows
    out = np.empty((n_rows, n_features), dtype=np.uint8)
    for j in range(n_features):
        out[:, j] = _col_to_numpy(batch.column(j if columns is None
                                               else columns[j]))
    return out


class ParquetBackend(EmbeddingBackend):
    """Single embeddings parquet holding features, tile_id and patch geometry."""

    kind = 'parquet'

    def __init__(self, path, batch_size=200_000, cache_features=False):
        self.path = path
        self.batch_size = batch_size
        self.cache_features = cache_features
        self._cache = None

        self.pf = pq.ParquetFile(path)
        names = list(self.pf.schema_arrow.names)
        self.feature_cols = [c for c in names if c not in ('tile_id', 'geometry')]
        self.n_features = len(self.feature_cols)
        self.n_patches = self.pf.metadata.num_rows

        self.source_crs = self._read_crs()
        log(f'Parquet backend: {self.n_patches:,} patches, '
            f'{self.n_features} features, {self.pf.metadata.num_row_groups} '
            f'row group(s)')
        self._read_centroids()

    def _read_crs(self):
        meta = self.pf.schema_arrow.metadata or {}
        try:
            geo = json.loads(meta[b'geo'])
            crs = geo['columns'][geo['primary_column']]['crs']
            return CRS.from_json_dict(crs) if crs else CRS.from_epsg(4326)
        except Exception:
            warn('Could not read CRS from parquet metadata; assuming EPSG:4326.')
            return CRS.from_epsg(4326)

    def _read_centroids(self):
        """Stream the geometry column, keeping only centroid coordinates.

        The 14.7 M patch polygons never all exist at once, and tile_ids are not
        retained: they would cost >1 GiB of Python strings, and row position is
        the identifier for this backend.
        """
        log('Reading patch geometry (streamed, keeping centroids only)...')
        xs = np.empty(self.n_patches, dtype='f8')
        ys = np.empty(self.n_patches, dtype='f8')
        offset = 0
        for batch in self.pf.iter_batches(batch_size=self.batch_size,
                                          columns=['geometry']):
            geoms = shapely.from_wkb(_col_to_numpy(batch.column('geometry')))
            coords = shapely.get_coordinates(shapely.centroid(geoms))
            k = len(coords)
            xs[offset:offset + k] = coords[:, 0]
            ys[offset:offset + k] = coords[:, 1]
            offset += k
            del geoms, coords
        if offset != self.n_patches:
            raise RuntimeError(
                f'Read {offset} geometries, expected {self.n_patches}.')
        self.centroids_ll = np.column_stack([xs, ys])
        self.ids = np.arange(self.n_patches)

    def _iter_feature_batches(self, batch_size=None):
        return self.pf.iter_batches(batch_size=batch_size or self.batch_size,
                                    columns=self.feature_cols)

    def _ensure_cache(self):
        if self._cache is not None:
            return
        gib = self.n_patches * self.n_features / 2**30
        log(f'Caching feature matrix in RAM ({gib:.1f} GiB)...')
        self._cache = np.empty((self.n_patches, self.n_features), dtype=np.uint8)
        offset = 0
        for batch in self._iter_feature_batches():
            arr = _batch_to_array(batch, self.n_features)
            self._cache[offset:offset + len(arr)] = arr
            offset += len(arr)

    def fetch(self, positions):
        positions = np.asarray(positions)
        if self.cache_features:
            self._ensure_cache()
            return self._cache[positions]

        order = np.argsort(positions)
        wanted = positions[order]
        out = np.empty((len(positions), self.n_features), dtype=np.uint8)
        offset, ptr = 0, 0
        for batch in self._iter_feature_batches():
            k = batch.num_rows
            end = ptr + np.searchsorted(wanted[ptr:], offset + k, side='left')
            if end > ptr:
                arr = _batch_to_array(batch, self.n_features)
                out[order[ptr:end]] = arr[wanted[ptr:end] - offset]
                ptr = end
            offset += k
            if ptr >= len(wanted):
                break
        if ptr < len(wanted):
            raise RuntimeError('Requested positions beyond end of parquet.')
        return out

    def iter_all(self, batch_size, keep_mask=None):
        if self.cache_features:
            self._ensure_cache()
            for start in range(0, self.n_patches, batch_size):
                pos = np.arange(start, min(start + batch_size, self.n_patches))
                if keep_mask is not None:
                    pos = pos[keep_mask[pos]]
                if len(pos):
                    yield pos, self._cache[pos]
            return

        offset = 0
        for batch in self._iter_feature_batches(batch_size):
            pos = np.arange(offset, offset + batch.num_rows)
            offset += batch.num_rows
            arr = _batch_to_array(batch, self.n_features)
            if keep_mask is not None:
                m = keep_mask[pos]
                pos, arr = pos[m], arr[m]
            if len(pos):
                yield pos, arr

    def describe(self):
        return {'backend': 'parquet', 'embeddings': os.path.abspath(self.path)}


class DuckDBBackend(EmbeddingBackend):
    """DuckDB table of embeddings keyed by tile_id, plus a centroids parquet.

    Built by scripts/build_duck_assets.py. Inference streams a single sequential
    scan rather than issuing a `WHERE tile_id IN (...)` per batch. DuckDB does
    push that filter down, so a per-batch query is not a full materialization of
    the table, but it still costs a pass over the id column plus a random-access
    fetch of the matching rows every time: measured 4-5x slower overall at 384
    dims, widening as the table grows.
    """

    kind = 'duckdb'

    def __init__(self, centroids_path, db_path, table='embeddings'):
        import duckdb  # imported here so the parquet path has no duckdb dep

        self.centroids_path = centroids_path
        self.db_path = db_path
        self.table = table

        log(f'Opening DuckDB {db_path} (read-only)...')
        try:
            self.con = duckdb.connect(db_path, read_only=True)
        except Exception as exc:
            raise SystemExit(
                f'Could not open {db_path} read-only: {exc}\n'
                'If the file is still being written or downloaded, wait for it '
                'to finish (a stale .wal alongside it is a sign of this).')

        cols = [r[0] for r in self.con.execute(
            f'DESCRIBE {self.table}').fetchall()]
        if 'tile_id' not in cols:
            raise SystemExit(f'Table {table} has no tile_id column.')
        self.feature_cols = [c for c in cols if c != 'tile_id']
        self.n_features = len(self.feature_cols)
        self._select = ', '.join(f'"{c}"' for c in self.feature_cols)

        n_db = self.con.execute(
            f'SELECT COUNT(*) FROM {self.table}').fetchone()[0]

        log('Reading centroids parquet...')
        tbl = pq.read_table(centroids_path, columns=['tile_id', 'geometry'])
        geoms = shapely.from_wkb(_col_to_numpy(tbl.column('geometry')))
        self.centroids_ll = shapely.get_coordinates(shapely.centroid(geoms))
        del geoms
        self.tile_ids = _col_to_numpy(tbl.column('tile_id'))
        self.ids = self.tile_ids
        self.n_patches = len(self.tile_ids)
        self.source_crs = self._read_crs(centroids_path)

        if n_db != self.n_patches:
            warn(f'DuckDB rows ({n_db:,}) != centroids ({self.n_patches:,}). '
                 'Patches missing from either side are skipped.')
        # A pandas Index rather than a dict: one C hash table, vectorized
        # lookups, and far less memory over ~15 M keys.
        self._index = pd.Index(self.tile_ids)
        log(f'DuckDB backend: {self.n_patches:,} patches, '
            f'{self.n_features} features')

    @staticmethod
    def _read_crs(path):
        meta = pq.ParquetFile(path).schema_arrow.metadata or {}
        try:
            geo = json.loads(meta[b'geo'])
            crs = geo['columns'][geo['primary_column']]['crs']
            return CRS.from_json_dict(crs) if crs else CRS.from_epsg(4326)
        except Exception:
            warn('Could not read CRS from centroids metadata; assuming '
                 'EPSG:4326.')
            return CRS.from_epsg(4326)

    def fetch(self, positions):
        positions = np.asarray(positions)
        want = self.tile_ids[positions]
        # A parameterized list rather than a registered Arrow table: DuckDB's
        # Arrow replacement scans need a newer pyarrow than is guaranteed here.
        query = self.con.execute(
            f'SELECT tile_id, {self._select} FROM {self.table} '
            'WHERE tile_id IN (SELECT UNNEST(?::VARCHAR[]))', [list(want)])
        # to_arrow_table is the current API; fetch_arrow_table is deprecated in
        # duckdb 1.5 but is the only one available in older versions.
        res = (query.to_arrow_table() if hasattr(query, 'to_arrow_table')
               else query.fetch_arrow_table())

        got = _col_to_numpy(res.column('tile_id'))
        if len(got) != len(want):
            missing = set(want) - set(got)
            raise SystemExit(
                f'{len(missing)} tile_ids present in centroids but not in the '
                f'{self.table} table, e.g. {list(missing)[:3]}.')
        arr = _batch_to_array(res, self.n_features,
                              columns=self.feature_cols)
        order = pd.Index(got).get_indexer(want)
        return arr[order]

    def iter_all(self, batch_size, keep_mask=None):
        res = self.con.execute(
            f'SELECT tile_id, {self._select} FROM {self.table}')
        # to_arrow_reader is the current API; fetch_record_batch is deprecated
        # in duckdb 1.5 but is the only one available in older versions.
        if hasattr(res, 'to_arrow_reader'):
            reader = res.to_arrow_reader(batch_size)
        else:
            reader = res.fetch_record_batch(batch_size)
        for batch in reader:
            tids = _col_to_numpy(batch.column(0))
            pos = self._index.get_indexer(tids)
            arr = _batch_to_array(batch, self.n_features,
                                  columns=list(range(1, self.n_features + 1)))
            m = pos >= 0
            if keep_mask is not None:
                m &= keep_mask[np.where(pos >= 0, pos, 0)]
            pos, arr = pos[m], arr[m]
            if len(pos):
                yield pos, arr

    def describe(self):
        return {'backend': 'duckdb',
                'duckdb': os.path.abspath(self.db_path),
                'table': self.table,
                'centroids': os.path.abspath(self.centroids_path)}


def make_backend(args):
    """Instantiate the backend implied by the CLI arguments."""
    if args.embeddings and (args.duckdb or args.centroids):
        raise SystemExit('Give either --embeddings or --centroids/--duckdb, '
                         'not both.')
    if args.embeddings:
        return ParquetBackend(args.embeddings, batch_size=args.batch_size,
                              cache_features=args.cache_features)
    if args.duckdb and args.centroids:
        if args.cache_features:
            warn('--cache-features applies to the parquet backend only; '
                 'ignored.')
        return DuckDBBackend(args.centroids, args.duckdb, table=args.table)
    raise SystemExit('Specify --embeddings PATH.parquet, or both --centroids '
                     'PATH.parquet and --duckdb PATH.db.')
