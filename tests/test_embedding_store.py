#!/usr/bin/env python3
"""Tests for embedding_store and the get_detections inference path.

    python3 tests/test_embedding_store.py

Covers the three fixes made to that path: iter_all() streaming, the raise on
ids missing from the store, and quoting of ids containing a single quote. The
central assertion is that streaming and the per-batch fallback produce identical
detections, on both backends, so the fast path cannot silently drift from the
one it replaced.

Runs on a synthetic in-memory fixture; no data files or network needed.
"""

from __future__ import annotations

import os
import sys
import types

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'src'))

try:
    import annoy  # noqa: F401
except ImportError:
    # embedding_store imports annoy at module level, but only get_annoy_index
    # uses it and nothing here touches that. Stub it so the test runs in an
    # environment without it rather than failing for an unrelated reason.
    annoy = types.ModuleType('annoy')
    annoy.AnnoyIndex = object
    sys.modules['annoy'] = annoy
    print('note: annoy not installed; stubbed (unused by these tests)')

import alt_workflow_ml_utils as ml  # noqa: E402
import embedding_store as es  # noqa: E402

N, D = 5000, 16
N_POSITIVE = 200
COLS = [f'e{j}' for j in range(D)]

PASS, FAIL = [], []


def check(label, ok, note=''):
    (PASS if ok else FAIL).append(label)
    print(f'  {"PASS" if ok else "FAIL"}  {label}'
          f'{"  [" + note + "]" if note else ""}')


class NoStream:
    """A mapper with iter_all hidden, to exercise the per-batch fallback."""

    def __init__(self, mapper):
        self._mapper = mapper

    def __getattr__(self, name):
        if name == 'iter_all':
            raise AttributeError(name)
        return getattr(self._mapper, name)


def build_fixture():
    """Separable positive cluster, one id carrying a single quote."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(N, D)).astype(np.float32)
    features[:N_POSITIVE] += 3.0
    ids = [f'tile_{i}' for i in range(N)]
    ids[7] = "tile_o'brien_7"

    frame = pd.DataFrame({'tile_id': ids,
                          **{c: features[:, j] for j, c in enumerate(COLS)}})
    geometry = [Point(x, y) for x, y in
                zip(-100 + rng.normal(size=N) * 0.5,
                    38 + rng.normal(size=N) * 0.5)]
    centroids = gpd.GeoDataFrame({'tile_id': ids}, geometry=geometry,
                                 crs='EPSG:4326')

    memory = es.from_dataframe(
        gpd.GeoDataFrame(frame, geometry=geometry, crs='EPSG:4326'))
    con = duckdb.connect()
    con.execute('CREATE TABLE emb AS SELECT * FROM frame')
    duck = es.from_duckdb(centroids.copy(), con, 'emb')

    labels = np.zeros(N, int)
    labels[:N_POSITIVE] = 1
    model = LogisticRegression(max_iter=1000).fit(
        pd.DataFrame(features, columns=COLS), labels)
    return features, ids, memory, duck, con, model


def main():
    features, ids, memory, duck, con, model = build_fixture()
    backends = (('memory', memory), ('duckdb', duck))

    check('quoted id round-trips',
          np.allclose(duck.get_vectors(["tile_o'brien_7"]).values[0],
                      features[7]))

    for name, store in backends:
        streamed, n_batches = {}, 0
        for batch_ids, vectors in store.iter_all(batch_size=777):
            if n_batches == 0:
                check(f'{name}: iter_all column order matches get_vectors',
                      list(vectors.columns) == COLS)
            n_batches += 1
            for row, tile_id in enumerate(np.asarray(batch_ids)):
                streamed[tile_id] = vectors.iloc[row].values
        check(f'{name}: iter_all batches', n_batches > 1,
              f'{n_batches} batches')
        check(f'{name}: iter_all covers every id exactly once',
              len(streamed) == N and set(streamed) == set(ids))
        check(f'{name}: iter_all values match the source',
              np.allclose(np.stack([streamed[i] for i in ids]), features))

    for name, store in backends:
        fast = ml.get_detections(store, model, 0.5, batch_size=777)
        slow = ml.get_detections(NoStream(store), model, 0.5, batch_size=777)
        fast = fast.sort_values('tile_id').reset_index(drop=True)
        slow = slow.sort_values('tile_id').reset_index(drop=True)
        check(f'{name}: streaming and fallback agree on ids',
              list(fast.tile_id) == list(slow.tile_id), f'{len(fast)} detections')
        check(f'{name}: streaming and fallback agree on probabilities',
              np.allclose(fast.probability, slow.probability))
        check(f'{name}: streaming and fallback agree on geometry',
              fast.geometry.equals(slow.geometry))
        check(f'{name}: detections recover the positive cluster',
              N_POSITIVE * 0.5 < len(fast) < N_POSITIVE * 2)

    detected = [ml.get_detections(store, model, 0.5).sort_values('tile_id')
                .tile_id.tolist() for _, store in backends]
    check('backends agree with each other', detected[0] == detected[1])

    for name, store in backends:
        try:
            store.get_vectors(['tile_1', 'no_such_tile', 'tile_2'])
            check(f'{name}: missing id raises', False)
        except KeyError as err:
            check(f'{name}: missing id raises', 'out of sync' in str(err))

        vectors = store.get_vectors(['tile_5', 'tile_5', 'tile_1'])
        check(f'{name}: duplicate ids are allowed',
              len(vectors) == 3
              and np.allclose(vectors.values[0], features[5])
              and np.allclose(vectors.values[2], features[1]))
        check(f'{name}: no NaN reaches the caller',
              not vectors.isna().any().any())
        check(f'{name}: empty request returns no rows',
              len(store.get_vectors([])) == 0)

    try:
        duck.get_vectors([f'no_such_tile_{i}' for i in range(9)])
        check('the error names the missing ids', False)
    except KeyError as err:
        check('the error names the missing ids',
              '9 requested id(s)' in str(err) and '...' in str(err))

    # An out-of-sync store is caught in get_detections too, where ids come from
    # the store rather than the caller.
    truncated = es.from_duckdb(duck.gdf.iloc[:100].copy(), con, 'emb')
    try:
        ml.get_detections(truncated, model, 0.5, batch_size=777)
        check('get_detections catches an out-of-sync store', False)
    except ValueError as err:
        check('get_detections catches an out-of-sync store',
              'out of sync' in str(err))

    con.close()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('failed: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
