#!/usr/bin/env python3
"""Triage positive points that a trained probe scores as negative.

For each low-scoring positive, this scores every patch in its neighbourhood too,
which separates the two failure modes:

    mislocated  - the point itself scores low but a nearby patch scores high, so
                  the facility is there and the point needs moving. The reported
                  offset says where.
    no_signal   - nothing in the neighbourhood scores high: either not a facility,
                  not visible in the imagery behind these embeddings, or a
                  facility form the model has not learned.

Usage:

    python3 review_rejected_positives.py \
        --positives USA_NM591_2026-06-04_centroids.geojson \
        --model runs/nm_t99_n10k_2026-08-11T1332_model.joblib \
        --centroids NMcentroids.parquet --duckdb NMembeddings.db \
        --out nm_positives_review.geojson

Writes every positive, sorted worst-first, with its own score, the best score in
its neighbourhood, and the offset to that best patch.
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import joblib
import numpy as np
import shapely
from scipy.spatial import cKDTree

from footprints.backends import DuckDBBackend
from footprints.geometry import choose_metric_crs, detect_stride, project


def main(positives, model, centroids, duckdb, out, radius_m=None,
         reject_below=0.01, promising_above=0.5, table='embeddings'):
    bundle = joblib.load(model)
    clf, threshold = bundle['model'], bundle['threshold']
    print(f'model {model}\n  threshold {threshold:.6f}')

    backend = DuckDBBackend(centroids, duckdb, table=table)
    metric_crs = choose_metric_crs(backend.centroids_ll, backend.source_crs)
    centroids_m = project(backend.centroids_ll, backend.source_crs, metric_crs)
    tree = cKDTree(centroids_m)
    stride = detect_stride(tree, centroids_m)
    radius = radius_m if radius_m is not None else 3 * stride
    print(f'  stride {stride:.1f} m; neighbourhood radius {radius:.0f} m')

    pos = gpd.read_file(positives)
    pos_m = pos.to_crs(metric_crs)
    xy = np.c_[pos_m.geometry.x, pos_m.geometry.y]

    # Own patch, plus every patch within the neighbourhood radius.
    own_d, own_i = tree.query(xy, k=1, workers=-1)
    groups = tree.query_ball_point(xy, r=radius, workers=-1)
    wanted = np.unique(np.concatenate([np.asarray(g, dtype='i8')
                                       for g in groups] + [own_i]))
    print(f'  scoring {len(wanted):,} patches for {len(pos):,} positives...')
    X = backend.fetch(wanted).astype(np.float32)
    probs = clf.predict_proba(X)[:, 1]
    prob_of = dict(zip(wanted.tolist(), probs.tolist()))

    rows = []
    for n, (own, group) in enumerate(zip(own_i, groups)):
        g = np.asarray(group, dtype='i8')
        gp = np.array([prob_of[int(k)] for k in g]) if len(g) else np.array([0.0])
        best = int(g[int(np.argmax(gp))]) if len(g) else int(own)
        own_p = prob_of[int(own)]
        best_p = float(gp.max())
        offset = centroids_m[best] - xy[n]
        rows.append({
            'own_probability': own_p,
            'best_nearby_probability': best_p,
            'offset_m': float(np.hypot(*offset)),
            'offset_east_m': float(offset[0]),
            'offset_north_m': float(offset[1]),
            'n_patches_scored': int(len(g)),
            'snap_dist_m': float(own_d[n]),
        })

    df = gpd.GeoDataFrame(rows, geometry=pos.geometry.to_numpy(), crs=pos.crs)
    for col in pos.columns:
        if col != 'geometry':
            df[col] = pos[col].to_numpy()

    def classify(r):
        if r.own_probability >= threshold:
            return 'ok'
        if r.best_nearby_probability >= promising_above:
            return 'mislocated'
        if r.own_probability < reject_below:
            return 'no_signal'
        return 'weak'

    df['verdict'] = df.apply(classify, axis=1)
    df = df.sort_values(['own_probability', 'best_nearby_probability'])

    df.to_file(out, driver='GeoJSON')
    print(f'\nwrote {out}: {len(df)} points')
    print(df.verdict.value_counts().to_string())
    mis = df[df.verdict == 'mislocated']
    if len(mis):
        print(f'\n"mislocated" offsets (m): median {mis.offset_m.median():.0f}, '
              f'max {mis.offset_m.max():.0f} — move the point this far to the '
              'best-scoring patch')
    print('\nSort by own_probability ascending in QGIS; verdict column groups '
          'the cases.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--positives', required=True)
    ap.add_argument('--model', required=True,
                    help='A *_model.joblib from a build_footprints run.')
    ap.add_argument('--centroids', required=True)
    ap.add_argument('--duckdb', required=True)
    ap.add_argument('--table', default='embeddings')
    ap.add_argument('--out', default='positives_review.geojson')
    ap.add_argument('--radius-m', type=float, default=None,
                    help='Neighbourhood radius. Default 3 x stride (~480 m).')
    ap.add_argument('--reject-below', type=float, default=0.01)
    ap.add_argument('--promising-above', type=float, default=0.5,
                    help='A nearby patch at or above this marks the point as '
                         'mislocated rather than signal-free.')
    main(**vars(ap.parse_args()))
