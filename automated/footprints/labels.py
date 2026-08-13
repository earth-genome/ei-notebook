"""Building the training set from point locations.

Positives are the known facility points, snapped to the nearest patch centroid.
Negatives are sampled at random from the AOI, held a minimum distance away from
any positive so that unmapped parts of a known facility are unlikely to be
labelled negative.
"""

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from .util import warn


def load_points(path, metric_crs, keep_field=None):
    """Read a vector file and return centroid geometry in the metric CRS.

    keep_field, when given, is carried through as a column (used for the
    provenance field that drives seeded admission).
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise SystemExit(f'{path} has no CRS.')
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    gdf = gdf.to_crs(metric_crs)
    cols = {'geometry': gdf.geometry.centroid}
    if keep_field:
        if keep_field not in gdf.columns:
            raise SystemExit(
                f'--source-field "{keep_field}" not present in {path}. '
                f'Columns are: {[c for c in gdf.columns if c != "geometry"]}')
        cols[keep_field] = gdf[keep_field].astype(str).to_numpy()
    return gpd.GeoDataFrame(cols, crs=metric_crs).reset_index(drop=True)


def snap_positives(pos_xy, tree, work_positions, max_snap_dist):
    """Snap positive points to their nearest in-AOI patch.

    Returns (kept_mask, positions, distances) where positions index the full
    patch array. Points farther than max_snap_dist from any patch are dropped:
    a nearest-neighbour query never fails, so a point outside the embedding
    extent would otherwise attach itself to an edge patch and quietly pollute
    both training and the recall denominator.
    """
    dists, local = tree.query(pos_xy, k=1, workers=-1)
    keep = dists <= max_snap_dist
    return keep, work_positions[local], dists


def sample_negatives(rng, work_positions, centroids_m, pos_xy, exclude,
                     n_target, min_dist):
    """Sample patches at least min_dist from any positive point.

    Oversamples and tops up, because the rejection rate is unknown in advance.
    """
    pos_tree = cKDTree(pos_xy)
    available = np.setdiff1d(work_positions, exclude, assume_unique=False)
    if n_target > len(available):
        warn(f'Requested {n_target:,} negatives but only {len(available):,} '
             'candidate patches exist; taking all of them.')
        n_target = len(available)

    chosen, seen = [], np.zeros(0, dtype='i8')
    n_drawn = n_rejected = 0
    attempts = 0
    while sum(len(c) for c in chosen) < n_target and attempts < 20:
        attempts += 1
        want = n_target - sum(len(c) for c in chosen)
        draw = min(len(available), int(want * 1.6) + 1000)
        cand = rng.choice(available, size=draw, replace=False)
        cand = np.setdiff1d(cand, seen, assume_unique=False)
        if len(cand) == 0:
            break
        seen = np.union1d(seen, cand)
        n_drawn += len(cand)
        d, _ = pos_tree.query(centroids_m[cand], k=1, workers=-1)
        ok = cand[d >= min_dist]
        n_rejected += len(cand) - len(ok)
        chosen.append(ok[:want])

    negatives = np.concatenate(chosen) if chosen else np.zeros(0, dtype='i8')
    if len(negatives) < n_target:
        warn(f'Only {len(negatives):,} of {n_target:,} negatives could be '
             f'placed at least {min_dist:.0f} m from a positive.')
    return negatives, n_drawn, n_rejected
