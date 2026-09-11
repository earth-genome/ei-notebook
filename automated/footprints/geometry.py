"""Reprojection, grid inference, and patch squares.

Everything downstream measures distance and area, so work happens in a metric
CRS chosen from the data. detect_stride() recovers the centroid spacing of the
embedding grid, which sets the footprint cell size.
"""

import os

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Transformer

from .util import warn


def project(coords, src_crs, dst_crs):
    """Transform an (N, 2) coordinate array between CRSs."""
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = tf.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def choose_metric_crs(centroids_ll, source_crs, override=None):
    """Pick a metric CRS: the override, or the UTM zone of the data bbox."""
    if override:
        return CRS.from_user_input(override)
    sample = centroids_ll[::max(1, len(centroids_ll) // 10_000)]
    box = shapely.box(sample[:, 0].min(), sample[:, 1].min(),
                      sample[:, 0].max(), sample[:, 1].max())
    return gpd.GeoSeries([box], crs=source_crs).estimate_utm_crs()


def detect_stride(tree, centroids_m, seed=42, n_sample=20_000):
    """Median nearest-neighbour centroid distance, in metres.

    Queried against the full tree (not a subsampled one, which would inflate
    the distance). For a 50%-overlapping sliding window this is half the patch
    width.
    """
    rng = np.random.default_rng(seed)
    n = len(centroids_m)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    dists, _ = tree.query(centroids_m[idx], k=2, workers=-1)
    return float(np.median(dists[:, 1]))


def build_squares(centroids_m, size_m):
    """Axis-aligned squares of side size_m centred on each point."""
    half = size_m / 2.0
    return shapely.box(centroids_m[:, 0] - half, centroids_m[:, 1] - half,
                       centroids_m[:, 0] + half, centroids_m[:, 1] + half)


def load_boundary(path, metric_crs):
    """Read a boundary file and return one geometry in the metric CRS.

    Parts are repaired before they are dissolved. Published administrative
    boundaries -- GADM especially -- routinely contain self-intersections and
    misordered rings, and unioning them raw throws "side location conflict"
    from GEOS. Repairing first costs a pass over the geometry and turns a crash
    into a usable AOI.
    """
    parts = gpd.read_file(path).to_crs(metric_crs).geometry.to_numpy()
    invalid = ~shapely.is_valid(parts)
    if invalid.any():
        warn(f'{int(invalid.sum())} of {len(parts)} boundary parts in '
             f'{os.path.basename(path)} are invalid; repairing.')
        parts = shapely.make_valid(parts)
    try:
        return shapely.union_all(parts)
    except shapely.errors.GEOSException:
        # Snapping to a millimetre grid clears the residual cases, well below
        # any precision a boundary carries.
        return shapely.union_all(parts, grid_size=0.001)


def boundary_mask(centroids_m, boundary_geom):
    """Boolean mask of centroids inside a (prepared) boundary geometry."""
    minx, miny, maxx, maxy = boundary_geom.bounds
    x, y = centroids_m[:, 0], centroids_m[:, 1]
    mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    idx = np.flatnonzero(mask)
    shapely.prepare(boundary_geom)
    inside = shapely.contains_xy(boundary_geom, x[idx], y[idx])
    mask[idx] = inside
    return mask
