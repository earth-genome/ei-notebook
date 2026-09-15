"""Reprojection, grid inference, and patch squares.

Everything downstream measures distance and area, so work happens in a metric
CRS chosen from the data. detect_stride() recovers the spacing of the embedding
grid, which sets the footprint cell size.

The grid is laid out geographically: rows of constant latitude a fixed number of
degrees apart, with each row's longitude step scaled by 1/cos(lat) so the ground
spacing stays constant. Individual chips, though, were cut from imagery in their
MGRS tile's UTM zone, which is the frame their footprint is actually square in.
CellFrames keeps those two facts apart -- the stride is read from the geographic
ladder, the cells are built per zone.
"""

import os

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer

from .util import log, warn

# Stride of the published embedding grid: a 320 m chip every 160 m. A constant
# of the product rather than something to be discovered, but still measured per
# run, because a mismatch means the inputs are not what the run assumes.
PRODUCT_STRIDE_M = 160.0


def project(coords, src_crs, dst_crs):
    """Transform an (N, 2) coordinate array between CRSs."""
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = tf.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def choose_metric_crs(centroids_ll, source_crs, override=None):
    """Pick a metric CRS: the override, or the UTM zone the data sits in.

    Taken from the median position rather than the bounding-box centre. A few
    distant patches drag the box centre a long way, and it can land in a zone
    holding none of the data at all: an Australian register with points near
    both Perth and Brisbane put the centre in UTM 53S, which contained 0% of
    the patches, rotating every footprint cell and inflating distances by 2%.
    The median moves with the data instead.
    """
    if override:
        return CRS.from_user_input(override)
    sample = centroids_ll[::max(1, len(centroids_ll) // 10_000)]
    lon, lat = float(np.median(sample[:, 0])), float(np.median(sample[:, 1]))

    # One UTM zone is 6 degrees wide. Data spanning several of them cannot be
    # measured accurately in any single zone, whichever is chosen.
    zones = np.floor((sample[:, 0] + 180) / 6).astype(int) + 1
    lo, hi = np.percentile(zones, [1, 99])
    if hi - lo >= 2:
        # A conformal conic rather than an equal-area one: areas are measured
        # on the ellipsoid now and do not depend on this choice, while the
        # thresholds that do depend on it are distances. Equal-area projections
        # buy exact area by distorting distance with direction, which would
        # bend the Voronoi boundaries the split is built from.
        warn(f'patches span UTM zones {int(lo)}-{int(hi)}; no single zone '
             f'measures all of them well, and distances far from {lon:.1f}E '
             'will be distorted -- consider --metric-crs with a conformal '
             'conic for the region (EPSG:3112 over Australia), or splitting '
             'the AOI.')
    point = shapely.points([[lon, lat]])[0]
    return gpd.GeoSeries([point], crs=source_crs).estimate_utm_crs()


def detect_stride(centroids_ll, expected_m=PRODUCT_STRIDE_M, tol=0.02):
    """Grid stride in metres, read off the latitude ladder.

    Measured in geographic coordinates and converted with a geodesic arc, so no
    projection is involved and the answer does not move with the working CRS.

    Nearest-neighbour distance, which this used to use, is the wrong statistic
    for this grid. Only the north-south spacing is a constant in metres: rows
    sit one fixed latitude step apart, while the east-west step is twice that
    and re-derived per row from 1/cos(lat), leaving neighbouring rows
    unregistered in longitude. The grid also has gaps, so a patch with no
    north-south neighbour has its nearest at the diagonal (~s*sqrt(2)) or in the
    next column (2*s). The median of that mixture ran 2-15% high, and every
    footprint area carried the square of the error.

    Rows are unaffected by either problem: a missing patch does not move the
    rows that remain, and the ladder is regular everywhere.
    """
    lat = np.asarray(centroids_ll)[:, 1]
    rows = np.unique(np.round(lat, 6))
    if len(rows) < 3:
        raise SystemExit(
            f'Only {len(rows)} distinct patch latitudes; too few to measure the '
            'grid stride. Pass --stride-m explicitly.')

    gaps = np.diff(rows)
    gaps = gaps[gaps > 1e-6]
    # Rows missing from a sparse area leave gaps of 2, 3, ... steps. Take a
    # robust small gap as the unit, then fit the step to every gap that is a
    # clean multiple of it, which uses the whole ladder rather than its minimum.
    base = np.percentile(gaps, 1)
    k = np.round(gaps / base)
    ok = (k >= 1) & (np.abs(gaps / base - k) < 0.1)
    if not ok.any():
        raise SystemExit('Patch latitudes are not on a regular ladder; the '
                         'grid stride cannot be measured. Pass --stride-m.')
    step_deg = float(gaps[ok].sum() / k[ok].sum())

    # Geodesic rather than a spherical radius: the metres in a degree of
    # latitude vary by 1% between equator and pole, which is the same size as
    # the discrepancy being checked for.
    lon0, lat0 = float(np.median(centroids_ll[:, 0])), float(np.median(lat))
    _, _, stride_m = Geod(ellps='WGS84').inv(lon0, lat0, lon0, lat0 + step_deg)
    stride_m = abs(float(stride_m))

    log(f'Grid stride {stride_m:.2f} m, from a latitude step of '
        f'{step_deg:.6f} deg over {len(rows):,} rows')
    if expected_m and abs(stride_m - expected_m) > tol * expected_m:
        warn(f'measured stride {stride_m:.2f} m differs from the expected '
             f'{expected_m:.0f} m by more than {tol:.0%}. These may not be '
             'standard embedding tiles, or the patch set may mix products. '
             'Check before trusting the areas, or pass --stride-m.')
    return stride_m


class CellFrames:
    """Which CRS each patch's cell is built in.

    Each chip was cut from imagery in its MGRS tile's UTM zone, so that is the
    frame it is genuinely square in. Building cells there and reprojecting keeps
    them true to the imagery whatever CRS the run measures in; building them in
    the working CRS instead rotates every cell by the angle between the two
    frames, which reached 10 degrees (47 m at the corners) on a continent-wide
    Australian run.

    Without an epsg column there is nothing to group by and cells fall back to
    axis-aligned in the working CRS, which is what every run before the column
    existed did.
    """

    def __init__(self, lonlat=None, epsg=None, source_crs=None, metric_crs=None):
        self.enabled = epsg is not None and lonlat is not None
        self.lonlat = lonlat
        self.epsg = None if epsg is None else np.asarray(epsg)
        self.source_crs = source_crs
        self.metric_crs = metric_crs

    def take(self, idx):
        """The frames for a subset of patches, positionally selected."""
        if not self.enabled:
            return self
        return CellFrames(np.asarray(self.lonlat)[idx], self.epsg[idx],
                          self.source_crs, self.metric_crs)

    def zones(self):
        return np.unique(self.epsg) if self.enabled else np.empty(0, dtype='i8')


def build_squares(centroids_m, size_m, frames=None):
    """Squares of side size_m centred on each patch, in the working CRS.

    Axis-aligned in the working CRS unless frames carries a per-patch epsg, in
    which case each square is built in its own zone and reprojected. Corners are
    transformed and joined by straight chords rather than curves; at 320 m that
    departs from the true edge by well under a millimetre.
    """
    half = size_m / 2.0
    if frames is None or not frames.enabled or not len(centroids_m):
        return shapely.box(centroids_m[:, 0] - half, centroids_m[:, 1] - half,
                           centroids_m[:, 0] + half, centroids_m[:, 1] + half)

    out = np.empty(len(centroids_m), dtype=object)
    for code in frames.zones():
        m = frames.epsg == code
        native = project(np.asarray(frames.lonlat)[m], frames.source_crs,
                         f'EPSG:{int(code)}')
        squares = shapely.box(native[:, 0] - half, native[:, 1] - half,
                              native[:, 0] + half, native[:, 1] + half)
        out[m] = gpd.GeoSeries(squares, crs=f'EPSG:{int(code)}').to_crs(
            frames.metric_crs).to_numpy()
    return out


def geodesic_area_ha(geoms, crs):
    """Polygon areas in hectares, measured on the ellipsoid.

    Projected area carries the square of the working CRS's scale factor: 4.8%
    per polygon on a continent-wide Australian run, and varying by 9 points
    across the area of interest. That lands on a number compared between
    regions and between runs, where the projection it happened to be measured
    in is invisible. Area on the ellipsoid is a property of the ground instead,
    and does not move with --metric-crs at all.

    Distances are deliberately left projected. They drive thresholds with
    kilometres of slack -- a 2.7% shift moves about one match per run -- and the
    merge, the buffer and the Voronoi split are planar operations with no
    geodesic equivalent, so mixing the two would compare a geodesic threshold
    against a boundary built in a projection.
    """
    geoms = np.asarray(geoms, dtype=object)
    if not len(geoms):
        return np.zeros(0)
    lonlat = gpd.GeoSeries(geoms, crs=crs).to_crs('EPSG:4326').to_numpy()
    geod = Geod(ellps='WGS84')
    return np.array([abs(geod.geometry_area_perimeter(g)[0])
                     for g in lonlat]) / 1e4


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
