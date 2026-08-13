"""Full-AOI inference and the patch-to-polygon step.

run_inference() streams the whole store once. merge_to_polygons() dissolves
detected patch squares into footprints; match_positives() is the filter that
keeps only footprints accounted for by a known positive.
"""

import time

import numpy as np
import shapely

from .util import log


def run_inference(backend, model, threshold, batch_size, keep_mask, n_expected):
    """Stream the AOI, returning detected patch positions and probabilities."""
    positions, probs = [], []
    seen = 0
    t_start = time.time()
    for pos, X in backend.iter_all(batch_size, keep_mask=keep_mask):
        p = model.predict_proba(X.astype(np.float32))[:, 1]
        hit = p >= threshold
        if hit.any():
            positions.append(pos[hit])
            probs.append(p[hit])
        seen += len(pos)
        if n_expected:
            pct = 100.0 * seen / n_expected
            rate = seen / max(1e-9, time.time() - t_start)
            log(f'  inference {seen:,}/{n_expected:,} ({pct:5.1f}%) '
                f'{rate/1000:.0f}k patches/s, '
                f'{sum(len(a) for a in positions):,} detections')
    if not positions:
        return np.zeros(0, dtype='i8'), np.zeros(0, dtype='f8')
    return np.concatenate(positions), np.concatenate(probs)


def merge_to_polygons(squares, merge_buffer_m, gap_close_m):
    """Union patch cells into connected footprint polygons.

    Performed in the metric CRS with a real metre buffer rather than the
    notebooks' degree-space buffer: patches from adjacent MGRS tiles carry
    different rotations, so degree-space seams can fail to close.
    """
    if len(squares) == 0:
        return np.zeros(0, dtype=object)
    grow = merge_buffer_m + gap_close_m
    merged = shapely.union_all(shapely.buffer(squares, grow, join_style='mitre'))
    parts = np.asarray(shapely.get_parts(merged), dtype=object)
    if grow > 0:
        parts = shapely.buffer(parts, -grow, join_style='mitre')
        parts = np.asarray(shapely.get_parts(shapely.union_all(parts)),
                           dtype=object)
    # The buffer round-trip leaves duplicated consecutive vertices behind.
    return np.asarray(shapely.remove_repeated_points(parts, tolerance=0.01),
                      dtype=object)


def assign_patches(polys, points):
    """Map each detection point to the index of its containing polygon."""
    owner = np.full(len(points), -1, dtype='i8')
    if len(polys) == 0 or len(points) == 0:
        return owner
    tree = shapely.STRtree(polys)
    pt_i, poly_i = tree.query(points, predicate='intersects')
    # First match wins; polygons are disjoint apart from shared boundaries.
    for p, q in zip(pt_i, poly_i):
        if owner[p] == -1:
            owner[p] = q
    return owner


def match_positives(polys, pos_points, tol_m):
    """Count known positives matched to each polygon.

    Containment is parameter-free and self-scales to facility size, unlike a
    fixed patch-to-point distance which truncates large facilities. The
    tolerance covers the case where an input centroid falls in a hole or just
    outside the blob, for an L-shaped or ring-shaped yard or when the model
    misses the single central patch.
    """
    counts = np.zeros(len(polys), dtype='i8')
    matched_pos = np.zeros(len(pos_points), dtype=bool)
    poly_of_pos = np.full(len(pos_points), -1, dtype='i8')
    if len(polys) == 0 or len(pos_points) == 0:
        return counts, matched_pos, poly_of_pos

    probe = (shapely.buffer(pos_points, tol_m) if tol_m > 0 else pos_points)
    tree = shapely.STRtree(polys)
    pos_i, poly_i = tree.query(probe, predicate='intersects')
    for p, q in zip(pos_i, poly_i):
        counts[q] += 1
        matched_pos[p] = True
        if poly_of_pos[p] == -1:
            poly_of_pos[p] = q
    return counts, matched_pos, poly_of_pos
