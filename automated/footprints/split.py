"""One footprint per facility, carved out of the merged detections.

A merged polygon can cover several neighbouring facilities, which leaves
downstream work -- pond extraction above all -- with no way to say which
facility a feature belongs to. This divides such a polygon between its
facilities by competing growth: each one claims its own patch and then grows
outward, and a patch goes to whichever facility reaches it by the shortest path
*through detected patches*. Travelling through the detections rather than across
open ground puts the boundary at the narrow neck between two facilities, where a
straight-line Voronoi split would cut wherever the midline happened to fall.

Adjacency and geometry are both taken from the merge rule rather than from the
patch grid. The embeddings sit on a Major Tom grid whose spacing varies and has
gaps, so anything keyed to a nominal stride strands patches the merge had joined.
"""

import heapq

import numpy as np
import shapely

from .geometry import build_squares
from .util import log


def _merge_geometry(squares, grow):
    """Union cells the way merge_to_polygons does, so seams close the same way."""
    if grow > 0:
        merged = shapely.union_all(shapely.buffer(squares, grow,
                                                  join_style='mitre'))
        return shapely.buffer(merged, -grow, join_style='mitre')
    return shapely.union_all(squares)


def _partition(xy, owner, parent, n_groups):
    """Divide the parent polygon between owners, with no overlap and no gap.

    Partitioning the patches does not partition the ground: cells are about
    twice the grid spacing, so neighbouring cells overlap by roughly half a
    cell, and two facilities would each claim the shared band -- exactly the
    ambiguity this is meant to remove. So ground is assigned instead to its
    nearest patch centroid, and through that to the patch's owner. Every point
    of the parent lands in exactly one part.
    """
    if len(xy) == 1:
        return {int(owner[0]): parent}
    cells = shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(xy), extend_to=parent))
    # voronoi_polygons does not return cells in input order, so each is matched
    # back to the point it contains.
    pts = shapely.points(xy)
    tree = shapely.STRtree(pts)
    owner_of_cell = np.full(len(cells), -1, dtype='i8')
    c_i, p_i = tree.query(cells, predicate='contains')
    for c, p in zip(c_i, p_i):
        if owner_of_cell[c] == -1:
            owner_of_cell[c] = owner[p]
    out = {}
    for g in range(n_groups):
        sel = np.flatnonzero(owner_of_cell == g)
        if not len(sel):
            continue
        piece = shapely.intersection(_union_cells(cells[sel]), parent)
        if not shapely.is_empty(piece) and shapely.area(piece) > 0:
            out[g] = piece
    return out


def _union_cells(cells):
    """Union Voronoi cells, tolerating GEOS robustness failures.

    The cells form a coverage -- they tile without overlapping -- so the
    coverage union applies, which is both faster and free of the overlay
    robustness problems that make a general union throw "unable to assign free
    hole to a shell" on long shared edges. The fallbacks exist because a
    degenerate cell can still break the coverage assumption.
    """
    try:
        return shapely.coverage_union_all(cells)
    except Exception:
        pass
    try:
        # Snapping to a 1 mm grid resolves almost all remaining cases, at a
        # precision far below anything that matters for a footprint.
        return shapely.union_all(cells, grid_size=0.001)
    except Exception:
        return shapely.union_all(shapely.make_valid(cells))


def _adjacency(squares, grow):
    """Neighbour lists under the merge rule: buffered cells that intersect."""
    cells = shapely.buffer(squares, grow, join_style='mitre')
    a_i, b_i = shapely.STRtree(cells).query(cells, predicate='intersects')
    adj = [[] for _ in range(len(squares))]
    for a, b in zip(a_i, b_i):
        if a != b:
            adj[a].append(int(b))
    return adj


def _grow(xy, adj, seed_patch, seed_xy):
    """Shortest-path competition. Returns (owner, distance) per patch.

    Distance accumulates real metres along the path rather than counting hops,
    because the grid spacing is irregular; a seed starts at its own distance
    from its patch centroid so the total measures reach from the facility point.
    """
    n = len(xy)
    owner = np.full(n, -1, dtype='i8')
    dist = np.full(n, np.inf)
    heap = []
    for s, p in enumerate(seed_patch):
        d0 = float(np.hypot(*(xy[p] - seed_xy[s])))
        if d0 < dist[p]:
            dist[p], owner[p] = d0, s
            heapq.heappush(heap, (d0, int(p), s))
    while heap:
        d, u, s = heapq.heappop(heap)
        if d > dist[u] or owner[u] != s:
            continue
        for v in adj[u]:
            step = float(np.hypot(*(xy[v] - xy[u])))
            nd = d + step
            # Strict improvement only, with the seed index breaking exact ties,
            # so the result does not depend on heap ordering.
            if nd < dist[v] - 1e-9 or (abs(nd - dist[v]) <= 1e-9
                                       and s < owner[v]):
                dist[v], owner[v] = nd, s
                heapq.heappush(heap, (nd, v, s))
    return owner, dist


def single_facility_reach(patch_xy, patch_poly, pos_xy, pairs_pos, pairs_poly):
    """Straight-line reach of every footprint that holds exactly one facility.

    These are the unambiguous cases, so they are the only honest basis for
    deciding how far a facility's own ground plausibly extends. Note the
    distribution has a long tail of its own -- a cap taken from it is generous,
    not strict.
    """
    n_per_poly = np.bincount(pairs_poly, minlength=patch_poly.max() + 1
                             if len(patch_poly) else 1)
    reach = []
    for p, q in zip(pairs_pos, pairs_poly):
        if n_per_poly[q] != 1:
            continue
        xy = patch_xy[patch_poly == q]
        if len(xy):
            reach.append(float(np.hypot(xy[:, 0] - pos_xy[p, 0],
                                        xy[:, 1] - pos_xy[p, 1]).max()))
    return np.array(reach)


def split_footprints(kept_idx, patch_xy, patch_poly, pos_xy, pos_ids,
                     pairs_pos, pairs_poly, cell_size_m, grow,
                     max_dist=None):
    """One row per covered facility, plus what could not be attributed.

    Every covered facility gets a row, including those whose footprint needed no
    dividing, so the layer is a complete alternative to the footprints file
    rather than a supplement to be merged with it. Ground beyond max_dist from
    any facility becomes an `unattributed` row for its parent, so the parts
    still reconstruct the parent exactly and nothing disappears silently.

    split_id is <poly_id>-<NN>, numbered from 01 in positive_id order, with 00
    reserved for the unattributed remainder.
    """
    by_poly = {}
    for p, q in zip(pairs_pos, pairs_poly):
        by_poly.setdefault(int(q), []).append(int(p))

    rows, n_split, n_capped = [], 0, 0
    for q, members in sorted(by_poly.items()):
        members = sorted(members, key=lambda i: str(pos_ids[i]))
        sel = np.flatnonzero(patch_poly == q)
        if len(sel) == 0:
            continue
        xy = patch_xy[sel]
        squares = build_squares(xy, cell_size_m)
        seed_xy = pos_xy[members]

        if len(members) == 1:
            owner = np.zeros(len(sel), dtype='i8')
            dist = np.hypot(xy[:, 0] - seed_xy[0, 0], xy[:, 1] - seed_xy[0, 1])
        else:
            adj = _adjacency(squares, grow)
            seed_patch = [int(np.argmin(np.hypot(xy[:, 0] - s[0],
                                                 xy[:, 1] - s[1])))
                          for s in seed_xy]
            owner, dist = _grow(xy, adj, seed_patch, seed_xy)
            n_split += 1

        over = (dist > max_dist) if max_dist else np.zeros(len(sel), bool)
        n_capped += int(over.sum())

        # Two facilities can land on the same patch; at this resolution they are
        # one location and no growth can separate them. Each still gets a row,
        # with identical geometry, so "one row per facility" stays true and the
        # duplication is visible instead of silent.
        # Everything in the parent is partitioned, including the over-cap
        # patches, which share group index len(members) and become the
        # unattributed remainder. Doing it in one pass keeps parts and remainder
        # mutually exclusive and exactly covering.
        parent = _merge_geometry(squares, grow)
        group = np.where(over, len(members), owner)
        pieces = _partition(xy, group, parent, len(members) + 1)
        seed_patch_of = {s: pieces.get(s) for s in range(len(members))}

        for s, i in enumerate(members):
            geom = seed_patch_of[s]
            duplicated = False
            if geom is None:
                # Unseparated from a neighbour: carry that neighbour's geometry.
                donor = next((seed_patch_of[t] for t in range(len(members))
                              if seed_patch_of[t] is not None), None)
                geom, duplicated = donor, donor is not None
            if geom is None:
                continue
            rows.append({
                'split_id': f'{int(kept_idx[q])}-{s + 1:02d}',
                'poly_id': int(kept_idx[q]),
                'positive_id': str(pos_ids[i]),
                'attributed': True,
                'geometry_shared': bool(duplicated),
                'n_patches': int(((owner == s) & ~over).sum()),
                'area_ha': float(shapely.area(geom) / 1e4),
                'geometry': geom,
            })
        if over.any() and pieces.get(len(members)) is not None:
            geom = pieces[len(members)]
            rows.append({
                'split_id': f'{int(kept_idx[q])}-00',
                'poly_id': int(kept_idx[q]),
                'positive_id': '',
                'attributed': False,
                'geometry_shared': False,
                'n_patches': int(over.sum()),
                'area_ha': float(shapely.area(geom) / 1e4),
                'geometry': geom,
            })
    log(f'  {n_split} footprints divided between facilities; '
        f'{len(rows)} rows'
        + (f', {n_capped:,} patches beyond {max_dist:,.0f} m left unattributed'
           if max_dist else ''))
    return rows
