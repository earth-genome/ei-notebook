#!/usr/bin/env python3
"""Build facility footprints from known point locations and patch embeddings.

Given point lat/lons for facilities we already know exist, but no spatial
footprints, this rebuilds a linear-probe binary classifier from the known points
and uses the merged patch-level inference output as the footprints.

    1. Load the points as positives, snapped to the nearest embedding patch.
    2. Randomly sample negatives at --neg-ratio : 1, at least --neg-min-dist-m
       from any known positive.
    3. Train a classifier (logistic regression by default).
    4. Choose the threshold from the peak of the F-beta curve, computed on
       cross-validated out-of-fold probabilities.
    5. Run inference across the full AOI.
    6. Merge detected patches to polygons and keep only those containing (or
       within --match-tol-m of) a known positive.
    7. Write a statistical summary.
    8. Write raw detections, filtered patches, and merged footprints.

Two storage backends, selected by which arguments are given:

    --embeddings PATH.parquet                  # single embeddings parquet
    --centroids PATH.parquet --duckdb PATH.db  # DuckDB + centroids parquet

Both stream, so neither requires the full embedding matrix in RAM.

Example:

    python build_footprints.py \
        --positives USA_KS1133_2025-05-02_centroids.geojson \
        --embeddings 20230101-20240101_USA_Kansas-deduped.parquet \
        --boundary usa_kansas.geojson \
        --reference-polygons USA_KS1133_2025-05-02.geojson \
        --tag ks_feedlots

Derived from ParquetEmbeddingsML / DuckDBEmbeddingsML and their ml_utils. The
stages live in the footprints package alongside this script, which holds only
argument parsing and the run sequence; see footprints/__init__.py for the module
map. Evaluation against reference polygons is optional and lives in
evaluate_footprints.py.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

# Chosen before footprints.modeling imports pyplot: this is a batch script and
# must not require a display.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from footprints.backends import make_backend  # noqa: E402
from footprints.geometry import (boundary_mask, build_squares,  # noqa: E402
                                choose_metric_crs, detect_stride, project)
from footprints.labels import (load_points, sample_negatives,  # noqa: E402
                               snap_positives)
from footprints.modeling import fbeta_curve_fig, pr_curve_fig  # noqa: E402
from footprints.pipeline import (detect_and_filter, label_metrics,  # noqa: E402
                                 mine_hard_negatives, seeded_admission,
                                 select_best_round, train_model)
from footprints.reporting import write_config, write_stats  # noqa: E402
from footprints.util import log, warn  # noqa: E402

try:
    import evaluate_footprints
except ImportError:  # evaluation is optional
    evaluate_footprints = None


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().isoformat(timespec='minutes').replace(':', '')
    basename = f'{args.tag}_{stamp}'
    ctx = {'basename': basename}

    # --- Backend and grid ---------------------------------------------------
    backend = make_backend(args)
    metric_crs = choose_metric_crs(backend.centroids_ll, backend.source_crs,
                                   args.metric_crs)
    ctx['metric_crs'] = metric_crs.to_string()
    log(f'Metric CRS: {ctx["metric_crs"]}')

    log('Projecting centroids...')
    centroids_m = project(backend.centroids_ll, backend.source_crs, metric_crs)

    log('Building spatial index...')
    full_tree = cKDTree(centroids_m)
    stride_m = args.stride_m or detect_stride(full_tree, centroids_m, args.seed)
    if args.footprint_geometry == 'patch':
        cell_size_m = args.patch_size_m or 2 * stride_m
    else:
        cell_size_m = stride_m
    ctx['stride_m'] = stride_m
    ctx['cell_size_m'] = cell_size_m
    log(f'Centroid stride {stride_m:.2f} m; footprint cell {cell_size_m:.2f} m '
        f'({cell_size_m**2/1e4:.2f} ha)')

    max_snap = (args.max_snap_dist_m if args.max_snap_dist_m is not None
                else 1.5 * stride_m)
    match_tol = (args.match_tol_m if args.match_tol_m is not None
                 else stride_m)
    nbr_radius = 1.5 * stride_m
    args.max_snap_dist_m, args.match_tol_m = max_snap, match_tol
    ctx['nbr_radius_m'] = nbr_radius

    # --- Working AOI --------------------------------------------------------
    boundary_geom = None
    if args.boundary:
        log(f'Clipping the patch set to {args.boundary}...')
        boundary_geom = gpd.read_file(args.boundary).to_crs(
            metric_crs).geometry.union_all()
        keep_mask = boundary_mask(centroids_m, boundary_geom)
    else:
        keep_mask = np.ones(backend.n_patches, dtype=bool)
    work_positions = np.flatnonzero(keep_mask)
    ctx['n_work'] = int(len(work_positions))
    log(f'Working AOI: {ctx["n_work"]:,} of {backend.n_patches:,} patches')
    if ctx['n_work'] == 0:
        raise SystemExit('No patches inside --boundary.')
    work_tree = cKDTree(centroids_m[work_positions])

    # --- Step 1: positives --------------------------------------------------
    pos = load_points(args.positives, metric_crs,
                      keep_field=args.source_field if args.trusted_source
                      else None)
    ctx['n_pos_read'] = int(len(pos))
    log(f'Positives: {ctx["n_pos_read"]:,} points read')
    pos_xy = np.column_stack([pos.geometry.x, pos.geometry.y])

    if boundary_geom is not None:
        inside = boundary_mask(pos_xy, boundary_geom)
        ctx['n_pos_outside'] = int((~inside).sum())
        if ctx['n_pos_outside']:
            warn(f'{ctx["n_pos_outside"]:,} positives fall outside '
                 '--boundary; dropped.')
        pos, pos_xy = pos[inside].reset_index(drop=True), pos_xy[inside]
    else:
        ctx['n_pos_outside'] = 0

    keep, pos_positions, snap_dists = snap_positives(
        pos_xy, work_tree, work_positions, max_snap)
    ctx['n_pos_far'] = int((~keep).sum())
    if ctx['n_pos_far']:
        warn(f'{ctx["n_pos_far"]:,} positives lie more than {max_snap:.0f} m '
             'from any patch centroid; dropped.')
    pos = pos[keep].reset_index(drop=True)
    pos_xy, pos_positions = pos_xy[keep], pos_positions[keep]
    ctx['snap_dists'] = snap_dists[keep]
    ctx['n_pos_used'] = int(len(pos))
    if ctx['n_pos_used'] == 0:
        raise SystemExit('No positives survived snapping; check the inputs.')

    pos_patches = np.unique(pos_positions)
    ctx['n_pos_patches'] = int(len(pos_patches))
    if ctx['n_pos_patches'] < ctx['n_pos_used']:
        log(f'  {ctx["n_pos_used"] - ctx["n_pos_patches"]:,} positives share a '
            'patch with another; training on unique patches.')

    # --- Step 2: negatives --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.n_negatives is not None:
        n_neg_target = int(args.n_negatives)
        log('Negative count set absolutely (--n-negatives), ignoring '
            '--neg-ratio.')
    else:
        n_neg_target = int(round(args.neg_ratio * ctx['n_pos_patches']))
        # A ratio alone under-samples the background when the positive set is
        # small relative to the AOI: 10:1 on 50 points is 500 negatives for a
        # whole state. The floor scales with the area actually being searched.
        floor = int(round(args.min_negatives_per_million
                          * ctx['n_work'] / 1e6))
        if n_neg_target < floor:
            log(f'Ratio gives {n_neg_target:,} negatives; raising to the floor '
                f'of {floor:,} '
                f'({args.min_negatives_per_million:g} per million of the '
                f'{ctx["n_work"]:,} patches in the AOI).')
            n_neg_target = floor
    ctx['n_neg_floor'] = (None if args.n_negatives is not None
                          else int(round(args.min_negatives_per_million
                                         * ctx['n_work'] / 1e6)))
    ctx['n_neg_target'] = n_neg_target
    log(f'Sampling {n_neg_target:,} negatives at least '
        f'{args.neg_min_dist_m:.0f} m from any positive...')
    neg_patches, n_drawn, n_rejected = sample_negatives(
        rng, work_positions, centroids_m, pos_xy, pos_patches,
        n_neg_target, args.neg_min_dist_m)
    ctx['n_neg_drawn'] = int(n_drawn)
    ctx['n_neg_rejected'] = int(n_rejected)
    ctx['n_neg_used'] = int(len(neg_patches))
    log(f'Negatives: {ctx["n_neg_used"]:,} kept, {n_rejected:,} rejected of '
        f'{n_drawn:,} drawn')

    # --- Steps 3-6, once per hard-negative round ----------------------------
    label_positions = np.concatenate([pos_patches, neg_patches])
    y = np.concatenate([np.ones(len(pos_patches), dtype='i8'),
                        np.zeros(len(neg_patches), dtype='i8')])
    label_source = np.array(['positive'] * len(pos_patches)
                            + ['random'] * len(neg_patches), dtype=object)
    log(f'Fetching embedding vectors for {len(label_positions):,} labeled '
        'patches...')
    X = backend.fetch(label_positions).astype(np.float32)

    env = {
        'backend': backend, 'args': args, 'centroids_m': centroids_m,
        'keep_mask': keep_mask if args.boundary else None,
        'n_work': ctx['n_work'], 'cell_size_m': cell_size_m,
        'match_tol': match_tol, 'pos_points': shapely.points(pos_xy),
        'pos_tree': cKDTree(pos_xy),
    }
    hard_min_dist = (args.hard_neg_min_dist_m
                     if args.hard_neg_min_dist_m is not None
                     else args.neg_min_dist_m)

    # Seeded admission, when the caller names one or more trusted provenances.
    # Runs before any rounds: train on the trusted subset, sharpen it with mining,
    # admit the rest on merit, and only then start the main rounds. Falls back to
    # the cross-validated gate below when no trusted source is named.
    if args.trusted_source:
        trusted = {s for spec in args.trusted_source for s in spec.split(',')}
        log(f'Seeded admission on --source-field "{args.source_field}", '
            f'trusting {sorted(trusted)}:')
        # Patch -> provenance. Where several positives share a patch, a trusted
        # provenance wins; otherwise the first seen.
        src_of_patch = {}
        for p, s in zip(pos_positions, pos[args.source_field].to_numpy()):
            if p not in src_of_patch or s in trusted:
                src_of_patch[p] = s
        src = np.array([src_of_patch.get(p, '') for p in label_positions],
                       dtype=object)

        keep, adm = seeded_admission(X, y, src, trusted, args,
                                     env=env, positions=label_positions,
                                     rng=rng)
        ctx['admission'] = adm
        if not keep.all():
            ctx['admission']['excluded_positions'] = \
                label_positions[~keep].tolist()
            X, y = X[keep], y[keep]
            label_positions, label_source = (label_positions[keep],
                                             label_source[keep])
            src = src[keep]
        label_source = np.where(
            y == 1,
            np.array([f'positive:{s or "?"}' for s in src], dtype=object),
            label_source)

    ctx['rounds'] = []
    # Every round's model, threshold, detections and label snapshot, so the run
    # can emit any round's output rather than only the last. Cheap: the bulky X
    # matrix is not duplicated, only the ~14k label arrays and the detections.
    rounds_state = []
    # Label-quality exclusions accumulate across rounds: each round's hard
    # negatives sharpen the boundary, so positives the embeddings do not support
    # keep surfacing. Once excluded a positive is never reinstated.
    lq_state = {'threshold': float(args.drop_positives_below),
                'n_positives_before': int((y == 1).sum()),
                'positions': [], 'oof': [], 'per_round': []}
    for rnd in range(args.hard_negative_rounds + 1):
        if args.hard_negative_rounds:
            log(f'--- Round {rnd} of {args.hard_negative_rounds} '
                f'({int((y == 0).sum()):,} negatives, '
                f'{int((y == 1).sum()):,} positives) ---')
        trained = train_model(X, y, args)

        # Label-quality gate, once, on round 0's out-of-fold probabilities. A
        # positive the model scores near zero when it was held out is a label
        # the embeddings do not support: a mislocated point, a facility invisible
        # in the imagery behind the embeddings, or a mistake. Training on such
        # points drags the boundary out over ordinary background, which is what
        # destroyed the first New Mexico run (14.2% such points there against
        # 1.9% in Kansas). Dropping them and refitting costs one extra CV pass.
        # Round 0 always gates; later rounds only if round 0 found real
        # contamination. Per-round gating rescued a contaminated AOI (New Mexico
        # union IoU 0.484 -> 0.667) but costs recall on a clean one (Kansas lost
        # 106 genuine facilities for tighter polygons). Round 0's exclusion
        # fraction is itself the diagnostic: 0.7% on clean Kansas, 8.8% on
        # contaminated New Mexico. Setting --label-quality-warn to 0 forces
        # per-round gating; setting it above 1 restricts gating to round 0.
        gate_now = args.drop_positives_below > 0 and (
            rnd == 0 or lq_state.get('per_round_enabled'))
        if gate_now:
            is_pos = y == 1
            bad = is_pos & (trained['oof'] < args.drop_positives_below)
            log(f'Label-quality check: {bad.sum():,} of {is_pos.sum():,} '
                f'remaining positive patches score below '
                f'{args.drop_positives_below:g} out-of-fold.')

            # Hard ceiling. Exclusions do not converge on their own -- each round
            # removes a roughly constant share of what remains, because dropping
            # the hardest positives narrows the class and makes the next-hardest
            # look unsupported. Without this the gate would eventually keep only
            # the most stereotypical facilities.
            budget = (int(args.max_positives_excluded
                          * lq_state['n_positives_before'])
                      - len(lq_state['positions']))
            if bad.any() and budget <= 0:
                log(f'  Ceiling reached '
                    f'({args.max_positives_excluded:.0%} of positives already '
                    'excluded); gating stops here.')
                lq_state['cap_hit'] = True
                bad = np.zeros_like(bad)
            elif bad.sum() > budget:
                idx = np.flatnonzero(bad)
                worst = idx[np.argsort(trained['oof'][idx])[:budget]]
                log(f'  Ceiling reached: excluding only the {budget:,} '
                    f'worst-scoring of {bad.sum():,}, to stay within '
                    f'{args.max_positives_excluded:.0%} of positives.')
                lq_state['cap_hit'] = True
                bad = np.zeros_like(bad)
                bad[worst] = True

            lq_state['per_round'].append(int(bad.sum()))
            if bad.any():
                lq_state['positions'] += label_positions[bad].tolist()
                # Captured before the refit: these points leave the labeled set,
                # so no later oof array has an entry for them.
                lq_state['oof'] += trained['oof'][bad].tolist()
                cum = len(lq_state['positions'])
                frac = cum / max(1, lq_state['n_positives_before'])
                log(f'  Excluding them and refitting on the remaining '
                    f'{int(is_pos.sum() - bad.sum()):,} positives '
                    f'({cum:,} excluded so far, {frac:.1%} of the original '
                    f'{lq_state["n_positives_before"]:,})...')
                keep = ~bad
                X, y = X[keep], y[keep]
                label_positions, label_source = (label_positions[keep],
                                                 label_source[keep])
                trained = train_model(X, y, args)

            if rnd == 0:
                frac0 = (len(lq_state['positions'])
                         / max(1, lq_state['n_positives_before']))
                lq_state['round0_frac'] = float(frac0)
                lq_state['per_round_enabled'] = frac0 >= args.label_quality_warn
                if lq_state['per_round_enabled']:
                    log(f'  Round 0 excluded {frac0:.1%} '
                        f'(>= {args.label_quality_warn:.0%}): the point set '
                        'looks contaminated, so the check will run every '
                        'round.')
                else:
                    log(f'  Round 0 excluded {frac0:.1%} '
                        f'(< {args.label_quality_warn:.0%}): the point set '
                        'looks clean, so no further gating.')

        # Round 0 can use a fixed threshold: before any hard negatives the
        # classes are near-separable, the F-beta curve is flat or ragged, and the
        # selected threshold is close to arbitrary. A conservative fixed value
        # yields a cleaner false-positive pool to mine from.
        threshold = trained['threshold']
        source = 'plateau'
        if rnd == 0 and args.round0_threshold is not None:
            threshold = float(args.round0_threshold)
            source = 'fixed'
            log(f'  Round 0: using fixed threshold {threshold:g} '
                f'(the plateau rule would have given '
                f'{trained["threshold"]:.6f})')

        result = detect_and_filter(env, trained['model'], threshold)

        # Per-round output quality, so rounds can be compared and any one of
        # them can become the run's output. All of these are reference-free.
        kept_polys = result['polys'][np.flatnonzero(result['kept'])]
        areas = (shapely.area(kept_polys) / 1e4 if len(kept_polys)
                 else np.zeros(0))
        # Recall of the positives actually in training, patch against patch.
        # Derived from label_positions rather than by subtracting known
        # exclusions, so it stays correct however a positive left the set --
        # label gate, seeded admission, or anything added later. (Subtracting
        # only the gate's exclusions gave recalls above 100% once admission
        # could also remove positives.)
        trained_patches = set(label_positions[y == 1].tolist())
        matched_patches = {int(p) for p, m in zip(pos_positions,
                                                  result['matched_pos']) if m}
        n_trained = len(trained_patches)
        n_excl_so_far = len(lq_state['positions'])
        matched_trained = len(trained_patches & matched_patches)
        rounds_state.append({
            'trained': trained, 'threshold': threshold, 'result': result,
            'label_positions': label_positions.copy(), 'y': y.copy(),
            'label_source': label_source.copy(),
        })
        ctx['rounds'].append({
            'round': rnd,
            'n_negatives': int((y == 0).sum()),
            'n_trained_positives': n_trained,
            'n_excluded_so_far': n_excl_so_far,
            'recall_trained': matched_trained / max(1, n_trained),
            'recall_all': int(result['matched_pos'].sum()) / max(
                1, ctx['n_pos_used']),
            'area_median_ha': float(np.median(areas)) if areas.size else 0.0,
            'area_max_ha': float(areas.max()) if areas.size else 0.0,
            'area_total_ha': float(areas.sum()) if areas.size else 0.0,
            'ha_per_trained_positive': (float(areas.sum()) / max(1, n_trained)),
            'det_per_trained_positive': (result['n_detections']
                                         / max(1, n_trained)),
            'threshold': threshold,
            'threshold_source': source,
            'n_detections': result['n_detections'],
            'n_polys_raw': result['n_polys_raw'],
            'n_polys_kept': result['n_polys_kept'],
            'n_patches_rejected': result['n_patches_rejected'],
            'redetect_poly': int(result['matched_pos'].sum()),
            'oof_fpr': trained['fpr'],
        })

        if rnd == args.hard_negative_rounds:
            break

        hard, info = mine_hard_negatives(
            env, result, label_positions, hard_min_dist,
            args.max_hard_negatives, rng)
        if info['near_positive']:
            log(f'  {info["near_positive"]:,} rejected patches lie within '
                f'{hard_min_dist:.0f} m of a positive and were not used as '
                'hard negatives.')
        if info['dropped_by_cap']:
            log(f'  Sampled {args.max_hard_negatives:,} of '
                f'{info["available"]:,} available hard negatives; '
                f'{info["dropped_by_cap"]:,} not used.')
        if len(hard) == 0:
            log('  No hard negatives available; stopping early.')
            ctx['rounds'][-1]['hard_negatives_added'] = 0
            break
        log(f'  Mining {len(hard):,} hard negatives for the next round...')
        ctx['rounds'][-1]['hard_negatives_added'] = int(len(hard))
        ctx['rounds'][-1]['hard_negative_info'] = info

        X = np.vstack([X, backend.fetch(hard).astype(np.float32)])
        y = np.concatenate([y, np.zeros(len(hard), dtype='i8')])
        label_positions = np.concatenate([label_positions, hard])
        label_source = np.concatenate([
            label_source, np.array([f'hard_r{rnd}'] * len(hard), dtype=object)])

    # --- Which round becomes the run's output -------------------------------
    # 'auto' applies the validated criterion (see select_best_round); 'last'
    # reproduces the original behaviour; an integer forces a round.
    if args.select_round == 'auto':
        pick = select_best_round(ctx['rounds'])
        r = ctx['rounds'][pick]
        log(f'Selected round {pick} of {len(rounds_state)}: lowest '
            f'ha/positive ({r["ha_per_trained_positive"]:.1f}) among rounds '
            f'holding recall(trained) '
            f'({r["recall_trained"]:.1%}).')
    elif args.select_round == 'last':
        pick = len(rounds_state) - 1
    else:
        pick = int(args.select_round)
        if not 0 <= pick < len(rounds_state):
            raise SystemExit(
                f'--select-round {pick} but the run has rounds '
                f'0..{len(rounds_state) - 1}.')
    ctx['selected_round'] = pick
    if pick != len(rounds_state) - 1:
        log(f'Emitting round {pick} as the run output '
            f'(of {len(rounds_state)} rounds).')
    chosen = rounds_state[pick]
    trained, result, threshold = (chosen['trained'], chosen['result'],
                                  chosen['threshold'])
    label_positions = chosen['label_positions']
    y, label_source = chosen['y'], chosen['label_source']
    # `threshold` is the effective one, which differs from the model's selected
    # threshold only when round 0 used --round0-threshold.
    model, oof = trained['model'], trained['oof']

    if args.drop_positives_below > 0:
        ctx['label_quality'] = {
            'threshold': lq_state['threshold'],
            'n_positives_before': lq_state['n_positives_before'],
            'n_excluded': len(lq_state['positions']),
            'frac_excluded': (len(lq_state['positions'])
                              / max(1, lq_state['n_positives_before'])),
            'excluded_positions': lq_state['positions'],
            'excluded_oof': lq_state['oof'],
            'per_round': lq_state['per_round'],
            'cap_hit': bool(lq_state.get('cap_hit')),
            'round0_frac': lq_state.get('round0_frac'),
            'per_round_enabled': bool(lq_state.get('per_round_enabled')),
        }
        if ctx['label_quality']['frac_excluded'] >= args.label_quality_warn:
            warn(f'{ctx["label_quality"]["frac_excluded"]:.1%} of positives are '
                 f'unsupported by the embeddings across all rounds '
                 f'(>= {args.label_quality_warn:.0%}). The point set likely '
                 'contains mislocated or mistaken labels; review the '
                 'excluded-positives output before trusting this run.')

    ctx['n_pos_trained'] = int((y == 1).sum())

    for key in ('model_repr', 'fold_stats', 'selection'):
        ctx[key] = trained[key]
    # Metrics must describe the threshold actually used for inference, which
    # differs from the selected one for a single-round --round0-threshold run.
    ctx.update(label_metrics(y, oof, threshold, args.beta))
    ctx['threshold_source'] = ctx['rounds'][-1]['threshold_source']
    ctx['expected_fp'] = ctx['fpr'] * ctx['n_work']

    det_positions = result['det_positions']
    det_probs = result['det_probs']
    det_xy, det_points = result['det_xy'], result['det_points']
    polys, owner, counts = result['polys'], result['owner'], result['counts']
    matched_pos, kept = result['matched_pos'], result['kept']
    keep_patch = result['keep_patch']
    for key in ('n_detections', 'n_polys_raw', 'n_polys_kept',
                'n_polys_rejected', 'n_patches_kept', 'n_patches_rejected'):
        ctx[key] = result[key]
    ctx['frac_polys_rejected'] = (ctx['n_polys_rejected']
                                  / max(1, ctx['n_polys_raw']))
    ctx['frac_patches_rejected'] = (ctx['n_patches_rejected']
                                    / max(1, ctx['n_detections']))

    for beta in dict.fromkeys([1.0, 0.25, args.beta]):
        fig = fbeta_curve_fig(y, oof, beta, threshold)
        fig.savefig(os.path.join(args.outdir,
                                 f'{basename}_fbeta_F{beta:g}.png'), dpi=140)
        plt.close(fig)
    fig = pr_curve_fig(y, oof)
    fig.savefig(os.path.join(args.outdir, f'{basename}_pr.png'), dpi=140)
    plt.close(fig)

    # --- Step 7: statistics -------------------------------------------------
    detected_set = np.zeros(backend.n_patches, dtype=bool)
    detected_set[det_positions] = True
    ctx['redetect_patch'] = int(detected_set[pos_positions].sum())
    nbrs = full_tree.query_ball_point(pos_xy, r=nbr_radius, workers=-1)
    ctx['redetect_nbr'] = int(sum(
        1 for group in nbrs if len(group) and detected_set[list(group)].any()))
    ctx['redetect_poly'] = int(matched_pos.sum())
    for key in ('patch', 'nbr', 'poly'):
        ctx[f'frac_redetect_{key}'] = (ctx[f'redetect_{key}']
                                       / max(1, ctx['n_pos_used']))
    log(f'Re-detection: patch {ctx["frac_redetect_patch"]:.1%}, '
        f'neighbourhood {ctx["frac_redetect_nbr"]:.1%}, '
        f'polygon {ctx["frac_redetect_poly"]:.1%}')

    kept_idx = np.flatnonzero(kept)
    kept_polys = polys[kept_idx]
    areas_ha = (shapely.area(kept_polys) / 1e4 if len(kept_polys)
                else np.zeros(0))
    ctx['areas_ha'] = areas_ha

    # Per-polygon aggregates in one grouped pass, rather than a mask per polygon.
    assigned = pd.DataFrame({'poly': owner, 'prob': det_probs})
    assigned = assigned[assigned.poly >= 0]
    grouped = assigned.groupby('poly')
    cells = grouped.size().reindex(kept_idx, fill_value=0).to_numpy()
    confidence = grouped.prob.mean().reindex(kept_idx).to_numpy()
    ctx['cells_per_poly'] = cells
    vals, cnts = np.unique(counts[kept_idx], return_counts=True)
    ctx['positives_per_poly'] = {int(v): int(c) for v, c in zip(vals, cnts)}

    ctx['n_fragment_neighbours'] = 0
    rejected_idx = np.flatnonzero(~kept)
    if len(kept_idx) and len(rejected_idx):
        tree = shapely.STRtree(polys[rejected_idx])
        hits = tree.query(shapely.buffer(kept_polys, stride_m),
                          predicate='intersects')
        ctx['n_fragment_neighbours'] = int(len(np.unique(hits[0])))

    # --- Step 8: outputs ----------------------------------------------------
    log('Writing outputs...')
    written = []

    def to_file(gdf, suffix):
        path = os.path.join(args.outdir, f'{basename}_{suffix}.geojson')
        if len(gdf) == 0:
            warn(f'{suffix}: nothing to write (0 features); file skipped.')
            return
        gdf.to_crs('EPSG:4326').to_file(path, driver='GeoJSON')
        written.append(path)

    raw_geom = (build_squares(det_xy, cell_size_m) if args.raw_as_polygons
                else det_points)
    raw = gpd.GeoDataFrame(
        {'probability': det_probs,
         'patch_id': backend.ids[det_positions],
         'poly_id': owner},
        geometry=list(raw_geom), crs=metric_crs)
    to_file(raw, 'detections_raw')

    filtered = gpd.GeoDataFrame(
        {'probability': det_probs[keep_patch],
         'patch_id': backend.ids[det_positions[keep_patch]],
         'poly_id': owner[keep_patch]},
        geometry=list(build_squares(det_xy[keep_patch], cell_size_m)),
        crs=metric_crs)
    to_file(filtered, 'patches_filtered')

    footprints = gpd.GeoDataFrame(
        {'poly_id': kept_idx,
         'confidence': confidence,
         'n_patches': cells,
         'area_ha': areas_ha,
         'n_positives': counts[kept_idx]},
        geometry=list(kept_polys), crs=metric_crs)
    to_file(footprints, 'footprints')

    labels = gpd.GeoDataFrame(
        {'int_class': y,
         'source': label_source,
         'patch_id': backend.ids[label_positions],
         'oof_probability': oof},
        geometry=list(shapely.points(centroids_m[label_positions])),
        crs=metric_crs)
    to_file(labels, 'labels')

    # The excluded positives, as the *input* points rather than patch centroids,
    # so they can be reviewed, corrected and fed back into a later run.
    lq = ctx.get('label_quality', {})
    if lq.get('n_excluded'):
        oof_of = dict(zip(lq['excluded_positions'], lq['excluded_oof']))
        drop = np.isin(pos_positions, np.array(lq['excluded_positions']))
        excluded = pos[drop].copy()
        excluded['oof_probability'] = [oof_of[p] for p in pos_positions[drop]]
        excluded['redetected_anyway'] = matched_pos[drop]
        to_file(excluded, 'positives_excluded')
        ctx['n_excluded_redetected'] = int(matched_pos[drop].sum())

    if args.save_unfiltered_polygons:
        # Every merged polygon, including those no known positive lands on.
        # Written alongside the filtered footprints rather than replacing them:
        # the positives filter is load-bearing for the run statistics, the
        # assessment thresholds and the round-selection criterion, all of which
        # are calibrated on filtered output. Filter on `retained = false` in GIS
        # to see the candidate discoveries.
        by_poly = assigned.groupby('poly')
        all_idx = np.arange(len(polys))
        unfiltered = gpd.GeoDataFrame(
            {'poly_id': all_idx,
             'n_patches': by_poly.size().reindex(all_idx,
                                                 fill_value=0).to_numpy(),
             'confidence': by_poly.prob.mean().reindex(all_idx).to_numpy(),
             'area_ha': shapely.area(polys) / 1e4,
             'n_positives': counts,
             'retained': kept},
            geometry=list(polys), crs=metric_crs)
        to_file(unfiltered, 'polygons_unfiltered')
        n_new = int((~kept).sum())
        log(f'Unfiltered polygons: {len(polys):,} total, {n_new:,} with no '
            f'known positive ({len(polys) / max(1, ctx["n_pos_trained"]):.1f} '
            'polygons per trained positive).')

    if args.save_round_outputs:
        for n, state in enumerate(rounds_state):
            res = state['result']
            idx = np.flatnonzero(res['kept'])
            if not len(idx):
                continue
            polys_n = res['polys'][idx]
            gdf = gpd.GeoDataFrame(
                {'poly_id': idx,
                 'n_patches': [int((res['owner'] == i).sum()) for i in idx],
                 'area_ha': shapely.area(polys_n) / 1e4,
                 'n_positives': res['counts'][idx]},
                geometry=list(polys_n), crs=metric_crs)
            to_file(gdf, f'round{n}_footprints')

    model_path = os.path.join(args.outdir, f'{basename}_model.joblib')
    joblib.dump({'model': model, 'threshold': threshold,
                 'feature_columns': backend.feature_cols,
                 'metric_crs': ctx['metric_crs']}, model_path)
    written.append(model_path)

    stats_path = os.path.join(args.outdir, f'{basename}_stats.txt')
    write_stats(stats_path, args, backend, ctx)
    written.append(stats_path)

    config_path = os.path.join(args.outdir, f'{basename}_config.txt')
    write_config(config_path, args, backend, ctx)
    written.append(config_path)

    # --- Step 7b: optional evaluation --------------------------------------
    if args.reference_polygons:
        if evaluate_footprints is None:
            warn('evaluate_footprints.py not importable; skipping evaluation.')
        else:
            log(f'Evaluating against {args.reference_polygons}...')
            ref = gpd.read_file(args.reference_polygons)
            report, layers = evaluate_footprints.evaluate(
                footprints, ref, metric_crs=metric_crs)
            written += evaluate_footprints.write_evaluation(
                report, layers, args.outdir, basename,
                footprints_path=f'{basename}_footprints.geojson',
                reference_path=args.reference_polygons)
            print(evaluate_footprints.format_report(
                report, f'{basename}_footprints.geojson',
                args.reference_polygons))

    log('Done. Wrote:')
    for path in written:
        print(f'  {path}')
    print()
    with open(stats_path) as f:
        print(f.read())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io = parser.add_argument_group('standard options')
    io.add_argument('--positives', required=True,
                    help='Vector file of known facility locations. Any '
                         'geometry type; centroids are used.')
    io.add_argument('--embeddings',
                    help='Embeddings parquet (features + tile_id + patch '
                         'geometry). Mutually exclusive with --duckdb.')
    io.add_argument('--centroids',
                    help='Centroids parquet (tile_id + point geometry), for '
                         'the DuckDB backend.')
    io.add_argument('--duckdb', help='DuckDB database of embeddings.')
    io.add_argument('--boundary',
                    help='Optional GeoJSON clipping the patch set used for '
                         'sampling and inference.')
    io.add_argument('--reference-polygons',
                    help='Optional reference footprints; triggers the '
                         'evaluation report.')
    io.add_argument('--outdir', default='runs', help='Output directory.')
    io.add_argument('--tag', default='run', help='Output basename prefix.')
    io.add_argument('--select-round', default='auto',
                    help='Which round becomes the run output: "auto" (default: '
                         'least footprint bloat among rounds that hold their '
                         'recall), "last", or a round number. Every round is '
                         'trained and tabulated regardless; add '
                         '--save-round-outputs to write them all.')
    io.add_argument('--save-round-outputs', action='store_true',
                    help='Also write each round\'s footprints as '
                         '*_round<N>_footprints.geojson, for comparing rounds '
                         'in GIS.')
    io.add_argument('--seed', type=int, default=42)

    lab = parser.add_argument_group('standard options: labeled set')
    lab.add_argument('--neg-ratio', type=float, default=10.0,
                     help='Negatives sampled per positive patch.')
    lab.add_argument('--min-negatives-per-million', type=float, default=1000.0,
                     help='Floor on the ratio-derived negative count, per '
                          'million patches in the working AOI, so a small '
                          'point set still gets a background sample '
                          'proportionate to the area searched. Ignored when '
                          '--n-negatives is given.')
    lab.add_argument('--neg-min-dist-m', type=float, default=3000.0,
                     help='Minimum distance from a negative to any known '
                          'positive. Encodes how far from a facility counts as '
                          'background. Its effect on footprint tightness is '
                          'untested.')

    adv = parser.add_argument_group('advanced options')
    adv.add_argument('--trusted-source', action='append', metavar='VALUE',
                     help='One or more trusted provenance values from '
                          '--source-field (repeat the flag, or comma-separate). '
                          'Seeds the model on these positives alone, then admits '
                          'the rest only if that model scores them at or above '
                          '--admit-above. Far stronger than the self-assessed '
                          'gate, which cannot see a whole batch of bad points. '
                          'Without this, the gate below is used instead.')
    adv.add_argument('--source-field', default='source',
                     help='Attribute in the positives file holding provenance, '
                          'used with --trusted-source.')
    adv.add_argument('--seed-rounds', type=int, default=2,
                     help='Hard-negative rounds used to sharpen the trusted-only '
                          'seed before it judges candidates. A round-0 seed is '
                          'far too permissive and admits the look-alikes the '
                          'whole mechanism exists to exclude. Each round costs '
                          'one inference pass. 0 disables sharpening.')
    adv.add_argument('--admit-above', type=float, default=0.5,
                     help='Probability from the trusted-seed model at or above '
                          'which an untrusted positive joins training.')
    adv.add_argument('--drop-positives-below', type=float, default=0.01,
                     help='Exclude positives whose out-of-fold probability '
                          'falls below this and refit once: labels the '
                          'embeddings do not support drag the decision '
                          'boundary over ordinary background. 0 disables.')
    adv.add_argument('--label-quality-warn', type=float, default=0.05,
                     help='Warn loudly when at least this fraction of '
                          'positives is excluded by the check above, and switch '
                          'the check from round 0 only to every round.')
    adv.add_argument('--max-positives-excluded', type=float, default=0.20,
                     help='Hard ceiling on cumulative label-quality '
                          'exclusions, as a fraction of the original '
                          'positives. Exclusions do not converge on their own, '
                          'so this stops the gate eating the positive class; '
                          'hitting it is itself flagged.')
    adv.add_argument('--save-unfiltered-polygons', action='store_true',
                     help='Also write every merged polygon, including those no '
                          'known positive lands on, with a "retained" column. '
                          'This is the only route to discovering unlisted '
                          'facilities -- buyer beware, see the README on '
                          'judging whether the output is usable.')
    adv.add_argument('--footprint-geometry', choices=['patch', 'stride'],
                     default='patch',
                     help='Cell contributed by each detection: the full '
                          'receptive field (2 x stride) or just the stride '
                          'cell.')
    adv.add_argument('--hard-negative-rounds', type=int, default=4,
                     help='Retraining rounds that add the previous round\'s '
                          'rejected detections as negatives. The default of 4 '
                          '(five training passes) is the expected workflow: '
                          'random negatives alone leave the model firing over '
                          'open country, and --select-round auto picks the best '
                          'of the five. 0 gives the original single-pass '
                          'behaviour.')
    adv.add_argument('--hard-neg-min-dist-m', type=float, default=None,
                     help='Minimum distance from a hard negative to any known '
                          'positive, so a fragment of a real facility is not '
                          'taught as a negative. Defaults to '
                          '--neg-min-dist-m.')
    adv.add_argument('--max-hard-negatives', type=int, default=150,
                     help='How many hard negatives to sample per round, '
                          'uniformly from those available. A couple of hundred '
                          'matches hand practice and outperformed using all of '
                          'them.')

    mod = parser.add_argument_group('advanced options: model and threshold')
    mod.add_argument('--model', choices=['logreg', 'mlp'], default='logreg')
    mod.add_argument('--hidden-layers', default='64,16',
                     help='Comma-separated MLP layer sizes.')
    mod.add_argument('--class-weight', choices=['none', 'balanced'],
                     default='none')
    mod.add_argument('--max-iter', type=int, default=1000)
    mod.add_argument('--cv-folds', type=int, default=5,
                     help='Stratified folds for out-of-fold probabilities.')
    mod.add_argument('--beta', type=float, default=1.0,
                     help='Beta of the F-beta curve used to pick the '
                          'threshold. Below 1 favours precision and yields '
                          'tighter footprints.')
    mod.add_argument('--plateau-tol', type=float, default=0.005,
                     help='Take the highest threshold within this of the '
                          'F-beta maximum, among the contiguous plateau '
                          'containing that maximum. 0 gives the literal '
                          'argmax.')
    mod.add_argument('--round0-threshold', type=float, default=None,
                     help='Fixed threshold for round 0 instead of the plateau '
                          'rule, e.g. 0.99. Later hard-negative rounds still '
                          'use the plateau rule. Useful where round 0 is '
                          'near-separable and its curve gives no real signal.')

    # Internal: grid geometry, execution details and overrides that a user has
    # no reason to touch. Hidden from --help via SUPPRESS but fully functional;
    # each is documented here in place of a help string.
    #
    #   --table                DuckDB table name (default 'embeddings').
    #   --n-negatives          Absolute negative count, overriding --neg-ratio
    #                          and the per-million floor.
    #   --max-snap-dist-m      Drop positives farther than this from any patch
    #                          centroid. Default 1.5 x stride.
    #   --stride-m             Override the auto-detected centroid stride.
    #   --patch-size-m         Override the patch cell size in patch mode.
    #   --match-tol-m          A polygon is kept if it comes within this of a
    #                          known positive; 0 requires strict containment.
    #                          Default 1 x stride.
    #   --gap-close-m          Morphological closing before merging, to bridge
    #                          gaps within a fragmented facility.
    #   --merge-buffer-m       Buffer tolerance when unioning cells.
    #   --metric-crs           CRS for distances and areas, e.g. EPSG:5070.
    #                          Defaults to the estimated UTM zone.
    #   --batch-size           Rows per inference batch.
    #   --cache-features       Hold the uint8 feature matrix in RAM (parquet
    #                          backend) to avoid a second read.
    #   --raw-as-polygons      Write raw detections as cells rather than points.
    hid = parser.add_argument_group('internal')
    for name, kwargs in (
            ('--table', dict(default='embeddings')),
            ('--n-negatives', dict(type=int, default=None)),
            ('--max-snap-dist-m', dict(type=float, default=None)),
            ('--stride-m', dict(type=float, default=None)),
            ('--patch-size-m', dict(type=float, default=None)),
            ('--match-tol-m', dict(type=float, default=None)),
            ('--gap-close-m', dict(type=float, default=0.0)),
            ('--merge-buffer-m', dict(type=float, default=2.0)),
            ('--metric-crs', dict(default=None)),
            ('--batch-size', dict(type=int, default=200_000)),
            ('--cache-features', dict(action='store_true')),
            ('--raw-as-polygons', dict(action='store_true'))):
        hid.add_argument(name, help=argparse.SUPPRESS, **kwargs)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
