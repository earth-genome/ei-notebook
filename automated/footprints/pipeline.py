"""One pass of the pipeline, and the loops built on top of it.

detect_and_filter() is a single train-threshold-infer-filter pass. Around it:
hard-negative mining, which feeds rejected detections back as negatives;
seeded admission, which trains on trusted provenance and admits the rest only
if they score well; and select_best_round(), which picks a mining round without
reference polygons to compare against.
"""

import numpy as np
import shapely
from sklearn import metrics as skmetrics

from .geometry import build_squares
from .inference import (assign_patches, match_positives, merge_to_polygons,
                        run_inference)
from .modeling import _fit, make_model, oof_probabilities, select_threshold
from .util import log, warn


def train_model(X, y, args):
    """Cross-validate, pick a threshold, and refit on everything.

    Pure with respect to the embedding store: it sees only the labeled matrix,
    so it is reusable across hard-negative rounds.
    """
    log(f'{args.cv_folds}-fold cross-validation for out-of-fold '
        'probabilities...')
    oof, fold_stats = oof_probabilities(X, y, args)
    selection = select_threshold(y, oof, args.beta, args.plateau_tol)
    threshold = selection['threshold']
    log(f'Selected threshold {threshold:.6f} '
        f'(F{args.beta:g} = {selection["fbeta_at_threshold"]:.4f}, '
        f'max {selection["fbeta_max"]:.4f})')
    if threshold > 1 - 1e-6:
        warn(f'Selected threshold {threshold:.10f} is at the top of the '
             'probability range; the classifier is saturated and the F-beta '
             'plateau may be degenerate.')

    log('Refitting on all labeled data...')
    model = _fit(make_model(args), X, y)

    out = {
        'model': model,
        'model_repr': repr(model),
        'oof': oof,
        'fold_stats': fold_stats,
        'selection': selection,
        'threshold': threshold,
        **label_metrics(y, oof, threshold, args.beta),
    }
    return out


def label_metrics(y, oof, threshold, beta):
    """Out-of-fold label-set metrics at a given threshold."""
    y_pred = (oof >= threshold).astype(int)
    cm = skmetrics.confusion_matrix(y, y_pred, labels=[0, 1])
    out = {
        'confusion': cm.tolist(),
        'accuracy': float(skmetrics.accuracy_score(y, y_pred)),
        'precision': float(skmetrics.precision_score(y, y_pred,
                                                     zero_division=0)),
        'recall': float(skmetrics.recall_score(y, y_pred, zero_division=0)),
        'f1': float(skmetrics.f1_score(y, y_pred, zero_division=0)),
        'fbeta': float(skmetrics.fbeta_score(y, y_pred, beta=beta,
                                             zero_division=0)),
        'fpr': float(cm[0][1] / max(1, cm[0].sum())),
    }
    if 0 < threshold < 1:
        out['margin'] = float(np.log(threshold / (1 - threshold)))
    return out


def detect_and_filter(env, model, threshold):
    """Run inference, merge to polygons, and filter against known positives.

    env carries the run-invariant setup (backend, geometry, positives), so this
    can be called once per hard-negative round.
    """
    log(f'Running inference over {env["n_work"]:,} patches...')
    det_positions, det_probs = run_inference(
        env['backend'], model, threshold, env['args'].batch_size,
        env['keep_mask'], env['n_work'])
    log(f'Detections: {len(det_positions):,}')

    det_xy = env['centroids_m'][det_positions]
    det_points = shapely.points(det_xy) if len(det_xy) else np.zeros(0, object)

    log('Merging detected patches into polygons...')
    squares = build_squares(det_xy, env['cell_size_m'])
    polys = merge_to_polygons(squares, env['args'].merge_buffer_m,
                              env['args'].gap_close_m)
    log(f'Merged into {len(polys):,} polygons')

    owner = assign_patches(polys, det_points)
    if (owner == -1).any():
        warn(f'{int((owner == -1).sum()):,} detected patches could not be '
             'assigned to a merged polygon; excluded from the filtered '
             'patch output.')

    counts, matched_pos, poly_of_pos = match_positives(
        polys, env['pos_points'], env['match_tol'])
    kept = counts > 0
    keep_patch = np.isin(owner, np.flatnonzero(kept))
    log(f'Retained {int(kept.sum()):,} polygons, rejected '
        f'{int((~kept).sum()):,}')

    return {
        'det_positions': det_positions,
        'det_probs': det_probs,
        'det_xy': det_xy,
        'det_points': det_points,
        'polys': polys,
        'owner': owner,
        'counts': counts,
        'matched_pos': matched_pos,
        'poly_of_pos': poly_of_pos,
        'kept': kept,
        'keep_patch': keep_patch,
        'n_detections': int(len(det_positions)),
        'n_polys_raw': int(len(polys)),
        'n_polys_kept': int(kept.sum()),
        'n_polys_rejected': int((~kept).sum()),
        'n_patches_kept': int(keep_patch.sum()),
        'n_patches_rejected': int(len(det_positions) - keep_patch.sum()),
    }


def mature_seed(X, y, positions, env, args, rng):
    """Train the trusted-only seed, then sharpen it with hard-negative rounds.

    A round-0 model trained on trusted positives plus *random* negatives is far
    too permissive to judge candidates: measured on New Mexico it scored a batch
    of near-distribution look-alikes at median probability 0.316, admitting 55 of
    116 and degrading the run below doing nothing at all (union IoU 0.533 against
    0.750 for hand curation). The same trusted-only model after mining scored them
    at median 0.000. Each round costs one full-AOI inference pass.
    """
    Xs, ys, ps = X, y, positions
    trained = train_model(Xs, ys, args)
    for r in range(args.seed_rounds):
        log(f'  Seed round {r + 1}/{args.seed_rounds}: mining against the '
            'trusted-only model...')
        result = detect_and_filter(env, trained['model'], trained['threshold'])
        hard, info = mine_hard_negatives(
            env, result, ps, args.hard_neg_min_dist_m or args.neg_min_dist_m,
            args.max_hard_negatives, rng)
        if not len(hard):
            log('    no hard negatives available; seed is already clean.')
            break
        log(f'    +{len(hard):,} hard negatives')
        Xs = np.vstack([Xs, env['backend'].fetch(hard).astype(np.float32)])
        ys = np.concatenate([ys, np.zeros(len(hard), dtype='i8')])
        ps = np.concatenate([ps, hard])
        trained = train_model(Xs, ys, args)
    return trained


def seeded_admission(X, y, sources, trusted, args, env=None, positions=None,
                     rng=None):
    """Train on trusted-provenance positives only, then admit the rest on merit.

    A probability floor applied by a model trained on *everything* cannot catch a
    batch of bad points that share an appearance: they define part of the positive
    class and the model scores them confidently. Measured on New Mexico, 108
    points from one source had median out-of-fold probability 1.000 under such a
    model and 0.000 under a model trained only on trusted points -- same points,
    same embeddings. Seeding from the trusted subset removes that circularity.

    Returns (keep_mask, info). keep_mask covers all rows of y: trusted positives
    and admitted untrusted positives are True, rejected untrusted positives False,
    and every negative is True.
    """
    is_pos = y == 1
    trusted_mask = is_pos & np.isin(sources, list(trusted))
    candidate_mask = is_pos & ~trusted_mask
    info = {'trusted': int(trusted_mask.sum()),
            'candidates': int(candidate_mask.sum()),
            'admitted': 0, 'by_source': {}}
    if not trusted_mask.any():
        raise SystemExit(
            f'--trusted-source {sorted(trusted)} matched no positives. '
            f'Values present: {sorted(set(sources[is_pos]))}')
    if not candidate_mask.any():
        log('  All positives are from a trusted source; nothing to admit.')
        return np.ones(len(y), bool), info

    seed_rows = trusted_mask | (y == 0)
    log(f'  Seeding on {int(trusted_mask.sum()):,} trusted positives '
        f'({", ".join(sorted(trusted))}) plus {int((y == 0).sum()):,} negatives...')
    if args.seed_rounds and env is not None:
        seed_trained = mature_seed(X[seed_rows], y[seed_rows],
                                   positions[seed_rows], env, args, rng)
        seed = seed_trained['model']
        info['seed_rounds'] = int(args.seed_rounds)
        info['seed_threshold'] = float(seed_trained['threshold'])
    else:
        seed = _fit(make_model(args), X[seed_rows], y[seed_rows])
        info['seed_rounds'] = 0
    probs = seed.predict_proba(X[candidate_mask])[:, 1]
    admit = probs >= args.admit_above

    keep = np.ones(len(y), bool)
    idx = np.flatnonzero(candidate_mask)
    keep[idx[~admit]] = False
    info['admitted'] = int(admit.sum())
    for s in sorted(set(sources[candidate_mask])):
        sel = sources[idx] == s
        info['by_source'][s] = {
            'n': int(sel.sum()),
            'admitted': int(admit[sel].sum()),
            'median_probability': float(np.median(probs[sel])),
        }
    log(f'  Admitted {int(admit.sum()):,} of {int(candidate_mask.sum()):,} '
        f'untrusted positives scoring >= {args.admit_above:g}; '
        f'{int((~admit).sum()):,} set aside.')
    for s, d in info['by_source'].items():
        log(f'    {s}: {d["admitted"]}/{d["n"]} admitted '
            f'(median probability {d["median_probability"]:.3f})')
    return keep, info


#: Rounds whose recall(trained) is within this of the best are treated as
#: equally good on recall, so the choice falls to footprint bloat. 0.5 percentage
#: points: wide enough to ignore one or two facilities of noise, narrow enough to
#: exclude the genuine recall drops observed (2.4 points in one New Mexico round).
RECALL_TOLERANCE = 0.005


def select_best_round(rounds, tol=RECALL_TOLERANCE):
    """Pick the round with the least footprint bloat that keeps its recall.

    Criterion: lowest ha per trained positive, among rounds whose recall of the
    positives they trained on is within `tol` of the best any round achieved.

    Both quantities are reference-free, which is the point -- in production there
    are no polygons to compare against. The justification is that where reference
    polygons *do* exist, reference IoU rises as ha/positive falls, including
    through a turning point: on Kansas's own centroids ha/positive bottomed out at
    round 2 (37.3) and rose to 41.3 by round 4, while reference IoU peaked at
    0.818 on round 2 and fell back to 0.667 -- so the last round was measurably
    not the best, and this rule picks the right one. Validated on three series
    across two states; see docs/automated-planning.md.
    """
    best_recall = max(r['recall_trained'] for r in rounds)
    eligible = [r for r in rounds if r['recall_trained'] >= best_recall - tol]
    return min(eligible, key=lambda r: r['ha_per_trained_positive'])['round']


def mine_hard_negatives(env, result, known_positions, min_dist, cap, rng):
    """Patches detected outside every retained polygon: the AOI's own hardest
    negatives.

    These are the look-alikes that uniform random sampling almost never finds --
    and they include the genuinely negative clusters worth learning from, such
    as suburban construction, so they are sampled uniformly with no filtering by
    blob size or shape.

    min_dist keeps hard negatives away from *known* positives, so a fragment of
    a listed facility is not taught as a negative. Nothing can protect against
    an *unlisted* real facility being mined; that risk is managed by keeping the
    budget small (see --max-hard-negatives), which dilutes any such mistake
    among the random negatives.

    Returns (positions, info).
    """
    info = {'available': 0, 'near_positive': 0, 'dropped_by_cap': 0}

    rejected = result['det_positions'][~result['keep_patch']]
    rejected = np.setdiff1d(rejected, known_positions)
    if len(rejected) == 0:
        return rejected, info

    dists, _ = env['pos_tree'].query(env['centroids_m'][rejected], k=1,
                                     workers=-1)
    far = dists >= min_dist
    info['near_positive'] = int((~far).sum())
    hard = rejected[far]
    info['available'] = int(len(hard))

    hard = hard[rng.permutation(len(hard))]
    if cap and len(hard) > cap:
        info['dropped_by_cap'] = len(hard) - cap
        hard = hard[:cap]
    return hard, info
