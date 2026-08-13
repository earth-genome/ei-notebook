"""Reference-free checks on whether a run went wrong.

Every check here is computable without ground-truth polygons, since the case
this pipeline exists for has none. Checks marked informational=True are context
for a human reader and are not counted as warning flags.
"""

import textwrap

import numpy as np


def assess_run(ctx, args):
    """Heuristics that flag a run whose footprints are probably unusable.

    Recovery statistics cannot detect the characteristic failure: when a weak
    model fires across a large share of the AOI, detections coalesce into
    enormous polygons, and any polygon touching a known positive is retained by
    design. New Mexico reported 100% of positives re-detected while producing 12x
    the reference area and a single 471 km^2 polygon covering 71 positives.

    Thresholds are calibrated on the two AOIs worked so far -- good values from
    Kansas and cleaned New Mexico, bad values from contaminated New Mexico -- so
    treat them as smoke alarms, not proof. Returns a list of check dicts.
    """
    checks = []

    def check(name, value, limit, worse, fmt='{:.3g}', note='',
              informational=False):
        """worse: 'above' or 'below' -- which side is the failure.

        informational=True reports the value without ever flagging: for signals
        that proved not to discriminate between good and bad runs.
        """
        if value is None:
            return
        bad = value > limit if worse == 'above' else value < limit
        checks.append({
            'name': name,
            'value': fmt.format(value),
            'limit': ('context only' if informational else
                      ('> ' if worse == 'above' else '< ') + fmt.format(limit)),
            'triggered': bool(bad) and not informational,
            'note': note,
        })

    lq = ctx.get('label_quality') or {}
    check('positives unsupported by the embeddings',
          lq.get('frac_excluded'), args.label_quality_warn, 'above',
          '{:.1%}', 'mislocated or mistaken labels')
    if lq.get('cap_hit'):
        checks.append({
            'name': 'label-quality exclusions hit their ceiling',
            'value': f'{args.max_positives_excluded:.0%}',
            'limit': 'must not bind',
            'triggered': True,
            'note': ('more positives were unsupported than the ceiling allows '
                     'removing, so contamination exceeds what was excluded'),
        })

    # Informational: measured across six runs, this does not separate good from
    # bad. Good New Mexico runs sit at 0.877-0.883 and failed ones at
    # 0.801-0.869 -- overlapping, because a harder AOI lowers it legitimately.
    check('peak F-beta on the labeled set', ctx['selection']['fbeta_max'],
          0.9, 'below', '{:.3f}', informational=True)

    # Informational: does not discriminate, and is not even monotonic. The best
    # Kansas run (union IoU 0.818, 99.9% recall) shows 4 disconnected plateaus
    # while the failed New Mexico runs show 2. Label gating reshapes the
    # probability distribution enough to fragment the curve on good runs.
    check('disconnected plateaus in the F-beta curve',
          ctx['selection'].get('n_substantive_runs'), 1, 'above', '{:.0f}',
          informational=True)

    n_pos = max(1, ctx['n_pos_used'])
    # Detections per known positive rather than per unit of AOI: the share of
    # the AOI depends on facility density, so it false-flags a dense region (and
    # the synthetic fixture). Per-positive separates cleanly on the runs so far:
    # 12-34 for acceptable runs, 71-306 for failed ones.
    check('detected patches per known positive',
          ctx['n_detections'] / n_pos, 50, 'above', '{:.0f}',
          'the model is firing far more than the known facilities explain')
    check('detections as a share of the AOI',
          ctx['n_detections'] / max(1, ctx['n_work']), 0.0075, 'above',
          '{:.3%}', informational=True)

    check('raw polygons rejected by the positives filter',
          ctx['frac_polys_rejected'], 0.95, 'above', '{:.1%}',
          'nearly everything detected was spurious')

    # Against the positives the final model actually trained on: gated-out
    # points are not targets, so counting them here understates coverage.
    n_target = max(1, ctx.get('n_pos_trained') or ctx['n_pos_used'])
    check('retained polygons per trained positive',
          ctx['n_polys_kept'] / n_target, 0.7, 'below', '{:.2f}',
          'footprints are merging across facilities')

    areas = np.asarray(ctx['areas_ha'], dtype=float)
    if areas.size:
        check('largest footprint (ha)', float(areas.max()), 2000.0, 'above',
              '{:,.0f}', 'larger than any plausible facility')
        # Concentration is only meaningful with enough polygons to concentrate:
        # with 25, the largest five hold a third of the area by construction.
        if areas.size >= 50:
            top5 = float(np.sort(areas)[-5:].sum() / max(1e-9, areas.sum()))
            check('share of total area in the 5 largest footprints', top5, 0.25,
                  'above', '{:.0%}', 'a few blobs dominate the output')

    ppp = ctx.get('positives_per_poly') or {}
    if ppp:
        # 20, not the 5 that Kansas alone suggested: a legitimate New Mexico run
        # merges up to 8 genuinely adjacent facilities, while failed runs reach
        # 29-71.
        check('most positives inside one footprint', float(max(ppp)), 20,
              'above', '{:.0f}',
              'one polygon is laundering many positives')

    return checks


def write_assessment(write, checks, echo=False):
    """Render the assessment block; returns the number of triggered checks."""
    fired = [c for c in checks if c['triggered']]
    lines = ['=' * 74]
    if fired:
        lines.append(f'RUN ASSESSMENT: {len(fired)} of {len(checks)} checks '
                     'flagged -- treat these footprints with suspicion')
    else:
        lines.append(f'RUN ASSESSMENT: all {len(checks)} checks passed')
    lines.append('=' * 74)
    for c in checks:
        mark = '!!' if c['triggered'] else 'ok'
        limit = (c['limit'] if c['limit'] == 'context only'
                 else f'flag if {c["limit"]}')
        lines.append(f'  {mark}  {c["name"]:<52} {c["value"]:>9}  ({limit})')
        if c['triggered'] and c['note']:
            lines.append(f'        -> {c["note"]}')
    if fired:
        lines.append('')
        lines.append(textwrap.fill(
            'Recovery statistics cannot detect this class of failure: a single '
            'enormous polygon "recovers" every positive it covers. Check the '
            'footprint areas below, and view the output against imagery before '
            'using it.', width=72, initial_indent='  ',
            subsequent_indent='  '))
    lines.append('')
    text = '\n'.join(lines) + '\n'
    write(text)
    if echo:
        print(text, flush=True)
    return len(fired)
