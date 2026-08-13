"""The two text artefacts a run leaves behind.

write_config() records every parameter, input fingerprint and validation metric,
so a run can be reproduced or audited later. write_stats() is the statistical
summary, and carries the assessment block.
"""

import hashlib
import os
import platform
import sys
import textwrap
from datetime import datetime

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely

from .assessment import assess_run, write_assessment
from .util import WARNINGS


def file_fingerprint(path):
    """Size plus a partial hash: full hashing of multi-GB inputs is too slow."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(str(size).encode())
    with open(path, 'rb') as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return f'{size:,} bytes, partial sha1 {h.hexdigest()[:12]}'


def describe_distribution(values, unit=''):
    """One-line summary of a numeric distribution."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 'none'
    return (f'n={v.size:,}  total={v.sum():,.1f}{unit}  min={v.min():,.2f}  '
            f'p10={np.percentile(v, 10):,.2f}  median={np.median(v):,.2f}  '
            f'mean={v.mean():,.2f}  p90={np.percentile(v, 90):,.2f}  '
            f'max={v.max():,.2f}{unit}')


def write_config(path, args, backend, ctx):
    """Every parameter, input fingerprint, and validation metric of the run."""
    cm = ctx['confusion']
    with open(path, 'w') as f:
        w = f.write
        w('=' * 74 + '\n')
        w('FOOTPRINT MODEL RUN CONFIGURATION\n')
        w('=' * 74 + '\n')
        w(f'Run:     {ctx["basename"]}\n')
        w(f'Written: {datetime.now().isoformat(timespec="seconds")}\n')
        w(f'Command: {" ".join(sys.argv)}\n\n')

        w('-- Inputs ' + '-' * 63 + '\n')
        for key, value in backend.describe().items():
            w(f'  {key}: {value}\n')
        for label, p in (('positives', args.positives),
                         ('boundary', args.boundary),
                         ('reference_polygons', args.reference_polygons)):
            if p:
                w(f'  {label}: {os.path.abspath(p)}\n')
                w(f'    {file_fingerprint(p)}\n')
        w('\n')

        w('-- Grid ' + '-' * 65 + '\n')
        w(f'  Patches in dataset:      {backend.n_patches:,}\n')
        w(f'  Patches in working AOI:  {ctx["n_work"]:,}\n')
        w(f'  Embedding dimension:     {backend.n_features}\n')
        w(f'  Detected centroid stride: {ctx["stride_m"]:.2f} m\n')
        w(f'  Footprint cell size:      {ctx["cell_size_m"]:.2f} m '
          f'({args.footprint_geometry} mode, '
          f'{ctx["cell_size_m"]**2/1e4:.2f} ha per cell)\n')
        w(f'  Metric CRS:               {ctx["metric_crs"]}\n')
        w(f'  Source CRS:               {backend.source_crs.to_string()}\n\n')

        w('-- Parameters ' + '-' * 59 + '\n')
        for k, v in sorted(vars(args).items()):
            w(f'  --{k.replace("_", "-")}: {v}\n')
        w('\n')

        w('-- Labeled set ' + '-' * 58 + '\n')
        w(f'  Positive points read:            {ctx["n_pos_read"]:,}\n')
        w(f'  Dropped, outside boundary:       {ctx["n_pos_outside"]:,}\n')
        w(f'  Dropped, snap distance > {args.max_snap_dist_m or 0:.0f} m: '
          f'{ctx["n_pos_far"]:,}\n')
        w(f'  Positive points used:            {ctx["n_pos_used"]:,}\n')
        w(f'  Unique positive patches:         {ctx["n_pos_patches"]:,}\n')
        w(f'  Snap distance (m):               '
          f'{describe_distribution(ctx["snap_dists"])}\n')
        w(f'  Negatives requested:             {ctx["n_neg_target"]:,}\n')
        w(f'  Negatives drawn:                 {ctx["n_neg_drawn"]:,}\n')
        w(f'  Negatives rejected (< {args.neg_min_dist_m:.0f} m from a '
          f'positive): {ctx["n_neg_rejected"]:,}\n')
        w(f'  Negatives used:                  {ctx["n_neg_used"]:,}\n')
        adm = ctx.get('admission')
        if adm:
            w('  Seeded admission (trusted-provenance seed):\n')
            w(f'    Trusted positives:   {adm["trusted"]:,}\n')
            w(f'    Untrusted candidates: {adm["candidates"]:,}\n')
            w(f'    Admitted:            {adm["admitted"]:,} '
              f'(>= {args.admit_above:g} from the seed model)\n')
            for src, d in adm['by_source'].items():
                w(f'      {src}: {d["admitted"]}/{d["n"]} admitted, '
                  f'median probability {d["median_probability"]:.3f}\n')
        lq = ctx.get('label_quality')
        if lq:
            w(f'  Label-quality check at out-of-fold prob < '
              f'{lq["threshold"]:g}:\n')
            w(f'    Positives excluded and refitted: {lq["n_excluded"]:,} of '
              f'{lq["n_positives_before"]:,} ({lq["frac_excluded"]:.1%})\n')
            if lq.get('per_round'):
                w('    Excluded per round: '
                  + ' + '.join(f'{n:,}' for n in lq['per_round']) + '\n')
            if lq.get('round0_frac') is not None:
                w(f'    Round 0 alone excluded {lq["round0_frac"]:.1%}, so '
                  + ('the set looks contaminated and the check ran every '
                     'round.\n' if lq['per_round_enabled'] else
                     'the set looks clean and gating stopped after round 0.\n'))
            if lq['n_excluded']:
                w(f'    Written to {ctx["basename"]}_positives_excluded.geojson '
                  'for review\n')
                if 'n_excluded_redetected' in ctx:
                    w(f'    Of those, still matched by a retained polygon: '
                      f'{ctx["n_excluded_redetected"]:,}\n')
        w(f'  Realized ratio:                  '
          f'{ctx["n_neg_used"] / max(1, ctx["n_pos_patches"]):.2f} : 1\n\n')

        if len(ctx.get('rounds', [])) > 1:
            w('-- Hard-negative rounds ' + '-' * 49 + '\n')
            w('  Each round adds the previous round\'s rejected detections as '
              'negatives.\n')
            w(f'  Hard negatives held at least '
              f'{args.hard_neg_min_dist_m or args.neg_min_dist_m:.0f} m from '
              'any known positive.\n\n')
            w('  round  negatives   threshold  src      detections  polys_raw  '
              'polys_kept  rejected_patches  redetected  mined  of_available\n')
            for r in ctx['rounds']:
                info = r.get('hard_negative_info', {})
                w(f'  {r["round"]:>5}  {r["n_negatives"]:>9,}  '
                  f'{r["threshold"]:>10.6f}  '
                  f'{r.get("threshold_source", "plateau"):<7}  '
                  f'{r["n_detections"]:>10,}  '
                  f'{r["n_polys_raw"]:>9,}  {r["n_polys_kept"]:>10,}  '
                  f'{r["n_patches_rejected"]:>16,}  '
                  f'{r["redetect_poly"]:>10,}  '
                  f'{r.get("hard_negatives_added", 0):>5,}  '
                  f'{info.get("available", 0):>12,}\n')
            sel = ctx.get('selected_round')
            w(f'\n  Metrics elsewhere in this file describe round {sel}, '
              'which was emitted as the run output.\n\n')

            w('-- Per-round output quality ' + '-' * 45 + '\n')
            w('  All reference-free, so these are usable for choosing a round '
              'where no\n  polygons exist to compare against.\n\n')
            w('  round  trained_pos  recall(trained)  recall(all)  '
              'det/pos  median_ha  max_ha  ha/pos\n')
            for r in ctx['rounds']:
                mark = ' *' if r['round'] == sel else '  '
                w(f'{mark}{r["round"]:>5}  {r.get("n_trained_positives", 0):>11,}  '
                  f'{r.get("recall_trained", 0):>15.1%}  '
                  f'{r.get("recall_all", 0):>11.1%}  '
                  f'{r.get("det_per_trained_positive", 0):>7.0f}  '
                  f'{r.get("area_median_ha", 0):>9.1f}  '
                  f'{r.get("area_max_ha", 0):>6.0f}  '
                  f'{r.get("ha_per_trained_positive", 0):>6.1f}\n')
            w('  (* = emitted round)\n\n')

        w('-- Model ' + '-' * 64 + '\n')
        w(f'  Estimator: {ctx["model_repr"]}\n')
        w('  No feature scaling (raw quantized uint8), as in the notebooks.\n')
        w(f'  Cross-validation: {args.cv_folds}-fold stratified, '
          'out-of-fold probabilities\n')
        for fs in ctx['fold_stats']:
            w(f'    fold {fs["fold"]}: n_test={fs["n_test"]:,}  '
              f'AP={fs["average_precision"]:.4f}  '
              f'ROC-AUC={fs["roc_auc"]:.4f}\n')
        w('\n')

        w('-- Threshold ' + '-' * 60 + '\n')
        sel = ctx['selection']
        w(f'  Beta:                     {args.beta}\n')
        w(f'  Plateau tolerance:        {args.plateau_tol}\n')
        w(f'  Selected threshold:       {sel["threshold"]:.6f}\n')
        w(f'  F{args.beta:g} at threshold:        '
          f'{sel["fbeta_at_threshold"]:.4f}\n')
        w(f'  F{args.beta:g} maximum:             {sel["fbeta_max"]:.4f}\n')
        w(f'  Literal argmax threshold: {sel["argmax_threshold"]:.6f}\n')
        w(f'  Peak plateau:             '
          f'{sel["plateau_threshold_range"][0]:.6f} to '
          f'{sel["plateau_threshold_range"][1]:.6f}'
          f'  ({sel["n_plateau_candidates"]:,} candidates)\n')
        if sel.get('n_eligible_runs', 1) > 1:
            w(f'  Within-tolerance points outside the peak plateau: '
              f'{sel["n_eligible_total"] - sel["n_plateau_candidates"]:,} in '
              f'{sel["n_eligible_runs"] - 1} disconnected run(s), discarded.\n')
            w('    A ragged F-beta curve is itself a sign of a weakly '
              'separating model.\n')
        if 'margin' in ctx:
            w(f'  Equivalent decision-function margin: {ctx["margin"]:.4f}\n')
        w('\n')

        w('-- Confusion matrix, out-of-fold, at the selected threshold ' + '-' * 14 + '\n')
        w('                   predicted 0   predicted 1\n')
        w(f'      actual 0     {cm[0][0]:>11,}   {cm[0][1]:>11,}\n')
        w(f'      actual 1     {cm[1][0]:>11,}   {cm[1][1]:>11,}\n')
        w(f'  accuracy  {ctx["accuracy"]:.4f}\n')
        w(f'  precision {ctx["precision"]:.4f}\n')
        w(f'  recall    {ctx["recall"]:.4f}\n')
        w(f'  F1        {ctx["f1"]:.4f}\n')
        w(f'  F{args.beta:g}      {ctx["fbeta"]:.4f}\n\n')

        w('-- Caveats ' + '-' * 62 + '\n')
        w(textwrap.fill(
            'The threshold was selected on the same out-of-fold predictions '
            'reported above, so these metrics carry mild optimism. They are '
            'still far less biased and less variable than picking a threshold '
            'by eye on a single held-out split.',
            width=72, initial_indent='  ', subsequent_indent='  ') + '\n\n')
        w(textwrap.fill(
            f'Precision above is measured against sampled negatives, where the '
            f'negative prior is about '
            f'{ctx["n_neg_used"]/max(1,ctx["n_neg_used"]+ctx["n_pos_patches"]):.0%}. '
            f'Across the working AOI the prior is roughly '
            f'{ctx["n_pos_patches"]/max(1,ctx["n_work"]):.1e}, so field '
            f'precision will be much lower. See the stats file for the '
            f'implied AOI-wide false positive count.',
            width=72, initial_indent='  ', subsequent_indent='  ') + '\n\n')
        w(textwrap.fill(
            f'Negatives were required to be at least {args.neg_min_dist_m:.0f} '
            'm from any known positive, which removes the most '
            'facility-adjacent hard negatives from the sampling frame. For '
            'footprint generation that is the desirable bias: the model should '
            'fire on the scruffy edges of a facility rather than learn sharp '
            'boundaries from near-misses. Whether shrinking it tightens '
            'footprints is untested supposition.',
            width=72, initial_indent='  ', subsequent_indent='  ') + '\n\n')

        if WARNINGS:
            w('-- Warnings ' + '-' * 61 + '\n')
            for m in WARNINGS:
                w(f'  ! {m}\n')
            w('\n')

        w('-- Environment ' + '-' * 58 + '\n')
        w(f'  python {platform.python_version()} on {platform.platform()}\n')
        import sklearn
        mods = {'numpy': np, 'pandas': pd, 'geopandas': gpd, 'shapely': shapely,
                'pyarrow': pa, 'scikit-learn': sklearn, 'joblib': joblib}
        try:
            import duckdb
            mods['duckdb'] = duckdb
        except ImportError:
            pass
        for name, mod in mods.items():
            w(f'  {name} {getattr(mod, "__version__", "?")}\n')


def write_stats(path, args, backend, ctx):
    """The step 7 statistical summary of inference versus the input positives."""
    with open(path, 'w') as f:
        w = f.write
        w('=' * 74 + '\n')
        w('INFERENCE SUMMARY\n')
        w('=' * 74 + '\n')
        w(f'Run:     {ctx["basename"]}\n')
        w(f'Written: {datetime.now().isoformat(timespec="seconds")}\n\n')
        # Not echoed: main prints the whole stats file when the run ends, and
        # this block is the first thing in it.
        ctx['n_checks_flagged'] = write_assessment(
            w, assess_run(ctx, args), echo=False)

        w('-- Area of interest ' + '-' * 54 + '\n')
        w(f'  Patches in dataset:       {backend.n_patches:,}\n')
        w(f'  Patches in working AOI:   {ctx["n_work"]:,}'
          f'{"  (clipped to --boundary)" if args.boundary else ""}\n')
        w(f'  Centroid stride:          {ctx["stride_m"]:.2f} m\n')
        w(f'  Footprint cell:           {ctx["cell_size_m"]:.2f} m '
          f'= {ctx["cell_size_m"]**2/1e4:.2f} ha ({args.footprint_geometry})\n')
        w(f'  Metric CRS:               {ctx["metric_crs"]}\n\n')

        w('-- Detections ' + '-' * 59 + '\n')
        w(f'  Selected threshold:       {ctx["selection"]["threshold"]:.6f}\n')
        w(f'  Raw detected patches:     {ctx["n_detections"]:,}'
          f'  ({100.0 * ctx["n_detections"] / max(1, ctx["n_work"]):.4f}% '
          'of the AOI)\n')
        w(f'  Merged polygons, raw:     {ctx["n_polys_raw"]:,}\n')
        w(f'  Implied AOI-wide false positives at the out-of-fold FPR '
          f'({ctx["fpr"]:.2e}): {ctx["expected_fp"]:,.0f}\n')
        if len(ctx.get('rounds', [])) > 1:
            w(textwrap.fill(
                'MEANINGLESS FOR THIS RUN. The extrapolation assumes the '
                'negatives are a random sample of the AOI. After '
                'hard-negative mining they are deliberately adversarial, so '
                'the out-of-fold FPR is pessimistic by a large and unknowable '
                'factor. Compare rounds in the table above instead.',
                width=72, initial_indent='    ', subsequent_indent='    '))
            w('\n\n')
        else:
            w('    A rough scaling of the sampled-negative false positive '
              'rate to the whole AOI.\n\n')

        w('-- Filtering against known positives ' + '-' * 37 + '\n')
        w(f'  Criterion: polygon contains a known positive, or lies within '
          f'{args.match_tol_m:.0f} m of one\n')
        w(f'  Polygons retained:        {ctx["n_polys_kept"]:,}\n')
        w(f'  Polygons rejected:        {ctx["n_polys_rejected"]:,}'
          f'  ({ctx["frac_polys_rejected"]:.1%} of raw polygons)\n')
        w(f'  Patches retained:         {ctx["n_patches_kept"]:,}\n')
        w(f'  Patches rejected:         {ctx["n_patches_rejected"]:,}'
          f'  ({ctx["frac_patches_rejected"]:.1%} of raw detections)\n')
        if len(ctx.get('rounds', [])) > 1:
            w('  (The implied-estimate comparison is omitted: see the note '
              'under Detections.)\n\n')
        else:
            w(f'  Rejected patches vs the implied estimate above: '
              f'{ctx["n_patches_rejected"]:,} observed against '
              f'{ctx["expected_fp"]:,.0f} predicted '
              f'({ctx["n_patches_rejected"] / max(1, ctx["expected_fp"]):.2f}'
              'x)\n')
            w(textwrap.fill(
                'Both counts are patches, so they are directly comparable. '
                'Ratios above 1 are expected and measure how much easier the '
                'sampled negatives are than the real AOI: they are uniform '
                'random and additionally excluded from the neighbourhood of '
                'known positives, so the hardest look-alikes are largely '
                'absent from training. A large ratio is the signal to add '
                'hard negatives rather than more random ones.',
                width=72, initial_indent='    ',
                subsequent_indent='    ') + '\n\n')

        w('-- Re-detection of the input positives ' + '-' * 35 + '\n')
        w(f'  Positive points used:     {ctx["n_pos_used"]:,}\n')
        w(f'  (1) snapped patch itself fires:        '
          f'{ctx["redetect_patch"]:,}  ({ctx["frac_redetect_patch"]:.1%})\n')
        w(f'  (2) any patch within {ctx["nbr_radius_m"]:.0f} m fires:      '
          f'{ctx["redetect_nbr"]:,}  ({ctx["frac_redetect_nbr"]:.1%})\n')
        w(f'  (3) matched by a retained polygon:     '
          f'{ctx["redetect_poly"]:,}  ({ctx["frac_redetect_poly"]:.1%})'
          '   <- headline recall\n')
        w(f'  Out-of-fold recall at this threshold:  {ctx["recall"]:.1%}\n')
        w('    Criterion (3) is the one that decides which footprints are '
          'written, and is\n'
          '    the client-facing recall: the fraction of their known '
          'facilities that got\n'
          '    a footprint. Read it together with the area statistics below '
          '-- it is\n'
          '    satisfiable by a single enormous blob covering many positives '
          'at once, and\n'
          '    read 100% on a New Mexico run whose footprints were 12x '
          'too large.\n')
        w('    Every positive is in the training set, so (1)-(3) are '
          'in-sample and\n'
          '    flatter the model relative to unseen facilities; the '
          'out-of-fold recall\n'
          '    is the honest patchwise figure.\n\n')

        w('-- Merge diagnostics ' + '-' * 53 + '\n')
        w('  Positives per retained polygon (merge collisions):\n')
        for k, v in sorted(ctx['positives_per_poly'].items()):
            w(f'    {k} positive(s): {v:,} polygon(s)\n')
        w(f'  Retained polygons with a rejected polygon within '
          f'{ctx["stride_m"]:.0f} m\n'
          f'    (fragmentation; consider --gap-close-m): '
          f'{ctx["n_fragment_neighbours"]:,}\n\n')

        w('-- Footprint areas ' + '-' * 55 + '\n')
        w(f'  Areas (ha): {describe_distribution(ctx["areas_ha"], " ha")}\n')
        w(f'  Cells per polygon: {describe_distribution(ctx["cells_per_poly"])}\n')
        w(textwrap.fill(
            f'Areas are quantized in {ctx["stride_m"]:.0f} m steps with a '
            f'floor of {ctx["cell_size_m"]**2/1e4:.2f} ha (one cell). Whether '
            'they are biased relative to true facility extent cannot be '
            'determined from the inputs to this run; it needs an independent '
            'footprint source or visual inspection against imagery.',
            width=72, initial_indent='  ', subsequent_indent='  ') + '\n')
