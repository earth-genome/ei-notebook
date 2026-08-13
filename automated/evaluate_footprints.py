#!/usr/bin/env python3
"""Compare a set of predicted facility footprints against reference polygons.

Runnable standalone:

    python evaluate_footprints.py --footprints run_footprints.geojson \
        --reference USA_KS1133_2025-05-02.geojson --outdir runs --tag ks_feedlots

or imported by build_footprints.py, which calls evaluate() / format_report() /
write_evaluation() directly.

Correspondence is computed in both directions, because one facility can fragment
into several output polygons and two nearby facilities can merge into one.

Caveat for the Kansas reference set: those polygons are themselves the merged
patch output of a prior model run on this same 160 m grid, not independently
measured footprints. High agreement therefore demonstrates reproduction of the
prior workflow, not independent accuracy, and the metrics here cannot arbitrate
between --footprint-geometry patch and stride (the reference is built from
320 m squares, so `patch` agrees with it by construction).
"""

from __future__ import annotations

import argparse
import os
import textwrap
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def _quantiles(values, qs=(0, 10, 25, 50, 75, 90, 100)):
    """Return a {label: value} dict of percentiles, empty if no values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return {f'p{q}': float(np.percentile(values, q)) for q in qs}


def _clean(gdf, metric_crs, name):
    """Reproject, drop empties, and repair invalid geometry."""
    if gdf.crs is None:
        raise ValueError(f'{name} has no CRS; cannot evaluate.')
    gdf = gdf.to_crs(metric_crs).reset_index(drop=True)
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].apply(
            shapely.make_valid)
    return gdf


def evaluate(footprints, reference, metric_crs=None):
    """Compare predicted footprints to reference polygons.

    Arguments:
        footprints: GeoDataFrame of predicted polygons.
        reference: GeoDataFrame of reference polygons.
        metric_crs: CRS for areas and distances. Defaults to the UTM zone
            estimated from the reference geometry.

    Returns: (report, layers), where report is a dict of metrics and layers is a
        dict of {name: GeoDataFrame} of diagnostic layers for GIS inspection.
    """
    if metric_crs is None:
        # Estimate from whichever layer has geometry; an empty one cannot.
        basis = reference if len(reference) else footprints
        metric_crs = (basis.estimate_utm_crs() if len(basis)
                      else reference.crs)

    ref = _clean(reference, metric_crs, 'reference')
    out = _clean(footprints, metric_crs, 'footprints')

    ref_area = ref.geometry.area.to_numpy()
    out_area = out.geometry.area.to_numpy()

    report = {
        'metric_crs': str(metric_crs),
        'n_reference': int(len(ref)),
        'n_output': int(len(out)),
        'reference_area_ha_total': float(ref_area.sum() / 1e4),
        'output_area_ha_total': float(out_area.sum() / 1e4),
    }

    if len(ref) == 0 or len(out) == 0:
        report['note'] = 'One of the two layers is empty; no correspondence.'
        report['n_overlapping_pairs'] = 0
        report['recovery_rate'] = 0.0
        report['agreement_rate'] = 0.0
        report['refs_missed'] = int(len(ref))
        report['outputs_novel'] = int(len(out))
        layers = {
            'missed_refs': ref.to_crs(reference.crs) if len(ref) else None,
            'novel_polys': out.to_crs(footprints.crs) if len(out) else None,
        }
        return report, {k: v for k, v in layers.items() if v is not None}

    # Candidate pairs, then drop mere edge contacts (zero-area intersections).
    tree = shapely.STRtree(out.geometry.to_numpy())
    ref_i, out_i = tree.query(ref.geometry.to_numpy(), predicate='intersects')
    if len(ref_i):
        inter = shapely.intersection(
            ref.geometry.to_numpy()[ref_i], out.geometry.to_numpy()[out_i])
        inter_area = shapely.area(inter)
        keep = inter_area > 0
        ref_i, out_i, inter_area = ref_i[keep], out_i[keep], inter_area[keep]
    else:
        inter_area = np.array([])

    pairs = pd.DataFrame({
        'ref_i': ref_i,
        'out_i': out_i,
        'inter_area': inter_area,
        'ref_area': ref_area[ref_i] if len(ref_i) else [],
        'out_area': out_area[out_i] if len(out_i) else [],
    })
    pairs['pair_iou'] = pairs.inter_area / (
        pairs.ref_area + pairs.out_area - pairs.inter_area)

    report['n_overlapping_pairs'] = int(len(pairs))

    if len(pairs) == 0:
        report['note'] = ('No output polygon overlaps any reference polygon. '
                          'Check that the two layers cover the same area.')
        report.update({
            'refs_matched': 0, 'refs_missed': int(len(ref)),
            'recovery_rate': 0.0, 'outputs_matched': 0,
            'outputs_novel': int(len(out)), 'agreement_rate': 0.0,
            'global_iou': 0.0,
        })
        return report, {
            'missed_refs': ref.to_crs(reference.crs),
            'novel_polys': out.to_crs(footprints.crs),
        }

    # --- Correspondence, both directions -------------------------------------
    outs_per_ref = pairs.groupby('ref_i').size().reindex(
        range(len(ref)), fill_value=0)
    refs_per_out = pairs.groupby('out_i').size().reindex(
        range(len(out)), fill_value=0)

    matched_refs = outs_per_ref > 0
    matched_outs = refs_per_out > 0
    report['refs_matched'] = int(matched_refs.sum())
    report['refs_missed'] = int((~matched_refs).sum())
    report['recovery_rate'] = float(matched_refs.mean())
    report['outputs_matched'] = int(matched_outs.sum())
    report['outputs_novel'] = int((~matched_outs).sum())
    report['agreement_rate'] = float(matched_outs.mean())

    one_to_one = 0
    for r, o in zip(pairs.ref_i, pairs.out_i):
        if outs_per_ref[r] == 1 and refs_per_out[o] == 1:
            one_to_one += 1
    report['correspondence'] = {
        'one_to_one': int(one_to_one),
        'refs_split_across_multiple_outputs': int((outs_per_ref > 1).sum()),
        'outputs_merging_multiple_refs': int((refs_per_out > 1).sum()),
        'refs_with_no_output': int((~matched_refs).sum()),
        'outputs_with_no_ref': int((~matched_outs).sum()),
        'outputs_per_ref_counts': {
            str(k): int(v) for k, v in
            outs_per_ref.value_counts().sort_index().items()},
        'refs_per_output_counts': {
            str(k): int(v) for k, v in
            refs_per_out.value_counts().sort_index().items()},
    }

    # --- Overlap quality, per reference --------------------------------------
    best_iou = pairs.groupby('ref_i').pair_iou.max()
    report['best_match_iou'] = {
        'mean': float(best_iou.mean()),
        **_quantiles(best_iou.to_numpy()),
    }

    # Union IoU: reference against the union of all its matching outputs. The
    # fragmentation-robust version, and the headline number.
    out_geoms = out.geometry.to_numpy()
    ref_geoms = ref.geometry.to_numpy()
    rows = []
    for r, grp in pairs.groupby('ref_i'):
        union = shapely.union_all(out_geoms[grp.out_i.to_numpy()])
        u_area = shapely.area(union)
        i_area = shapely.area(shapely.intersection(ref_geoms[r], union))
        denom = ref_area[r] + u_area - i_area
        rows.append({
            'ref_i': r,
            'union_iou': i_area / denom if denom > 0 else 0.0,
            'coverage': i_area / ref_area[r] if ref_area[r] > 0 else 0.0,
            'excess': (max(0.0, u_area - i_area) / u_area if u_area > 0
                       else 0.0),
            'area_ratio': u_area / ref_area[r] if ref_area[r] > 0 else np.nan,
            'n_outputs': len(grp),
        })
    per_ref = pd.DataFrame(rows)

    report['union_iou'] = {
        'mean': float(per_ref.union_iou.mean()),
        **_quantiles(per_ref.union_iou.to_numpy()),
    }
    report['coverage_of_reference'] = {
        'mean': float(per_ref.coverage.mean()),
        **_quantiles(per_ref.coverage.to_numpy()),
    }
    report['excess_beyond_reference'] = {
        'mean': float(per_ref.excess.mean()),
        **_quantiles(per_ref.excess.to_numpy()),
    }
    report['area_ratio_output_over_reference'] = {
        'median': float(per_ref.area_ratio.median()),
        'mean': float(per_ref.area_ratio.mean()),
        'total': float(out_area.sum() / ref_area.sum()),
    }

    # Matching-free aggregate. Assumes output polygons are mutually disjoint
    # (they come from a unary union) and reference polygons nearly so.
    inter_total = float(pairs.inter_area.sum())
    union_total = float(ref_area.sum() + out_area.sum() - inter_total)
    report['global_iou'] = inter_total / union_total if union_total > 0 else 0.0

    # --- Diagnostic layers ---------------------------------------------------
    layers = {}
    missed = ref.loc[~matched_refs.to_numpy()]
    if len(missed):
        layers['missed_refs'] = missed.to_crs(reference.crs)
    novel = out.loc[~matched_outs.to_numpy()]
    if len(novel):
        layers['novel_polys'] = novel.to_crs(footprints.crs)

    best_pair = pairs.loc[pairs.groupby('out_i').pair_iou.idxmax()]
    matched = out.loc[best_pair.out_i.to_numpy()].copy()
    matched['ref_id'] = best_pair.ref_i.to_numpy()
    matched['pair_iou'] = best_pair.pair_iou.to_numpy()
    lookup = per_ref.set_index('ref_i')
    matched['union_iou'] = lookup.union_iou.reindex(matched.ref_id).to_numpy()
    matched['coverage'] = lookup.coverage.reindex(matched.ref_id).to_numpy()
    matched['excess'] = lookup.excess.reindex(matched.ref_id).to_numpy()
    matched['area_ha'] = matched.geometry.area.to_numpy() / 1e4
    if len(matched):
        layers['matched'] = matched.to_crs(footprints.crs)

    return report, layers


def format_report(report, footprints_path=None, reference_path=None):
    """Render an evaluation report dict as readable text."""
    lines = []
    add = lines.append

    add('=' * 74)
    add('FOOTPRINT EVALUATION AGAINST REFERENCE POLYGONS')
    add('=' * 74)
    add(f'Written: {datetime.now().isoformat(timespec="seconds")}')
    if footprints_path:
        add(f'Footprints: {footprints_path}')
    if reference_path:
        add(f'Reference:  {reference_path}')
    add(f'Metric CRS: {report["metric_crs"]}')
    add('')

    add('-- Counts ' + '-' * 63)
    add(f'  Reference polygons:            {report["n_reference"]:>10,}')
    add(f'  Output polygons:               {report["n_output"]:>10,}')
    add(f'  Overlapping pairs:             {report.get("n_overlapping_pairs", 0):>10,}')
    add(f'  Reference total area (ha):     {report["reference_area_ha_total"]:>10,.1f}')
    add(f'  Output total area (ha):        {report["output_area_ha_total"]:>10,.1f}')
    add('')

    if 'note' in report:
        add(textwrap.fill(report['note'], width=72, initial_indent='  ',
                          subsequent_indent='  '))
        add('')
    if 'refs_matched' not in report:
        return '\n'.join(lines) + '\n'

    add('-- Correspondence ' + '-' * 55)
    add('  (Agreement with a prior polygon set. The recall that matters to a')
    add('   client is "positives matched by a retained polygon" in the stats')
    add('   file -- their own record of facilities -- not these numbers.)')
    add(f'  Reference polygons recovered:  {report["refs_matched"]:>10,}'
        f'  ({report["recovery_rate"]:.1%})')
    add(f'  References missed:             {report["refs_missed"]:>10,}')
    add(f'  Outputs with a reference:      {report["outputs_matched"]:>10,}'
        f'  ({report["agreement_rate"]:.1%})')
    add(f'  Outputs with no reference:     {report["outputs_novel"]:>10,}')
    if 'correspondence' not in report:
        add('')
        add(f'  Global IoU: {report["global_iou"]:.3f}')
        return '\n'.join(lines) + '\n'
    c = report['correspondence']
    add(f'  One-to-one pairs:              {c["one_to_one"]:>10,}')
    add(f'  References split across >1 output (fragmentation): '
        f'{c["refs_split_across_multiple_outputs"]:,}')
    add(f'  Outputs merging >1 reference (merge collision):    '
        f'{c["outputs_merging_multiple_refs"]:,}')
    add(f'  Outputs per reference: {c["outputs_per_ref_counts"]}')
    add(f'  References per output: {c["refs_per_output_counts"]}')
    add('')

    def block(title, d, fmt='{:.3f}'):
        add(f'  {title}')
        keys = [k for k in ('mean', 'median', 'total', 'p0', 'p10', 'p25',
                            'p50', 'p75', 'p90', 'p100') if k in d]
        add('    ' + '  '.join(f'{k}={fmt.format(d[k])}' for k in keys))

    add('-- Overlap quality (per reference, over recovered references) ' + '-' * 12)
    block('Union IoU (reference vs union of its outputs) [headline]',
          report['union_iou'])
    block('Best single-pair IoU', report['best_match_iou'])
    block('Coverage of reference (intersection / reference area)',
          report['coverage_of_reference'])
    block('Excess beyond reference (output area outside ref / output area)',
          report['excess_beyond_reference'])
    block('Area ratio (output / reference)',
          report['area_ratio_output_over_reference'])
    add('')
    add(f'  Global IoU (total intersection / total union): '
        f'{report["global_iou"]:.3f}')
    add('')

    add('-- Caveats ' + '-' * 62)
    add(textwrap.fill(
        'If the reference polygons were themselves produced by a patch-based '
        'model run on the same embedding grid, agreement here measures '
        'reproduction of that workflow rather than independent accuracy, and '
        'says nothing about absolute footprint area. If the positives used for '
        'training were derived from these reference polygons, the comparison '
        'is additionally in-sample.',
        width=72, initial_indent='  ', subsequent_indent='  '))

    return '\n'.join(lines) + '\n'


def write_evaluation(report, layers, outdir, basename,
                     footprints_path=None, reference_path=None):
    """Write the evaluation text report and diagnostic geojson layers."""
    os.makedirs(outdir, exist_ok=True)
    written = []

    eval_path = os.path.join(outdir, f'{basename}_eval.txt')
    with open(eval_path, 'w') as f:
        f.write(format_report(report, footprints_path, reference_path))
    written.append(eval_path)

    for name, gdf in layers.items():
        if gdf is None or len(gdf) == 0:
            continue
        path = os.path.join(outdir, f'{basename}_eval_{name}.geojson')
        gdf.to_file(path, driver='GeoJSON')
        written.append(path)

    return written


def main(footprints, reference, outdir='.', tag=None, metric_crs=None):
    fp = gpd.read_file(footprints)
    ref = gpd.read_file(reference)
    report, layers = evaluate(fp, ref, metric_crs=metric_crs)
    if tag is None:
        tag = os.path.basename(footprints).split('.geojson')[0]
    written = write_evaluation(report, layers, outdir, tag,
                               footprints_path=footprints,
                               reference_path=reference)
    print(format_report(report, footprints, reference))
    for path in written:
        print(f'Wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare predicted facility footprints to reference '
                    'polygons: correspondence counts, IoU distributions, '
                    'coverage and excess.')
    parser.add_argument('--footprints', required=True,
                        help='GeoJSON/GPKG of predicted footprint polygons.')
    parser.add_argument('--reference', required=True,
                        help='GeoJSON/GPKG of reference polygons.')
    parser.add_argument('--outdir', default='.',
                        help='Directory for the eval report and layers.')
    parser.add_argument('--tag', default=None,
                        help='Output basename. Defaults to the footprints '
                             'filename.')
    parser.add_argument('--metric-crs', default=None,
                        help='CRS for areas/distances, e.g. EPSG:5070. '
                             'Defaults to the estimated UTM zone.')
    args = parser.parse_args()
    main(**vars(args))
