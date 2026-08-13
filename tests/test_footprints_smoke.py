#!/usr/bin/env python3
"""End-to-end smoke test for automated/build_footprints.py.

Run it after changing defaults.

    python3 tests/test_footprints_smoke.py         # ~1 minute
    python3 tests/test_footprints_smoke.py --keep  # keep the working directory

Builds a small synthetic AOI whose structure matches the real embedding data --
50%-overlapping ~320 m patches on a 160 m stride, 384 uint8 features -- with
planted facilities and deliberately confusable decoys so that hard-negative
mining has something to find. Then exercises both storage backends and asserts
the invariants that have actually broken during development:

  * the pipeline completes and writes every expected output
  * all trained positives are recovered, and footprints are plausibly sized
  * the run-assessment block passes on known-good data
  * hard-negative rounds run (the fixture must not be so clean it early-stops)
  * the label-quality gate finds nothing to exclude in clean data
  * the DuckDB and parquet backends agree exactly
  * --select-round emits the round asked for
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
BUILD = os.path.join(ROOT, 'automated', 'build_footprints.py')
# The repo's asset builder, shared with the interactive workflow.
DUCK = os.path.join(ROOT, 'scripts', 'build_duck_assets.py')

FIXTURE = r'''
import numpy as np, geopandas as gpd, shapely, pyarrow as pa

rng = np.random.default_rng(0)
lon0, lat0 = -100.0, 38.0
dlon = 160 / (111320 * np.cos(np.radians(lat0)))
dlat = 160 / 110570
nx = ny = 200
ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
lon, lat = lon0 + ix.ravel() * dlon, lat0 + iy.ravel() * dlat
n = len(lon)

n_fac = 25
fac_i = rng.choice(nx - 6, n_fac, replace=False) + 3
fac_j = rng.choice(ny - 6, n_fac, replace=False) + 3
extents = rng.integers(0, 3, n_fac)
pos_mask = np.zeros((nx, ny), bool)
for a, b, e in zip(fac_i, fac_j, extents):
    pos_mask[a - e:a + e + 1, b - e:b + e + 1] = True
pos_mask = pos_mask.ravel()

X = rng.integers(60, 90, size=(n, 384)).astype(np.uint8)
X[pos_mask, :8] = rng.integers(200, 255, size=(pos_mask.sum(), 8))
# Confusable decoys away from the facilities, so the model produces false
# positives and the hard-negative rounds have something to mine. Without these
# the fixture is separable, mining finds nothing and the multi-round path is
# never exercised.
decoy = rng.choice(np.flatnonzero(~pos_mask), 250, replace=False)
X[np.ix_(decoy, np.arange(8))] = rng.integers(195, 250, size=(len(decoy), 8))

geom = shapely.box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
tile_id = np.array([f'14SXX_{i}_{j}' for i, j in zip(ix.ravel(), iy.ravel())],
                   dtype=object)
tbl = pa.table({**{f'vit-dino-patch16_{k}': X[:, k] for k in range(384)},
                'tile_id': pa.array(tile_id)})
gpd.GeoDataFrame(tbl.to_pandas(), geometry=list(geom), crs='EPSG:4326'
                 ).to_parquet('synth_embeddings.parquet', index=False)
gpd.GeoDataFrame(geometry=[shapely.Point(lon0 + a * dlon, lat0 + b * dlat)
                           for a, b in zip(fac_i, fac_j)], crs='EPSG:4326'
                 ).to_file('synth_positives.geojson', driver='GeoJSON')
polys = []
for a, b, e in zip(fac_i, fac_j, extents):
    gx, gy = np.meshgrid(lon0 + np.arange(a - e, a + e + 1) * dlon,
                         lat0 + np.arange(b - e, b + e + 1) * dlat,
                         indexing='ij')
    polys.append(shapely.union_all(shapely.box(
        gx.ravel() - dlon, gy.ravel() - dlat,
        gx.ravel() + dlon, gy.ravel() + dlat)))
gpd.GeoDataFrame(geometry=polys, crs='EPSG:4326').to_file(
    'synth_reference.geojson', driver='GeoJSON')
gpd.GeoDataFrame(geometry=[shapely.box(lon.min() - dlon, lat.min() - dlat,
                                       lon.max() + dlon, lat.max() + dlat)],
                 crs='EPSG:4326').to_file('synth_boundary.geojson',
                                          driver='GeoJSON')
print(f'fixture: {n} patches, {n_fac} facilities, {len(decoy)} decoys')
'''

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  [{detail}]' if detail
                                                     else ''))


def run(cmd, cwd, log):
    """Run a command, tee-ing output to a log file. Returns (rc, text)."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    with open(os.path.join(cwd, log), 'w') as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


def field(path, label, pattern=r'([\d,.]+)'):
    for line in open(path):
        if label in line:
            m = re.search(pattern, line.split(label)[1])
            if m:
                return m.group(1).replace(',', '')
    return None


def main(keep=False):
    work = tempfile.mkdtemp(prefix='footprints_smoke_')
    print(f'working directory: {work}\n')
    try:
        with open(os.path.join(work, 'fixture.py'), 'w') as f:
            f.write(FIXTURE)
        rc, out = run([sys.executable, 'fixture.py'], work, 'fixture.log')
        summary = next((l for l in out.splitlines()
                        if l.startswith('fixture:')), '')
        check('fixture builds', rc == 0, summary)
        if rc:
            return

        common = ['--positives', 'synth_positives.geojson',
                  '--reference-polygons', 'synth_reference.geojson',
                  '--boundary', 'synth_boundary.geojson',
                  '--outdir', 'out', '--batch-size', '5000']

        # --- parquet backend, default settings --------------------------------
        rc, out = run([sys.executable, BUILD, *common,
                       '--embeddings', 'synth_embeddings.parquet',
                       '--tag', 'pq', '--save-round-outputs'],
                      work, 'pq.log')
        check('parquet run completes', rc == 0,
              '' if rc == 0 else out.strip().splitlines()[-1])
        if rc:
            return

        cfgs = glob.glob(os.path.join(work, 'out', 'pq_*_config.txt'))
        cfg = cfgs[0]
        stats = cfg.replace('_config.txt', '_stats.txt')
        for suffix in ('_footprints.geojson', '_patches_filtered.geojson',
                       '_detections_raw.geojson', '_labels.geojson',
                       '_model.joblib', '_stats.txt', '_eval.txt',
                       '_fbeta_F1.png', '_pr.png'):
            check(f'writes {suffix.lstrip("_")}',
                  os.path.exists(cfg.replace('_config.txt', suffix)))

        text = open(stats).read()
        check('assessment passes on clean data',
              'RUN ASSESSMENT: all' in text,
              re.search(r'RUN ASSESSMENT: [^\n]*', text).group(0))

        # recall(trained) must never exceed 100% -- it did once, when a new way
        # of removing positives was not reflected in the numerator.
        bad = [l for l in text.splitlines()
               if re.match(r'^\s*\*?\s+\d+\s+[\d,]+\s+1[0-9][0-9]\.\d%', l)]
        check('no recall above 100% in the per-round table', not bad,
              bad[0].strip() if bad else '')

        rec = field(stats, 'matched by a retained polygon:')
        check('all positives recovered', rec == '25', f'{rec}/25')

        gate = field(cfg, 'Positives excluded and refitted:')
        check('label gate excludes nothing in clean data', gate == '0',
              f'{gate} excluded')

        rounds = re.findall(r'^ ?\*?\s+(\d)\s+\d', text, re.M)
        n_rounds = len(glob.glob(os.path.join(
            work, 'out', 'pq_*_round*_footprints.geojson')))
        check('hard-negative rounds actually run', n_rounds >= 2,
              f'{n_rounds} rounds emitted')

        import geopandas as gpd
        fp = gpd.read_file(cfg.replace('_config.txt', '_footprints.geojson'))
        check('footprint count is plausible', 20 <= len(fp) <= 40,
              f'{len(fp)} polygons')
        check('no absurd merged polygon', fp.area_ha.max() < 500,
              f'max {fp.area_ha.max():.0f} ha')

        # --- --select-round ---------------------------------------------------
        rc, out = run([sys.executable, BUILD, *common,
                       '--embeddings', 'synth_embeddings.parquet',
                       '--tag', 'sel0', '--select-round', '0'],
                      work, 'sel0.log')
        check('--select-round 0 runs', rc == 0)
        if rc == 0:
            c0 = glob.glob(os.path.join(work, 'out', 'sel0_*_config.txt'))[0]
            check('--select-round 0 emits round 0',
                  'round 0, which was emitted' in open(
                      c0.replace('_config.txt', '_stats.txt')).read()
                  or field(c0, 'describe round') == '0')

        # --- duckdb backend, must agree exactly -------------------------------
        rc, out = run([sys.executable, DUCK, 'synth_embeddings.parquet'],
                      work, 'duck_build.log')
        check('duckdb assets build', rc == 0)
        if rc == 0:
            rc, out = run([sys.executable, BUILD, *common,
                           '--centroids', 'centroids.parquet',
                           '--duckdb', 'embeddings.db', '--tag', 'db'],
                          work, 'db.log')
            check('duckdb run completes', rc == 0,
                  '' if rc == 0 else out.strip().splitlines()[-1])
            if rc == 0:
                import shapely
                dbcfg = glob.glob(os.path.join(work, 'out',
                                               'db_*_config.txt'))[0]
                a = gpd.read_file(cfg.replace('_config.txt',
                                              '_footprints.geojson'))
                b = gpd.read_file(dbcfg.replace('_config.txt',
                                                '_footprints.geojson'))
                crs = a.estimate_utm_crs()
                ua = shapely.union_all(a.to_crs(crs).geometry.to_numpy())
                ub = shapely.union_all(b.to_crs(crs).geometry.to_numpy())
                diff = shapely.area(
                    shapely.symmetric_difference(ua, ub)) / 1e4
                check('backends agree exactly', len(a) == len(b) and diff < 0.01,
                      f'{len(a)} vs {len(b)} polygons, sym-diff {diff:.4f} ha')
    finally:
        print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
        if FAIL:
            print('failed: ' + ', '.join(FAIL))
        if keep:
            print(f'kept: {work}')
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--keep', action='store_true',
                    help='Leave the working directory in place for inspection.')
    sys.exit(main(**vars(ap.parse_args())))
