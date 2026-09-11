#!/usr/bin/env python3
"""Fetch, assemble and dedupe patch embeddings for an area of interest.

Turns a STAC catalogue of per-MGRS-tile embeddings into the single parquet that
build_footprints.py consumes, and optionally the DuckDB pair for large areas:

    1. Search the catalogue for tiles covering the area.
    2. Download each tile's embeddings asset (resumable; existing files are kept).
    3. Quantize to uint8, name the feature columns, assign tile_id.
    4. Drop patches duplicated where MGRS tiles overlap.
    5. Write one deduped parquet, and optionally build the DuckDB assets.

Which tiles to fetch can be chosen two ways. `--region AOI.geojson` takes every
tile intersecting the area, which suits a region where the targets are widely
spread. `--positives POINTS.geojson` keeps only tiles that actually contain a
known facility, which for a country with large empty tracts is the difference
between downloading a continent and downloading a dozen tiles. The two compose:
the region bounds the search, the positives filter the result.

Example:

    python3 automated/fetch_embeddings.py \
        --region Australia/australia_east.geojson \
        --positives Australia/AUS_for_emission_modeling_20250918.geojson \
        --clip Australia/australia_east.geojson \
        --start 2025-01-01 --end 2025-12-31 \
        --name australia_east --outdir Australia --build-duckdb

Ask for the calendar year you want. Yearly embeddings run Jan 1 to Jan 1, so
that window also touches the previous vintage and the catalogue returns both;
the one the window genuinely overlaps is kept and the rest dropped, and the
period the catalogue reports is what goes into the output filename.

Memory is the constraint that shaped this: the assembly never holds more than
one tile at a time, and deduplication works on centroids alone, so peak usage is
set by the largest single tile rather than by the size of the region. The old
notebook path concatenated every tile into one GeoDataFrame first, which is what
made 64 GB machines swap.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import shapely

from footprints.util import log, warn

STAC_URL = 'https://stac.earthgenome.org'
COLLECTION = 'ssl4eo_yearly_embeddings'
FEATURE_PREFIX = 'vit-dino-patch16'


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def available_gb():
    """Free RAM plus free swap, in GB -- what a big allocation can actually use."""
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return (vm.available + sw.free) / 1e9


def check_headroom(need_gb, what, limit_gb=None):
    """Stop before an allocation that the machine cannot hold.

    Aborting here costs the user a re-run with different arguments; discovering
    it by swapping costs an afternoon, which is the failure this guards.
    """
    have = limit_gb if limit_gb is not None else available_gb()
    log(f'  {what}: needs ~{need_gb:.1f} GB, {have:.1f} GB available')
    if need_gb > have:
        raise SystemExit(
            f'Not enough memory for {what}: needs ~{need_gb:.1f} GB, '
            f'{have:.1f} GB available (RAM + swap). Fetch fewer tiles '
            f'(--positives narrows the set sharply), or raise --memory-limit-gb '
            f'if you know better than this estimate.')
    if need_gb > 0.7 * have:
        warn(f'{what} will use ~{100 * need_gb / have:.0f}% of available '
             'memory; expect swapping.')


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def utc_isoformat(date_str):
    """A plain YYYY-MM-DD as the UTC instant the catalogue expects."""
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return dt.isoformat().replace('+00:00', 'Z')


def search_items(region_geom, start, end, stac_url, collection, max_items):
    """STAC items whose footprint covers the region over the date range.

    The date range selects which embedding year to fetch, and is not the period
    the embeddings cover: a search in February returns that year's annual
    embeddings. See --period.
    """
    import pystac_client
    log(f'Searching {stac_url} [{collection}] {start}..{end}')
    catalog = pystac_client.Client.open(stac_url)
    # Queried by bounding box, then filtered locally against the real outline.
    # A GADM boundary has thousands of vertices, and asking the server to
    # intersect against it takes minutes where a bbox takes a second.
    results = catalog.search(
        bbox=shapely.bounds(region_geom).tolist(),
        datetime=f'{utc_isoformat(start)}/{utc_isoformat(end)}',
        collections=[collection],
        max_items=max_items)
    items = list(results.items())
    if items:
        tree = shapely.STRtree([shapely.geometry.shape(it.geometry)
                                for it in items])
        hits = set(tree.query(region_geom, predicate='intersects').tolist())
        dropped = len(items) - len(hits)
        items = [it for i, it in enumerate(items) if i in hits]
        if dropped:
            log(f'  {dropped} tiles in the bounding box but outside the region')
    if len(items) >= max_items:
        raise SystemExit(
            f'Search hit --max-items ({max_items}); the result is truncated and '
            'would silently miss tiles. Raise it or narrow the search.')
    log(f'  {len(items)} tiles match')
    return items


def item_period(item):
    """The embedding period an item covers, as YYYYMMDD-YYYYMMDD.

    Taken from the item id, which is <MGRS>_<start>_<end>, falling back to the
    start/end datetime properties. These items carry no `datetime`, only a
    range, so the period is not something the search window reveals.
    """
    parts = item.id.split('_')
    if len(parts) >= 3:
        try:
            a, b = parts[-2], parts[-1]
            return (f"{a.replace('-', '')}-{b.replace('-', '')}")
        except Exception:
            pass
    p = item.properties
    a, b = p.get('start_datetime', ''), p.get('end_datetime', '')
    return f"{a[:10].replace('-', '')}-{b[:10].replace('-', '')}"


def choose_period(items, start, end):
    """Keep one embedding vintage, the one the requested window actually asks for.

    Yearly embeddings are dated Jan 1 to Jan 1, so a search for a calendar year
    also touches the end of the previous year's period and the catalogue returns
    both. Mixing them would put two patches at every location, and deduplication
    being geometric would keep one of each pair at random, with nothing
    downstream showing the year had been shuffled.

    Rather than making the caller hand-tune dates around that, the period with
    the greatest true overlap with the requested window wins. A window touching
    the previous period only at its boundary gives zero overlap, so the natural
    request -- 1 January to 31 December -- resolves to the year asked for.
    """
    if not items:
        return None, items
    want_a = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    want_b = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    def overlap_days(period):
        a, b = period.split('-')
        pa = datetime.strptime(a, '%Y%m%d').replace(tzinfo=timezone.utc)
        pb = datetime.strptime(b, '%Y%m%d').replace(tzinfo=timezone.utc)
        return max(0.0, (min(want_b, pb) - max(want_a, pa)).total_seconds()) / 86400

    by_period = {}
    for it in items:
        by_period.setdefault(item_period(it), []).append(it)
    if len(by_period) == 1:
        return next(iter(by_period)), items

    ranked = sorted(by_period, key=overlap_days, reverse=True)
    best, runner = ranked[0], ranked[1]
    if overlap_days(runner) > 0.25 * max(overlap_days(best), 1e-9):
        raise SystemExit(
            f'The window {start}..{end} straddles two embedding periods '
            f'({best} and {runner}) too evenly to choose between them. '
            'Narrow it to the year you want.')
    dropped = sum(len(v) for k, v in by_period.items() if k != best)
    log(f'  {len(by_period)} periods returned; keeping {best} '
        f'({overlap_days(best):.0f} days of overlap), dropping {dropped} tiles '
        f'from {", ".join(k for k in ranked[1:])}')
    return best, by_period[best]


def filter_to_positives(items, positives_path, buffer_km):
    """Keep only tiles a known positive lands on.

    Filtering here rather than in the STAC query keeps the request small and the
    rule explicit: the catalogue returns candidate tiles for the region, and a
    tile survives only if a facility sits on it.
    """
    pts = gpd.read_file(positives_path).to_crs('EPSG:4326')
    log(f'Filtering {len(items)} tiles against {len(pts):,} positives'
        + (f', buffered {buffer_km} km' if buffer_km else ''))
    probe = pts.geometry.to_numpy()
    if buffer_km:
        metric = pts.to_crs(pts.estimate_utm_crs())
        probe = gpd.GeoSeries(
            shapely.buffer(metric.geometry.to_numpy(), buffer_km * 1000),
            crs=metric.crs).to_crs('EPSG:4326').to_numpy()
    tree = shapely.STRtree(probe)
    kept = [it for it in items
            if len(tree.query(shapely.geometry.shape(it.geometry),
                              predicate='intersects'))]
    log(f'  {len(kept)} of {len(items)} tiles contain a positive')
    if not kept:
        raise SystemExit(
            'No tile contains a positive. Check that the positives and the '
            'region are the same area, or drop --positives.')
    return kept


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download(items, dest, asset='embeddings'):
    """Fetch each tile's asset, skipping any already on disk."""
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    paths, fetched = [], 0
    for i, item in enumerate(items, 1):
        url = item.assets[asset].href
        path = dest / Path(url).name
        paths.append(path)
        if path.exists() and path.stat().st_size > 0:
            continue
        log(f'  [{i}/{len(items)}] {path.name}')
        tmp = path.with_suffix(path.suffix + '.part')
        with requests.get(url, stream=True) as r:
            if r.status_code >= 300:
                raise SystemExit(f'{url} returned HTTP {r.status_code}')
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        # Rename only once complete, so an interrupted run resumes cleanly
        # instead of leaving a truncated file that looks downloaded.
        tmp.rename(path)
        fetched += 1
    log(f'  {fetched} downloaded, {len(paths) - fetched} already present')
    return paths


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #

def quantize(values, lower=-5.0, upper=5.0):
    """Clip to [lower, upper] and scale onto uint8, as the published tiles are."""
    clipped = np.clip(values, lower, upper)
    return ((clipped - lower) / (upper - lower) * 255).astype(np.uint8)


def tile_centroids(path):
    """Centroid coordinates of one tile, as rounded integer keys.

    Only the geometry column is read. Keys are integers rather than floats so
    that equality is exact; duplicated patches come from the same source grid,
    so a 1e-7 degree grid (about 1 cm) separates distinct patches safely.
    """
    geom = gpd.read_parquet(path, columns=['geometry']).geometry
    c = shapely.centroid(geom.to_numpy())
    return (np.round(shapely.get_x(c) * 1e7).astype('i8'),
            np.round(shapely.get_y(c) * 1e7).astype('i8'))


def plan_dedupe(paths, clip_geom=None):
    """Decide, across all tiles, which rows survive dedup and the AOI clip.

    MGRS tiles overlap, so the same patch appears in more than one download.
    Working on centroids alone keeps this to about 16 bytes a row, so the whole
    region's key set fits in memory even when the embeddings do not.

    The clip rides along here because the centroids are already in hand: an MGRS
    tile reaches well past the area of interest, and those patches are carried
    through every later step for nothing.
    """
    log('Planning deduplication from centroids...')
    xs, ys, owner = [], [], []
    for i, p in enumerate(paths):
        x, y = tile_centroids(p)
        xs.append(x)
        ys.append(y)
        owner.append(np.full(len(x), i, dtype='i4'))
        log(f'  [{i + 1}/{len(paths)}] {p.name}: {len(x):,} patches')
    x, y = np.concatenate(xs), np.concatenate(ys)
    owner = np.concatenate(owner)
    total = len(x)

    keep = ~pd.DataFrame({'x': x, 'y': y}).duplicated().to_numpy()
    log(f'  {total:,} patches, {int((~keep).sum()):,} duplicated at tile '
        f'overlaps')
    if clip_geom is not None:
        inside = shapely.contains_xy(clip_geom, x / 1e7, y / 1e7)
        log(f'  {int((keep & ~inside).sum()):,} outside the clip boundary')
        keep &= inside
    log(f'  {int(keep.sum()):,} patches kept')
    if not keep.any():
        raise SystemExit('Nothing survived the clip; check --clip covers the '
                         'tiles that were fetched.')
    return [keep[owner == i] for i in range(len(paths))], int(keep.sum()), total


def write_shards(paths, keep_masks, n_kept, shard_dir, quantized=True):
    """One deduped, quantized GeoParquet shard per input tile.

    Shards rather than one streamed writer because geopandas owns the GeoParquet
    metadata: handing pyarrow a table built by hand would write the geometry as
    opaque binary and the result would not load back as spatial. They also give
    build_duck_assets.py bounded inputs, since it reads each path whole.

    tile_id is the row number across the whole region, zero-padded to a fixed
    width so every id is the same length and ids sort lexically.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(max(0, n_kept - 1)))
    shards, next_id = [], 0
    for i, (path, keep) in enumerate(zip(paths, keep_masks), 1):
        gdf = gpd.read_parquet(path)
        gdf = gdf[keep]
        if gdf.empty:
            log(f'  [{i}/{len(paths)}] {path.name}: nothing new')
            continue
        feats = np.stack(gdf['embedding'].to_numpy())
        if quantized:
            feats = quantize(feats)
        cols = {f'{FEATURE_PREFIX}_{k}': feats[:, k]
                for k in range(feats.shape[1])}
        cols['tile_id'] = [f'{n:0{width}d}'
                           for n in range(next_id, next_id + len(gdf))]
        next_id += len(gdf)
        # The CRS is set explicitly: some published tiles carry none, and a
        # missing CRS surfaces much later as a silent reprojection error.
        out = gpd.GeoDataFrame(cols, geometry=gdf.geometry.to_numpy(),
                               crs='EPSG:4326')
        shard = shard_dir / f'shard_{i:04d}.parquet'
        out.to_parquet(shard, index=False)
        shards.append(shard)
        log(f'  [{i}/{len(paths)}] {path.name}: {len(gdf):,} patches '
            f'({next_id:,}/{n_kept:,})')
        del gdf, feats, cols, out
    if next_id != n_kept:
        warn(f'wrote {next_id:,} rows but planned {n_kept:,}; '
             'the inputs may have changed mid-run.')
    return shards, next_id


def concat_shards(shards, out_path):
    """Concatenate shards into one parquet, carrying the GeoParquet metadata.

    The schema is taken from the first shard, so the geo metadata geopandas
    wrote survives into the combined file. One shard is in memory at a time.
    """
    writer = None
    try:
        for i, shard in enumerate(shards, 1):
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            log(f'  [{i}/{len(shards)}] {shard.name}')
            del table
    finally:
        if writer is not None:
            writer.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.download_dir or outdir / 'downloads-tmp')

    period = args.period
    if args.tiles:
        paths = sorted(Path(p) for p in args.tiles)
        log(f'Using {len(paths)} tile parquets given on the command line')
    else:
        if not args.region:
            raise SystemExit('Give --region (or --tiles to skip the download).')
        region = gpd.read_file(args.region).to_crs('EPSG:4326').union_all()
        items = search_items(region, args.start, args.end, args.stac_url,
                             args.collection, args.max_items)
        if args.positives:
            items = filter_to_positives(items, args.positives,
                                        args.positives_buffer_km)
        found, items = choose_period(items, args.start, args.end)
        if period and found and period != found:
            warn(f'--period {period} but the items cover {found}; '
                 'using the value given.')
        period = period or found
        log(f'Embedding period: {period}')
        if args.dry_run:
            for it in items:
                print(it.id)
            log(f'--dry-run: {len(items)} tiles would be fetched')
            return
        log(f'Downloading {len(items)} tiles to {tmp}')
        paths = download(items, tmp)
    if not period:
        raise SystemExit('Could not determine the embedding period; '
                         'pass --period YYYYMMDD-YYYYMMDD.')

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f'{len(missing)} tile files are missing, e.g. '
                         f'{missing[0]}')

    clip_geom = None
    if args.clip:
        clip_geom = gpd.read_file(args.clip).to_crs('EPSG:4326').union_all()
        log(f'Clipping to {args.clip}')
    keep_masks, n_kept, n_total = plan_dedupe(paths, clip_geom)

    # One tile at a time, but the features are float64 on the way in.
    biggest = max(p.stat().st_size for p in paths) / 1e9
    check_headroom(biggest * 12, 'assembling the largest tile',
                   args.memory_limit_gb)

    out_path = outdir / f'{period}_{args.name}-deduped.parquet'
    shard_dir = Path(args.shard_dir or outdir / 'shards-tmp')
    log(f'Assembling shards in {shard_dir}')
    shards, written = write_shards(paths, keep_masks, n_kept, shard_dir,
                                   quantized=not args.no_quantize)

    log(f'Combining {len(shards)} shards into {out_path}')
    concat_shards(shards, out_path)
    size_gb = out_path.stat().st_size / 1e9
    log(f'Wrote {written:,} patches ({size_gb:.2f} GB) to {out_path}')

    if args.build_duckdb:
        script = Path(__file__).resolve().parent.parent / 'scripts' \
            / 'build_duck_assets.py'
        db = outdir / f'{args.name}_embeddings.db'
        centroids = outdir / f'{args.name}_centroids.parquet'
        # Given the shards rather than the combined parquet: the script reads
        # each input whole, so one 6 GB file is a 6 GB allocation while the
        # shards are bounded by the largest tile. Its own tile_id dedup is a
        # no-op here, the ids already being unique across shards.
        log(f'Building DuckDB assets via {script.name}')
        cmd = [sys.executable, str(script), *[str(s) for s in shards],
               '--db_path', str(db), '--centroids_path', str(centroids)]
        rc = subprocess.run(cmd).returncode
        if rc:
            raise SystemExit(f'{script.name} failed with exit code {rc}')
        log(f'Wrote {db} and {centroids}')

    if args.keep_shards:
        log(f'Shards kept in {shard_dir}')
    else:
        for s in shards:
            s.unlink()
        shard_dir.rmdir() if not any(shard_dir.iterdir()) else None
        log('Shards removed (--keep-shards to retain them)')

    log('Done.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group('what to fetch')
    src.add_argument('--region', help='GeoJSON bounding the tile search.')
    src.add_argument('--positives', metavar='POINTS.geojson',
                     help='Keep only tiles containing one of these points. For '
                          'an area with large empty tracts this is the '
                          'difference between a continent and a dozen tiles.')
    src.add_argument('--positives-buffer-km', type=float, default=0.0,
                     help='Also keep tiles within this distance of a positive.')
    src.add_argument('--tiles', nargs='+', metavar='PARQUET',
                     help='Skip search and download; assemble these instead.')
    src.add_argument('--clip', metavar='AOI.geojson',
                     help='Drop patches whose centre falls outside this. An '
                          'MGRS tile reaches well past the area of interest, '
                          'and those patches cost disk, memory and inference '
                          'time in every later step. Use a real AOI boundary, '
                          'not a tight buffer around the positives: negative '
                          'sampling and hard-negative mining both need '
                          'surrounding country to work.')

    cat = p.add_argument_group('catalogue')
    cat.add_argument('--start', default=None,
                     help='Search window start, YYYY-MM-DD. For a calendar '
                          'year, just give 1 January.')
    cat.add_argument('--end', default=None,
                     help='Search window end, YYYY-MM-DD. Yearly embeddings run '
                          'Jan 1 to Jan 1, so a calendar-year window also '
                          'touches the previous vintage; the one the window '
                          'genuinely overlaps is the one kept, and its real '
                          'period goes into the filename.')
    cat.add_argument('--stac-url', default=STAC_URL)
    cat.add_argument('--collection', default=COLLECTION,
                     help=f'STAC collection (default {COLLECTION}). Vintages '
                          'before 2024 are in sentinel2-embeddings.')
    cat.add_argument('--max-items', type=int, default=2000,
                     help='Refuse a truncated search above this.')

    out = p.add_argument_group('output')
    out.add_argument('--name', required=True,
                     help='Region name used in the output filenames.')
    out.add_argument('--period', default=None, metavar='YYYYMMDD-YYYYMMDD',
                     help='Period the embeddings cover, for the filename. '
                          'Derived from the item ids when omitted; this is not '
                          'the search window, since a February search returns '
                          'that whole year\'s embeddings.')
    out.add_argument('--outdir', default='.')
    out.add_argument('--download-dir', default=None,
                     help='Where tiles are cached (default <outdir>/downloads-tmp).')
    out.add_argument('--shard-dir', default=None,
                     help='Where per-tile shards are staged (default '
                          '<outdir>/shards-tmp). They are the DuckDB build\'s '
                          'input and are removed unless --keep-shards.')
    out.add_argument('--keep-shards', action='store_true',
                     help='Keep the per-tile shards after assembly.')
    out.add_argument('--build-duckdb', action='store_true',
                     help='Also build the DuckDB + centroids pair, for AOIs too '
                          'large to hold as one parquet.')
    out.add_argument('--no-quantize', action='store_true',
                     help='Keep float features instead of uint8. Quadruples the '
                          'output size; build_footprints expects uint8.')
    out.add_argument('--memory-limit-gb', type=float, default=None,
                     help='Override the detected RAM+swap for the headroom '
                          'check.')
    out.add_argument('--dry-run', action='store_true',
                     help='List the tiles that would be fetched, then stop.')

    args = p.parse_args(argv)
    if not args.tiles and not (args.start and args.end):
        p.error('--start and --end are required unless --tiles is given')
    if args.tiles and not args.period:
        p.error('--period is required with --tiles (no items to derive it from)')
    return args


if __name__ == '__main__':
    main(parse_args())
