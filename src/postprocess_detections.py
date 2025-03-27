import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import glob
import argparse
import os
from typing import Optional, List


def process_mgrs_tile(mgrs_id: str, tiles_dir: str, detections: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """
    Process a single MGRS tile by finding its geometry file and filtering to detected tiles.

    Args:
        mgrs_id: MGRS tile identifier
        tiles_dir: Directory containing tile geometry files
        detections: GeoDataFrame containing tile detections to filter by

    Returns:
        GeoDataFrame containing tile geometries that match detections, or None if no matching file found
    """
    matching_files = glob.glob(f"{tiles_dir}/*{mgrs_id}*.parquet")
    if not matching_files:
        return None
    
    tile_file = matching_files[0]
    tile_gdf = gpd.read_parquet(tile_file).to_crs(epsg=4326)
    tile_gdf = tile_gdf[tile_gdf['tile_id'].isin(detections['tile_id'])]
    tile_gdf['utm_zone'] = tile_file.split('/')[-1].split('_')[1]
    return tile_gdf


def main(args: argparse.Namespace) -> None:
    """
    Process tile classifier predictions into dissolved polygons with area calculations.

    Args:
        args: Command line arguments containing:
            input_file: Path to input parquet file with predictions
            tiles_dir: Directory containing tile geometry files
            output_dir: Directory to save output files
            prob_threshold: Probability threshold for filtering predictions
    """
    detections = gpd.read_parquet(args.input_file)
    detections = detections[detections.prediction_probability > args.prob_threshold]

    detections['mgrs_id'] = [x[:5] for x in detections['tile_id']]
    
    tile_gdfs = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_mgrs_tile)(mgrs_id, args.tiles_dir, detections)
        for mgrs_id in detections['mgrs_id'].unique()
    )

    tile_gdfs = [gdf for gdf in tile_gdfs if gdf is not None]
    if tile_gdfs:
        detection_tiles = pd.concat(tile_gdfs)
        detections = detections.merge(
            detection_tiles, on='tile_id', suffixes=('', '_tile')).set_geometry('geometry_tile')

    unioned = detections.geometry_tile.union_all()
    exploded_gdf = gpd.GeoDataFrame(geometry=[unioned], crs=detection_tiles.crs).explode(index_parts=True)

    exploded_gdf['centroid'] = exploded_gdf.geometry.centroid
    exploded_gdf['utm_zone'] = exploded_gdf.apply(
        lambda row: int(((row.centroid.x + 180) / 6) % 60) + 1, axis=1)
    exploded_gdf['hemisphere'] = exploded_gdf.apply(
        lambda row: 'N' if row.centroid.y >= 0 else 'S', axis=1)

    def get_area(row: pd.Series) -> float:
        """Calculate area in square meters for a geometry in its local UTM zone."""
        utm_crs = f"EPSG:{'326' if row.hemisphere == 'N' else '327'}{row.utm_zone:02d}"
        return gpd.GeoDataFrame(geometry=[row.geometry], crs='epsg:4326').to_crs(utm_crs).area.iloc[0]

    exploded_gdf['area'] = exploded_gdf.apply(get_area, axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    input_fname = os.path.basename(args.input_file)
    fname_no_ext = os.path.splitext(input_fname)[0]
    output_fname = f"{fname_no_ext}_prob_{args.prob_threshold}_postprocess.parquet"
    output_file = os.path.join(args.output_dir, output_fname)
    gpd.GeoDataFrame(exploded_gdf.drop(columns=['centroid', 'utm_zone', 'hemisphere']), geometry='geometry').to_parquet(
        output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tile classifier predictions')
    parser.add_argument('--input_file', help='Path to input parquet file containing predictions')
    parser.add_argument('--tiles_dir', help='Directory containing tile files')
    parser.add_argument('--output_dir', help='Directory to save output files')
    parser.add_argument('--prob_threshold', type=float, default=0.9,
                       help='Probability threshold for filtering predictions (default: 0.9)')
    
    args = parser.parse_args()
    main(args)