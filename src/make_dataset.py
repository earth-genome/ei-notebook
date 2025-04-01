import argparse
import geopandas as gpd
import pandas as pd
import duckdb
from pathlib import Path
from typing import Union, Optional

import os


def process_data(
        full_dataset_path: Optional[Union[str, Path]] = None,
        pos_gdf_path: Optional[Union[str, Path]] = None,
        neg_ei_gdf_path: Optional[Union[str, Path]] = None,
        neg_lulc_gdf_path: Optional[Union[str, Path]] = None,
        centroid_gdf_path: Optional[Union[str, Path]] = None,
        embedding_db_path: Union[str, Path] = None,
        output_dir: Union[str, Path] = None,
        region_name: str = None,
        version: str = None
    ) -> None:
    """
    Process and combine positive and negative sample data with embeddings.

    Args:
        full_dataset_path: Optional path to pre-labeled dataset with tile_ids
        pos_gdf_path: Optional path to positive samples GeoParquet file
        neg_ei_gdf_path: Optional path to negative Earth Index samples GeoParquet file 
        neg_lulc_gdf_path: Optional path to negative LULC samples GeoParquet file
        centroid_gdf_path: Optional path to tile centroid GeoParquet file
        embedding_db_path: Path to DuckDB database containing embeddings
        output_dir: Directory to save output files
        region_name: Name of the region being processed
        version: Version string for output filenames

    Returns:
        None. Saves processed datasets to output_dir.
    """
    print(f"Processing {region_name} data")

    if full_dataset_path:
        # Use pre-labeled dataset with tile_ids
        full_gdf = gpd.read_parquet(full_dataset_path)
        if 'label' not in full_gdf.columns:
            raise ValueError("Full dataset must contain a 'label' column")
        if 'tile_id' not in full_gdf.columns:
            raise ValueError("Full dataset must contain a 'tile_id' column")
        if 'class' not in full_gdf.columns:
            full_gdf['class'] = full_gdf['label'].map({0: 'sample_neg', 1: 'sample_pos'})
    else:
        # Process separate positive and negative samples
        if not all([pos_gdf_path, centroid_gdf_path]):
            raise ValueError("Must provide either full_dataset or pos_gdf and centroid_gdf")

       

        pos_gdf = gpd.read_parquet(pos_gdf_path)
        centroid_gdf = gpd.read_parquet(centroid_gdf_path)

        assert 'tile_id' in centroid_gdf.columns, "centroid_gdf must have a 'tile_id' column"

        def add_utm_info(gdf):
            gdf['utm_zone'] = ((gdf.geometry.x + 180) / 6 + 1).astype(int)
            gdf['hemisphere'] = gdf.geometry.y.apply(lambda y: 'N' if y >= 0 else 'S')
            gdf['utm_epsg'] = gdf.apply(lambda x: f"EPSG:{'326' if x.hemisphere == 'N' else '327'}{x.utm_zone:02d}", axis=1)
            return gdf
        
       

        centroid_gdf = add_utm_info(centroid_gdf)
        pos_gdf = add_utm_info(pos_gdf)

    

        neg_samples = []

        if neg_lulc_gdf_path:
            neg_lulc_gdf = gpd.read_parquet(neg_lulc_gdf_path).drop(columns=['remapped'], errors='ignore')
            neg_lulc_gdf = add_utm_info(neg_lulc_gdf)
            
            neg_lulc_with_tiles = []
            for utm_epsg in neg_lulc_gdf.utm_epsg.unique():
                zone_neg = neg_lulc_gdf[neg_lulc_gdf.utm_epsg == utm_epsg].to_crs(utm_epsg)
                zone_cent = centroid_gdf[centroid_gdf.utm_epsg == utm_epsg].to_crs(utm_epsg)
                
                zone_joined = gpd.sjoin_nearest(zone_neg, zone_cent, how='left')
                zone_joined = zone_joined.to_crs('EPSG:4326')
                neg_lulc_with_tiles.append(zone_joined)
            
            neg_lulc_with_tiles = pd.concat(neg_lulc_with_tiles)
            neg_lulc_with_tiles = neg_lulc_with_tiles.drop(columns=['utm_zone', 'hemisphere', 'utm_epsg', 'index_right'], errors='ignore')
            neg_lulc_with_tiles['label'] = 0
            neg_samples.append(neg_lulc_with_tiles)

        if neg_ei_gdf_path:
            neg_ei_gdf = gpd.read_parquet(neg_ei_gdf_path)
            if 'class' not in neg_ei_gdf.columns:
                neg_ei_gdf['class'] = 'ei_neg'
            neg_ei_gdf['label'] = 0
            neg_samples.append(neg_ei_gdf)

        pos_with_tiles = []
        for utm_epsg in pos_gdf.utm_epsg.unique():
            zone_pos = pos_gdf[pos_gdf.utm_epsg == utm_epsg].to_crs(utm_epsg)
            zone_cent = centroid_gdf[centroid_gdf.utm_epsg == utm_epsg].to_crs(utm_epsg)
            
            zone_joined = gpd.sjoin_nearest(zone_pos, zone_cent[['tile_id', 'geometry']], how='left')
            zone_joined = zone_joined.rename(columns={'tile_id_right': 'tile_id'})
            
            zone_joined = zone_joined.to_crs('EPSG:4326')
            pos_with_tiles.append(zone_joined)

        
        pos_with_tiles = pd.concat(pos_with_tiles)
        pos_with_tiles = pos_with_tiles.drop(columns=['utm_zone', 'hemisphere', 'utm_epsg', 'index_right'], errors='ignore')
        if 'class' not in pos_with_tiles.columns:
            pos_with_tiles['class'] = 'ei_pos'
        pos_with_tiles['label'] = 1
        
        if neg_samples:
            full_neg_gdf = pd.concat(neg_samples)
            full_gdf = pd.concat([full_neg_gdf, pos_with_tiles])
            print(f"Number of negatives: {len(full_neg_gdf)}, Number of positives: {len(pos_with_tiles)}")
        else:
            full_gdf = pos_with_tiles
            print(f"No negative samples provided. Number of positives: {len(pos_with_tiles)}")

    print(f"Full GeoDataFrame has {len(full_gdf)} rows")
    full_gdf = full_gdf.drop_duplicates(subset=['tile_id'])
    print(f"Number of negatives: {len(full_gdf[full_gdf.label == 0])}, Number of positives: {len(full_gdf[full_gdf.label == 1])}")
    print(f"Full GeoDataFrame has {len(full_gdf)} rows after dropping duplicates")
    os.makedirs(output_dir, exist_ok=True)
    full_gdf.to_parquet(f"{output_dir}/tile_classifier_dataset_{version}_{region_name}.parquet")
    print(f"Saved {region_name} data to {output_dir}/tile_classifier_dataset_{version}_{region_name}.parquet")

    print(f"Loading embeddings for {region_name}")
    conn = duckdb.connect(f"{embedding_db_path}")
    tile_ids = tuple(full_gdf.tile_id.dropna().unique())
    embeddings_df = conn.execute(f"""
        SELECT *
        FROM embeddings 
        WHERE tile_id IN {tile_ids}
    """).df()
    
    full_df_embeddings = gpd.GeoDataFrame(
        embeddings_df.merge(full_gdf[['tile_id', 'label', 'geometry', 'class']], 
        on='tile_id', 
        how='inner'), 
        geometry='geometry'
    )
    full_df_embeddings = full_df_embeddings.drop(columns=['row_number'])
    full_df_embeddings.to_parquet(
        f"{output_dir}/tile_classifier_dataset_{version}_{region_name}_embeddings.parquet")
    print(
        f"Saved {region_name} embeddings to {output_dir}/tile_classifier_dataset_{version}_{region_name}_embeddings.parquet")


def main() -> None:
    """
    Parse command line arguments and run data processing pipeline.
    
    Either provide a full pre-labeled dataset with tile_ids, or provide separate 
    positive and negative samples to be joined with centroids.
    """
    parser = argparse.ArgumentParser(description='Process GeoDataFrames for CXR10 classifier')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--full-dataset', help='Path to pre-labeled dataset with tile_ids')
    input_group.add_argument('--pos-gdf', help='Path to positive GeoDataFrame parquet file')
    
    parser.add_argument('--neg-ei-gdf', required=False, help='Path to negative EI GeoDataFrame parquet file')
    parser.add_argument('--neg-lulc-gdf', required=False, help='Path to negative LULC GeoDataFrame parquet file')
    parser.add_argument('--centroid-gdf', required=False, help='Path to centroid GeoDataFrame parquet file')
    parser.add_argument('--embedding-db', required=True, help='Path to embedding database')
    parser.add_argument('--output-dir', required=True, help='Directory to save output files')
    parser.add_argument('--region-name', required=True, help='Name of the region')
    parser.add_argument('--version', required=True, help='Version string for output files')
    args = parser.parse_args()

    if args.pos_gdf and not args.centroid_gdf:
        parser.error("--centroid-gdf is required when using --pos-gdf")
    
    process_data(
        args.full_dataset,
        args.pos_gdf,
        args.neg_ei_gdf,
        args.neg_lulc_gdf,
        args.centroid_gdf,
        args.embedding_db,
        args.output_dir,
        args.region_name,
        args.version
    )


if __name__ == "__main__":
    main()