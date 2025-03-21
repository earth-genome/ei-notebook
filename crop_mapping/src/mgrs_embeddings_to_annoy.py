import argparse
import os
import time
import logging
from pathlib import Path

import annoy
import duckdb
import geopandas as gpd
import pystac_client
from utils.geometry import get_mgrs_items_intersecting_roi
from google.cloud import storage
import re
import pandas as pd 
from tqdm import tqdm
from joblib import Parallel, delayed
import importlib
from typing import Dict, List, Optional
from utils.geometry import get_crs_from_tile
from utils.models import MGRSTileGrid
from data_loader.utils.stac import CLIENT
from demeter.crop_mapping.src.crop_utils import download_wgs84_tiles

def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def process_tile_centroids(local_tile_dir: str, output_dir: str) -> None:
    """
    Process tile centroids from parquet files and save to a single parquet file.

    Args:
        local_tile_dir: Directory containing tile parquet files
        output_dir: Directory to save the output centroid file

    Raises:
        ValueError: If no parquet files found in local_tile_dir
    """
    output_path = Path(output_dir) / 'centroid_gdf.parquet'
    if output_path.exists():
        logging.info(f"Centroid file already exists at {output_path}. Skipping...")
        return

    tiles_gdf = []
    parquet_files = sorted(Path(local_tile_dir).glob('*.parquet'))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {local_tile_dir}")
        
    for parquet_file in parquet_files:
        gdf = gpd.read_parquet(parquet_file)
        gdf['geometry'] = gdf.centroid
        gdf = gdf.to_crs(epsg=4326)
        tiles_gdf.append(gdf)

    centroid_gdf = gpd.GeoDataFrame(pd.concat(tiles_gdf).reset_index(drop=True))
    centroid_gdf.to_parquet(output_path)
    logging.info(f"Tile centroids saved to: {output_path}")


def build_annoy_index(local_embedding_dir: str, n_trees: int, output_path: str) -> tuple[annoy.AnnoyIndex, int]:
    """
    Build an Annoy index from embedding files.

    Args:
        local_embedding_dir: Directory containing embedding parquet files
        n_trees: Number of trees to use in the Annoy index
        output_path: Path to save the Annoy index

    Returns:
        tuple: (Annoy index object or None if already exists, vector dimension)

    Raises:
        ValueError: If no embedding files found in local_embedding_dir
    """
    if Path(output_path).exists():
        logging.info(f"Annoy index already exists at {output_path}. Skipping...")
        # Still need to return vec_dim for UI config
        sample_df = pd.read_parquet(sorted(Path(local_embedding_dir).glob('*.parquet'))[0])
        return None, len(sample_df.drop(columns=['geometry', 'tile_id'], errors='ignore').columns)

    sorted_paths = sorted(Path(local_embedding_dir).glob('*.parquet'))
    if not sorted_paths:
        raise ValueError(f"No embedding files found in {local_embedding_dir}")

    sample_df = pd.read_parquet(sorted_paths[0])
    vec_dim = len(sample_df.drop(columns=['geometry', 'tile_id'], errors='ignore').columns)
    annoy_index = annoy.AnnoyIndex(vec_dim, 'angular')
    
    current_index = 0
    for path in tqdm(sorted_paths, desc="Adding Items to Annoy index"):
        logging.info(f"Indexing file: {path}")
        df = pd.read_parquet(path)
        vecs = df.drop(columns=['geometry', 'tile_id'], errors='ignore').values
        for v in vecs:
            annoy_index.add_item(current_index, v)
            current_index += 1
    
    start_time = time.time()
    annoy_index.build(n_trees, n_jobs=-1)
    build_time = time.time() - start_time
    logging.info(f"Building index took {build_time:.2f} seconds")
    
    return annoy_index, vec_dim


def create_duckdb_database(output_db_path: str, sorted_embedding_paths: List[str]) -> None:
    """
    Create a DuckDB database from embedding parquet files.

    Args:
        output_db_path: Path to save the DuckDB database
        sorted_embedding_paths: List of paths to embedding parquet files
    """
    if Path(output_db_path).exists():
        logging.info(f"DuckDB database already exists at {output_db_path}. Skipping...")
        return

    start_time = time.time()
    conn = duckdb.connect(output_db_path)
    try:
        sql_embedding_str = ",".join([f"'{f}'" for f in sorted_embedding_paths])
        conn.execute(f"""
            CREATE TABLE embeddings AS 
            SELECT *, ROW_NUMBER() OVER () - 1 as row_number 
            FROM read_parquet([{sql_embedding_str}])
        """)
        conn.execute("CREATE INDEX row_number_idx ON embeddings (row_number)")
        conn.execute("CREATE INDEX tile_id_idx ON embeddings (tile_id)")
        conn.commit()
    finally:
        conn.close()
    
    db_time = time.time() - start_time
    logging.info(f"Creating DuckDB database took {db_time:.2f} seconds")


def download_embedding_file_for_mgrs_id(
        mgrs_id: str,
        bucket_name: str,
        folder_path: str,
        local_embedding_dir: str) -> Optional[str]:
    """
    Get the Parquet file for a given MGRS ID.

    Args:
        mgrs_id: MGRS tile ID
        bucket_name: GCS bucket name
        folder_path: Folder path in the bucket
        local_embedding_dir: Local directory to store downloaded embeddings

    Returns:
        Local path to downloaded parquet file or None if no matching file found
    """
    storage_client = storage.Client(project='earthindex')
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    def extract_mgrs_id(filename: str) -> Optional[str]:
        match = re.search(r'(\d{2}[A-Z]{3})', filename)
        return match.group(1) if match else None

    for blob in blobs:
        if extract_mgrs_id(blob.name) == mgrs_id:
            local_path = f"{local_embedding_dir}/{os.path.basename(blob.name)}"
            if os.path.exists(local_path):
                logging.info(f"Embedding file {local_path} already exists. Skipping download...")
                return local_path
            logging.info(f"Downloading embedding file {blob.name} to {local_path}...")
            blob.download_to_filename(local_path)
            return local_path

    logging.warning(f"No matching embedding file found for MGRS ID: {mgrs_id}")
    return None

def main(args: argparse.Namespace) -> None:
    """
    Main function to process MGRS embeddings.

    Args:
        args: Command line arguments containing:
            - roi_geojson: Path to ROI GeoJSON file
            - image_config: Image configuration module path
            - embeddings_config: Embeddings configuration module path
            - n_jobs: Number of parallel jobs
            - annoy_output_file: Output Annoy index file path
            - n_trees: Number of trees for Annoy index
            - ui_config_output: Output path for UI config JSON file
            - output_dir: Base output directory for tiles and embeddings
    """
    setup_logging()
    
    image_config = importlib.import_module(args.image_config).config
    embeddings_config = importlib.import_module(args.embeddings_config).config

    stac_config = image_config['stac']
    tiling_config = image_config['tiling']
    bucket_name = embeddings_config['output_dir'].split('/')[2]
    folder_path = '/'.join(embeddings_config['output_dir'].split('/')[3:])

    # Create output directories
    local_tile_dir = os.path.join(args.output_dir, 'tiles')
    local_embedding_dir = os.path.join(args.output_dir, 'embeddings')
    os.makedirs(local_tile_dir, exist_ok=True)
    os.makedirs(local_embedding_dir, exist_ok=True)
    output_db_path = os.path.join(args.output_dir, 'embeddings.db')
    output_annoy_path = os.path.join(args.output_dir, 'embeddings.ann')

    client = pystac_client.Client.open("https://stac.earthindex.dev")
    intersecting_mgrs_items = get_mgrs_items_intersecting_roi(client, args.roi_geojson)
    intersecting_mgrs_ids = [x.id for x in intersecting_mgrs_items]

    try:
        # Download tiles and embeddings
        logging.info("Downloading tiles")
        Parallel(n_jobs=args.n_jobs, verbose=30)(
            delayed(download_wgs84_tiles)(
                mgrs_id, stac_config, tiling_config, local_tile_dir
            ) for mgrs_id in tqdm(intersecting_mgrs_ids, desc="Downloading tiles")
        )

        logging.info("Downloading embeddings")
        Parallel(n_jobs=args.n_jobs, verbose=30)(
            delayed(download_embedding_file_for_mgrs_id)(
                mgrs_id, bucket_name, folder_path, local_embedding_dir
            ) for mgrs_id in tqdm(intersecting_mgrs_ids, desc="Downloading embeddings")
        )

        # Process centroids
        logging.info("Processing tile centroids")
        process_tile_centroids(local_tile_dir, args.output_dir)

        # Build Annoy index
        logging.info("Building Annoy index")
        annoy_index, vec_dim = build_annoy_index(local_embedding_dir, args.n_trees, output_annoy_path)
        if annoy_index is not None:
            annoy_index.save(output_annoy_path)
            logging.info(f"Annoy index saved to: {output_annoy_path}")

        # Create DuckDB database
        logging.info("Creating DuckDB database")
        sorted_paths = sorted([str(p) for p in Path(local_embedding_dir).glob('*.parquet')])
        create_duckdb_database(output_db_path, sorted_paths)

        # Generate UI config
        ui_config = {
            "local_dir": args.output_dir,
            "mgrs_ids": intersecting_mgrs_ids,
            "index_dim": vec_dim,
            "start_date": image_config['imagery']['start_datetime'],
            "end_date": image_config['imagery']['end_datetime'],
            "imagery": image_config['imagery']['gs_base']
        }

        if args.ui_config_output:
            import json
            with open(args.ui_config_output, 'w') as f:
                json.dump(ui_config, f, indent=4)
            logging.info(f"UI config saved to: {args.ui_config_output}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MGRS embeddings")
    parser.add_argument("--roi_geojson", type=str, required=True, help="Path to ROI GeoJSON file")
    parser.add_argument("--image_config", type=str, required=True, help="Image configuration module path")
    parser.add_argument("--embeddings_config", type=str, required=True, help="Embeddings configuration module path")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--n_trees", type=int, default=10, help="Number of trees for Annoy index")
    parser.add_argument("--ui_config_output", type=str, help="Output path for UI config JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for tiles and embeddings")
    args = parser.parse_args()

    main(args)
