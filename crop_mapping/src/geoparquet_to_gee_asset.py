import importlib
import geopandas as gpd
import os
import subprocess
from pathlib import Path
from google.cloud import storage
import tempfile
import logging
import argparse
import pystac_client
from joblib import Parallel, delayed
from tqdm import tqdm

from demeter.crop_mapping.src.crop_utils import download_wgs84_tiles
from utils.geometry import get_mgrs_items_intersecting_roi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_and_upload_to_gee(local_parquet_dir, shapefile_dir, gee_project, gee_folder):
    """
    Convert local parquet files to shapefiles, upload to GCS, and create GEE assets.
    
    Parameters:
    -----------
    local_parquet_dir : str
        Local directory containing parquet files
    shapefile_dir : str
        GCS path to upload shapefiles to (e.g. gs://bucket/path)
    gee_project : str
        GEE project ID
    gee_folder : str
        GEE folder to upload assets to
    """
    storage_client = storage.Client()
    bucket_name = shapefile_dir.replace('gs://', '').split('/')[0]
    prefix = '/'.join(shapefile_dir.replace('gs://', '').split('/')[1:])
    bucket = storage_client.bucket(bucket_name)

    try:
        # Get list of local parquet files
        parquet_files = list(Path(local_parquet_dir).glob('*.parquet'))
        
        for parquet_file in tqdm(parquet_files, desc="Converting parquet files to shapefiles"):
            try:
                # Read geoparquet file
                gdf = gpd.read_parquet(parquet_file)
                
                # Generate output shapefile path
                base_name = os.path.splitext(parquet_file.name)[0]
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_shapefile = os.path.join(tmp_dir, f"{base_name}.shp")
                    
                    # Save as shapefile
                    gdf.to_file(local_shapefile)
                    logger.info(f"Converted {parquet_file} to {local_shapefile}")
                    
                    # Upload all shapefile components to GCS
                    for ext in ['.shp', '.shx', '.dbf', '.prj']:
                        local_file = os.path.join(tmp_dir, f"{base_name}{ext}")
                        gcs_path = f"{prefix}/{base_name}{ext}"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(local_file)
                        logger.info(f"Uploaded {local_file} to gs://{bucket_name}/{gcs_path}")
                
                # Generate GEE asset ID
                asset_id = f"projects/{gee_project}/assets/{gee_folder}/{base_name}"
                
                # Upload to GEE using earthengine CLI
                gcs_shapefile = f"{shapefile_dir}/{base_name}.shp"
                command = [
                    'earthengine', 'upload', 'table',
                    '--asset_id', asset_id,
                    gcs_shapefile
                ]
                
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Successfully uploaded {gcs_shapefile} to {asset_id}")
                logger.info(result.stdout)
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Convert geoparquet files from GCS to GEE assets')
    parser.add_argument('--image-config', required=True, help='Image configuration module path')
    parser.add_argument('--embeddings-config', required=True, help='Embeddings configuration module path')
    parser.add_argument('--local-parquet-dir', required=True, help='Local directory containing parquet files')
    parser.add_argument('--shapefile-dir', required=True, help='GCS path to upload shapefiles to (e.g. gs://bucket/path)')
    parser.add_argument('--roi-geojson', required=True, help='Path to ROI GeoJSON file')
    parser.add_argument('--gee-project', default='earthindex', help='GEE project ID')
    parser.add_argument('--gee-folder', default='cocoa-tiles', help='GEE folder to upload assets to')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_config = importlib.import_module(args.image_config).config
    embeddings_config = importlib.import_module(args.embeddings_config).config

    client = pystac_client.Client.open("https://stac.earthindex.dev")
    intersecting_mgrs_items = get_mgrs_items_intersecting_roi(client, args.roi_geojson)
    intersecting_mgrs_ids = [x.id for x in intersecting_mgrs_items]

    os.makedirs(args.local_parquet_dir, exist_ok=True)

    stac_config = image_config['stac']
    tiling_config = image_config['tiling']

    logging.info("Downloading tiles")
    Parallel(n_jobs=-1, verbose=30)(
        delayed(download_wgs84_tiles)(
            mgrs_id, stac_config, tiling_config, args.local_parquet_dir, project_to_wgs84=True
        ) for mgrs_id in tqdm(intersecting_mgrs_ids, desc="Downloading tiles")
    )
    convert_and_upload_to_gee(
        local_parquet_dir=args.local_parquet_dir,
        shapefile_dir=args.shapefile_dir,
        gee_project=args.gee_project,
        gee_folder=args.gee_folder
    )