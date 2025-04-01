import argparse
import geopandas as gpd
import pandas as pd
import ee
import geemap
import shapely
import logging
import json
from pathlib import Path
from typing import Dict, Any

from gee import initialize_ee_with_credentials

def filter_samples(samples_gdf: gpd.GeoDataFrame, 
                  pos_gdf: gpd.GeoDataFrame, 
                  buffer_size: float, 
                  utm_crs: str) -> gpd.GeoDataFrame:
    """
    Filter out samples that are too close to positive points.

    Args:
        samples_gdf: GeoDataFrame containing sample points to filter
        pos_gdf: GeoDataFrame containing positive points to buffer around
        buffer_size: Size of buffer in meters to create around positive points
        utm_crs: UTM CRS string to project geometries for accurate buffering

    Returns:
        GeoDataFrame containing filtered samples in WGS84 (EPSG:4326)
    """
    # Project to UTM for accurate buffering
    pos_gdf_utm = pos_gdf.to_crs(utm_crs)
    samples_utm = samples_gdf.to_crs(utm_crs)

    # Create buffer around positive points
    pos_gdf_buffered = pos_gdf_utm.copy()
    pos_gdf_buffered.geometry = pos_gdf_utm.geometry.buffer(buffer_size)
    pos_union = pos_gdf_buffered.geometry.unary_union

    # Remove samples that intersect with buffered positive points
    samples_filtered_utm = samples_utm[~samples_utm.geometry.intersects(pos_union)]

    # Project back to original CRS
    samples_filtered = samples_filtered_utm.to_crs('EPSG:4326')

    logging.info(f"Number of samples after filtering: {len(samples_filtered)}")
    return samples_filtered

def map_classes(samples_gdf: gpd.GeoDataFrame, 
                class_mapping: Dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Map numeric class values to human-readable class names.

    Args:
        samples_gdf: GeoDataFrame containing samples with numeric class values
        class_mapping: Dictionary containing mapping between class values and names

    Returns:
        GeoDataFrame with added 'class' column containing class names
    """
    class_mapping['class_names'] = {int(k): v for k, v in class_mapping['class_names'].items()}
    samples_gdf['class'] = samples_gdf['remapped'].map(class_mapping['class_names'])
    return samples_gdf

def main():
    """
    Main function to sample negative points using ESRI LULC data.
    
    Loads configuration, initializes Earth Engine, samples points from specified 
    LULC classes, filters points near positive samples, and saves results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Sample Negative Points Using ESRI LULC')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Initialize Earth Engine
    initialize_ee_with_credentials()

    # Load positive points
    pos_gdf = gpd.read_parquet(config['input']['positive_points'])

    # Load AOI
    aoi = gpd.read_file(config['input']['aoi'])

    # Load and remap LULC
    esri_lulc = ee.ImageCollection(config['lulc']['collection'])
    lulc_image = esri_lulc.filterDate(config['lulc']['start_date'], config['lulc']['end_date']).mosaic()
    lulc_remapped = lulc_image.remap(config['lulc']['class_mapping']['input_classes'], 
                                    config['lulc']['class_mapping']['output_classes'])

    # Define sampling parameters
    samples = lulc_remapped.stratifiedSample(
        region=ee.Geometry(shapely.geometry.mapping(aoi.geometry.iloc[0])),
        scale=config['sampling']['scale'],
        numPoints=0,
        classValues=config['sampling']['class_values'],
        classPoints=config['sampling']['class_points'],
        seed=config['sampling']['seed'],
        geometries=True
    )

    # Convert samples to GeoDataFrame: NB this will fail for large sample numbers  in which case you will have to export it
    samples_gdf = geemap.ee_to_gdf(samples)
    logging.info(f"Original number of samples: {len(samples_gdf)}")

    # Determine UTM zone based on first positive point
    first_point = pos_gdf.geometry.iloc[0]
    utm_zone = int(((first_point.x + 180) / 6) + 1)
    hemisphere = 'N' if first_point.y >= 0 else 'S'
    utm_crs = f'EPSG:326{utm_zone:02d}' if hemisphere == 'N' else f'EPSG:327{utm_zone:02d}'

    # Do some initial cleaning
    before = len(samples_gdf)
    samples_gdf = samples_gdf[~samples_gdf.geometry.isin(pos_gdf.geometry)]
    after = len(samples_gdf)
    logging.info(f"Removed {before - after} samples that exactly matched positive points.")

    # Filter samples
    samples_filtered = filter_samples(
        samples_gdf=samples_gdf,
        pos_gdf=pos_gdf,
        buffer_size=config['sampling']['buffer_size'],
        utm_crs=utm_crs
    )

    # Map class integers to names
    samples_filtered = map_classes(samples_filtered, config['lulc']['class_mapping'])

    # Save filtered samples
    samples_filtered.to_parquet(config['output']['filtered_samples'])
    logging.info(f"Filtered samples saved to {config['output']['filtered_samples']}")

if __name__ == "__main__":
    main()