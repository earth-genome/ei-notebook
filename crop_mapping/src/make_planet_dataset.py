import argparse
import geopandas as gpd
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from google.cloud import storage


def save_to_gcs(df, output_path):
    # Parse GCS path
    bucket_name = output_path.split('/')[2]
    blob_path = '/'.join(output_path.split('/')[3:])
    
    # Save locally first
    tmp_path = f"/tmp/{Path(blob_path).name}"
    df.to_parquet(tmp_path)
    
    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(tmp_path)
    
    # Clean up temp file
    os.remove(tmp_path)
    return output_path


def make_single_monthly_planet_dataset(planet_gdf_path, v2_dataset, output_dir):
    # Create output filename
    input_fname = Path(planet_gdf_path).stem
    output_fname = f"{input_fname}_v2_embeddings.parquet"
    output_path = os.path.join(output_dir, output_fname)

    # Check if file exists in GCS
    storage_client = storage.Client()
    bucket_name = output_dir.split('/')[2]  # Assuming format gs://bucket/path
    bucket = storage_client.bucket(bucket_name)
    output_blob_path = '/'.join(output_dir.split('/')[3:] + [output_fname])
    blob = bucket.blob(output_blob_path)

    if blob.exists():
        print(f"Output file gs://{bucket_name}/{output_blob_path} already exists, skipping...")
        return f"gs://{bucket_name}/{output_blob_path}"

    # Read planet GDF
    planet_gdf = gpd.read_parquet(planet_gdf_path, columns=["id", "geometry"])
    
    # Calculate UTM zone
    planet_gdf['utm_zone'] = ((planet_gdf.geometry.centroid.x + 180) / 6 + 1).astype(int)
    
    # Adjust for southern hemisphere
    if (planet_gdf.geometry.centroid.y < 0).all():
        planet_gdf['utm_zone'] = planet_gdf['utm_zone'] + 30
    
    results = []
    # Process each UTM zone group
    for zone, group in tqdm(planet_gdf.groupby('utm_zone'), desc="Processing UTM zones"):
        utm_crs = f'EPSG:327{zone:02d}'
        projected_group = group.to_crs(utm_crs)
        projected_group.geometry = projected_group.geometry.centroid
        reprojected_group = projected_group.to_crs('EPSG:4326')
        results.append(reprojected_group)

    planet_centroid_gdf = gpd.GeoDataFrame(pd.concat(results))

    v2_dataset['utm_zone'] = ((v2_dataset.geometry.centroid.x + 180) / 6 + 1).astype(int)
    final_results = []

    for zone, v2_group in v2_dataset.groupby('utm_zone'):
        results_subset = planet_centroid_gdf[planet_centroid_gdf.utm_zone == zone].copy()
        
        if len(results_subset) == 0:
            continue
            
        if (v2_group.geometry.centroid.y < 0).all():
            utm_crs = f'EPSG:327{zone:02d}'
        else:
            utm_crs = f'EPSG:326{zone:02d}'
            
        v2_utm = v2_group.to_crs(utm_crs)
        results_utm = results_subset.to_crs(utm_crs)
        joined = gpd.sjoin_nearest(v2_utm, results_utm, how='left', distance_col='distance')
        final_results.append(joined.to_crs('EPSG:4326'))

    final_df = pd.concat(final_results, ignore_index=True)
    filtered_df = final_df[final_df['distance'] <= 160]
    
    print(f"Dropped {len(final_df) - len(filtered_df)} rows "
          f"({(len(final_df) - len(filtered_df))/len(final_df)*100:.1f}%) due to distance > 160m")

    ids_to_match = filtered_df['id'].tolist()
    embeddings_df = pd.read_parquet(
        planet_gdf_path,
        columns=['id', 'embedding'],
        filters=[('id', 'in', ids_to_match)]
    )

    final_df = filtered_df.merge(embeddings_df, on='id', how='left')
    final_df = final_df.drop_duplicates(subset='id', keep='first')
    
    # Save file to GCS using new helper function
    return save_to_gcs(final_df, output_path)


def main():
    parser = argparse.ArgumentParser(description='Process Planet embeddings with v2 dataset')
    parser.add_argument('--input_dir', required=True, help='Directory containing planet_gdf files')
    parser.add_argument('--v2_dataset', required=True, help='Path to v2_dataset file')
    parser.add_argument('--output_dir', required=True, help='Directory for output files')
    
    args = parser.parse_args()
    
    # Read v2_dataset once
    v2_dataset = gpd.read_parquet(args.v2_dataset)
    
    # Process all parquet files in input directory
    input_dir = Path(args.input_dir)
    
    for input_file in tqdm(input_dir.glob('*.parquet'), desc="Processing files"):
        print(f"\nProcessing {input_file.name}...")
        output_path = make_single_monthly_planet_dataset(str(input_file), v2_dataset, args.output_dir)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()