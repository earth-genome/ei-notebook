
import argparse
import glob

import joblib
from joblib import Parallel, delayed
import geopandas as gpd
import pandas as pd

def predict_embeddings(model, threshold, embedding_dir, feature_cols,
                       output_dir, centroid_path, version, region,
                       tile_classifier_dataset_path, pos_weight):
    embedding_files = glob.glob(f"{embedding_dir}/*.parquet")
    
    predictions = Parallel(n_jobs=20, verbose=10)(
        delayed(process_embedding_file)(f, model, threshold, feature_cols) 
        for f in embedding_files
    )
    predictions = [p for p in predictions if p is not None]

    if predictions:
        prediction_df = pd.concat(predictions, ignore_index=True)
        
        centroid_gdf = gpd.read_parquet(centroid_path)
        prediction_gdf = prediction_df.merge(centroid_gdf, on='tile_id', how='left')
        prediction_gdf = gpd.GeoDataFrame(
            prediction_gdf[~prediction_gdf.geometry.isnull()],
            geometry='geometry')
        
        unfiltered_output_path = f"{output_dir}/tile_classifier_predictions_{version}_{region}_posw{pos_weight}.parquet"
        print(f"Saving unfiltered predictions to {unfiltered_output_path}")
        prediction_gdf.to_parquet(unfiltered_output_path)

def process_embedding_file(embedding_file, model, threshold, feature_cols):
    embeddings = pd.read_parquet(embedding_file)
    tile_ids = embeddings['tile_id']
    features = embeddings[feature_cols]
    proba = model.predict_proba(features)[:,1]
    high_conf_mask = proba > threshold
    if high_conf_mask.any():
        high_conf_embeddings = pd.DataFrame({'tile_id': tile_ids[high_conf_mask]})
        high_conf_embeddings['prediction_probability'] = proba[high_conf_mask]
        return high_conf_embeddings
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--classifier-dataset', required=True, help='Path to tile classifier dataset parquet file')
    parser.add_argument('--embedding-dir', required=True, help='Directory containing embedding parquet files')
    parser.add_argument('--centroid-path', required=True, help='Path to centroid GeoDataFrame parquet file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--version', default='v1', help='Version identifier for output files')
    parser.add_argument('--region', required=True, help='Region identifier (e.g., java)')
    parser.add_argument('--pos-weight', type=float, default=1.0, 
                       help='Weight multiplier for positive samples (default: 1.0)')
    parser.add_argument('--model-path', required=True, help='Path to saved sklearn model')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Prediction probability threshold')
    
    
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    
    # Get feature columns
    tile_classifier_dataset = gpd.read_parquet(args.classifier_dataset)
    feature_cols = [col for col in tile_classifier_dataset.columns if 'vit' in col]
    
    # Generate predictions
    predict_embeddings(
        model,
        args.threshold,
        args.embedding_dir,
        feature_cols,
        args.output_dir,
        args.centroid_path,
        args.version,
        args.region,
        args.classifier_dataset,
        args.pos_weight
    )


