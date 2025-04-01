import argparse
import glob
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

import os


def train_classifier(tile_classifier_dataset_path, output_dir, pos_weight=1.0):
    tile_classifier_dataset = gpd.read_parquet(tile_classifier_dataset_path).reset_index(drop=True)
    feature_cols = [col for col in tile_classifier_dataset.columns if 'vit' in col]
    X = tile_classifier_dataset[feature_cols]
    y = tile_classifier_dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25,
        random_state=42,
        stratify=tile_classifier_dataset['class']
    )

    # Create sample weights
    sample_weight = np.ones_like(y_train, dtype=float)
    sample_weight[y_train == 1] = pos_weight

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    print(f"Weighted Precision: {precision:.3f}")
    print(f"Weighted Recall: {recall:.3f}") 
    print(f"Weighted F1 Score: {f1:.3f}")

    # Per-class metrics
    class_report = metrics.classification_report(y_test, y_pred)
    print("\nPer-class metrics:")
    print(class_report)

    # Plot ROC and PR curves
    plot_curves(y_test, y_pred_proba, output_dir)

    return model, X_train.index.tolist()


def plot_curves(y_test, y_pred_proba, output_dir):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
    pr_auc = metrics.average_precision_score(y_test, y_pred_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_curves.png')
    plt.close()


def process_embedding_file(embedding_file, model, feature_cols):
    embeddings = pd.read_parquet(embedding_file)
    tile_ids = embeddings['tile_id']
    features = embeddings[feature_cols]
    proba = model.predict_proba(features)[:,1]
    high_conf_mask = proba > 0.5
    if high_conf_mask.any():
        high_conf_embeddings = pd.DataFrame({'tile_id': tile_ids[high_conf_mask]})
        high_conf_embeddings['prediction_probability'] = proba[high_conf_mask]
        return high_conf_embeddings
    return None


def predict_embeddings(model, embedding_dir, feature_cols, output_dir, centroid_path, train_indices,
                       version, region, tile_classifier_dataset_path, pos_weight):
    embedding_files = glob.glob(f"{embedding_dir}/*.parquet")
    
    predictions = Parallel(n_jobs=20, verbose=10)(
        delayed(process_embedding_file)(f, model, feature_cols) 
        for f in embedding_files
    )
    predictions = [p for p in predictions if p is not None]

    if predictions:
        prediction_df = pd.concat(predictions, ignore_index=True)
        
        centroid_gdf = gpd.read_parquet(centroid_path)
        prediction_gdf = prediction_df.merge(centroid_gdf, on='tile_id', how='left')
        prediction_gdf = gpd.GeoDataFrame(prediction_gdf, geometry='geometry')
        
        unfiltered_output_path = f"{output_dir}/tile_classifier_predictions_{version}_{region}_posw{pos_weight}.parquet"
        print(f"Saving unfiltered predictions to {unfiltered_output_path}")
        prediction_gdf.to_parquet(unfiltered_output_path)
        print(f"Saved unfiltered predictions to {unfiltered_output_path}")

        tile_classifier_dataset = gpd.read_parquet(tile_classifier_dataset_path)
        training_tile_ids = tile_classifier_dataset[tile_classifier_dataset.index.isin(train_indices)]['tile_id']
        filtered_prediction_gdf = prediction_gdf[~prediction_gdf['tile_id'].isin(training_tile_ids)]
        
        output_path = f"{output_dir}/tile_classifier_predictions_{version}_{region}_posw{pos_weight}_filtered_no_train.parquet"
        filtered_prediction_gdf.to_parquet(output_path)
        print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train tile classifier and generate predictions')
    parser.add_argument('--classifier-dataset', required=True, help='Path to tile classifier dataset parquet file')
    parser.add_argument('--embedding-dir', required=True, help='Directory containing embedding parquet files')
    parser.add_argument('--centroid-path', required=True, help='Path to centroid GeoDataFrame parquet file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--version', default='v1', help='Version identifier for output files')
    parser.add_argument('--region', required=True, help='Region identifier (e.g., java)')
    parser.add_argument('--pos-weight', type=float, default=1.0, 
                       help='Weight multiplier for positive samples (default: 1.0)')
    
    args = parser.parse_args()
    
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir
    # Train model with pos_weight
    model, train_indices = train_classifier(args.classifier_dataset, args.output_dir, args.pos_weight)
    
    # Get feature columns
    tile_classifier_dataset = gpd.read_parquet(args.classifier_dataset)
    feature_cols = [col for col in tile_classifier_dataset.columns if 'vit' in col]
    
    # Generate predictions
    predict_embeddings(
        model, 
        args.embedding_dir,
        feature_cols,
        output_dir,
        args.centroid_path,
        train_indices,
        args.version,
        args.region,
        args.classifier_dataset,
        args.pos_weight
    )


if __name__ == '__main__':
    main()
