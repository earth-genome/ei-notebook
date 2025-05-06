import argparse
import glob
from pathlib import Path

from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np

import os


def train_classifier(tile_classifier_dataset_path, output_dir,
                     mlp_layer_sizes=None, pos_weight=1.0):
    tile_classifier_dataset = gpd.read_parquet(tile_classifier_dataset_path).reset_index(drop=True)
    feature_cols = [col for col in tile_classifier_dataset.columns if 'vit' in col]
    X = tile_classifier_dataset[feature_cols]
    y = tile_classifier_dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25,
        random_state=42,
        #stratify=tile_classifier_dataset['class']
    )

    # Create sample weights
    sample_weight = np.ones_like(y_train, dtype=float)
    sample_weight[y_train == 1] = pos_weight

    now = datetime.today().isoformat()[:16]
    if mlp_layer_sizes:
        model = MLPClassifier(
            hidden_layer_sizes=mlp_layer_sizes, n_iter_no_change=40,
            max_iter=1000, verbose=True)
        model_path = os.path.join(
            output_dir,
            f'MLP{"-".join([str(s) for s in mlp_layer_sizes])}_{now}.joblib')
        print('Warning: Sample weight not implemented for MLP classifier.')
        model.fit(X_train, y_train)
    else:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42, n_jobs=-1
        )
        model_path = os.path.join(output_dir, f'XGB_{now}.joblib')
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

    print(f'Model saved to: {model_path}')
    joblib.dump(model, model_path)


def plot_curves(y_test, y_pred_proba, output_dir,
                thresholds=np.arange(0, 1.025, .025)):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
    pr_auc = metrics.average_precision_score(y_test, y_pred_proba)
    f1s = [metrics.f1_score(y_test, (y_pred_proba > t).astype('int'))
               for t in thresholds]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

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

    ax3.plot(thresholds, f1s, label='F1 score')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1 score')
    ax3.legend(loc='lower left')

    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_curves.png')
    plt.close()

def parse_tuple(s):
    try:
        return tuple(int(x) for x in s.strip("()").split(","))
    except:
        raise argparse.ArgumentTypeError(
            'Give tuple in form (int,int,...), e.g. (64,16).')

def main():
    parser = argparse.ArgumentParser(description='Train tile classifier')
    parser.add_argument('--classifier-dataset', required=True,
                        help='Path to tile classifier dataset parquet file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--pos-weight', type=float, default=1.0, 
        help='Weight multiplier for positive samples (default: 1.0)')
    parser.add_argument('--mlp-layer-sizes', type=parse_tuple, default=())
    
    
    args = parser.parse_args()
    train_classifier(args.classifier_dataset, args.output_dir,
                     args.mlp_layer_sizes, args.pos_weight)


if __name__ == '__main__':
    main()
