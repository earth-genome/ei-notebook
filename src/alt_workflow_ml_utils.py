"""ML and validation utilities for satellite embedding models.

Used by ei_alt_workflow.ipynb. Expects an EmbeddingMapper (from embedding_store)
with .gdf, .get_vectors(ids), and .id_column (gdf.index.name). predict_df and
detections_to_rectpolys use embeddings.id_column so the user does not pass id/tile
column names explicitly.
"""

import math
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

warnings.simplefilter("ignore", category=FutureWarning)


def predict(X, model, threshold):
    """Run model inference.

    Arguments:
        X: A dataframe or array of model input vectors.
        model: An sklearn model with predict_proba() and binary outputs.
        threshold: Numerical cutoff for positive class assignment.

    Returns:
        (probs, y_pred): Positive class probabilities and binary predictions.
    """
    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype("int")
    return probs, y_pred


def predict_df(df, embeddings, model, threshold=1):
    """Run model inference on a dataframe of labeled geographic points."""
    tile_ids = df[embeddings.id_column]
    X = embeddings.get_vectors(tile_ids)
    return predict(X, model, threshold)


def score(y_pred, y_true):
    """Return a confusion matrix and model metrics."""
    tricks = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "confusion": metrics.confusion_matrix,
    }
    scores = {t: f(y_true, y_pred) for t, f in tricks.items()}
    for t, s in scores.items():
        print(f"{t}: {s}")
    scores["confusion"] = scores["confusion"].tolist()
    return scores


def f1_curve(y_true, probs, thresholds=np.arange(0, 1.025, 0.025)):
    """Compute F1 curve for a range of thresholds."""
    f1s = [
        metrics.f1_score(y_true, (probs >= t).astype(int)) for t in thresholds
    ]
    fig, ax = plt.subplots()
    ax.plot(thresholds, f1s)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")
    ax.legend(loc="lower left")
    return fig, ax


def prec_rec_curve(y_true, probs, annotate_thresholds=False):
    """Compute precision-recall curve."""
    prec, rec, thresholds = metrics.precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    if annotate_thresholds:
        step = max(1, len(thresholds) // 20)
        for x, y, txt in zip(rec[::step], prec[::step], thresholds[::step]):
            ax.annotate(np.round(txt, 3), (x, y - 0.04))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    return fig, ax


def roc_curve(y_true, probs, annotate_thresholds=False):
    """Compute ROC curve."""
    fpr, tpr, thresholds = metrics.roc_curve(y_true, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Patchwise")
    if annotate_thresholds:
        step = max(1, len(thresholds) // 20)
        for x, y, txt in zip(fpr[::step], tpr[::step], thresholds[::step]):
            ax.annotate(np.round(txt, 2), (x, y - 0.04))
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks[::-1])
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right")
    return fig, ax


def get_detections(embeddings, model, threshold, boundary_path=None, batch_size=10000):
    """Run model over all embedding centroids and return positive detections.

    embeddings: EmbeddingMapper (embedding_store) with .gdf and .get_vectors(ids).
    boundary_path: Optional path to GeoJSON/shapefile to clip detections.
    Output GeoDataFrame has an id column with the same name as embeddings.id_column,
    plus geometry and probability.
    """
    gdf = embeddings.gdf
    tile_ids = gdf.index.to_numpy()
    n_batches = math.ceil(len(tile_ids) / batch_size)
    batches = [
        tile_ids[i * batch_size : (i + 1) * batch_size]
        for i in range(n_batches)
    ]
    detections_list = []
    for batch in batches:
        X = embeddings.get_vectors(batch)
        probs, y_pred = predict(X, model, threshold)
        mask = y_pred.astype(bool)
        out = gdf.loc[batch[mask]].copy()
        out["probability"] = probs[mask]
        detections_list.append(out.reset_index())
    detections = pd.concat(detections_list, ignore_index=True)
    detections = gpd.GeoDataFrame(detections, geometry="geometry")
    if boundary_path:
        boundary = gpd.read_file(boundary_path)
        detections = gpd.clip(detections, boundary).reset_index(drop=True)
    return detections


def detections_to_rectpolys(
    embeddings,
    detections,
    patch_width=320,
    buffer_width=0.00001,
    confidence_measure="probability",
):
    """Convert point detections to merged polygons with confidence.

    detections: GeoDataFrame from get_detections (must have id column and geometry).
    Uses embeddings.id_column to find the id column in detections.
    """
    id_col = embeddings.id_column
    centroids = embeddings.gdf.loc[detections[id_col]]
    first_point = centroids.geometry.iloc[0]
    zone = int((first_point.x + 180) / 6) + 1
    epsg = 32600 + zone if first_point.y >= 0 else 32700 + zone
    centroids_utm = centroids.geometry.to_crs(f"EPSG:{epsg}")
    boxes = centroids_utm.buffer(int(patch_width / 2), cap_style=3)
    boxes = gpd.GeoSeries(boxes).set_crs(f"EPSG:{epsg}").to_crs("EPSG:4326")
    merged = boxes.buffer(buffer_width, join_style=2).union_all()
    polys = gpd.GeoDataFrame(geometry=[merged]).explode(index_parts=False)
    polys = polys.buffer(-buffer_width, join_style=2)
    polys = gpd.GeoDataFrame(geometry=polys).set_crs("EPSG:4326")
    polys["confidence"] = polys.geometry.apply(
        lambda poly: detections.iloc[detections.sindex.query(poly)][
            confidence_measure
        ].mean()
    )
    return polys
