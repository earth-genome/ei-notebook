"""ML and validation utilities for satellite embedding models.

Used by ei_alt_workflow.ipynb. Expects an EmbeddingMapper (from embedding_store)
with .gdf, .get_vectors(ids), and .id_column. predict_df and detections_to_rectpolys
use embeddings.id_column so the user does not pass id/tile column names explicitly.
"""

import math
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from shapely.geometry import box

warnings.simplefilter("ignore", category=FutureWarning)


def predict(X, model, threshold=0.5):
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


def predict_df(df, embeddings, model, threshold=0.5):
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
    return fig, ax


def prec_rec_curve(y_true, probs, annotate_thresholds=False):
    """Compute precision-recall curve."""
    prec, rec, thresholds = metrics.precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="Patchwise")
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

    Batches by gdf range index; converts to ids only for get_vectors(ids).
    boundary_path: Optional path to GeoJSON/shapefile to clip detections.
    Output GeoDataFrame has an id column with the same name as embeddings.id_column,
    plus geometry and probability.
    """
    gdf = embeddings.gdf
    id_column = embeddings.id_column
    n = len(gdf)
    n_batches = math.ceil(n / batch_size)
    detections_list = []
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_positions = np.arange(start, end)
        ids = gdf[id_column].iloc[batch_positions]
        X = embeddings.get_vectors(ids)
        probs, y_pred = predict(X, model, threshold)
        mask = y_pred.astype(bool)
        if not mask.any():
            continue
        out = gdf.iloc[batch_positions[mask]].copy()
        out["probability"] = probs[mask]
        detections_list.append(out)
    if not detections_list:
        empty = gdf.iloc[0:0].copy()
        empty["probability"] = pd.Series(dtype=float)
        return empty
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
    patch_width: patch size in meters; converted to degrees at the centroid of
        embeddings.gdf bounds so buffering stays in 4326 and edges align.
    """
    id_column = embeddings.id_column
    gdf = embeddings.gdf
    centroids = gdf[gdf[id_column].isin(detections[id_column])].copy()

    def meters_to_degrees_half(meters, ref_lon, ref_lat):
        """Infer half-extent in degrees (lat, lon) for a given half-extent in meters at a reference point (lon, lat)."""
        # Approx: 1 deg lat ~ 111320 m; 1 deg lon ~ 111320 * cos(lat) m
        half_m = meters / 2.0
        lat_rad = math.radians(ref_lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(lat_rad)
        half_lat_deg = half_m / m_per_deg_lat
        half_lon_deg = half_m / m_per_deg_lon
        return half_lat_deg, half_lon_deg

    # Reference point: centroid of embeddings.gdf bounds (for consistent degree scale)
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy (lon, lat)
    ref_lon = (bounds[0] + bounds[2]) / 2.0
    ref_lat = (bounds[1] + bounds[3]) / 2.0
    half_lat_deg, half_lon_deg = meters_to_degrees_half(patch_width, ref_lon, ref_lat)

    # Build axis-aligned squares in 4326 (no UTM)
    def make_box(geom):
        lon, lat = geom.x, geom.y
        return box(
            lon - half_lon_deg,
            lat - half_lat_deg,
            lon + half_lon_deg,
            lat + half_lat_deg,
        )

    boxes = gpd.GeoSeries(centroids.geometry.apply(make_box), crs="EPSG:4326")
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
