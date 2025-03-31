# Crop Mapping Pipeline

This repository contains a pipeline for training a tile-based crop classifier using satellite imagery and interactive labeling.

## Overview

The pipeline consists of several steps:

1. Interactive labeling of positive examples
2. Sample negative examples
3. Create training dataset
4. Train tile classifier and run inference
5. Postprocess detections

## Pipeline Steps

### 1. Interactive labeling of examples
The Geolabeler provides an interactive map interface for labeling geographic points. Key features:

- Interactive map with multiple basemap options (Maptiler satellite, HSV median composite, Google Hybrid)
- Point and lasso selection modes for efficient labeling
- Positive/negative/erase labeling options
- Direct Google Maps linking for reference
- Automatic saving of labeled points as GeoJSON files

To launch the labeling interface: run the `duckdb_ei.ipynb` cells. This will allow the user to search over the AOI that was processed in step 1 by performing similarity serarch in the embeddings using the `annoy` index.
The outputs of this step will be a `geojson` of positive samples (and optionally negative samples) with the tile ID of the closest tile centroid as an identifier.

### 2. Sample Negative Points

It is possible to output positive and negative samples from the previous interactive labeling step. If that is your preferred method, you can skip right to step 3.

However, we have found in experimentation that it is often best to have a training set class balance that roughly follows the real-world classs frequency of tiles. In the case of most applications, therefore, it is helpful to have many more negative samples than positive ones. 

The script `sample_negatives.py` can automate the sampling of negative points, by intelligent sampling from the ESRI Global Land Use/Land Cover dataset.

Run `sample_negatives.py` to generate negative samples using the ESRI Global Land Use/Land Cover dataset. This script:

- Takes positive samples and an AOI as input
- Samples points from specified LULC classes (e.g. water, trees, built, rangeland)
- Filters out points that are too close to positive samples using a buffer
- Maps LULC class integers to human-readable names
- Outputs a parquet file of filtered negative samples

The parameters, including the year of the ESRI LC map and, should be stored in a config file. An example that could work for the pineapple mapping model is shown here:
```json
{
  "input": {
    "aoi": "gs://demeter-labs/tea/geometries/ra_java_only_aoi.geojson",
    "positive_points": "gs://demeter-labs/tea/ei-datasets/pos_gdf_v1_sumatra_2024-11-10.parquet"
  },
  "lulc": {
    "collection": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "class_mapping": {
      "input_classes": [1, 2, 4, 5, 7, 8, 9, 10, 11],
      "output_classes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
      "class_names": {
        "1": "Water",
        "2": "Trees",
        "3": "Built",
        "4": "Crops",
        "5": "Crops",
        "6": "Flooded Vegetation",
        "7": "Built",
        "8": "Bare Ground",
        "9": "Rangeland"
      }
    }
  },
  "sampling": {
    "scale": 320,
    "class_values": [1, 2, 4, 5, 9],
    "class_points": [200, 3000, 3000, 3000, 1000],
    "seed": 0,
    "buffer_size": 320
  },
  "output": {
    "filtered_samples": "gs://demeter-labs/tea/samples/java_neg_water_built_tree_rangeland_samples_10091.parquet"
  }
} 
```


### 3. Create Training Dataset

Run `make_dataset.py` to combine positive and negative samples with embeddings into a training dataset. This script:

- Takes positive samples, negative samples from EI and LULC, and tile centroids as input
- Joins samples with their nearest MGRS tile centroids 
- Assigns binary labels (1 for positive, 0 for negative)
- Adds class labels for each point type (ei_pos, ei_neg, Water, Trees, etc.)
- Merges with embedding values from a DuckDB database
- Outputs two parquet files:
  - A dataset with just the samples and labels
  - A dataset that includes the embeddings for model training

Example usage:
```
python src/make_dataset.py \
  --pos-gdf $LOCAL_DIR/pos_gdf_v1_java_2024-11-10.parquet \
  --neg-ei-gdf $LOCAL_DIR/neg_gdf_v1_java_2024-11-10.parquet \
  --neg-lulc-gdf $LOCAL_DIR/java_neg_water_built_tree_rangeland_samples_10091.parquet \
  --centroid-gdf $LOCAL_DIR/mgrs_tiles/centroid_gdf.parquet \
  --embedding-db $LOCAL_DIR/embeddings/embeddings.duckdb \
  --output-dir $LOCAL_DIR/training_data \
  --region-name costa_rica \
  --version v1
```

### 4. Train Tile Classifier and Deploy

Run `tile_classifier.py` to train an XGBoost binary classifier on the embedding dataset. This script:

- Loads the training dataset with embeddings
- Splits data into train/test sets (75%/25%) stratified by class
- Trains an XGBoost classifier with optional positive class weighting
- Evaluates model performance with:
  - Training and test accuracy
  - Precision, recall and F1 scores
  - Per-class metrics
  - ROC and PR curves
- Generates predictions on all tiles in the embedding directory
- Filters out training tiles from predictions
- Outputs:
  - Model performance curves plot
  - Two prediction files:
    - All predictions above 0.5 confidence
    - Predictions filtered to exclude training tiles

Example usage:
```
python src/tile_classifier.py \
  --train-data $LOCAL_DIR/training_data/tile_classifier_dataset_v1_java_embeddings.parquet \
  --embedding-dir $LOCAL_DIR/embeddings \
  --output-dir $LOCAL_DIR/predictions \
  --region-name costa_rica \
  --version v1 \
  --pos-weight 2.0 \
  --test-size 0.25 \
  --random-seed 42
```

### 5. Postprocess Detections

Run `postprocess_detections.py` to process the tile classifier predictions into final polygons. This script:

- Loads the prediction parquet file and filters by probability threshold
- For each MGRS tile ID in predictions:
  - Finds corresponding tile geometry file
  - Filters to only predicted tiles
- Merges prediction probabilities with tile geometries
- Dissolves overlapping tile geometries into continuous polygons
- Calculates area for each polygon in its local UTM zone
- Outputs a parquet file with:
  - Dissolved polygons
  - Area calculations
  - Original prediction probabilities

Example usage:
```
python src/postprocess_detections.py \
  $LOCAL_DIR/predictions/tile_classifier_predictions_v1_java.parquet \
  $LOCAL_DIR/tiles \
  $LOCAL_DIR/predictions \
  --prob_threshold 0.9

```
















