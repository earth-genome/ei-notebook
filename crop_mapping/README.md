# Crop Mapping Pipeline

This repository contains a pipeline for training a tile-based crop classifier using satellite imagery and interactive labeling.

## Overview

The pipeline consists of several steps:

1. Generate embeddings and search index
2. Interactive labeling of positive examples
3. Sample negative examples
4. Create training dataset
5. Train tile classifier

## Pipeline Steps

### 1. Generate Embeddings and Search Index

First, run `mgrs_embeddings_to_annoy.py` to:
- Download tile embeddings for the region of interest
- Create an Annoy index for fast similarity search
- Generate a DuckDB database of embeddings
- Create a tile centroid GeoDataFrame 

```python
python src/mgrs_embeddings_to_annoy.py \
--roi_geojson path/to/region.geojson \
--image_config configs.image_config \
--embeddings_config configs.embeddings_config \
--n_jobs -1 \
--n_trees 10 \
--ui_config_output output/ui_config.json \
--output_dir output/embeddings
```
This will generate several outputs in the specified output directory:

- `embeddings/`: Directory containing downloaded embedding parquet files
- `tiles/`: Directory containing downloaded tile parquet files 
- `centroid_gdf.parquet`: GeoDataFrame with tile centroids
- `embeddings.ann`: Annoy index file for similarity search
- `embeddings.db`: DuckDB database containing embeddings
- `ui_config.json`: Configuration file for the labeling UI containing:
  - Local directory paths
  - List of MGRS tile IDs
  - Embedding dimension
  - Date range
  - Imagery path

NB: this should be run on a VM with fairly large RAM. For Java + Sumatra this was run on a `c3d-standard-30` instance with 128 GB of memory.


### 2. Use Geolabeler to Generate Positives
The Geolabeler provides an interactive map interface for labeling geographic points. Key features:

- Interactive map with multiple basemap options (Maptiler satellite, HSV median composite, Google Hybrid)
- Point and lasso selection modes for efficient labeling
- Positive/negative/erase labeling options
- Direct Google Maps linking for reference
- Automatic saving of labeled points as GeoJSON files

To launch the labeling interface: run the `duckdb_ei.ipynb` cells. This will allow the user to search over the AOI that was processed in step 1 by performing similarity serarch in the embeddings using the `annoy` index.
The outputs of this step will be a `geojson` of positive samples with the tile ID of the closest tile centroid as an identifier.

### 3. Sample Negative Points

Run `sample_negatives.py` to generate negative samples using the ESRI Global Land Use/Land Cover dataset. This script:

- Takes positive samples and an AOI as input
- Samples points from specified LULC classes (e.g. water, trees, built, rangeland)
- Filters out points that are too close to positive samples using a buffer
- Maps LULC class integers to human-readable names
- Outputs a parquet file of filtered negative samples

The parameters, including the year of the ESRI LC map and, should be stored in a config file. An example for the tea v0 model is shown here:
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


### 4. Create Training Dataset

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
  --pos-gdf gs://demeter-labs/tea/ei-datasets/pos_gdf_v1_java_2024-11-10.parquet \
  --neg-ei-gdf gs://demeter-labs/tea/ei-datasets/neg_gdf_v1_java_2024-11-10.parquet \
  --neg-lulc-gdf gs://demeter-labs/tea/samples/java_neg_water_built_tree_rangeland_samples_10091.parquet \
  --centroid-gdf gs://demeter-labs/tea/mgrs_tiles/centroid_gdf.parquet \
  --embedding-db gs://demeter-labs/tea/embeddings/embeddings.duckdb \
  --output-dir gs://demeter-labs/tea/training_data \
  --region-name java \
  --version v1
```

### 5. Train Tile Classifier and Deploy

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
  --train-data gs://demeter-labs/tea/training_data/tile_classifier_dataset_v1_java_embeddings.parquet \
  --embedding-dir gs://demeter-labs/tea/embeddings \
  --output-dir gs://demeter-labs/tea/predictions \
  --region-name java \
  --version v1 \
  --pos-weight 2.0 \
  --test-size 0.25 \
  --random-seed 42
```

### 6. Postprocess Detections

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
  gs://demeter-labs/tea/predictions/tile_classifier_predictions_v1_java.parquet \
  gs://demeter-labs/tea/tiles \
  gs://demeter-labs/tea/predictions \
  --prob_threshold 0.9

```
















