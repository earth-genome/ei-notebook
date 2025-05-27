# Crop Mapping Pipeline

This repository contains a pipeline for training a tile-based crop classifier using satellite imagery and interactive labeling.

## Prerequisites
First, there are some basic system requirements to take care of. Please refer to the linked documentation for installation.
1. `gcloud` CLI: https://cloud.google.com/sdk/docs/install
2. `earthengine` python client library: https://developers.google.com/earth-engine/guides/python_install

Please make sure you are authenticated with these tools before starting.

Next, please download the supporting files. These include a pre-generated `annoy` search index, tile geometries, and raw embeddings.

`gsutil cp -r gs://ei-notebook-assets/costa_rica/ /path/to/loca/data/dir`

Assets are available for Costa Rica (Central America), Sumatra (and Java), and West Africa. Please note that West Africa and Sumatra are larger AOIs and have larger associated files, which likely require remotely working in a high-memory virtual machine. The Central America AOI is relatively smaller and may be a better place to begin. 

Next, set the LOCAL_DATA_DIR shell variable to the downloaded folder, e.g.

`LOCAL_DATA_DIR=/Users/ben/EarthGenome/data/costa_rica`

This location will be referenced throughout this workflow.

Before starting work, set up your local code environment. It is recommended to use `conda` or `mamba` to manage python dependencies.

```
mamba create -n ei-notebook python=3.12 -y
mamba activate ei-notebook
mamba install -c conda-forge --file ./requirements_v2.txt -y
```

## Pipeline Overview

The pipeline consists of several steps:

1. Interactive labeling of positive examples
2. Sample negative examples
3. Create training dataset
4. Train tile classifier and run inference
5. Postprocess detections
6. Iterate!

## Pipeline Steps

### 1. Interactive labeling of examples
The Geolabeler provides an interactive map interface for labeling geographic points. Key features:

- Interactive map with multiple basemap options (Maptiler satellite, RGB/HSV median composite, Google Hybrid). **Please note that the RGB/HSV are generated on the fly from GEE. The dates for this imagery should be set using the config.**
- Point and lasso selection modes for efficient labeling
- Positive/negative/erase labeling options
- Direct Google Maps linking for reference
- Automatic saving of labeled points as GeoJSON files

To launch the labeling interface, first set up your `config` file. An example is provided at `./config/ui_config.json`. Next, run the `duckdb_ei.ipynb` cells. This will allow the user to search in the AOI by performing similarity serarch in the embeddings using a supplied `annoy` index. Please use the python environment you set up earlier to run the notebook.

The outputs of this step will be a `parquet` of positive samples (and optionally negative samples) with the tile ID of the closest tile centroid as an identifier.

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

The parameters, including the year of the ESRI LC map and, should be stored in a config file. An example that might work for in Costa Rica mapping is shown here. Please note that local files paths should be changed to match your set up.

```json
{
  "input": {
    "aoi": "/Users/ben/EarthGenome/code/ei-notebook/places/costa_rica.geojson",
    "positive_points": "/Users/ben/EarthGenome/data/costa_rica/positive_labels.parquet"
  },
  "lulc": {
    "collection": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "class_mapping": {
      "input_classes": [1, 2, 4, 5, 7, 8, 9, 10, 11],
      "output_classes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
      "class_names": {
        "1": "Water",
        "2": "Trees",
        "3": "Flooded Vegetation",
        "4": "Crops",
        "5": "Built Area",
        "6": "Bare Ground",
        "7": "Snow/Ice",
        "8": "Clouds",
        "9": "Rangeland"
      }
    }
  },
  "sampling": {
    "scale": 200,
    "class_values": [1, 2, 4, 5, 9],
    "class_points": [200, 4500, 4500, 4500, 1000],
    "seed": 0,
    "buffer_size": 200
  },
  "output": {
    "filtered_samples": "/Users/ben/EarthGenome/data/costa_rica/costa_rica_neg_samples.parquet"
  }
} 
```

Example usage:

```
python src/sample_negatives.py --config config/sample_negatives_config.json
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

Example usage, if you used the notebook to create positive labels and automatically sampled negative examples:
```
python src/make_dataset.py \
 --pos-gdf $LOCAL_DATA_DIR/positive_labels.parquet \
 --neg-lulc-gdf $LOCAL_DATA_DIR/costa_rica_neg_samples.parquet \
 --centroid-gdf $LOCAL_DATA_DIR/centroid_gdf.parquet \
 --embedding-db $LOCAL_DATA_DIR/embeddings.db \
 --output-dir $LOCAL_DATA_DIR/training_data \
 --region-name costa_rica \
 --version v1

```

Example usage where you exported both positive and negative examples from the notebook:
```
python src/make_dataset.py \
 --full-dataset $LOCAL_DATA_DIR/full_labels.parquet \
 --centroid-gdf $LOCAL_DATA_DIR/centroid_gdf.parquet \
 --embedding-db $LOCAL_DATA_DIR/embeddings.db \
 --output-dir $LOCAL_DATA_DIR/training_data \
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
    --classifier-dataset $LOCAL_DATA_DIR/training_data/tile_classifier_dataset_v1_costa_rica_embeddings.parquet \
    --embedding-dir $LOCAL_DATA_DIR/embeddings \
    --centroid-path $LOCAL_DATA_DIR/centroid_gdf.parquet \
    --output-dir $LOCAL_DATA_DIR//output \
    --version 1 \
    --region costa_rica \
    --pos-weight 1.0

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
    --input_file $LOCAL_DATA_DIR/output/tile_classifier_predictions_1_costa_rica_posw1.0.parquet \
    --tiles_dir $LOCAL_DATA_DIR/tiles \
    --output_dir $LOCAL_DATA_DIR/output \
    --prob_threshold 0.90
```

### 6. Iterate!
You should now be able to return the interactive notebook, load in the new detections and iterate on mapping.
















