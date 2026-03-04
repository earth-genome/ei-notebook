"""Build DuckDB database and a centroids GeoDataFrame from multiple embeddings
parquet files.

Requires an id column (default "tile_id") in all parquets to deduplicate
embeddings across overlapping regions. Geometry column is configurable via
--geometry_col (default "geometry").
"""

import argparse
import gc

import duckdb
import geopandas as gpd
import pandas as pd
import psutil
from tqdm import tqdm

def print_mem_usage(note=''):
    mem = psutil.virtual_memory()
    print(
        f"{note}: Memory usage: {mem.percent}% used, "
        f"{mem.available / (1024**3):.2f} GiB available"
    )
    
def main(
    parquet_paths,
    clip_path=None,
    db_path="embeddings.db",
    table_name="embeddings",
    centroids_path="centroids.parquet",
    id_col="tile_id",
    geometry_col="geometry",
):
    """Create DuckDB table and centroids parquet for out-of-memory embeddings ML.

    id_col must be present in all parquets; it is used to deduplicate across
    overlapping regions. geometry_col names the geometry column (default
    "geometry").
    """
    con = duckdb.connect(db_path)
    seen_tile_ids = set()
    centroid_dfs = []
    print_mem_usage("Start")

    for parquet_path in tqdm(parquet_paths):
        gdf = gpd.read_parquet(parquet_path)
        print_mem_usage("GDF loaded")

        if id_col not in gdf.columns:
            raise ValueError(
                f"id column '{id_col}' not in {parquet_path}. "
                "All parquets must have the id column for deduplication."
            )
        if geometry_col not in gdf.columns:
            raise ValueError(
                f"geometry column '{geometry_col}' not in {parquet_path}."
            )

        if clip_path:
            boundary = gpd.read_file(clip_path)
            gdf = gpd.clip(gdf, boundary)
        gdf = gdf.reset_index(drop=True)

        # Deduplicate within and across parquets
        gdf = gdf.drop_duplicates(subset=id_col)
        gdf = gdf[~gdf[id_col].isin(seen_tile_ids)]
        seen_tile_ids.update(gdf[id_col])

        update_duck_db(con, gdf, table_name, geometry_col=geometry_col)
        print_mem_usage("DuckDB updated")

        centroid_gdf = gdf.loc[:, [id_col, geometry_col]].copy()
        centroid_gdf[geometry_col] = centroid_gdf[geometry_col].centroid
        centroid_dfs.append(centroid_gdf)

        del gdf
        gc.collect()

    pd.concat(centroid_dfs, ignore_index=True).to_parquet(
        centroids_path, index=False
    )
    
    result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    print(f"Inserted {result[0]} rows into {table_name}.")
    sample = con.execute(
        f"SELECT * FROM {table_name} USING SAMPLE 2 ROWS").fetchdf()
    print(f"Sample rows:\n{sample}")
    con.close()

def update_duck_db(con, gdf, table_name="embeddings", geometry_col="geometry"):
    """Write chunk of embeddings GeoDataFrame to DuckDB (no geometry column)."""
    df = gdf.drop(columns=geometry_col)
    con.register("df", df)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0"
    )
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    con.unregister("df")
    del df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build embeddings.db and centroids parquet from embedding parquets."
    )
    parser.add_argument(
        "parquet_paths",
        type=str,
        nargs="+",
        help="Paths to embeddings parquet files (vectors + geometry).",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="GeoJSON boundary to clip embeddings.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="embeddings.db",
        help="Output DuckDB path.",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default="embeddings",
        help="DuckDB table name.",
    )
    parser.add_argument(
        "--centroids_path",
        type=str,
        default="centroids.parquet",
        help="Output centroids parquet path.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="tile_id",
        help="Id column name (must exist in all parquets; used for deduplication).",
    )
    parser.add_argument(
        "--geometry_col",
        type=str,
        default="geometry",
        help="Geometry column name.",
    )
    args = parser.parse_args()
    main(**vars(args))
