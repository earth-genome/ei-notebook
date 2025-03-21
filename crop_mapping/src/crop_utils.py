import os
from typing import Dict, List, Optional

import geopandas as gpd

from utils.geometry import get_crs_from_tile
from utils.models import MGRSTileGrid
from data_loader.utils.stac import CLIENT


def download_wgs84_tiles(
        mgrs_id: str,
        stac_config: Dict[str, str],
        tiling_config: Dict[str, str],
        local_tile_dir: str,
        project_to_wgs84: bool = False) -> Optional[gpd.GeoDataFrame]:
    """
    Get WGS84 tiles for a given MGRS ID.

    Args:
        mgrs_id: MGRS tile ID
        stac_config: STAC configuration
        tiling_config: Tiling configuration
        local_tile_dir: Local directory to store downloaded tiles

    Returns:
        Tiles in WGS84 projection or None if MGRS item not found
    """
    search = CLIENT.search(
        collections=[stac_config['mgrs_id']],
        ids=[mgrs_id]
    )
    mgrs_items = [i for i in search.items()]
    if not mgrs_items:
        print(f"MGRS item {mgrs_id} not found. Skipping...")
        return None
    mgrs_item = mgrs_items[0]

    crs = get_crs_from_tile(mgrs_item)

    grid = MGRSTileGrid(
        mgrs_tile_id=mgrs_id,
        crs=crs,
        tilesize=tiling_config['tilesize'],
        overlap=tiling_config['overlap'],
        resolution=tiling_config['resolution']
    )

    gs_dir = f"{tiling_config['gs_bucket']}/{tiling_config['gs_base']}"
    tile_fname = grid.prefix
    gs_out = f"gs://{gs_dir}/{tile_fname}.parquet"
    local_path = f"{local_tile_dir}/{tile_fname}.parquet"
    if os.path.exists(local_path):
        return None
    gdf = gpd.read_parquet(gs_out)
    if project_to_wgs84:
        gdf = gdf.to_crs(4326)
    gdf.to_parquet(local_path)
    return None