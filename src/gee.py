from dotenv import load_dotenv
import ee
import os

def initialize_ee_with_credentials():
    """Initialize Earth Engine with service account credentials if available, otherwise default."""
    load_dotenv()
    try:
        credentials = ee.ServiceAccountCredentials(
            os.getenv('GEE_SERVICE_ACCOUNT', ''),
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
        )
        ee.Initialize(project='earthindex', credentials=credentials)
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}, defaulting to ")
        ee.Initialize(project='earthindex')


def get_s2_rgb_median(aoi: ee.Geometry,
                      start_date: str = '2023-01-01',
                      end_date: str = '2024-12-31',
                      clear_threshold: float = 0.80,
                      scale_factor: float = 1) -> ee.Image:
    """Get median RGB composite from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        scale_factor: Factor to divide image values by
        
    Returns:
        Median RGB composite as Earth Engine Image with bands B4 (R), B3 (G), B2 (B)
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    QA_BAND = 'cs_cdf'
    filtered_collection = s2.filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .linkCollection(csPlus, [QA_BAND]) \
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(clear_threshold))) \
        
    return filtered_collection.select(['B4', 'B3', 'B2']).median().divide(scale_factor)


def get_s2_hsv_median(aoi: ee.Geometry,
                      start_date: str = '2023-01-01',
                      end_date: str = '2024-12-31',
                      clear_threshold: float = 0.80) -> ee.Image:
    """Get median HSV composite from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        
    Returns:
        Median HSV composite as Earth Engine Image with bands hue, saturation, value
    """
    rgb_median = get_s2_rgb_median(aoi, start_date, end_date, clear_threshold).divide(10000)
    hsv_median = ee.Image.rgbToHsv(rgb_median)
    return hsv_median.select(['hue', 'saturation', 'value'])
    

def get_ee_image_url(image: ee.Image, vis_params: dict) -> str:
    """Get tile URL for displaying Earth Engine image in web map.
    
    Args:
        image: Earth Engine Image to display
        vis_params: Dictionary of visualization parameters
        
    Returns:
        URL template string for map tiles
    """
    map_id = image.getMapId(vis_params)
    return map_id['tile_fetcher'].url_format


def load_gcs_rgb_composites(mgrs_ids: list, imagery: str,
                            start_date: str, end_date: str) -> ee.Image:
    """Load and mosaic RGB composite GeoTIFFs from Google Cloud Storage.
    This tends to be faster than loading from EE, but these are currently generated 
    using s2cloudless, not the newer cloud score plus.
    
    Args:
        mgrs_ids: List of MGRS tile IDs
        imagery: Name of imagery product
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Mosaiced Earth Engine Image
    """
    print("Loading RGB composites")
    images = []
    for mgrs_id in mgrs_ids:
        image = ee.Image.loadGeoTIFF(
            f'gs://ei-imagery/{imagery}/{start_date}_{end_date}/{imagery}_{mgrs_id}_{start_date}_{end_date}_RGB.tif')
        images.append(image)

    image_collection = ee.ImageCollection(images)
    return image_collection.mosaic()


def get_planet_rgb_median(aoi: ee.Geometry,
                          start_date: str = '2023-01-01',
                          end_date: str = '2024-12-31') -> ee.Image:
    """Get median RGB composite from Planet NICFI basemaps.
    
    Args:
        aoi: Earth Engine Geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format  
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Median RGB composite as Earth Engine Image with bands R, G, B
    """
    collection = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/asia") \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi)
    
    return collection.median().select(['R', 'G', 'B'])


def get_planet_hsv_median(aoi: ee.Geometry,
                         start_date: str = '2023-01-01',
                         end_date: str = '2024-12-31') -> ee.Image:
    """Get median HSV from Planet NICFI basemaps.
    
    Args:
        aoi: Earth Engine Geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Median HSV as Earth Engine Image with bands hue, saturation, value
    """
    rgb = get_planet_rgb_median(aoi, start_date, end_date).divide(10000)
    hsv = rgb.rgbToHsv()
    return hsv.rename(['hue', 'saturation', 'value'])


def get_planet_ndvi_median(aoi: ee.Geometry,
                          start_date: str = '2023-01-01',
                          end_date: str = '2024-12-31') -> ee.Image:
    """Get median NDVI from Planet NICFI basemaps.
    
    Args:
        aoi: Earth Engine Geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Median NDVI as Earth Engine Image
    """
    collection = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/asia") \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi)
    
    median = collection.median()
    ndvi = median.normalizedDifference(['N', 'R']).rename('NDVI')
    return ndvi