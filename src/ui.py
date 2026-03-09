"""Map labeling interface and training utility functions for
machine learning on top of satellite foundation model embeddings."""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import ipyleaflet as ipyl
from ipyleaflet import Map, DrawControl, GeoJSON
from IPython.display import display
from ipywidgets import Button, HTML, Layout, VBox, HBox
import pandas as pd
from shapely.geometry import Point

warnings.simplefilter("ignore", category=FutureWarning)

# Get API keys from environment variables
MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')
if not MAPTILER_API_KEY:
    MAPTILER_API_KEY = 'YOUR_MAPTILER_API_KEY'
    warnings.warn("MAPTILER_API_KEY environment variable not set. Using placeholder. Please set it for full functionality.")

MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')
if not MAPBOX_ACCESS_TOKEN:
    MAPBOX_ACCESS_TOKEN = 'YOUR_MAPBOX_ACCESS_TOKEN'
    warnings.warn("MAPBOX_ACCESS_TOKEN environment variable not set. Using placeholder. Please set it for full functionality.")

DEFAULT_BASEMAP_TILES = {
    'MAPTILER': f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
    'GOOGLE_HYBRID': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    'MAPBOX': f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}.png?access_token={MAPBOX_ACCESS_TOKEN}"
}

DEFAULT_BASEMAP_ATTRIBUTIONS = {
    'MAPTILER': '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>',
    'GOOGLE_HYBRID': '© Airbus, Landsat, Copernicus, Maxar; Map data © Google',
    'MAPBOX': '<a href="https://www.mapbox.com/" target="_blank">&copy; Mapbox</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>',
}

class GeoLabeler:
    """Interactive Leaflet map for labeling geographic features relative to
    satellite image embedding tiles.

    Attributes:
        gdf: GeoDataFrame (centroids or full embedding rows) with geometry.
        map: ipyleaflet Map widget.
        save_dir: Directory for saved GeoJSON files (default: cwd).
        basemap_tiles: Dict of basemap name -> tile URL, copied from
            DEFAULT_BASEMAP_TILES at init. add_ee_basemaps() appends to this.
        basemap_attributions: Dict of basemap name -> attribution string.
        current_basemap: Key into basemap_tiles for the active basemap.
        basemap_layer: The TileLayer shown on the map.
        pos_indices, neg_indices: Lists of iloc positions for labeled points.
        pos_layer, neg_layer, erase_layer, points: GeoJSON map layers.
        select_val: 1/0/-100/2 for pos/neg/erase/Google Maps mode.
        detection_gdf: Optional; set by notebook for lasso selection over
            search results.
    """

    def __init__(
            self, gdf, geojson_path, custom_baselayer_url=None,
            custom_attribution=None, save_dir=None):
        print("Initializing GeoLabeler...")
        self.gdf = gdf.copy()
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.basemap_tiles = dict(DEFAULT_BASEMAP_TILES)
        self.basemap_attributions = dict(DEFAULT_BASEMAP_ATTRIBUTIONS)
        if custom_baselayer_url is not None:
            self.basemap_tiles['CUSTOM'] = custom_baselayer_url
            self.current_basemap = 'CUSTOM'
            initial_basemap_url = custom_baselayer_url
        else:
            self.current_basemap = 'GOOGLE_HYBRID'
            initial_basemap_url = self.basemap_tiles['GOOGLE_HYBRID']
        self._custom_attribution = custom_attribution
        attribution = self.basemap_attributions.get(self.current_basemap) or self._custom_attribution or ''
        self.basemap_layer = ipyl.TileLayer(
            url=initial_basemap_url, no_wrap=True, name='basemap',
            attribution=attribution)
        cen = gdf.geometry.unary_union.centroid
        self.map = Map(
            basemap=self.basemap_layer,
            center=(cen.y, cen.x), zoom=7, layout={'height': '600px'},
            scroll_wheel_zoom=True, attribution_control=False)
        # Single attribution box that we keep in sync when toggling basemaps
        self._attribution_html = HTML(
            value=f'<div style="font-size: 10px; color: #333;">{attribution}</div>',
            layout=Layout(margin='0', padding='2px 4px'))
        self.map.add(ipyl.WidgetControl(widget=self._attribution_html, position='bottomright'))

        print("Adding controls...")
        self.pos_button = Button(description='Positive')
        self.neg_button = Button(description='Negative')
        self.erase_button = Button(description='Erase')
        self.google_maps_button = Button(description='Google Maps')
        self.toggle_mode_button = Button(description='Toggle Lasso Mode')
        self.toggle_basemap_button = Button(description=f'Basemap: {self.current_basemap}')
        self.save_button = Button(description='Save Dataset')
        self.pos_button.on_click(self.pos_click)
        self.neg_button.on_click(self.neg_click)
        self.erase_button.on_click(self.erase_click)
        self.google_maps_button.on_click(self.google_maps_click)
        self.toggle_mode_button.on_click(self.toggle_mode)
        self.toggle_basemap_button.on_click(self.toggle_basemap)
        self.save_button.on_click(self.save_dataset)
        self.map.on_interaction(self.label_point)
        self.execute_label_point = True
        self.select_val = -100  # Initialize to _erase_
        self.pos_indices = []
        self.neg_indices = []
        self.detection_gdf = None
        self.lasso_mode = False
        
        geojson_path_str = geojson_path if isinstance(geojson_path, (str, bytes)) else str(geojson_path)
        with open(geojson_path_str) as f:
            region_data = json.load(f)
        region_layer = ipyl.GeoJSON(
            name="region",
            data=region_data,
            style={
                'color': '#FFFFFF',
                'weight': 2,
                'opacity': 1,
                'fillOpacity': 0,
            },
        )
        self.map.add_layer(region_layer)
        # Fit initial viewport to the boundary GeoJSON
        boundary_gdf = gpd.read_file(geojson_path_str)
        (minx, miny, maxx, maxy) = boundary_gdf.total_bounds
        self.map.fit_bounds([[miny, minx], [maxy, maxx]])


        # layer to contain positive labeled points
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'green',
                'radius': 3,
                'fillColor': '#00FF00',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.pos_layer)

        # layer to contain negative labeled points
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'red',
                'radius': 3,
                'fillColor': '#FF0000',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.neg_layer)

        # erased points
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'white',
                'radius': 3,
                'fillColor': '#000000',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.erase_layer)
        
        # generic points layer for visualization
        self.points = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'black',
                'radius': 3,
                'fillColor': '#ffe014',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            },
            hover_style={
                'fillColor': '#ffe014',
                'fillOpacity': 0.5
            }
        )
        self.map.add_layer(self.points)
        
        # Add DrawControl for lasso selection
        self.draw_control = DrawControl(
            polygon={"shapeOptions": {"color": "#6be5c3", "fillOpacity": 0.5}},
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        self.draw_control.polygon = {"shapeOptions": {"color": "#6be5c3"}}
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        self.draw_control.clear()

        display(VBox([
            self.map, 
            HBox([
                self.pos_button, 
                self.neg_button, 
                self.erase_button,
                self.google_maps_button,
                self.toggle_mode_button,
                self.toggle_basemap_button,
                self.save_button
            ])
        ]))
        
    def pos_click(self, b):
        self.select_val = 1

    def neg_click(self, b):
        self.select_val = 0

    def erase_click(self, b):
        self.select_val = -100
        
    def google_maps_click(self, b):
        self.select_val = 2
        # Force single point mode when google maps is selected
        self.lasso_mode = False
        self.toggle_mode_button.description = 'Toggle Lasso Mode'
        self.draw_control.polygon = {}
        self.draw_control.clear()

    def toggle_mode(self, b):
        prev_select_val = self.select_val
        self.lasso_mode = not self.lasso_mode
        if self.lasso_mode:
            self.toggle_mode_button.description = 'Toggle Single Point Mode'
            self.draw_control.polygon = {"shapeOptions": {"color": "#6be5c3"}}
            # Restore previous selection mode
            self.select_val = prev_select_val
        else:
            self.toggle_mode_button.description = 'Toggle Lasso Mode'
            self.draw_control.polygon = {}
        self.draw_control.clear()

    def toggle_basemap(self, b):
        basemap_keys = list(self.basemap_tiles.keys())
        current_idx = basemap_keys.index(self.current_basemap)
        next_idx = (current_idx + 1) % len(basemap_keys)
        self.current_basemap = basemap_keys[next_idx]

        self.basemap_layer.url = self.basemap_tiles[self.current_basemap]
        attr = self.basemap_attributions.get(self.current_basemap, self._custom_attribution or '')
        self.basemap_layer.attribution = attr
        self._attribution_html.value = f'<div style="font-size: 10px; color: #333;">{attr}</div>'
        self.toggle_basemap_button.description = f'Basemap: {self.current_basemap}'

    def save_dataset(self, b=None):
        """Save positive and negative points to GeoJSON in save_dir.

        Call from the UI via the Save Dataset button (b is the button) or from the
        notebook as labeler.save_dataset().
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save positive points (pos_indices are iloc positions)
        if self.pos_indices:
            pos_gdf = self.gdf.iloc[self.pos_indices][["geometry"]].copy()
            path = self.save_dir / f"positive_points_{timestamp}.geojson"
            pos_gdf.to_file(path, driver="GeoJSON")
            print(f"Saved positive points to {path}")
        else:
            print("No positive points to save")

        # Save negative points (neg_indices are iloc positions)
        if self.neg_indices:
            neg_gdf = self.gdf.iloc[self.neg_indices][["geometry"]].copy()
            path = self.save_dir / f"negative_points_{timestamp}.geojson"
            neg_gdf.to_file(path, driver="GeoJSON")
            print(f"Saved negative points to {path}")
        else:
            print("No negative points to save")

    def handle_draw(self, target, action, geo_json):
        if action != 'created':
            return
        self.polygon = gpd.GeoDataFrame.from_features([geo_json])

        # Convert the GeoJSON layer data to a GeoDataFrame
        self.points_gdf = gpd.GeoDataFrame.from_features(self.points.data['features'])
        
        self.points_inside = self.detection_gdf[
            self.detection_gdf.geometry.within(self.polygon.geometry.iloc[0])]
        
        print(self.points_inside)
        for idx in self.points_inside.index:
            if idx in self.pos_indices:
                self.pos_indices.remove(idx)
            if idx in self.neg_indices:
                self.neg_indices.remove(idx)
            
            if self.select_val == 1:
                self.pos_indices.append(idx)
            elif self.select_val == 0:
                self.neg_indices.append(idx)
        
        self.update_layers()
        self.draw_control.clear()

    def update_layers(self):
        self.pos_layer.data = json.loads(self.gdf.iloc[self.pos_indices][["geometry"]].to_json())
        self.neg_layer.data = json.loads(self.gdf.iloc[self.neg_indices][["geometry"]].to_json())

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        if not self.execute_label_point or self.lasso_mode:
            return
        
        action = kwargs.get('type') 
        if action not in ['click']:
            return
                 
        # find the closest point in the dataframe to the clicked point
        lat, lon = kwargs.get('coordinates')
        
        if self.select_val == 2:
            import webbrowser
            url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            webbrowser.open(url, new=2, autoraise=True)
            # print(f"Please open this URL in your local browser: {url}")
            return
        idx = self.gdf.sindex.nearest(Point(lon, lat))[1][0]
        
        if idx in self.pos_indices:
            self.pos_indices.remove(idx)
        if idx in self.neg_indices:
            self.neg_indices.remove(idx)
                
        if self.select_val == 1:
            self.pos_indices.append(idx)
            self.pos_layer.data = json.loads(self.gdf.iloc[self.pos_indices][["geometry"]].to_json())
        elif self.select_val == 0:
            self.neg_indices.append(idx)
            self.neg_layer.data = json.loads(self.gdf.iloc[self.neg_indices][["geometry"]].to_json())
        else:
            self.erase_layer.data = json.loads(self.gdf.iloc[[idx]][["geometry"]].to_json())
            self.pos_layer.data = json.loads(self.gdf.iloc[self.pos_indices][["geometry"]].to_json())
            self.neg_layer.data = json.loads(self.gdf.iloc[self.neg_indices][["geometry"]].to_json())

    def update_layer(self, layer, new_data):
        """Add points to the map for visualization, without changing labels."""
        self.execute_label_point = False
        layer.data = new_data
        self.execute_label_point = True

    def add_ee_basemaps(self, geojson_path, start_date, end_date):
        """Add Earth Engine HSV and RGB median basemaps to this labeler's
        basemap toggle. The current basemap is left unchanged. Triggers EE
        initialization and authentication.

        Usage:
            labeler = GeoLabeler(gdf, geojson_path, ...)
            labeler.add_ee_basemaps(geojson_path, start_date, end_date)
        """
        import shapely
        import ee
        from gee import (
            get_s2_hsv_median,
            get_s2_rgb_median,
            get_ee_image_url,
            initialize_ee_with_credentials,
        )
        initialize_ee_with_credentials()
        boundary = gpd.read_file(geojson_path).geometry.iloc[0]
        ee_boundary = ee.Geometry(shapely.geometry.mapping(boundary))
        hsv_median = get_s2_hsv_median(ee_boundary, start_date, end_date)
        hsv_url = get_ee_image_url(hsv_median, {
            'min': [0, 0, 0], 'max': [1, 1, 1],
            'bands': ['hue', 'saturation', 'value']})
        self.basemap_tiles['HSV_MEDIAN'] = hsv_url
        self.basemap_attributions['HSV_MEDIAN'] = '© Copernicus via Earth Engine'
        rgb_median = get_s2_rgb_median(
            ee_boundary, start_date, end_date, scale_factor=10000)
        rgb_url = get_ee_image_url(rgb_median, {
            'min': [0, 0, 0], 'max': [0.25, 0.25, 0.25],
            'bands': ['B4', 'B3', 'B2']})
        self.basemap_tiles['RGB_MEDIAN'] = rgb_url
        self.basemap_attributions['RGB_MEDIAN'] = '© Copernicus via Earth Engine'
