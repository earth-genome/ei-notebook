from dotenv import load_dotenv
load_dotenv()

import os
import requests
import json
import planet
import os
import copy
import asyncio
import math
import geopandas as gpd
import argparse
import logging
from datetime import datetime

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PL_API_KEY = os.getenv('PL_API_KEY')
GCP_CREDENTIALS = os.getenv('B64_GCP_CREDENTIALS')
auth = planet.Auth.from_key(PL_API_KEY)

BASEMAP_API_URL = 'https://api.planet.com/basemaps/v1/mosaics'

session = requests.Session()
session.auth = (PL_API_KEY, "")

response = requests.get(BASEMAP_API_URL, auth=session.auth)


def handle_pagination(session, url, params, key='items'):
    """
    Handle paginated URLs by making multiple requests and yielding individual items.

    Parameters:
        session (requests.Session): The session object to make HTTP requests.
        url (str): The base URL for the paginated resource.
        params (dict): Query parameters to be sent with the request.
        key (str, optional): The key in the response JSON containing the list of items. Defaults to 'items'.

    Yields:
        dict: Individual items from the paginated resource.

    Raises:
        requests.HTTPError: If any HTTP errors occur during the requests.

    """
    while True:
        # Make a GET request to the specified URL with the given parameters
        response = session.get(url, params=params)

        # Raise an exception if the response has an HTTP error status code
        response.raise_for_status()

        # Parse the response body as JSON
        body = response.json()

        # Iterate over each item in the 'key' list of the response body and yield it
        for item in body[key]:
            yield item

        # Check if there is a next page link in the response body
        if '_next' in body['_links']:
            # Update the URL to the next page URL
            url = body['_links']['_next']
        else:
            # If there is no next page link, break the loop and stop pagination
            break


def polygon_search(mosaic_name, geometry):
    """Searches for quad ID's within a polygon geometry using the Planet Basemaps API.

    Parameters:
        mosaic_name (str): The name of the mosaic to search within.
        geometry (dict): The polygon geometry to search with.

    Yields:
        dict: The quad IDs found within the polygon geometry.

    Raises:
        requests.exceptions.HTTPError: If any HTTP error occurs during the API requests.

    """
    base_url = 'https://api.planet.com/basemaps/v1'

    # Configure retry logic for handling rate limiting (status code 429)
    retries = Retry(total=5, backoff_factor=0.2, status_forcelist=[429])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Retrieve the mosaic ID from the mosaic name
    rv = session.get(f'{base_url}/mosaics', params={'name__is': mosaic_name})
    rv.raise_for_status()
    logging.info(f"Found mosaic {mosaic_name} with ID {rv.json()}")
    mosaics = rv.json()['mosaics']
    if not mosaics:
        logging.warning(f"No mosaic found for name {mosaic_name}")
        return None
    mosaic_id = mosaics[0]['id']

    url = None
    while True:
        if url is None:
            # Initial request to search for quads within the mosaic
            url = f'{base_url}/mosaics/{mosaic_id}/quads/search'
            rv = session.post(url, json=geometry)
        else:
            # Request subsequent pages of quad search results
            rv = session.get(url)
        rv.raise_for_status()
        response = rv.json()

        # Yield item information for each result item
        for item in response['items']:
            yield item

        # Check if there are more pages of results
        if '_next' in response['_links']:
            url = response['_links']['_next']
        else:
            break

async def create_and_deliver_order(order_params, client):
    '''Create an order and wait for it to delivered

    Parameters:
        order_params: An order request
        client: An Order client object
    '''
    with planet.reporting.StateBar(state='creating') as reporter:
        # Place an order to the Orders API
        order = await client.create_order(order_params)
        reporter.update(state='created', order_id=order['id'])
        # Wait while the order is being completed
        await client.wait(order['id'], callback=reporter.update_state,
                          max_attempts=0)
        
async def batch_lists_and_place_orders(quad_ids, order_params):
    """
    Process quad IDs in batches and create orders.

    Parameters:
        quad_ids (list): A list of quad IDs to be processed in batches.
        order_params (dict): The order parameters dictionary that contains the details of the order.

    """

    # Calculate the number of batches
    num_batches = math.ceil(len(quad_ids) / 100)

    # Create batched quad IDs lists
    batched_quad_ids = [
        quad_ids[i:i + 100] for i in range(0, len(quad_ids), 100)
    ]

    # Duplicate the order_params dictionary for each batch
    all_order_params = [
        copy.deepcopy(order_params) for _ in range(num_batches)
    ]

    # Assign batched quad IDs to each order_params dictionary
    for i, params in enumerate(all_order_params):
        params['products'][0]['quad_ids'] = batched_quad_ids[i]

    async with planet.Session() as ps:
        # The Orders API client
        client = ps.client('orders')

        # Create the order and deliver it to GCP for each batch
        await asyncio.gather(*[
            create_and_deliver_order(params, client) 
            for params in all_order_params
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Download Planet Basemaps to GEE')
    parser.add_argument('--aoi_file', type=str, required=True,
                        help='Path to AOI GeoJSON file')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date in YYYY-MM format')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date in YYYY-MM format')
    parser.add_argument('--gee_collection', type=str, required=True,
                        help='GEE collection path for output')
    return parser.parse_args()


def get_mosaic_name(date_str):
    return f"ps_monthly_sen2_normalized_analytic_8b_sr_subscription_{date_str}_mosaic"


def get_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m')
    end = datetime.strptime(end_date, '%Y-%m')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y_%m'))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return dates


def main():
    args = parse_args()
    
    # Load AOI
    aoi = gpd.read_file(args.aoi_file)
    geometry = eval(aoi.to_json())['features'][0]['geometry']
    
    # Get list of dates to process
    dates = get_date_range(args.start_date, args.end_date)
    
    for date in dates:
        mosaic_name = get_mosaic_name(date)
        logging.info(f"Processing {mosaic_name}...")
        quad_ids = []
        
        # Search for quad IDs
        for quad in polygon_search(mosaic_name, geometry):
            quad_ids.append(quad['id'])
            
        if not quad_ids:
            logging.warning(f"No quads found for {date}, skipping...")
            continue
            
        order_params = {
            "name": f"basemap_order_{date}",
            "source_type": "basemaps",
            "products": [
                {
                    "mosaic_name": mosaic_name,
                    "quad_ids": quad_ids
                }
            ],
            "delivery": {
                "google_earth_engine": {
                    "project": "earthindex",
                    "collection": args.gee_collection,
                    "credentials": GCP_CREDENTIALS,
                }
            }
        }
        
        asyncio.run(batch_lists_and_place_orders(quad_ids, order_params))


if __name__ == "__main__":
    main()


