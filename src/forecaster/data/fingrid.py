import os
import time
import requests
import numpy as np
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv

from forecaster.utils import get_project_root

# Forecasts and the IDs in Fingrid's API
EXTERNAL_FEATURES = {
    '246': 'wind_power',
    '247': 'solar_power',
    '165': 'consumption',
    '242': 'production'
}


def getFeatureDictFromKeys(keys):
    # Identify any keys that are not in the original dictionary
    missing_keys = [key for key in keys if key not in EXTERNAL_FEATURES]
    
    # Assert that all requested keys are present, with a detailed message if not
    assert not missing_keys, (
        f"The following IDs are not available as features: {missing_keys}. "
        f"Available feature IDs are: {list(EXTERNAL_FEATURES.keys())}"
    )
    return {key: EXTERNAL_FEATURES[key] for key in keys if key in EXTERNAL_FEATURES}

def load_fingrid_data(dataset_id):

    project_root_path = get_project_root()
    data = pd.read_csv(f'{project_root_path}/data/{dataset_id}.csv')

    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(data.iloc[0, 0])
    dt_end = datetime.fromisoformat(data.iloc[-1, 0])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    # Convert the timestamps to datetime index for the dataframe
    data.index = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')
    # Drop the original startTIme column
    data.drop(columns='startTime', inplace=True)

    return data

def fetch_dataset_shorts():
    """
    Fetches short descriptions of all datasets available in the Fingrid API.

    Returns:
    - list: Contains the short descriptions of the datasets.
    """

    params = {'pageSize': 20000, 'orderBy': 'id'}
    end_point = "datasets"

    all_data = fetch_data(end_point, params)
    return all_data


def print_dataset_shorts(dataset_shorts):
    """
    Prints the short descriptions of the datasets available in the Fingrid API.

    Args:
    - dataset_shorts (list): Contains the short descriptions of the datasets.

    Returns:
    - None
    """

    for dataset in dataset_shorts:
        print(f"{dataset['id']} - {dataset['nameEn']} ({dataset['dataPeriodEn']})")
        print("Description:", dataset['descriptionEn'])
        print("-"*20)


def fetch_dataset_data(dataset_id, start_time, end_time):
    """
    Fetches data from the Fingrid API for the specified dataset and time range.

    Args:
    - dataset_id (str): ID of the dataset to fetch data from.
    - start_time (str): Start time for the data in the format 'YYYY-MM-DDTHH:MM:SSZ'.
    - end_time (str): End time for the data in the format 'YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
    - list: Contains the data fetched from the API.
    """

    params = {
        'startTime': start_time, 
        'endTime': end_time, 
        'format': 'json', 
        'oneRowPerTimePeriod': 'true', 
        'pageSize': 20000, 
        'locale': 'en', 
        'sortBy': 'startTime', 
        'sortOrder': 'asc'
        }
    end_point = f"datasets/{dataset_id}/data"

    all_data = fetch_data(end_point, params)
    return all_data


def fetch_data(end_point, params):
    """
    Fetches data from the Fingrid API using the specified endpoint and parameters with a persistent session.

    Args:
    - end_point (str): Endpoint for the API.
    - params (dict): Parameters to be passed to the API.

    Returns:
    - list: Contains the data fetched from the API.
    """

    # Set your API key here
    load_dotenv()
    API_KEY = os.getenv("FINGRID_API_KEY")

    # Base URL for Fingrid API
    BASE_URL = "https://data.fingrid.fi/api"

    headers = {
        'x-api-key': API_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        "Accept-Encoding": "*",
        "Connection": "keep-alive",
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }

    all_data = []
    session = requests.Session()
    session.headers.update(headers)

    # Fetch the first page to check the pagination info
    response = session.get(f"{BASE_URL}/{end_point}", params=params)
    response.raise_for_status()
    data = response.json()
    print(data['pagination'], data.keys())
    all_data.extend(data['data'])

    # Determine total number of pages
    total_pages = data['pagination'].get('lastPage', 1)

    # Loop through all pages if more than one
    for page in range(2, total_pages + 1):
        print("Sleeding for 6 seconds to avoid rate limiting")
        time.sleep(6)  # sleep for 6 second to avoid rate limiting

        params['page'] = page
        response = session.get(f"{BASE_URL}/{end_point}", params=params)
        response.raise_for_status()
        data = response.json()
        print(data['pagination'], data.keys())
        all_data.extend(data['data'])
        
    return all_data