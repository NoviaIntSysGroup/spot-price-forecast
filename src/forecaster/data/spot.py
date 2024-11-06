
import numpy as np
import pandas as pd
from datetime import datetime


def load_spot_data(filepath, date_col, price_col, raw=False):
    """
    Preprocesses the spot prices by loading it from a CSV file, converting the HourEUR column to a datetime format, and handling numeric columns.

    Args:
    - filepath (str): Path to the CSV file.
    - date_col (str): Name of the column containing the date.
    - price_col (str): Name of the column containing the target variable.

    Returns:
    - pd.DataFrame: Preprocessed spot prices.
    """
    # Load the data with the correct delimiter and decimal handling
    spot_data = pd.read_csv(filepath, decimal=',')

    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(spot_data.loc[spot_data.index[0], date_col])
    dt_end = datetime.fromisoformat(spot_data.loc[spot_data.index[-1], date_col])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    # Convert the timestamps to datetime index for the dataframe
    spot_data.index = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')
    # Sort data by index
    spot_data.sort_index(inplace=True)
    # Handling numeric columns (removing spaces and converting to float)
    spot_data[price_col] = spot_data[price_col].apply(lambda x: float(str(x).replace(' ', '')))

    if raw:
        return spot_data
    else:
        # Replace the hours with a bidding error to be 0 €/MWh instead of -500 €/MWh
        # https://yle.fi/a/74-20061943
        specific_datetime = pd.to_datetime('2023-11-24 15:00+02:00')
        spot_data.loc[specific_datetime:specific_datetime+pd.Timedelta(hours=10), price_col] = 0

        # Rename columns to comply with statsforecast library
        spot_data.rename(columns={price_col: 'y'}, inplace=True)

        return spot_data[['y']]