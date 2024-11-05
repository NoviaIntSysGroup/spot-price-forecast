
import pytz
import pandas as pd
from datetime import datetime


def date_to_timestamp(iso_datetime):
    """It converts a date (iso format, Helsinki) to a timestamp."""
    # Define the timezone (making sure we are dealing with Helsinki time)
    tz = pytz.timezone('Europe/Helsinki')
    # Convert the date to a datetime object
    dt = datetime.fromisoformat(iso_datetime)
    # Localize the datetime to the specified timezone
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    # Convert the datetime to a timestamp
    timestamp = dt.timestamp()
    return int(timestamp)

def timestamp_to_date(timestamp):
    """It converts a timestamp to a datetime (iso format, Helsinki)."""
    # Define the timezone (making sure we are dealing with Helsinki time)
    tz = pytz.timezone('Europe/Helsinki')
    return datetime.fromtimestamp(timestamp, tz)

def get_hours_in_date_interval(start_date, end_date):
    """It returns the number of hours in an interval."""
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)
    return int((end_timestamp - start_timestamp) / 3600)

def get_hourly_datetimes_for_date_interval(start_date, end_date):
    """It returns a list of datetimes for an interval."""
    n_hours = get_hours_in_date_interval(start_date, end_date) + 1
    start_ts = date_to_timestamp(start_date)
    datetimes = [timestamp_to_date(start_ts + 3600 * i) for i in range(n_hours)]
    return datetimes

def load_spot_data(filepath, date_col, price_col):
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
    spot_price = pd.read_csv(filepath, decimal=',')

    # Change the original dates to datetimes with time zone and daylight saving
    # The originals are in European/Helsinki time zone. So we are simply
    # adding that information explicitly.
    spot_price_dt = get_hourly_datetimes_for_date_interval(
        spot_price.iloc[0, 0],
        spot_price.iloc[-1, 0]
    )
    spot_price[date_col] = spot_price_dt


    # Replace the hours with a bidding error to be 0 €/MWh instead of -500 €/MWh
    # https://yle.fi/a/74-20061943
    specific_datetime = pd.to_datetime('2023-11-24 15:00+02:00')
    matching_row_index = spot_price.index[spot_price[date_col] == specific_datetime]
    spot_price.loc[matching_row_index[0]:matching_row_index[0]+10, price_col] = 0
    
    # Set the date column as the index
    spot_price.set_index(date_col, inplace=True)
    
    # Handling numeric columns (removing spaces and converting to float)
    spot_price[price_col] = spot_price[price_col].apply(lambda x: float(str(x).replace(' ', '')))

    # Rename columns to comply with statsforecast library
    spot_price.rename(columns={price_col: 'y'}, inplace=True)

    # Sort data by index
    spot_price.sort_index(inplace=True)
    
    return spot_price[['y']]