import numpy as np
import pandas as pd

from forecaster.data import fingrid

def create_daily_lag_features(df, column_name="y", lags=[1], average=False):
    """
    Creates lagged features for the given column at a daily level.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the features.
    - column_name (str): Name of the column for which to create lagged features.
    - lags (list): List of lags to create.
    - average (bool): Whether to create daily average lagged features.

    Returns:
    - pd.DataFrame: Timeseries DataFrame with the lagged features added.
    """
    
    # Precompute the hour index
    hour_idx = df.index.hour

    # Iterate over the lags
    for lag in lags:
        # Create the lagged daily data in a vectorized way
        lagged_data = df[column_name].shift(24 * lag)
        
        if average:
            # Calculate the average across the 24 hours in a vectorized way
            daily_avg_lag = lagged_data.groupby(lagged_data.index.date).transform('mean')
            df[f'{column_name}_lag_avg_{lag}'] = daily_avg_lag
        else:
            # Initialize a DataFrame to hold lagged hourly data
            lagged_hourly_df = pd.DataFrame(index=df.index)

            # Vectorized assignment of lagged data by hour
            for hour in range(24):
                # Create a mask for the specific hour
                mask = (hour_idx == hour)
                # Extract lagged values for the hour
                lagged_hourly_data = lagged_data[mask]
                # Group by date to align with the original's pivot logic
                lagged_hourly_column = lagged_hourly_data.groupby(lagged_hourly_data.index.date).first()

                # Map the values back to the DataFrame, filling NaNs where no lagged data is available
                lagged_hourly_df[f'{column_name}_lag_{lag}_h{hour}'] = df.index.map(
                    lambda x: lagged_hourly_column.loc[x.date()] if x.date() in lagged_hourly_column.index else np.nan
                )

            # Assign the lagged hourly columns back to the original DataFrame
            df = pd.concat([df, lagged_hourly_df], axis=1)
    
    return df

def extract_time_features(df):
    """
    Extracts time features (dummy variables) for weekend and weekdays from the datetime index of the dataframe.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the target variable.

    Returns:
    - pd.DataFrame: Timeseries DataFrame with the time features added.
    """
    # Check if the index of the dataframe is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The index of the dataframe must be a DatetimeIndex.")

    # Initialize all feature columns to zero
    for hour in range(24):
        df[f'weekday_hour_{hour}'] = 0
        df[f'weekend_hour_{hour}'] = 0
    
    # Set values for weekday and weekend hours
    for hour in range(24):
        # Weekday columns (Monday=0, ..., Friday=4)
        df.loc[(df.index.weekday < 5) & (df.index.hour == hour), f'weekday_hour_{hour}'] = 1
        
        # Weekend columns (Saturday=5, Sunday=6)
        df.loc[(df.index.weekday >= 5) & (df.index.hour == hour), f'weekend_hour_{hour}'] = 1
    
    return df

def add_external_features(df, features_to_add):
    """
    Adds external features to the dataset by loading the data from the data folder and merging it with the existing dataset.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the features and target variable.
    - features_to_add (dict): Dictionary with the external feature ids as keys and names as values

    Returns:
    - pd.DataFrame: Timeseries DataFrame with the external features added.
    """

    for feature_key in features_to_add.keys():
        # Load the external feature data
        data = fingrid.load_fingrid_data(feature_key)
        data.columns.values[0] = features_to_add[feature_key]
        # Merge the data with the existing DataFrame
        df = df.merge(data, how='left', left_index=True, right_index=True)
        # Fill missing values with 0, the external data should not contain NaN values but
        # it might not cover the same time period as the original data, in which case
        # the missing values get NaN values that need to be filled
        df = df.fillna(0)

    return df