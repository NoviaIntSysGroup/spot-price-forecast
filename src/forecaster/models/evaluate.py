from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

def calculate_metrics(y_test, y_pred):
    """
    Calculates the mean absolute error and the root mean squared error.

    Args:
    - y_test (np.array): True target values.
    - y_pred (np.array): Predicted target values.

    Returns:
    - dict: Contains the mean absolute error and the root mean squared error.
    """
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Root Mean Squared Error
    rmse = root_mean_squared_error(y_test, y_pred)

    # Return a dictionary with all three metrics
    return {
        'mean_absolute_error': mae,
        'root_mean_squared_error': rmse
    }

def calculate_price_extremes(df, price_column='y', date_column=None, top_k=1):
    """
    Calculates the top-k hours with the maximum and minimum prices for each day in the dataset.
    Works with both a datetime index or a separate date column.

    Parameters:
    - df: pandas DataFrame containing the data
    - price_column: str, name of the column containing the price data (default is 'y')
    - date_column: str, name of the column containing the date data, if not using the index (default is None)
    - top_k: int, number of top maximum and minimum price hours to return (default is 1)

    Returns:
    - daily_extremes: pandas DataFrame with columns 'day', 'max_price_hours', 'min_price_hours', and 'year'
    """
    df_copy = df.copy()
    
    # Handle datetime conversion
    if date_column:
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy['day'] = df_copy[date_column].dt.date
        df_copy['hour'] = df_copy[date_column].dt.hour
    else:
        df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['day'] = df_copy.index.date
        df_copy['hour'] = df_copy.index.hour
    
    # Extract year
    df_copy['year'] = pd.to_datetime(df_copy['day']).dt.year
    
    # Sort data to allow easy nlargest/nsmallest calculation
    df_copy.sort_values(by=['day', price_column], ascending=[True, False], inplace=True)
    
    # Get top-k max prices per day
    max_price_hours = df_copy.groupby('day').head(top_k).groupby('day')['hour'].apply(list)
    
    # Sort data to get min prices
    df_copy.sort_values(by=['day', price_column], ascending=[True, True], inplace=True)
    
    # Get top-k min prices per day
    min_price_hours = df_copy.groupby('day').head(top_k).groupby('day')['hour'].apply(list)
    
    # Combine results into a DataFrame
    daily_extremes = pd.DataFrame({
        'day': max_price_hours.index,
        'max_price_hours': max_price_hours.values,
        'min_price_hours': min_price_hours.reindex(max_price_hours.index).values,
        'year': pd.to_datetime(max_price_hours.index).year
    })

    return daily_extremes.reset_index(drop=True)


def calculate_prediction_accuracy(actual_extremes, predicted_extremes, year_on_year=False, order=False, top_k=1):
    """
    Calculates the percentage of correct predictions for top-k hours, either year-by-year or overall.

    Args:
    - actual_extremes (pd.DataFrame): DataFrame containing the actual extreme hours.
    - predicted_extremes (pd.DataFrame): DataFrame containing the predicted extreme hours.
    - year_on_year (bool): If True, calculate accuracy separately for each year. If False, calculate overall accuracy (default is False).
    - order (bool): If True, the order of the hours must match exactly (default is False).
    - top_k (int): Number of top hours to consider (default is 1).

    Returns:
    - accuracy_dict (dict): Dictionary containing the accuracy percentages for max and min hours.
        - Key: 'max' or 'min'
        - Value: Dictionary with keys as k values and values as accuracy percentages
    """
    accuracy_dict = {'max': {}, 'min': {}}
    
    # Merge actual and predicted data on 'day'
    merged_df = pd.merge(actual_extremes, predicted_extremes, on='day', suffixes=('_actual', '_predicted'))
    
    # Group by year if year_on_year is True
    if year_on_year:
        grouped = merged_df.groupby('year_actual')
    else:
        grouped = [(None, merged_df)]
    
    # Calculate accuracy for each group
    for year, group in grouped:
        # Calculate accuracy for each k value
        for k in range(1, top_k + 1):
            # Calculate accuracy for max and min hours
            for key in ['max', 'min']:
                # Get the actual and predicted columns
                actual_col = f'{key}_price_hours_actual'
                predicted_col = f'{key}_price_hours_predicted'
                
                if order:
                    # Compare the lists directly for ordered comparison
                    correct_predictions = (group[actual_col].str[:k] == group[predicted_col].str[:k]).sum()
                else:
                    # Convert to sets for unordered comparison
                    correct_predictions = (group[actual_col].str[:k].apply(set) == group[predicted_col].str[:k].apply(set)).sum()
                
                if year_on_year:
                    if year not in accuracy_dict[key]:
                        accuracy_dict[key][year] = {}
                    accuracy_dict[key][year][k] = (correct_predictions / len(group)) * 100
                else:
                    accuracy_dict[key][k] = (correct_predictions / len(group)) * 100

    return accuracy_dict
