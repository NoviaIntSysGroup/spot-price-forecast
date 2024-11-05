import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


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


############################

def plot_spot_price_predictions(
    y_true, 
    y_pred, 
    true_label='True Spot Price', 
    pred_label='Prediction', 
    title='Spot Price Prediction', 
    xlabel='Time', 
    ylabel='Spot Price', 
    line_width=2, 
    tick_fontsize=16, 
    axis_fontsize=20, 
    rotation=45, 
    figsize=(14, 7)
):
    """
    Plots the true values and predictions for spot price.

    Parameters:
    - y_true: array-like, true values of the spot price.
    - y_pred: array-like, predicted values of the spot price.
    - true_label: str, label for the true values line.
    - pred_label: str, label for the predicted values line.
    - title: str, title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - line_width: float, width of the lines.
    - tick_fontsize: int, fontsize for ticks.
    - axis_fontsize: int, fontsize for axis labels.
    - rotation: int, rotation angle for x-axis ticks.
    - figsize: tuple, size of the figure.
    """

    # Create figure and plot space
    plt.figure(figsize=figsize)

    # Plotting true values and predictions
    plt.plot(y_true, label=true_label, color='#182D7C', linewidth=line_width)
    plt.plot(y_pred, label=pred_label, color='#990000', linewidth=line_width)

    # Adding titles and labels
    plt.title(title, fontsize=axis_fontsize + 2, pad=20)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)

    # Customize ticks and spines
    plt.xticks(fontsize=tick_fontsize, rotation=rotation)
    plt.yticks(fontsize=tick_fontsize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    # Increase tick width and length
    plt.gca().tick_params(width=2)
    plt.gca().tick_params(axis='x', length=10)
    plt.gca().tick_params(axis='y', length=10)

    # Adding a legend with a modern look
    plt.legend(frameon=False, fontsize=12, loc='best')

    # Show plot
    plt.tight_layout()
    plt.show()




def fill_missing_values(df, missing_mapping=None):
    """
    Fill missing values in the dataframe using the specified fill type.

    Args:
    - df (pd.DataFrame): The dataframe containing the missing values.
    - missing_mapping (dict): A dictionary mapping columns to the fill type. There can be multiple fill types in an array.

    Returns:
    - pd.DataFrame: The dataframe with missing values filled.
    """
    # Copy the dataframe to avoid modifying the original
    df = df.copy()

    # Define the missing mapping for all columns if not provided
    if missing_mapping is None:
        missing_mapping = { col: ['interpolate', 'ffill', 'bfill'] for col in df.columns if df[col].isnull().sum() > 0 }

    # Iterate over the missing mapping
    for col, fill_types in missing_mapping.items():
        # Check if the fill types is a list
        if not isinstance(fill_types, list):
            fill_types = [fill_types]
        # Iterate over the fill types
        for fill in fill_types:
            if fill not in ['interpolate', 'ffill', 'bfill']:
                raise ValueError(f"Invalid fill type '{fill}' provided. Use 'interpolate', 'ffill', or 'bfill'.")
            if df[col].isnull().sum() == 0:
                break
            if fill == 'interpolate':
                df[col] = df[col].interpolate()
            elif fill == 'ffill':
                df[col] = df[col].ffill()
            elif fill == 'bfill':
                df[col] = df[col].bfill()
    return df

def add_external_features(df, features_to_add):
    """
    Adds external features to the dataset by loading the data from the data folder and merging it with the existing dataset.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the features and target variable.
    - features_to_add (list): List of feature ids to be added to the dataset.

    Returns:
    - pd.DataFrame: Timeseries DataFrame with the external features added.
    """
    df_copy = df.copy()
    # load df from '../data/Elspotprices.csv'
    for feature in features_to_add:
        data = pd.read_csv(f'../data/{feature}.csv', index_col=0, parse_dates=True)
        
        # convert to Europe/Helsinki timezone
        data.index = data.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

        data = data.resample('h').sum()
        df_copy = df_copy.merge(data, how='left', left_index=True, right_index=True)

    return df_copy



def ts_split_train_test(df, test_size=0.2):
    """
    Splits the timeseries DataFrame into training and testing sets.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the features and target variable.
    - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    - tuple: Contains the training features, testing features, training target, and testing target.
    """
    
    X = df.drop(columns='y')
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test

def ts_split_train_test_by_date(df, train_start, train_end, test_start, test_end):
    """
    Splits the timeseries DataFrame into training and testing sets based on the specified dates.

    Args:
    - df (pd.DataFrame): Timeseries DataFrame containing the features and target variable.
    - train_start (str): Start date for the training set.
    - train_end (str): End date for the training set.
    - test_start (str): Start date for the testing set.
    - test_end (str): End date for the testing set.

    Returns:
    - tuple: Contains the training features, testing features, training target, and testing target.
    """
    
    X_train = df.loc[train_start:train_end].drop(columns='y')
    y_train = df.loc[train_start:train_end]['y']
    
    X_test = df.loc[test_start:test_end].drop(columns='y')
    y_test = df.loc[test_start:test_end]['y']

    return X_train, X_test, y_train, y_test





def plot_coefficients(coefficients, col_keyword):
    """
    Plots the coefficients of the linear regression model for weekday or weekend features

    Args:
    - coefficients (pd.Series): Coefficients of the linear regression model.
    - weekend (bool): Whether to plot coefficients for weekend features.

    Returns:
    - None
    """
    # Filter coefficients for weekday or weekend features
    if col_keyword:
        coefficients = coefficients[coefficients.index.str.contains(col_keyword)]
    
    # Plot the coefficients
    plt.figure(figsize=(10, 6))
    plt.plot(coefficients.index, coefficients)
    plt.xlabel('Coefficient Value')
    plt.title(f'Coefficients for {col_keyword} features')
    plt.xticks(rotation=90)
    plt.show()

def plot_year_over_year_coefficients(coeffs, keyword=False, model_name="", years=None):
    """
    Plots the coefficients for each year for the linear regression model.

    Args:
    - coeffs (dict): Dictionary containing the coefficients for each year.
    - keyword (str): Keyword to filter the coefficients.
    - model_name (str): Name of the model.
    - years (list): List of years to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))

    if years:
        coeffs = {year: coeffs[year] for year in years}

    if not coeffs:
        print('No coefficients found for the specified years.')
        return

    for year, coeff in coeffs.items():
        if keyword:
            coeff = coeff[coeff.index.str.contains(keyword)]
            if len(coeff) == 1:
                plt.bar(year, coeff.iloc[0])
            else:
                plt.plot(coeff.index, coeff)
    plt.xlabel('Coefficient Value')
    plt.title(f'{model_name} Coefficients for "{keyword}" Features')
    plt.xticks(rotation=90)  
    # add legend only if its a bar plot
    if len(coeff) != 1 and len(coeffs) > 1:
        plt.legend([f'Year {year}' for year in coeffs.keys()], title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

    # add year in title if only one year is selected
    if len(coeffs) == 1:
        plt.title(f'{model_name} Coefficients for "{keyword}" Features in {year}')  
    plt.show()

def plot_metrics(metrics):
    """
    Plots separate metrics for models.

    Args:
    - metrics (dict): Dictionary containing the metrics for each model
    """
    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Plot each metric separately
    for metric in metrics_df.columns:
        plt.figure(figsize=(12, 6))

        # Sort the values for the current metric
        sorted_df = metrics_df.sort_values(by=metric)

        # Plot each model for the current metric
        for index, value in enumerate(sorted_df[metric]):
            plt.bar(sorted_df.index[index], value, label=sorted_df.index[index])

            # Annotate values on top of the bars
            plt.text(index, 
                     value + (value * 0.02),  # Adjusted position
                     format(value, '.2f'), 
                     ha='center', va='bottom', 
                     fontsize=10)
        
        plt.ylabel('Value')
        plt.title(f'{metric.capitalize()} for Each Model')

        # Add legend for models on the right
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Remove x-ticks and labels as they're represented in the legend
        plt.xticks([])

        plt.tight_layout()
        plt.show()


def plot_predictions(predictions, y_test):
    """
    Plots the predictions for each model.

    Args:
    - predictions (dict): Dictionary containing the predictions for each model
    - y_test (pd.Series): Testing target.
    """
    fig = go.Figure()
    # add real value
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Real Value'))

    for model, data in predictions.items():
        fig.add_trace(go.Scatter(x=data[0]['predictions'].index, y=data[0]['predictions'], mode='lines', name=model))
    

    # add zoom
    fig.update_layout(xaxis_rangeslider_visible=True)
    fig.update_layout(title='Predictions Over Time', xaxis_title='Time', yaxis_title='Predictions')
    fig.show()

def plot_mae(predictions, y_test):
    """
    Plots the Mean Absolute Error for each model.

    Args:
    - predictions (dict): Dictionary containing the predictions for each model
    - y_test (pd.Series): Testing target.
    """
    fig = go.Figure()
    for model, data in predictions.items():
        mae = np.abs(y_test - data[0]['predictions'])
        fig.add_trace(go.Scatter(x=mae.index, y=mae, mode='lines', name=model))
    
    fig.update_layout(xaxis_rangeslider_visible=True)
    fig.update_layout(title='Mean Absolute Error Over Time', xaxis_title='Time', yaxis_title='Mean Absolute Error')
    fig.show()




def visualize_predictions(processed_data, predictions):
    """
    Visualizes the true spot prices and predictions different models using Dash.

    Args:
    - processed_data (pd.DataFrame): DataFrame containing the processed data.
    - predictions (dict): Dictionary containing the predictions of different models.
        - Key: Model name
        - Value: List of dictionaries containing the predictions for each fold.
            - 'model': Trained sklearn model
            - 'predictions': Predictions for the fold (pd.Series)
            - 'metrics': Evaluation metrics for the fold
                - 'mean_squared_error'
                - 'mean_absolute_error'
                - 'root_mean_squared_error'

    Returns:
    - Dash app: Dash app for visualizing the data.
    """
    app = dash.Dash(__name__)

    # Initialize the figure object
    fig = go.Figure()

    # filter processed data from 2017 to 2019
    processed_data = processed_data[processed_data.index.year.isin([2017, 2018, 2019])]

    # Add traces for true spot prices
    fig.add_trace(go.Scatter(
        x=processed_data.index, 
        y=processed_data['y'], 
        mode='lines', 
        name='True Spot Prices',
        # dark grey line
        line=dict(color='rgba(0, 0, 0, 0.8)', width=2)
    ))

    # Define a color palette
    colors = ['#D7263D', '#1EAAF1', '#F2C14E', '#F78154', '#6F2DBD', '#A3DA8D', '#F49D6E', '#8E5572', '#FFD166', '#345995', '#F25C63', '#45B39D', '#FF8A5C', '#5A69AF', '#EFB7B7', '#28AFB0', '#FF7F11', '#B76761', '#D4A5A5', '#7A457D']

    # Add traces for each model's predictions
    color_dict = {model: colors[i % len(colors)] for i, model in enumerate(predictions.keys())}
    for model_name, results in predictions.items():
        for index, fold in enumerate(results):
            predictions_df = pd.DataFrame(fold, columns=['predictions'])
            fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['predictions'],
                mode='lines',
                name=model_name,
                line=dict(color=color_dict[model_name], width=2),
            ))

    # Add interactive components and configurations
    fig.update_layout(
        title='Model Predictions vs Actual Spot Prices',
        yaxis_title='Spot Price',
        legend_title='Model',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type='date',  # This ensures that the range slider works with date indices
            rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
                ])
            )
        ),
        height=800
    )

    # Layout for the app
    app.layout = html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=processed_data.index.min(),
            end_date=processed_data.index.max(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Dropdown(
            id='model-selector',
            options=[{'label': name, 'value': name} for name in predictions.keys()],
            value=list(predictions.keys()),
            multi=True
        ),
        dcc.Graph(id='graph-with-filters', figure=fig)
    ])

    # Callback to update the graph based on selected date range and models
    @app.callback(
        Output('graph-with-filters', 'figure'),
        [Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('model-selector', 'value')]
    )
    def update_figure(start_date, end_date, selected_models):
        filtered_data = processed_data[(processed_data.index >= start_date) & (processed_data.index <= end_date)]
        fig = go.Figure()

        # Add trace for actual prices
        fig.add_trace(go.Scatter(
            x=filtered_data.index, 
            y=filtered_data['y'], 
            mode='lines', 
            name='True Spot Prices',
            line=dict(color='rgba(0, 0, 0, 0.8)', width=2)
        ))

        # Add traces for each selected model
        for model_name in selected_models:
            results = predictions[model_name]
            for index, fold in enumerate(results):
                predictions_df = pd.DataFrame(fold['predictions'], columns=['predictions'])
                fig.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['predictions'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=color_dict[model_name], width=2),
                ))

        # Update layout
        fig.update_layout(
            title='Model Predictions vs Actual Spot Prices',
            yaxis_title='Spot Price',
            legend_title='Model',
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type='date',  # This ensures that the range slider works with date indices
                rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
                )
            ),
            height=800
        )

        return fig

    return app

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


def plot_prediction_accuracy_histogram(accuracy_dict, title, year_on_year=False):
    """
    Plots separate bar graphs for max and min price hour prediction accuracies in the same image.
    Displays the accuracy values above each bar.
    
    Parameters:
    - accuracy_dict: Dictionary with keys as k values and values as accuracy percentages for max and min hours
        - Key: 'max' or 'min'
        - Value: Dictionary with keys as k values and values as accuracy percentages
    - title: Title for the entire figure
    - year_on_year: Boolean indicating if the data is for each year separately
    """
    if year_on_year:
        years = sorted(accuracy_dict['max'].keys())
        x = list(accuracy_dict['max'][years[0]].keys())  # Assuming all years have the same k values

        # Setup the plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Adjust bar width and spacing
        bar_width = 0.15  # Smaller bar width for better spacing
        space_between_groups = 0.3  # Space between groups of bars
        positions = [i * (bar_width * len(years) + space_between_groups) for i in range(len(x))]

        # Plot Max and Min Price Hours Accuracy for each year
        for i, (key, ax, label) in enumerate(zip(['max', 'min'], axes, ['Max', 'Min'])):
            for j, year in enumerate(years):
                # Get the accuracy values for the current year
                y_values = list(accuracy_dict[key][year].values())
                # Adjust positions for each year bar
                ax.bar(
                    [pos + j * bar_width for pos in positions], 
                    y_values, 
                    width=bar_width, 
                    label=f'Year {year}'
                )

            # Set labels and titles
            ax.set_xlabel('Number of Top-k Hours')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{label} Price Hours Accuracy')
            ax.set_xticks([pos + bar_width * (len(years) - 1) / 2 for pos in positions])
            ax.set_xticklabels(x)
            ax.set_ylim(0, 100)

            # Add value labels on top of each bar
            for j, year in enumerate(years):
                y_values = list(accuracy_dict[key][year].values())
                for k, v in enumerate(y_values):
                    ax.text(positions[k] + j * bar_width, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)
            
            # Add legend
            ax.legend()

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    else:
        x = list(accuracy_dict['max'].keys())
        y_max = list(accuracy_dict['max'].values())
        y_min = list(accuracy_dict['min'].values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot Max and Min Price Hours Accuracy
        for ax, y, label, color in zip(axes, [y_max, y_min], ['Max', 'Min'], ['skyblue', 'lightcoral']):
            ax.bar(x, y, color=color)
            ax.set_xlabel('Number of Top-k Hours')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{label} Price Hours Accuracy')
            ax.set_xticks(x)
            ax.set_ylim(0, 100)
            for i, v in enumerate(y):
                ax.text(i + 1, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

def plot_overall_accuracy_comparison(accuracy_dicts, models, top_k_values, add_to_title):
    """
    Plots a multi-bar chart comparing overall accuracies across different models for different top-k values.

    Parameters:
    - accuracy_dicts: List of accuracy dictionaries for each model
    - models: List of model names
    - top_k_values: List of top-k values considered
    - add_to_title: Additional text to add to the title
    """
    x = top_k_values
    bar_width = 0.2
    space_between_groups = 0.3  # Space between groups of bars
    positions = [i * (bar_width * len(models) + space_between_groups) for i in range(len(x))]

    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Max and Min Price Hours Accuracy for each model
    for i, (key, ax, label) in enumerate(zip(['max', 'min'], axes, ['Max', 'Min'])):
        for j, model in enumerate(models):
            # Get the accuracy values for the current model
            y_values = [accuracy_dicts[model][key][k] for k in x]
            # Adjust positions for each model bar
            ax.bar(
                [pos + j * bar_width for pos in positions], 
                y_values, 
                width=bar_width, 
                label=f'Model: {model}'
            )

        # Set labels and titles
        ax.set_xlabel('Number of Top-k Hours')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{label} Price Hours Accuracy')
        ax.set_xticks([pos + bar_width * (len(models) - 1) / 2 for pos in positions])
        ax.set_xticklabels(x)
        ax.set_ylim(0, 100)

        # Add value labels on top of each bar
        for j, model in enumerate(models):
            y_values = [accuracy_dicts[model][key][k] for k in x]
            for k, v in enumerate(y_values):
                ax.text(positions[k] + j * bar_width, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)
        
        # Add legend
        ax.legend()

    fig.suptitle(f'Overall Prediction Accuracy Comparison Across Models {add_to_title}')
    plt.tight_layout()
    plt.show()